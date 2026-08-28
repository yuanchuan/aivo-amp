//! Amp CLI bridge.
//!
//! Stands up a localhost HTTP server that intercepts every call Amp CLI
//! would make to its backend (`AMP_URL`) so:
//!
//! 1. The **LLM plane** (`/api/provider/<protocol>/...`) gets routed to a
//!    user-configured upstream (deepseek, openrouter, etc.). Anthropic-
//!    protocol calls go through aivo's `AnthropicToOpenAIRouter` for
//!    on-the-fly Anthropic→OpenAI translation when the upstream isn't
//!    natively Anthropic.
//! 2. The **management plane** (`/api/internal?<method>`, `/api/user/*`,
//!    `/api/telemetry/*`, `/api/auth/*`) is **stubbed locally by default**
//!    so no traffic leaks to ampcode.com. Stub shapes are mirrored from
//!    real ampcode.com responses so amp's auth check (`isAuthenticated`)
//!    flips true and amp progresses to the LLM call.
//!
//! Setting `AIVO_AMP_PASSTHROUGH=1` flips management traffic to the real
//! ampcode.com endpoint (using the token from `~/.local/share/amp/
//! secrets.json`) — useful if the user wants their thread history /
//! telemetry on Sourcegraph. Off by default for privacy.
//!
//! When `--debug` is on, each request/response is appended to a JSONL
//! trace at `~/.config/aivo/logs/amp-trace-<ts>-<pid>.jsonl`. Without
//! `--debug` the bridge writes nothing to disk. Unhandled paths always
//! emit a loud `[amp-bridge] UNHANDLED` on stderr regardless of `--debug`,
//! so users discovering an unstubbed RPC can re-run with `--debug` to
//! capture the body.

use anyhow::Result;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::AsyncWriteExt;

use crate::amp_threads;
use aivo::constants::CONTENT_TYPE_JSON;
use aivo::services::http_utils::{self, router_http_client};
use aivo::services::percent_codec;

#[derive(Clone)]
pub struct AmpBridgeConfig {
    pub upstream_base_url: String,
    pub upstream_api_key: String,
    /// JSONL file the bridge appends every observed request to.
    /// `Some` only when `--debug` is on; `None` skips trace I/O entirely
    /// so a normal `aivo amp` run touches no log files.
    pub trace_log_path: Option<PathBuf>,
    /// When set, `/api/internal?<method>` and other management routes are
    /// forwarded to a real Amp endpoint (typically `https://ampcode.com/`)
    /// instead of being stubbed. Lets the user keep Amp's auth, threads, and
    /// telemetry plane working against Sourcegraph while only the LLM plane
    /// (`/api/provider/<X>/...`) gets routed at `upstream_base_url`.
    pub native_amp_url: Option<String>,
    pub native_amp_key: Option<String>,
    /// Port of an upstream-targeting `AnthropicToOpenAIRouter` running on
    /// localhost. When set, `/api/provider/anthropic/...` paths are
    /// forwarded to it (translation: Anthropic /v1/messages → OpenAI
    /// /v1/chat/completions). When None, Anthropic requests go directly
    /// to the upstream — only correct when the upstream natively speaks
    /// Anthropic protocol.
    pub anthropic_translation_port: Option<u16>,
    /// Port of an upstream-targeting `ResponsesToChatRouter` running on
    /// localhost. When set, `/api/provider/openai/v1/responses` (the
    /// OpenAI Responses API endpoint amp uses for interactive chat) gets
    /// forwarded there for Responses → /v1/chat/completions translation.
    /// Most non-OpenAI upstreams (deepseek, openrouter, …) only have
    /// /v1/chat/completions, so this translation is mandatory.
    pub responses_translation_port: Option<u16>,
    /// When set, the bridge rewrites the `model` field in `/api/provider/<X>`
    /// request bodies to this value before forwarding. Amp picks Claude
    /// model names internally based on its agent mode; non-Amp upstreams
    /// (deepseek, openrouter, etc.) won't accept those names. Threaded
    /// from `aivo run amp -m <model>`.
    pub force_model: Option<String>,
    /// Directory the bridge persists `uploadThread` payloads to (and
    /// reads back on `getThread` / `listThreads`). Mirrors what
    /// ampcode.com does server-side so `amp threads continue T-<id>`
    /// works after `aivo amp` exits.
    pub threads_dir: PathBuf,
    /// Stats sink: token usage of each LLM turn is recorded against
    /// `usage_key_id` in this store, labeled `amp`, so `aivo stats --by amp`
    /// reflects aivo-routed usage. amp's own threads don't carry usage (neo
    /// computes it server-side), so the bridge — the one point that sees every
    /// provider response — is the source of truth. `None` disables accounting.
    pub usage_store: Option<aivo::services::session_store::SessionStore>,
    pub usage_key_id: String,
}

pub struct AmpBridge {
    config: AmpBridgeConfig,
}

#[derive(Clone)]
struct AmpBridgeState {
    config: Arc<AmpBridgeConfig>,
    client: reqwest::Client,
}

impl AmpBridge {
    pub fn new(mut config: AmpBridgeConfig) -> Self {
        // Drop the trailing slash once so per-request URL building doesn't
        // re-trim on every forwarded call.
        let trimmed = config.upstream_base_url.trim_end_matches('/').to_string();
        config.upstream_base_url = trimmed;
        if let Some(url) = config.native_amp_url.as_mut() {
            let trimmed = url.trim_end_matches('/').to_string();
            *url = trimmed;
        }
        Self { config }
    }

    /// Binds to a random local port and runs the bridge in the background.
    /// Caller sets `AMP_URL=http://127.0.0.1:<port>` before spawning amp.
    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let state = AmpBridgeState {
            config: Arc::new(self.config.clone()),
            client: router_http_client(),
        };
        let handle = tokio::spawn(async move { run_bridge(listener, state).await });
        Ok((port, handle))
    }
}

/// Tries to read the user's Sourcegraph Amp token from the canonical
/// `~/.local/share/amp/secrets.json` file. The file format is
/// `{"apiKey@<url>": "<token>"}`. Returns `(url, token)` of the first entry
/// found, or `None` if the file doesn't exist / can't be parsed.
pub fn detect_native_amp_credentials() -> Option<(String, String)> {
    let home = aivo::services::system_env::home_dir()?;
    let path = aivo::services::system_env::join_segments(
        &home,
        &[".local", "share", "amp", "secrets.json"],
    );
    let raw = std::fs::read_to_string(&path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let obj = value.as_object()?;
    for (k, v) in obj {
        if let Some(token) = v.as_str().filter(|t| !t.is_empty())
            && let Some(url) = k.strip_prefix("apiKey@")
        {
            return Some((url.to_string(), token.to_string()));
        }
    }
    None
}

/// True if `url` points at an Amp-protocol-compatible endpoint that doesn't
/// need the bridge — Sourcegraph's hosted endpoint or anything on localhost
/// (typical of self-hosted Sourcegraph or CLIProxyAPI deployments).
pub fn is_amp_native_endpoint(url: &str) -> bool {
    let Ok(parsed) = url::Url::parse(url) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let host = host.to_ascii_lowercase();
    host == "ampcode.com"
        || host.ends_with(".ampcode.com")
        || host == "sourcegraph.com"
        || host.ends_with(".sourcegraph.com")
        || host == "localhost"
        || host == "127.0.0.1"
        || host == "0.0.0.0"
        || host == "::1"
}

/// Response returned by `dispatch`. The streaming variant lets us deliver
/// SSE chunks to amp as they arrive instead of buffering the whole answer
/// — important for interactive chat where token-by-token rendering is
/// the difference between "feels alive" and "stares at a blank screen
/// for 10 seconds".
enum BridgeResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: String,
    },
    Streaming {
        status: u16,
        content_type: String,
        upstream: reqwest::Response,
        /// Apply the reasoning content_part filter incrementally per SSE
        /// event. Set when forwarding `/api/provider/openai/v1/responses`
        /// — those events sometimes carry `part.type == "reasoning"`,
        /// which amp's parser doesn't recognize.
        filter_reasoning: bool,
    },
}

async fn run_bridge(listener: tokio::net::TcpListener, state: AmpBridgeState) -> Result<()> {
    loop {
        let (mut socket, _peer) = listener.accept().await?;
        let state = state.clone();

        tokio::spawn(async move {
            let request_bytes = match http_utils::read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(err) => {
                    let response = http_utils::http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
            };

            let request = String::from_utf8_lossy(&request_bytes);
            let method = request.split_whitespace().next().unwrap_or("").to_string();
            let full_path = http_utils::extract_request_path(&request);

            // amp's new "Neo" architecture (2026-05-28) opens a Rivet
            // actor WebSocket on the same AMP_URL host:port. Detect the
            // upgrade and switch to the WS handler before falling through
            // to the HTTP dispatch.
            if http_utils::header_value(&request, "upgrade")
                .map(|v| v.eq_ignore_ascii_case("websocket"))
                .unwrap_or(false)
            {
                handle_websocket(socket, &request, &full_path, state.clone()).await;
                return;
            }

            let body = http_utils::extract_request_body(&request)
                .unwrap_or("")
                .to_string();

            log_request(
                state.config.trace_log_path.as_deref(),
                &method,
                &full_path,
                &body,
            )
            .await;

            let dispatch_result = dispatch(&state, &request, &method, &full_path, &body).await;
            match dispatch_result {
                Ok(BridgeResponse::Buffered {
                    status,
                    content_type,
                    body,
                }) => {
                    log_response_buffered(
                        state.config.trace_log_path.as_deref(),
                        &full_path,
                        status,
                        &body,
                    )
                    .await;
                    let _ = http_utils::write_buffered_response(
                        &mut socket,
                        status,
                        &content_type,
                        body.as_bytes(),
                    )
                    .await;
                }
                Ok(BridgeResponse::Streaming {
                    status,
                    content_type,
                    upstream,
                    filter_reasoning,
                }) => {
                    let captured = stream_through_socket(
                        &mut socket,
                        status,
                        &content_type,
                        upstream,
                        filter_reasoning,
                    )
                    .await;
                    log_response_buffered(
                        state.config.trace_log_path.as_deref(),
                        &full_path,
                        status,
                        &captured,
                    )
                    .await;
                }
                Err(err) => {
                    eprintln!("[amp-bridge] dispatch error: {err}");
                    let raw = http_utils::http_error_response(500, "amp-bridge error");
                    let _ = socket.write_all(raw.as_bytes()).await;
                }
            }
        });
    }
}

/// Streams `upstream` through `socket` as chunked HTTP. Captures the bytes
/// in a buffer (returned to the caller for trace logging) while writing
/// them to the socket — buffer growth is bounded by the upstream's natural
/// response size. When `filter_reasoning` is set, runs the SSE byte stream
/// through `IncrementalReasoningFilter` so events with
/// `part.type == "reasoning"` are dropped before they reach amp.
async fn stream_through_socket(
    socket: &mut tokio::net::TcpStream,
    status: u16,
    content_type: &str,
    mut upstream: reqwest::Response,
    filter_reasoning: bool,
) -> String {
    let head = http_utils::http_chunked_response_head(status, content_type);
    if socket.write_all(head.as_bytes()).await.is_err() {
        return String::new();
    }
    let mut captured = String::new();
    let mut filter = IncrementalReasoningFilter::new();
    while let Ok(Some(chunk)) = upstream.chunk().await {
        let bytes = if filter_reasoning {
            filter.feed(&chunk)
        } else {
            chunk.to_vec()
        };
        if !bytes.is_empty() {
            captured.push_str(&String::from_utf8_lossy(&bytes));
            let formatted = http_utils::format_http_chunk(&bytes);
            if socket.write_all(&formatted).await.is_err() {
                break;
            }
        }
    }
    if filter_reasoning {
        let tail = filter.flush();
        if !tail.is_empty() {
            captured.push_str(&String::from_utf8_lossy(&tail));
            let formatted = http_utils::format_http_chunk(&tail);
            let _ = socket.write_all(&formatted).await;
        }
    }
    let _ = socket.write_all(b"0\r\n\r\n").await;
    captured
}

/// Streaming SSE filter: buffers incoming bytes, emits complete events
/// (delimited by `\n\n`) one at a time after running each through the
/// reasoning strip. Partial events stay in the buffer until the next
/// `feed()` call. `flush()` emits whatever's left at end-of-stream.
struct IncrementalReasoningFilter {
    buffer: String,
}

impl IncrementalReasoningFilter {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    fn feed(&mut self, chunk: &[u8]) -> Vec<u8> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut out = Vec::new();
        while let Some(idx) = self.buffer.find("\n\n") {
            let event = self.buffer[..idx].to_string();
            self.buffer.drain(..idx + 2);
            if event_is_reasoning_content_part(&event) {
                continue;
            }
            let cleaned = strip_reasoning_from_event_data(&event);
            out.extend_from_slice(cleaned.as_bytes());
            out.extend_from_slice(b"\n\n");
        }
        out
    }

    fn flush(&mut self) -> Vec<u8> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let event = std::mem::take(&mut self.buffer);
        if event_is_reasoning_content_part(&event) {
            return Vec::new();
        }
        let cleaned = strip_reasoning_from_event_data(&event);
        cleaned.into_bytes()
    }
}

/// Completes the WebSocket handshake on the already-read request, then
/// pumps frames in a logging-only loop. For each inbound text frame we
/// JSON-parse the payload; if it carries a JSON-RPC `id`, we reply with
/// `{"jsonrpc":"2.0","id":<id>,"result":{}}` so amp's actor client stops
/// re-trying. The goal is not to drive amp end-to-end — it's to keep the
/// socket open long enough that the trace log captures the protocol
/// shape (method names, params) so a follow-up PR can stub specific
/// methods properly.
async fn handle_websocket(
    socket: tokio::net::TcpStream,
    request: &str,
    full_path: &str,
    state: AmpBridgeState,
) {
    use futures::StreamExt;
    use tokio_tungstenite::tungstenite::Message;

    let trace = state.config.trace_log_path.clone();
    let trace = trace.as_deref();

    let Some(key) = http_utils::header_value(request, "sec-websocket-key") else {
        let resp = http_utils::http_error_response(400, "missing Sec-WebSocket-Key");
        let mut s = socket;
        let _ = s.write_all(resp.as_bytes()).await;
        return;
    };
    let accept = tokio_tungstenite::tungstenite::handshake::derive_accept_key(key.as_bytes());
    // amp's Rivet WS client closes with code 1002 ("Server did not
    // respond with sent protocols") unless the server echoes one of the
    // values it sent in `Sec-WebSocket-Protocol`. Echo the first one
    // (typically `rivetkit`).
    let protocol_header = http_utils::header_value(request, "sec-websocket-protocol")
        .and_then(|v| v.split(',').next().map(str::trim))
        .filter(|v| !v.is_empty())
        .map(|p| format!("Sec-WebSocket-Protocol: {p}\r\n"))
        .unwrap_or_default();
    let handshake = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept}\r\n\
         {protocol_header}\
         \r\n"
    );
    let mut socket = socket;
    if socket.write_all(handshake.as_bytes()).await.is_err() {
        return;
    }
    log_ws_event(trace, full_path, "open", "").await;

    let ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
        socket,
        tokio_tungstenite::tungstenite::protocol::Role::Server,
        None,
    )
    .await;

    let mut ws_state = WsState::from_path(full_path);
    // If the user did `amp threads continue T-<id>`, the WS opens
    // with an existing thread id and amp expects the actor to know
    // about the prior conversation. Pull it off disk so the
    // client_resume arm has something to replay.
    ws_state.hydrate_from_disk(&state.config.threads_dir).await;
    let cancel_flag = ws_state.cancel_flag.clone();
    let (sink, mut stream) = ws.split();
    // Three concurrent tasks share the connection so the LLM call in
    // the worker doesn't block ping/pong on the read+write paths
    // (otherwise amp closes with code 4000 "Pong timeout" after 10s
    // and we lose any in-flight tool_lease delivery).
    let (write_tx, write_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (worker_tx, worker_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let trace_owned = state.config.trace_log_path.clone();
    let full_path_owned = full_path.to_string();

    let writer = tokio::spawn(ws_writer_task(
        sink,
        write_rx,
        trace_owned.clone(),
        full_path_owned.clone(),
    ));
    let worker = tokio::spawn(ws_worker_task(
        state.clone(),
        ws_state,
        worker_rx,
        write_tx.clone(),
        trace_owned.clone(),
        full_path_owned.clone(),
    ));

    while let Some(msg) = stream.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(err) => {
                log_ws_event(trace, full_path, "error", &err.to_string()).await;
                break;
            }
        };
        match msg {
            Message::Text(text) => {
                let s = text.to_string();
                log_ws_event(trace, full_path, "recv", &s).await;
                // Application-level pings — answer immediately, don't
                // queue behind whatever the worker is doing.
                if s == "ping" {
                    log_ws_event(trace, full_path, "send", "pong").await;
                    if write_tx.send("pong".to_string()).is_err() {
                        break;
                    }
                    continue;
                }
                // client_cancel must take effect immediately. The worker
                // is usually blocked inside the LLM streaming await, so
                // dequeuing would happen only after the turn finishes
                // — too late. Set the cancel flag right here so the
                // streaming loop can poll it and bail.
                if is_client_cancel_frame(&s) {
                    cancel_flag.store(true, Ordering::SeqCst);
                }
                if worker_tx.send(s).is_err() {
                    break;
                }
            }
            Message::Binary(bytes) => {
                log_ws_event(
                    trace,
                    full_path,
                    "recv-binary",
                    &format!("<{} bytes>", bytes.len()),
                )
                .await;
            }
            Message::Close(frame) => {
                let detail = frame
                    .map(|f| format!("code={} reason={}", f.code, f.reason))
                    .unwrap_or_else(|| "<no frame>".to_string());
                log_ws_event(trace, full_path, "close", &detail).await;
                break;
            }
            Message::Ping(p) => {
                log_ws_event(
                    trace,
                    full_path,
                    "recv-ping",
                    &format!("<{} bytes>", p.len()),
                )
                .await;
            }
            Message::Pong(p) => {
                log_ws_event(
                    trace,
                    full_path,
                    "recv-pong",
                    &format!("<{} bytes>", p.len()),
                )
                .await;
            }
            _ => {}
        }
    }
    drop(worker_tx);
    drop(write_tx);
    let _ = worker.await;
    let _ = writer.await;
}

async fn ws_writer_task(
    mut sink: futures::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        tokio_tungstenite::tungstenite::Message,
    >,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    _trace: Option<PathBuf>,
    _full_path: String,
) {
    use futures::SinkExt;
    while let Some(event) = rx.recv().await {
        if sink
            .send(tokio_tungstenite::tungstenite::Message::Text(event.into()))
            .await
            .is_err()
        {
            break;
        }
    }
}

async fn ws_worker_task(
    state: AmpBridgeState,
    mut ws_state: WsState,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    write_tx: tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<PathBuf>,
    full_path: String,
) {
    let trace_ref = trace.as_deref();
    while let Some(text) = rx.recv().await {
        if let Some(reply) = ws_stub_reply(&text) {
            log_ws_event(trace_ref, &full_path, "send", &reply).await;
            if write_tx.send(reply).is_err() {
                break;
            }
        }
        ws_followup_events(
            &state,
            &mut ws_state,
            &text,
            &write_tx,
            trace_ref,
            &full_path,
        )
        .await;
    }
}

/// Helper that ships a single outbound event: log it, write it to
/// the channel. Used by `agent_turn` to stream delta events.
async fn send_event(
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    event: String,
) {
    log_ws_event(trace, full_path, "send", &event).await;
    let _ = write_tx.send(event);
}

/// amp's Zod schemas accept thread/message ids only in fixed shapes
/// (binary `Ix` / `c3`):
///   threadId: `T-<8hex>-<4hex>-<4hex>-<4hex>-<12hex>` (UUID v7 style)
///   messageId: `M-<22 chars [0-9A-Za-z]>`
/// Anything else fails `kZR.safeParse` and amp silently drops the
/// server event — which is why a `T-aivo-<nanos>` thread id meant
/// every `message_added` ack was discarded and the user-message
/// outbox kept retrying.
fn new_thread_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!(
        "T-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

fn new_message_id() -> String {
    format!("M-{}", random_base62_22())
}

/// amp's `U3` Zod template literal requires `TU-<22 base62>`. The LLM
/// returns tool_use IDs like `call_00_...` (OpenAI style) or
/// `toolu_...` (Anthropic style), neither of which validate. We
/// rewrite to `TU-...` form before emitting and use the same id when
/// the result comes back from the executor.
fn new_tool_call_id() -> String {
    format!("TU-{}", random_base62_22())
}

/// Normalizes a tool_use block's `id` to amp's `TU-<22>` shape, in
/// place. amp's Zod parser silently drops any frame whose tool_use id
/// doesn't validate, including streamed `delta` events — so without
/// normalizing at SSE block-start time, the mid-stream tool_use delta
/// gets thrown away by the TUI even though the final `message_updated`
/// renders correctly. Idempotent on already-valid TU ids so a repeated
/// pass (e.g. SSE content_block_start + later agent_turn_finish
/// fold-up) reuses the existing id instead of burning a fresh one.
/// Returns the final id for blocks the caller needs to track, else
/// `None` for non-tool_use blocks.
fn ensure_tool_use_id(block: &mut serde_json::Value) -> Option<String> {
    if block.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
        return None;
    }
    let current = block
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let id = if is_valid_tu_id(&current) {
        current
    } else {
        let fresh = new_tool_call_id();
        block["id"] = json!(fresh.clone());
        fresh
    };
    Some(id)
}

// A tool result has two on-the-wire shapes that share `type:"tool_result"`
// but differ in their id field, so the field name disambiguates them:
//
//   amp neo  : {type:"tool_result", toolUseID, run:{status, result, …}}
//   Anthropic: {type:"tool_result", tool_use_id, content, is_error}
//
// amp's thread store + reload path speak neo; the upstream LLM speaks
// Anthropic. We persist neo so `amp threads continue` / the thread
// switcher can reload a thread that contains tool calls — with the
// Anthropic shape, amp's loader reads `block.toolUseID` (undefined) and
// runs `M-${undefined.replace(/^TU-/,"")}`, throwing "undefined is not an
// object (evaluating 'R.replace')" (the stuck half of aivo#14). Both
// converters are shape-aware and idempotent: a block already in the
// target shape (or a non-tool_result block) passes through untouched, so
// they also upconvert *legacy* threads persisted in the Anthropic shape
// before this fix.

fn is_tool_result(block: &serde_json::Value) -> bool {
    block.get("type").and_then(|t| t.as_str()) == Some("tool_result")
}

/// neo → Anthropic, for blocks fed to the upstream LLM.
fn block_to_anthropic(block: &serde_json::Value) -> serde_json::Value {
    // Not a tool_result, or already Anthropic → leave it alone.
    if !is_tool_result(block) || block.get("tool_use_id").is_some() {
        return block.clone();
    }
    let run = block.get("run");
    let status = run
        .and_then(|r| r.get("status"))
        .and_then(|s| s.as_str())
        .unwrap_or("error");
    let content = match run.and_then(|r| r.get("result")) {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => format!("(tool exited with status: {status})"),
    };
    json!({
        "type": "tool_result",
        "tool_use_id": block.get("toolUseID").cloned().unwrap_or_else(|| json!("")),
        "content": content,
        "is_error": status != "done",
    })
}

/// Anthropic → neo, for blocks served to amp's TUI / reload path.
fn block_to_neo(block: &serde_json::Value) -> serde_json::Value {
    // Not a tool_result, or already neo → leave it alone.
    if !is_tool_result(block) || block.get("toolUseID").is_some() {
        return block.clone();
    }
    let is_error = block
        .get("is_error")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);
    json!({
        "type": "tool_result",
        "toolUseID": block.get("tool_use_id").cloned().unwrap_or_else(|| json!("")),
        "run": {
            "status": if is_error { "error" } else { "done" },
            "result": block.get("content").cloned().unwrap_or_else(|| json!("")),
        },
    })
}

/// Applies `f` to every content block of a message `content` value
/// (block-array messages only; strings/other shapes pass through).
fn map_content_blocks(
    content: &serde_json::Value,
    f: fn(&serde_json::Value) -> serde_json::Value,
) -> serde_json::Value {
    match content {
        serde_json::Value::Array(blocks) => {
            serde_json::Value::Array(blocks.iter().map(f).collect())
        }
        other => other.clone(),
    }
}

/// Normalizes every message's content in a loaded thread payload to amp's
/// neo shape before serving it to amp (getThread / getThreadTail). Makes
/// both freshly-persisted (already-neo) and legacy (Anthropic) threads
/// reload without the `toolUseID`-undefined crash.
fn normalize_thread_payload_to_neo(payload: &mut serde_json::Value) {
    if let Some(messages) = payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for msg in messages {
            if let Some(content) = msg.get("content") {
                let neo = map_content_blocks(content, block_to_neo);
                msg["content"] = neo;
            }
        }
    }
}

/// Marks a served thread payload as actor-native so neo's switch/resume
/// path connects directly instead of running the import-into-actor dance.
///
/// Neo decides whether a thread must be imported with
/// `needsImport = thread.meta.usesThreadActors !== true` (`cVR` in the
/// binary). When false, neo POSTs the thread to `/actors/.../import`,
/// marks it imported, then resumes — a multi-step Rivet handshake the
/// bridge only half-satisfies, leaving `connectingToThreadID` stuck and
/// the composer frozen on "Loading thread" after a thread switch. Since
/// the bridge already serves the actor WS and hydrates history from
/// disk, we advertise every thread as actor-native (`meta.usesThreadActors
/// = true`) so neo skips import and resumes straight away — the same path
/// a freshly-created thread takes.
fn mark_thread_actor_native(payload: &mut serde_json::Value) {
    payload["usesThreadActors"] = json!(true);
    payload["usesDtw"] = json!(false);
    if !payload.get("meta").map(|m| m.is_object()).unwrap_or(false) {
        payload["meta"] = json!({});
    }
    payload["meta"]["usesThreadActors"] = json!(true);

    // Report a version that already accounts for every message we're
    // serving. neo loads the full history from this `getThread`, then
    // tells the actor "resume me from `v`" (`client_resume`). The
    // bridge's resume arm replays `persisted_messages[v..]`, so a stale
    // `v` (the old hardcoded `2`) makes it re-emit nearly the whole
    // thread amp just loaded — a burst of duplicate `message_added`
    // events on *every* thread switch. Pin `v` to the message count so
    // the replay cursor lands at the end and nothing is re-sent.
    let msg_count = payload
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    payload["v"] = json!(msg_count);
}

/// Extracts the first text from a user message's `content`, whether
/// it's a plain string or an Anthropic-style content-block array.
/// Returns an empty string when nothing usable is found.
fn first_text_from_content(content: &serde_json::Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        for block in arr {
            if block.get("type").and_then(|t| t.as_str()) == Some("text")
                && let Some(t) = block.get("text").and_then(|v| v.as_str())
            {
                return t.to_string();
            }
        }
    }
    String::new()
}

/// Cheap structural test for whether a WS text frame is an amp
/// `client_cancel` notification. Parses the JSON so we don't fire on
/// a literal "client_cancel" substring embedded in a user message.
fn is_client_cancel_frame(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|v| {
            v.get("method")
                .and_then(|m| m.as_str())
                .map(|s| s == "client_cancel")
        })
        .unwrap_or(false)
}

fn is_valid_tu_id(s: &str) -> bool {
    s.len() == 25 && s.starts_with("TU-") && s[3..].chars().all(|c| c.is_ascii_alphanumeric())
}

fn random_base62_22() -> String {
    use rand::Rng;
    const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let mut rng = rand::thread_rng();
    (0..22)
        .map(|_| ALPHABET[rng.gen_range(0..ALPHABET.len())] as char)
        .collect()
}

/// Per-WS-connection state. Tracks the thread id (extracted from the
/// Rivet `rvt-key` URL param), a monotonic `seq` counter for
/// `message_added`/`message_updated` events, the conversation history
/// we hand to the upstream LLM on each turn, the registered tools
/// snapshot, and any in-flight tool calls (executor runs tool locally,
/// sends back `executor_tool_result` which we map to a Anthropic
/// `tool_result` block and feed into the next LLM call).
struct WsState {
    thread_id: String,
    seq: u64,
    /// Anthropic-format conversation history. Each entry is
    /// `{role: "user"|"assistant", content: [...blocks...]}`.
    messages: Vec<serde_json::Value>,
    /// Anthropic-format tools list, accumulated from
    /// `executor_tools_register` events at handshake.
    tools: Vec<serde_json::Value>,
    /// tool_use_id → still waiting flag. Set when an assistant turn
    /// emits tool_use blocks; cleared as `executor_tool_result`
    /// messages arrive.
    pending_tool_uses: std::collections::HashSet<String>,
    /// amp neo tool_result blocks (`{type, toolUseID, run}`), accumulated
    /// as the executor returns each. When `pending_tool_uses` is empty
    /// these fold into a user message: persisted verbatim (neo) and
    /// converted to the Anthropic shape for the LLM via
    /// `neo_block_to_anthropic`.
    tool_results: Vec<serde_json::Value>,
    /// Set of user `messageId`s we've already started processing. amp's
    /// outbox retries `client_append_user_msg` every ~500ms until it
    /// sees a matching `message_added` ack; the bridge's LLM call
    /// blocks the WS read for several seconds, so those retries queue
    /// up and would otherwise each kick off a fresh `agent_turn`. We
    /// always re-send the ack on a retry (so the outbox eventually
    /// stops) but only run `agent_turn` on the first appearance.
    seen_user_message_ids: std::collections::HashSet<String>,
    /// True once amp has sent `executor_tools_bootstrap_complete`.
    /// amp can dispatch a `client_append_user_msg` BEFORE it sends
    /// the executor snapshots / `executor_tools_register`; if we run
    /// `agent_turn` immediately the LLM gets called with zero tools
    /// and ends up hallucinating fake tool names. Queue user messages
    /// until bootstrap completes, then drain.
    bootstrap_complete: bool,
    /// Placeholder `messageId`s of assistant turns that are
    /// pre-emitted (so amp's UI shows a "generating" slot
    /// immediately) but whose actual LLM call is deferred until
    /// bootstrap_complete fires.
    pending_assistant_ids: Vec<String>,
    /// Messages persisted in amp's E2R/D2R shape so we can write the
    /// thread file. Updated when we ack a user message and when we
    /// finalize an assistant message (placeholder doesn't count).
    /// `amp threads continue T-<id>` and `listThreads` read from
    /// this file via the bridge's `getThread` / `listThreads` RPCs.
    persisted_messages: Vec<serde_json::Value>,
    /// Most recent agentMode seen in `client_append_user_msg` —
    /// included in the persisted thread metadata.
    last_agent_mode: Option<serde_json::Value>,
    /// ISO-8601 UTC timestamp of when the thread was opened. amp's CLI
    /// thread list reads this as `created` and feeds it to
    /// `new Date(...).toISOString()`. Null / missing crashes the CLI
    /// with `RangeError: Invalid Date`.
    created_at: String,
    /// Thread title: either set explicitly by `client_set_thread_title`
    /// or auto-derived from the first user message. amp's TUI shows
    /// `(untitled)` for nulls, but listThreads and resume both look
    /// nicer with a real title.
    title: Option<String>,
    /// Most recent `reasoningEffort` from client_append_user_msg
    /// (`low|medium|high|none`). Threaded into the upstream body as
    /// `output_config.effort` so the existing translator pipeline picks
    /// it up and routes it to whichever effort dial the upstream wants.
    /// Without this every turn runs at the upstream default regardless
    /// of amp's selected mode.
    last_reasoning_effort: Option<String>,
    /// Cooperative cancellation flag for the in-flight turn. The reader
    /// task flips this true the moment it sees a `client_cancel` frame
    /// (without waiting for the worker to dequeue, since the worker is
    /// usually blocked inside the LLM streaming await). The streaming
    /// loop polls between events and bails; agent_turn_finish resets it
    /// before the next turn starts.
    cancel_flag: Arc<AtomicBool>,
    /// Tool-approval queue. amp's executor sends
    /// `executor_tool_approval_request` when its local policy says a
    /// tool needs human approval; we add the approval entry here and
    /// rebroadcast `tool_approval_queue` so amp's TUI observers can
    /// render the prompt. On `client_tool_approval_response` (the
    /// TUI's verdict), we forward it back to the executor as
    /// `executor_tool_approval_response` and drop the entry from the
    /// queue. Without this passthrough every tool runs silently under
    /// `dangerouslyAllowAll` (the existing behavior).
    pending_approvals: Vec<serde_json::Value>,
    /// Parallel index mapping `persisted_messages[k]` to its position
    /// in `messages`. The two vecs diverge because tool_result fold-up
    /// user messages get pushed to `messages` between turns but never
    /// to `persisted_messages`. The map lets `client_edit_message` and
    /// `client_retry` truncate both vecs consistently when the user
    /// rewinds the conversation.
    persisted_to_msgs_idx: Vec<usize>,
    /// Most recent skill snapshot from `executor_skill_snapshot` —
    /// custom user-defined commands (frontmatter `name`/`description`).
    /// Surfaced to the LLM so it knows what skill invocations are
    /// available in the current workspace.
    skills: Vec<serde_json::Value>,
    /// AGENTS.md / amp/agents.md files reported by
    /// `executor_guidance_snapshot`. These are user-authored agent
    /// instructions; we inject their content into the system prompt
    /// so the LLM actually follows them (without this the agent
    /// ignores the user's per-repo conventions).
    guidance_files: Vec<serde_json::Value>,
    /// Workspace environment from `executor_environment_snapshot`
    /// (cwd, repo, git branch, …). Surfaced as a "Workspace:" block
    /// in the system prompt so the model doesn't have to grope for
    /// pwd/branch on every turn.
    environment: Option<serde_json::Value>,
    /// Per-thread settings overrides from
    /// `client_update_thread_settings` — fields include
    /// `reasoning.effort`, `internal.model`, `tools.disable/enable`,
    /// `anthropic.thinking.enabled`, etc. Stored as-is and rebroadcast
    /// so amp's TUI observers see the update; actionable fields
    /// (reasoning effort, model override) are picked up by the next
    /// agent_turn.
    thread_settings: serde_json::Value,
    /// True only when `hydrate_from_disk` loaded prior thread state
    /// from a `.json` file on disk. Gates the `client_resume` arm's
    /// replay: for a *fresh* thread (no on-disk state), the messages
    /// in `persisted_messages` were emitted inline by their handlers
    /// already, so replaying them would duplicate them in amp's UI.
    hydrated: bool,
    /// Set once we've surfaced the "unrecognized amp protocol" error to
    /// amp's TUI, so a future amp build that streams several unhandled
    /// `client_*` frames per turn raises one error bar, not a flood.
    protocol_warned: bool,
}

impl WsState {
    fn from_path(full_path: &str) -> Self {
        let query = full_path.split('?').nth(1).unwrap_or("");
        let thread_id = query
            .split('&')
            .find_map(|kv| kv.strip_prefix("rvt-key="))
            .map(percent_codec::decode)
            .unwrap_or_else(|| "T-aivo-unknown".to_string());
        Self {
            thread_id,
            seq: 1,
            messages: Vec::new(),
            tools: Vec::new(),
            pending_tool_uses: std::collections::HashSet::new(),
            tool_results: Vec::new(),
            seen_user_message_ids: std::collections::HashSet::new(),
            bootstrap_complete: false,
            pending_assistant_ids: Vec::new(),
            persisted_messages: Vec::new(),
            last_agent_mode: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            title: None,
            last_reasoning_effort: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            pending_approvals: Vec::new(),
            persisted_to_msgs_idx: Vec::new(),
            skills: Vec::new(),
            guidance_files: Vec::new(),
            environment: None,
            thread_settings: json!({}),
            hydrated: false,
            protocol_warned: false,
        }
    }

    /// On WS open for an existing thread id, load the persisted state
    /// off disk so the `client_resume` arm has messages to replay and
    /// the LLM history matches what the user already saw. Without
    /// this, `amp threads continue T-<id>` connects to a fresh-looking
    /// actor and the conversation appears to have evaporated.
    async fn hydrate_from_disk(&mut self, threads_dir: &Path) {
        let Some(payload) = amp_threads::load_thread(threads_dir, &self.thread_id).await else {
            return;
        };
        self.hydrated = true;
        if let Some(title) = payload.get("title").and_then(|v| v.as_str()) {
            self.title = Some(title.to_string());
        }
        if let Some(created) = payload.get("created").and_then(|v| v.as_str()) {
            self.created_at = created.to_string();
        }
        if let Some(mode) = payload.get("agentMode") {
            self.last_agent_mode = Some(mode.clone());
        }
        let Some(messages) = payload.get("messages").and_then(|v| v.as_array()) else {
            return;
        };
        for msg in messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = msg.get("content").cloned().unwrap_or(json!([]));
            // `messages` feeds the upstream LLM (Anthropic tool_result
            // shape); `persisted_messages` is replayed to amp's TUI on
            // client_resume and re-saved by thread_payload, so normalize it
            // to neo. Both converters are shape-aware, so a legacy thread
            // persisted in the Anthropic shape upgrades cleanly on resume.
            self.messages.push(json!({
                "role": role,
                "content": map_content_blocks(&content, block_to_anthropic),
            }));
            let mut neo_msg = msg.clone();
            if let Some(c) = neo_msg.get("content") {
                neo_msg["content"] = map_content_blocks(c, block_to_neo);
            }
            self.persisted_messages.push(neo_msg);
            self.persisted_to_msgs_idx.push(self.messages.len() - 1);
            if role == "user"
                && let Some(id) = msg.get("messageId").and_then(|v| v.as_str())
            {
                // Pre-populate dedup so amp's outbox retries on resume
                // don't re-trigger the agent loop for already-acked
                // user messages.
                self.seen_user_message_ids.insert(id.to_string());
            }
        }
    }

    /// Builds the on-disk thread payload from accumulated state.
    /// Format matches what amp's `getThread` expects in `result.thread.data`.
    /// `usesThreadActors:true` (+ `meta.usesThreadActors`) marks the thread
    /// actor-native so neo resumes it directly instead of running the
    /// import-into-actor handshake that freezes the composer; see
    /// [`mark_thread_actor_native`].
    fn thread_payload(&self) -> serde_json::Value {
        let mut payload = json!({
            "id": self.thread_id,
            "v": 2,
            "messages": self.persisted_messages,
            "title": self.title.clone(),
            "created": self.created_at,
            "usesDtw": false,
            "usesThreadActors": true,
            "meta": {"usesThreadActors": true},
        });
        if let Some(mode) = &self.last_agent_mode {
            payload["agentMode"] = mode.clone();
        }
        payload
    }

    /// Sets the title if none has been assigned yet. Used both for
    /// auto-derivation from the first user message and for explicit
    /// `client_set_thread_title` events (which always overwrite).
    fn auto_title_from_user_content(&mut self, content: &serde_json::Value) {
        if self.title.is_some() {
            return;
        }
        let raw = first_text_from_content(content);
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return;
        }
        let truncated: String = trimmed.chars().take(60).collect();
        let title = if trimmed.chars().count() > 60 {
            format!("{truncated}…")
        } else {
            truncated
        };
        self.title = Some(title);
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }
}

/// Server-initiated event frames the bridge should push *after*
/// replying to an inbound JSON-RPC request. amp's parser (`kZR` in the
/// binary) treats any WS frame with `method` and no `id` as a
/// notification and reshapes it to `{type: method, ...params}` before
/// dispatching to `handleServerMessage`. So events must be sent in
/// JSON-RPC notification form, not the post-reshape form.
async fn ws_followup_events(
    state: &AmpBridgeState,
    ws_state: &mut WsState,
    text: &str,
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
) {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    let method = val.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let params = val.get("params").cloned().unwrap_or(json!({}));
    match method {
        "executor_connect" => {
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "executor_connected",
                    "params": {
                        "executorId": "exec-aivo",
                        "registeredToolCount": 0,
                        "guidanceInventory": [],
                        "resumeBootstrap": false,
                    },
                })
                .to_string(),
            )
            .await;
        }
        // Track registered tools by name so re-registration on
        // reconnect doesn't accumulate duplicates. amp uses
        // `inputSchema` (camelCase) while Anthropic's /v1/messages
        // expects `input_schema`; translate while copying.
        // Bootstrap is complete — drain any deferred assistant turns
        // that were queued because they arrived before tools were
        // registered.
        "executor_tools_bootstrap_complete" => {
            ws_state.bootstrap_complete = true;
            let ids: Vec<String> = ws_state.pending_assistant_ids.drain(..).collect();
            for id in ids {
                agent_turn_finish(state, ws_state, id, write_tx, trace, full_path).await;
            }
        }
        "executor_tools_register" => {
            if let Some(tools) = params.get("tools").and_then(|v| v.as_array()) {
                for tool in tools {
                    let name = tool.get("name").cloned().unwrap_or(json!(""));
                    let description = tool.get("description").cloned().unwrap_or(json!(""));
                    let input_schema = tool
                        .get("inputSchema")
                        .or_else(|| tool.get("input_schema"))
                        .cloned()
                        .unwrap_or(json!({"type": "object", "properties": {}}));
                    let entry = json!({
                        "name": name.clone(),
                        "description": description,
                        "input_schema": input_schema,
                    });
                    let name_s = name.as_str().unwrap_or("").to_string();
                    if let Some(existing) = ws_state
                        .tools
                        .iter_mut()
                        .find(|t| t.get("name").and_then(|v| v.as_str()) == Some(&name_s))
                    {
                        *existing = entry;
                    } else {
                        ws_state.tools.push(entry);
                    }
                }
            }
        }
        // Mirror of `executor_tools_register`: amp drops tools when an
        // MCP server disconnects or a plugin tool goes away. Without
        // pruning, the next `agent_turn` keeps advertising the dead
        // tool to the LLM, which then calls a tool the executor can no
        // longer run. Schema: `{toolNames: string[]}`.
        "executor_tools_unregister" => {
            if let Some(names) = params.get("toolNames").and_then(|v| v.as_array()) {
                let drop: std::collections::HashSet<&str> =
                    names.iter().filter_map(|n| n.as_str()).collect();
                ws_state.tools.retain(|t| {
                    !drop.contains(t.get("name").and_then(|v| v.as_str()).unwrap_or(""))
                });
            }
        }
        // amp's outbox retries this every ~500ms until it sees a
        // `message_added` server event with the matching `messageId`
        // (`TZR` in the binary). Retries that arrive while we're
        // awaiting the LLM call queue up in the worker channel; if we
        // re-emit the ack for each of them, the later acks get a
        // higher `seq` than the assistant response we already sent,
        // and amp's UI then renders the user message AFTER its own
        // reply. Drop duplicates silently — amp's outbox stops as
        // soon as it sees the first ack.
        "client_append_user_msg" => {
            let message_id = params
                .get("messageId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !ws_state.seen_user_message_ids.insert(message_id.clone()) {
                return;
            }
            let content = params.get("content").cloned().unwrap_or(json!([]));
            let agent_mode = params.get("agentMode").cloned();
            let created_at = chrono::Utc::now().to_rfc3339();
            let mut user_msg = json!({
                "threadId": ws_state.thread_id,
                "role": "user",
                "messageId": message_id,
                "content": content.clone(),
                "createdAt": created_at,
            });
            if let Some(mode) = agent_mode {
                user_msg["agentMode"] = mode.clone();
                ws_state.last_agent_mode = Some(mode);
            }
            // amp drives reasoningEffort from the current agent mode
            // (smart=medium, deep=high, rush=low, …). Capture so the
            // next upstream call hits the right effort tier.
            if let Some(effort) = params
                .get("reasoningEffort")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                ws_state.last_reasoning_effort = Some(effort.to_string());
            }
            ws_state.auto_title_from_user_content(&content);
            ws_state
                .messages
                .push(json!({"role": "user", "content": content}));
            ws_state.persisted_messages.push(user_msg.clone());
            ws_state
                .persisted_to_msgs_idx
                .push(ws_state.messages.len() - 1);

            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "message_added",
                    "params": {
                        "message": user_msg,
                        "seq": ws_state.next_seq(),
                    },
                })
                .to_string(),
            )
            .await;
            persist_thread(state, ws_state).await;
            let assistant_id = agent_turn_start(ws_state, write_tx, trace, full_path).await;
            if ws_state.bootstrap_complete {
                agent_turn_finish(state, ws_state, assistant_id, write_tx, trace, full_path).await;
            } else {
                // Defer the LLM call until executor_tools_register +
                // executor_tools_bootstrap_complete have populated
                // ws_state.tools — otherwise the model gets called
                // with zero tools and hallucinates fake ones.
                ws_state.pending_assistant_ids.push(assistant_id);
            }
        }
        // The reader task flips the cancel flag the moment this frame
        // arrives so an in-flight stream can poll it and bail. When a turn
        // IS live, that bailed stream runs finish_turn_as_cancelled — which
        // clears the flag and emits the `cancelled` / `message_updated` /
        // `agent_state idle` chain — BEFORE the (single-threaded) worker
        // ever dequeues this frame. So reaching this arm with the flag
        // already cleared means the cancel was handled: nothing to do.
        //
        // But if we get here with the flag STILL set, no live turn absorbed
        // it: the status bar is parked on `error` (a settled upstream
        // failure) or `running_tools` (a stuck tool, e.g. the Librarian in
        // aivo#14) with no in-flight stream left to move it. Esc would be a
        // no-op and the badge/spinner would never clear. Abandon any
        // orphaned tool leases and drop the agent back to idle so the user
        // gets their prompt back.
        "client_cancel" => {
            // Flag still set => no live turn absorbed the cancel (an
            // in-flight stream would have cleared it via
            // finish_turn_as_cancelled before we dequeued this frame).
            let unhandled = ws_state.cancel_flag.swap(false, Ordering::SeqCst);
            if unhandled {
                ws_state.pending_tool_uses.clear();
                ws_state.tool_results.clear();
                emit_agent_state(write_tx, trace, full_path, "idle", None).await;
            }
        }
        // Read-state + error-bar notifications neo fires on its own:
        // `client_mark_message_read` auto-emits whenever a message
        // scrolls into view with the terminal focused, `*_unread` is its
        // inverse, `client_dismiss_active_error` fires when the user
        // dismisses the error bar. All are fire-and-forget `send()`
        // frames (not load-bearing `request()`s) — aivo models no
        // read state or server-side error bar, so dropping them is
        // correct. Matched explicitly so they don't trip the
        // unrecognized-protocol error below.
        "client_mark_message_read"
        | "client_mark_message_unread"
        | "client_dismiss_active_error" => {}
        // Known `request()` frames the bridge has no server-side state
        // to mutate. `ws_stub_reply` has already answered each with an
        // empty `result`, which is all amp's `.then()` awaits — so the
        // ONLY thing left to do is *not* fall through to the
        // unrecognized-protocol arm below, which would raise a bogus
        // "update aivo" error bar and park `agent_state` on `error` for
        // routine user actions:
        //   - remove/steer a queued message: aivo runs no server-side
        //     message queue (queued msgs arrive later as
        //     `client_append_user_msg`), so there's nothing to drop.
        //   - upsert_notification_subscription: aivo models no
        //     server-side notification state.
        //   - spawn_executor: only the local executor connects (via
        //     `executor_connect`); amp never requests a sandbox spawn
        //     against the bridge, but ack it defensively.
        "client_remove_queued_msg"
        | "client_steer_queued_msg"
        | "client_upsert_notification_subscription"
        | "client_spawn_executor" => {}
        // Manual `$ <cmd>` invocation. `ws_stub_reply`'s empty result
        // satisfies the request, so this no longer errors — but the
        // bridge does not yet run the command or append its output, so
        // the feature is a graceful no-op rather than fully wired.
        // Schema: `{args, run, hidden}`.
        "client_append_manual_bash_invocation" => {}
        // amp's executor uploads project context as snapshots on
        // connect (and refreshes them via *_update events). Without
        // these the LLM has no idea what workspace it's in — pwd,
        // git branch, available skills, AGENTS.md conventions all
        // get lost. We accumulate into WsState and surface to the
        // model via build_system_prompt on every turn.
        //
        // `isLast` segments snapshots that arrive across multiple
        // frames; we replace on isLast=true so partial state during
        // the in-flight snapshot doesn't bleed into the next turn.
        "executor_skill_snapshot" => {
            if let Some(skills) = params.get("skills").and_then(|v| v.as_array()) {
                if params
                    .get("isLast")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true)
                {
                    ws_state.skills = skills.clone();
                } else {
                    ws_state.skills.extend(skills.iter().cloned());
                }
            }
        }
        "executor_guidance_snapshot" => {
            if let Some(files) = params.get("files").and_then(|v| v.as_array()) {
                if params
                    .get("isLast")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true)
                {
                    ws_state.guidance_files = files.clone();
                } else {
                    ws_state.guidance_files.extend(files.iter().cloned());
                }
            }
        }
        "executor_environment_snapshot" | "executor_environment_update" => {
            if let Some(env) = params.get("environment").cloned() {
                ws_state.environment = Some(env);
            }
        }
        // amp's TUI sends this when the user changes per-thread
        // overrides (model swap, reasoning effort, disabled tools,
        // …). Merge the incoming settings into our stored copy and
        // rebroadcast as `thread_settings` so other observers in the
        // same amp process pick up the update. Actionable fields
        // (reasoning.effort, internal.model) are read off WsState by
        // the next agent_turn.
        "client_update_thread_settings" => {
            if let Some(settings) = params.get("settings").cloned() {
                if let Some(map) = settings.as_object() {
                    let target = ws_state
                        .thread_settings
                        .as_object_mut()
                        .expect("initialized as empty object");
                    for (k, v) in map {
                        target.insert(k.clone(), v.clone());
                    }
                }
                send_event(
                    write_tx,
                    trace,
                    full_path,
                    json!({
                        "jsonrpc": "2.0",
                        "method": "thread_settings",
                        "params": {"settings": ws_state.thread_settings.clone()},
                    })
                    .to_string(),
                )
                .await;
            }
        }
        // amp sends this on connect when resuming a thread (or after
        // a reconnect). The actor is expected to bring amp's TUI in
        // sync with the actor-side state by replaying every persisted
        // message past the client's known `version`. ws_stub_reply
        // already handled the JSON-RPC response (id-matched empty
        // result) so amp's `request("client_resume")` promise resolves;
        // here we additionally walk the hydrated history and re-emit
        // each message as `message_added` so the TUI can render the
        // prior conversation.
        "client_resume" => {
            // Only replay when we hydrated from disk. For a fresh
            // thread (no prior on-disk state), the messages already
            // in `persisted_messages` were emitted inline by their
            // own handlers — the user msg ack from
            // `client_append_user_msg`, the assistant from
            // `agent_turn_finish`. Replaying them here would
            // duplicate them in amp's TUI.
            //
            // Also skip emitting `agent_state idle`. amp sends
            // client_resume on every connect, often right after
            // handshake while the very first user msg is mid-turn.
            // Emitting idle here wipes the `working` state
            // `agent_turn_start` just set; the user sees the
            // spinner flash off and the turn appears to go silent
            // even though the LLM call is still running.
            // ws_stub_reply already responded to the JSON-RPC
            // request itself with `result:{}`, which is what amp's
            // `request("client_resume")` promise actually awaits.
            if !ws_state.hydrated {
                return;
            }
            let cursor = params.get("version").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let messages_snapshot = ws_state.persisted_messages.clone();
            for (idx, msg) in messages_snapshot.iter().enumerate() {
                if idx < cursor {
                    continue;
                }
                send_event(
                    write_tx,
                    trace,
                    full_path,
                    json!({
                        "jsonrpc": "2.0",
                        "method": "message_added",
                        "params": {
                            "message": msg,
                            "seq": ws_state.next_seq(),
                        },
                    })
                    .to_string(),
                )
                .await;
            }
        }
        // Server-routed file/directory reads and git commands. Both
        // TUI and executor share the WS so the bridge relays between
        // them: forward `client_*` requests as `executor_*`, forward
        // `executor_*_result` responses back as `client_*_result`. No
        // policy added — just transport. amp's `@file` mention UX
        // and the git-context status bar both depend on this routing.
        "client_filesystem_read_file"
        | "client_filesystem_read_directory"
        | "client_git_command"
        | "client_git_diff_snapshot" => {
            let forwarded = method.replacen("client_", "executor_", 1);
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": forwarded,
                    "params": params.clone(),
                })
                .to_string(),
            )
            .await;
        }
        "executor_filesystem_read_file_result"
        | "executor_filesystem_read_directory_result"
        | "executor_git_command_result"
        | "executor_git_diff_snapshot_result" => {
            let forwarded = method.replacen("executor_", "client_", 1);
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": forwarded,
                    "params": params.clone(),
                })
                .to_string(),
            )
            .await;
        }
        // amp's executor sends `tool_progress` while a long-running
        // tool (Bash output, large Read, …) emits intermediate state.
        // Both executor and TUI share the WS connection but TUI
        // observers only see frames *from* the server, so the bridge
        // has to echo it back for `onToolProgress` to fire and the
        // TUI to render partial output. Schema:
        // `{toolCallId, progress?: snapshot|unknown, parentToolCallId?}`.
        "tool_progress" => {
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "tool_progress",
                    "params": params.clone(),
                })
                .to_string(),
            )
            .await;
        }
        // amp's executor sends this when its local policy gates a
        // tool call on human approval (e.g. Bash on a risky pattern).
        // Add the approval entry to our queue and rebroadcast
        // `tool_approval_queue` so amp's TUI observers (in the same
        // process) render the prompt. The matching response comes
        // back as `client_tool_approval_response`.
        "executor_tool_approval_request" => {
            if let Some(approval) = params.get("approval").cloned() {
                let tool_call_id = approval
                    .get("toolCallId")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                if let Some(id) = &tool_call_id {
                    ws_state
                        .pending_approvals
                        .retain(|a| a.get("toolCallId").and_then(|v| v.as_str()) != Some(id));
                }
                ws_state.pending_approvals.push(approval);
                emit_tool_approval_queue(write_tx, trace, full_path, ws_state).await;
                // Flip status to "Waiting for approval" so the user
                // sees the agent is paused for their input, not stuck.
                emit_agent_state(
                    write_tx,
                    trace,
                    full_path,
                    "awaiting_approval",
                    tool_call_id.as_deref(),
                )
                .await;
            }
        }
        // amp's TUI sends this with the user's approve/deny verdict.
        // Forward to the executor as `executor_tool_approval_response`
        // (the schema the executor's observer expects) so the executor
        // can either proceed with the lease or skip the tool, then
        // rebroadcast the now-shrunken queue.
        "client_tool_approval_response" => {
            let tool_call_id = params
                .get("toolCallId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !tool_call_id.is_empty() {
                ws_state.pending_approvals.retain(|a| {
                    a.get("toolCallId").and_then(|v| v.as_str()) != Some(&tool_call_id)
                });
            }
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "executor_tool_approval_response",
                    "params": params.clone(),
                })
                .to_string(),
            )
            .await;
            emit_tool_approval_queue(write_tx, trace, full_path, ws_state).await;
            // Clear the "awaiting_approval" indicator. If more
            // approvals are still pending, return to that state;
            // otherwise the executor is now free to run leases so
            // running_tools is the closer match.
            let next = if ws_state.pending_approvals.is_empty() {
                "running_tools"
            } else {
                "awaiting_approval"
            };
            emit_agent_state(write_tx, trace, full_path, next, None).await;
        }
        // Drop the last assistant turn (plus any tool_result fold-up
        // user messages between it and its triggering user message)
        // and re-run agent_turn against the trimmed history. amp's UI
        // sends this from `retryAgentLoop()` — typically after an
        // error_set or when the user wants a different answer.
        "client_retry" => {
            if !rewind_to_before_last_assistant(ws_state) {
                return;
            }
            let assistant_id = agent_turn_start(ws_state, write_tx, trace, full_path).await;
            if ws_state.bootstrap_complete {
                agent_turn_finish(state, ws_state, assistant_id, write_tx, trace, full_path).await;
            } else {
                ws_state.pending_assistant_ids.push(assistant_id);
            }
        }
        // Replace a previous user message's content and resume the
        // conversation from that point — drops every message after
        // the edit (assistant turns + tool_result fold-ups) so the
        // model doesn't see contradictory context. agentMode +
        // reasoningEffort on the frame override the captured WsState
        // values just like a fresh client_append_user_msg.
        "client_edit_message" => {
            let Some(message_id) = params.get("messageId").and_then(|v| v.as_str()) else {
                return;
            };
            let new_content = params.get("content").cloned().unwrap_or(json!([]));
            if !rewind_to_user_message(ws_state, message_id, new_content.clone()) {
                return;
            }
            if let Some(mode) = params.get("agentMode") {
                ws_state.last_agent_mode = Some(mode.clone());
            }
            if let Some(effort) = params
                .get("reasoningEffort")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                ws_state.last_reasoning_effort = Some(effort.to_string());
            }
            // Re-emit message_added for the edited user message so
            // amp's UI re-renders it with the new content + the
            // editId echo.
            let edited = ws_state
                .persisted_messages
                .last()
                .cloned()
                .unwrap_or(json!({}));
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "message_updated",
                    "params": {"message": edited, "seq": ws_state.next_seq()},
                })
                .to_string(),
            )
            .await;
            persist_thread(state, ws_state).await;
            let assistant_id = agent_turn_start(ws_state, write_tx, trace, full_path).await;
            if ws_state.bootstrap_complete {
                agent_turn_finish(state, ws_state, assistant_id, write_tx, trace, full_path).await;
            } else {
                ws_state.pending_assistant_ids.push(assistant_id);
            }
        }
        // amp's TUI fires this when the user (or amp's auto-title-gen)
        // names the thread. The new title shows up in listThreads + the
        // sidebar; persist it immediately so a relaunch sees the same
        // name without waiting for the next turn's persist_thread.
        "client_set_thread_title" => {
            if let Some(title) = params.get("title").and_then(|v| v.as_str()) {
                let trimmed = title.trim();
                ws_state.title = if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                };
                persist_thread(state, ws_state).await;
            }
        }
        // The executor finished running a tool we emitted. Record the
        // result; when all pending tool_uses resolve, fold them into a
        // user message and run another agent turn.
        "executor_tool_result" => {
            let Some(tool_use_id) = params.get("toolCallId").and_then(|v| v.as_str()) else {
                return;
            };
            let tool_use_id_owned = tool_use_id.to_string();
            // Accumulate in amp's neo shape (`{toolUseID, run}`) — that's
            // what we persist, and what `amp threads continue` reloads.
            // `run` is the executor's own run object (status + result +
            // any extras its renderer wants). amp dereferences
            // `block.run.status` on reload, so guarantee an object with a
            // status even if the executor sent neither. The Anthropic
            // shape the model needs is derived later via
            // neo_block_to_anthropic.
            let mut run = params
                .get("run")
                .cloned()
                .filter(serde_json::Value::is_object)
                .unwrap_or_else(|| json!({}));
            if run.get("status").and_then(|s| s.as_str()).is_none() {
                run["status"] = json!("error");
            }
            ws_state.tool_results.push(json!({
                "type": "tool_result",
                "toolUseID": tool_use_id,
                "run": run,
            }));
            ws_state.pending_tool_uses.remove(tool_use_id);

            // Confirm receipt to amp's executor. Without this, amp's
            // executor leaves the tool in a "result-sent-but-unacked"
            // limbo and its TUI keeps showing the per-tool `::` spinner
            // instead of settling on `$`. amp's binary only RECEIVES
            // `executor_tool_result_ack` (via onExecutorToolResultAck);
            // it expects the actor (us) to emit it after processing.
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "executor_tool_result_ack",
                    "params": {"toolCallId": tool_use_id_owned},
                })
                .to_string(),
            )
            .await;

            if ws_state.pending_tool_uses.is_empty() && !ws_state.tool_results.is_empty() {
                // All tools resolved — add a user message with the
                // accumulated tool_result blocks and run another turn.
                // Bootstrap is always complete by this point because
                // tools_register fires before any tool can run.
                let neo_blocks: Vec<_> = ws_state.tool_results.drain(..).collect();
                // The model gets the Anthropic shape; the persisted
                // fold-up (below) keeps the neo shape amp reloads.
                let anthropic_blocks: Vec<_> = neo_blocks.iter().map(block_to_anthropic).collect();
                ws_state.messages.push(json!({
                    "role": "user",
                    "content": anthropic_blocks,
                }));
                // Persist the tool-result fold-up too. Without this,
                // `amp threads continue T-<id>` resumes with no record
                // of what tools returned — the model would see its
                // own tool_use blocks but think they came back empty,
                // and likely re-run the same commands. The share log
                // / `aivo logs show` also loses the tool output the
                // user actually saw during the live session.
                let fold_msg = json!({
                    "threadId": ws_state.thread_id,
                    "role": "user",
                    "messageId": new_message_id(),
                    "content": neo_blocks,
                    "createdAt": chrono::Utc::now().to_rfc3339(),
                });
                ws_state.persisted_messages.push(fold_msg);
                ws_state
                    .persisted_to_msgs_idx
                    .push(ws_state.messages.len() - 1);
                persist_thread(state, ws_state).await;
                let assistant_id = agent_turn_start(ws_state, write_tx, trace, full_path).await;
                agent_turn_finish(state, ws_state, assistant_id, write_tx, trace, full_path).await;
            }
        }
        // Unknown `client_*` frame — amp's client protocol moved past
        // what this bridge handles. Every load-bearing client method
        // (append_user_msg, resume, retry, edit_message, tool_approval,
        // …) is matched explicitly above, so an unrecognized one means
        // a turn the user expects will silently never fire. Surface one
        // friendly error into amp's error bar instead of leaving the
        // outbox to retry forever. Non-`client_*` unknowns are amp's own
        // notifications we don't need; keep dropping those quietly.
        other if other.starts_with("client_") && !ws_state.protocol_warned => {
            ws_state.protocol_warned = true;
            eprintln!("[amp-bridge] UNHANDLED client method: {other}");
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "error_set",
                    "params": {
                        "seq": ws_state.next_seq(),
                        "error": {"message": format!(
                            "aivo's amp bridge doesn't recognize this amp version's protocol \
                             (unhandled message: {other}). Update aivo with `aivo update`, or \
                             install a known-good amp release."
                        )},
                    },
                })
                .to_string(),
            )
            .await;
            emit_agent_state(write_tx, trace, full_path, "error", None).await;
        }
        _ => {}
    }
}

/// Runs one iteration of the agent loop: calls the upstream LLM with
/// the accumulated state, then emits either a final `message_added`
/// (text-only response) or a `message_added` carrying tool_use blocks
/// (and registers them in `pending_tool_uses` so the next
/// `executor_tool_result` events drive the loop forward).
/// Writes the current thread state to
/// `~/.config/aivo/amp-threads/T-<id>.json` so `amp threads continue
/// T-<id>` / `listThreads` work across `aivo run amp` invocations. In
/// old amp, the CLI uploaded this payload via the `uploadThread` RPC
/// after every turn; in neo the CLI doesn't upload, so we materialize
/// it ourselves from the WS-side `persisted_messages`.
async fn persist_thread(state: &AmpBridgeState, ws_state: &WsState) {
    let payload = ws_state.thread_payload();
    if let Err(err) = amp_threads::save_thread(&state.config.threads_dir, &payload).await {
        eprintln!("[amp-bridge] persist_thread failed: {err}");
    }
}

/// Emits an `agent_state: "working"` event + placeholder assistant
/// message so amp's TUI shows the status indicator and the
/// in-progress slot immediately. Returns the assistant `messageId`
/// the caller passes to `agent_turn_finish` once tools are ready
/// (or right away if already bootstrapped). amp's status bar reads
/// from `agent_state` events with values from the `I$` enum: idle /
/// working / streaming / tool_use / running_tools /
/// awaiting_approval / error.
async fn agent_turn_start(
    ws_state: &mut WsState,
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
) -> String {
    let assistant_message_id = new_message_id();
    emit_agent_state(
        write_tx,
        trace,
        full_path,
        "working",
        Some(&assistant_message_id),
    )
    .await;
    let placeholder = json!({
        "threadId": ws_state.thread_id,
        "role": "assistant",
        "messageId": assistant_message_id.clone(),
        "content": [],
    });
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "message_added",
            "params": {"message": placeholder, "seq": ws_state.next_seq()},
        })
        .to_string(),
    )
    .await;
    assistant_message_id
}

/// Closes out an assistant turn that was cancelled mid-flight. Emits
/// the `cancelled` event amp's UI listens for to clear the spinner,
/// a final `message_updated` with `state:{type:"cancelled"}`, and
/// resets the agent_state back to idle. Doesn't touch tool_leases /
/// next-turn — the user explicitly stopped the loop.
async fn finish_turn_as_cancelled(
    ws_state: &mut WsState,
    assistant_message_id: &str,
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
) {
    let seq = ws_state.next_seq();
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "cancelled",
            "params": {"seq": seq, "messageId": assistant_message_id},
        })
        .to_string(),
    )
    .await;
    let final_msg = json!({
        "threadId": ws_state.thread_id,
        "role": "assistant",
        "messageId": assistant_message_id,
        "content": [],
        "state": {"type": "cancelled"},
        "createdAt": chrono::Utc::now().to_rfc3339(),
    });
    ws_state.persisted_messages.push(final_msg.clone());
    ws_state
        .persisted_to_msgs_idx
        .push(ws_state.messages.len().saturating_sub(1));
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "message_updated",
            "params": {"message": final_msg, "seq": ws_state.next_seq()},
        })
        .to_string(),
    )
    .await;
    emit_agent_state(write_tx, trace, full_path, "idle", None).await;
    // Reset so a follow-up turn doesn't pre-cancel.
    ws_state.cancel_flag.store(false, Ordering::SeqCst);
}

async fn emit_inference_tools(
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    ws_state: &WsState,
    assistant_message_id: &str,
) {
    let tool_names: Vec<String> = ws_state
        .tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .filter(|n| !n.is_empty())
        .collect();
    let agent_mode = ws_state
        .last_agent_mode
        .clone()
        .unwrap_or_else(|| json!("smart"));
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "inference_tools",
            "params": {
                "messageId": assistant_message_id,
                "agentMode": agent_mode,
                "tools": tool_names,
            },
        })
        .to_string(),
    )
    .await;
}

/// Trims state back to "just acked the last user message" so the
/// caller can replay an agent turn. Drops every `persisted_messages`
/// entry from the most recent assistant onward (plus any tool_result
/// fold-up user messages in `messages` between that assistant and
/// its triggering user message). Clears the in-flight bookkeeping
/// the dropped turn owned. Returns `false` (and leaves state
/// untouched) when there's nothing to rewind to.
fn rewind_to_before_last_assistant(ws_state: &mut WsState) -> bool {
    let last_assistant_persisted = ws_state
        .persisted_messages
        .iter()
        .rposition(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"));
    let Some(k) = last_assistant_persisted else {
        return false;
    };
    let msgs_cutoff = ws_state.persisted_to_msgs_idx.get(k).copied().unwrap_or(0);
    ws_state.persisted_messages.truncate(k);
    ws_state.persisted_to_msgs_idx.truncate(k);
    ws_state.messages.truncate(msgs_cutoff);
    clear_in_flight_turn_state(ws_state);
    true
}

/// Rewinds to a specific user message identified by `message_id` and
/// replaces its content. Truncates everything after it in both
/// `messages` and `persisted_messages`, then pushes a fresh user
/// entry with `new_content` so the caller can run a follow-up turn.
/// Returns `false` (and leaves state untouched) when the message id
/// isn't found or doesn't belong to a user message.
fn rewind_to_user_message(
    ws_state: &mut WsState,
    message_id: &str,
    new_content: serde_json::Value,
) -> bool {
    let k = match ws_state.persisted_messages.iter().position(|m| {
        m.get("messageId").and_then(|v| v.as_str()) == Some(message_id)
            && m.get("role").and_then(|v| v.as_str()) == Some("user")
    }) {
        Some(idx) => idx,
        None => return false,
    };
    let msgs_cutoff = ws_state.persisted_to_msgs_idx.get(k).copied().unwrap_or(0);
    ws_state.persisted_messages.truncate(k);
    ws_state.persisted_to_msgs_idx.truncate(k);
    ws_state.messages.truncate(msgs_cutoff);
    clear_in_flight_turn_state(ws_state);

    // Rebuild the user message in both vecs with the new content.
    let mut user_msg = json!({
        "threadId": ws_state.thread_id,
        "role": "user",
        "messageId": message_id,
        "content": new_content.clone(),
        "createdAt": chrono::Utc::now().to_rfc3339(),
    });
    if let Some(mode) = &ws_state.last_agent_mode {
        user_msg["agentMode"] = mode.clone();
    }
    ws_state
        .messages
        .push(json!({"role": "user", "content": new_content}));
    ws_state.persisted_messages.push(user_msg);
    ws_state
        .persisted_to_msgs_idx
        .push(ws_state.messages.len() - 1);
    true
}

/// Clears the bookkeeping that belonged to a turn we're about to
/// discard (rewind paths). Tool leases that hadn't completed, queued
/// approvals, accumulated tool_results, and any deferred-bootstrap
/// assistant ids all become invalid once their owning assistant
/// message is gone.
fn clear_in_flight_turn_state(ws_state: &mut WsState) {
    ws_state.pending_tool_uses.clear();
    ws_state.tool_results.clear();
    ws_state.pending_approvals.clear();
    ws_state.pending_assistant_ids.clear();
    ws_state.cancel_flag.store(false, Ordering::SeqCst);
}

/// Broadcasts the current pending-approvals list. amp's TUI dispatches
/// `tool_approval_queue` to `onToolApprovalQueue` observers which feed
/// `pendingApprovals` — that's what drives the approval prompt UI.
async fn emit_tool_approval_queue(
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    ws_state: &WsState,
) {
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "tool_approval_queue",
            "params": {"approvals": ws_state.pending_approvals.clone()},
        })
        .to_string(),
    )
    .await;
}

async fn emit_agent_state(
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    state: &str,
    message_id: Option<&str>,
) {
    let mut params = json!({"state": state});
    if let Some(mid) = message_id {
        params["messageId"] = json!(mid);
    }
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "agent_state",
            "params": params,
        })
        .to_string(),
    )
    .await;
}

/// Runs the LLM call and emits the final `message_updated` (plus any
/// `tool_lease` events) for an assistant turn whose placeholder was
/// already emitted by `agent_turn_start`.
async fn agent_turn_finish(
    state: &AmpBridgeState,
    ws_state: &mut WsState,
    assistant_message_id: String,
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
) {
    // Clear any cancel flag left over from a prior turn so a stale
    // `client_cancel` (e.g. arrived after the previous turn finished
    // but before this one started) doesn't pre-cancel the new run.
    ws_state.cancel_flag.store(false, Ordering::SeqCst);

    // Tell amp's status bar which tools are active for this turn. Amp
    // displays the count next to the agent mode (e.g. "deep · 14 tools")
    // and tracks per-mode availability across turns. Emit before the
    // LLM call so the indicator updates as soon as the spinner starts.
    emit_inference_tools(write_tx, trace, full_path, ws_state, &assistant_message_id).await;

    let cancel_flag = ws_state.cancel_flag.clone();
    let agent_mode_owned = ws_state
        .last_agent_mode
        .as_ref()
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let call_result = call_upstream_chat_streaming(
        state,
        &ws_state.messages,
        &ws_state.tools,
        &assistant_message_id,
        write_tx,
        trace,
        full_path,
        &cancel_flag,
        ws_state.last_reasoning_effort.as_deref(),
        ws_state.environment.as_ref(),
        &ws_state.skills,
        &ws_state.guidance_files,
        agent_mode_owned.as_deref(),
    )
    .await;

    // Cancel takes precedence over any error/result: if the user
    // bailed mid-turn we close the assistant slot as cancelled rather
    // than surfacing a spurious "stream aborted" upstream error in the
    // amp error bar.
    if cancel_flag.load(Ordering::SeqCst) {
        finish_turn_as_cancelled(ws_state, &assistant_message_id, write_tx, trace, full_path).await;
        return;
    }

    let (blocks, turn_usage) = match call_result {
        Ok(b) => b,
        Err(e) => {
            let err_msg = e.to_string();
            // amp's UI displays `error_set` notifications in a dedicated
            // error bar; without it the user would only see the failure as
            // body text inside the assistant turn. The follow-up
            // message_updated still fires so the in-progress slot closes
            // out instead of spinning forever.
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "error_set",
                    "params": {
                        "seq": ws_state.next_seq(),
                        "error": {"message": err_msg.clone()},
                    },
                })
                .to_string(),
            )
            .await;
            let final_msg = json!({
                "threadId": ws_state.thread_id,
                "role": "assistant",
                "messageId": assistant_message_id,
                "content": [{"type": "text", "text": format!("(aivo bridge upstream error: {err_msg})")}],
                "state": {"type": "complete"},
                "createdAt": chrono::Utc::now().to_rfc3339(),
            });
            send_event(
                write_tx,
                trace,
                full_path,
                json!({
                    "jsonrpc": "2.0",
                    "method": "message_updated",
                    "params": {"message": final_msg, "seq": ws_state.next_seq()},
                })
                .to_string(),
            )
            .await;
            // Show "Error" in the status badge before going idle, so
            // the user has a visual cue something failed even if they
            // missed the error_set frame in the error bar. (Some amp
            // builds clear active errors on the next user input;
            // sitting on `error` until then is the load-bearing signal.)
            emit_agent_state(write_tx, trace, full_path, "error", None).await;
            return;
        }
    };

    // Collect tool_lease events for every tool_use block. The id was
    // already normalized to amp's `TU-<22>` shape upstream (SSE block-
    // start in stream_anthropic_sse, or the Buffered-branch normalization
    // in call_upstream_chat_streaming); ensure_tool_use_id is idempotent
    // so calling it again here is a safety net for any path that returns
    // blocks without going through those points.
    let mut rewritten_blocks = Vec::with_capacity(blocks.len());
    let mut tool_leases: Vec<serde_json::Value> = Vec::new();
    for mut block in blocks {
        if let Some(tu_id) = ensure_tool_use_id(&mut block) {
            ws_state.pending_tool_uses.insert(tu_id.clone());
            tool_leases.push(json!({
                "toolCallId": tu_id,
                "toolName": block.get("name").cloned().unwrap_or(json!("")),
                "args": block.get("input").cloned().unwrap_or(json!({})),
                "messageId": assistant_message_id.clone(),
            }));
        }
        rewritten_blocks.push(block);
    }

    ws_state.messages.push(json!({
        "role": "assistant",
        "content": rewritten_blocks.clone(),
    }));

    let has_tools = !tool_leases.is_empty();
    let mut final_msg = json!({
        "threadId": ws_state.thread_id,
        "role": "assistant",
        "messageId": assistant_message_id.clone(),
        "content": rewritten_blocks,
        "state": {"type": "complete"},
        "createdAt": chrono::Utc::now().to_rfc3339(),
    });
    // Carry the turn's token usage on the persisted message (pre-neo amp's own
    // schema), so `--aivo-stats` reads it back per session.
    if let Some(u) = turn_usage {
        final_msg["usage"] = u;
    }
    ws_state.persisted_messages.push(final_msg.clone());
    ws_state
        .persisted_to_msgs_idx
        .push(ws_state.messages.len() - 1);
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "message_updated",
            "params": {"message": final_msg, "seq": ws_state.next_seq()},
        })
        .to_string(),
    )
    .await;
    persist_thread(state, ws_state).await;
    // tool_use blocks in messages are UI/history only — the executor
    // only RUNS a tool when it receives a `tool_lease` notification.
    for lease in tool_leases {
        send_event(
            write_tx,
            trace,
            full_path,
            json!({
                "jsonrpc": "2.0",
                "method": "tool_lease",
                "params": lease,
            })
            .to_string(),
        )
        .await;
    }
    // Update the status bar: if the model emitted tool_use we move to
    // `running_tools` until the executor reports back; otherwise the
    // turn is done and we go back to idle.
    let next_state = if has_tools { "running_tools" } else { "idle" };
    emit_agent_state(
        write_tx,
        trace,
        full_path,
        next_state,
        Some(&assistant_message_id),
    )
    .await;
}

/// Returns the user's override prompt for `mode` if they've written one,
/// else the per-mode override, else `None`. Search path:
///   `~/.config/aivo/amp/prompts/<mode>.md`
///   `~/.config/aivo/amp/prompts/default.md`
/// File reads are best-effort — missing/unreadable files just fall
/// through to the built-in. Live-loaded each turn so editing the file
/// doesn't require an amp restart.
async fn resolve_prompt_override(mode: Option<&str>) -> Option<String> {
    let home = aivo::services::system_env::home_dir()?;
    let dir = home
        .join(".config")
        .join("aivo")
        .join("amp")
        .join("prompts");
    if let Some(m) = mode.filter(|s| !s.is_empty()) {
        let path = dir.join(format!("{m}.md"));
        if let Ok(s) = tokio::fs::read_to_string(&path).await {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    let path = dir.join("default.md");
    if let Ok(s) = tokio::fs::read_to_string(&path).await {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Base prompt — covers identity, tool-use philosophy, code-change
/// discipline, communication, verification, ambiguity, failure
/// recovery, workspace awareness. Authored independently for the
/// aivo bridge; amp's per-mode prompts were used only as a coverage
/// checklist (topic-level), never as a source of phrasing.
const BASE_PROMPT: &str = "\
You are a coding agent operating inside the Sourcegraph amp CLI. Your \
requests are routed through the aivo bridge to a user-configured LLM \
provider, which may not be Claude. Tool definitions are registered by \
amp's local executor; use them as the primary way to inspect and \
modify the workspace.\n\
\n\
## Tool use\n\
- Prefer tool calls over describing what you would do. If the next \
useful step is a Read, an Edit, or a Bash command, do it; don't \
narrate intent first.\n\
- When several tool calls are independent of each other (no result \
feeds the next), emit them in a single response so they run in \
parallel.\n\
- Don't speculate about something a single tool call can verify.\n\
- When invoking an unfamiliar CLI, read `--help` on the specific \
subcommand you're about to run, not just the parent command. \
Subcommand flags (e.g. `--password-stdin`, `--input-file`, \
`--from-env`, `--non-interactive`) are usually the difference \
between a one-shot success and a long thrash through hacks.\n\
\n\
## Editing code\n\
- Match the surrounding style — naming, indentation, error-handling \
patterns, comment density. Don't introduce a different idiom for the \
sake of \"cleaner\" code.\n\
- Keep the change minimal in scope. Don't refactor neighbouring code \
that's unrelated to the request.\n\
- Don't add comments unless the WHY is non-obvious. \"// loop over \
items\" above `for item in items` adds noise. \"// retry once because \
the upstream returns 429 on first call after auth\" adds value.\n\
- Don't introduce abstractions for hypothetical future requirements. \
Three similar lines beats a premature helper.\n\
\n\
## Communication\n\
- Be terse. Skip \"Sure!\" / \"I'll go ahead and...\" preambles.\n\
- Don't recap what you just did at the end of a turn. The diff and \
tool outputs are the record.\n\
- State load-bearing assumptions explicitly when you make them.\n\
- When citing a file location, use `path:line` so the user can click \
through.\n\
\n\
## Verification\n\
- For non-trivial changes, run the project's typecheck / test / lint \
before claiming the task is done. Use the workspace context below to \
figure out which one applies.\n\
- Stop verifying once the verification stops producing new \
information. Don't run the same check three times.\n\
\n\
## Handling ambiguity\n\
- Default to making a sensible assumption and proceeding. State the \
assumption in one line so the user can correct it.\n\
- Only ask when the alternatives have meaningfully different blast \
radius (e.g. \"delete the file or rename it?\") or when no reasonable \
default exists.\n\
\n\
## When stuck\n\
- Re-read the error message carefully. Search for context in the \
codebase before guessing.\n\
- If two attempts at the same approach failed, change approach \
instead of trying a third variant.\n\
- A TTY / `/dev/tty` / \"device not configured\" error means the \
command wants interactive input. Before reaching for `expect`, \
`script`, pty wrappers, or other hacks, re-read the command's \
`--help` for a non-interactive flag (`--password-stdin`, \
`--yes`, `--batch`, `--no-prompt`, env-var alternatives). The flag \
almost always exists; missing it on the first try and immediately \
piling on workarounds is the failure mode.\n\
- Report blockers clearly: what you tried, what failed, what would \
unblock you.\n\
\n\
## Safety\n\
- Destructive operations (`rm -rf`, `git push --force`, dropping \
database state, deleting branches) require an explicit user request. \
Don't reach for them as a shortcut around an obstacle.\n\
- Investigate unfamiliar state (unexpected files, uncommitted \
changes, lock files) before deleting or overwriting.\n\
\n\
## Workspace\n\
- The user has registered specific tools via amp's executor — those \
are listed below. Use them in preference to ad-hoc Bash where one \
applies.\n\
- If project guidance (AGENTS.md and friends) is provided below, \
follow it. It outranks generic conventions.";

/// Per-mode addendum: amp's agent modes (rush/smart/deep/large/
/// frontier) bias the agent toward different speed/depth trade-offs.
/// The base prompt is the same; this is the calibration delta.
fn mode_addendum(mode: Option<&str>) -> &'static str {
    match mode.unwrap_or("smart") {
        "rush" => {
            "\n\n## Mode: rush\n\
- Optimize for the shortest path to a working result.\n\
- For small / well-scoped changes, skip the verification step unless \
the change touches something risky.\n\
- Default to acting over reading more context. If you've seen enough \
to make a defensible attempt, attempt it.\n\
- Single-shot answers are fine when the question is genuinely simple."
        }
        "deep" => {
            "\n\n## Mode: deep\n\
- Think before acting. Spend the reasoning budget on understanding \
the problem before you reach for tools.\n\
- Verify more aggressively than usual: run tests, re-read the \
changed file, check for callers of edited functions.\n\
- When the task has hidden complexity, surface it to the user \
before committing to an approach."
        }
        "large" => {
            "\n\n## Mode: large\n\
- The task likely spans multiple files / multiple areas. Plan before \
executing.\n\
- Output a short plan (3-7 bullets) when the work isn't obviously a \
single edit, then work through it. Update the plan if you discover \
the original was wrong.\n\
- Group related edits into single tool calls where possible to keep \
the diff coherent."
        }
        "frontier" => {
            "\n\n## Mode: frontier\n\
- This is a novel or exploratory task. Don't pattern-match a \
familiar solution onto it without checking the fit.\n\
- Favor reasoning over recall: the answer probably isn't in your \
training data verbatim.\n\
- When you discover a non-obvious constraint, name it explicitly so \
the user sees what you learned."
        }
        _ => "", // smart / default / unknown — no addendum, base prompt is the balanced version.
    }
}

/// Builds the system prompt handed to the upstream LLM. The base
/// prompt is either a user-authored override loaded from
/// `~/.config/aivo/amp/prompts/<mode>.md` (see `resolve_prompt_override`)
/// or our built-in BASE_PROMPT + mode_addendum, in that order. The
/// workspace/guidance/skills/tools sections always append regardless,
/// so the user's override only replaces the base directives — they
/// still get live workspace context for free.
fn build_system_prompt(
    tools: &[serde_json::Value],
    environment: Option<&serde_json::Value>,
    skills: &[serde_json::Value],
    guidance_files: &[serde_json::Value],
    agent_mode: Option<&str>,
    override_base: Option<&str>,
) -> String {
    let mut s = match override_base {
        Some(text) => text.to_string(),
        None => {
            let mut base = String::from(BASE_PROMPT);
            base.push_str(mode_addendum(agent_mode));
            base
        }
    };

    // Workspace context surfaced by `executor_environment_snapshot`.
    // Including cwd + branch saves the model from running pwd/git
    // status on every cold turn.
    if let Some(env) = environment {
        let cwd = env
            .get("workingDirectory")
            .or_else(|| env.get("workspaceRoot"))
            .and_then(|v| v.as_str());
        let branch = env
            .get("git")
            .and_then(|g| g.get("branch"))
            .and_then(|v| v.as_str());
        if cwd.is_some() || branch.is_some() {
            s.push_str("\n\nWorkspace:\n");
            if let Some(c) = cwd {
                s.push_str(&format!("- cwd: {c}\n"));
            }
            if let Some(b) = branch {
                s.push_str(&format!("- git branch: {b}\n"));
            }
        }
    }

    // AGENTS.md / amp/agents.md guidance from
    // `executor_guidance_snapshot`. These are user-authored
    // instructions for the agent; including their content (not just
    // paths) is the whole point — otherwise the user's per-repo
    // conventions get ignored.
    if !guidance_files.is_empty() {
        s.push_str("\n\nProject guidance (from AGENTS.md / amp/agents.md):\n");
        for file in guidance_files.iter().take(8) {
            let uri = file.get("uri").and_then(|v| v.as_str()).unwrap_or("");
            let content = file
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if content.is_empty() {
                continue;
            }
            s.push_str(&format!("\n### {uri}\n"));
            // Cap each guidance file at ~4KB to keep the system
            // prompt bounded; AGENTS.md files can be much larger.
            let truncated: String = content.chars().take(4096).collect();
            s.push_str(&truncated);
            if content.chars().count() > 4096 {
                s.push_str("\n…(truncated)");
            }
            s.push('\n');
        }
    }

    // User-defined skills (custom commands). Naming them lets the
    // model suggest "you can run /<skill>" — skill execution itself
    // is amp's concern, not the bridge's.
    if !skills.is_empty() {
        s.push_str("\n\nAvailable user skills:\n");
        for skill in skills.iter().take(20) {
            let name = skill
                .get("name")
                .and_then(|v| v.as_str())
                .or_else(|| {
                    skill
                        .get("frontmatter")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                })
                .unwrap_or("(unnamed)");
            let desc = skill
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if desc.is_empty() {
                s.push_str(&format!("- {name}\n"));
            } else {
                s.push_str(&format!("- {name}: {desc}\n"));
            }
        }
    }

    if !tools.is_empty() {
        s.push_str("\n\nAvailable tools:\n");
        for tool in tools.iter().take(40) {
            let name = tool
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("(unnamed)");
            s.push_str("- ");
            s.push_str(name);
            s.push('\n');
        }
        if tools.len() > 40 {
            s.push_str(&format!("- ... and {} more\n", tools.len() - 40));
        }
    }
    s
}

/// Streams the upstream LLM call, emitting a `delta` event to amp for
/// every Anthropic SSE chunk that mutates the assistant message. The
/// returned `Vec<Value>` is the final accumulated content blocks, used
/// by the caller for history and tool_use bookkeeping.
///
/// Routes through the bridge's own `dispatch` via a synthetic `POST
/// /api/provider/anthropic/v1/messages` so the existing in-process
/// Anthropic→OpenAI translator, `force_model` rewrite, and auth
/// injection all apply.
#[allow(clippy::too_many_arguments)]
async fn call_upstream_chat_streaming(
    state: &AmpBridgeState,
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
    assistant_message_id: &str,
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    cancel_flag: &Arc<AtomicBool>,
    reasoning_effort: Option<&str>,
    environment: Option<&serde_json::Value>,
    skills: &[serde_json::Value],
    guidance_files: &[serde_json::Value],
    agent_mode: Option<&str>,
) -> Result<(Vec<serde_json::Value>, Option<serde_json::Value>)> {
    let model = state
        .config
        .force_model
        .clone()
        .unwrap_or_else(|| "claude-haiku-4-5-20251001".to_string());

    // Best-effort load the user's prompt override. None means we use
    // the built-in BASE_PROMPT + mode_addendum. Live-loaded so editing
    // ~/.config/aivo/amp/prompts/<mode>.md takes effect on the next
    // turn without an amp restart.
    let override_base = resolve_prompt_override(agent_mode).await;

    let mut body = json!({
        "model": model,
        "max_tokens": 4096,
        "system": build_system_prompt(
            tools,
            environment,
            skills,
            guidance_files,
            agent_mode,
            override_base.as_deref(),
        ),
        "messages": messages,
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    // Forward amp's reasoningEffort through the Anthropic→OpenAI
    // translator's effort extraction pipeline. `output_config.effort`
    // is the canonical Anthropic form the extractor honors; "none"
    // drops effort entirely.
    if let Some(level) = reasoning_effort.filter(|v| matches!(*v, "low" | "medium" | "high")) {
        body["output_config"] = json!({"effort": level});
    }
    let body_str = serde_json::to_string(&body)?;

    let fake_request = format!(
        "POST /api/provider/anthropic/v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{body_str}"
    );

    let resp = dispatch(
        state,
        &fake_request,
        "POST",
        "/api/provider/anthropic/v1/messages",
        &body_str,
    )
    .await?;

    match resp {
        // Translator returned a buffered JSON response (non-streaming
        // upstream, or the translator chose to buffer). Synthesize
        // progressive deltas by growing the visible text block(s) in
        // chunks so amp's UI doesn't see a wall-of-text appear at once,
        // then close out with a `complete` delta.
        BridgeResponse::Buffered { status, body, .. } => {
            if !(200..300).contains(&status) {
                anyhow::bail!(
                    "upstream {status}: {}",
                    body.chars().take(400).collect::<String>()
                );
            }
            let mut turn_usage = None;
            if let Some(usage) = crate::usage::parse_token_usage(body.as_bytes()) {
                record_amp_usage(&state.config, &model, &usage).await;
                turn_usage = Some(amp_message_usage(&model, &usage));
            }
            let parsed: serde_json::Value = serde_json::from_str(&body)
                .map_err(|e| anyhow::anyhow!("parse upstream JSON: {e}"))?;
            let mut blocks: Vec<serde_json::Value> = parsed
                .get("content")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if blocks.is_empty() {
                anyhow::bail!("empty assistant response");
            }
            for block in blocks.iter_mut() {
                ensure_tool_use_id(block);
            }
            emit_buffered_blocks_progressively(
                write_tx,
                trace,
                full_path,
                assistant_message_id,
                &blocks,
                cancel_flag,
            )
            .await;
            Ok((blocks, turn_usage))
        }
        BridgeResponse::Streaming {
            status, upstream, ..
        } => {
            if !(200..300).contains(&status) {
                let body = upstream.text().await.unwrap_or_default();
                anyhow::bail!(
                    "upstream {status}: {}",
                    body.chars().take(400).collect::<String>()
                );
            }
            let mut sniffer =
                crate::usage::StreamUsageSniffer::new(state.config.usage_store.is_some());
            let blocks = stream_anthropic_sse(
                upstream,
                assistant_message_id,
                write_tx,
                trace,
                full_path,
                cancel_flag,
                &mut sniffer,
            )
            .await?;
            let mut turn_usage = None;
            if let Some(usage) = sniffer.finish() {
                record_amp_usage(&state.config, &model, &usage).await;
                turn_usage = Some(amp_message_usage(&model, &usage));
            }
            Ok((blocks, turn_usage))
        }
    }
}

/// Amp-thread-schema usage object stamped onto the persisted assistant message
/// (`model` + camelCase token fields) — the shape pre-neo amp wrote and
/// `collect_thread_sessions` reads, so `--aivo-stats` windows neo turns too
/// (neo amp's own uploads carry no usage).
fn amp_message_usage(model: &str, usage: &crate::usage::TokenUsage) -> serde_json::Value {
    json!({
        "model": model,
        "inputTokens": usage.prompt,
        "outputTokens": usage.completion,
        "cacheReadInputTokens": usage.cache_read,
        "cacheCreationInputTokens": usage.cache_creation,
    })
}

/// Record one LLM turn's token usage against the configured stats key, labeled
/// `amp`. Best-effort and fire-and-forget — never fails an amp turn.
async fn record_amp_usage(config: &AmpBridgeConfig, model: &str, usage: &crate::usage::TokenUsage) {
    if let Some(store) = &config.usage_store {
        let _ = store
            .record_tokens(
                &config.usage_key_id,
                Some("amp"),
                Some(model),
                usage.prompt,
                usage.completion,
                usage.cache_read,
                usage.cache_creation,
            )
            .await;
    }
}

/// Consumes an Anthropic SSE stream (`event: content_block_*`,
/// `message_start`, `message_stop` …), accumulates content blocks,
/// and emits a `delta` event to amp every time the message visibly
/// changes. Returns the final blocks.
async fn stream_anthropic_sse(
    upstream: reqwest::Response,
    assistant_message_id: &str,
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    cancel_flag: &Arc<AtomicBool>,
    sniffer: &mut crate::usage::StreamUsageSniffer,
) -> Result<Vec<serde_json::Value>> {
    use futures::StreamExt;
    let mut byte_stream = upstream.bytes_stream();
    let mut sse_buf = String::new();
    // Internal snapshot of the full assistant content. Kept so the
    // caller (agent_turn_finish) can emit `message_updated` with the
    // final blocks and persist them; the WS deltas we emit during the
    // stream are INCREMENTS, not snapshots — amp's IZR reducer appends
    // text/thinking on each delta, so sending snapshots produces
    // exponential duplication in the streaming content buffer.
    let mut blocks: Vec<serde_json::Value> = Vec::new();
    // Parallel array of partial JSON strings, one per block, for
    // accumulating `input_json_delta` chunks until the block closes.
    let mut tool_json_partials: Vec<String> = Vec::new();

    // Minimum wall-clock interval between emitted `generating` deltas.
    // When the translator delivers a single-chunk SSE re-frame (non-
    // streaming upstream that we wrap in SSE shape), all events parse
    // synchronously and amp would otherwise see the whole turn arrive
    // in microseconds with no visible streaming. Sleeping here for the
    // gap between emits paces it to ~125Hz, which is fast enough to
    // be invisible on real upstream streaming (tokens arrive at ~30ms
    // intervals and the elapsed check skips the sleep) but slow enough
    // to render visibly progressive on the buffered path.
    const MIN_EMIT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(8);
    let mut last_emit = std::time::Instant::now()
        .checked_sub(MIN_EMIT_INTERVAL)
        .unwrap_or_else(std::time::Instant::now);
    // amp's TUI shows "Waiting" while agent_state is "working" and
    // switches to "Thinking" / "Streaming" once we flip to
    // "streaming" (refined by the most recent delta's block kind).
    // We sit on "working" until the first delta is about to fire,
    // then promote — that way the status badge tracks reality.
    let mut emitted_streaming_state = false;
    let mut stream_error: Option<String> = None;

    'outer: while let Some(chunk_res) = byte_stream.next().await {
        if cancel_flag.load(Ordering::Relaxed) {
            break 'outer;
        }
        let chunk = chunk_res?;
        sniffer.observe(&chunk);
        sse_buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(end) = sse_buf.find("\n\n") {
            if cancel_flag.load(Ordering::Relaxed) {
                break 'outer;
            }
            let event_text: String = sse_buf.drain(..end + 2).collect();
            let Some(data) = event_text.lines().find_map(|l| l.strip_prefix("data: ")) else {
                continue;
            };
            if data == "[DONE]" {
                break 'outer;
            }
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };

            // The increment we'll emit for this event (None for events
            // that don't visibly change content, e.g. message_start).
            let mut increment: Option<(usize, serde_json::Value)> = None;

            match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "message_stop" => break 'outer,
                "error" => {
                    stream_error = Some(
                        ev.get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown upstream stream error")
                            .to_string(),
                    );
                    break 'outer;
                }
                "content_block_start" => {
                    let idx = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let mut block = ev.get("content_block").cloned().unwrap_or(json!({}));
                    if block.get("type").and_then(|t| t.as_str()) == Some("thinking") {
                        if block.get("thinking").is_none() {
                            block["thinking"] = json!("");
                        }
                        if block.get("signature").is_none() {
                            block["signature"] = json!("");
                        }
                    }
                    ensure_tool_use_id(&mut block);
                    while blocks.len() <= idx {
                        blocks.push(json!({}));
                        tool_json_partials.push(String::new());
                    }
                    blocks[idx] = block.clone();
                    // amp's IZR replaces the hidden placeholder at this
                    // index with the block as-is (UZR check). Empty
                    // text/thinking is harmless — subsequent deltas
                    // append.
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        // Tool_use needs `inputPartialJSON.json` so
                        // amp's ZZR can append input_json_deltas.
                        // `complete:false` keeps the block open.
                        let mut shell = block.clone();
                        shell["complete"] = json!(false);
                        shell["inputPartialJSON"] = json!({"json": ""});
                        increment = Some((idx, shell));
                    } else {
                        increment = Some((idx, block));
                    }
                }
                "content_block_delta" => {
                    let idx = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    if idx >= blocks.len() {
                        continue;
                    }
                    if let Some(delta) = ev.get("delta") {
                        match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                            "text_delta" => {
                                if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                                    let cur = blocks[idx]
                                        .get("text")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    blocks[idx]["text"] = json!(cur + t);
                                    // Emit only the new text — amp
                                    // appends via _.text + r.text.
                                    increment = Some((idx, json!({"type": "text", "text": t})));
                                }
                            }
                            "thinking_delta" => {
                                if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                                    let cur = blocks[idx]
                                        .get("thinking")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    blocks[idx]["thinking"] = json!(cur + t);
                                    // Empty signature on the increment
                                    // — amp's spread keeps the existing
                                    // signature unless r.signature is
                                    // truthy. Set to "" so the field is
                                    // schema-valid.
                                    increment = Some((
                                        idx,
                                        json!({
                                            "type": "thinking",
                                            "thinking": t,
                                            "signature": "",
                                        }),
                                    ));
                                }
                            }
                            "signature_delta" => {
                                if let Some(s) = delta.get("signature").and_then(|v| v.as_str()) {
                                    blocks[idx]["signature"] = json!(s);
                                    // Empty thinking on the increment
                                    // — amp's append yields existing +
                                    // "" = unchanged. The spread sets
                                    // signature to s.
                                    increment = Some((
                                        idx,
                                        json!({
                                            "type": "thinking",
                                            "thinking": "",
                                            "signature": s,
                                        }),
                                    ));
                                }
                            }
                            "input_json_delta" => {
                                if let Some(p) = delta.get("partial_json").and_then(|v| v.as_str())
                                {
                                    tool_json_partials[idx].push_str(p);
                                    let tu_id = blocks[idx]
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    // amp's ZZR appends inputPartialJSONDelta.json
                                    // to the existing inputPartialJSON.json.
                                    // Block id must match for the merge to apply.
                                    increment = Some((
                                        idx,
                                        json!({
                                            "type": "tool_use",
                                            "id": tu_id,
                                            "complete": false,
                                            "inputPartialJSONDelta": {"json": p},
                                        }),
                                    ));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_stop" => {
                    let idx = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    if idx < blocks.len()
                        && blocks[idx].get("type").and_then(|t| t.as_str()) == Some("tool_use")
                    {
                        let partial = std::mem::take(&mut tool_json_partials[idx]);
                        if let Ok(input) = serde_json::from_str::<serde_json::Value>(&partial) {
                            blocks[idx]["input"] = input.clone();
                            // Mark the tool_use complete so amp stops
                            // expecting input_json deltas. Replaces
                            // via IZR's else branch (not dLT, not
                            // text/thinking).
                            let mut shell = blocks[idx].clone();
                            shell["complete"] = json!(true);
                            increment = Some((idx, shell));
                        }
                    }
                }
                _ => {}
            }

            if let Some((idx, inc_block)) = increment {
                if !emitted_streaming_state {
                    emit_agent_state(
                        write_tx,
                        trace,
                        full_path,
                        "streaming",
                        Some(assistant_message_id),
                    )
                    .await;
                    emitted_streaming_state = true;
                }
                let elapsed = last_emit.elapsed();
                if elapsed < MIN_EMIT_INTERVAL {
                    tokio::time::sleep(MIN_EMIT_INTERVAL - elapsed).await;
                }
                emit_incremental_delta(
                    write_tx,
                    trace,
                    full_path,
                    assistant_message_id,
                    &inc_block,
                    idx,
                    "generating",
                )
                .await;
                last_emit = std::time::Instant::now();
            }
        }
    }
    let cancelled = cancel_flag.load(Ordering::Relaxed);

    if !cancelled
        && blocks.is_empty()
        && let Some(salvaged) = parse_buffered_message_blocks(&sse_buf)
    {
        sniffer.observe_json_body(&sse_buf);
        blocks = salvaged;
        emit_buffered_blocks_progressively(
            write_tx,
            trace,
            full_path,
            assistant_message_id,
            &blocks,
            cancel_flag,
        )
        .await;
        return Ok(blocks);
    }

    // Terminal delta has no blocks — amp's KZR ignores `blocks` when
    // state is "complete" / "aborted" and only updates the message
    // state. The accumulated content already lives in amp's buffer
    // from the increments above (and gets replaced by message_updated
    // shortly after).
    emit_terminal_delta(
        write_tx,
        trace,
        full_path,
        assistant_message_id,
        if cancelled { "aborted" } else { "complete" },
    )
    .await;
    if cancelled {
        return Ok(blocks);
    }
    if blocks.is_empty() {
        match stream_error {
            Some(msg) => anyhow::bail!("upstream stream error: {msg}"),
            None => anyhow::bail!("empty assistant response (no SSE blocks parsed)"),
        }
    }
    Ok(blocks)
}

fn parse_buffered_message_blocks(buf: &str) -> Option<Vec<serde_json::Value>> {
    let parsed: serde_json::Value = serde_json::from_str(buf.trim()).ok()?;
    if parsed.get("type").and_then(|t| t.as_str()) != Some("message") {
        return None;
    }
    let mut blocks = parsed.get("content")?.as_array()?.clone();
    if blocks.is_empty() {
        return None;
    }
    for block in blocks.iter_mut() {
        ensure_tool_use_id(block);
    }
    Some(blocks)
}

/// Emits a single-block incremental delta targeting `block_index`.
/// amp's IZR reducer interprets the entry as an increment to the
/// existing block at that index (appending text/thinking, applying
/// input_json deltas, or replacing on the first content_block_start).
async fn emit_incremental_delta(
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    assistant_message_id: &str,
    block: &serde_json::Value,
    block_index: usize,
    state_str: &str,
) {
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "delta",
            "params": {
                "messageId": assistant_message_id,
                "role": "assistant",
                "blocks": [block],
                "blockIndex": block_index,
                "state": state_str,
            },
        })
        .to_string(),
    )
    .await;
}

/// Emits the closing delta for a turn — no blocks, just the state
/// transition. amp's KZR ignores `blocks` when state is `complete`
/// or `aborted` (forces `blocks: []` itself), so we don't waste a
/// frame restating the content.
async fn emit_terminal_delta(
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    assistant_message_id: &str,
    state_str: &str,
) {
    send_event(
        write_tx,
        trace,
        full_path,
        json!({
            "jsonrpc": "2.0",
            "method": "delta",
            "params": {
                "messageId": assistant_message_id,
                "role": "assistant",
                "blocks": [],
                "state": state_str,
            },
        })
        .to_string(),
    )
    .await;
}

/// Emit a fully-buffered set of blocks as a sequence of `generating`
/// deltas followed by a `complete` delta, growing visible text blocks
/// chunk-by-chunk so amp's UI shows progressive output even when the
/// upstream couldn't stream. Non-text blocks (tool_use, thinking) ride
/// along whole on the first delta that needs them.
async fn emit_buffered_blocks_progressively(
    write_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    trace: Option<&Path>,
    full_path: &str,
    assistant_message_id: &str,
    blocks: &[serde_json::Value],
    cancel_flag: &Arc<AtomicBool>,
) {
    const CHUNK_CHARS: usize = 24;
    const PACE_MS: u64 = 16;

    // amp's IZR reducer needs an initial block at each index before
    // text deltas can append. Send each block's shell first (empty
    // text/thinking for text/thinking blocks, full body for tool_use)
    // so amp can build up its content array slot-by-slot.
    for (idx, b) in blocks.iter().enumerate() {
        if cancel_flag.load(Ordering::Relaxed) {
            break;
        }
        let shell = match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => json!({"type": "text", "text": ""}),
            Some("thinking") => json!({"type": "thinking", "thinking": "", "signature": ""}),
            _ => b.clone(),
        };
        emit_incremental_delta(
            write_tx,
            trace,
            full_path,
            assistant_message_id,
            &shell,
            idx,
            "generating",
        )
        .await;
    }

    // Now stream the text content in increments so amp's UI sees
    // progressive growth. Non-text blocks (tool_use, thinking) were
    // delivered whole above and don't need further deltas.
    'paced: for (idx, b) in blocks.iter().enumerate() {
        let Some(text) = b
            .get("text")
            .and_then(|v| v.as_str())
            .filter(|_| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        else {
            continue;
        };
        let chars: Vec<char> = text.chars().collect();
        let mut cursor = 0;
        while cursor < chars.len() {
            if cancel_flag.load(Ordering::Relaxed) {
                break 'paced;
            }
            let end = (cursor + CHUNK_CHARS).min(chars.len());
            let slice: String = chars[cursor..end].iter().collect();
            emit_incremental_delta(
                write_tx,
                trace,
                full_path,
                assistant_message_id,
                &json!({"type": "text", "text": slice}),
                idx,
                "generating",
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(PACE_MS)).await;
            cursor = end;
        }
    }

    let cancelled = cancel_flag.load(Ordering::Relaxed);
    emit_terminal_delta(
        write_tx,
        trace,
        full_path,
        assistant_message_id,
        if cancelled { "aborted" } else { "complete" },
    )
    .await;
}

/// JSON-RPC 2.0 stub. Returns a per-method result shape for the
/// handful of actor methods we've observed amp invoke. Falls back to
/// `result:{}` for unknown methods so the caller doesn't immediately
/// disconnect on a missing field.
fn ws_stub_reply(text: &str) -> Option<String> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    let id = val.get("id")?.clone();
    let method = val.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let result = match method {
        // amp's Executor handshake. Required fields verified in the binary's
        // response Zod schema: `executorId`, `registeredToolCount`,
        // `guidanceInventory` (array), `resumeBootstrap` (optional bool).
        "executor_connect" => json!({
            "executorId": "exec-aivo",
            "registeredToolCount": 0,
            "guidanceInventory": [],
            "resumeBootstrap": false,
        }),
        _ => json!({}),
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string())
}

async fn log_ws_event(path: Option<&Path>, full_path: &str, phase: &str, body: &str) {
    let Some(path) = path else { return };
    let entry = json!({
        "ts": http_utils::current_unix_ts(),
        "phase": format!("ws_{phase}"),
        "path": full_path,
        "body": body,
    });
    append_trace(path, &entry).await;
}

async fn log_request(path: Option<&Path>, method: &str, full_path: &str, body: &str) {
    let Some(path) = path else { return };
    let entry = json!({
        "ts": http_utils::current_unix_ts(),
        "phase": "request",
        "method": method,
        "path": full_path,
        "body": body,
    });
    append_trace(path, &entry).await;
}

async fn log_response_buffered(path: Option<&Path>, full_path: &str, status: u16, body: &str) {
    let Some(path) = path else { return };
    let entry = json!({
        "ts": http_utils::current_unix_ts(),
        "phase": "response",
        "path": full_path,
        "status": status,
        "body": body,
    });
    append_trace(path, &entry).await;
}

async fn append_trace(path: &Path, entry: &serde_json::Value) {
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    if let Ok(mut f) = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        let line = format!("{entry}\n");
        let _ = f.write_all(line.as_bytes()).await;
    }
}

async fn dispatch(
    state: &AmpBridgeState,
    request: &str,
    method: &str,
    full_path: &str,
    body: &str,
) -> Result<BridgeResponse> {
    let (path, query) = match full_path.split_once('?') {
        Some((p, q)) => (p, q),
        None => (full_path, ""),
    };

    if let Some(rest) = path.strip_prefix("/api/provider/") {
        return forward_to_upstream(state, rest, body, request).await;
    }

    // Management surface: if the user has a real Amp account configured,
    // forward to the real endpoint so amp's auth/threads/telemetry plane
    // works for real. Otherwise fall back to stubs.
    let is_management = path == "/api/internal"
        || path.starts_with("/api/user")
        || path.starts_with("/api/telemetry")
        || path.starts_with("/api/otel")
        || path.starts_with("/api/auth");
    if is_management
        && state.config.native_amp_url.is_some()
        && state.config.native_amp_key.is_some()
    {
        return forward_to_native_amp(state, full_path, body, request).await;
    }

    if path == "/api/internal" {
        let body_text = handle_internal_rpc(state, query, body).await;
        return Ok(stub_buffered(body_text, CONTENT_TYPE_JSON));
    }

    // amp's per-user actor fetches its Rivet WS creds here; the parser
    // throws + retry-loops (wedging the UI on "Loading thread") unless
    // all three fields are present and non-empty. Must precede the
    // `/api/user` catch-all, which this path starts_with.
    if path == "/api/user-actor-credentials" {
        return Ok(stub_buffered(
            r#"{"userId":"user_aivo_local","wsToken":"aivo-bridge","poolName":"default"}"#
                .to_string(),
            CONTENT_TYPE_JSON,
        ));
    }

    if path.starts_with("/api/user") {
        return Ok(stub_buffered(
            r#"{"userEmail":"aivo@local","isInternalUser":false,"features":[],"team":null,"mysteriousMessage":""}"#.to_string(),
            CONTENT_TYPE_JSON,
        ));
    }

    if path.starts_with("/api/telemetry")
        || path.starts_with("/api/otel")
        || path.starts_with("/api/auth")
    {
        return Ok(stub_buffered("{}".to_string(), CONTENT_TYPE_JSON));
    }

    // New in amp 0.0.1779927513 ("neo" rebrand, 2026-05-28). amp now
    // POSTs `/api/thread-actors` on every launch to provision a thread.
    // Without a 2xx response, amp throws "Failed to create thread-actors
    // thread" before the chat UI is usable. We return a synthetic thread
    // descriptor with `usesThreadActors:false` + `usesDtw:false` —
    // amp's downstream Rivet WebSocket transport (`thread-client`)
    // attempt will still fail (the bridge has no WS layer), but at
    // least the immediate HTTP error is gone and the UI proceeds. The
    // mark-imported follow-up (`POST /api/thread-actors/<id>`) is
    // acknowledged with an empty object.
    if let Some(rest) = path.strip_prefix("/api/thread-actors") {
        let body_text = handle_thread_actors(rest, body);
        return Ok(stub_buffered(body_text, CONTENT_TYPE_JSON));
    }

    // RivetKit HTTP API. aivo's `RIVET_PUBLIC_ENDPOINT` env points amp's
    // actor client at `<bridge>/actors`, so all rivet calls land here:
    //   GET    /actors/metadata?namespace=...   — service discovery
    //   GET    /actors?actor_ids=...             — get by id
    //   GET    /actors?name=...&key=...          — get by name+key
    //   PUT    /actors                           — create
    //   POST   /actors                           — get-or-create
    //   DELETE /actors/<id>                      — destroy
    // The Rivet client uses `J.any()` schemas so we just return shapes
    // that look plausible. The WS upgrade for the actor connection
    // itself is handled separately by `handle_websocket`.
    if path == "/actors" || path.starts_with("/actors/") || path.starts_with("/actors?") {
        return Ok(stub_buffered(
            handle_rivet_actors(method, path, body),
            CONTENT_TYPE_JSON,
        ));
    }

    // Amp polls `<AMP_URL>/news.rss` for announcement banners. Return an
    // empty but well-formed feed so the check is a silent no-op.
    if path == "/news.rss" {
        return Ok(stub_buffered(
            r#"<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>aivo</title><link>http://localhost</link><description></description></channel></rss>"#.to_string(),
            "application/rss+xml",
        ));
    }

    // amp's Librarian tool and the `amp threads <query>` CLI both search
    // the user's threads via `GET /api/threads/find?q=<query>&limit=N&
    // offset=M`, parsing `{threads:[...], hasMore:bool}`. On the real
    // ampcode.com backend this is server-side full-text search; the
    // bridge has to answer it from locally-persisted threads or the
    // Librarian hangs with no result and (pre-fix) couldn't be cancelled
    // — the stuck-Librarian half of aivo#14.
    if path == "/api/threads/find" {
        const DEFAULT_LIMIT: usize = 20;
        let mut q = String::new();
        let mut limit = DEFAULT_LIMIT;
        let mut offset = 0usize;
        for (k, v) in url::form_urlencoded::parse(query.as_bytes()) {
            match k.as_ref() {
                "q" => q = v.into_owned(),
                "limit" => limit = v.parse().unwrap_or(DEFAULT_LIMIT),
                "offset" => offset = v.parse().unwrap_or(0),
                _ => {}
            }
        }
        let (threads, has_more) =
            amp_threads::find_threads(&state.config.threads_dir, &q, limit, offset).await;
        let body = json!({"threads": threads, "hasMore": has_more}).to_string();
        return Ok(stub_buffered(body, CONTENT_TYPE_JSON));
    }

    eprintln!("[amp-bridge] UNHANDLED: {method} {full_path}");
    if state.config.trace_log_path.is_none() {
        eprintln!("[amp-bridge] re-run with --debug to capture the request body");
    }
    Ok(BridgeResponse::Buffered {
        status: 404,
        content_type: CONTENT_TYPE_JSON.to_string(),
        body: r#"{"error":{"code":"not-found","message":"unhandled by amp-bridge"}}"#.to_string(),
    })
}

fn stub_buffered(body: String, content_type: &str) -> BridgeResponse {
    BridgeResponse::Buffered {
        status: 200,
        content_type: content_type.to_string(),
        body,
    }
}

/// Forwards a management-plane request verbatim to the real Amp endpoint,
/// using the user's stored Sourcegraph token. The path (including query
/// string) is preserved so amp's RPC framework gets a real response.
async fn forward_to_native_amp(
    state: &AmpBridgeState,
    full_path: &str,
    body: &str,
    request: &str,
) -> Result<BridgeResponse> {
    let native_url = state
        .config
        .native_amp_url
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("native_amp_url unset"))?;
    let native_key = state
        .config
        .native_amp_key
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("native_amp_key unset"))?;

    let url = format!("{native_url}{full_path}");
    let mut headers = http_utils::extract_passthrough_headers(request)?;
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {native_key}"))?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(CONTENT_TYPE_JSON));

    let response = state
        .client
        .post(&url)
        .headers(headers)
        .body(body.to_string())
        .send()
        .await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    if content_type.contains("text/event-stream") {
        return Ok(BridgeResponse::Streaming {
            status,
            content_type,
            upstream: response,
            filter_reasoning: false,
        });
    }
    let resp_body = response.text().await?;
    Ok(BridgeResponse::Buffered {
        status,
        content_type,
        body: resp_body,
    })
}

/// Handles `/api/internal?<method>` requests, persisting and serving
/// thread state on disk for `getThread`/`uploadThread`/`listThreads`/
/// `deleteThread` so amp's resume flow works across `aivo amp`
/// invocations. Everything else delegates to the static stub map.
async fn handle_internal_rpc(state: &AmpBridgeState, query: &str, body: &str) -> String {
    let rpc_method = percent_codec::decode(query);
    let dir = state.config.threads_dir.as_path();
    match rpc_method.as_str() {
        "uploadThread" => {
            // Capture the FULL thread payload amp uploads on every turn
            // so getThread/listThreads can serve real data.
            if let Some(payload) = amp_threads::extract_thread_payload_from_request(body)
                && let Err(err) = amp_threads::save_thread(dir, &payload).await
            {
                eprintln!("[amp-bridge] uploadThread save failed: {err}");
            }
            r#"{"ok":true}"#.to_string()
        }
        "getThread" => {
            let Some(id) = amp_threads::extract_thread_id_from_request(body) else {
                return r#"{"ok":false,"error":{"code":"thread-not-found","message":"Thread not found"}}"#
                    .to_string();
            };
            match amp_threads::load_thread(dir, &id).await {
                Some(mut payload) => {
                    // Serve neo-shaped tool_result blocks so amp's loader
                    // doesn't crash on legacy Anthropic-shaped threads.
                    normalize_thread_payload_to_neo(&mut payload);
                    // Advertise actor-native so neo's resume skips the
                    // import-into-actor handshake that freezes the composer.
                    mark_thread_actor_native(&mut payload);
                    json!({
                        "ok": true,
                        "result": {"thread": {"data": payload}},
                    })
                    .to_string()
                }
                None => r#"{"ok":false,"error":{"code":"thread-not-found","message":"Thread not found"}}"#
                    .to_string(),
            }
        }
        "listThreads" => {
            let limit = amp_threads::extract_list_limit(body);
            let threads = amp_threads::list_threads(dir, limit).await;
            json!({"ok": true, "result": {"threads": threads}}).to_string()
        }
        // neo's "Switch Thread" picker calls this per highlighted row to
        // render the preview pane. amp's consumer is `{...result.thread
        // .data, messages: result.messages}`, so we hand back the full
        // thread object plus the message tail. Without this, the generic
        // stub's `result:null` shows "Preview Unavailable".
        "getThreadTail" => {
            let Some(id) = amp_threads::extract_thread_id_from_request(body) else {
                return r#"{"ok":false,"error":{"code":"thread-not-found","message":"Thread not found"}}"#
                    .to_string();
            };
            let limit = amp_threads::extract_tail_limit(body);
            match amp_threads::load_thread_tail(dir, &id, limit).await {
                Some((mut payload, tail)) => {
                    let tail = serde_json::Value::Array(
                        tail.iter()
                            .map(|m| {
                                let mut m = m.clone();
                                if let Some(c) = m.get("content") {
                                    m["content"] = map_content_blocks(c, block_to_neo);
                                }
                                m
                            })
                            .collect(),
                    );
                    // Keep the thread object's own `messages` consistent
                    // with the tail so the merged preview thread is
                    // self-contained and we don't ship full history twice.
                    payload["messages"] = tail.clone();
                    mark_thread_actor_native(&mut payload);
                    json!({
                        "ok": true,
                        "result": {"thread": {"data": payload}, "messages": tail},
                    })
                    .to_string()
                }
                None => r#"{"ok":false,"error":{"code":"thread-not-found","message":"Thread not found"}}"#
                    .to_string(),
            }
        }
        "deleteThread" => {
            if let Some(id) = amp_threads::extract_thread_id_from_request(body) {
                amp_threads::delete_thread(dir, &id).await;
            }
            r#"{"ok":true,"result":null}"#.to_string()
        }
        _ => internal_rpc_stub_body(query),
    }
}

/// RivetKit HTTP handler. Returns plausible shapes for the actor
/// service discovery + CRUD calls the amp client makes. Iteratively
/// stubbed — start by unblocking `/metadata` and `POST /actors`
/// (get-or-create); add specific replies as new endpoints show up in
/// the trace.
fn handle_rivet_actors(method: &str, full_path: &str, _body: &str) -> String {
    let path_no_query = full_path.split('?').next().unwrap_or(full_path);
    if path_no_query == "/actors/metadata" {
        // RivetKit client only logs `runtime/version/envoy` and reads
        // `clientEndpoint/clientNamespace/clientToken` for optional
        // overrides. All schemas are `J.any()` so we can be minimal.
        return r#"{"runtime":"aivo-bridge","version":"0","envoy":null}"#.to_string();
    }
    if path_no_query == "/actors" {
        match method {
            "GET" => return r#"{"actors":[]}"#.to_string(),
            "PUT" | "POST" => {
                // create / get-or-create. Return a synthetic actor.
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let actor_id = format!("act-aivo-{nanos}");
                return json!({
                    "actor": {
                        "id": actor_id,
                        "name": "threadActor",
                        "key": null,
                        "createdAt": 0,
                        "destroyedAt": null,
                    }
                })
                .to_string();
            }
            _ => {}
        }
    }
    if path_no_query.starts_with("/actors/") && method == "DELETE" {
        return r#"{}"#.to_string();
    }
    // Generic catch-all for unanticipated rivet paths. Empty 200 keeps
    // the client from looping while we observe new shapes in the trace.
    r#"{}"#.to_string()
}

fn handle_thread_actors(path_suffix: &str, body: &str) -> String {
    let trimmed = path_suffix.trim_start_matches('/');
    if trimmed.is_empty() {
        // amp posts here twice in execute mode: first to create a new
        // thread (no `threadId` in body), then again after the Rivet
        // WS connect_failed timeout — the retry includes the original
        // `threadId` and expects it echoed back. Generating a fresh id
        // on the retry triggers an infinite reconnect loop (seen in
        // amp-trace 20260528-115522).
        let req: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::Value::Null);
        let thread_id = req
            .get("threadId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(new_thread_id);
        let mut resp = json!({
            "threadId": thread_id,
            "wsToken": "aivo-bridge",
            "ownerUserId": "user_aivo_local",
            "threadVersion": 1,
            "usesThreadActors": false,
            "usesDtw": false,
            "executorType": serde_json::Value::Null,
        });
        if let Some(mode) = req.get("agentMode") {
            resp["agentMode"] = mode.clone();
        }
        resp.to_string()
    } else {
        r#"{}"#.to_string()
    }
}

fn internal_rpc_stub_body(query: &str) -> String {
    // Amp's RPC envelope (captured from real ampcode.com responses): caller
    // expects `{ok: true, result: ...}` or `{ok: false, error: {...}}`.
    // Stubs below mirror the real response shapes so amp considers itself
    // authenticated and proceeds, without any network traffic to
    // ampcode.com.
    let rpc_method = percent_codec::decode(query);
    match rpc_method.as_str() {
        "getUserInfo" => {
            // Schema mirrored from a real ampcode.com response. Auth check
            // on the amp client side requires a non-empty `email` and the
            // `accept-abuse-data-retention` feature flag.
            //
            // `isInternalUser: true` unlocks experimental agent modes and
            // honors `amp.internal.*` settings (notably `internal.model`,
            // which lets the user override the primary model — handy for
            // pointing amp at a Gemini-3 catalog entry to bypass the
            // ~300k context cap on Claude Opus while the bridge actually
            // serves requests via the configured upstream).
            r#"{"ok":true,"result":{"id":"user_aivo_local","username":null,"githubLogin":null,"slackUserID":null,"email":"aivo@local","firstName":"aivo","lastName":"local","emailVerified":true,"profilePictureUrl":null,"lastSignInAt":"2026-01-01T00:00:00.000Z","createdAt":"2026-01-01T00:00:00.000Z","updatedAt":"2026-01-01T00:00:00.000Z","siteAdmin":true,"isInternalUser":true,"features":[{"name":"accept-abuse-data-retention","enabled":true}],"mysteriousMessage":null}}"#.to_string()
        }
        "loadPlugins" => r#"{"ok":true,"result":[]}"#.to_string(),
        "getUserFreeTierStatus" => r#"{"ok":true,"result":{"canUseAmpFree":false}}"#.to_string(),
        // amp's resume flow calls `getThreadLinkInfo` for two checks:
        // (1) `result.creatorUserID` matched against the viewer to gate
        //     "cannot resume thread created by another user". We pin
        //     `creatorUserID` to the same `user_aivo_local` id we hand
        //     out in `getUserInfo` so ownership always matches.
        // (2) `result.usesThreadActors` — if true, amp refuses to resume
        //     the thread in the legacy CLI ("created with the Neo TUI").
        //     Aivo doesn't drive Neo, so always false.
        // Without this stub, the generic `result:null` arm below caused
        // amp's resume to throw "Unexpected error inside Amp CLI" while
        // dereferencing `null.creatorUserID`.
        "getThreadLinkInfo" => {
            r#"{"ok":true,"result":{"creatorUserID":"user_aivo_local","usesThreadActors":false}}"#
                .to_string()
        }
        // `getThread` / `getThreadTail` / `uploadThread` / `listThreads`
        // / `deleteThread` are intercepted by `handle_internal_rpc` for
        // disk persistence before reaching this stub. They never fall
        // through here.
        // amp's server-side LLM-reachable tools — normally executed by
        // ampcode.com, not by the LLM. The bridge has no implementation,
        // and the previous generic `result:null` stub caused amp's caller
        // code to dereference `result.results` / `result.fullContent` /
        // `result.tasks` and silently fail (red X in the UI), prompting the
        // model to retry. Returning an explicit error makes amp surface a
        // real tool_result error to the model on the first call, so it
        // falls back (Bash/curl for web; in-context TODO list for tasks)
        // immediately instead of looping.
        //
        // Web tools: `web_search` and `read_web_page`.
        // Task tools (single LLM-facing tool, sub-actions create/list/get/
        // update/delete): `createTask`, `listTasks`, `getTask`,
        // `updateTask`, `deleteTask`.
        "webSearch2" | "extractWebPageContent" => {
            r#"{"ok":false,"error":{"code":"not-supported","message":"web search/fetch tools are not implemented in aivo's amp bridge — use Bash with curl instead"}}"#.to_string()
        }
        "createTask" | "listTasks" | "getTask" | "updateTask" | "deleteTask" => {
            r#"{"ok":false,"error":{"code":"not-supported","message":"amp Task tool is not implemented in aivo's amp bridge — track work in-conversation instead"}}"#.to_string()
        }
        // amp's chat UI polls `notices` at startup and every 60s. The client
        // code is `this.notices = result; this.notices[0]` — so `result` MUST
        // be an array, not the catch-all `null`, or the UI crashes with
        // `null is not an object (evaluating 'this.notices[0]')`. Empty
        // array = no notices shown.
        "notices" => r#"{"ok":true,"result":[]}"#.to_string(),
        // Fires when the user dismisses a notice; null result is safe.
        "logNoticeAction" => r#"{"ok":true,"result":null}"#.to_string(),
        _ => {
            // Generic success for unknown methods. The trace log captures
            // the call shape so we can add a real stub later if needed.
            r#"{"ok":true,"result":null}"#.to_string()
        }
    }
}

async fn forward_to_upstream(
    state: &AmpBridgeState,
    rest: &str,
    body: &str,
    request: &str,
) -> Result<BridgeResponse> {
    // rest is e.g. "anthropic/v1/messages", "openai/v1/chat/completions",
    // or "google/v1beta/models/<model>:generateContent".
    let (provider, after) = match rest.split_once('/') {
        Some(parts) => parts,
        None => ("", rest),
    };

    // Single-pass body rewrite:
    // - force-model: amp picks Claude model names from its internal agent
    //   mode; non-Amp upstreams won't recognize them.
    // - strip forced anthropic tool_choice on /api/provider/anthropic: amp's
    //   title-generation call sends `{"type":"tool",...}` which reasoning
    //   models reject with "does not support this tool_choice".
    // Parsing and re-serializing the body is the heaviest per-request cost
    // on large /v1/messages payloads, so do both transforms in one pass.
    let body_owned = rewrite_request_body(
        body,
        state.config.force_model.as_deref(),
        provider == "anthropic",
    );

    // Anthropic-protocol requests route through the in-process translator
    // when the upstream isn't natively Anthropic.
    if provider == "anthropic"
        && let Some(port) = state.config.anthropic_translation_port
    {
        let url = format!("http://127.0.0.1:{port}/{after}");
        return forward_via_url(state, &url, &body_owned, request, false, false).await;
    }

    // OpenAI Responses API (`/v1/responses`) — amp's interactive chat
    // path. Translate to /v1/chat/completions via the responses router,
    // then filter reasoning content_part events on the way back so amp's
    // parser doesn't choke. Streamed when upstream sends SSE.
    if provider == "openai"
        && after.trim_start_matches('/').starts_with("v1/responses")
        && let Some(port) = state.config.responses_translation_port
    {
        let url = format!("http://127.0.0.1:{port}/{after}");
        return forward_via_url(state, &url, &body_owned, request, false, true).await;
    }

    // Direct passthrough — strip the `<provider>/` prefix so the upstream
    // sees a normal request path.
    let url = format!("{}/{after}", state.config.upstream_base_url);
    forward_via_url(state, &url, &body_owned, request, true, false).await
}

/// Strips reasoning-related Responses-API SSE events that amp's parser
/// doesn't recognize. The upstream may emit `response.content_part.added/
/// done` events whose `part.type == "reasoning"` (deepseek-reasoner,
/// gpt-5.5, deepseek-v4-pro at high effort, …). Amp throws "unexpected
/// content_part.added for output message: reasoning" on those. Also
/// strips reasoning entries from `content` arrays in `output_item.done`
/// / `response.completed` snapshots.
fn filter_reasoning_sse(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    for chunk in body.split("\n\n") {
        if chunk.is_empty() {
            continue;
        }
        // Drop content_part events whose `part.type == "reasoning"`. JSON
        // key order isn't guaranteed (the upstream may emit
        // `{"reasoning":"","type":"reasoning"}` vs `{"type":"reasoning",...}`),
        // so a substring match on a fixed key order misses cases — parse
        // the data line as JSON and check the part type directly.
        if event_is_reasoning_content_part(chunk) {
            continue;
        }
        // For events that carry a full message snapshot (output_item.done,
        // response.completed), strip reasoning entries from the content
        // array so amp's final-message parser doesn't reject the snapshot.
        let cleaned = strip_reasoning_from_event_data(chunk);
        out.push_str(&cleaned);
        out.push_str("\n\n");
    }
    out
}

fn event_is_reasoning_content_part(chunk: &str) -> bool {
    let Some(json_text) = extract_sse_data(chunk) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json_text) else {
        return false;
    };
    let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if event_type != "response.content_part.added" && event_type != "response.content_part.done" {
        return false;
    }
    value
        .get("part")
        .and_then(|p| p.get("type"))
        .and_then(|t| t.as_str())
        == Some("reasoning")
}

fn extract_sse_data(chunk: &str) -> Option<&str> {
    if let Some(stripped) = chunk.strip_prefix("data: ") {
        return Some(stripped);
    }
    let idx = chunk.find("\ndata: ")?;
    Some(&chunk[idx + "\ndata: ".len()..])
}

fn strip_reasoning_from_event_data(chunk: &str) -> String {
    // SSE event format is `event: <name>\ndata: <json>`. Find the data:
    // line, parse the JSON, surgically remove reasoning content entries,
    // re-emit. Tolerant: if anything goes sideways we return the chunk
    // unmodified rather than corrupting the stream.
    let Some(data_start) = chunk.find("\ndata: ").or_else(|| {
        if chunk.starts_with("data: ") {
            Some(0)
        } else {
            None
        }
    }) else {
        return chunk.to_string();
    };
    let prefix_len = if chunk.starts_with("data: ") {
        "data: ".len()
    } else {
        data_start + "\ndata: ".len()
    };
    let json_text = &chunk[prefix_len..];
    let mut value: serde_json::Value = match serde_json::from_str(json_text) {
        Ok(v) => v,
        Err(_) => return chunk.to_string(),
    };

    let mut changed = false;
    walk_strip_reasoning(&mut value, &mut changed);

    if !changed {
        return chunk.to_string();
    }
    let new_json = value.to_string();
    let mut out = String::with_capacity(chunk.len());
    out.push_str(&chunk[..prefix_len]);
    out.push_str(&new_json);
    out
}

fn walk_strip_reasoning(value: &mut serde_json::Value, changed: &mut bool) {
    match value {
        serde_json::Value::Array(items) => {
            let original_len = items.len();
            items.retain(|v| v.get("type").and_then(|t| t.as_str()) != Some("reasoning"));
            if items.len() != original_len {
                *changed = true;
            }
            for item in items {
                walk_strip_reasoning(item, changed);
            }
        }
        serde_json::Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                walk_strip_reasoning(v, changed);
            }
        }
        _ => {}
    }
}

/// In-place body edits applied in a single parse/serialize pass:
/// - `force_model`: replaces the top-level `model` field when present.
/// - `strip_anthropic_forced_tool_choice`: removes a forced Anthropic-style
///   `{"type":"tool","name":"..."}` tool_choice. `null` and `"auto"` are
///   untouched. Some upstream reasoning models (notably `deepseek-reasoner`)
///   reject any non-`auto` tool_choice with "does not support this
///   tool_choice"; the model still has the tool in `tools[]` and the system
///   prompt usually instructs the behavior, so dropping is safe enough for
///   amp's title-generation case.
/// - **always** rewrites descriptions for `web_search` and `read_web_page`
///   in `tools[]`. The bridge can't serve these — they'd hit a
///   `not-supported` stub mid-conversation. Replacing the schema text
///   with a curl-pointer (~20 tokens vs the original ~100) lets the model
///   see the tool exists but route directly to Bash without a wasted
///   round-trip. amp's system prompt frames web access as a tool-only
///   capability, so stripping the tools entirely caused the model to
///   apologize and give up (2026-05-08 regression). Native amp launches
///   never hit this function (different code path), so ampcode.com's
///   real implementation is unaffected.
///
/// Returns the body verbatim when no edit applies or the body isn't JSON.
fn rewrite_request_body(
    body: &str,
    force_model: Option<&str>,
    strip_anthropic_forced_tool_choice: bool,
) -> String {
    // Cheap substring guard: parsing and re-serializing is the heaviest
    // per-request cost. Skip the round-trip when no edit could possibly
    // apply. False positives (e.g. literal "web_search" inside an unrelated
    // string) just take the slow path and the rewrite no-ops, which is
    // strictly correct.
    let body_might_have_unsupported_tools =
        body.contains("\"web_search\"") || body.contains("\"read_web_page\"");
    if force_model.is_none()
        && !strip_anthropic_forced_tool_choice
        && !body_might_have_unsupported_tools
    {
        return body.to_string();
    }
    let mut value: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return body.to_string(),
    };
    let Some(obj) = value.as_object_mut() else {
        return body.to_string();
    };

    let mut changed = false;
    if let Some(forced) = force_model
        && obj.contains_key("model")
    {
        obj.insert(
            "model".to_string(),
            serde_json::Value::String(forced.to_string()),
        );
        changed = true;
    }
    if strip_anthropic_forced_tool_choice {
        let is_forced = obj
            .get("tool_choice")
            .and_then(|tc| tc.as_object())
            .and_then(|tc| tc.get("type"))
            .and_then(|t| t.as_str())
            == Some("tool");
        if is_forced {
            obj.remove("tool_choice");
            changed = true;
        }
    }
    if rewrite_unsupported_tool_descriptions(obj) {
        changed = true;
    }
    if !changed {
        return body.to_string();
    }
    value.to_string()
}

/// Replaces the `description` field of any `tools[]` entry whose `name` is
/// a bridge-unsupported web tool with a short pointer to Bash + curl.
///
/// Covers two body shapes:
///   - Anthropic Messages: `tools[].name` + `tools[].description`
///   - OpenAI Chat/Responses: `tools[].function.name` + `tools[].function.description`
///
/// Returns `true` if any tool was rewritten.
fn rewrite_unsupported_tool_descriptions(
    obj: &mut serde_json::Map<String, serde_json::Value>,
) -> bool {
    let Some(tools) = obj.get_mut("tools").and_then(|v| v.as_array_mut()) else {
        return false;
    };
    let mut changed = false;
    for entry in tools {
        let Some(item) = entry.as_object_mut() else {
            continue;
        };
        // Anthropic shape: `{name, description, input_schema}`
        if let Some(replacement) = item
            .get("name")
            .and_then(|n| n.as_str())
            .and_then(unsupported_tool_replacement)
        {
            item.insert(
                "description".to_string(),
                serde_json::Value::String(replacement.to_string()),
            );
            changed = true;
            continue;
        }
        // OpenAI shape: `{type:"function", function:{name, description, parameters}}`
        if let Some(func) = item.get_mut("function").and_then(|f| f.as_object_mut())
            && let Some(replacement) = func
                .get("name")
                .and_then(|n| n.as_str())
                .and_then(unsupported_tool_replacement)
        {
            func.insert(
                "description".to_string(),
                serde_json::Value::String(replacement.to_string()),
            );
            changed = true;
        }
    }
    changed
}

/// Returns the replacement description for a tool name we know the bridge
/// can't serve, or `None` for any other tool. Wording is identical for
/// both web tools because the actionable workaround is the same and
/// identical strings cache better in the upstream's prompt cache.
///
/// The recommended fetch command is platform-gated at compile time:
/// - Unix (Linux/macOS): `curl` or `wget`, both standard
/// - Windows: `curl.exe` (ships in System32 since Win10 1803) is preferred
///   because PowerShell's bare `curl` is an alias for `Invoke-WebRequest`
///   with incompatible flags. Fall back to `Invoke-WebRequest`/`iwr` when
///   `curl.exe` is unavailable (older Windows / stripped images).
#[cfg(not(windows))]
const WEB_TOOL_REPLACEMENT: &str = "DISABLED in this environment — calling will return an error. \
     To search the web or fetch a URL's contents, use the Bash tool with `curl` or `wget` instead.";
#[cfg(windows)]
const WEB_TOOL_REPLACEMENT: &str = "DISABLED in this environment — calling will return an error. \
     To search the web or fetch a URL's contents, use the Bash tool with `curl.exe` (or PowerShell's `Invoke-WebRequest` / `iwr` if `curl.exe` is unavailable). Note: bare `curl` in PowerShell is an alias for `Invoke-WebRequest` and rejects standard curl flags — always call `curl.exe` explicitly.";
fn unsupported_tool_replacement(name: &str) -> Option<&'static str> {
    match name {
        "web_search" | "read_web_page" => Some(WEB_TOOL_REPLACEMENT),
        _ => None,
    }
}

/// Forwards a request to `url` and returns either a buffered or streaming
/// response based on the upstream's content-type. SSE responses are
/// streamed chunk-by-chunk back to amp so its TUI sees tokens as they
/// arrive — buffering would make the chat feel frozen until the whole
/// answer landed.
///
/// - `inject_auth=true` rewrites the Authorization header with the upstream
///   API key (direct upstream calls). `false` for in-process translator
///   proxies which inject auth themselves.
/// - `filter_reasoning=true` strips `response.content_part.added/done`
///   events whose `part.type == "reasoning"` from the SSE stream and
///   from `output_item.done` / `response.completed` snapshots.
async fn forward_via_url(
    state: &AmpBridgeState,
    url: &str,
    body: &str,
    request: &str,
    inject_auth: bool,
    filter_reasoning: bool,
) -> Result<BridgeResponse> {
    let mut headers = http_utils::extract_passthrough_headers(request)?;
    if inject_auth {
        let auth_value = format!("Bearer {}", state.config.upstream_api_key);
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
    }
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(CONTENT_TYPE_JSON));

    let response = state
        .client
        .post(url)
        .headers(headers)
        .body(body.to_string())
        .send()
        .await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    // SSE → stream; everything else → buffer. The reasoning filter still
    // applies in both modes (incremental in streaming, post-hoc in
    // buffered) so amp's parser doesn't see reasoning content_part events
    // either way.
    if content_type.contains("text/event-stream") {
        Ok(BridgeResponse::Streaming {
            status,
            content_type,
            upstream: response,
            filter_reasoning,
        })
    } else {
        let resp_body = response.text().await?;
        let body = if filter_reasoning {
            filter_reasoning_sse(&resp_body)
        } else {
            resp_body
        };
        Ok(BridgeResponse::Buffered {
            status,
            content_type,
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_endpoint_detection() {
        assert!(is_amp_native_endpoint("https://ampcode.com/"));
        assert!(is_amp_native_endpoint("https://amp.ampcode.com"));
        assert!(is_amp_native_endpoint("https://ampcode.com"));
        assert!(is_amp_native_endpoint("https://sourcegraph.com/.api/amp/"));
        assert!(is_amp_native_endpoint("http://localhost:8317/"));
        assert!(is_amp_native_endpoint("http://127.0.0.1:8317/"));
        assert!(!is_amp_native_endpoint("https://api.deepseek.com"));
        assert!(!is_amp_native_endpoint("https://openrouter.ai/api/v1"));
        // Path-based spoofing must not pass — host has to be the actual
        // ampcode.com / sourcegraph.com domain (or a subdomain), not a
        // string occurring in the path.
        assert!(!is_amp_native_endpoint(
            "https://attacker.example/ampcode.com"
        ));
        assert!(!is_amp_native_endpoint(
            "https://attacker.example/sourcegraph.com"
        ));
        assert!(!is_amp_native_endpoint(
            "https://ampcode.com.attacker.example"
        ));
        // Garbage input
        assert!(!is_amp_native_endpoint("not a url"));
        assert!(!is_amp_native_endpoint(""));
    }

    #[test]
    fn internal_rpc_stub_known_method_returns_envelope() {
        let body = internal_rpc_stub_body("getUserInfo");
        assert!(body.contains(r#""ok":true"#));
        // Real ampcode.com schema uses `email` (not `userEmail`); amp's auth
        // check requires a non-empty value here to flip isAuthenticated=true.
        assert!(body.contains(r#""email":"aivo@local""#));
        // Required for amp's "data retention accepted" gate.
        assert!(body.contains("accept-abuse-data-retention"));
    }

    #[test]
    fn internal_rpc_stub_unimplemented_llm_tools_return_explicit_error() {
        // The previous generic `{ok:true, result:null}` stub made amp
        // dereference `result.results` / `result.fullContent` /
        // `result.tasks` and silently fail, prompting the model to retry.
        // An explicit error envelope surfaces a real tool_result error so
        // the model falls back on the first call.
        for method in [
            "webSearch2",
            "extractWebPageContent",
            "createTask",
            "listTasks",
            "getTask",
            "updateTask",
            "deleteTask",
        ] {
            let body = internal_rpc_stub_body(method);
            assert!(body.contains(r#""ok":false"#), "{method}");
            assert!(body.contains(r#""code":"not-supported""#), "{method}");
        }
    }

    #[test]
    fn internal_rpc_stub_notices_returns_empty_array_not_null() {
        // amp's chat UI does `this.notices = result; this.notices[0]` on
        // every poll. A null result crashes the element rebuild.
        let body = internal_rpc_stub_body("notices");
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert_eq!(parsed["ok"], serde_json::Value::Bool(true));
        assert!(parsed["result"].is_array(), "result must be an array");
        assert_eq!(parsed["result"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn handle_thread_actors_create_returns_required_shape() {
        let body = handle_thread_actors("", r#"{"agentMode":"smart"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        // amp's `Ix` Zod template literal requires `T-<8hex>-<4hex>-...`.
        let tid = parsed["threadId"].as_str().unwrap();
        assert!(tid.starts_with("T-"), "{tid}");
        assert_eq!(tid.len(), "T-00000000-0000-0000-0000-000000000000".len());
        assert!(tid[2..].chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        assert_eq!(parsed["wsToken"], "aivo-bridge");
        assert_eq!(parsed["ownerUserId"], "user_aivo_local");
        assert!(parsed["threadVersion"].is_number());
        assert_eq!(parsed["usesThreadActors"], serde_json::Value::Bool(false));
        assert_eq!(parsed["agentMode"], "smart");
    }

    #[test]
    fn handle_thread_actors_echoes_supplied_thread_id() {
        // On the post-WS-failure retry, amp resubmits the original
        // threadId. Returning a fresh id would loop the reconnect.
        let body = handle_thread_actors("", r#"{"threadId":"T-aivo-existing","agentMode":"deep"}"#);
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert_eq!(parsed["threadId"], "T-aivo-existing");
    }

    #[test]
    fn handle_thread_actors_mark_imported_returns_empty_object() {
        let body = handle_thread_actors("/T-abc", "{}");
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert!(parsed.is_object() && parsed.as_object().unwrap().is_empty());
    }

    #[test]
    fn internal_rpc_stub_log_notice_action_returns_null_ok() {
        let body = internal_rpc_stub_body("logNoticeAction");
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert_eq!(parsed["ok"], serde_json::Value::Bool(true));
        assert!(parsed["result"].is_null());
    }

    #[test]
    fn rewrite_request_body_drops_anthropic_forced_selection() {
        let body = r#"{"model":"x","tool_choice":{"type":"tool","name":"set_title","disable_parallel_tool_use":true},"tools":[]}"#;
        let out = rewrite_request_body(body, None, true);
        assert!(!out.contains("tool_choice"));
        assert!(out.contains(r#""model":"x""#));
        assert!(out.contains(r#""tools":[]"#));
    }

    #[test]
    fn rewrite_request_body_passes_through_auto_or_null_tool_choice() {
        // null tool_choice — used by amp's normal chat call. Leave it.
        let body = r#"{"model":"x","tool_choice":null}"#;
        assert_eq!(rewrite_request_body(body, None, true), body);
        // auto tool_choice — leave it.
        let auto = r#"{"model":"x","tool_choice":"auto"}"#;
        assert_eq!(rewrite_request_body(auto, None, true), auto);
    }

    #[test]
    fn rewrite_request_body_replaces_top_level_model() {
        let body = r#"{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"hi"}]}"#;
        let out = rewrite_request_body(body, Some("deepseek-v4-pro"), false);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["model"], "deepseek-v4-pro");
        assert!(parsed["messages"].is_array());
    }

    #[test]
    fn rewrite_request_body_passes_through_invalid_json() {
        let out = rewrite_request_body("not json", Some("x"), true);
        assert_eq!(out, "not json");
    }

    #[test]
    fn rewrite_request_body_applies_both_edits_in_one_pass() {
        let body = r#"{"model":"claude-haiku-4-5","tool_choice":{"type":"tool","name":"x"}}"#;
        let out = rewrite_request_body(body, Some("deepseek-v4-pro"), true);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["model"], "deepseek-v4-pro");
        assert!(parsed.get("tool_choice").is_none());
    }

    #[test]
    fn rewrite_request_body_short_circuits_when_no_edits_requested() {
        let body = r#"{"model":"x","tool_choice":{"type":"tool","name":"y"}}"#;
        assert_eq!(rewrite_request_body(body, None, false), body);
    }

    #[test]
    fn filter_reasoning_drops_content_part_added_events() {
        // Real upstream JSON emits keys in alphabetical order so the part
        // ends up as `{"reasoning":"","type":"reasoning"}` — make sure the
        // filter handles both orderings via JSON parse, not substring match.
        let body = "event: response.content_part.added\n\
                    data: {\"content_index\":1,\"part\":{\"reasoning\":\"\",\"type\":\"reasoning\"},\"type\":\"response.content_part.added\"}\n\n\
                    event: response.output_text.delta\n\
                    data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
        let out = filter_reasoning_sse(body);
        assert!(!out.contains("content_part.added"));
        assert!(out.contains("output_text.delta"));
        assert!(out.contains(r#""delta":"hi""#));
    }

    #[test]
    fn filter_reasoning_strips_reasoning_from_content_array_in_snapshot() {
        // response.completed and output_item.done events carry the full
        // assistant message in a `content` array. Reasoning entries there
        // also need to go away.
        let body = "event: response.completed\n\
                    data: {\"type\":\"response.completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"4\"},{\"type\":\"reasoning\",\"reasoning\":\"think\"}]}]}\n\n";
        let out = filter_reasoning_sse(body);
        assert!(out.contains(r#""text":"4""#));
        assert!(!out.contains(r#""type":"reasoning""#));
    }

    #[test]
    fn filter_reasoning_passes_through_when_no_reasoning() {
        let body = "event: response.output_text.delta\n\
                    data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
        let out = filter_reasoning_sse(body);
        assert!(out.contains(r#""delta":"hi""#));
    }

    #[test]
    fn rewrite_request_body_passes_through_when_model_absent() {
        let body = r#"{"messages":[]}"#;
        let out = rewrite_request_body(body, Some("x"), false);
        assert_eq!(out, body);
    }

    #[test]
    fn rewrite_request_body_replaces_anthropic_web_tool_descriptions() {
        // Anthropic Messages shape: tools[i] is `{name, description, input_schema}`.
        // The bridge rewrites web_search / read_web_page descriptions to point
        // at Bash + curl since the bridge can't serve them. Other tools (Bash,
        // create_file) are left verbatim.
        let body = r#"{"model":"claude-haiku-4-5","tools":[
            {"name":"web_search","description":"Search the web for current info.","input_schema":{"type":"object"}},
            {"name":"Bash","description":"Run a shell command.","input_schema":{"type":"object"}},
            {"name":"read_web_page","description":"Fetch a URL and return its contents.","input_schema":{"type":"object"}}
        ]}"#;
        let out = rewrite_request_body(body, None, false);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let tools = parsed["tools"].as_array().unwrap();
        assert!(
            tools[0]["description"]
                .as_str()
                .unwrap()
                .contains("DISABLED"),
            "web_search description should be replaced"
        );
        assert!(
            tools[0]["description"].as_str().unwrap().contains("curl"),
            "should mention curl as the workaround"
        );
        assert_eq!(
            tools[1]["description"], "Run a shell command.",
            "Bash description must not be touched"
        );
        assert!(
            tools[2]["description"]
                .as_str()
                .unwrap()
                .contains("DISABLED"),
            "read_web_page description should be replaced"
        );
    }

    #[test]
    #[cfg(windows)]
    fn rewrite_request_body_windows_description_points_at_curl_exe() {
        // Windows-only: bare `curl` in PowerShell is an alias for
        // Invoke-WebRequest and rejects standard curl flags. The model
        // must be steered toward `curl.exe` (ships in System32 since
        // Win10 1803) or PowerShell's Invoke-WebRequest, never bare
        // `curl` in PowerShell.
        let body = r#"{"tools":[{"name":"web_search","description":"x","input_schema":{}}]}"#;
        let out = rewrite_request_body(body, None, false);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let desc = parsed["tools"][0]["description"].as_str().unwrap();
        assert!(desc.contains("curl.exe"));
        assert!(desc.contains("Invoke-WebRequest"));
    }

    #[test]
    #[cfg(not(windows))]
    fn rewrite_request_body_unix_description_keeps_plain_curl() {
        // Unix-only: the description should NOT mention curl.exe or
        // Invoke-WebRequest — those are dead weight on macOS/Linux where
        // bare `curl` and `wget` are universally available.
        let body = r#"{"tools":[{"name":"web_search","description":"x","input_schema":{}}]}"#;
        let out = rewrite_request_body(body, None, false);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let desc = parsed["tools"][0]["description"].as_str().unwrap();
        assert!(desc.contains("curl"));
        assert!(desc.contains("wget"));
        assert!(!desc.contains("curl.exe"));
        assert!(!desc.contains("Invoke-WebRequest"));
    }

    #[test]
    fn rewrite_request_body_replaces_openai_web_tool_descriptions() {
        // OpenAI Chat/Responses shape: tools[i] is `{type:"function", function:{name, description, parameters}}`.
        let body = r#"{"model":"gpt-5","tools":[
            {"type":"function","function":{"name":"web_search","description":"Search.","parameters":{}}},
            {"type":"function","function":{"name":"create_file","description":"Make a file.","parameters":{}}}
        ]}"#;
        let out = rewrite_request_body(body, None, false);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let tools = parsed["tools"].as_array().unwrap();
        assert!(
            tools[0]["function"]["description"]
                .as_str()
                .unwrap()
                .contains("DISABLED")
        );
        assert_eq!(tools[1]["function"]["description"], "Make a file.");
    }

    #[test]
    fn rewrite_request_body_skips_parse_when_no_unsupported_tools_or_edits() {
        // Fast-path: when none of the trigger conditions apply (no
        // force_model, no tool_choice strip, no web_search/read_web_page
        // substring), the function must short-circuit and return the body
        // byte-for-byte. Verifies the substring guard works.
        let body =
            r#"{"model":"claude-haiku-4-5","tools":[{"name":"Bash","description":"shell"}]}"#;
        let out = rewrite_request_body(body, None, false);
        assert_eq!(out, body);
    }

    #[test]
    fn internal_rpc_stub_unknown_method_wraps_in_ok_envelope() {
        // Amp's RPC client checks `response.ok` — a bare null or unwrapped
        // object crashes with `e.ok is not an object`. The stub must wrap.
        let body = internal_rpc_stub_body("someUnknownThing");
        assert_eq!(body, r#"{"ok":true,"result":null}"#);
    }

    #[test]
    fn incremental_reasoning_filter_drops_event_across_chunks() {
        // SSE event arrives split into two reqwest chunks. Filter buffers
        // until the `\n\n` event boundary, then drops the reasoning event.
        let mut filter = IncrementalReasoningFilter::new();
        let part1 = b"event: response.content_part.added\n\
                      data: {\"part\":{\"reasoning\":\"\",\"type\":\"reason";
        let part2 = b"ing\"},\"type\":\"response.content_part.added\"}\n\n\
                      event: response.output_text.delta\n\
                      data: {\"delta\":\"hi\",\"type\":\"response.output_text.delta\"}\n\n";
        let out1 = filter.feed(part1);
        // First chunk has no complete event yet → emit nothing.
        assert!(out1.is_empty());
        let out2 = filter.feed(part2);
        let s = String::from_utf8(out2).unwrap();
        // Reasoning event dropped, output_text.delta passed through.
        assert!(!s.contains("content_part.added"));
        assert!(s.contains(r#""delta":"hi""#));
    }

    /// Buffered branch sends an initial shell delta to establish the
    /// block at amp's index, then text increments (so amp's IZR
    /// `_.text + r.text` rebuilds the full content), then a terminal
    /// delta with empty blocks. Concatenating the text increments
    /// must yield the original input.
    #[tokio::test]
    async fn buffered_branch_emits_text_increments_then_terminal() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let text = "x".repeat(96);
        let blocks = vec![json!({"type": "text", "text": text})];

        let cancel = Arc::new(AtomicBool::new(false));
        emit_buffered_blocks_progressively(&tx, None, "/test", "M-test", &blocks, &cancel).await;
        drop(tx);

        let mut deltas: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            deltas.push(serde_json::from_str(&s).unwrap());
        }

        // First emit: shell (empty text), blockIndex 0.
        let first = &deltas[0]["params"];
        assert_eq!(first["state"].as_str(), Some("generating"));
        assert_eq!(first["blockIndex"].as_u64(), Some(0));
        assert_eq!(first["blocks"][0]["type"].as_str(), Some("text"));
        assert_eq!(first["blocks"][0]["text"].as_str(), Some(""));

        // Last emit: terminal complete with empty blocks.
        let last = deltas.last().unwrap();
        assert_eq!(last["params"]["state"].as_str(), Some("complete"));
        assert_eq!(last["params"]["blocks"].as_array().unwrap().len(), 0);

        // Concatenated text increments rebuild the full content.
        let rebuilt: String = deltas[1..deltas.len() - 1]
            .iter()
            .filter_map(|d| d["params"]["blocks"][0]["text"].as_str())
            .collect();
        assert_eq!(rebuilt.len(), 96);
        assert_eq!(rebuilt, "x".repeat(96));
    }

    /// tool_use-only response: the shell delta carries the whole tool
    /// block (id/name/input/etc.) and the terminal delta closes the
    /// turn. No text increments because there's no text content.
    #[tokio::test]
    async fn buffered_branch_with_only_tool_use_emits_shell_then_terminal() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let blocks = vec![json!({
            "type": "tool_use",
            "id": "TU-abc",
            "name": "Bash",
            "input": {"cmd": "ls"}
        })];

        let cancel = Arc::new(AtomicBool::new(false));
        emit_buffered_blocks_progressively(&tx, None, "/test", "M-test", &blocks, &cancel).await;
        drop(tx);

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        assert_eq!(frames.len(), 2, "shell + terminal only");
        assert_eq!(frames[0]["params"]["state"].as_str(), Some("generating"));
        assert_eq!(
            frames[0]["params"]["blocks"][0]["type"].as_str(),
            Some("tool_use")
        );
        assert_eq!(frames[1]["params"]["state"].as_str(), Some("complete"));
    }

    #[test]
    fn tu_id_validator_accepts_proper_shape_only() {
        // 25 chars total: "TU-" + 22 base62 chars.
        assert!(is_valid_tu_id("TU-aB3cdEfghiJklmNopqrsTu"));
        assert!(is_valid_tu_id("TU-0123456789ABCDEFGHIJKL"));
        // Too short, too long, wrong prefix, illegal char.
        assert!(!is_valid_tu_id("TU-abc"));
        assert!(!is_valid_tu_id("TU-abcdefghijklmnopqrstuvw")); // 23 chars after TU-
        assert!(!is_valid_tu_id("XX-aB3cdEfghiJklmNopqrsTu"));
        assert!(!is_valid_tu_id("TU-aB3cdEfghiJklmNopqr-Tu"));
        assert!(!is_valid_tu_id(""));
    }

    #[test]
    fn parse_buffered_message_blocks_salvages_mislabeled_json_body() {
        let body = json!({
            "id": "gen_x",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "call_1", "name": "skill", "input": {"name": "x"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 2}
        })
        .to_string();
        let blocks = parse_buffered_message_blocks(&body).expect("salvages message JSON");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["text"].as_str().unwrap(), "Hello");
        assert!(is_valid_tu_id(blocks[1]["id"].as_str().unwrap()));
    }

    #[test]
    fn parse_buffered_message_blocks_rejects_non_message_bodies() {
        assert!(parse_buffered_message_blocks("event: message_stop\n").is_none());
        assert!(
            parse_buffered_message_blocks(r#"{"type":"error","error":{"message":"boom"}}"#)
                .is_none()
        );
        assert!(parse_buffered_message_blocks(r#"{"type":"message","content":[]}"#).is_none());
    }

    #[test]
    fn ensure_tool_use_id_rewrites_llm_native_ids() {
        // OpenAI-shaped call id — must be replaced.
        let mut block = json!({
            "type": "tool_use",
            "id": "call_AbcDef12345",
            "name": "Bash",
            "input": {"cmd": "ls"}
        });
        let id = ensure_tool_use_id(&mut block).expect("returns id for tool_use");
        assert!(is_valid_tu_id(&id), "got id {id:?}");
        assert_eq!(block["id"].as_str().unwrap(), id);

        // Anthropic-shaped toolu id — also replaced.
        let mut block = json!({
            "type": "tool_use",
            "id": "toolu_01abc",
            "name": "Bash",
            "input": {}
        });
        let id = ensure_tool_use_id(&mut block).expect("returns id for tool_use");
        assert!(is_valid_tu_id(&id), "got id {id:?}");
    }

    #[test]
    fn ensure_tool_use_id_is_idempotent_on_valid_tu_id() {
        let original = "TU-aB3cdEfghiJklmNopqrsTu".to_string();
        let mut block = json!({
            "type": "tool_use",
            "id": original.clone(),
            "name": "Bash",
            "input": {}
        });
        let id = ensure_tool_use_id(&mut block).unwrap();
        assert_eq!(id, original, "valid TU id should be preserved verbatim");
        assert_eq!(block["id"].as_str().unwrap(), original);
    }

    #[test]
    fn ensure_tool_use_id_ignores_non_tool_use_blocks() {
        let mut text = json!({"type": "text", "text": "hello"});
        assert!(ensure_tool_use_id(&mut text).is_none());
        assert_eq!(text["text"].as_str().unwrap(), "hello");

        let mut thinking = json!({
            "type": "thinking",
            "thinking": "...",
            "signature": ""
        });
        assert!(ensure_tool_use_id(&mut thinking).is_none());
    }

    #[test]
    fn ensure_tool_use_id_assigns_when_id_missing() {
        let mut block = json!({"type": "tool_use", "name": "Bash"});
        let id = ensure_tool_use_id(&mut block).expect("returns id");
        assert!(is_valid_tu_id(&id));
        assert_eq!(block["id"].as_str().unwrap(), id);
    }

    fn fake_bridge_state() -> AmpBridgeState {
        AmpBridgeState {
            config: Arc::new(AmpBridgeConfig {
                upstream_base_url: "http://127.0.0.1:1".to_string(),
                upstream_api_key: "test".to_string(),
                trace_log_path: None,
                native_amp_url: None,
                native_amp_key: None,
                anthropic_translation_port: None,
                responses_translation_port: None,
                force_model: None,
                threads_dir: std::env::temp_dir().join("aivo-amp-bridge-test"),
                usage_store: None,
                usage_key_id: String::new(),
            }),
            client: reqwest::Client::new(),
        }
    }

    fn fake_ws_state() -> WsState {
        WsState::from_path("/actors?rvt-key=T-0001-0002-0003-0004-000000000005")
    }

    fn bridge_state_with_threads_dir(dir: std::path::PathBuf) -> AmpBridgeState {
        let mut state = fake_bridge_state();
        let mut config = (*state.config).clone();
        config.threads_dir = dir;
        state.config = Arc::new(config);
        state
    }

    /// The two tool_result converters are shape-aware and idempotent, and
    /// leave non-tool_result blocks untouched — the property that lets the
    /// same code path handle both fresh (neo) and legacy (Anthropic) data.
    #[test]
    fn tool_result_converters_are_idempotent_and_passthrough() {
        let anthropic = json!({"type": "tool_result", "tool_use_id": "TU-x",
                               "content": "out", "is_error": true});
        let neo = json!({"type": "tool_result", "toolUseID": "TU-x",
                        "run": {"status": "error", "result": "out"}});

        // Cross-convert lands on the other shape…
        assert_eq!(block_to_neo(&anthropic)["toolUseID"], json!("TU-x"));
        assert_eq!(block_to_neo(&anthropic)["run"]["status"], json!("error"));
        assert_eq!(block_to_anthropic(&neo)["tool_use_id"], json!("TU-x"));
        assert_eq!(block_to_anthropic(&neo)["is_error"], json!(true));

        // …and re-applying the same direction is a no-op (already in shape).
        assert_eq!(block_to_neo(&neo), neo);
        assert_eq!(block_to_anthropic(&anthropic), anthropic);

        // Non-tool_result blocks pass through both ways untouched.
        for b in [
            json!({"type": "text", "text": "hi"}),
            json!({"type": "thinking", "thinking": "…", "signature": "sig"}),
            json!({"type": "tool_use", "id": "TU-y", "name": "Bash", "input": {}}),
        ] {
            assert_eq!(block_to_neo(&b), b);
            assert_eq!(block_to_anthropic(&b), b);
        }
    }

    /// aivo#14 legacy threads: a thread persisted in the OLD Anthropic
    /// tool_result shape (`tool_use_id`/`content`) must be served to amp
    /// in the neo shape (`toolUseID`/`run`) — otherwise amp's loader runs
    /// `M-${undefined.replace(...)}` and TypeErrors. getThread/getThreadTail
    /// upconvert on the fly so already-saved threads load without migration.
    #[tokio::test]
    async fn get_thread_upconverts_legacy_anthropic_tool_results_to_neo() {
        let dir = tempfile::tempdir().unwrap();
        let id = "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0";
        let tu = "TU-bbbbbbbbbbbbbbbbbbbbbb";
        // Legacy on-disk shape (what every pre-fix thread looks like).
        let payload = json!({
            "id": id, "v": 2, "title": "legacy", "created": "2026-06-09T00:00:00Z",
            "messages": [
                {"role": "assistant", "messageId": "M-a0000000000000000000000",
                 "content": [{"type": "tool_use", "id": tu, "name": "Bash", "input": {}}]},
                {"role": "user", "messageId": "M-f0000000000000000000000",
                 "content": [{"type": "tool_result", "tool_use_id": tu,
                              "content": "legacy output", "is_error": false}]},
            ],
        });
        amp_threads::save_thread(dir.path(), &payload)
            .await
            .unwrap();
        let state = bridge_state_with_threads_dir(dir.path().to_path_buf());

        for method in ["getThread", "getThreadTail"] {
            let body =
                format!(r#"{{"method":"{method}","params":{{"thread":"{id}","limit":76}}}}"#);
            let resp = handle_internal_rpc(&state, method, &body).await;
            let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
            let block = &v["result"]["thread"]["data"]["messages"][1]["content"][0];
            // Served as neo: defined toolUseID (no R.replace crash) + run.
            assert_eq!(block["toolUseID"].as_str(), Some(tu), "{method}");
            assert_eq!(block["run"]["status"].as_str(), Some("done"), "{method}");
            assert_eq!(
                block["run"]["result"].as_str(),
                Some("legacy output"),
                "{method}"
            );
            assert!(
                block["tool_use_id"].is_null(),
                "{method}: snake_case id dropped"
            );
        }
    }

    /// neo's switch-thread preview: `getThreadTail` must serve the
    /// thread object + message tail so the picker's preview pane renders.
    /// Regression for the empty "Preview Unavailable" pane (generic stub
    /// returned `result:null`).
    #[tokio::test]
    async fn get_thread_tail_serves_thread_and_message_tail() {
        let dir = tempfile::tempdir().unwrap();
        let id = "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0";
        let messages: Vec<serde_json::Value> = (0..5)
            .map(|i| json!({"role": "user", "messageId": format!("M-{i}"), "content": []}))
            .collect();
        let payload = json!({
            "id": id, "v": 2, "title": "preview me",
            "created": "2026-05-29T00:00:00Z", "messages": messages,
        });
        amp_threads::save_thread(dir.path(), &payload)
            .await
            .unwrap();
        let state = bridge_state_with_threads_dir(dir.path().to_path_buf());

        let body =
            format!(r#"{{"method":"getThreadTail","params":{{"thread":"{id}","limit":3}}}}"#);
        let resp = handle_internal_rpc(&state, "getThreadTail", &body).await;
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(v["ok"], true);
        // amp's consumer merges `{...result.thread.data, messages}`.
        assert_eq!(v["result"]["thread"]["data"]["title"], "preview me");
        let tail = v["result"]["messages"].as_array().unwrap();
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0]["messageId"], "M-2");
        assert_eq!(tail[2]["messageId"], "M-4");
        // data.messages stays consistent with the tail (no double history).
        assert_eq!(v["result"]["thread"]["data"]["messages"], json!(tail));
    }

    /// aivo#14 stuck-Librarian: `GET /api/threads/find?q=...` used to
    /// fall through to the 404 UNHANDLED branch, leaving the Librarian
    /// tool hung. dispatch now serves the `{threads, hasMore}` envelope
    /// amp parses, with `+`/`%XX`-encoded query params decoded.
    #[tokio::test]
    async fn dispatch_serves_threads_find_envelope() {
        let dir = tempfile::tempdir().unwrap();
        amp_threads::save_thread(
            dir.path(),
            &json!({"id": "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0", "v": 2,
                    "title": "Librarian design", "created": "2026-05-29T00:00:00Z",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}),
        )
        .await
        .unwrap();
        let state = bridge_state_with_threads_dir(dir.path().to_path_buf());

        // `+` is a space, so q = "librarian OR libraian".
        let full_path = "/api/threads/find?q=librarian+OR+libraian&limit=10";
        let req = format!("GET {full_path} HTTP/1.1\r\n\r\n");
        let resp = dispatch(&state, &req, "GET", full_path, "").await.unwrap();
        let BridgeResponse::Buffered { status, body, .. } = resp else {
            panic!("expected buffered response");
        };
        assert_eq!(status, 200, "must not 404 into the UNHANDLED branch");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        // Exact shape amp's Librarian destructures: `{threads, hasMore}`.
        let threads = v["threads"].as_array().expect("threads array");
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0]["id"], "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0");
        assert_eq!(v["hasMore"], false);
    }

    /// `/api/user-actor-credentials` must win over the `/api/user`
    /// catch-all and return non-empty userId/wsToken/poolName, else amp
    /// retry-loops on "Loading thread".
    #[tokio::test]
    async fn dispatch_serves_user_actor_credentials_shape() {
        let state = fake_bridge_state();
        let full_path = "/api/user-actor-credentials";
        let req = format!("POST {full_path} HTTP/1.1\r\n\r\n");
        let resp = dispatch(&state, &req, "POST", full_path, "{}")
            .await
            .unwrap();
        let BridgeResponse::Buffered { status, body, .. } = resp else {
            panic!("expected buffered response");
        };
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["userId"], "user_aivo_local");
        assert_eq!(v["wsToken"], "aivo-bridge");
        assert!(
            v["poolName"].as_str().is_some_and(|s| !s.is_empty()),
            "poolName must be a non-empty string, got {v:?}"
        );
        assert!(v.get("userEmail").is_none(), "misrouted to /api/user stub");
    }

    /// A find query matching nothing must still return a well-formed
    /// empty envelope (status 200) — never the 404 the Librarian hangs on.
    #[tokio::test]
    async fn dispatch_threads_find_empty_result_is_well_formed() {
        let dir = tempfile::tempdir().unwrap();
        let state = bridge_state_with_threads_dir(dir.path().to_path_buf());
        let full_path = "/api/threads/find?q=nomatch&limit=10";
        let req = format!("GET {full_path} HTTP/1.1\r\n\r\n");
        let resp = dispatch(&state, &req, "GET", full_path, "").await.unwrap();
        let BridgeResponse::Buffered { status, body, .. } = resp else {
            panic!("expected buffered response");
        };
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["threads"], json!([]));
        assert_eq!(v["hasMore"], false);
    }

    #[tokio::test]
    async fn get_thread_tail_unknown_thread_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let state = bridge_state_with_threads_dir(dir.path().to_path_buf());
        let body = r#"{"method":"getThreadTail","params":{"thread":"T-missing","limit":76}}"#;
        let resp = handle_internal_rpc(&state, "getThreadTail", body).await;
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        // `thread-not-found` lets neo's wrapper return null cleanly
        // instead of throwing "Unexpected error inside Amp CLI".
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"]["code"], "thread-not-found");
    }

    /// executor_tool_result arrives for one of two pending tools: state
    /// folds the result into `tool_results`, decrements pending,
    /// emits `executor_tool_result_ack` back to the executor (so its
    /// per-tool TUI marker can transition from `::` spinner to `$`
    /// completed), but does NOT trigger the next turn until the second
    /// tool resolves.
    #[tokio::test]
    async fn executor_tool_result_folds_into_state_and_acks_and_waits_for_siblings() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        ws_state.bootstrap_complete = true;
        let tu_a = new_tool_call_id();
        let tu_b = new_tool_call_id();
        ws_state.pending_tool_uses.insert(tu_a.clone());
        ws_state.pending_tool_uses.insert(tu_b.clone());

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "executor_tool_result",
            "params": {
                "toolCallId": tu_a,
                "run": {"status": "done", "result": "hello stdout"},
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        assert_eq!(ws_state.pending_tool_uses.len(), 1);
        assert!(ws_state.pending_tool_uses.contains(&tu_b));
        assert_eq!(ws_state.tool_results.len(), 1);
        // Accumulated in amp's neo shape: {toolUseID, run:{status,result}}.
        let r = &ws_state.tool_results[0];
        assert_eq!(r["type"].as_str(), Some("tool_result"));
        assert_eq!(r["toolUseID"].as_str(), Some(tu_a.as_str()));
        assert_eq!(r["run"]["status"].as_str(), Some("done"));
        assert_eq!(r["run"]["result"].as_str(), Some("hello stdout"));

        // We must ack the result back to the executor — otherwise
        // amp's TUI keeps the per-tool spinner on `::` indefinitely.
        let ack: serde_json::Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
        assert_eq!(ack["method"].as_str(), Some("executor_tool_result_ack"));
        assert_eq!(ack["params"]["toolCallId"].as_str(), Some(tu_a.as_str()));
        // Next turn must NOT have fired (sibling still pending) — no
        // more outbound frames.
        assert!(rx.recv().await.is_none(), "expected only the ack");
    }

    /// executor_skill_snapshot with isLast=true REPLACES skills,
    /// isLast=false APPENDS — covers amp's multi-frame snapshot
    /// streaming pattern.
    #[tokio::test]
    async fn executor_skill_snapshot_replaces_or_appends() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        ws_state.skills = vec![json!({"name": "old", "description": ""})];

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "executor_skill_snapshot",
            "params": {
                "snapshotId": "snap-1",
                "skills": [{"name": "new", "description": "fresh"}],
                "isLast": true,
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        assert_eq!(ws_state.skills.len(), 1);
        assert_eq!(ws_state.skills[0]["name"].as_str(), Some("new"));

        // Appending stream (isLast=false) preserves earlier entries.
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "executor_skill_snapshot",
            "params": {
                "snapshotId": "snap-2",
                "skills": [{"name": "more", "description": ""}],
                "isLast": false,
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        assert_eq!(ws_state.skills.len(), 2);
    }

    /// executor_guidance_snapshot stores AGENTS.md files for later
    /// injection into the system prompt.
    #[tokio::test]
    async fn executor_guidance_snapshot_stores_files() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "executor_guidance_snapshot",
            "params": {
                "snapshotId": "g-1",
                "files": [
                    {"uri": "file:///proj/AGENTS.md", "content": "Always run cargo fmt."},
                ],
                "isLast": true,
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        assert_eq!(ws_state.guidance_files.len(), 1);
        assert_eq!(
            ws_state.guidance_files[0]["content"].as_str(),
            Some("Always run cargo fmt.")
        );
    }

    /// executor_environment_snapshot + executor_environment_update
    /// both fill the environment slot. The system prompt builder
    /// surfaces cwd and git branch from it.
    #[tokio::test]
    async fn executor_environment_snapshot_populates_workspace() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "executor_environment_snapshot",
            "params": {
                "environment": {
                    "workingDirectory": "/tmp/proj",
                    "git": {"branch": "main"},
                },
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        let env = ws_state.environment.unwrap();
        assert_eq!(env["workingDirectory"].as_str(), Some("/tmp/proj"));
        assert_eq!(env["git"]["branch"].as_str(), Some("main"));
    }

    /// build_system_prompt weaves the workspace + guidance + skills
    /// into a single prompt string the upstream LLM sees.
    #[test]
    fn system_prompt_surfaces_workspace_guidance_skills() {
        let env = json!({
            "workingDirectory": "/tmp/repo",
            "git": {"branch": "feature/x"},
        });
        let guidance = vec![json!({
            "uri": "file:///tmp/repo/AGENTS.md",
            "content": "Use snake_case for module names.",
        })];
        let skills = vec![json!({
            "name": "deploy",
            "description": "Ship the current branch",
        })];
        let prompt = build_system_prompt(&[], Some(&env), &skills, &guidance, None, None);
        assert!(prompt.contains("cwd: /tmp/repo"));
        assert!(prompt.contains("git branch: feature/x"));
        assert!(prompt.contains("AGENTS.md"));
        assert!(prompt.contains("Use snake_case"));
        assert!(prompt.contains("deploy: Ship the current branch"));
    }

    /// Pinned guidance from real failure-mode analysis (e.g. amp
    /// session T-bb00e7 vs pi session 019e6edf, where pi succeeded
    /// by running `aivo keys export --help` and discovering
    /// `--password-stdin` while amp thrashed through pty workarounds).
    /// Don't drop these without re-litigating.
    #[test]
    fn base_prompt_pins_subcommand_help_and_tty_failure_guidance() {
        let prompt = build_system_prompt(&[], None, &[], &[], None, None);
        // "Read --help on the specific subcommand" — addresses the
        // failure where amp skipped from parent --help to running the
        // command, missing `--password-stdin`.
        assert!(
            prompt.contains("--password-stdin"),
            "subcommand --help guidance dropped"
        );
        // "After a TTY error, re-read --help instead of pty hacks" —
        // addresses the thrash through `expect`, `script`, python pty.
        assert!(
            prompt.contains("/dev/tty") || prompt.contains("device not configured"),
            "TTY-failure recovery guidance dropped"
        );
    }

    /// The built-in base prompt covers every category on our coverage
    /// checklist. Substring assertions on header text — change the
    /// BASE_PROMPT and these tests tell you what you dropped.
    #[test]
    fn base_prompt_covers_all_checklist_sections() {
        let prompt = build_system_prompt(&[], None, &[], &[], None, None);
        for section in [
            "## Tool use",
            "## Editing code",
            "## Communication",
            "## Verification",
            "## Handling ambiguity",
            "## When stuck",
            "## Safety",
            "## Workspace",
        ] {
            assert!(
                prompt.contains(section),
                "missing checklist section {section}"
            );
        }
    }

    /// Per-mode addendums kick in for known modes and are absent for
    /// smart / unknown modes (base prompt is the balanced version).
    #[test]
    fn mode_addendums_calibrate_per_mode() {
        for (mode, expected) in [
            (Some("rush"), "Mode: rush"),
            (Some("deep"), "Mode: deep"),
            (Some("large"), "Mode: large"),
            (Some("frontier"), "Mode: frontier"),
        ] {
            let prompt = build_system_prompt(&[], None, &[], &[], mode, None);
            assert!(
                prompt.contains(expected),
                "mode {mode:?} missing addendum {expected}"
            );
        }
        // smart and unknown modes don't get an addendum.
        for mode in [Some("smart"), Some("ufo"), None] {
            let prompt = build_system_prompt(&[], None, &[], &[], mode, None);
            assert!(
                !prompt.contains("## Mode:"),
                "mode {mode:?} unexpectedly carries an addendum"
            );
        }
    }

    /// override_base completely replaces the base prompt — workspace /
    /// guidance / skills / tools sections still append.
    #[test]
    fn override_base_replaces_built_in_but_keeps_appended_sections() {
        let env = json!({"workingDirectory": "/proj", "git": {"branch": "main"}});
        let prompt = build_system_prompt(
            &[],
            Some(&env),
            &[],
            &[],
            Some("deep"),
            Some("CUSTOM USER PROMPT GOES HERE"),
        );
        // Override is the base.
        assert!(prompt.starts_with("CUSTOM USER PROMPT GOES HERE"));
        // Built-in BASE_PROMPT sections are NOT present.
        assert!(!prompt.contains("## Tool use"));
        // Mode addendum is suppressed when override is used.
        assert!(!prompt.contains("## Mode:"));
        // Workspace context still appends.
        assert!(prompt.contains("cwd: /proj"));
        assert!(prompt.contains("git branch: main"));
    }

    /// resolve_prompt_override: per-mode file wins, then default.md,
    /// then None. File reads point at a temp HOME to keep the test
    /// hermetic.
    #[tokio::test]
    async fn resolve_prompt_override_picks_per_mode_then_default() {
        let home =
            std::env::temp_dir().join(format!("aivo-amp-prompt-override-{}", std::process::id()));
        let dir = home
            .join(".config")
            .join("aivo")
            .join("amp")
            .join("prompts");
        std::fs::create_dir_all(&dir).unwrap();

        // Stand-in for system_env::home_dir() via env var override.
        // (system_env::home_dir reads HOME on unix.)
        let saved = std::env::var("HOME").ok();
        // SAFETY: tokio::test runs each #[tokio::test] in its own
        // single-threaded runtime; no other test threads concurrently
        // touch HOME. We restore at the end of the test.
        unsafe {
            std::env::set_var("HOME", &home);
        }

        // No files yet → None.
        assert!(resolve_prompt_override(Some("rush")).await.is_none());

        // default.md only → used for any mode.
        std::fs::write(dir.join("default.md"), "default override").unwrap();
        assert_eq!(
            resolve_prompt_override(Some("rush")).await.as_deref(),
            Some("default override"),
        );
        assert_eq!(
            resolve_prompt_override(None).await.as_deref(),
            Some("default override"),
        );

        // Mode-specific file takes precedence.
        std::fs::write(dir.join("rush.md"), "rush override").unwrap();
        assert_eq!(
            resolve_prompt_override(Some("rush")).await.as_deref(),
            Some("rush override"),
        );
        // Other modes still see default.
        assert_eq!(
            resolve_prompt_override(Some("deep")).await.as_deref(),
            Some("default override"),
        );

        // Empty (whitespace-only) override file falls through to
        // built-in (treated as absent).
        std::fs::write(dir.join("deep.md"), "   \n\t\n  ").unwrap();
        assert_eq!(
            resolve_prompt_override(Some("deep")).await.as_deref(),
            Some("default override"),
        );

        // Cleanup.
        let _ = std::fs::remove_dir_all(&home);
        unsafe {
            if let Some(h) = saved {
                std::env::set_var("HOME", h);
            } else {
                std::env::remove_var("HOME");
            }
        }
    }

    /// client_update_thread_settings merges the incoming settings into
    /// WsState and rebroadcasts as `thread_settings` so amp's TUI
    /// observers see the update.
    #[tokio::test]
    async fn client_update_thread_settings_merges_and_broadcasts() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        // First update sets reasoning.effort.
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_update_thread_settings",
            "params": {
                "settings": {"reasoning.effort": "high"},
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        // Second update adds a different field — both should survive.
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_update_thread_settings",
            "params": {
                "settings": {"internal.model": "claude-opus-4-7"},
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        assert_eq!(
            ws_state.thread_settings["reasoning.effort"].as_str(),
            Some("high")
        );
        assert_eq!(
            ws_state.thread_settings["internal.model"].as_str(),
            Some("claude-opus-4-7")
        );

        // Two thread_settings broadcasts, second carries merged settings.
        let mut broadcasts: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            broadcasts.push(serde_json::from_str(&s).unwrap());
        }
        assert_eq!(broadcasts.len(), 2);
        assert_eq!(broadcasts[1]["method"].as_str(), Some("thread_settings"));
        let merged = &broadcasts[1]["params"]["settings"];
        assert_eq!(merged["reasoning.effort"].as_str(), Some("high"));
        assert_eq!(merged["internal.model"].as_str(), Some("claude-opus-4-7"));
    }

    /// `amp threads continue T-<id>` opens a WS with the same thread
    /// id and immediately sends client_resume. The bridge must hydrate
    /// from disk so the resume has something to replay; the resume arm
    /// then re-emits each persisted message as `message_added` plus a
    /// final agent_state idle.
    #[tokio::test]
    async fn client_resume_replays_persisted_history() {
        let state = fake_bridge_state();
        let dir = std::env::temp_dir().join(format!(
            "aivo-amp-bridge-resume-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let thread_id = "T-resume00-0000-0000-0000-000000000001";

        // Write a thread file with one user + one assistant message.
        let payload = json!({
            "id": thread_id,
            "v": 2,
            "title": "earlier work",
            "messages": [
                {
                    "messageId": "M-user000000000000000000a",
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "messageId": "M-asst000000000000000000a",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi back"}],
                    "state": {"type": "complete"},
                },
            ],
        });
        std::fs::write(
            dir.join(format!("{thread_id}.json")),
            serde_json::to_vec(&payload).unwrap(),
        )
        .unwrap();

        // Build state pointing at this temp dir.
        let bridge_state = AmpBridgeState {
            config: Arc::new(AmpBridgeConfig {
                threads_dir: dir.clone(),
                ..(*state.config).clone()
            }),
            client: state.client.clone(),
        };
        let mut ws_state = WsState::from_path(&format!("/actors?rvt-key={thread_id}"));
        ws_state.hydrate_from_disk(&dir).await;

        // Hydration loaded the prior conversation into both vecs.
        assert_eq!(ws_state.persisted_messages.len(), 2);
        assert_eq!(ws_state.messages.len(), 2);
        assert_eq!(ws_state.title.as_deref(), Some("earlier work"));
        assert!(
            ws_state
                .seen_user_message_ids
                .contains("M-user000000000000000000a")
        );

        // client_resume → emits message_added for both messages + idle.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_resume",
            "params": {"version": 0},
        })
        .to_string();
        ws_followup_events(&bridge_state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        let methods: Vec<&str> = frames
            .iter()
            .map(|v| v["method"].as_str().unwrap_or(""))
            .collect();
        // Replays both persisted messages; does NOT emit agent_state
        // (that would clobber whatever working/streaming state the
        // active turn has set).
        assert_eq!(methods, vec!["message_added", "message_added"]);
        // First message_added is the user, second is the assistant.
        assert_eq!(
            frames[0]["params"]["message"]["role"].as_str(),
            Some("user")
        );
        assert_eq!(
            frames[1]["params"]["message"]["role"].as_str(),
            Some("assistant")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression for the "status flashes then disappears" bug:
    /// client_resume on a fresh (non-hydrated) thread must NOT
    /// replay messages or emit agent_state. The messages already in
    /// persisted_messages were emitted inline by their handlers
    /// (e.g. the user-msg ack from client_append_user_msg); replaying
    /// would duplicate them in amp's TUI. Emitting agent_state idle
    /// would wipe the working/streaming state of the in-flight turn.
    #[tokio::test]
    async fn client_resume_is_noop_for_fresh_thread() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        // Simulate the state after client_append_user_msg ran: a user
        // message persisted but `hydrated` is false (this is a fresh
        // thread, not loaded from disk).
        ws_state.persisted_messages.push(json!({
            "messageId": "M-user000000000000000000a",
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
            "createdAt": "2026-05-28T00:00:00Z",
        }));
        ws_state.persisted_to_msgs_idx.push(0);
        assert!(!ws_state.hydrated);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_resume",
            "params": {"version": 0},
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        // No frames at all — replay skipped, no spurious idle.
        assert!(
            rx.recv().await.is_none(),
            "client_resume on a fresh thread must not emit anything"
        );
    }

    /// Hydration is a no-op when the thread file doesn't exist
    /// (new thread / first connection) — state stays empty.
    #[tokio::test]
    async fn hydrate_from_disk_is_noop_for_unknown_thread() {
        let dir = std::env::temp_dir().join(format!(
            "aivo-amp-bridge-hydrate-noop-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut ws_state = fake_ws_state();
        let before_len = ws_state.persisted_messages.len();
        ws_state.hydrate_from_disk(&dir).await;
        assert_eq!(ws_state.persisted_messages.len(), before_len);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// When ALL pending tool_uses resolve, the bridge synthesizes a
    /// user message with the tool_result blocks AND persists it. Prior
    /// to this fix, persisted_messages was missing the fold-up, so
    /// `amp threads continue T-<id>` resumed with no record of what
    /// tools returned.
    #[tokio::test]
    async fn executor_tool_result_persists_fold_up_when_all_resolve() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        ws_state.bootstrap_complete = true;
        // Seed a user → assistant turn so the persisted_to_msgs_idx
        // tracking has real entries before the fold-up arrives.
        seed_one_turn(&mut ws_state, "M-userseed00000000000000", "do thing");
        let persisted_before = ws_state.persisted_messages.len();
        let idx_before = ws_state.persisted_to_msgs_idx.len();

        let tu = new_tool_call_id();
        ws_state.pending_tool_uses.insert(tu.clone());

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "executor_tool_result",
            "params": {
                "toolCallId": tu,
                "run": {"status": "done", "result": "stdout text"},
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;

        // The fold-up was persisted: one new user message with the
        // tool_result block, plus a matching messages-index entry.
        assert_eq!(ws_state.persisted_messages.len(), persisted_before + 1);
        assert_eq!(ws_state.persisted_to_msgs_idx.len(), idx_before + 1);

        let fold_up = ws_state.persisted_messages.last().unwrap();
        assert_eq!(fold_up["role"].as_str(), Some("user"));
        assert_eq!(fold_up["threadId"], json!(ws_state.thread_id));
        assert!(
            fold_up["messageId"]
                .as_str()
                .unwrap_or("")
                .starts_with("M-"),
            "messageId got {:?}",
            fold_up["messageId"]
        );
        assert!(fold_up["createdAt"].is_string());
        let blocks = fold_up["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        // Persisted in amp's neo shape so `threads continue` can reload a
        // thread with tool calls (aivo#14: an Anthropic `tool_use_id`
        // block leaves amp's `toolUseID` undefined → `R.replace` crash).
        assert_eq!(blocks[0]["type"].as_str(), Some("tool_result"));
        assert_eq!(blocks[0]["toolUseID"].as_str(), Some(tu.as_str()));
        assert_eq!(blocks[0]["run"]["status"].as_str(), Some("done"));
        assert_eq!(blocks[0]["run"]["result"].as_str(), Some("stdout text"));
    }

    /// Round-trip: a persisted neo tool_result must (a) reload with a
    /// defined `toolUseID` (so amp's loader doesn't TypeError) and (b)
    /// rehydrate into the Anthropic `tool_use_id`/`content` shape the
    /// upstream LLM expects on `amp threads continue`.
    #[tokio::test]
    async fn persisted_neo_tool_result_rehydrates_to_anthropic_for_llm() {
        let dir = tempfile::tempdir().unwrap();
        let tu = "TU-aaaaaaaaaaaaaaaaaaaaaa";
        let payload = json!({
            "id": "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0", "v": 2,
            "title": "tool thread", "created": "2026-06-09T00:00:00Z",
            "messages": [
                {"role": "user", "messageId": "M-u0000000000000000000000",
                 "content": [{"type": "text", "text": "run it"}]},
                {"role": "assistant", "messageId": "M-a0000000000000000000000",
                 "content": [{"type": "tool_use", "id": tu, "name": "Bash", "input": {}}]},
                {"role": "user", "messageId": "M-f0000000000000000000000",
                 "content": [{"type": "tool_result", "toolUseID": tu,
                              "run": {"status": "error", "result": "boom"}}]},
            ],
        });
        amp_threads::save_thread(dir.path(), &payload)
            .await
            .unwrap();

        let mut ws_state = fake_ws_state();
        ws_state.thread_id = "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0".to_string();
        ws_state.hydrate_from_disk(dir.path()).await;

        // On disk: neo shape preserved (toolUseID defined — no R.replace).
        let disk = amp_threads::load_thread(dir.path(), &ws_state.thread_id)
            .await
            .unwrap();
        let neo = &disk["messages"][2]["content"][0];
        assert_eq!(neo["toolUseID"].as_str(), Some(tu));

        // In the LLM history: converted to Anthropic shape.
        let llm = ws_state.messages.last().unwrap();
        let block = &llm["content"][0];
        assert_eq!(block["type"].as_str(), Some("tool_result"));
        assert_eq!(block["tool_use_id"].as_str(), Some(tu));
        assert_eq!(block["content"].as_str(), Some("boom"));
        assert_eq!(block["is_error"].as_bool(), Some(true));
        // The assistant tool_use block (identical across formats) is intact.
        assert_eq!(ws_state.messages[1]["content"][0]["id"].as_str(), Some(tu));
    }

    #[test]
    fn mark_thread_actor_native_skips_import_and_replay() {
        // Legacy thread served by getThread: low `v`, no actor flags,
        // an existing `meta` object that must be preserved.
        let mut payload = json!({
            "id": "T-legacy",
            "v": 2,
            "meta": {"traces": []},
            "messages": [{"role": "user"}, {"role": "assistant"}, {"role": "user"}],
        });
        mark_thread_actor_native(&mut payload);
        // needsImport = meta.usesThreadActors !== true → false, so neo
        // resumes directly instead of running the import dance.
        assert_eq!(payload["meta"]["usesThreadActors"].as_bool(), Some(true));
        assert_eq!(payload["usesThreadActors"].as_bool(), Some(true));
        // Existing meta keys survive.
        assert!(payload["meta"]["traces"].is_array());
        // `v` lands at the message count so client_resume's replay
        // cursor is at the end — no duplicate message_added burst.
        assert_eq!(payload["v"].as_u64(), Some(3));
    }

    #[test]
    fn thread_payload_carries_required_fields() {
        let ws_state = fake_ws_state();
        let payload = ws_state.thread_payload();
        // Schema fields amp's resume + listThreads expect.
        assert!(payload["id"].is_string());
        assert_eq!(payload["v"].as_u64(), Some(2));
        assert!(payload["messages"].is_array());
        assert_eq!(payload["usesDtw"].as_bool(), Some(false));
        // Actor-native so neo resumes directly instead of importing —
        // see `mark_thread_actor_native`.
        assert_eq!(payload["usesThreadActors"].as_bool(), Some(true));
        assert_eq!(payload["meta"]["usesThreadActors"].as_bool(), Some(true));
        // `created` is set at WsState construction and must be a parseable
        // ISO-8601 timestamp — amp passes it to `new Date(...)`.
        let created = payload["created"].as_str().expect("created is string");
        chrono::DateTime::parse_from_rfc3339(created).expect("parseable rfc3339");
    }

    #[test]
    fn auto_title_derives_from_first_user_message() {
        let mut ws_state = fake_ws_state();
        ws_state.auto_title_from_user_content(&json!([
            {"type": "text", "text": "How do I run the test suite?"}
        ]));
        assert_eq!(
            ws_state.title.as_deref(),
            Some("How do I run the test suite?")
        );
        // Second user message must not overwrite an auto-derived title.
        ws_state.auto_title_from_user_content(&json!([
            {"type": "text", "text": "Another question"}
        ]));
        assert_eq!(
            ws_state.title.as_deref(),
            Some("How do I run the test suite?")
        );
    }

    #[test]
    fn auto_title_truncates_long_messages_with_ellipsis() {
        let mut ws_state = fake_ws_state();
        let long = "x".repeat(120);
        ws_state.auto_title_from_user_content(&json!([
            {"type": "text", "text": long}
        ]));
        let title = ws_state.title.unwrap();
        assert!(title.ends_with('…'));
        assert_eq!(title.chars().count(), 61); // 60 chars + ellipsis
    }

    #[test]
    fn auto_title_handles_plain_string_content() {
        // amp may send `content` as a bare string instead of a blocks array.
        let mut ws_state = fake_ws_state();
        ws_state.auto_title_from_user_content(&json!("hello world"));
        assert_eq!(ws_state.title.as_deref(), Some("hello world"));
    }

    #[test]
    fn auto_title_skips_empty_or_whitespace_content() {
        let mut ws_state = fake_ws_state();
        ws_state.auto_title_from_user_content(&json!([{"type": "text", "text": "   "}]));
        assert_eq!(ws_state.title, None);
        ws_state.auto_title_from_user_content(&json!([]));
        assert_eq!(ws_state.title, None);
    }

    #[tokio::test]
    async fn inference_tools_emits_tool_names_and_agent_mode() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let mut ws_state = fake_ws_state();
        ws_state.tools = vec![
            json!({"name": "Bash", "description": "", "input_schema": {}}),
            json!({"name": "Read", "description": "", "input_schema": {}}),
            json!({"description": "no name field"}),
        ];
        ws_state.last_agent_mode = Some(json!("deep"));

        emit_inference_tools(&tx, None, "/test", &ws_state, "M-abc").await;
        drop(tx);

        let frame: serde_json::Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
        assert_eq!(frame["method"].as_str(), Some("inference_tools"));
        let params = &frame["params"];
        assert_eq!(params["messageId"].as_str(), Some("M-abc"));
        assert_eq!(params["agentMode"].as_str(), Some("deep"));
        let tools: Vec<&str> = params["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(tools, vec!["Bash", "Read"]);
    }

    /// Seeds a WsState with a one-turn history: one user message + one
    /// assistant reply. Both vecs and the parallel index get the right
    /// entries so the rewind helpers have something to rewind from.
    fn seed_one_turn(ws_state: &mut WsState, user_msg_id: &str, user_text: &str) {
        ws_state.messages.push(json!({
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }));
        ws_state.persisted_messages.push(json!({
            "messageId": user_msg_id,
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }));
        ws_state
            .persisted_to_msgs_idx
            .push(ws_state.messages.len() - 1);

        ws_state.messages.push(json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }));
        ws_state.persisted_messages.push(json!({
            "messageId": new_message_id(),
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }));
        ws_state
            .persisted_to_msgs_idx
            .push(ws_state.messages.len() - 1);
    }

    /// rewind_to_before_last_assistant drops just the last assistant
    /// turn (no tools, no fold-ups) so the next agent turn replays
    /// against the user message that triggered it.
    #[test]
    fn rewind_drops_last_assistant_and_keeps_user() {
        let mut ws_state = fake_ws_state();
        seed_one_turn(&mut ws_state, "M-user1111111111111111111", "hi");
        ws_state.pending_tool_uses.insert(new_tool_call_id());
        ws_state.tool_results.push(json!({"x": 1}));

        assert!(rewind_to_before_last_assistant(&mut ws_state));
        // User message stays.
        assert_eq!(ws_state.messages.len(), 1);
        assert_eq!(ws_state.persisted_messages.len(), 1);
        assert_eq!(ws_state.persisted_to_msgs_idx, vec![0]);
        assert_eq!(
            ws_state.persisted_messages[0]["role"].as_str(),
            Some("user")
        );
        // In-flight state cleared.
        assert!(ws_state.pending_tool_uses.is_empty());
        assert!(ws_state.tool_results.is_empty());
    }

    /// rewind on a fresh state (no messages) returns false and leaves
    /// state untouched.
    #[test]
    fn rewind_noop_when_no_assistant_in_history() {
        let mut ws_state = fake_ws_state();
        // Only a user message — nothing to rewind to.
        ws_state
            .messages
            .push(json!({"role": "user", "content": []}));
        ws_state.persisted_messages.push(json!({
            "messageId": "M-only00000000000000000000",
            "role": "user",
            "content": [],
        }));
        ws_state.persisted_to_msgs_idx.push(0);

        assert!(!rewind_to_before_last_assistant(&mut ws_state));
        assert_eq!(ws_state.messages.len(), 1);
        assert_eq!(ws_state.persisted_messages.len(), 1);
    }

    /// rewind correctly walks past tool_result fold-up user messages
    /// that live in `messages` but never made it to `persisted_messages`.
    #[test]
    fn rewind_walks_past_tool_result_fold_ups() {
        let mut ws_state = fake_ws_state();
        // Turn 1: user → assistant (with tool_use) → tool_result fold → assistant.
        seed_one_turn(&mut ws_state, "M-aaaaaaaaaaaaaaaaaaaaaa", "step 1");
        // Insert a tool_result fold-up (not persisted).
        ws_state.messages.push(json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "TU-x", "content": ""}],
        }));
        // Second assistant turn that follows the fold-up.
        ws_state.messages.push(json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "final"}],
        }));
        ws_state.persisted_messages.push(json!({
            "messageId": new_message_id(),
            "role": "assistant",
            "content": [{"type": "text", "text": "final"}],
        }));
        ws_state
            .persisted_to_msgs_idx
            .push(ws_state.messages.len() - 1);

        assert!(rewind_to_before_last_assistant(&mut ws_state));
        // Drops the FIRST assistant (index 1) onwards — both the
        // tool_result fold-up and the second assistant disappear.
        // Wait: we drop from the LAST assistant. The last assistant
        // is at persisted_messages[2]. Its messages index is 3
        // (the second assistant in messages). So messages truncates
        // to 3 — keeps user(0), assistant(1), fold-up(2).
        //
        // After this rewind there's still the first assistant turn
        // visible — the user can retry the second one specifically.
        assert_eq!(ws_state.persisted_messages.len(), 2);
        assert_eq!(ws_state.messages.len(), 3);
    }

    /// rewind_to_user_message replaces the user's content and drops
    /// everything after — the assistant turn that followed disappears.
    #[test]
    fn rewind_to_user_message_truncates_and_replaces() {
        let mut ws_state = fake_ws_state();
        let mid = "M-edit00000000000000000000".to_string();
        seed_one_turn(&mut ws_state, &mid, "original");

        let new_content = json!([{"type": "text", "text": "rewritten"}]);
        assert!(rewind_to_user_message(
            &mut ws_state,
            &mid,
            new_content.clone()
        ));

        // Assistant gone, user message replaced with new content.
        assert_eq!(ws_state.messages.len(), 1);
        assert_eq!(ws_state.persisted_messages.len(), 1);
        assert_eq!(ws_state.persisted_messages[0]["content"], new_content,);
        assert_eq!(ws_state.messages[0]["content"], new_content);
        assert_eq!(
            ws_state.persisted_messages[0]["messageId"].as_str(),
            Some(mid.as_str())
        );
    }

    /// Unknown messageId is a no-op (don't corrupt state if amp races
    /// or sends an edit for a message we've already truncated away).
    #[test]
    fn rewind_to_user_message_noop_when_id_unknown() {
        let mut ws_state = fake_ws_state();
        seed_one_turn(&mut ws_state, "M-aaaaaaaaaaaaaaaaaaaaaa", "hi");
        let before_msgs = ws_state.messages.clone();
        let before_persisted = ws_state.persisted_messages.clone();

        assert!(!rewind_to_user_message(
            &mut ws_state,
            "M-nothere00000000000000000",
            json!([{"type": "text", "text": "x"}]),
        ));
        assert_eq!(ws_state.messages, before_msgs);
        assert_eq!(ws_state.persisted_messages, before_persisted);
    }

    /// client_filesystem_read_file (from TUI) forwards verbatim as
    /// executor_filesystem_read_file (to executor). Same routing
    /// applies to read_directory and git_command on the request side.
    #[tokio::test]
    async fn client_to_executor_request_routes_are_forwarded() {
        let state = fake_bridge_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let cases = [
            (
                "client_filesystem_read_file",
                "executor_filesystem_read_file",
                json!({"requestId": "req-1", "uri": "file:///tmp/a"}),
            ),
            (
                "client_filesystem_read_directory",
                "executor_filesystem_read_directory",
                json!({"requestId": "req-2", "uri": "file:///tmp"}),
            ),
            (
                "client_git_command",
                "executor_git_command",
                json!({"requestId": "req-3", "args": ["status"]}),
            ),
            (
                "client_git_diff_snapshot",
                "executor_git_diff_snapshot",
                json!({"requestId": "req-4", "baseRevision": "HEAD"}),
            ),
        ];
        for (inbound, expected_outbound, params) in cases {
            let mut ws_state = fake_ws_state();
            let frame = json!({
                "jsonrpc": "2.0",
                "method": inbound,
                "params": params,
            })
            .to_string();
            ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
            let v: serde_json::Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
            assert_eq!(
                v["method"].as_str(),
                Some(expected_outbound),
                "inbound {inbound}",
            );
            // Params survive verbatim — the bridge adds no policy.
            assert_eq!(
                v["params"]["requestId"], params["requestId"],
                "inbound {inbound}"
            );
        }
    }

    /// executor_*_result (from executor) forwards as client_*_result
    /// (to TUI) so the TUI's pending-request resolver can complete.
    #[tokio::test]
    async fn executor_to_client_result_routes_are_forwarded() {
        let state = fake_bridge_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let cases = [
            (
                "executor_filesystem_read_file_result",
                "client_filesystem_read_file_result",
                json!({"requestId": "req-1", "ok": true, "contentBase64": "aGk="}),
            ),
            (
                "executor_filesystem_read_directory_result",
                "client_filesystem_read_directory_result",
                json!({"requestId": "req-2", "ok": true, "entries": []}),
            ),
            (
                "executor_git_command_result",
                "client_git_command_result",
                json!({"requestId": "req-3", "ok": true, "exitCode": 0, "stdout": "", "stderr": ""}),
            ),
            (
                "executor_git_diff_snapshot_result",
                "client_git_diff_snapshot_result",
                json!({"requestId": "req-4", "ok": true, "files": []}),
            ),
        ];
        for (inbound, expected_outbound, params) in cases {
            let mut ws_state = fake_ws_state();
            let frame = json!({
                "jsonrpc": "2.0",
                "method": inbound,
                "params": params,
            })
            .to_string();
            ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
            let v: serde_json::Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
            assert_eq!(v["method"].as_str(), Some(expected_outbound));
            assert_eq!(v["params"]["requestId"], params["requestId"]);
            assert_eq!(v["params"]["ok"].as_bool(), Some(true));
        }
    }

    /// tool_progress from the executor must be rebroadcast verbatim
    /// so amp's TUI `onToolProgress` observer fires and the partial
    /// tool output (e.g. Bash stdout) renders during execution.
    #[tokio::test]
    async fn tool_progress_is_rebroadcast_to_tui() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let tu = new_tool_call_id();

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "tool_progress",
            "params": {
                "toolCallId": tu.clone(),
                "progress": {"type": "snapshot", "value": "partial stdout..."},
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        let echoed: serde_json::Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
        assert_eq!(echoed["method"].as_str(), Some("tool_progress"));
        assert_eq!(echoed["params"]["toolCallId"].as_str(), Some(tu.as_str()));
        assert_eq!(
            echoed["params"]["progress"]["value"].as_str(),
            Some("partial stdout...")
        );
    }

    /// executor_tool_approval_request enqueues the approval entry and
    /// rebroadcasts the full queue so amp's TUI observers can render
    /// the approve/deny prompt.
    #[tokio::test]
    async fn executor_tool_approval_request_enqueues_and_broadcasts() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let tu = new_tool_call_id();

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "executor_tool_approval_request",
            "params": {
                "approval": {
                    "id": "approval-1",
                    "toolCallId": tu.clone(),
                    "toolName": "Bash",
                    "args": {"cmd": "rm -rf /"},
                    "context": {},
                    "timestamp": 1_700_000_000,
                    "matchedRule": {"tool": "Bash", "action": "ask"},
                }
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        assert_eq!(ws_state.pending_approvals.len(), 1);
        assert_eq!(
            ws_state.pending_approvals[0]["toolCallId"].as_str(),
            Some(tu.as_str())
        );

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        // Queue broadcast + agent_state transition to awaiting_approval.
        let methods: Vec<&str> = frames
            .iter()
            .map(|v| v["method"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(methods, vec!["tool_approval_queue", "agent_state"]);
        let approvals = frames[0]["params"]["approvals"].as_array().unwrap();
        assert_eq!(approvals.len(), 1);
        assert_eq!(approvals[0]["toolName"].as_str(), Some("Bash"));
        // Status badge flips to "Waiting for approval".
        assert_eq!(
            frames[1]["params"]["state"].as_str(),
            Some("awaiting_approval")
        );
        assert_eq!(frames[1]["params"]["messageId"].as_str(), Some(tu.as_str()));
    }

    /// Duplicate request for the same toolCallId overwrites instead
    /// of double-queueing (defensive against executor retries).
    #[tokio::test]
    async fn executor_tool_approval_request_dedups_by_tool_call_id() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let tu = new_tool_call_id();

        for reason in ["initial", "retry"] {
            let frame = json!({
                "jsonrpc": "2.0",
                "method": "executor_tool_approval_request",
                "params": {
                    "approval": {
                        "id": format!("approval-{reason}"),
                        "toolCallId": tu.clone(),
                        "toolName": "Bash",
                        "args": {},
                        "context": {},
                        "timestamp": 1_700_000_000,
                        "matchedRule": {"tool": "Bash", "action": "ask"},
                        "reason": reason,
                    }
                },
            })
            .to_string();
            ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        }

        assert_eq!(ws_state.pending_approvals.len(), 1);
        // The retry's payload wins.
        assert_eq!(
            ws_state.pending_approvals[0]["reason"].as_str(),
            Some("retry")
        );
    }

    /// client_tool_approval_response forwards the verdict to the
    /// executor (renamed to `executor_tool_approval_response` per the
    /// schema) AND removes the entry from the pending queue, then
    /// rebroadcasts the shrunken queue.
    #[tokio::test]
    async fn client_tool_approval_response_forwards_and_dequeues() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let tu_a = new_tool_call_id();
        let tu_b = new_tool_call_id();

        // Seed the queue with two pending approvals.
        for tu in [&tu_a, &tu_b] {
            ws_state.pending_approvals.push(json!({
                "id": format!("approval-{tu}"),
                "toolCallId": tu,
                "toolName": "Bash",
                "args": {},
                "context": {},
                "timestamp": 1_700_000_000,
                "matchedRule": {"tool": "Bash", "action": "ask"},
            }));
        }

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_tool_approval_response",
            "params": {
                "toolCallId": tu_a,
                "accepted": true,
                "input": {"askAnswers": {}},
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        // Pending approvals shrunk to just tu_b.
        assert_eq!(ws_state.pending_approvals.len(), 1);
        assert_eq!(
            ws_state.pending_approvals[0]["toolCallId"].as_str(),
            Some(tu_b.as_str())
        );

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        // First: executor_tool_approval_response carrying the verdict.
        assert_eq!(
            frames[0]["method"].as_str(),
            Some("executor_tool_approval_response")
        );
        assert_eq!(
            frames[0]["params"]["toolCallId"].as_str(),
            Some(tu_a.as_str())
        );
        assert_eq!(frames[0]["params"]["accepted"].as_bool(), Some(true));
        // Then: tool_approval_queue with the surviving entry.
        assert_eq!(frames[1]["method"].as_str(), Some("tool_approval_queue"));
        assert_eq!(
            frames[1]["params"]["approvals"].as_array().unwrap().len(),
            1
        );
        // Final: agent_state transition. Still pending approvals, so
        // we stay on "awaiting_approval" rather than going back to
        // running_tools.
        assert_eq!(frames[2]["method"].as_str(), Some("agent_state"));
        assert_eq!(
            frames[2]["params"]["state"].as_str(),
            Some("awaiting_approval")
        );
    }

    /// When the LAST pending approval resolves, the status transitions
    /// to "running_tools" — the executor is now free to run the leases.
    #[tokio::test]
    async fn client_tool_approval_response_transitions_to_running_tools_when_queue_empties() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let tu = new_tool_call_id();
        ws_state.pending_approvals.push(json!({
            "id": "approval-only",
            "toolCallId": tu.clone(),
            "toolName": "Bash",
            "args": {},
            "context": {},
            "timestamp": 1_700_000_000,
            "matchedRule": {"tool": "Bash", "action": "ask"},
        }));

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_tool_approval_response",
            "params": {"toolCallId": tu, "accepted": true},
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        // Last frame is the agent_state transition back to running_tools.
        let last = frames.last().unwrap();
        assert_eq!(last["method"].as_str(), Some("agent_state"));
        assert_eq!(last["params"]["state"].as_str(), Some("running_tools"));
    }

    /// Denial path carries denyFeedback through to the executor so
    /// the model can read why the tool was rejected.
    #[tokio::test]
    async fn client_tool_approval_response_forwards_deny_feedback() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let tu = new_tool_call_id();
        ws_state.pending_approvals.push(json!({
            "id": "approval-1",
            "toolCallId": tu.clone(),
            "toolName": "Bash",
            "args": {},
            "context": {},
            "timestamp": 1_700_000_000,
            "matchedRule": {"tool": "Bash", "action": "ask"},
        }));

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_tool_approval_response",
            "params": {
                "toolCallId": tu,
                "accepted": false,
                "input": {"denyFeedback": "too risky"},
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        let first: serde_json::Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
        assert_eq!(
            first["params"]["input"]["denyFeedback"].as_str(),
            Some("too risky")
        );
        assert_eq!(first["params"]["accepted"].as_bool(), Some(false));
    }

    /// client_append_user_msg captures reasoningEffort onto WsState
    /// so the next agent_turn_finish can thread it into the upstream
    /// body via output_config.effort. The translator's existing
    /// effort-extraction pipeline takes over from there.
    #[tokio::test]
    async fn client_append_user_msg_records_reasoning_effort() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        ws_state.bootstrap_complete = false; // don't fire LLM call
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_append_user_msg",
            "params": {
                "messageId": "M-effort0000000000000000",
                "content": [{"type": "text", "text": "do the hard thing"}],
                "reasoningEffort": "high",
                "agentMode": "deep",
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;

        assert_eq!(ws_state.last_reasoning_effort.as_deref(), Some("high"));
        // Sanity: agentMode also captured (existing path).
        assert_eq!(
            ws_state.last_agent_mode.as_ref().and_then(|v| v.as_str()),
            Some("deep")
        );
    }

    #[tokio::test]
    async fn client_append_user_msg_ignores_blank_reasoning_effort() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        ws_state.last_reasoning_effort = Some("medium".to_string());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        // Frame omits reasoningEffort — previous value stays put.
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_append_user_msg",
            "params": {
                "messageId": "M-noeffort00000000000000",
                "content": [{"type": "text", "text": "follow-up"}],
            },
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        assert_eq!(ws_state.last_reasoning_effort.as_deref(), Some("medium"));
    }

    #[test]
    fn is_client_cancel_frame_parses_method() {
        assert!(is_client_cancel_frame(
            r#"{"jsonrpc":"2.0","method":"client_cancel","params":{}}"#
        ));
        // Different method — must not match.
        assert!(!is_client_cancel_frame(
            r#"{"jsonrpc":"2.0","method":"client_append_user_msg","params":{}}"#
        ));
        // No method field.
        assert!(!is_client_cancel_frame(r#"{"jsonrpc":"2.0"}"#));
        // User message that mentions cancel — substring matching would
        // false-positive, structured matching shouldn't.
        assert!(!is_client_cancel_frame(
            r#"{"method":"client_append_user_msg","params":{"content":"please client_cancel my order"}}"#
        ));
        // Garbage.
        assert!(!is_client_cancel_frame("not json"));
    }

    /// finish_turn_as_cancelled emits the precise event sequence amp's
    /// UI expects when a turn is bailed: cancelled notification, then
    /// the assistant message_updated with state:{type:"cancelled"},
    /// then agent_state idle to clear the spinner.
    #[tokio::test]
    async fn finish_turn_as_cancelled_emits_expected_sequence() {
        let mut ws_state = fake_ws_state();
        ws_state.cancel_flag.store(true, Ordering::SeqCst);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        finish_turn_as_cancelled(
            &mut ws_state,
            "M-cancelled000000000000000",
            &tx,
            None,
            "/test",
        )
        .await;
        drop(tx);

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        let methods: Vec<&str> = frames
            .iter()
            .map(|v| v["method"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(methods, vec!["cancelled", "message_updated", "agent_state"]);
        // cancelled carries the messageId.
        assert_eq!(
            frames[0]["params"]["messageId"].as_str(),
            Some("M-cancelled000000000000000")
        );
        // message_updated has state:{type:"cancelled"}.
        assert_eq!(
            frames[1]["params"]["message"]["state"]["type"].as_str(),
            Some("cancelled")
        );
        // agent_state drops to idle so amp's spinner clears.
        assert_eq!(frames[2]["params"]["state"].as_str(), Some("idle"));
        // Flag is reset so a follow-up turn isn't pre-cancelled.
        assert!(!ws_state.cancel_flag.load(Ordering::SeqCst));
    }

    /// emit_buffered_blocks_progressively stops chunking text and
    /// closes with state:"aborted" when the cancel flag flips mid-run.
    #[tokio::test]
    async fn buffered_emit_aborts_on_cancel_flag() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let cancel = Arc::new(AtomicBool::new(true)); // already cancelled
        let blocks = vec![json!({"type": "text", "text": "x".repeat(200)})];
        emit_buffered_blocks_progressively(&tx, None, "/test", "M-test", &blocks, &cancel).await;
        drop(tx);

        let mut states: Vec<String> = Vec::new();
        while let Some(s) = rx.recv().await {
            let v: serde_json::Value = serde_json::from_str(&s).unwrap();
            states.push(v["params"]["state"].as_str().unwrap_or("").to_string());
        }
        // No "generating" deltas (we bailed before the first chunk),
        // just the closing aborted state.
        assert_eq!(states, vec!["aborted".to_string()]);
    }

    /// agent_turn_finish must emit error_set when the upstream LLM
    /// call fails, so amp's UI shows the failure in its error bar
    /// instead of just rendering the error text inside the assistant
    /// turn. The final agent_state transitions to "error" (not
    /// "idle"), so the status badge visibly reflects the failure
    /// until the user takes their next action.
    #[tokio::test]
    async fn agent_turn_finish_emits_error_set_and_error_state_on_upstream_failure() {
        let state = fake_bridge_state(); // points at 127.0.0.1:1 (refused)
        let mut ws_state = fake_ws_state();
        ws_state.bootstrap_complete = true;
        ws_state.messages.push(json!({
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
        }));

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        agent_turn_finish(
            &state,
            &mut ws_state,
            "M-aaaaaaaaaaaaaaaaaaaaaa".to_string(),
            &tx,
            None,
            "/test",
        )
        .await;
        drop(tx);

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        let methods: Vec<&str> = frames
            .iter()
            .map(|v| v["method"].as_str().unwrap_or(""))
            .collect();
        assert!(methods.contains(&"inference_tools"));
        assert!(
            methods.contains(&"error_set"),
            "expected error_set in {methods:?}"
        );
        assert!(methods.contains(&"message_updated"));
        // Final agent_state must be "error", not "idle", so the
        // status badge shows the failure visually.
        let last_agent_state = frames
            .iter()
            .rev()
            .find(|v| v["method"].as_str() == Some("agent_state"))
            .expect("at least one agent_state frame");
        assert_eq!(
            last_agent_state["params"]["state"].as_str(),
            Some("error"),
            "final agent_state should be \"error\", got {last_agent_state}"
        );
    }

    /// aivo#14: once a turn settles on `error` (or parks on
    /// `running_tools` waiting on a stuck tool), there's no in-flight
    /// stream to absorb a `client_cancel`. The reader still flips the
    /// cancel flag, so the worker reaches the arm with the flag SET — and
    /// must drop the agent back to `idle` (clearing orphaned tool leases)
    /// so Esc isn't a no-op and the badge/spinner clears.
    #[tokio::test]
    async fn client_cancel_with_no_live_turn_resets_stuck_status_to_idle() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        // Simulate the stuck-Librarian state: a pending tool lease that
        // never returned, status parked on running_tools, and the reader
        // having flipped the flag the instant client_cancel landed.
        ws_state.pending_tool_uses.insert(new_tool_call_id());
        ws_state.cancel_flag.store(true, Ordering::SeqCst);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let frame = json!({"jsonrpc": "2.0", "method": "client_cancel", "params": {}}).to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        let frames: Vec<serde_json::Value> = {
            let mut v = Vec::new();
            while let Some(s) = rx.recv().await {
                v.push(serde_json::from_str(&s).unwrap());
            }
            v
        };
        // Exactly one agent_state frame, and it returns to idle.
        let agent_states: Vec<&str> = frames
            .iter()
            .filter(|v| v["method"].as_str() == Some("agent_state"))
            .map(|v| v["params"]["state"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(agent_states, vec!["idle"], "got {frames:?}");
        // Orphaned lease abandoned and flag reset so the next turn is clean.
        assert!(ws_state.pending_tool_uses.is_empty());
        assert!(!ws_state.cancel_flag.load(Ordering::SeqCst));
    }

    /// The mirror case: a `client_cancel` that an in-flight turn already
    /// absorbed (finish_turn_as_cancelled cleared the flag before the
    /// worker dequeued the frame) must NOT re-emit idle — doing so would
    /// race a freshly-started follow-up turn's `working` state back off.
    #[tokio::test]
    async fn client_cancel_already_handled_is_a_noop() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        // Flag already cleared => the live-turn path handled the cancel.
        ws_state.cancel_flag.store(false, Ordering::SeqCst);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let frame = json!({"jsonrpc": "2.0", "method": "client_cancel", "params": {}}).to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        let mut emitted_any = false;
        while let Some(_s) = rx.recv().await {
            emitted_any = true;
        }
        assert!(
            !emitted_any,
            "no frames when the cancel was already handled"
        );
    }

    /// An unrecognized `client_*` frame (amp's client protocol moved
    /// past what the bridge handles) surfaces one friendly error_set +
    /// error agent_state into amp's TUI, instead of vanishing into the
    /// silent fallthrough and leaving the outbox to retry forever.
    /// Guarded so a second unknown frame stays quiet.
    #[tokio::test]
    async fn unknown_client_method_emits_one_friendly_error() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let frame = json!({"method": "client_some_future_thing", "params": {}}).to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        // Second unknown frame must not raise another error bar.
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        drop(tx);

        let mut frames: Vec<serde_json::Value> = Vec::new();
        while let Some(s) = rx.recv().await {
            frames.push(serde_json::from_str(&s).unwrap());
        }
        let error_sets = frames
            .iter()
            .filter(|v| v["method"].as_str() == Some("error_set"))
            .count();
        assert_eq!(error_sets, 1, "exactly one error_set, got {frames:?}");
        assert!(
            frames[0]["params"]["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("client_some_future_thing"),
            "error message names the unhandled method"
        );
        assert!(
            frames
                .iter()
                .any(|v| v["method"].as_str() == Some("agent_state")
                    && v["params"]["state"].as_str() == Some("error")),
            "agent_state goes to error"
        );
    }

    /// A `client_*` method the bridge DOES handle (and unknown
    /// non-`client_` notifications) must not trip the protocol warning.
    #[tokio::test]
    async fn handled_and_nonclient_methods_do_not_warn() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        // An amp notification we intentionally drop — not client-originated.
        let noise = json!({"method": "executor_some_notification", "params": {}}).to_string();
        ws_followup_events(&state, &mut ws_state, &noise, &tx, None, "/test").await;
        drop(tx);

        let mut had_error = false;
        while let Some(s) = rx.recv().await {
            let v: serde_json::Value = serde_json::from_str(&s).unwrap();
            had_error |= v["method"].as_str() == Some("error_set");
        }
        assert!(!had_error, "non-client notification must not warn");
        assert!(!ws_state.protocol_warned);
    }

    /// Routine user actions (remove/steer a queued message, manual `$`
    /// bash, notification subscribe, executor spawn) send `client_*`
    /// `request()` frames the bridge has no state for. They must be
    /// silently accepted — NOT routed to the unrecognized-protocol arm,
    /// which would raise a bogus "update aivo" error bar and park the
    /// agent on `error` mid-session.
    #[tokio::test]
    async fn benign_client_requests_do_not_warn_or_error() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        for method in [
            "client_remove_queued_msg",
            "client_steer_queued_msg",
            "client_upsert_notification_subscription",
            "client_spawn_executor",
            "client_append_manual_bash_invocation",
        ] {
            let frame = json!({"id": "r1", "method": method, "params": {}}).to_string();
            ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        }
        drop(tx);

        let mut emitted = false;
        while let Some(_s) = rx.recv().await {
            emitted = true;
        }
        assert!(!emitted, "benign requests emit no follow-up events");
        assert!(
            !ws_state.protocol_warned,
            "benign requests must not trip the protocol warning"
        );
    }

    /// executor_tools_unregister prunes the named tools so the next
    /// agent_turn stops advertising a tool the executor can no longer
    /// run (e.g. after an MCP server disconnects).
    #[tokio::test]
    async fn executor_tools_unregister_prunes_named_tools() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        ws_state.tools = vec![
            json!({"name": "Bash", "description": "", "input_schema": {}}),
            json!({"name": "mcp__x__do", "description": "", "input_schema": {}}),
            json!({"name": "Read", "description": "", "input_schema": {}}),
        ];
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let frame = json!({
            "method": "executor_tools_unregister",
            "params": {"toolNames": ["mcp__x__do"]},
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;

        let names: Vec<&str> = ws_state
            .tools
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(
            names,
            vec!["Bash", "Read"],
            "only the named tool is dropped"
        );
    }

    /// neo fires read-state / error-dismiss notifications
    /// (`client_mark_message_read` auto-emits on message visibility);
    /// these are benign no-ops and must not trip the protocol warning.
    #[tokio::test]
    async fn benign_client_notifications_do_not_warn() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        for method in [
            "client_mark_message_read",
            "client_mark_message_unread",
            "client_dismiss_active_error",
        ] {
            let frame = json!({"method": method, "params": {"messageId": "M-x"}}).to_string();
            ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        }
        drop(tx);

        let mut had_error = false;
        while let Some(s) = rx.recv().await {
            let v: serde_json::Value = serde_json::from_str(&s).unwrap();
            had_error |= v["method"].as_str() == Some("error_set");
        }
        assert!(!had_error, "benign read-state notifications must not warn");
        assert!(!ws_state.protocol_warned);
    }

    #[tokio::test]
    async fn inference_tools_defaults_agent_mode_when_unset() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let ws_state = fake_ws_state();
        emit_inference_tools(&tx, None, "/test", &ws_state, "M-abc").await;
        drop(tx);
        let frame: serde_json::Value = serde_json::from_str(&rx.recv().await.unwrap()).unwrap();
        assert_eq!(frame["params"]["agentMode"].as_str(), Some("smart"));
        assert_eq!(
            frame["params"]["tools"].as_array().unwrap().len(),
            0,
            "no tools registered yet"
        );
    }

    #[tokio::test]
    async fn client_set_thread_title_overwrites_existing() {
        let state = fake_bridge_state();
        let mut ws_state = fake_ws_state();
        ws_state.title = Some("auto-derived".to_string());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_set_thread_title",
            "params": {"title": "Refactor amp bridge"},
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        assert_eq!(ws_state.title.as_deref(), Some("Refactor amp bridge"));

        // Empty title clears (defensive — unclear if amp ever sends this).
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "client_set_thread_title",
            "params": {"title": "   "},
        })
        .to_string();
        ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
        assert_eq!(ws_state.title, None);
    }

    /// status != "done" maps to is_error:true so the model sees the
    /// failure. Covers error / cancelled / rejected-by-user.
    #[tokio::test]
    async fn executor_tool_result_marks_non_done_status_as_error() {
        let state = fake_bridge_state();
        let (tx, mut _rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        for status in ["error", "cancelled", "rejected-by-user"] {
            let mut ws_state = fake_ws_state();
            ws_state.bootstrap_complete = true;
            let tu_a = new_tool_call_id();
            // Keep a sibling pending so the next-turn trigger doesn't fire.
            ws_state.pending_tool_uses.insert(tu_a.clone());
            ws_state.pending_tool_uses.insert(new_tool_call_id());

            let frame = json!({
                "jsonrpc": "2.0",
                "method": "executor_tool_result",
                "params": {
                    "toolCallId": tu_a,
                    "run": {"status": status},
                },
            })
            .to_string();
            ws_followup_events(&state, &mut ws_state, &frame, &tx, None, "/test").await;
            // Stored neo-shape, carrying the precise executor status.
            let r = &ws_state.tool_results[0];
            assert_eq!(r["run"]["status"].as_str(), Some(status), "status={status}");
            // Converting to the Anthropic shape maps non-done → is_error,
            // and the result was absent so the fallback names the status.
            let a = block_to_anthropic(r);
            assert_eq!(a["is_error"].as_bool(), Some(true), "status={status}");
            assert!(
                a["content"].as_str().unwrap_or("").contains(status),
                "status={status} content={:?}",
                a["content"]
            );
        }
    }
}
