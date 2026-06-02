//! Disk persistence for amp threads observed by the bridge.
//!
//! Real ampcode.com persists every `uploadThread` server-side, then serves
//! it back via `getThread` so `amp threads continue T-<id>` works across
//! invocations. Aivo's bridge stubs auth/threads locally — we have to do
//! the same job ourselves or amp's resume flow is dead.
//!
//! Layout: each thread is one JSON file at
//! `~/.config/aivo/amp-threads/T-<id>.json`. The body is the exact
//! `params.thread` payload amp uploaded — round-trips cleanly into
//! `getThread`'s `result.thread.data` slot.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use tokio::fs;

use aivo::services::system_env;

/// `~/.config/aivo/amp-threads/`. Falls back to a relative path if the
/// home directory can't be resolved (matches the trace-log fallback).
pub fn default_threads_dir() -> PathBuf {
    let home = system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".config").join("aivo").join("amp-threads")
}

/// `~/.cache/amp/logs/threads/` — where amp's CLI writes per-thread
/// debug logs (JSONL of WS frames + internal events). Even though neo
/// amp's actual thread CONTENT lives server-side on ampcode.com, these
/// log files contain enough metadata (titles, agentMode, user-msg
/// markers) for `aivo logs` to surface native sessions that the user
/// ran with `amp` directly (not via `aivo run amp`).
///
/// Found by mining amp's binary: `qM = Ym.join(rbT, "threads")` where
/// `rbT = Ym.join(xx, "logs")` and `xx` resolves to `~/.cache/amp`.
#[allow(dead_code)] // retained for a future `amp threads`/`logs` integration
pub fn default_native_amp_logs_dir() -> PathBuf {
    let home = system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("amp").join("logs").join("threads")
}

/// True if the payload was written by the neo-era bridge (post the
/// May-2026 amp upgrade): `v: 2`, string `created`, string
/// `messageId`s. Pre-neo uploads carried `v: 23` with numeric
/// `messageId` and a unix-ms `created`; `list_threads` filters those
/// out so `aivo logs` doesn't surface threads amp's neo TUI can't
/// render either. Defensive — checks at least two distinguishing
/// signals so a malformed neo payload still passes and a stray
/// schema match in old data doesn't slip through.
fn is_neo_thread(payload: &Value) -> bool {
    let v_is_two = payload.get("v").and_then(|v| v.as_u64()) == Some(2);
    let created_is_string = payload
        .get("created")
        .map(|c| c.is_string())
        .unwrap_or(false);
    v_is_two && created_is_string
}

/// `T-<ulid-with-dashes>` — anything else is rejected so a malicious
/// thread ID can't traverse out of the threads dir.
fn valid_thread_id(id: &str) -> bool {
    id.len() > 2
        && id.len() <= 64
        && id.starts_with("T-")
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Writes the thread payload to disk. Called from the bridge's
/// `uploadThread` handler; amp uploads the FULL thread on every turn,
/// so a plain overwrite is the right semantic.
pub async fn save_thread(dir: &Path, payload: &Value) -> Result<String> {
    let id = payload
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("uploadThread payload missing string `id`"))?
        .to_string();
    if !valid_thread_id(&id) {
        return Err(anyhow!("rejecting unsafe thread id: {id}"));
    }
    fs::create_dir_all(dir)
        .await
        .with_context(|| format!("creating threads dir {}", dir.display()))?;
    let path = dir.join(format!("{id}.json"));
    let body = serde_json::to_vec(payload)?;
    fs::write(&path, body)
        .await
        .with_context(|| format!("writing thread {}", path.display()))?;
    Ok(id)
}

/// Loads a previously-saved thread by ID, or `None` if missing/corrupt.
pub async fn load_thread(dir: &Path, id: &str) -> Option<Value> {
    if !valid_thread_id(id) {
        return None;
    }
    let path = dir.join(format!("{id}.json"));
    let bytes = fs::read(&path).await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Loads a thread plus its last `limit` messages, or `None` if missing.
/// Backs the bridge's `getThreadTail` RPC — neo's switch-thread picker
/// calls it per highlighted row to fill the preview pane. amp's consumer
/// builds `{...thread.data, messages}`, so we hand back the full thread
/// object and the message tail as siblings.
pub async fn load_thread_tail(dir: &Path, id: &str, limit: usize) -> Option<(Value, Vec<Value>)> {
    let payload = load_thread(dir, id).await?;
    let tail = payload
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|all| all[all.len().saturating_sub(limit)..].to_vec())
        .unwrap_or_default();
    Some((payload, tail))
}

/// Deletes a saved thread; silently ignores missing files (matches amp's
/// idempotent `deleteThread` semantics).
pub async fn delete_thread(dir: &Path, id: &str) {
    if !valid_thread_id(id) {
        return;
    }
    let _ = fs::remove_file(dir.join(format!("{id}.json"))).await;
}

/// Returns up to `limit` most recently modified threads as listThreads
/// summary objects. Shape mirrors what ampcode.com would return: each
/// item carries the fields amp's CLI displays (`id`, `title`, `created`,
/// `updatedAt`, `messageCount`, `creatorUserID`).
pub async fn list_threads(dir: &Path, limit: usize) -> Vec<Value> {
    if limit == 0 {
        return Vec::new();
    }
    let Ok(mut rd) = fs::read_dir(dir).await else {
        return Vec::new();
    };
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        entries.push((mtime, path));
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.0));

    let mut out = Vec::with_capacity(entries.len().min(limit));
    for (mtime, path) in entries.into_iter().take(limit) {
        let Ok(bytes) = fs::read(&path).await else {
            continue;
        };
        let Ok(payload) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        // Skip pre-neo amp uploads (v=23, numeric created, integer
        // messageIds). They were persisted back when the bridge
        // intercepted amp's `uploadThread` HTTP RPC; neo amp uses
        // WebSocket sync and never calls uploadThread, but the old
        // files still sit on disk. Different schema (integer
        // messageIds, different message envelope) so amp's neo TUI
        // can't render them in `amp threads list` either — they'd
        // just clutter `aivo logs` with "(amp thread, N messages)"
        // rows lacking titles.
        if !is_neo_thread(&payload) {
            continue;
        }
        let id = payload.get("id").and_then(|s| s.as_str()).unwrap_or("");
        if id.is_empty() {
            continue;
        }
        let title = payload.get("title").cloned().unwrap_or(Value::Null);
        let created = payload.get("created").cloned().unwrap_or(Value::Null);
        let agent_mode = payload
            .get("agentMode")
            .cloned()
            .unwrap_or_else(|| Value::String("smart".to_string()));
        let message_count = payload
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let updated_at = chrono::DateTime::<chrono::Utc>::from(mtime).to_rfc3339();
        // amp's CLI thread list reads `userLastInteractedAt` and feeds it
        // to `new Date(...).toISOString()`. Without this field amp crashes
        // with `RangeError: Invalid Date`. We mirror `updatedAt` since
        // we can't tell the two apart without amp's own activity tracking.
        out.push(json!({
            "id": id,
            "title": title,
            "agentMode": agent_mode,
            "created": created,
            "updatedAt": updated_at,
            "userLastInteractedAt": updated_at,
            "messageCount": message_count,
            "creatorUserID": "user_aivo_local",
            // Neo's switch picker hard-derefs `relationships.find` per row; omit it and the picker build crashes.
            "relationships": [],
        }));
    }
    out
}

/// Per-session metadata recovered from one native amp log file.
/// `updated_at` is the file mtime (amp appends on every WS frame).
/// `title` is `None` when amp's title-gen RPC hasn't completed yet.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NativeAmpThreadMeta {
    pub id: String,
    pub title: Option<String>,
    pub agent_mode: String,
    pub message_count: usize,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Parse one native amp log file into metadata. `None` when the path is
/// unreadable or the file stem isn't a valid `T-<id>`.
#[allow(dead_code)]
pub async fn read_native_amp_thread_meta(path: &Path) -> Option<NativeAmpThreadMeta> {
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)?;
    if !valid_thread_id(&id) {
        return None;
    }
    let meta = fs::metadata(path).await.ok()?;
    let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let text = fs::read_to_string(path).await.ok()?;
    let (title, agent_mode, message_count) = parse_native_amp_log_metadata(&text);
    Some(NativeAmpThreadMeta {
        id,
        title,
        agent_mode,
        message_count,
        updated_at: chrono::DateTime::<chrono::Utc>::from(mtime),
    })
}

/// Lists native amp sessions (run via `amp` directly, not `aivo run amp`)
/// by scanning amp's per-thread log files. Returns the same row shape
/// as `list_threads` so the caller can merge both.
///
/// Sessions are matched by filename (`T-<id>.log`). For each, the log
/// file is scanned line-by-line for two markers: a `"title"` field
/// (set by amp's title-gen RPC response — usually within the first
/// ~100 lines) and an `"agentMode"` field. `messageCount` is the
/// count of `[observer] onMessageAdded` log lines with `"role":"user"`.
///
/// Cheap fallbacks: missing title → `"(amp session, <N> messages)"`;
/// missing agentMode → `"smart"`. The intent is "show me native
/// sessions ran today, even if metadata extraction was lossy"
/// rather than perfect reconstruction.
#[allow(dead_code)]
pub async fn list_native_amp_thread_logs(dir: &Path, limit: usize) -> Vec<Value> {
    if limit == 0 {
        return Vec::new();
    }
    let Ok(mut rd) = fs::read_dir(dir).await else {
        return Vec::new();
    };
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("log") {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        entries.push((mtime, path));
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.0));

    let mut out = Vec::with_capacity(entries.len().min(limit));
    for (_mtime, path) in entries.into_iter().take(limit) {
        let Some(meta) = read_native_amp_thread_meta(&path).await else {
            continue;
        };
        let updated_at = meta.updated_at.to_rfc3339();
        let title = meta
            .title
            .clone()
            .unwrap_or_else(|| format!("(amp session, {} messages)", meta.message_count));
        out.push(json!({
            "id": meta.id,
            "title": title,
            "agentMode": meta.agent_mode,
            "created": Value::Null,
            "updatedAt": updated_at,
            "userLastInteractedAt": updated_at,
            "messageCount": meta.message_count,
            "creatorUserID": "amp_native",
            "source": "amp_native",
            // Required by neo's picker — see `list_threads`.
            "relationships": [],
        }));
    }
    out
}

/// Pulls (title, agentMode, messageCount) out of a thread log file.
/// Cheap line-oriented scan — no full JSON parsing of each line.
#[allow(dead_code)]
fn parse_native_amp_log_metadata(text: &str) -> (Option<String>, String, usize) {
    let mut title: Option<String> = None;
    let mut agent_mode: Option<String> = None;
    let mut message_count: usize = 0;
    for line in text.lines() {
        if title.is_none()
            && let Some(t) = extract_quoted_value(line, "\"title\":\"")
        {
            // amp's title can be empty until generation completes; only
            // accept non-empty.
            if !t.is_empty() {
                title = Some(t);
            }
        }
        if agent_mode.is_none()
            && let Some(m) = extract_quoted_value(line, "\"agentMode\":\"")
        {
            agent_mode = Some(m);
        }
        // Count messages: `[observer] onMessageAdded` with role:"user"
        // fires once per user turn (skips assistant + placeholders).
        if line.contains("onMessageAdded") && line.contains("\"role\":\"user\"") {
            message_count += 1;
        }
    }
    (
        title,
        agent_mode.unwrap_or_else(|| "smart".to_string()),
        message_count,
    )
}

/// Find `prefix` in `haystack` and return the substring between
/// `prefix` and the next unescaped `"`. Returns `None` if the prefix
/// isn't found or the closing quote is missing. Cheap; doesn't unescape
/// JSON sequences — fine for short display strings.
#[allow(dead_code)]
fn extract_quoted_value(haystack: &str, prefix: &str) -> Option<String> {
    let start = haystack.find(prefix)? + prefix.len();
    let rest = &haystack[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse amp's `created` field — a unix-ms integer (older threads) or an
/// RFC3339 string (neo) — into UTC. `None` when absent or unrecognized.
fn thread_created_utc(created: Option<&Value>) -> Option<chrono::DateTime<chrono::Utc>> {
    match created {
        Some(Value::Number(n)) => n.as_i64().and_then(chrono::DateTime::from_timestamp_millis),
        Some(Value::String(s)) => chrono::DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|d| d.with_timezone(&chrono::Utc)),
        _ => None,
    }
}

/// Accumulated token usage for one model across amp thread files.
#[derive(Debug, Clone, Default)]
pub struct ModelUsage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}

/// One thread's per-model usage plus its creation time. `created` is `None`
/// when the thread carries no recognizable `created` field (the host then can't
/// time-filter it). Only threads with ≥1 usage-bearing turn are emitted.
#[derive(Debug, Clone)]
pub struct ThreadUsage {
    pub created: Option<chrono::DateTime<chrono::Utc>>,
    pub by_model: std::collections::BTreeMap<String, ModelUsage>,
}

/// Walk a thread directory and return per-thread (session) usage — the raw data
/// the host needs to window + aggregate itself. Each assistant message carries a
/// `usage` object (`model` + the four token dimensions). Best-effort: any
/// unreadable/unparseable file is skipped. No filtering happens here — that's
/// the host's job.
pub async fn collect_thread_sessions(dir: &Path) -> Vec<ThreadUsage> {
    use std::collections::BTreeMap;
    let mut out = Vec::new();
    let Ok(mut rd) = fs::read_dir(dir).await else {
        return out;
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = fs::read(&path).await else {
            continue;
        };
        let Ok(thread) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        // amp's `created` is unix-ms (older threads) or an RFC3339 string (neo).
        let created = thread_created_utc(thread.get("created"));
        let Some(messages) = thread.get("messages").and_then(|m| m.as_array()) else {
            continue;
        };
        let mut by_model: BTreeMap<String, ModelUsage> = BTreeMap::new();
        for msg in messages {
            let Some(u) = msg.get("usage").filter(|u| u.is_object()) else {
                continue;
            };
            let model = u
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("(unknown)")
                .to_string();
            let n = |k: &str| u.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
            let e = by_model.entry(model).or_default();
            e.input += n("inputTokens");
            e.output += n("outputTokens");
            e.cache_read += n("cacheReadInputTokens");
            e.cache_write += n("cacheCreationInputTokens");
        }
        if !by_model.is_empty() {
            out.push(ThreadUsage { created, by_model });
        }
    }
    out
}

/// Pulls the thread ID out of a `getThread` / `deleteThread` request body.
pub fn extract_thread_id_from_request(body: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(body).ok()?;
    parsed
        .get("params")
        .and_then(|p| p.get("thread"))
        .and_then(|t| t.as_str())
        .map(str::to_string)
}

/// Pulls the thread payload out of an `uploadThread` request body.
pub fn extract_thread_payload_from_request(body: &str) -> Option<Value> {
    let parsed: Value = serde_json::from_str(body).ok()?;
    parsed
        .get("params")
        .and_then(|p| p.get("thread"))
        .filter(|v| v.is_object())
        .cloned()
}

/// Pulls `params.limit` out of an RPC request body, falling back to
/// `default` when absent or unparseable.
fn extract_param_limit(body: &str, default: usize) -> usize {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("params").and_then(|p| p.get("limit")).cloned())
        .and_then(|l| l.as_u64())
        .map(|n| n as usize)
        .unwrap_or(default)
}

/// `listThreads` limit; default 200 mirrors the amp CLI's own request.
pub fn extract_list_limit(body: &str) -> usize {
    extract_param_limit(body, 200)
}

/// `getThreadTail` limit. amp always sends an explicit value (76 observed
/// from neo's switch picker); the fallback only guards a malformed body.
pub fn extract_tail_limit(body: &str) -> usize {
    extract_param_limit(body, 100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn valid_thread_id_accepts_real_ulid_format() {
        assert!(valid_thread_id("T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0"));
    }

    #[test]
    fn valid_thread_id_rejects_path_traversal() {
        assert!(!valid_thread_id("T-../etc/passwd"));
        assert!(!valid_thread_id("T-/abs"));
        assert!(!valid_thread_id("../sneaky"));
    }

    #[test]
    fn valid_thread_id_rejects_empty_or_too_long() {
        assert!(!valid_thread_id(""));
        assert!(!valid_thread_id("T-"));
        assert!(valid_thread_id("T-a"));
        assert!(!valid_thread_id(&format!("T-{}", "a".repeat(80))));
    }

    #[tokio::test]
    async fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let payload = json!({
            "v": 1,
            "id": "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0",
            "created": 1778211465768u64,
            "messages": [],
            "agentMode": "smart",
        });
        let id = save_thread(dir.path(), &payload).await.unwrap();
        assert_eq!(id, "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0");
        let loaded = load_thread(dir.path(), &id).await.unwrap();
        assert_eq!(loaded, payload);
    }

    #[tokio::test]
    async fn save_rejects_payload_without_id() {
        let dir = tempfile::tempdir().unwrap();
        let payload = json!({"v": 1, "messages": []});
        assert!(save_thread(dir.path(), &payload).await.is_err());
    }

    #[tokio::test]
    async fn save_rejects_unsafe_id() {
        let dir = tempfile::tempdir().unwrap();
        let payload = json!({"id": "T-../etc/passwd"});
        assert!(save_thread(dir.path(), &payload).await.is_err());
    }

    #[tokio::test]
    async fn load_returns_none_for_missing_thread() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_thread(dir.path(), "T-does-not-exist").await.is_none());
    }

    #[tokio::test]
    async fn list_threads_orders_by_recency_and_caps() {
        let dir = tempfile::tempdir().unwrap();
        for (i, suffix) in ["aaa", "bbb", "ccc"].iter().enumerate() {
            let payload = json!({
                "id": format!("T-{suffix}"),
                "title": format!("title-{i}"),
                "v": 2,
                "created": format!("2026-05-28T00:00:0{i}Z"),
                "agentMode": "smart",
                "messages": [{"role": "user"}, {"role": "assistant"}],
            });
            save_thread(dir.path(), &payload).await.unwrap();
            // ensure mtime differs reliably across saves
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let listed = list_threads(dir.path(), 10).await;
        assert_eq!(listed.len(), 3);
        assert_eq!(listed[0]["id"], "T-ccc");
        assert_eq!(listed[2]["id"], "T-aaa");
        assert_eq!(listed[0]["messageCount"], 2);
        assert_eq!(listed[0]["title"], "title-2");
        assert_eq!(listed[0]["agentMode"], "smart");
        // The `userLastInteractedAt` mirror is what stops amp's CLI
        // listing renderer from crashing with `Invalid Date`.
        assert_eq!(listed[0]["updatedAt"], listed[0]["userLastInteractedAt"]);

        let listed_capped = list_threads(dir.path(), 1).await;
        assert_eq!(listed_capped.len(), 1);
        assert_eq!(listed_capped[0]["id"], "T-ccc");

        let none = list_threads(dir.path(), 0).await;
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn list_native_amp_thread_logs_extracts_title_and_counts_users() {
        let dir = tempfile::tempdir().unwrap();
        // Realistic log fragment based on what amp writes during a
        // session: connection-state events, message_added user msgs,
        // a title-gen response, agentMode fields. One unrelated event
        // that mentions "title" inside body text shouldn't trip the
        // extractor (it's looking for the JSON key form).
        let id = "T-019e6f4a-8f1f-7309-9aa5-d390080d1abf";
        let log = r#"{"@timestamp":"2026-05-28T15:53:40.379Z","level":"INFO","message":"[thread-client] Transport connection state changed","logger":"","threadId":"T-019e6f4a-8f1f-7309-9aa5-d390080d1abf","previousState":"connecting","nextState":"connected","previousRole":null,"nextRole":null,"clientId":null,"pid":99029}
{"@timestamp":"2026-05-28T15:53:40.500Z","level":"INFO","message":"[observer] onAgentState","data":{"agentMode":"smart","reasoningEffort":"medium"},"pid":99029}
{"@timestamp":"2026-05-28T15:53:50.000Z","level":"INFO","message":"[observer] onMessageAdded","data":{"role":"user","messageId":"M-aaaaaaaaaaaaaaaaaaaaaa"},"pid":99029}
{"@timestamp":"2026-05-28T15:53:55.000Z","level":"INFO","message":"[observer] onMessageAdded","data":{"role":"assistant","messageId":"M-bbbbbbbbbbbbbbbbbbbbbb"},"pid":99029}
{"@timestamp":"2026-05-28T15:54:30.000Z","level":"INFO","message":"[observer] onThreadTitle","data":{"title":"Export aivo keys to file"},"pid":99029}
{"@timestamp":"2026-05-28T15:54:40.000Z","level":"INFO","message":"[observer] onMessageAdded","data":{"role":"user","messageId":"M-cccccccccccccccccccccc"},"pid":99029}
"#;
        std::fs::write(dir.path().join(format!("{id}.log")), log).unwrap();
        // A file with an invalid id name should be ignored.
        std::fs::write(dir.path().join("not-a-thread.log"), "garbage").unwrap();
        // A non-log file in the dir should also be ignored.
        std::fs::write(dir.path().join("README"), "scratch").unwrap();

        let listed = list_native_amp_thread_logs(dir.path(), 10).await;
        assert_eq!(listed.len(), 1);
        let row = &listed[0];
        assert_eq!(row["id"], id);
        assert_eq!(row["title"], "Export aivo keys to file");
        assert_eq!(row["agentMode"], "smart");
        assert_eq!(row["messageCount"], 2); // two user messages
        assert_eq!(row["source"], "amp_native");
        assert!(row["updatedAt"].is_string());
    }

    #[tokio::test]
    async fn list_native_amp_thread_logs_falls_back_when_no_title() {
        let dir = tempfile::tempdir().unwrap();
        let id = "T-019e6f4a-8f1f-7309-9aa5-d390080d1ab2";
        // No title generated yet; one user message.
        let log = r#"{"message":"[observer] onMessageAdded","data":{"role":"user"}}
"#;
        std::fs::write(dir.path().join(format!("{id}.log")), log).unwrap();
        let listed = list_native_amp_thread_logs(dir.path(), 10).await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["title"], "(amp session, 1 messages)");
        assert_eq!(listed[0]["agentMode"], "smart"); // fallback
    }

    #[test]
    fn extract_quoted_value_handles_basic_cases() {
        assert_eq!(
            extract_quoted_value(r#"foo "title":"hello" bar"#, "\"title\":\""),
            Some("hello".to_string())
        );
        // No prefix found.
        assert_eq!(
            extract_quoted_value(r#"no title here"#, "\"title\":\""),
            None
        );
        // Prefix found but no closing quote.
        assert_eq!(
            extract_quoted_value(r#""title":"unterminated"#, "\"title\":\""),
            None
        );
    }

    /// Regression: pre-neo amp uploads (v=23, numeric `created`,
    /// integer `messageId`s) used to clutter `aivo logs` with rows
    /// that amp's neo TUI couldn't render either. They live in the
    /// same directory because the previous bridge intercepted amp's
    /// HTTP `uploadThread` RPC; the neo bridge writes its own format
    /// (`v: 2`, ISO `created`). list_threads now filters by schema.
    #[tokio::test]
    async fn list_threads_filters_out_pre_neo_uploads() {
        let dir = tempfile::tempdir().unwrap();

        // Pre-neo upload — what we used to capture from amp's
        // HTTP uploadThread. Now ignored.
        let pre_neo = json!({
            "id": "T-019e1842-b0f2-768e-9c42-04c85fb73c1f",
            "v": 23,
            "created": 1778213296920u64,
            "env": {},
            "nextMessageId": 0,
            "messages": [{
                "role": "user",
                "messageId": 0,
                "content": [{"type": "text", "text": "hi"}],
            }],
        });
        save_thread(dir.path(), &pre_neo).await.unwrap();

        // Neo-format thread written by the current bridge.
        let neo = json!({
            "id": "T-bb00e7ba-4a83-8c3f-86da-b1b772c111c5",
            "v": 2,
            "title": "neo session",
            "created": "2026-05-28T22:54:00Z",
            "usesDtw": false,
            "usesThreadActors": false,
            "messages": [],
        });
        save_thread(dir.path(), &neo).await.unwrap();

        let listed = list_threads(dir.path(), 10).await;
        assert_eq!(listed.len(), 1, "pre-neo upload must be filtered out");
        assert_eq!(listed[0]["id"], "T-bb00e7ba-4a83-8c3f-86da-b1b772c111c5");
        assert_eq!(
            listed[0]["relationships"],
            json!([]),
            "every listThreads row needs a `relationships` array or neo's picker build crashes"
        );
    }

    #[tokio::test]
    async fn collect_thread_sessions_merges_turns_and_carries_timestamps() {
        let dir = tempfile::tempdir().unwrap();
        // Thread with two opus turns (unix-ms created) — merged within the session.
        std::fs::write(
            dir.path().join("T-old.json"),
            json!({
                "id": "T-old", "v": 2, "created": 1_700_000_000_000i64,
                "messages": [
                    {"role": "user", "content": []},
                    {"role": "assistant", "content": [], "usage": {
                        "model": "deepseek-v4-flash", "inputTokens": 10, "outputTokens": 100,
                        "cacheReadInputTokens": 1000, "cacheCreationInputTokens": 50}},
                    {"role": "assistant", "content": [], "usage": {
                        "model": "deepseek-v4-flash", "inputTokens": 5, "outputTokens": 20,
                        "cacheReadInputTokens": 500, "cacheCreationInputTokens": 0}},
                ],
            })
            .to_string(),
        )
        .unwrap();
        // A thread with no usage-bearing turn is omitted entirely.
        std::fs::write(
            dir.path().join("T-empty.json"),
            json!({"id": "T-empty", "v": 2, "created": 1_700_000_000_001i64,
                   "messages": [{"role": "user", "content": []}]})
            .to_string(),
        )
        .unwrap();

        let sessions = collect_thread_sessions(dir.path()).await;
        assert_eq!(sessions.len(), 1, "empty thread is dropped");
        let s = &sessions[0];
        // created (unix-ms 1_700_000_000_000 → 2023-11-14T...) parsed.
        assert_eq!(s.created.unwrap().timestamp(), 1_700_000_000);
        let ds = &s.by_model["deepseek-v4-flash"];
        assert_eq!(
            (ds.input, ds.output, ds.cache_read, ds.cache_write),
            (15, 120, 1500, 50),
            "turns within the session are summed"
        );
    }

    #[tokio::test]
    async fn delete_thread_removes_file_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let payload = json!({"id": "T-aaa"});
        save_thread(dir.path(), &payload).await.unwrap();
        assert!(load_thread(dir.path(), "T-aaa").await.is_some());
        delete_thread(dir.path(), "T-aaa").await;
        assert!(load_thread(dir.path(), "T-aaa").await.is_none());
        // No-op the second time.
        delete_thread(dir.path(), "T-aaa").await;
    }

    #[test]
    fn extract_thread_id_from_request_parses_real_payload() {
        let body = r#"{"method":"getThread","params":{"thread":"T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0"}}"#;
        assert_eq!(
            extract_thread_id_from_request(body).as_deref(),
            Some("T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0"),
        );
    }

    #[test]
    fn extract_thread_id_returns_none_for_garbage() {
        assert!(extract_thread_id_from_request("not json").is_none());
        assert!(extract_thread_id_from_request("{}").is_none());
    }

    #[test]
    fn extract_thread_payload_returns_object_only() {
        let body = r#"{"method":"uploadThread","params":{"thread":{"id":"T-x","messages":[]},"createdOnServer":false}}"#;
        let v = extract_thread_payload_from_request(body).unwrap();
        assert_eq!(v["id"], "T-x");
    }

    #[test]
    fn extract_list_limit_uses_request_value_or_default() {
        assert_eq!(
            extract_list_limit(r#"{"params":{"limit":50,"usesThreadActors":false}}"#),
            50,
        );
        assert_eq!(extract_list_limit(r#"{"params":{}}"#), 200);
        assert_eq!(extract_list_limit("not json"), 200);
    }

    #[test]
    fn extract_tail_limit_uses_request_value_or_default() {
        // Real shape neo's switch picker sends.
        assert_eq!(
            extract_tail_limit(
                r#"{"method":"getThreadTail","params":{"thread":"T-x","limit":76}}"#
            ),
            76,
        );
        assert_eq!(extract_tail_limit(r#"{"params":{}}"#), 100);
        assert_eq!(extract_tail_limit("not json"), 100);
    }

    #[tokio::test]
    async fn load_thread_tail_returns_last_n_messages() {
        let dir = tempfile::tempdir().unwrap();
        let messages: Vec<Value> = (0..10)
            .map(|i| json!({"role": "user", "messageId": format!("M-{i}"), "content": []}))
            .collect();
        let payload = json!({
            "id": "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0",
            "v": 2,
            "title": "tail test",
            "created": "2026-05-29T00:00:00Z",
            "messages": messages,
        });
        save_thread(dir.path(), &payload).await.unwrap();
        let id = payload["id"].as_str().unwrap();

        let (full, tail) = load_thread_tail(dir.path(), id, 3).await.unwrap();
        assert_eq!(full["title"], "tail test");
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0]["messageId"], "M-7");
        assert_eq!(tail[2]["messageId"], "M-9");

        // Limit larger than history returns every message.
        let (_, all) = load_thread_tail(dir.path(), id, 100).await.unwrap();
        assert_eq!(all.len(), 10);
    }

    #[tokio::test]
    async fn load_thread_tail_missing_thread_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_thread_tail(dir.path(), "T-nope", 10).await.is_none());
    }
}
