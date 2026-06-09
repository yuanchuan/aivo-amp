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
        if let Some(summary) = thread_summary(&payload, mtime) {
            out.push(summary);
        }
    }
    out
}

/// Builds one listThreads / `threads/find` summary row from a thread
/// payload + its file mtime, or `None` when the payload isn't a
/// renderable neo thread.
///
/// Skips pre-neo amp uploads (v=23, numeric created, integer
/// messageIds). They were persisted back when the bridge intercepted
/// amp's `uploadThread` HTTP RPC; neo amp uses WebSocket sync and never
/// calls uploadThread, but the old files still sit on disk. Different
/// schema (integer messageIds, different message envelope) so amp's neo
/// TUI can't render them in `amp threads list` either — they'd just
/// clutter `aivo logs` with "(amp thread, N messages)" rows lacking
/// titles.
fn thread_summary(payload: &Value, mtime: std::time::SystemTime) -> Option<Value> {
    if !is_neo_thread(payload) {
        return None;
    }
    let id = payload.get("id").and_then(|s| s.as_str()).unwrap_or("");
    if id.is_empty() {
        return None;
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
    Some(json!({
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
    }))
}

/// Serves amp's Librarian tool + the `amp threads <query>` CLI, both of
/// which issue `GET /api/threads/find?q=<query>&limit=N&offset=M` and
/// parse `{threads:[...], hasMore:bool}`. Each row carries the same
/// fields as `list_threads` plus an optional `matchedSearchText`
/// snippet. Returns `(page, has_more)`.
///
/// The real ampcode.com backend does server-side full-text search with
/// boolean operators; locally we scan the on-disk neo threads
/// (most-recent first) and keep any whose title or message text
/// contains a query token (case-insensitive). `OR`/`AND` keywords are
/// dropped, so the Librarian's habitual `term OR misspelling` queries
/// match either spelling. An empty/all-operator query lists everything,
/// like the bare CLI.
pub async fn find_threads(
    dir: &Path,
    query: &str,
    limit: usize,
    offset: usize,
) -> (Vec<Value>, bool) {
    let tokens: Vec<String> = query
        .split_whitespace()
        .filter(|t| !t.eq_ignore_ascii_case("or") && !t.eq_ignore_ascii_case("and"))
        .map(str::to_ascii_lowercase)
        .filter(|t| !t.is_empty())
        .collect();

    let Ok(mut rd) = fs::read_dir(dir).await else {
        return (Vec::new(), false);
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

    // Collect every match first (cheap — title + text scan per file),
    // then page with offset/limit so `hasMore` is accurate.
    let mut matched: Vec<Value> = Vec::new();
    for (mtime, path) in entries {
        let Ok(bytes) = fs::read(&path).await else {
            continue;
        };
        let Ok(payload) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        let Some(mut summary) = thread_summary(&payload, mtime) else {
            continue;
        };
        let haystack = thread_search_text(&payload);
        let haystack_lc = haystack.to_lowercase();
        let hit = tokens.is_empty() || tokens.iter().any(|t| haystack_lc.contains(t));
        if !hit {
            continue;
        }
        if let Some(snippet) = matched_snippet(&haystack, &haystack_lc, &tokens) {
            summary["matchedSearchText"] = Value::String(snippet);
        }
        matched.push(summary);
    }

    let total = matched.len();
    let page: Vec<Value> = matched.into_iter().skip(offset).take(limit).collect();
    let has_more = offset.saturating_add(page.len()) < total;
    (page, has_more)
}

/// Concatenates a thread's title + every message's text content into one
/// searchable string. tool_use/tool_result and other non-text blocks are
/// skipped — the Librarian searches conversational text, not tool I/O.
fn thread_search_text(payload: &Value) -> String {
    let mut buf = String::new();
    if let Some(t) = payload.get("title").and_then(|v| v.as_str()) {
        buf.push_str(t);
        buf.push('\n');
    }
    let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) else {
        return buf;
    };
    for msg in messages {
        match msg.get("content") {
            Some(Value::String(s)) => {
                buf.push_str(s);
                buf.push('\n');
            }
            Some(Value::Array(blocks)) => {
                for b in blocks {
                    if b.get("type").and_then(|t| t.as_str()) == Some("text")
                        && let Some(t) = b.get("text").and_then(|v| v.as_str())
                    {
                        buf.push_str(t);
                        buf.push('\n');
                    }
                }
            }
            _ => {}
        }
    }
    buf
}

/// A short context window around the first matching token, for the
/// Librarian's `matchedSearchText`. Best-effort + cosmetic: the field is
/// optional (amp guards it with `?.`), so any miss just omits it.
fn matched_snippet(text: &str, text_lc: &str, tokens: &[String]) -> Option<String> {
    const WINDOW: usize = 80;
    let pos = tokens
        .iter()
        .filter_map(|t| text_lc.find(t.as_str()))
        .min()?;
    let start = floor_char_boundary(text, pos.saturating_sub(WINDOW));
    let end = ceil_char_boundary(text, (pos + WINDOW).min(text.len()));
    // Collapse whitespace so the snippet stays on one line.
    let mut snippet = text[start..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if snippet.is_empty() {
        return None;
    }
    if start > 0 {
        snippet.insert(0, '…');
    }
    if end < text.len() {
        snippet.push('…');
    }
    Some(snippet)
}

/// `str::floor_char_boundary` is still unstable, so snap manually: walk
/// back to the nearest UTF-8 boundary at or before `i`.
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Nearest UTF-8 boundary at or after `i`.
fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
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

    /// find_threads backs amp's Librarian + `amp threads <query>` CLI.
    /// It matches the query against title + message text (case-insensitive,
    /// `OR`/`AND` dropped), returns most-recent-first, and surfaces a
    /// `matchedSearchText` snippet so the Librarian sees why a thread hit.
    #[tokio::test]
    async fn find_threads_matches_title_and_body_case_insensitively() {
        let dir = tempfile::tempdir().unwrap();
        let mk = |id: &str, title: &str, text: &str| {
            json!({
                "id": id, "v": 2, "title": title,
                "created": "2026-05-28T00:00:00Z", "agentMode": "smart",
                "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
            })
        };
        for (i, t) in [
            mk("T-aaa", "Refactor parser", "nothing relevant here"),
            mk(
                "T-bbb",
                "Daily notes",
                "We discussed the Librarian tool design",
            ),
            mk("T-ccc", "LIBRARIAN deep dive", "more text"),
        ]
        .into_iter()
        .enumerate()
        {
            save_thread(dir.path(), &t).await.unwrap();
            let _ = i;
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        // `librarian OR libraian`: OR is dropped, either token matches.
        let (hits, has_more) = find_threads(dir.path(), "librarian OR libraian", 10, 0).await;
        let ids: Vec<&str> = hits.iter().map(|t| t["id"].as_str().unwrap()).collect();
        assert_eq!(
            ids,
            vec!["T-ccc", "T-bbb"],
            "title + body match, recency order"
        );
        assert!(!has_more);
        // Body match carries a snippet naming the term.
        let bbb = hits.iter().find(|t| t["id"] == "T-bbb").unwrap();
        assert!(
            bbb["matchedSearchText"]
                .as_str()
                .unwrap_or("")
                .to_lowercase()
                .contains("librarian"),
            "snippet should quote the matched text, got {bbb:?}"
        );
        // Shape the Librarian client reads.
        assert!(bbb["title"].is_string());
        assert_eq!(bbb["creatorUserID"], "user_aivo_local");
        assert!(bbb["updatedAt"].is_string());
    }

    #[tokio::test]
    async fn find_threads_paginates_with_offset_limit_and_has_more() {
        let dir = tempfile::tempdir().unwrap();
        for n in 0..5 {
            let t = json!({
                "id": format!("T-{n}{n}{n}"), "v": 2, "title": format!("topic {n}"),
                "created": "2026-05-28T00:00:00Z",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "shared keyword"}]}],
            });
            save_thread(dir.path(), &t).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        // First page of 2 of 5 → hasMore.
        let (page1, more1) = find_threads(dir.path(), "keyword", 2, 0).await;
        assert_eq!(page1.len(), 2);
        assert!(more1);
        // Last page exhausts the set → no more.
        let (page3, more3) = find_threads(dir.path(), "keyword", 2, 4).await;
        assert_eq!(page3.len(), 1);
        assert!(!more3);
    }

    #[tokio::test]
    async fn find_threads_empty_query_lists_all_and_skips_pre_neo() {
        let dir = tempfile::tempdir().unwrap();
        save_thread(
            dir.path(),
            &json!({"id": "T-neo", "v": 2, "title": "kept",
                    "created": "2026-05-28T00:00:00Z", "messages": []}),
        )
        .await
        .unwrap();
        // Pre-neo (v=23) must never surface, even on an empty query.
        save_thread(
            dir.path(),
            &json!({"id": "T-old", "v": 23, "created": 1778213296920u64, "messages": []}),
        )
        .await
        .unwrap();
        let (hits, _) = find_threads(dir.path(), "  ", 10, 0).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["id"], "T-neo");
    }

    /// A multibyte body must not panic the snippet windower (char-boundary
    /// snapping) and a non-matching query must return nothing.
    #[tokio::test]
    async fn find_threads_handles_unicode_body_and_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!("{}librarian{}", "日本語".repeat(40), "café ☕".repeat(40));
        save_thread(
            dir.path(),
            &json!({"id": "T-uni", "v": 2, "title": "unicode",
                    "created": "2026-05-28T00:00:00Z",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": body}]}]}),
        )
        .await
        .unwrap();
        let (hit, _) = find_threads(dir.path(), "librarian", 10, 0).await;
        assert_eq!(hit.len(), 1);
        assert!(hit[0]["matchedSearchText"].is_string());
        let (miss, more) = find_threads(dir.path(), "zzzznomatch", 10, 0).await;
        assert!(miss.is_empty());
        assert!(!more);
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
