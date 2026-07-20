//! Workspace MCP server trust store for the amp bridge.
//!
//! Direct `amp` gates workspace `.amp/settings.json` MCP servers behind
//! `amp mcp approve <name>` so a hostile checkout can't auto-run an MCP
//! server with the user's credentials. aivo's bridge merges workspace
//! settings directly into the temp file passed via `--settings-file`,
//! which bypasses amp's approval workflow — so we mirror it here.
//!
//! Approval is keyed by `(workspace_settings_absolute_path, server_name,
//! server_config_sha256)`. Re-approval is required when the server config
//! changes (e.g. package version bump, command swap) — that's the whole
//! point of the hash.
//!
//! Storage: `~/.config/aivo/amp-trust.json`. Format:
//!
//! ```json
//! {
//!   "version": 1,
//!   "approvals": {
//!     "/abs/path/to/.amp/settings.json": {
//!       "approved_at": "2026-05-08T12:34:56Z",
//!       "servers": { "<name>": "<sha256-hex>" }
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Walks up from `start` looking for the nearest `.amp/settings.json` (or
/// `.amp/settings.jsonc`), per amp's documented workspace-discovery order:
/// "searched upward to repo root". Returns the first hit. Stops at:
/// - the `.git` directory (inclusive — that level still counted)
/// - `$HOME` (inclusive — last directory checked when `home` is set)
/// - filesystem root
///
/// `home` is threaded explicitly so tests can pin the search ceiling
/// without leaning on the real environment.
pub fn find_workspace_amp_settings(start: &Path, home: Option<&Path>) -> Option<PathBuf> {
    let mut current = start;
    loop {
        for name in ["settings.json", "settings.jsonc"] {
            let candidate = current.join(".amp").join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        if current.join(".git").exists() {
            return None;
        }
        if let Some(h) = home
            && current == h
        {
            return None;
        }
        current = current.parent()?;
    }
}

/// Reads an amp settings file, accepting both strict JSON and JSONC (with
/// `//` and `/* */` comments). Returns `None` when the file is missing or
/// can't be parsed even after stripping comments — callers that need the
/// error surfaced should use `parse_amp_settings_file` instead.
pub fn read_amp_settings_file(path: &Path) -> Option<serde_json::Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
        return Some(v);
    }
    serde_json::from_str::<serde_json::Value>(&strip_jsonc_comments(&raw)).ok()
}

/// Like `read_amp_settings_file` but propagates I/O and parse errors so
/// the CLI can show them. The bridge prefers the silent `Option` form to
/// stay conservative on malformed workspace files.
pub fn parse_amp_settings_file(path: &Path) -> Result<serde_json::Value> {
    let raw = std::fs::read_to_string(path)?;
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
        return Ok(v);
    }
    Ok(serde_json::from_str(&strip_jsonc_comments(&raw))?)
}

/// Strips `//` line comments and `/* */` block comments from JSONC,
/// preserving anything that looks like a comment but lives inside a string
/// literal. Byte-level state machine: safe for UTF-8 because every
/// continuation byte has the high bit set and never collides with the ASCII
/// delimiters (`/`, `*`, `"`, `\`).
pub fn strip_jsonc_comments(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let mut in_string = false;
    let mut escape = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            out.push(b);
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push(b);
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'/' => {
                    // Line comment: skip until newline. Leave the newline
                    // in place so line/column counts in any downstream
                    // parse error remain meaningful.
                    i += 2;
                    while i < bytes.len() && bytes[i] != b'\n' {
                        i += 1;
                    }
                    continue;
                }
                b'*' => {
                    // Block comment: skip until closing `*/`. If the
                    // comment is unterminated, swallow the rest of the
                    // input — better than emitting a half-comment that
                    // breaks the JSON parse.
                    i += 2;
                    while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                        i += 1;
                    }
                    i = (i + 2).min(bytes.len());
                    continue;
                }
                _ => {}
            }
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

const TRUST_FILE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AmpTrustStore {
    pub version: u32,
    pub approvals: BTreeMap<String, WorkspaceApprovals>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceApprovals {
    pub approved_at: String,
    pub servers: BTreeMap<String, String>,
}

impl AmpTrustStore {
    /// Returns the canonical trust file path. `None` only when we can't
    /// locate `$HOME` — bridge will fall back to "no approvals" semantics
    /// (drop every workspace MCP server) which is the safe default.
    pub fn path() -> Option<PathBuf> {
        aivo::services::system_env::home_dir()
            .map(|h| h.join(".config").join("aivo").join("amp-trust.json"))
    }

    /// Loads the trust store. Returns an empty store on any read/parse
    /// failure so a corrupt file fails closed (no servers approved)
    /// rather than open.
    pub fn load() -> Self {
        let Some(path) = Self::path() else {
            return Self::default_v1();
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::default_v1();
        };
        serde_json::from_str(&raw).unwrap_or_else(|_| Self::default_v1())
    }

    fn default_v1() -> Self {
        Self {
            version: TRUST_FILE_VERSION,
            approvals: BTreeMap::new(),
        }
    }

    pub fn save(&self) -> Result<()> {
        let path =
            Self::path().ok_or_else(|| anyhow::anyhow!("cannot locate $HOME for trust file"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// True iff `(workspace, name, config)` has been approved with this
    /// exact config hash. Hash mismatch → not approved (caller should
    /// treat as a new server).
    pub fn is_approved(&self, workspace: &Path, name: &str, config: &serde_json::Value) -> bool {
        let key = canonical_workspace_key(workspace);
        let Some(ws) = self.approvals.get(&key) else {
            return false;
        };
        let Some(stored_hash) = ws.servers.get(name) else {
            return false;
        };
        stored_hash == &hash_server_config(config)
    }

    pub fn approve(&mut self, workspace: &Path, name: &str, config: &serde_json::Value) {
        let key = canonical_workspace_key(workspace);
        let entry = self.approvals.entry(key).or_default();
        entry.approved_at = Utc::now().to_rfc3339();
        entry
            .servers
            .insert(name.to_string(), hash_server_config(config));
    }

    pub fn revoke(&mut self, workspace: &Path, name: &str) -> bool {
        let key = canonical_workspace_key(workspace);
        let Some(ws) = self.approvals.get_mut(&key) else {
            return false;
        };
        let removed = ws.servers.remove(name).is_some();
        if ws.servers.is_empty() {
            self.approvals.remove(&key);
        }
        removed
    }

    pub fn approved_servers_for(&self, workspace: &Path) -> Vec<String> {
        let key = canonical_workspace_key(workspace);
        self.approvals
            .get(&key)
            .map(|w| w.servers.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// Canonical key for a workspace settings file: absolute, with symlinks
/// resolved when possible. Falls back to the raw path on canonicalize
/// failure (e.g. file removed mid-launch) so the lookup stays
/// deterministic even when filesystem state is shifting under us.
fn canonical_workspace_key(path: &Path) -> String {
    std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
}

/// SHA-256 of the canonical (sorted-key) JSON serialization of a server
/// config. Sorting makes the hash insensitive to JSON object key order
/// — the user formatting their settings.json with their editor of
/// choice shouldn't invalidate approvals.
pub fn hash_server_config(config: &serde_json::Value) -> String {
    let canonical = canonical_json(config);
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

/// Recursive sorted-key JSON serialization. We don't pull in an external
/// canonical-JSON crate for one function — serde_json's `Map` preserves
/// insertion order, so we walk the value tree and rebuild objects with
/// sorted keys before serializing.
fn canonical_json(value: &serde_json::Value) -> String {
    fn walk(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(m) => {
                let mut sorted: BTreeMap<String, serde_json::Value> = BTreeMap::new();
                for (k, v) in m {
                    sorted.insert(k.clone(), walk(v));
                }
                serde_json::Value::Object(sorted.into_iter().collect())
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(walk).collect())
            }
            other => other.clone(),
        }
    }
    serde_json::to_string(&walk(value)).unwrap_or_default()
}

/// Filters `amp.mcpServers` in `workspace_settings` to only include
/// servers approved in `trust` for `workspace_path`. Returns the names
/// of dropped servers so the caller can warn the user. Mutates the
/// settings value in place; if `amp.mcpServers` ends up empty it's
/// removed entirely so the merged settings file doesn't carry a
/// useless empty object.
pub fn filter_workspace_mcp_servers(
    workspace_path: &Path,
    workspace_settings: &mut serde_json::Value,
    trust: &AmpTrustStore,
) -> Vec<String> {
    let Some(obj) = workspace_settings.as_object_mut() else {
        return Vec::new();
    };
    let Some(mcp_value) = obj.get_mut("amp.mcpServers") else {
        return Vec::new();
    };
    let Some(mcp_obj) = mcp_value.as_object_mut() else {
        return Vec::new();
    };

    let mut dropped: Vec<String> = Vec::new();
    let names: Vec<String> = mcp_obj.keys().cloned().collect();
    for name in names {
        let approved = mcp_obj
            .get(&name)
            .map(|cfg| trust.is_approved(workspace_path, &name, cfg))
            .unwrap_or(false);
        if !approved {
            mcp_obj.remove(&name);
            dropped.push(name);
        }
    }
    if mcp_obj.is_empty() {
        obj.remove("amp.mcpServers");
    }
    dropped.sort();
    dropped
}

/// Reads `amp.mcpServers` from a workspace settings file as a flat list
/// of `(name, config)` pairs. Returns an empty vec when the key is
/// absent or shaped wrong. Used by `aivo amp trust` to enumerate
/// candidates for approval.
pub fn workspace_mcp_servers(settings: &serde_json::Value) -> Vec<(String, serde_json::Value)> {
    let Some(obj) = settings.get("amp.mcpServers").and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let mut out: Vec<(String, serde_json::Value)> =
        obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hash_is_stable_across_key_order() {
        let a = json!({"command": "npx", "args": ["-y", "pkg"], "env": {"K": "v"}});
        let b = json!({"env": {"K": "v"}, "args": ["-y", "pkg"], "command": "npx"});
        assert_eq!(hash_server_config(&a), hash_server_config(&b));
    }

    #[test]
    fn hash_changes_when_config_changes() {
        // A package version bump (or any config change) MUST invalidate
        // the prior approval — that's the whole reason we hash instead
        // of trusting by name.
        let a = json!({"command": "npx", "args": ["-y", "pkg@1"]});
        let b = json!({"command": "npx", "args": ["-y", "pkg@2"]});
        assert_ne!(hash_server_config(&a), hash_server_config(&b));
    }

    #[test]
    fn approve_then_is_approved_returns_true() {
        let mut trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let cfg = json!({"command": "npx", "args": []});
        assert!(!trust.is_approved(&workspace, "fs", &cfg));
        trust.approve(&workspace, "fs", &cfg);
        assert!(trust.is_approved(&workspace, "fs", &cfg));
    }

    #[test]
    fn approval_invalidated_when_config_changes() {
        // User approved server X with config A. Tomorrow workspace
        // bumps config to B (e.g. malicious upstream). Approval must
        // not carry over.
        let mut trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let cfg_a = json!({"command": "npx", "args": ["-y", "pkg@1"]});
        let cfg_b = json!({"command": "npx", "args": ["-y", "pkg@2"]});
        trust.approve(&workspace, "fs", &cfg_a);
        assert!(trust.is_approved(&workspace, "fs", &cfg_a));
        assert!(!trust.is_approved(&workspace, "fs", &cfg_b));
    }

    #[test]
    fn approval_scoped_per_workspace() {
        // Approving server "fs" in repo A doesn't grant trust to a
        // server with the same name and config in repo B — the path
        // is part of the trust key.
        let mut trust = AmpTrustStore::default_v1();
        let cfg = json!({"command": "npx"});
        let repo_a = std::path::PathBuf::from("/repo-a/.amp/settings.json");
        let repo_b = std::path::PathBuf::from("/repo-b/.amp/settings.json");
        trust.approve(&repo_a, "fs", &cfg);
        assert!(trust.is_approved(&repo_a, "fs", &cfg));
        assert!(!trust.is_approved(&repo_b, "fs", &cfg));
    }

    #[test]
    fn revoke_returns_true_only_when_entry_existed() {
        let mut trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let cfg = json!({"command": "npx"});
        assert!(!trust.revoke(&workspace, "fs"));
        trust.approve(&workspace, "fs", &cfg);
        assert!(trust.revoke(&workspace, "fs"));
        assert!(!trust.is_approved(&workspace, "fs", &cfg));
    }

    #[test]
    fn revoke_cleans_up_empty_workspace_entry() {
        // After revoking the last server in a workspace, the
        // workspace key itself goes away — keeps the trust file tidy.
        let mut trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let cfg = json!({"command": "npx"});
        trust.approve(&workspace, "fs", &cfg);
        trust.revoke(&workspace, "fs");
        assert!(trust.approvals.is_empty());
    }

    #[test]
    fn filter_drops_unapproved_servers_and_returns_their_names() {
        let trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let mut settings = json!({
            "amp.mcpServers": {
                "filesystem": {"command": "npx"},
                "github": {"command": "npx"},
            },
            "amp.showCosts": true,
        });
        let dropped = filter_workspace_mcp_servers(&workspace, &mut settings, &trust);
        assert_eq!(dropped, vec!["filesystem", "github"]);
        // Unrelated keys survive verbatim.
        assert_eq!(settings["amp.showCosts"], true);
        // Empty map removed entirely so the merged file isn't carrying
        // a meaningless `"amp.mcpServers": {}`.
        assert!(settings.get("amp.mcpServers").is_none());
    }

    #[test]
    fn filter_keeps_approved_servers() {
        let mut trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let fs_cfg = json!({"command": "npx", "args": ["-y", "fs-mcp"]});
        let gh_cfg = json!({"command": "npx", "args": ["-y", "gh-mcp"]});
        trust.approve(&workspace, "filesystem", &fs_cfg);
        // Note: github NOT approved — should drop only that one.

        let mut settings = json!({
            "amp.mcpServers": {
                "filesystem": fs_cfg,
                "github": gh_cfg,
            }
        });
        let dropped = filter_workspace_mcp_servers(&workspace, &mut settings, &trust);
        assert_eq!(dropped, vec!["github"]);
        assert!(settings["amp.mcpServers"]["filesystem"].is_object());
        assert!(settings["amp.mcpServers"].get("github").is_none());
    }

    #[test]
    fn filter_is_noop_when_no_mcp_key() {
        let trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let mut settings = json!({"amp.showCosts": true});
        let dropped = filter_workspace_mcp_servers(&workspace, &mut settings, &trust);
        assert!(dropped.is_empty());
        assert_eq!(settings["amp.showCosts"], true);
    }

    #[test]
    fn workspace_mcp_servers_returns_sorted_pairs() {
        // Stable order so the trust UI lists servers consistently
        // regardless of how the user arranged them in the JSON.
        let settings = json!({
            "amp.mcpServers": {
                "zebra": {"command": "z"},
                "alpha": {"command": "a"},
            }
        });
        let pairs = workspace_mcp_servers(&settings);
        assert_eq!(pairs[0].0, "alpha");
        assert_eq!(pairs[1].0, "zebra");
    }

    #[test]
    fn workspace_mcp_servers_returns_empty_when_absent_or_misshaped() {
        assert!(workspace_mcp_servers(&json!({})).is_empty());
        assert!(workspace_mcp_servers(&json!({"amp.mcpServers": "not an object"})).is_empty());
    }

    #[test]
    fn find_workspace_amp_settings_finds_settings_json_in_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let amp_dir = tmp.path().join(".amp");
        std::fs::create_dir_all(&amp_dir).unwrap();
        let settings = amp_dir.join("settings.json");
        std::fs::write(&settings, "{}").unwrap();

        let found = find_workspace_amp_settings(tmp.path(), None);
        assert_eq!(found.as_deref(), Some(settings.as_path()));
    }

    #[test]
    fn find_workspace_amp_settings_walks_up_to_repo_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        let amp_dir = tmp.path().join(".amp");
        std::fs::create_dir_all(&amp_dir).unwrap();
        let settings = amp_dir.join("settings.json");
        std::fs::write(&settings, "{}").unwrap();

        let nested = tmp.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).unwrap();

        let found = find_workspace_amp_settings(&nested, None);
        assert_eq!(found.as_deref(), Some(settings.as_path()));
    }

    #[test]
    fn find_workspace_amp_settings_stops_at_git_boundary() {
        // `.amp/settings.json` lives ABOVE the repo root. Walking should
        // halt once it sees `.git` and not leak into the parent project's
        // settings — that file is for a different repo.
        let tmp = tempfile::tempdir().unwrap();
        let outer_amp = tmp.path().join(".amp");
        std::fs::create_dir_all(&outer_amp).unwrap();
        std::fs::write(outer_amp.join("settings.json"), "{}").unwrap();

        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        let found = find_workspace_amp_settings(&repo, None);
        assert!(found.is_none());
    }

    #[test]
    fn find_workspace_amp_settings_stops_at_home_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let outer_amp = tmp.path().join(".amp");
        std::fs::create_dir_all(&outer_amp).unwrap();
        std::fs::write(outer_amp.join("settings.json"), "{}").unwrap();

        let home = tmp.path().join("user");
        std::fs::create_dir_all(&home).unwrap();
        let nested = home.join("project");
        std::fs::create_dir_all(&nested).unwrap();

        let found = find_workspace_amp_settings(&nested, Some(&home));
        assert!(found.is_none());
    }

    #[test]
    fn find_workspace_amp_settings_prefers_json_over_jsonc() {
        let tmp = tempfile::tempdir().unwrap();
        let amp_dir = tmp.path().join(".amp");
        std::fs::create_dir_all(&amp_dir).unwrap();
        let json = amp_dir.join("settings.json");
        let jsonc = amp_dir.join("settings.jsonc");
        std::fs::write(&json, "{}").unwrap();
        std::fs::write(&jsonc, "{}").unwrap();

        let found = find_workspace_amp_settings(tmp.path(), None);
        assert_eq!(found.as_deref(), Some(json.as_path()));
    }

    #[test]
    fn find_workspace_amp_settings_falls_back_to_jsonc() {
        let tmp = tempfile::tempdir().unwrap();
        let amp_dir = tmp.path().join(".amp");
        std::fs::create_dir_all(&amp_dir).unwrap();
        let jsonc = amp_dir.join("settings.jsonc");
        std::fs::write(&jsonc, "{}").unwrap();

        let found = find_workspace_amp_settings(tmp.path(), None);
        assert_eq!(found.as_deref(), Some(jsonc.as_path()));
    }

    #[test]
    fn find_workspace_amp_settings_returns_none_when_nothing_present() {
        let tmp = tempfile::tempdir().unwrap();
        let found = find_workspace_amp_settings(tmp.path(), Some(tmp.path()));
        assert!(found.is_none());
    }

    #[test]
    fn strip_jsonc_comments_removes_line_comments() {
        let input = "{\n  // a setting\n  \"k\": 1 // trailing\n}";
        let out = strip_jsonc_comments(input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["k"], 1);
    }

    #[test]
    fn strip_jsonc_comments_removes_block_comments() {
        let input = "{ /* block */ \"k\": /* mid */ 2 }";
        let out = strip_jsonc_comments(input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["k"], 2);
    }

    #[test]
    fn strip_jsonc_comments_preserves_comment_like_strings() {
        // Inside a string, `//` and `/*` aren't comments — stripping them
        // would silently corrupt MCP server URLs that contain `//`.
        let input = r#"{"url": "https://example.com/path", "note": "/* not a comment */"}"#;
        let out = strip_jsonc_comments(input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["url"], "https://example.com/path");
        assert_eq!(v["note"], "/* not a comment */");
    }

    #[test]
    fn strip_jsonc_comments_handles_escaped_quotes_in_strings() {
        // Backslash-escaped quote inside a string must NOT terminate the
        // string — otherwise the next `//` outside the perceived string
        // gets eaten and breaks the JSON.
        let input = r#"{"k": "a\"b // still in string"}"#;
        let out = strip_jsonc_comments(input);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["k"], r#"a"b // still in string"#);
    }

    #[test]
    fn read_amp_settings_file_falls_back_to_jsonc() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("settings.jsonc");
        std::fs::write(&path, "{\n  // hi\n  \"amp.showCosts\": true\n}").unwrap();
        let v = read_amp_settings_file(&path).unwrap();
        assert_eq!(v["amp.showCosts"], true);
    }

    #[test]
    fn read_amp_settings_file_returns_none_for_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("does-not-exist.json");
        assert!(read_amp_settings_file(&path).is_none());
    }

    #[test]
    fn read_amp_settings_file_returns_none_for_unparseable() {
        // Garbage even after stripping comments → bail rather than crash
        // the bridge or write nonsense into the override.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("broken.json");
        std::fs::write(&path, "this is not json {{{ }}}").unwrap();
        assert!(read_amp_settings_file(&path).is_none());
    }

    #[test]
    fn parse_amp_settings_file_handles_jsonc_with_comments() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("settings.jsonc");
        std::fs::write(&path, "{\n  // approval test\n  \"amp.mcpServers\": {}\n}").unwrap();
        let v = parse_amp_settings_file(&path).unwrap();
        assert!(v["amp.mcpServers"].is_object());
    }

    #[test]
    fn save_then_load_roundtrip() {
        // Sanity check the on-disk format. Use a tempdir as $HOME to
        // avoid touching the user's real trust file.
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: `set_var` is called before any thread interacts with
        // HOME; this test is single-threaded by serial-test conventions.
        unsafe {
            std::env::set_var("HOME", tmp.path());
        }
        let mut trust = AmpTrustStore::default_v1();
        let workspace = std::path::PathBuf::from("/repo/.amp/settings.json");
        let cfg = json!({"command": "npx", "args": ["-y", "fs-mcp"]});
        trust.approve(&workspace, "fs", &cfg);
        trust.save().unwrap();

        let reloaded = AmpTrustStore::load();
        assert!(reloaded.is_approved(&workspace, "fs", &cfg));
    }
}
