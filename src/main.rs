//! `aivo-amp` — Sourcegraph Amp support for aivo, shipped as a sibling-binary
//! plugin. The aivo host execs this for `aivo amp …` / `aivo run amp …`,
//! handing off `AIVO_CONFIG_DIR` (and `AIVO_DEBUG_LOG` under `--debug`). The
//! plugin self-resolves the key from that config dir and launches amp through
//! aivo's provider bridge.

mod amp_bridge;
mod amp_threads;
mod amp_trust;
mod cli;
mod commands;
mod launch;
mod mode_models;
mod usage;

use std::path::PathBuf;

use clap::Parser;

use aivo::errors::ExitCode;
use aivo::key_resolution::{KeyLookupMode, KeyResolution, resolve_key_override};
use aivo::services::key_compat::KeyCompatContext;
use aivo::services::session_store::SessionStore;
use aivo::style;

use crate::mode_models::AMP_AGENT_MODES;

// current_thread runtime: the ported `ensure_loopback_no_proxy_in_process_env`
// mutates process env, which is only sound on a single-threaded runtime.
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let code = run().await;
    std::process::exit(code);
}

async fn run() -> i32 {
    let argv: Vec<String> = std::env::args().collect();

    // Self-description for `aivo plugins` — see aivo's docs/PLUGIN-PROTOCOL.md.
    // Must run before `LaunchCli::parse`, whose trailing_var_arg would otherwise
    // swallow the flag into `amp_args`. Prints one JSON manifest and exits.
    if argv.get(1).map(String::as_str) == Some("--aivo-manifest") {
        print_manifest();
        return 0;
    }

    // Usage stats for `aivo stats --by amp` — amp keeps complete per-turn token
    // usage in its own native thread store, so the plugin reads + normalizes it
    // (the host doesn't know amp's format). See aivo's docs/PLUGIN-PROTOCOL.md.
    if argv.get(1).map(String::as_str) == Some("--aivo-stats") {
        print_stats().await;
        return 0;
    }

    // Management subcommand: `aivo amp trust …`.
    if argv.get(1).map(String::as_str) == Some("trust") {
        let args = cli::AmpArgs::parse();
        return commands::AmpCommand::new().execute(args).await.code();
    }

    // Launch path. `parse()` handles --help/--version and exits on error.
    let launch = cli::LaunchCli::parse();
    match launch_amp(launch).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{} {e}", style::red("Error:"));
            ExitCode::UserError.code()
        }
    }
}

/// Emit amp's plugin manifest (protocol v1). `type: "coding-agent"` opts amp
/// into host-side run accounting — aivo records each launch in `aivo stats` /
/// `aivo logs` (count, duration, exit), so the plugin no longer logs its own
/// `[run]` rows (see launch.rs). `requires` declares the wrapped `amp` binary so
/// `aivo plugins install` offers the same consent-gated install native tools
/// get. Capabilities honestly disclose the host power the plugin uses. amp is a
/// "fat" plugin: it links the aivo library and reads the encrypted key store
/// directly (`AIVO_CONFIG_DIR`), so its key access *is* `config-read` — it never
/// consumes the `raw-key`/`endpoint` env handoff (which is why neither is
/// declared). `config-read`/`config-write` read and persist to aivo's key store
/// (active/starter key, learned routes) and shared `logs.db`; `spawn` launches
/// the real `amp` binary.
fn print_manifest() {
    // Build the manifest as raw JSON rather than aivo's `PluginManifest` type:
    // that type is `pub(crate)` by design, since the protocol boundary is
    // language-agnostic JSON, not a shared Rust struct. `json!` (vs hand-rolled
    // string concat) keeps serde escaping the env-sourced fields for us.
    let manifest = serde_json::json!({
        "name": "amp",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "1",
        "description": "Sourcegraph Amp coding agent, backed by aivo's key store",
        "type": "coding-agent",
        "roles": ["subcommand"],
        // amp's own `--help` already lists -k/-m/--debug (with amp-specific
        // descriptions), so tell aivo to skip its duplicate help banner.
        "documents_aivo_flags": true,
        "capabilities": ["config-read", "config-write", "spawn"],
        "requires": [{ "bin": "amp", "install": "npm install -g @ampcode/cli" }],
        "homepage": env!("CARGO_PKG_REPOSITORY"),
    });
    println!("{manifest}");
}

/// Emit amp's usage as one `aivo.stats/v1` JSON object: one **timestamped
/// per-session** record per thread, read from the bridge's store
/// (`~/.config/aivo/amp-threads/`) — the runs amp made through aivo's loopback,
/// on aivo-managed keys/models (e.g. deepseek, aivo/starter), NOT amp's own
/// backend runs. The plugin only provides data; aivo applies `--since` filtering
/// and aggregation host-side.
async fn print_stats() {
    let dir = amp_threads::default_threads_dir();
    let sessions = amp_threads::collect_thread_sessions(&dir).await;

    let sessions_json: Vec<serde_json::Value> = sessions
        .iter()
        .map(|s| {
            let models: Vec<serde_json::Value> = s
                .by_model
                .iter()
                .map(|(name, u)| {
                    serde_json::json!({
                        "name": name,
                        "input_tokens": u.input,
                        "output_tokens": u.output,
                        "cache_read_tokens": u.cache_read,
                        "cache_write_tokens": u.cache_write,
                    })
                })
                .collect();
            let mut obj = serde_json::json!({ "models": models });
            if let Some(ts) = s.created {
                obj["ts"] = serde_json::json!(ts.to_rfc3339());
            }
            obj
        })
        .collect();

    let report = serde_json::json!({
        "schema": "aivo.stats/v1",
        "tool": "amp",
        "source": "aivo-routed amp threads (~/.config/aivo/amp-threads)",
        "sessions": sessions_json,
    });
    println!("{report}");
}

async fn launch_amp(launch: cli::LaunchCli) -> anyhow::Result<i32> {
    // Activate trace logging if requested via flag or host-provided env.
    if let Some(path) = debug_log_path(&launch) {
        let _ = aivo::services::http_debug::init(path).await;
    }

    // Opt-in management-plane passthrough. SAFETY: current_thread runtime, set
    // before any task/thread reads it.
    if launch.passthrough {
        unsafe { std::env::set_var("AIVO_AMP_PASSTHROUGH", "1") };
    }

    // Validate the initial mode up front (matches aivo's run.rs gate).
    let modes = launch.to_mode_models();
    if let Some(mode) = modes.initial_mode.as_deref().map(str::trim)
        && !AMP_AGENT_MODES.contains(&mode)
    {
        anyhow::bail!(
            "unknown --mode '{mode}'. Valid: {}",
            AMP_AGENT_MODES.join(", ")
        );
    }

    let store = session_store();
    // New users: provision + activate the starter key so `aivo amp` works
    // out of the box, mirroring the aivo host.
    if let Some((starter, is_new)) = store.ensure_starter_key().await
        && is_new
    {
        let _ = store.set_active_key(&starter.id).await;
    }

    // Resolve the key the same way `aivo run` does (explicit -k → last
    // selection → active → interactive picker).
    let key_flag = launch.key.as_deref();
    let resolution = resolve_key_override(
        &store,
        key_flag,
        KeyLookupMode::RequireActiveOrPrompt,
        KeyCompatContext::None,
    )
    .await?;
    let key = match resolution {
        KeyResolution::Selected(k) => k,
        KeyResolution::Cancelled => return Ok(ExitCode::Success.code()),
        KeyResolution::MissingAuth => {
            eprintln!(
                "{} No API key available. Add one with {}.",
                style::red("Error:"),
                style::cyan("aivo keys add")
            );
            return Ok(ExitCode::AuthError.code());
        }
    };

    // For a `type: "coding-agent"` plugin the host owns `-m`/`--model`: it strips
    // the flag from our argv (so it never reaches our CLI) and instead *persists*
    // the resolved model on the key (`set_chat_model`) before launching us — the
    // same way it persists the `-k` choice as the last selection. So when the
    // flag didn't arrive, recover it from the store for the key we just resolved,
    // mirroring how `key` itself came back via the persisted last-selection. A
    // direct `aivo-amp -m …` invocation (no host) still uses the parsed flag.
    // (The host only hands `AIVO_KEY_MODEL` to plugins holding the `endpoint`
    // cap, which amp doesn't — it reads the store directly, its `config-read`.)
    let model_owned = match launch.model.clone().filter(|m| !m.trim().is_empty()) {
        Some(m) => Some(m),
        None => store
            .get_chat_model(&key.id)
            .await
            .ok()
            .flatten()
            .filter(|m| !m.trim().is_empty()),
    };
    let model = model_owned.as_deref();
    launch::run_amp(&store, &key, model, &modes, &launch.amp_args).await
}

/// Resolves the session store from the host-provided config dir, falling back
/// to aivo's default location when invoked directly.
fn session_store() -> SessionStore {
    match std::env::var("AIVO_CONFIG_DIR") {
        Ok(dir) if !dir.is_empty() => {
            SessionStore::with_path(PathBuf::from(dir).join("config.json"))
        }
        _ => SessionStore::new(),
    }
}

/// Resolves the trace-log path: `--debug[=path]` flag wins, else the host's
/// `AIVO_DEBUG_LOG`, else none.
fn debug_log_path(launch: &cli::LaunchCli) -> Option<PathBuf> {
    if let Some(p) = launch.debug.as_deref() {
        return Some(if p.is_empty() {
            aivo::services::http_debug::default_log_path()
        } else {
            PathBuf::from(p)
        });
    }
    match std::env::var("AIVO_DEBUG_LOG") {
        Ok(p) if !p.is_empty() => Some(PathBuf::from(p)),
        _ => None,
    }
}
