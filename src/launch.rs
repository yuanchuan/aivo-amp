//! Amp launch orchestration, ported from aivo's `environment_injector::for_amp`
//! and `launch_runtime`'s amp glue. Builds the bridge + sub-routers, writes the
//! merged settings file, and spawns the `amp` child through the bridge.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;

use aivo::constants::{
    AIVO_STARTER_MODEL, AIVO_STARTER_REAL_URL, AIVO_STARTER_SENTINEL, PLACEHOLDER_LOOPBACK_URL,
};
use aivo::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
use aivo::services::codex_oauth::CODEX_OAUTH_SENTINEL;
use aivo::services::copilot_auth::CopilotTokenManager;
use aivo::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;
use aivo::services::provider_profile::provider_profile_for_key;
use aivo::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use aivo::services::route_cache::{PersistedRoute, RouteCache};
use aivo::services::session_store::{ApiKey, SessionStore};
use aivo::services::{
    AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig, CopilotRouter, CopilotRouterConfig,
    ResponsesToChatRouter, ResponsesToChatRouterConfig,
};

use crate::amp_bridge::{
    AmpBridge, AmpBridgeConfig, detect_native_amp_credentials, is_amp_native_endpoint,
};
use crate::mode_models::AmpModeModels;

// Route-persistence namespaces in `ApiKey.protocol_routes`. amp drives two
// translator channels with disjoint protocol families, so each gets its own
// namespace: sharing one would let an `anthropic` route (messages channel)
// seed the responses channel, and seeded slots start *confirmed* — a wrong
// protocol would stick instead of being re-probed.
const AMP_ROUTE_NS_MESSAGES: &str = "amp";
const AMP_ROUTE_NS_RESPONSES: &str = "amp-responses";

/// Orchestrates a full `aivo amp` launch: build env → start bridge + routers →
/// patch the child env → write the merged settings file → spawn `amp` → wait →
/// persist learned routes. Returns the amp child's exit code.
pub async fn run_amp(
    store: &SessionStore,
    key: &ApiKey,
    model: Option<&str>,
    modes: &AmpModeModels,
    amp_args: &[String],
) -> Result<i32> {
    let mut env = for_amp(key, model, modes);

    if env.contains_key("AIVO_USE_CURSOR_ROUTER") {
        anyhow::bail!(
            "amp over cursor-agent (ACP) keys isn't supported by the aivo-amp plugin yet — pick a key with a real base URL, or `copilot`/`ollama`"
        );
    }

    // Sub-router caches captured for post-session persistence, each tagged with
    // the `protocol_routes` namespace it should be merged into. Empty on the
    // native-amp-endpoint path (no bridge, nothing to learn).
    let mut route_caches: Vec<(&'static str, Arc<RouteCache>)> = Vec::new();
    if env.contains_key("AIVO_USE_AMP_BRIDGE") {
        // Real limits of the model amp is forced to (cascade per the plugin
        // protocol's `AIVO_MODEL_*` contract) — `for_amp` already resolved the
        // effective wire model, starter default included.
        let limits =
            crate::model_limits::resolve(key, env.get("AIVO_AMP_FORCE_MODEL").map(String::as_str))
                .await;
        // Seed each translator from the key's persisted amp routes so a repeat
        // launch skips the protocol probe — same as the native tools.
        let messages_seed = key.routes_for_tool(AMP_ROUTE_NS_MESSAGES);
        let responses_seed = key.routes_for_tool(AMP_ROUTE_NS_RESPONSES);
        let (port, caches) = start_amp_bridge(
            &mut env,
            messages_seed,
            responses_seed,
            store,
            &key.id,
            limits,
        )
        .await?;
        route_caches = caches;
        env.insert("AMP_URL".to_string(), format!("http://127.0.0.1:{port}"));
        // Real key never reaches Amp; the bridge holds it and forwards.
        env.insert("AMP_API_KEY".to_string(), "aivo-bridge".to_string());
        // amp's Rivet thread-actors WS client (the 2026-05-28 "neo" rebrand)
        // ignores AMP_URL when it points at localhost — override so the WS
        // upgrade reaches the bridge.
        env.insert(
            "RIVET_PUBLIC_ENDPOINT".to_string(),
            format!("http://127.0.0.1:{port}/actors"),
        );
        // A user's HTTP_PROXY must not intercept amp's localhost call to the bridge.
        ensure_loopback_no_proxy(&mut env);
    }

    // The bridge plumbs the merged settings file path here; pass it to amp as
    // `--settings-file <path>` (prepended), then drop the aivo-internal marker.
    let settings_file = env.remove("AIVO_AMP_SETTINGS_FILE");
    let mut args: Vec<String> = Vec::new();
    if let Some(sf) = settings_file.as_deref() {
        args.push("--settings-file".to_string());
        args.push(sf.to_string());
    }
    // Pin the initial agent mode through amp's own `--mode` flag (it controls
    // the model, system prompt, and tool selection); the bridge then picks the
    // same mode up from the first turn's `agentMode`. `--mode` is consumed by
    // our own CLI, so the resolved value is re-added here.
    if let Some(mode) = modes
        .initial_mode
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    {
        args.push("--mode".to_string());
        args.push(mode.to_string());
    }
    args.extend(amp_args.iter().cloned());

    // Run accounting — the `[run]` started/finished pair in `aivo logs` /
    // `aivo stats` — is the host's job now that the manifest declares
    // `type: "coding-agent"`: aivo wraps every coding-agent launch in that pair.
    // The plugin used to write its own pair here; doing both would double-log.
    let result = spawn_amp_and_wait(&args, env).await;

    if let Some(sf) = settings_file {
        let _ = std::fs::remove_file(sf);
    }

    // Persist the protocol routes confirmed this session (best-effort), so the
    // next launch reuses them. `dirty_routes` only returns confirmed routes.
    persist_amp_routes(store, &key.id, &route_caches).await;

    result
}

/// Merges each captured router's confirmed routes back into the key's amp
/// namespace, mirroring the host's `persist_runtime_discoveries`.
async fn persist_amp_routes(
    store: &SessionStore,
    key_id: &str,
    caches: &[(&'static str, Arc<RouteCache>)],
) {
    for (namespace, cache) in caches {
        let dirty = cache.dirty_routes();
        if dirty.is_empty() {
            continue;
        }
        if let Err(e) = store.merge_routes(key_id, namespace, &dirty).await {
            eprintln!("aivo: failed to persist amp routes ({namespace}): {e}");
        }
    }
}

/// Builds the amp-specific environment for `key`. Ported from
/// `environment_injector::for_amp`. Produces the `AIVO_AMP_*` scaffolding that
/// `start_amp_bridge` consumes, or `AMP_URL`/`AMP_API_KEY` directly for a
/// native amp endpoint.
pub fn for_amp(
    key: &ApiKey,
    model: Option<&str>,
    amp_modes: &AmpModeModels,
) -> HashMap<String, String> {
    // Cursor ACP path: chain amp → amp_bridge → cursor router. (run_amp bails
    // before launch since the cursor router isn't wired into the plugin yet.)
    if key.is_cursor_acp() {
        let mut env = HashMap::new();
        env.insert("AIVO_USE_CURSOR_ROUTER".to_string(), "1".to_string());
        env.insert(
            "AIVO_CURSOR_KEY_SECRET".to_string(),
            key.key.as_str().to_string(),
        );
        env.insert("AIVO_USE_AMP_BRIDGE".to_string(), "1".to_string());
        env.insert(
            "AIVO_AMP_UPSTREAM_BASE_URL".to_string(),
            PLACEHOLDER_LOOPBACK_URL.to_string(),
        );
        env.insert(
            "AIVO_AMP_UPSTREAM_KEY".to_string(),
            "aivo-cursor".to_string(),
        );
        env.insert("AMP_SKIP_UPDATE_CHECK".to_string(), "1".to_string());

        let internal_mode_model = amp_modes.to_internal_model_value();
        let suppress_user_force = internal_mode_model.is_some();
        let resolved_force_model = model
            .filter(|_| !suppress_user_force)
            .filter(|m| !m.trim().is_empty() && *m != "__default__");
        if let Some(m) = resolved_force_model {
            env.insert("AIVO_AMP_FORCE_MODEL".to_string(), m.to_string());
        }

        const BRIDGE_UNSUPPORTED_TOOLS: &[&str] = &["Task"];
        let mut disable_tools = amp_modes.disable_tools.clone();
        for tool in BRIDGE_UNSUPPORTED_TOOLS {
            if !disable_tools.iter().any(|t| t == tool) {
                disable_tools.push((*tool).to_string());
            }
        }
        env.insert(
            "AIVO_AMP_TOOLS_DISABLE".to_string(),
            disable_tools.join(","),
        );

        if let Some(modes_obj) = internal_mode_model {
            env.insert(
                "AIVO_AMP_INTERNAL_MODEL_JSON".to_string(),
                modes_obj.to_string(),
            );
        }
        return env;
    }

    let mut env = HashMap::new();
    if is_amp_native_endpoint(&key.base_url) {
        env.insert("AMP_URL".to_string(), key.base_url.clone());
        env.insert("AMP_API_KEY".to_string(), key.key.to_string());
    } else {
        env.insert("AIVO_USE_AMP_BRIDGE".to_string(), "1".to_string());
        // Resolve the aivo-starter sentinel to its real backing URL.
        let profile = provider_profile_for_key(key);
        let (upstream_url, upstream_key) = if profile.serve_flags.is_starter {
            (
                AIVO_STARTER_REAL_URL.to_string(),
                AIVO_STARTER_SENTINEL.to_string(),
            )
        } else {
            (key.base_url.clone(), key.key.to_string())
        };
        env.insert("AIVO_AMP_UPSTREAM_BASE_URL".to_string(), upstream_url);
        env.insert("AIVO_AMP_UPSTREAM_KEY".to_string(), upstream_key);
        if profile.serve_flags.is_starter {
            env.insert("AIVO_AMP_IS_STARTER".to_string(), "1".to_string());
        }

        // Force the model the user picked; suppress when per-mode overrides
        // route their own models. Starter defaults to aivo/starter only when
        // no per-mode override exists.
        let internal_mode_model = amp_modes.to_internal_model_value();
        let suppress_user_force = internal_mode_model.is_some();
        let resolved_force_model = model
            .filter(|_| !suppress_user_force)
            .filter(|m| !m.trim().is_empty() && *m != "__default__");
        if let Some(m) = resolved_force_model {
            env.insert("AIVO_AMP_FORCE_MODEL".to_string(), m.to_string());
        } else if profile.serve_flags.is_starter && !suppress_user_force {
            env.insert(
                "AIVO_AMP_FORCE_MODEL".to_string(),
                AIVO_STARTER_MODEL.to_string(),
            );
        }

        // Auto-disable bridge-unsupported tools (`Task` only; web_search /
        // read_web_page are kept and their descriptions rewritten instead).
        const BRIDGE_UNSUPPORTED_TOOLS: &[&str] = &["Task"];
        let mut disable_tools = amp_modes.disable_tools.clone();
        for tool in BRIDGE_UNSUPPORTED_TOOLS {
            if !disable_tools.iter().any(|t| t == tool) {
                disable_tools.push((*tool).to_string());
            }
        }
        env.insert(
            "AIVO_AMP_TOOLS_DISABLE".to_string(),
            disable_tools.join(","),
        );

        if let Some(modes_obj) = internal_mode_model {
            env.insert(
                "AIVO_AMP_INTERNAL_MODEL_JSON".to_string(),
                modes_obj.to_string(),
            );
        }

        // Privacy default: stub the management plane locally. Opt back in to
        // forwarding auth/threads/telemetry with `AIVO_AMP_PASSTHROUGH=1`.
        if std::env::var("AIVO_AMP_PASSTHROUGH").as_deref() == Ok("1")
            && let Some((amp_url, amp_token)) = detect_native_amp_credentials()
        {
            env.insert("AIVO_AMP_NATIVE_URL".to_string(), amp_url);
            env.insert("AIVO_AMP_NATIVE_KEY".to_string(), amp_token);
        }
        env.insert("AMP_SKIP_UPDATE_CHECK".to_string(), "1".to_string());
    }
    env
}

/// Spawns the Amp bridge on a random local port. Strips the `AIVO_AMP_*`
/// scaffolding env vars so they don't leak into the spawned amp child.
/// Ported from `launch_runtime::start_amp_bridge`, adapted to the current
/// router config/`start_background` shapes.
async fn start_amp_bridge(
    env: &mut HashMap<String, String>,
    messages_seed: BTreeMap<String, PersistedRoute>,
    responses_seed: BTreeMap<String, PersistedRoute>,
    store: &SessionStore,
    key_id: &str,
    limits: crate::model_limits::ModelLimits,
) -> Result<(u16, Vec<(&'static str, Arc<RouteCache>)>)> {
    sweep_stale_amp_settings_files();
    let mut route_caches: Vec<(&'static str, Arc<RouteCache>)> = Vec::new();

    let upstream_base_url = env
        .remove("AIVO_AMP_UPSTREAM_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_AMP_UPSTREAM_BASE_URL"))?;
    let upstream_api_key = env
        .remove("AIVO_AMP_UPSTREAM_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_AMP_UPSTREAM_KEY"))?;

    // OAuth sentinels need credential refreshers the bridge translators don't
    // wire up yet — bail loudly before spawning.
    if matches!(
        upstream_base_url.as_str(),
        CLAUDE_OAUTH_SENTINEL | CODEX_OAUTH_SENTINEL | GEMINI_OAUTH_SENTINEL
    ) {
        anyhow::bail!(
            "amp doesn't yet support `{upstream_base_url}` keys — pick a key with a real base URL, or `copilot`/`ollama`"
        );
    }

    // Resolve sentinel upstreams.
    let copilot_github_token = (upstream_base_url == "copilot").then(|| upstream_api_key.clone());
    let upstream_base_url = if upstream_base_url == "ollama" {
        "http://localhost:11434/v1".to_string()
    } else {
        upstream_base_url
    };
    let native_amp_url = env.remove("AIVO_AMP_NATIVE_URL");
    let native_amp_key = env.remove("AIVO_AMP_NATIVE_KEY");
    let force_model = env.remove("AIVO_AMP_FORCE_MODEL");
    // `_JSON` (object form from per-mode flags) wins over the bare string form.
    let internal_model: Option<serde_json::Value> = env
        .remove("AIVO_AMP_INTERNAL_MODEL_JSON")
        .and_then(|s| serde_json::from_str(&s).ok())
        .or_else(|| {
            env.remove("AIVO_AMP_INTERNAL_MODEL")
                .map(serde_json::Value::String)
        });
    // Believed-window snap (see `model_limits`): amp's context meter and
    // compaction budget come from its compiled-in catalog entry for the mode's
    // model, so when the real window is known, pin every mode to the catalog
    // entry that matches it. Only when the wire model is forced (the bridge
    // rewrites amp's model names regardless, so the believed name is purely
    // amp-internal) and the user set no per-mode override of their own.
    let default_internal_model: Option<serde_json::Value> =
        match (&internal_model, &force_model, limits.context) {
            (None, Some(_), Some(context)) => {
                let snapped =
                    crate::model_limits::snap_internal_model(context, limits.output).to_string();
                let modes = crate::mode_models::AMP_AGENT_MODES
                    .iter()
                    .map(|mode| (mode.to_string(), serde_json::Value::String(snapped.clone())));
                Some(serde_json::Value::Object(modes.collect()))
            }
            _ => None,
        };
    let tools_disable: Vec<String> = env
        .remove("AIVO_AMP_TOOLS_DISABLE")
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let is_starter = env.remove("AIVO_AMP_IS_STARTER").as_deref() == Some("1");
    env.remove("AIVO_USE_AMP_BRIDGE");

    // The bridge + sub-routers build reqwest clients in this process; their
    // localhost hops must bypass any ambient HTTP_PROXY.
    ensure_loopback_no_proxy_in_process_env();

    let (anthropic_translation_port, responses_translation_port, upstream_base_url) =
        if let Some(github_token) = copilot_github_token {
            // Copilot: a CopilotRouter (speaks /v1/messages natively) fills the
            // anthropic slot; a Copilot-mode ResponsesToChatRouter fills the
            // responses slot.
            let copilot_router = CopilotRouter::new(CopilotRouterConfig {
                github_token: github_token.clone(),
            });
            let (anthropic_port, anthropic_handle) = copilot_router.start_background().await?;
            tokio::spawn(async move {
                if let Ok(Err(e)) = anthropic_handle.await {
                    eprintln!("aivo: amp-bridge copilot router exited: {e}");
                }
            });

            let responses_router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
                target_base_url: String::new(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: limits.output,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            })
            .with_tool(AMP_ROUTE_NS_RESPONSES)
            .with_seed_routes(responses_seed.clone());
            let (responses_port, responses_cache, _learned, responses_handle) =
                responses_router.start_background().await?;
            route_caches.push((AMP_ROUTE_NS_RESPONSES, responses_cache));
            tokio::spawn(async move {
                if let Ok(Err(e)) = responses_handle.await {
                    eprintln!("aivo: amp-bridge copilot responses translator exited: {e}");
                }
            });

            (
                Some(anthropic_port),
                Some(responses_port),
                format!("http://127.0.0.1:{anthropic_port}"),
            )
        } else {
            // Non-Copilot: spawn an AnthropicToOpenAIRouter unless the upstream
            // is natively Anthropic, and a ResponsesToChatRouter unless it's
            // natively a Responses API host.
            let upstream_protocol = detect_provider_protocol(&upstream_base_url);
            let anthropic_port = if upstream_protocol == ProviderProtocol::Anthropic {
                None
            } else {
                let translator = AnthropicToOpenAIRouter::new(AnthropicToOpenAIRouterConfig {
                    target_base_url: upstream_base_url.clone(),
                    target_api_key: upstream_api_key.clone(),
                    seed_routes: messages_seed.clone(),
                    strip_cache_control: false,
                    model_prefix: None,
                    requires_reasoning_content: false,
                    // Clamp amp's requested max_tokens (it asks for its
                    // believed catalog entry's ceiling) to the real model's
                    // published output limit.
                    max_tokens_cap: limits.output,
                    is_starter,
                });
                let (port, cache, _learned, handle) = translator.start_background().await?;
                route_caches.push((AMP_ROUTE_NS_MESSAGES, cache));
                tokio::spawn(async move {
                    if let Ok(Err(e)) = handle.await {
                        eprintln!("aivo: amp-bridge anthropic translator exited: {e}");
                    }
                });
                Some(port)
            };

            let responses_port = if upstream_protocol == ProviderProtocol::ResponsesApi {
                None
            } else {
                let translator = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
                    target_base_url: upstream_base_url.clone(),
                    api_key: upstream_api_key.clone(),
                    target_protocol: upstream_protocol,
                    target_path_variant: None,
                    copilot_token_manager: None,
                    model_prefix: None,
                    requires_reasoning_content: false,
                    actual_model: None,
                    max_tokens_cap: limits.output,
                    responses_api_supported: Some(false),
                    is_starter,
                    aivo_prefix_models: Vec::new(),
                })
                .with_tool(AMP_ROUTE_NS_RESPONSES)
                .with_seed_routes(responses_seed.clone());
                let (port, cache, _learned, handle) = translator.start_background().await?;
                route_caches.push((AMP_ROUTE_NS_RESPONSES, cache));
                tokio::spawn(async move {
                    if let Ok(Err(e)) = handle.await {
                        eprintln!("aivo: amp-bridge responses translator exited: {e}");
                    }
                });
                Some(port)
            };

            (anthropic_port, responses_port, upstream_base_url)
        };

    // Only allocate a trace file when `--debug` is on.
    let trace_log_path = aivo::services::http_debug::is_debug_active().then(|| {
        let home = aivo::services::system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let now = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let pid = std::process::id();
        home.join(".config")
            .join("aivo")
            .join("logs")
            .join(format!("amp-trace-{now}-{pid}.jsonl"))
    });

    // Write a merged settings file only when there's something to override.
    if internal_model.is_some() || default_internal_model.is_some() || !tools_disable.is_empty() {
        let path = write_amp_settings_override(
            internal_model.as_ref(),
            default_internal_model.as_ref(),
            &tools_disable,
        )?;
        env.insert(
            "AIVO_AMP_SETTINGS_FILE".to_string(),
            path.to_string_lossy().into_owned(),
        );
    }

    let threads_dir = crate::amp_threads::default_threads_dir();
    let bridge = AmpBridge::new(AmpBridgeConfig {
        upstream_base_url,
        upstream_api_key,
        trace_log_path,
        native_amp_url,
        native_amp_key,
        anthropic_translation_port,
        responses_translation_port,
        force_model,
        threads_dir,
        usage_store: Some(store.clone()),
        usage_key_id: key_id.to_string(),
    });
    let (port, handle) = bridge.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: amp bridge exited unexpectedly: {e}");
        }
    });
    Ok((port, route_caches))
}

// ---- settings-file merge (ported from launch_runtime) -----------------------

fn sweep_stale_amp_settings_files() {
    let Some(home) = aivo::services::system_env::home_dir() else {
        return;
    };
    let dir = home.join(".config").join("aivo").join("cache");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let cutoff =
        std::time::SystemTime::now().checked_sub(std::time::Duration::from_secs(24 * 60 * 60));
    for entry in entries.flatten() {
        let path = entry.path();
        let is_amp_settings = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("amp-settings-") && n.ends_with(".json"));
        if !is_amp_settings {
            continue;
        }
        let too_old = match (
            cutoff,
            entry.metadata().ok().and_then(|m| m.modified().ok()),
        ) {
            (Some(cutoff_at), Some(mtime)) => mtime < cutoff_at,
            _ => false,
        };
        if too_old {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Generates a settings.json that mirrors amp's own discovery order — user +
/// workspace (trust-filtered) + aivo overrides + managed (corp policy wins).
fn write_amp_settings_override(
    internal_model: Option<&serde_json::Value>,
    default_internal_model: Option<&serde_json::Value>,
    tools_disable: &[String],
) -> Result<PathBuf> {
    let home = aivo::services::system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let user_settings_path = home.join(".config").join("amp").join("settings.json");
    let user_value = crate::amp_trust::read_amp_settings_file(&user_settings_path);

    let workspace_path = aivo::services::system_env::current_dir()
        .and_then(|cwd| crate::amp_trust::find_workspace_amp_settings(&cwd, Some(&home)));
    let workspace_value = match workspace_path.as_deref() {
        Some(p) => filter_workspace_settings(p, crate::amp_trust::read_amp_settings_file(p)),
        None => None,
    };

    let merged_existing = merge_amp_settings_layers(user_value, workspace_value);
    let with_aivo_overrides = build_amp_settings_override(
        merged_existing,
        internal_model,
        default_internal_model,
        tools_disable,
    );

    // Managed settings: corporate policy wins over everything.
    let managed_value = find_managed_amp_settings()
        .as_deref()
        .and_then(crate::amp_trust::read_amp_settings_file);
    let final_value = merge_amp_settings_layers(Some(with_aivo_overrides), managed_value)
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    let pid = std::process::id();
    let cache_dir = home.join(".config").join("aivo").join("cache");
    std::fs::create_dir_all(&cache_dir)?;
    let path = cache_dir.join(format!("amp-settings-{pid}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&final_value)?)?;
    Ok(path)
}

/// Drops unapproved workspace `amp.mcpServers` via the trust store; warns once.
fn filter_workspace_settings(
    workspace_path: &Path,
    settings: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    let mut value = settings?;
    let trust = crate::amp_trust::AmpTrustStore::load();
    let dropped =
        crate::amp_trust::filter_workspace_mcp_servers(workspace_path, &mut value, &trust);
    if !dropped.is_empty() {
        let count = dropped.len();
        let names = dropped.join(", ");
        eprintln!(
            "aivo: skipped {count} unapproved workspace MCP server(s) from {}: {names}",
            workspace_path.display()
        );
        eprintln!("       run `aivo amp trust` from this repo to approve");
    }
    Some(value)
}

fn find_managed_amp_settings() -> Option<PathBuf> {
    managed_settings_paths().into_iter().find(|p| p.is_file())
}

fn managed_settings_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    #[cfg(target_os = "macos")]
    {
        paths.push(PathBuf::from(
            "/Library/Application Support/ampcode/managed-settings.json",
        ));
    }
    #[cfg(target_os = "linux")]
    {
        paths.push(PathBuf::from("/etc/ampcode/managed-settings.json"));
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(prog_data) = std::env::var("ProgramData") {
            paths.push(
                PathBuf::from(prog_data)
                    .join("ampcode")
                    .join("managed-settings.json"),
            );
        }
    }
    paths
}

/// Shallow top-level-key replacement: each workspace key wins entirely over
/// the user's value.
fn merge_amp_settings_layers(
    user: Option<serde_json::Value>,
    workspace: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match (user, workspace) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(user_v), Some(ws_v)) => {
            let mut user_obj = match user_v {
                serde_json::Value::Object(m) => m,
                _ => serde_json::Map::new(),
            };
            if let serde_json::Value::Object(ws_obj) = ws_v {
                for (k, v) in ws_obj {
                    user_obj.insert(k, v);
                }
            }
            Some(serde_json::Value::Object(user_obj))
        }
    }
}

/// Adds aivo's `internal.model` + `tools.disable` (dual-written, prefixed and
/// bare) plus a small set of set-if-absent bridge-aligned defaults.
/// `internal_model` (from per-mode flags) always wins; `default_internal_model`
/// (the believed-window snap) is set-if-absent so a user's own
/// `internal.model` in settings.json stays authoritative.
fn build_amp_settings_override(
    existing: Option<serde_json::Value>,
    internal_model: Option<&serde_json::Value>,
    default_internal_model: Option<&serde_json::Value>,
    tools_disable: &[String],
) -> serde_json::Value {
    let mut value = match existing {
        Some(v) if v.is_object() => v,
        _ => serde_json::Value::Object(serde_json::Map::new()),
    };
    let Some(obj) = value.as_object_mut() else {
        return value;
    };
    // amp's binary reads `T["internal.model"]` directly; write both forms.
    let user_has_internal_model =
        obj.contains_key("amp.internal.model") || obj.contains_key("internal.model");
    let effective_internal_model =
        internal_model.or_else(|| default_internal_model.filter(|_| !user_has_internal_model));
    if let Some(model) = effective_internal_model {
        obj.insert("amp.internal.model".to_string(), model.clone());
        obj.insert("internal.model".to_string(), model.clone());
    }

    if !tools_disable.is_empty() {
        let read_existing = |key: &str| -> Vec<String> {
            obj.get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut merged = read_existing("amp.tools.disable");
        for entry in read_existing("tools.disable") {
            if !merged.iter().any(|e| e == &entry) {
                merged.push(entry);
            }
        }
        for tool in tools_disable {
            if !merged.iter().any(|existing| existing == tool) {
                merged.push(tool.clone());
            }
        }
        let arr =
            serde_json::Value::Array(merged.into_iter().map(serde_json::Value::String).collect());
        obj.insert("amp.tools.disable".to_string(), arr.clone());
        obj.insert("tools.disable".to_string(), arr);
    }

    for (key, default) in [
        ("amp.showCosts", serde_json::Value::Bool(false)),
        (
            "amp.git.commit.coauthor.enabled",
            serde_json::Value::Bool(false),
        ),
        (
            "amp.git.commit.ampThread.enabled",
            serde_json::Value::Bool(false),
        ),
        (
            "amp.updates.mode",
            serde_json::Value::String("disabled".to_string()),
        ),
        ("amp.notifications.enabled", serde_json::Value::Bool(false)),
        (
            "amp.network.timeout",
            serde_json::Value::Number(serde_json::Number::from(600)),
        ),
    ] {
        if !obj.contains_key(key) {
            obj.insert(key.to_string(), default);
        }
    }
    value
}

// ---- NO_PROXY loopback helpers (ported from launch_runtime) ------------------

const NO_PROXY_VAR_NAMES: &[&str] = &["NO_PROXY", "no_proxy"];
const LOOPBACK_HOSTS: &[&str] = &["localhost", "127.0.0.1", "::1"];

fn ensure_loopback_no_proxy(env: &mut HashMap<String, String>) {
    for var in NO_PROXY_VAR_NAMES {
        let existing = env.get(*var).cloned().unwrap_or_default();
        env.insert((*var).to_string(), merge_loopback_entries(&existing));
    }
}

/// Same, on the current process env. SAFETY: the runtime spawns the bridge
/// before any concurrent env access; this runs before reqwest clients are built.
fn ensure_loopback_no_proxy_in_process_env() {
    // SAFETY: see fn-level comment.
    unsafe {
        for var in NO_PROXY_VAR_NAMES {
            let existing = std::env::var(var).unwrap_or_default();
            std::env::set_var(var, merge_loopback_entries(&existing));
        }
    }
}

fn merge_loopback_entries(existing: &str) -> String {
    let mut entries: Vec<String> = existing
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    for host in LOOPBACK_HOSTS {
        if !entries.iter().any(|e| e.eq_ignore_ascii_case(host)) {
            entries.push((*host).to_string());
        }
    }
    entries.join(",")
}

// ---- amp child spawn --------------------------------------------------------

fn locate_amp_binary() -> Option<PathBuf> {
    use aivo::services::path_search::{collect_path_dirs, find_in_dirs};
    let mut dirs = Vec::new();
    if let Some(home) = aivo::services::system_env::home_dir() {
        dirs.push(home.join(".amp").join("bin"));
    }
    dirs.extend(collect_path_dirs());
    find_in_dirs("amp", &dirs)
}

/// Spawns `amp` with the prepared env overlaid on the inherited environment,
/// inherits stdio, forwards termination signals on unix, and returns its code.
async fn spawn_amp_and_wait(args: &[String], env: HashMap<String, String>) -> Result<i32> {
    use std::process::Stdio;

    let amp_bin = locate_amp_binary().ok_or_else(|| {
        anyhow::anyhow!(
            "`amp` not found on PATH or in ~/.amp/bin — install it from https://ampcode.com/"
        )
    })?;

    let mut cmd = tokio::process::Command::new(&amp_bin);
    cmd.args(args);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);

    let mut child = cmd.spawn()?;

    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let pid = child.id();
        let mut sigint = signal(SignalKind::interrupt())?;
        let mut sigterm = signal(SignalKind::terminate())?;
        loop {
            tokio::select! {
                status = child.wait() => {
                    return Ok(status?.code().unwrap_or(1));
                }
                _ = sigint.recv() => { forward_signal(pid, libc::SIGINT); }
                _ = sigterm.recv() => { forward_signal(pid, libc::SIGTERM); }
            }
        }
    }
    #[cfg(not(unix))]
    {
        let status = child.wait().await?;
        Ok(status.code().unwrap_or(1))
    }
}

#[cfg(unix)]
fn forward_signal(pid: Option<u32>, sig: libc::c_int) {
    if let Some(pid) = pid {
        // SAFETY: kill(2) with a known pid and signal number has no memory effects.
        unsafe {
            libc::kill(pid as libc::pid_t, sig);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn snap_default_fills_internal_model_when_absent() {
        let snap = json!({"smart": "anthropic:claude-opus-4-8"});
        let out = build_amp_settings_override(None, None, Some(&snap), &[]);
        assert_eq!(out["amp.internal.model"], snap);
        assert_eq!(out["internal.model"], snap);
    }

    #[test]
    fn snap_default_never_clobbers_user_internal_model() {
        for key in ["amp.internal.model", "internal.model"] {
            let existing = json!({ key: "openai:my-custom" });
            let snap = json!({"smart": "anthropic:claude-opus-4-8"});
            let out = build_amp_settings_override(Some(existing), None, Some(&snap), &[]);
            assert_eq!(out[key], json!("openai:my-custom"), "{key}");
            // The untouched spelling must not be introduced with the snap.
            let other = if key == "internal.model" {
                "amp.internal.model"
            } else {
                "internal.model"
            };
            assert!(out.get(other).is_none(), "{other}");
        }
    }

    #[test]
    fn per_mode_flags_win_over_snap_and_user_settings() {
        let existing = json!({"internal.model": "openai:my-custom"});
        let forced = json!({"smart": "openai:flagged"});
        let snap = json!({"smart": "anthropic:claude-opus-4-8"});
        let out = build_amp_settings_override(Some(existing), Some(&forced), Some(&snap), &[]);
        assert_eq!(out["amp.internal.model"], forced);
        assert_eq!(out["internal.model"], forced);
    }
}
