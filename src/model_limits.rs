//! Real context/output limits for the launched model, applied to amp.
//!
//! aivo's plugin protocol (docs/plugin-protocol.md → Endpoint handoff) hands
//! `endpoint`-granted plugins two advisory env vars — `AIVO_MODEL_CONTEXT_WINDOW`
//! and `AIVO_MODEL_MAX_OUTPUT_TOKENS` — so the wrapped CLI doesn't assume a
//! default window for unknown models. amp is a fat plugin (no `endpoint` cap),
//! so alongside honoring those vars when a host sets them, it self-resolves
//! through the host's own canonical cascade (`model_metadata::resolve_limits`:
//! live models-cache → embedded models.dev snapshot). Absent limits mean
//! "unknown" — amp keeps its own defaults, per the protocol.
//!
//! The limits feed two amp-side knobs (see `launch`):
//! - `output` → `max_tokens_cap` on both bridge translators, clamping amp's
//!   requested `max_tokens` to what the model actually accepts;
//! - `context` → a believed-model snap: amp's context meter, compaction budget,
//!   and requested `max_tokens` all come from its *compiled-in* catalog entry
//!   for the mode's model (it has no context-window setting), so we pick the
//!   catalog entry whose window best matches the real model and pin it via
//!   `internal.model` — the same trick the bridge docs describe doing by hand.

use aivo::services::model_metadata::resolve_limits;
use aivo::services::model_names::strip_context_suffix;
use aivo::services::models_cache::ModelsCache;
use aivo::services::session_store::ApiKey;

/// Resolved limits for the model amp is forced to on the wire. `None` fields
/// are unknown, not zero — leave amp's defaults alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelLimits {
    pub context: Option<u64>,
    pub output: Option<u64>,
}

/// Resolves limits for `model` on `key`. Per-field cascade, first hit wins:
/// host env (`AIVO_MODEL_*`, the protocol handoff — also a manual override
/// channel) → the host's canonical cascade (`model_metadata::resolve_limits`:
/// models-cache metadata under the key's full-catalog then picker namespace,
/// falling back to the embedded models.dev snapshot).
pub async fn resolve(key: &ApiKey, model: Option<&str>) -> ModelLimits {
    let mut limits = ModelLimits {
        context: env_limit("AIVO_MODEL_CONTEXT_WINDOW"),
        output: env_limit("AIVO_MODEL_MAX_OUTPUT_TOKENS"),
    };
    let Some(model) = model.map(strip_context_suffix).filter(|m| !m.is_empty()) else {
        return limits;
    };
    if limits.context.is_some() && limits.output.is_some() {
        return limits;
    }

    let cache = ModelsCache::new();
    let resolved = resolve_limits(&cache, Some(&key.base_url), model).await;
    limits.context = limits.context.or(resolved.context);
    limits.output = limits.output.or(resolved.output);
    limits
}

fn env_limit(var: &str) -> Option<u64> {
    std::env::var(var).ok()?.trim().parse().ok()
}

/// Picks the amp catalog entry whose believed window best matches the real
/// `context` (floor snap — amp must compact no later than reality allows).
/// Anthropic-family entries only, so amp keeps speaking `/v1/messages` to the
/// bridge's well-tuned anthropic channel regardless of the snap. Windows and
/// outputs are amp's compiled-in values (`q3` catalog in the amp binary):
///
/// - ≥ 1M  → `claude-sonnet-4-6`            (1M / 64k)
/// - ≥ 332k → `claude-opus-4-8`             (332k / 32k)
/// - else  → 200k tier, split by real output: `claude-haiku-4-5-…` (200k / 64k)
///   or `claude-opus-4-5-…` (200k / 32k). Below 200k amp's catalog has nothing
///   smaller — 200k is the closest believable window (the wire `max_tokens_cap`
///   still clamps output exactly).
///
/// If a future amp drops one of these ids, its catalog lookup falls back to
/// the per-provider default (sonnet-class) — today's behavior, not an error.
pub fn snap_internal_model(context: u64, output: Option<u64>) -> &'static str {
    if context >= 1_000_000 {
        "anthropic:claude-sonnet-4-6"
    } else if context >= 332_000 {
        "anthropic:claude-opus-4-8"
    } else if output.is_some_and(|o| o >= 64_000) {
        "anthropic:claude-haiku-4-5-20251001"
    } else {
        "anthropic:claude-opus-4-5-20251101"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_model_resolves_from_embedded_snapshot() {
        // The snapshot leg of the host cascade — no cache file, no env, no
        // network. Pins that the resolve_limits port still yields real limits.
        let key = ApiKey::new_with_protocol(
            "id".into(),
            "n".into(),
            "https://api.example.com/v1".into(),
            None,
            "secret".into(),
        );
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let limits = rt.block_on(resolve(&key, Some("claude-sonnet-4-6")));
        assert_eq!(limits.context, Some(1_000_000));
        assert_eq!(limits.output, Some(64_000));
        // Unknown model → unknown limits, not zeros.
        let unknown = rt.block_on(resolve(&key, Some("no-such-model-xyz")));
        assert_eq!(unknown, ModelLimits::default());
    }

    #[test]
    fn snap_picks_floor_tier_by_context() {
        // ≥1M → sonnet's 1M entry, even for larger windows (2M grok).
        assert_eq!(
            snap_internal_model(2_000_000, None),
            "anthropic:claude-sonnet-4-6"
        );
        assert_eq!(
            snap_internal_model(1_000_000, None),
            "anthropic:claude-sonnet-4-6"
        );
        // 400k (gpt-5-class) floors to the 332k entry, never rounds up.
        assert_eq!(
            snap_internal_model(400_000, None),
            "anthropic:claude-opus-4-8"
        );
        assert_eq!(
            snap_internal_model(332_000, None),
            "anthropic:claude-opus-4-8"
        );
        // 200k tier splits on the real output limit.
        assert_eq!(
            snap_internal_model(256_000, Some(64_000)),
            "anthropic:claude-haiku-4-5-20251001"
        );
        assert_eq!(
            snap_internal_model(200_000, Some(32_000)),
            "anthropic:claude-opus-4-5-20251101"
        );
        // Below 200k there's nothing smaller; unknown output stays conservative.
        assert_eq!(
            snap_internal_model(131_072, None),
            "anthropic:claude-opus-4-5-20251101"
        );
    }
}
