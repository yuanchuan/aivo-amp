//! Real context/output limits for the launched model, applied to amp.
//!
//! aivo's plugin protocol (docs/plugin-protocol.md → Endpoint handoff) hands
//! `endpoint`-granted plugins two advisory env vars — `AIVO_MODEL_CONTEXT_WINDOW`
//! and `AIVO_MODEL_MAX_OUTPUT_TOKENS` — so the wrapped CLI doesn't assume a
//! default window for unknown models. amp is a fat plugin (no `endpoint` cap),
//! so alongside honoring those vars when a host sets them, it self-resolves the
//! same cascade the host uses: live models-cache (harvested from the key's
//! `/v1/models`) → static long-context table. Absent limits mean "unknown" —
//! amp keeps its own defaults, per the protocol.
//!
//! The limits feed two amp-side knobs (see `launch`):
//! - `output` → `max_tokens_cap` on both bridge translators, clamping amp's
//!   requested `max_tokens` to what the model actually accepts;
//! - `context` → a believed-model snap: amp's context meter, compaction budget,
//!   and requested `max_tokens` all come from its *compiled-in* catalog entry
//!   for the mode's model (it has no context-window setting), so we pick the
//!   catalog entry whose window best matches the real model and pin it via
//!   `internal.model` — the same trick the bridge docs describe doing by hand.

use aivo::services::context_window::static_context_window;
use aivo::services::model_names::strip_context_suffix;
use aivo::services::models_cache::{ModelsCache, full_catalog_key};
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
/// channel) → models-cache metadata under the key's full-catalog then picker
/// namespace (TTL-free: limits are stable once published) → static
/// long-context table (context only).
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
    let meta = match cache
        .get_metadata(&full_catalog_key(&key.base_url), model)
        .await
    {
        Some(m) => Some(m),
        None => cache.get_metadata(&key.base_url, model).await,
    };
    if let Some(meta) = meta {
        limits.context = limits.context.or(meta.context_window);
        // The cache stores max-output as the `aivo models` display string
        // ("128K", "1M"); parse it back.
        limits.output = limits
            .output
            .or_else(|| meta.max_output.as_deref().and_then(parse_token_count));
    }
    limits.context = limits.context.or_else(|| static_context_window(model));
    limits
}

fn env_limit(var: &str) -> Option<u64> {
    std::env::var(var).ok()?.trim().parse().ok()
}

/// Parses a token count in the formats aivo renders/accepts: plain digits
/// ("131072"), K ("128K" → 128_000), M with one optional decimal ("1M",
/// "1.5M"). Mirrors the host's `format_token_count` output, so round-tripping
/// the cache's display string loses at most sub-K precision — and always
/// rounds down, the safe direction for a cap.
fn parse_token_count(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Ok(n) = s.parse::<u64>() {
        return Some(n);
    }
    let (num, scale) = match s.char_indices().last()? {
        (i, 'k') | (i, 'K') => (&s[..i], 1_000u64),
        (i, 'm') | (i, 'M') => (&s[..i], 1_000_000u64),
        _ => return None,
    };
    if let Ok(n) = num.parse::<u64>() {
        return Some(n * scale);
    }
    // "1.5M" form: one fractional digit, M only (format_token_count never
    // emits fractional K).
    let (whole, frac) = num.split_once('.')?;
    if scale != 1_000_000 || frac.len() != 1 {
        return None;
    }
    let whole: u64 = whole.parse().ok()?;
    let frac: u64 = frac.parse().ok()?;
    Some(whole * scale + frac * 100_000)
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
    fn parse_token_count_reads_plain_k_and_m_forms() {
        assert_eq!(parse_token_count("131072"), Some(131_072));
        assert_eq!(parse_token_count("128K"), Some(128_000));
        assert_eq!(parse_token_count("200k"), Some(200_000));
        assert_eq!(parse_token_count("1M"), Some(1_000_000));
        assert_eq!(parse_token_count("1.5M"), Some(1_500_000));
        assert_eq!(parse_token_count(" 64K "), Some(64_000));
    }

    #[test]
    fn parse_token_count_rejects_garbage() {
        for s in ["", "K", "1.5K", "1.55M", "12B", "abc", "-1"] {
            assert_eq!(parse_token_count(s), None, "{s:?}");
        }
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
