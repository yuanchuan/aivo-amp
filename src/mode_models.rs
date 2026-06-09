//! Amp per-mode model overrides, ported from aivo's `environment_injector`.
//! Populated from `aivo amp` flags and rendered into amp's `internal.model`.

use serde_json::{Map, Value};

/// Amp-only run overrides. Set from `aivo amp` flags:
/// - `--rush-model / --smart-model / --deep-model / --large-model` populate
///   `rush/smart/deep/large`. When any is non-empty, the bridge writes
///   `amp.internal.model` as an *object* keyed by mode name in the
///   generated settings.json.
/// - `--disable-tool <name>` (repeatable) populates `disable_tools`. The
///   bridge writes `tools.disable: [...]` so amp strips the named tool
///   from the request to the upstream — useful when the upstream lacks
///   server-backed tools (`web_search`, `read_web_page`).
#[derive(Debug, Clone, Default)]
pub struct AmpModeModels {
    pub rush: Option<String>,
    pub smart: Option<String>,
    pub deep: Option<String>,
    pub large: Option<String>,
    pub disable_tools: Vec<String>,
    /// `--mode <smart|rush|deep|large>`: pin the initial agent mode for
    /// this thread. Amp locks the agent mode after the first message
    /// lands, so this only applies before the first send.
    pub initial_mode: Option<String>,
}

/// Canonical amp agent modes (order matches amp's own catalog). Used for
/// `--mode` validation.
pub const AMP_AGENT_MODES: [&str; 4] = ["smart", "rush", "deep", "large"];

/// One-line description per mode, same order as [`AMP_AGENT_MODES`], mirrored
/// from amp's own mode catalog. Shown as the dim hint in the `--mode` picker.
pub const AMP_AGENT_MODE_DESCRIPTIONS: [&str; 4] = [
    "Strong intelligence for any task",
    "Fast, low-token mode for small, well-defined tasks",
    "The most capable coding mode with deep reasoning",
    "The biggest context window possible (1M tokens), for large tasks",
];

impl AmpModeModels {
    /// Renders the override as the JSON object form amp expects:
    /// `{"<mode>": "<provider>:<model>", ...}`. Modes with no override
    /// are omitted; if the user's value lacks a `provider:` prefix we
    /// add `openai:` since amp validates the format and the bridge
    /// rewrites the on-the-wire model name regardless of provider.
    pub fn to_internal_model_value(&self) -> Option<Value> {
        let mut obj = Map::new();
        for (mode, value) in [
            ("rush", &self.rush),
            ("smart", &self.smart),
            ("deep", &self.deep),
            ("large", &self.large),
        ] {
            if let Some(m) = value.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
                let provider_prefixed = if m.contains(':') {
                    m.to_string()
                } else {
                    format!("openai:{m}")
                };
                obj.insert(mode.to_string(), Value::String(provider_prefixed));
            }
        }
        if obj.is_empty() {
            None
        } else {
            Some(Value::Object(obj))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `--mode` picker zips modes with descriptions; a length mismatch
    /// would silently drop the trailing mode(s) from the picker.
    #[test]
    fn modes_and_descriptions_stay_aligned() {
        assert_eq!(AMP_AGENT_MODES.len(), AMP_AGENT_MODE_DESCRIPTIONS.len());
        assert!(AMP_AGENT_MODE_DESCRIPTIONS.iter().all(|d| !d.is_empty()));
    }
}
