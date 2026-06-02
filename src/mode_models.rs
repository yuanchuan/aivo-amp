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
    /// lands, so this only applies before the first send. Bare flag
    /// (`Some("")`) requests an interactive picker.
    pub initial_mode: Option<String>,
}

/// Canonical amp agent modes. Order matches both amp's own catalog
/// (rush/smart/deep/large) and the JSON object emitted to
/// `internal.model`. Used for `--mode` validation and the picker UI.
pub const AMP_AGENT_MODES: [(&str, &str); 4] = [
    ("smart", "Default — most capable model + tools"),
    ("rush", "Fast/cheap for small, well-defined tasks"),
    ("deep", "Deep reasoning"),
    ("large", "Biggest context window (1M)"),
];

impl AmpModeModels {
    /// True when at least one per-mode slot was passed bare (empty string),
    /// meaning the caller wants an interactive picker.
    pub fn has_any_picker_request(&self) -> bool {
        [&self.rush, &self.smart, &self.deep, &self.large]
            .iter()
            .any(|v| matches!(v, Some(s) if s.is_empty()))
    }

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
