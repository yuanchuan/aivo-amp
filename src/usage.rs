//! Token-usage extraction for amp turns, vendored from aivo's `serve_router`
//! so the plugin builds against public aivo without reaching into crate
//! internals. `parse_token_usage` reads a buffered response's `usage` block;
//! `StreamUsageSniffer` accumulates usage off a forwarded SSE stream. The
//! bridge feeds the result to `record_amp_usage`.

use serde_json::Value;

/// Token counts pulled from a response `usage` block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TokenUsage {
    pub(crate) prompt: u64,
    pub(crate) completion: u64,
    pub(crate) cache_read: u64,
    pub(crate) cache_creation: u64,
}

impl TokenUsage {
    fn is_zero(&self) -> bool {
        *self == TokenUsage::default()
    }

    /// Per-field max. Merges partial usage from successive stream events —
    /// Anthropic reports input in `message_start` and output in `message_delta`,
    /// and providers send cumulative counts, so the max is the final total.
    fn merge_max(&mut self, other: &TokenUsage) {
        self.prompt = self.prompt.max(other.prompt);
        self.completion = self.completion.max(other.completion);
        self.cache_read = self.cache_read.max(other.cache_read);
        self.cache_creation = self.cache_creation.max(other.cache_creation);
    }
}

/// Pull a `TokenUsage` out of any provider's response JSON object: OpenAI chat
/// (`usage` with `prompt_tokens`/`completion_tokens`), Responses (`usage` with
/// `input_tokens`/`output_tokens`, or nested under `response`), Anthropic
/// (`usage`, or nested under `message`), or Gemini (`usageMetadata`). Returns
/// `None` when there's no usage or it's all zero.
fn extract_usage_from_value(v: &Value) -> Option<TokenUsage> {
    if let Some(u) = v
        .get("usage")
        .or_else(|| v.get("message").and_then(|m| m.get("usage")))
        .or_else(|| v.get("response").and_then(|r| r.get("usage")))
    {
        let num = |a: &str, b: &str| -> u64 {
            u.get(a)
                .or_else(|| u.get(b))
                .and_then(|x| x.as_u64())
                .unwrap_or(0)
        };
        let cache_read = u
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .or_else(|| {
                u.get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
            })
            .or_else(|| u.get("cache_read_input_tokens"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let usage = TokenUsage {
            prompt: num("prompt_tokens", "input_tokens"),
            completion: num("completion_tokens", "output_tokens"),
            cache_read,
            cache_creation: u
                .get("cache_creation_input_tokens")
                .and_then(|x| x.as_u64())
                .unwrap_or(0),
        };
        return (!usage.is_zero()).then_some(usage);
    }
    if let Some(um) = v.get("usageMetadata") {
        let n = |k: &str| um.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
        let usage = TokenUsage {
            prompt: n("promptTokenCount"),
            completion: n("candidatesTokenCount"),
            cache_read: n("cachedContentTokenCount"),
            cache_creation: 0,
        };
        return (!usage.is_zero()).then_some(usage);
    }
    None
}

/// Extract token usage from a buffered JSON response body.
pub(crate) fn parse_token_usage(body: &[u8]) -> Option<TokenUsage> {
    let v: Value = serde_json::from_slice(body).ok()?;
    extract_usage_from_value(&v)
}

/// `data: {...}` → `{...}`. `None` for non-data SSE lines.
fn sse_data_payload(line: &str) -> Option<&str> {
    line.strip_prefix("data:").map(str::trim_start)
}

/// Accumulates token usage from a forwarded SSE stream by scanning `data:` lines
/// for any provider's usage event. A no-op when `enabled` is false (native
/// launches don't account usage). `finish()` yields the merged per-field max.
pub(crate) struct StreamUsageSniffer {
    enabled: bool,
    pending: String,
    usage: TokenUsage,
    seen: bool,
}

impl StreamUsageSniffer {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pending: String::new(),
            usage: TokenUsage::default(),
            seen: false,
        }
    }

    /// Feed a raw upstream chunk (native provider SSE bytes).
    pub(crate) fn observe(&mut self, chunk: &[u8]) {
        if !self.enabled {
            return;
        }
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        // Parse complete lines; keep any trailing partial line buffered. Usage
        // only rides on `data:` lines, so skip everything else.
        while let Some(nl) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=nl).collect();
            let Some(json) = sse_data_payload(line.trim()) else {
                continue;
            };
            if json.is_empty() || json == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(json)
                && let Some(u) = extract_usage_from_value(&v)
            {
                self.usage.merge_max(&u);
                self.seen = true;
            }
        }
    }

    pub(crate) fn finish(self) -> Option<TokenUsage> {
        (self.enabled && self.seen).then_some(self.usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_openai_chat_shape() {
        let body = json!({
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 20},
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!(
            (u.prompt, u.completion, u.cache_read, u.cache_creation),
            (30, 12, 20, 0)
        );
    }

    #[test]
    fn parses_responses_and_anthropic_nesting() {
        // Responses-style `input_tokens`/`output_tokens` nested under `response`.
        let resp = json!({"response": {"usage": {"input_tokens": 5, "output_tokens": 7}}});
        let u = parse_token_usage(resp.to_string().as_bytes()).unwrap();
        assert_eq!((u.prompt, u.completion), (5, 7));
        // Anthropic cache_creation under `message.usage`.
        let anth = json!({"message": {"usage": {
            "input_tokens": 1, "output_tokens": 2,
            "cache_read_input_tokens": 9, "cache_creation_input_tokens": 4,
        }}});
        let u = parse_token_usage(anth.to_string().as_bytes()).unwrap();
        assert_eq!((u.cache_read, u.cache_creation), (9, 4));
    }

    #[test]
    fn none_when_absent_or_all_zero() {
        assert!(parse_token_usage(b"{}").is_none());
        assert!(parse_token_usage(b"not json").is_none());
        let zero = json!({"usage": {"prompt_tokens": 0, "completion_tokens": 0}});
        assert!(parse_token_usage(zero.to_string().as_bytes()).is_none());
    }

    #[test]
    fn sniffer_merges_max_across_chunks_and_respects_enabled() {
        let mut s = StreamUsageSniffer::new(true);
        // Split a usage event across two observe() calls; input arrives first.
        s.observe(b"data: {\"usage\":{\"input_tokens\":100,\"output_tokens\":0}}\n");
        s.observe(b"data: {\"usage\":{\"input_tokens\":100,\"output_tokens\":40}}\n");
        s.observe(b"data: [DONE]\n");
        let u = s.finish().unwrap();
        assert_eq!((u.prompt, u.completion), (100, 40));

        // Disabled sniffer accounts nothing.
        let mut off = StreamUsageSniffer::new(false);
        off.observe(b"data: {\"usage\":{\"input_tokens\":5,\"output_tokens\":5}}\n");
        assert!(off.finish().is_none());
    }
}
