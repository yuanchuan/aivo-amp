//! CLI surface for the `aivo-amp` plugin. The aivo host dispatches both
//! `aivo amp …` and `aivo run amp …` to this binary with the same argv, so we
//! own the parse: a `trust` management subcommand, otherwise a launch.

use clap::Parser;

use crate::mode_models::AmpModeModels;

/// `aivo amp trust …` — workspace MCP-approval management. Mirrors aivo's
/// original `AmpArgs`; consumed by [`crate::commands::AmpCommand`].
#[derive(Debug, Default, Parser)]
#[command(
    name = "aivo-amp",
    disable_help_subcommand = true,
    about = "amp configuration commands"
)]
pub struct AmpArgs {
    /// Subcommand: `trust` (currently the only option).
    #[arg(value_name = "ACTION")]
    pub action: Option<String>,

    /// Approve every pending workspace MCP server without prompting.
    #[arg(long)]
    pub all: bool,

    /// List approved MCP servers for the current workspace and exit.
    #[arg(long)]
    pub list: bool,

    /// Revoke approval for a specific server name in the current workspace.
    #[arg(long, value_name = "NAME")]
    pub revoke: Option<String>,
}

/// `aivo amp [flags] [amp-args…]` — launch amp through aivo's bridge.
/// Known flags are consumed here; everything else is forwarded to `amp`.
#[derive(Debug, Parser)]
#[command(
    name = "aivo-amp",
    about = "Launch Sourcegraph Amp through aivo's provider bridge",
    disable_help_subcommand = true
)]
pub struct LaunchCli {
    /// API key id or name. Bare `-k` opens the key picker.
    #[arg(short = 'k', long = "key", num_args = 0..=1, default_missing_value = "", value_name = "ID|NAME")]
    pub key: Option<String>,

    /// Force this model on the wire (bridge rewrites amp's mode model names).
    #[arg(short = 'm', long = "model", num_args = 0..=1, default_missing_value = "", value_name = "MODEL")]
    pub model: Option<String>,

    /// Pin the initial agent mode: smart | rush | deep | large. Bare `--mode`
    /// opens an interactive picker.
    #[arg(long = "mode", num_args = 0..=1, default_missing_value = "", value_name = "MODE")]
    pub mode: Option<String>,

    /// Per-mode model override for `rush` (`[key::]model`; bare flag or `key::` opens pickers).
    #[arg(long = "rush-model", num_args = 0..=1, default_missing_value = "", value_name = "MODEL")]
    pub rush_model: Option<String>,
    /// Per-mode model override for `smart` (`[key::]model`; bare flag or `key::` opens pickers).
    #[arg(long = "smart-model", num_args = 0..=1, default_missing_value = "", value_name = "MODEL")]
    pub smart_model: Option<String>,
    /// Per-mode model override for `deep` (`[key::]model`; bare flag or `key::` opens pickers).
    #[arg(long = "deep-model", num_args = 0..=1, default_missing_value = "", value_name = "MODEL")]
    pub deep_model: Option<String>,
    /// Per-mode model override for `large` (`[key::]model`; bare flag or `key::` opens pickers).
    #[arg(long = "large-model", num_args = 0..=1, default_missing_value = "", value_name = "MODEL")]
    pub large_model: Option<String>,

    /// Strip a tool from amp's request to the upstream (repeatable).
    #[arg(long = "disable-tool", value_name = "NAME")]
    pub disable_tool: Vec<String>,

    /// Forward amp's management plane (auth/threads/telemetry) to the URL in
    /// the user's amp secrets.json instead of stubbing it locally.
    #[arg(long = "passthrough")]
    pub passthrough: bool,

    /// Capture bridge + upstream traffic to a JSONL trace. Bare `--debug`
    /// uses the default path under ~/.config/aivo/logs.
    #[arg(long = "debug", num_args = 0..=1, default_missing_value = "", value_name = "PATH")]
    pub debug: Option<String>,

    /// Remaining args are passed through to the `amp` binary verbatim.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub amp_args: Vec<String>,
}

impl LaunchCli {
    /// Builds the per-mode override carrier from the parsed flags.
    pub fn to_mode_models(&self) -> AmpModeModels {
        AmpModeModels {
            rush: self.rush_model.clone(),
            smart: self.smart_model.clone(),
            deep: self.deep_model.clone(),
            large: self.large_model.clone(),
            disable_tools: self.disable_tool.clone(),
            initial_mode: self.mode.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bare `--mode` parses to `Some("")`, the sentinel `launch_amp` turns into
    /// an interactive picker (mirrors bare `-k`/`-m`).
    #[test]
    fn bare_mode_flag_parses_as_empty() {
        let cli = LaunchCli::try_parse_from(["aivo-amp", "--mode"]).unwrap();
        assert_eq!(cli.mode.as_deref(), Some(""));
    }

    #[test]
    fn explicit_mode_value_parses() {
        let cli = LaunchCli::try_parse_from(["aivo-amp", "--mode", "deep"]).unwrap();
        assert_eq!(cli.mode.as_deref(), Some("deep"));
    }

    /// Bare `--mode` must not swallow a following flag as its value.
    #[test]
    fn bare_mode_does_not_consume_following_flag() {
        let cli = LaunchCli::try_parse_from(["aivo-amp", "--mode", "--debug"]).unwrap();
        assert_eq!(cli.mode.as_deref(), Some(""));
        assert!(cli.debug.is_some());
    }

    #[test]
    fn no_mode_flag_is_none() {
        let cli = LaunchCli::try_parse_from(["aivo-amp"]).unwrap();
        assert_eq!(cli.mode, None);
    }

    /// Bare per-mode model flags parse to `Some("")` — the sentinel
    /// `launch_amp` turns into a model picker (mirrors bare `-k`/`-m`/`--mode`).
    #[test]
    fn bare_mode_model_flags_parse_as_empty() {
        let cli = LaunchCli::try_parse_from(["aivo-amp", "--rush-model", "--deep-model"]).unwrap();
        assert_eq!(cli.rush_model.as_deref(), Some(""));
        assert_eq!(cli.deep_model.as_deref(), Some(""));
        assert_eq!(cli.smart_model, None);
    }

    #[test]
    fn explicit_mode_model_value_parses() {
        let cli =
            LaunchCli::try_parse_from(["aivo-amp", "--smart-model", "deepseek-chat"]).unwrap();
        assert_eq!(cli.smart_model.as_deref(), Some("deepseek-chat"));
    }
}
