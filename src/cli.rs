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

    /// Pin the initial agent mode: smart | rush | deep | large.
    #[arg(long = "mode", value_name = "MODE")]
    pub mode: Option<String>,

    /// Per-mode model override for `rush`.
    #[arg(long = "rush-model", value_name = "MODEL")]
    pub rush_model: Option<String>,
    /// Per-mode model override for `smart`.
    #[arg(long = "smart-model", value_name = "MODEL")]
    pub smart_model: Option<String>,
    /// Per-mode model override for `deep`.
    #[arg(long = "deep-model", value_name = "MODEL")]
    pub deep_model: Option<String>,
    /// Per-mode model override for `large`.
    #[arg(long = "large-model", value_name = "MODEL")]
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
