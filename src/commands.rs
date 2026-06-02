//! `aivo amp` subcommand — currently scoped to `aivo amp trust`, which
//! mirrors the workspace MCP approval workflow that direct `amp` exposes
//! via `amp mcp approve`.
//!
//! Background: the bridge writes a merged `--settings-file` so workspace
//! `.amp/settings.json` is preserved end-to-end. That bypasses amp's
//! per-server approval gate, which is precisely what stops a hostile
//! checkout from auto-launching an MCP server with the user's
//! credentials. `aivo amp trust` is our own approval gate that the
//! bridge consults before merging workspace MCP servers.

use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::amp_trust::{
    AmpTrustStore, find_workspace_amp_settings, hash_server_config, parse_amp_settings_file,
    workspace_mcp_servers,
};
use crate::cli::AmpArgs;
use aivo::errors::ExitCode;
use aivo::services::system_env;
use aivo::style;

#[derive(Default)]
pub struct AmpCommand;

impl AmpCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: AmpArgs) -> ExitCode {
        match args.action.as_deref() {
            Some("trust") => trust_dispatch(&args),
            Some(other) => {
                eprintln!(
                    "{} unknown subcommand '{}'. Valid: trust",
                    style::red("Error:"),
                    other
                );
                ExitCode::UserError
            }
            None => {
                Self::print_help(None);
                ExitCode::Success
            }
        }
    }

    pub fn print_help(action: Option<&str>) {
        match action {
            Some("trust") => print_help_trust(),
            _ => print_help_overview(),
        }
    }
}

fn print_help_overview() {
    println!("{}", style::bold("aivo amp"));
    println!("  amp-specific configuration commands.\n");
    println!("{}", style::bold("Usage:"));
    println!("  aivo amp trust                       interactively approve workspace MCP servers");
    println!(
        "  aivo amp trust --all                 approve every pending server in this workspace"
    );
    println!(
        "  aivo amp trust --list                list approved servers (for current workspace)"
    );
    println!(
        "  aivo amp trust --revoke <NAME>       revoke an approved server in this workspace\n"
    );
    println!("{}", style::bold("Notes:"));
    println!("  - 'trust' applies only to MCP servers declared in the workspace's");
    println!("    `.amp/settings.json` (or `.jsonc`). Servers in your user-level");
    println!("    `~/.config/amp/settings.json` are not gated.");
    println!("  - Approvals are scoped per workspace settings file path AND per");
    println!("    server config hash. A package version bump or command change");
    println!("    requires re-approval.");
    println!("  - For amp's own thread management (list/markdown/delete/continue/...),");
    println!("    just use `aivo amp threads <subcommand>` — those flow through to amp,");
    println!("    which talks to aivo's bridge for persisted thread state.");
}

fn print_help_trust() {
    println!("{} aivo amp trust [OPTIONS]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Approve workspace MCP servers declared in `.amp/settings.json`.")
    );
    println!(
        "{}",
        style::dim("With no flags, walks pending servers interactively.")
    );
    println!();
    let row = |flag: &str, desc: &str| {
        println!(
            "  {}{}",
            style::cyan(format!("{:<22}", flag)),
            style::dim(desc)
        );
    };
    println!("{}", style::bold("Options:"));
    row("--all", "Approve every pending server without prompting");
    row("--list", "List approved servers for the current workspace");
    row("--revoke <NAME>", "Revoke approval for a specific server");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo amp trust"));
    println!("  {}", style::dim("aivo amp trust --list"));
    println!("  {}", style::dim("aivo amp trust --revoke playwright"));
}

fn trust_dispatch(args: &AmpArgs) -> ExitCode {
    let cwd = match system_env::current_dir() {
        Some(c) => c,
        None => {
            eprintln!("{} cannot resolve current directory", style::red("Error:"));
            return ExitCode::UserError;
        }
    };
    let home = system_env::home_dir();

    let workspace = match find_workspace_amp_settings(&cwd, home.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!(
                "{} no workspace .amp/settings.json found walking up from {}",
                style::yellow("note:"),
                cwd.display()
            );
            return ExitCode::Success;
        }
    };

    if let Some(name) = args.revoke.as_deref() {
        return revoke_server(&workspace, name);
    }
    if args.list {
        return list_approvals(&workspace);
    }
    interactive_approval(&workspace, args.all)
}

fn revoke_server(workspace: &Path, name: &str) -> ExitCode {
    let mut trust = AmpTrustStore::load();
    if !trust.revoke(workspace, name) {
        eprintln!(
            "{} no approval found for server '{}' in {}",
            style::yellow("note:"),
            name,
            workspace.display()
        );
        return ExitCode::Success;
    }
    if let Err(e) = trust.save() {
        eprintln!("{} failed to save trust file: {}", style::red("Error:"), e);
        return ExitCode::UserError;
    }
    println!(
        "{} revoked '{}' for {}",
        style::green("✓"),
        name,
        workspace.display()
    );
    ExitCode::Success
}

fn list_approvals(workspace: &Path) -> ExitCode {
    let trust = AmpTrustStore::load();
    let approved = trust.approved_servers_for(workspace);
    println!("{} {}", style::bold("Workspace:"), workspace.display());
    if approved.is_empty() {
        println!("  {}", style::dim("(no approved servers)"));
    } else {
        for name in approved {
            println!("  {} {}", style::green("✓"), name);
        }
    }
    ExitCode::Success
}

fn interactive_approval(workspace: &Path, approve_all: bool) -> ExitCode {
    let settings = match parse_amp_settings_file(workspace) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "{} failed to parse {}: {}",
                style::red("Error:"),
                workspace.display(),
                e
            );
            return ExitCode::UserError;
        }
    };

    let servers = workspace_mcp_servers(&settings);
    if servers.is_empty() {
        println!(
            "{} no `amp.mcpServers` declared in {}",
            style::dim("note:"),
            workspace.display()
        );
        return ExitCode::Success;
    }

    let mut trust = AmpTrustStore::load();
    println!("{} {}", style::bold("Workspace:"), workspace.display());
    println!();

    let mut approved = 0usize;
    let mut skipped = 0usize;
    let mut already = 0usize;
    let mut quit = false;

    for (name, config) in &servers {
        let already_approved = trust.is_approved(workspace, name, config);
        let label_status = if already_approved {
            style::green("✓ already approved")
        } else {
            style::yellow("? pending")
        };
        println!("  {} {}", style::bold(name.as_str()), label_status);
        println!("    {}", style::dim(summarize_server_config(config)));
        println!(
            "    {}",
            style::dim(format!("hash: {}", &hash_server_config(config)[..16]))
        );

        if already_approved {
            already += 1;
            continue;
        }
        if quit {
            skipped += 1;
            continue;
        }

        let decision = if approve_all {
            ApprovalChoice::Approve
        } else {
            prompt_choice()
        };
        match decision {
            ApprovalChoice::Approve => {
                trust.approve(workspace, name, config);
                approved += 1;
                println!("    {}", style::green("→ approved"));
            }
            ApprovalChoice::Skip => {
                skipped += 1;
                println!("    {}", style::dim("→ skipped"));
            }
            ApprovalChoice::Quit => {
                skipped += 1;
                quit = true;
                println!("    {}", style::dim("→ quit (remaining will be skipped)"));
            }
        }
        println!();
    }

    if approved > 0
        && let Err(e) = trust.save()
    {
        eprintln!("{} failed to save trust file: {}", style::red("Error:"), e);
        return ExitCode::UserError;
    }

    println!(
        "{}: approved {}, skipped {}, already approved {}",
        style::bold("Summary"),
        approved,
        skipped,
        already
    );
    ExitCode::Success
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApprovalChoice {
    Approve,
    Skip,
    Quit,
}

fn prompt_choice() -> ApprovalChoice {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("    [a]pprove / [s]kip / [q]uit: ");
        let _ = stdout.flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            return ApprovalChoice::Skip;
        }
        match line.trim().to_lowercase().as_str() {
            "a" | "approve" | "y" | "yes" => return ApprovalChoice::Approve,
            "s" | "skip" | "n" | "no" => return ApprovalChoice::Skip,
            "q" | "quit" => return ApprovalChoice::Quit,
            _ => continue,
        }
    }
}

/// One-line, human-readable preview of a server config — `command + args`
/// for stdio servers, `url` for HTTP servers, otherwise a compact JSON
/// rendering as a last resort.
fn summarize_server_config(config: &serde_json::Value) -> String {
    if let Some(url) = config.get("url").and_then(|v| v.as_str()) {
        return format!("url: {url}");
    }
    if let Some(cmd) = config.get("command").and_then(|v| v.as_str()) {
        let args: Vec<String> = config
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| a.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if args.is_empty() {
            return cmd.to_string();
        }
        return format!("{} {}", cmd, args.join(" "));
    }
    config.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn summarize_server_config_renders_command_with_args() {
        let cfg = json!({"command": "npx", "args": ["-y", "mcp-pkg"]});
        assert_eq!(summarize_server_config(&cfg), "npx -y mcp-pkg");
    }

    #[test]
    fn summarize_server_config_renders_url_for_http_servers() {
        let cfg = json!({"url": "https://example.com/mcp"});
        assert_eq!(
            summarize_server_config(&cfg),
            "url: https://example.com/mcp"
        );
    }

    #[test]
    fn summarize_server_config_falls_back_to_json_when_neither() {
        let cfg = json!({"weird": "shape"});
        assert!(summarize_server_config(&cfg).contains("weird"));
    }
}
