# aivo-amp

An [aivo](https://github.com/yuanchuan/aivo) plugin that runs Sourcegraph
[**Amp**](https://ampcode.com/) on your aivo-managed keys, models, and
endpoints. It ships as a standalone sibling binary so the default `aivo` stays
lean, and routes Amp through aivo's in-process bridge — translating Amp's
Anthropic/Responses calls to whatever your key points at and stubbing Amp's
management plane (auth/threads/telemetry) locally.

## Install

```bash
aivo plugins install github:yuanchuan/aivo-amp
```
## Usage

```bash
aivo amp                          # interactive on your active aivo key
aivo amp -m <model> "fix tests"   # pick the model (aivo resolves it)
aivo amp -k work -m <model>       # pick the key + model
aivo amp -k                       # bare -k → aivo's key picker
```

aivo resolves `-k`/`-m` before launch (bare `-k` opens its key picker; a new key
with no saved model opens its model picker, remembered per key). Every other flag
passes straight through to `amp`. Native `ampcode.com` keys talk to Amp directly
with no bridge.

The plugin adds a few flags of its own:

| Option | Meaning |
| --- | --- |
| `--mode <smart\|rush\|deep\|large>` | Pin the initial agent mode. |
| `--rush-model` / `--smart-model` / `--deep-model` / `--large-model` | Per-mode model override. |
| `--disable-tool <name>` | Strip a tool from Amp's upstream request (repeatable). |
| `--passthrough` | Forward Amp's management plane upstream instead of stubbing it. |
| `--debug[=path]` | Capture bridge + upstream traffic to a JSONL trace. |

`aivo amp trust` gates the MCP servers a workspace's `.amp/settings.json`
declares (mirrors `amp mcp approve`); run it to approve, `--list`, or
`--revoke <name>`.

## Develop

Requires a Rust toolchain (edition 2024, rustc ≥ 1.88). The plugin depends on the
`aivo` lib by path (`aivo = { path = "../aivo" }`), so it must sit next to an
`aivo` checkout.

```bash
cargo build --release   # release binary
cargo test              # unit tests
cargo clippy --all-targets -- -D warnings
```

## License

MIT
