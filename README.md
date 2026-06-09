# aivo-amp

An [aivo](https://github.com/yuanchuan/aivo) plugin that runs Sourcegraph
[**Amp**](https://ampcode.com/) on your aivo-managed keys, models, and endpoints.
It ships as a standalone sibling binary and routes Amp through aivo's in-process
bridge, stubbing Amp's management plane (auth/threads/telemetry) locally.

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

aivo resolves `-k`/`-m` before launch (bare `-k` opens the key picker; a new key
with no model opens the model picker, remembered per key). Other flags pass
through to `amp`. Native `ampcode.com` keys skip the bridge.

The plugin adds a few flags of its own:

| Option | Meaning |
| --- | --- |
| `--mode <smart\|rush\|deep\|large>` | Pin the initial agent mode. |
| `--rush-model` / `--smart-model` / `--deep-model` / `--large-model` | Per-mode model override. |
| `--disable-tool <name>` | Strip a tool from Amp's upstream request (repeatable). |
| `--passthrough` | Forward Amp's management plane upstream instead of stubbing it. |
| `--debug[=path]` | Capture bridge + upstream traffic to a JSONL trace. |

`aivo amp trust` gates the MCP servers a workspace's `.amp/settings.json`
declares (mirrors `amp mcp approve`): run it bare to walk pending servers, `--all`
to approve them all, `--list` to show approvals, or `--revoke <name>` to drop one.
Approvals are scoped per settings path **and** config hash, so a command or
version change forces re-approval.

`aivo amp threads <list|…>` passes through to `amp` (state lives in the bridge);
`aivo stats --by amp` reports token usage for threads run through your aivo keys.

## License

MIT
