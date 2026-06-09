# aivo-amp

Run Sourcegraph [**Amp**](https://ampcode.com/) on any
[aivo](https://github.com/yuanchuan/aivo)-managed provider — no ampcode.com
subscription required.

## Install

```bash
aivo plugins install github:yuanchuan/aivo-amp
```

## Usage

### Launch Amp

| Command | What it does |
|---------|-------------|
| `aivo amp` | Launch on your active key |
| `aivo amp "fix tests"` | Launch with a prompt |
| `aivo amp -k work` | Pick a key |
| `aivo amp -k work -m gpt-4o` | Pick a key and model |
| `aivo amp -m` | Opens the model picker |
| `aivo amp --mode deep` | Launch in deep-reasoning mode |
| `aivo amp --mode` | Opens the agent mode picker |
| `aivo amp -- -y` | Forward `-y` through to amp |

aivo resolves `-k`/`-m` before launch. Bare `-k`/`-m` open their respective
pickers. Native `ampcode.com` keys skip the bridge.

### Plugin flags

| Flag | Meaning |
| --- | --- |
| `--mode [smart\|rush\|deep\|large]` | Pin initial agent mode. Bare `--mode` opens a picker. |
| `--rush-model <MODEL>` | Model override for rush mode |
| `--smart-model <MODEL>` | Model override for smart mode |
| `--deep-model <MODEL>` | Model override for deep mode |
| `--large-model <MODEL>` | Model override for large mode |
| `--disable-tool <NAME>` | Strip a tool from Amp's request (repeatable) |
| `--passthrough` | Forward management plane to ampcode.com instead of stubbing |
| `--debug[=PATH]` | Capture bridge traffic to a JSONL trace |

### Management

| Command | What it does |
|---------|-------------|
| `aivo amp threads list` | List threads |
| `aivo amp threads continue T-<id>` | Resume a thread |
| `aivo stats --by amp` | Show token usage |

## License

MIT
