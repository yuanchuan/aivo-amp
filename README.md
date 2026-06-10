# aivo-amp

Run Sourcegraph [**Amp**](https://ampcode.com/) on any
[aivo](https://github.com/yuanchuan/aivo)-managed provider — no ampcode.com
subscription required.

## Install

```sh
aivo plugins install github:yuanchuan/aivo-amp
```

## Usage

```
aivo amp                          # launch on your active aivo key
aivo amp "fix tests"              # launch with a prompt
aivo amp -k work -m gpt-4o        # pick a key and model
aivo amp -m                       # bare -k/-m → picker
aivo amp --mode deep              # launch in a given agent mode
aivo amp -- -y                    # forward flags through to amp
```

aivo resolves `-k`/`-m` before launch; native `ampcode.com` keys skip the
bridge. When the model's real limits are known, the plugin aligns Amp's
`max_tokens`, context meter, and compaction budget with them.

### Plugin flags

| Flag | Meaning |
| --- | --- |
| `--mode [smart\|rush\|deep\|large]` | Pin initial agent mode. Bare `--mode` opens a picker. |
| `--rush/smart/deep/large-model <MODEL>` | Per-mode model override |
| `--disable-tool <NAME>` | Strip a tool from Amp's request (repeatable) |
| `--passthrough` | Forward management plane to ampcode.com instead of stubbing |
| `--debug[=PATH]` | Capture bridge traffic to a JSONL trace |

### Management

```
aivo amp threads list             # list threads
aivo amp threads continue T-<id>  # resume a thread
aivo stats --by amp               # show token usage
```

## License

MIT
