# Safety splits

Each file contains one HumanML3D motion ID per line.

- `lcr_wholeset_reimplementation_unsafe.txt`: an independent reimplementation of the unsafe whole-set split described by the earlier LCR paper; the original split was not released.
- `lcr_official_test_unsafe.txt`: the test-only unsafe split released with the later official LCR implementation.
- `safemo_unsafe.txt`: precomputed SafeMo Engine labels.

Use the official test list only with `--dataset-scope test`. The other lists can be evaluated on `all` or intersected with `test`.

| File | IDs | SHA256 |
|---|---:|---|
| `lcr_wholeset_reimplementation_unsafe.txt` | 2,194 | `322543e751dcf489d78108cdb4bde3bf1899cd91f84a119e6c1d21a5dde9bcee` |
| `lcr_official_test_unsafe.txt` | 304 | `03dd3d493e41a2d46c388c8e9d78a0a3e90ddc0a126905eb5c80a4c9665c8b64` |
| `safemo_unsafe.txt` | 2,618 | `827939bd1e1bb2266643012953ed2fca31c87fa3861fdbf622984e108e836ce4` |

For each evaluation scope, the forget subset is the intersection with the selected unsafe list and the retain subset is its complement.
