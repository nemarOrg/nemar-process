# The smoke fixture

`smoke.set` is **synthetic**: eight channels, twenty seconds at 250 Hz, written by `scripts/make_smoke_fixture.py` from a fixed seed.
It is a smoke input only, used to check that the runtime image can load a file and run one eegprep step; it stands for no recording and is never a scientific input.
The scientific fixtures belong to the recipes.

- The data are seeded noise plus a slow sawtooth, in microvolts.
- The last channel (`Ch8`) is flat from the fifth second on, so `clean_flatlines` has exactly one channel to remove.
- The file is deterministic: the same bytes on every machine, with the one clock time in a MAT-file header replaced by a fixed line.
  `python scripts/make_smoke_fixture.py --check` fails if the committed file differs from what the script writes.
- sha256: `454324a186b81911dff2013f2fd6e7e19ca11a7f0d854b1451fa4ab908d9588b`

Regenerate it in the runtime environment, then copy the new digest above:

```bash
export UV_PROJECT_ENVIRONMENT=.cache/runtime-venv
uv sync --frozen --no-dev --group runtime --extra cpu
PYTHONPATH=recipes .cache/runtime-venv/bin/python scripts/make_smoke_fixture.py
```
