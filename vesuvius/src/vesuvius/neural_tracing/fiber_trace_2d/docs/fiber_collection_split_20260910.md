# Downloaded fiber collection: length-based split

Output: `/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910/`.
Use `train/<source>/**/*.json` and `test/<source>/**/*.json` (recursive globs).
These are relative symlinks to the unchanged source files under sibling `all/`.

Seed 20260910. Within each of six sources, order complete fibers by SHA256
of `[seed, source, relative_path]`, then select the shortest prefix whose total
polyline arc length reaches at least 5% of that source's total. Length sums
Euclidean distances of successive `line_points`, in source base-voxel units.
No interpolation, rescaling or fiber cutting. Both Paris4 collections are
independent. Random order is independent of enumeration order and other sources.

| Source | Train files | Test files | Test length % |
|---|---:|---:|---:|
| PHerc0139 | 197 | 10 | 5.2901% |
| PHerc0332 | 74 | 3 | 9.5189% |
| PHerc1229 | 77 | 5 | 5.1029% |
| PHerc1451 | 5 | 2 | 6.5622% |
| PHercParis4_eval_fibers | 463 | 24 | 5.0605% |
| PHercParis4_fibers | 130 | 6 | 5.0237% |
| Total files | 946 | 50 | — |

Rounding up means the last whole fiber can overshoot the quota substantially,
as in PHerc0332. Removing the final selected fiber drops every source below
its 5% quota. This is not subset-sum optimization toward exactly 5%.

All 996 JSON files parse with the shared version-3 format parser and have
positive finite lengths. No identical polylines (including reversed order)
were found. There are 74 cross-split branch references: this is a fiber-level
random split, not a spatial or connected-component holdout. Training consumers
must honor membership and must not expand through branch references into test.
Source JSON metadata and original bytes remain unchanged.

`manifest.json` records each path, checksum, length, rank, split and geometry
hash, plus source statistics and branch/duplicate audit. `README.md` reports
results; `SOURCES.md` and `SOURCE_SHA256SUMS` preserve download provenance.

Reproduce from repository root (omit `--apply` for a read-only preview; output
must not exist when applying):

```bash
PYTHONPATH=/home/hendrik/villa2/vesuvius/src /home/hendrik/.venv_las/bin/python \
  /home/hendrik/villa2/vesuvius/scripts/split_fiber_collection.py \
  /ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/all \
  /ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910 \
  --apply
```

Tests: `PYTHONPATH=/home/hendrik/villa2/vesuvius/src /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/scripts/test_split_fiber_collection.py` (7 passed).
No training config is switched to the collection by this task.
