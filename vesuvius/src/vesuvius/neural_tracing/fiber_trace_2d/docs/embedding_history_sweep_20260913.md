# Embedding history falloff sweep

The experiment changes embedding reference spacing, not geometric tracing step.
For reference age `d` and history window `W`, use normalized weights:

- uniform: `1` (default, existing behavior).
- linear: `max(0, 1-d/W)`.
- Gaussian: `exp(-2*(d/W)^2)` (sigma `W/2`).

All truncate beyond W. Cost is the weighted mean of `1-cosine`, multiplied by
the embedding weight and added to existing costs. Zero total weight gives zero
penalty. History stays causal and per hypothesis; no GT enters the tracer.

Native flag: `--embedding-history-falloff uniform|linear|gaussian`.
Other flags remain `--embedding-history-spacing-base-voxels`,
`--embedding-history-window-base-voxels`, `--embedding-history-count`.

Run the configurable experiment through the approved launcher:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-trace-sweep
```

Settings are in `vesuvius/scripts/fiber3d_experiment.json` under
`embedding_trace_sweep`. Uses frozen raw update5400, hard PHerc1299 fiber, both
directions, and existing signed cache. Shared cached-only readiness errors on
missing coverage, never silently supplies fill. The initial audit found35/36 new
variants lacked coverage. Optional `fill_missing` therefore uses the existing live
inference implementation in the original namespace under one non-exclusive20GiB
GPU/16CPU manager grant. Frozen checkpoint and implementation identity must match.
Training is not stopped, but cooperative resizing briefly reinitializes its workers.
Only missing predictions are generated; per-run inferred tile counts are recorded.

Grid: spacing8/16/32, window32/64/128/256, three shapes, weight0.25/count64;
off and original32/256/count8 controls. Sequential native processes, two case
workers each, decoded cache2GiB, 180s limit for cached-only replay; gap filling
uses the existing live request timeout300s. Rank only complete replays by
fewest bidirectional failures; stable ties retain the earlier variant. Repeat best.
This is exploratory tuning on one selected difficult fiber, not held-out evidence
of overall quality. Wall timings include concurrent training and cache warmup.

Validation: native tests2/2 and Python106/106 passed via
`run_fiber3d_cooperative_tests` (20260913_154610_537245; Python45.73s).
GCC13 QuickBuild, `-O3 -g1 -march=native`, build concurrency4; no fast-math.

## Completed results

Frozen raw update **5400 / 691,200 samples**, checkpoint SHA256
`7ea0f9c89e4bd4ce8c48dbae920b7ed24c810c178ab34576767e30b930057b35`.
Fiber: `/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910/test/PHerc1229/lt_20260825T075316065_000069.json`; source folder PHerc1229 maps to canonical
PHerc1299/20260309130042. Both directions, 35.725227 mm per direction.

| Setting | Failures (lower better) | Mean segment length (mm) |
|---|---:|---:|
| Embeddings off | 96 | 0.729086 |
| Original embedding: spacing32/window256/uniform/count8 | 109 | 0.643698 |
| Best: spacing8/window256/Gaussian/count64 | **88** | **0.793894** |

Best reduces failures by **8.33%** versus off. Mean segment length increases
8.89%; on this fixed-length fiber these are related metrics, not independent evidence.
Gaussian sigma128, all distances in base voxels; embedding weight remains0.25.
This changes reference spacing, **not the geometric tracing step**.

### All 36 combinations

Cells are total bidirectional failures. All runs completed.

| Spacing (base voxels) | Window (base voxels) | Uniform | Linear | Gaussian |
|---:|---:|---:|---:|---:|
| 8 | 32 | 102 | 96 | 98 |
| 8 | 64 | 98 | 97 | 100 |
| 8 | 128 | 95 | 100 | 99 |
| 8 | 256 | 103 | 105 | **88** |
| 16 | 32 | 101 | 107 | 104 |
| 16 | 64 | 105 | 92 | 99 |
| 16 | 128 | 100 | 107 | 99 |
| 16 | 256 | 98 | 102 | 103 |
| 32 | 32 | 102 | 102 | 102 |
| 32 | 64 | 97 | 96 | 102 |
| 32 | 128 | 99 | 99 | 97 |
| 32 | 256 | 109 | 104 | 102 |

### Reuse, timing and reproducibility

- 388 additional inference tiles across36 variants;23 variants needed none.
  Missing coverage was filled under the cooperative non-exclusive grant.
- Per-variant wall time min/median/max **40.65/48.82/66.66 s**.
  These are different parameter settings with different missing coverage, not
  repeated identical performance trials. CPU tracing used two case workers;
  gap inference used one GPU, while training continued.
- Winner first pass56.43s (12 inferred tiles); cached-only repeat38.99s (zero).
  Full replay hash matched exactly:
  `99c46419a2fc902453de169bfb92ac8a8308eaac3220baa394ac6f2382561ab0`.
- After cache expansion, off and original controls again matched their original
  full replay hashes and96/109 failures; original repeat also matched exactly.
- Main sweep manager grant released; training is the only active manager task.
- Defaults unchanged. Selected on **one difficult fiber**; improvement needs
  validation on other fibers before adopting as default.

Raw reports and replay outputs:

- `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_trace_sweep_20260913_v4`
- Final controls: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_trace_sweep_controls_20260913_v1`
- Earlier v1–v3 attempts retained for coverage/API failure inspection.

For another sweep, set a **new output directory** in the runner configuration;
existing output directories intentionally fail rather than overwrite results.

## Weight neighbor search (same day)

Set `weight_search: true` in the same approved runner configuration. Fixed
spacing8/window256/Gaussian/count64, same frozen raw5400 and fiber. Start at0.25,
evaluate both adjacent nonnegative0.05 weights, move only for fewer failures;
ties retain center (equally improving neighbors prefer lower weight). Stop at a
local minimum. Incomplete replays abort, rather than imply no improvement.

| Weight | Bidirectional failures | Wall seconds |
|---:|---:|---:|
| 0 (off control) | 96 | 3.60 |
| 0.20 | 98 | 50.81 |
| **0.25** | **88** | 38.72 |
| 0.30 | 103 | 41.96 |
| 0.25 exact repeat | 88 | 44.51 |

Both neighbors were worse, so no further hill-climbing steps were warranted.
This establishes only a **local minimum at0.05 resolution on this fiber**, not
a global best weight. All runs inferred zero tiles. Original-history control
also reproduced109 failures; winner full replay hash is identical to the earlier
history sweep. Per-direction winner failures37/51. Timings are single runs of
two CPU trace workers with concurrent training, not isolated speed benchmarks.

Results: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_weight_sweep_20260913_v1`.
Tests:111 Python passed40.70s and two native suites, approved cooperative runner
`20260913_162317_150528`. No production default changes.

## Full history tuning at weight0.30

Same frozen raw5400 /691,200 samples, bidirectional hard fiber, history count64,
native binary/build and inference cache as above. Only grid weight changed.
The configurable runner now accepts `grid_weight` (default0.25); original
off/0.25 controls remain fixed. With `weight_search:false`, all36 grid rows must
complete and only grid rows are eligible to win. No further weights were tried.

| Comparison | Best history (spacing/window/falloff) | Failures | Mean segment (mm) |
|---|---|---:|---:|
| Off | — | 96 | 0.729086 |
| Weight0.25 | 8/256/Gaussian | **88** | **0.793894** |
| Weight0.30 | 8/64/Gaussian | 93 | 0.752110 |

0.30 best has5 more failures than0.25 best, but3 fewer than off.
At identical histories,0.30 improves11, ties1 and worsens24 of36 compared with0.25.
These are correlated trials on one fiber, not statistical/generalization evidence.
Keep0.25/8/256/Gaussian as the best measured setting; production defaults unchanged.

### Side-by-side failures (lower better)

| Spacing | Window | Uniform .25 | Uniform .30 | Linear .25 | Linear .30 | Gaussian .25 | Gaussian .30 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 102 | 110 | 96 | 99 | 98 | 102 |
| 8 | 64 | 98 | 109 | 97 | 103 | 100 | **93** |
| 8 | 128 | 95 | 96 | 100 | 103 | 99 | 101 |
| 8 | 256 | 103 | 100 | 105 | 102 | **88** | 103 |
| 16 | 32 | 101 | 97 | 107 | 110 | 104 | 101 |
| 16 | 64 | 105 | 111 | 92 | 101 | 99 | 104 |
| 16 | 128 | 100 | 105 | 107 | 104 | 99 | 106 |
| 16 | 256 | 98 | 103 | 102 | 102 | 103 | 97 |
| 32 | 32 | 102 | 100 | 102 | 104 | 102 | 100 |
| 32 | 64 | 97 | 98 | 96 | 99 | 102 | 106 |
| 32 | 128 | 99 | 105 | 99 | 101 | 97 | 101 |
| 32 | 256 | 109 | 96 | 104 | 106 | 102 | 98 |

Both controls matched original full replay hashes. Best0.30 exact cached repeat
matched hash `e747e26ecc9cb2b963180462c3aa50c9bd8772841ddaeaa993404a3b4ef67146`
and93 failures; repeat38.57s, zero new inference.
Grid added56 missing tiles total under nonexclusive16CPU/20GiB GPU0 grant.
Grant released; training remains applied. Two native case workers per variant;
grid wall min/median/max37.27/46.58/62.79s, sum27.96min.
These timings include concurrent training and variable cache misses, not isolated
identical-input repetitions.

Run command unchanged:
`/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-trace-sweep`.
Output:
`/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_history_weight030_20260913_v1`.
Use a fresh output directory for another run; do not overwrite this evidence.
Tests:111 Python passed34.58s, native2/2 via approved cooperative runner
`20260913_163117_386059`; `git diff --check` clean.
