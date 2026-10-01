# Live historical slab regression

The direct follower predicts continuous paths and causal survival confidence from
one current crop and up to eight live historical slabs. The main encoder, local
128-point history, 16-point forecast and adaptive refinement policy are unchanged.
See [architecture and supervision](TRAJECTORY_MEMORY.md).

## Training and tracing

Use the existing project environment from `fiber_follow`:

```bash
bash scripts/launch_patch4_memory.sh
# Or the convolutional main encoder:
bash scripts/launch_memory.sh
```

Set `RUN_NAME`, `BATCH_SIZE`, `MICROBATCH_SIZE`, `WORKERS`, and `BANK_PATH` as needed.
The patch launcher defaults to 16 decisions per update and microbatch, eight
loader workers, and a fresh `axial_patch4_overlap_tokens_slabs_v11_run1` destination.
The convolutional launcher defaults to `axial_survival_slabs_v10_run1`.
Existing output directories are never overwritten.

When resuming, `--lr` may change the base learning rate while retaining AdamW
moments, EMA, and the existing warmup/cosine schedule position. Omit
`--reset-optimizer` to preserve that state; the active LR includes cosine decay.

To prepare a separate continuation with more refinement stages, use a completed
training checkpoint as the source of all effective settings:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.refinement_resume \
  --checkpoint output/patch4_run3/ckpt_017000.pt \
  --name patch4_run3_refine3 --refinement-steps 3
```

This only prepares the new directory; `migration.json` records the exact trainer
command to launch. It preserves weights, EMA, RNG, update count, LR schedule and
existing AdamW moments. Added stage embeddings copy the last learned stage;
their moment rows start at zero, retaining the embedding tensor's Adam step.
The frozen recovery fixture is copied unchanged and published replay caches are
referenced from an independent replay index. The source run must be quiescent
while preparing the fork. Existing destinations, zero-stage source models, and
nonstandard optimizer layouts are rejected. As with ordinary resumes, loader
random streams restart. No source config defaults are used.

Architecture identifiers are `axial_fiber_slabs_v10`,
`axial_patch4_overlap_fiber_slabs_v11`, and `axial_patch4_overlap_tokens_fiber_slabs_v11`.
Patch4 now uses a learned 6x6x6 convolution with stride 4 and padding 1, after
padding the high image edges to multiples of four. Adjacent neighborhoods overlap
by two voxels. The 30x26x26 token grid and centers at offset 1.5 input samples
are preserved; dense reconstruction still uses 4x4x4 output cells.
Overlapping patch models require fresh training; v10 nonoverlapping patch
checkpoints are rejected rather than silently reshaped. The conv model remains
v10, and replay v6 remains compatible. Recurrent checkpoints and older replay
are rejected; there is no weight migration or memory compatibility interface.
The existing regression `train`, `collect`, and `infer` module entry points remain.

`--batch` is the minimum number of independent supervised decisions per optimizer
update; `--microbatch` is the number loaded and predicted together. Complete
microbatches are retained, including both members of matched pairs. Every decision
gets equal weight. Geometry, generated survival and supplied-candidate survival
keep their coefficients (1, .5, 1). There are no observation-only steps, streamed
loss budgets, writer replay or cached historical main-encoder features.

The current crop remains 120x101x101 at spacing .5, with CT/presence and optional
six direction moments. History reads only CT plus a rendered observed-path
heatmap: neither presence nor direction volumes are read for slabs. Slabs are
8x65x65 at spacing .5, with depth offsets -2 through +1.5 tracing voxels.
Selection uses actual observed arclength: seed, up to four uniformly spaced older
observations, then available anchors 96, 64 and 32 voxels behind the head. All
valid observations are at least 32 arclength voxels apart, including the seed.
Unavailable slots are padded and perform no image reads or convolution work.

Training samples a complete synthetic prefix before truncating the main model's
local history, retaining short/no-history draws. Matched pairs and synthetic
wrong-turn histories retain their actual constructed paths, including bridges.
Annotations affect labels only. Every selected slab is checked against the
training holdout before images are loaded; either unsafe member rejects a pair.
Replay and resumed recovery use saved committed prefixes, without connecting a
remote seed to a truncated local history.

The shared slab encoder uses three residual convolution stages, widths 8/16/32,
and strides (1,2,2), (2,2,2), (2,2,2). Each valid slab contributes 162 spatial
feature tokens, projected to model width (128 by default). Position, pose, age,
slot and seed role are embedded inside the encoder. Generator and scorer have
separate residual history attention shared across their respective decoder
layers. History features stay attached and are reused across all attempts and
supplied candidates within a decision. Fully masked history contributes zero.

Training compiles supervised prediction, supplied-candidate scoring, and losses. Valid
slabs are gathered and encoded before those fixed-shape compiler boundaries.
Current-image attention uses batched, padding-masked SDPA with cuDNN preferred
on CUDA. It needs no per-row key compaction or dynamic key dimensions.
Current-image and historical attention K/V projections remain attached and are
reused within the decision, including supplied-candidate scoring.
CUDA uses BF16 autocast with FP32 survival and coordinate policy; the slab encoder
and other parameters have independent gradient clipping (5 and 100).
`--activation-checkpointing` still controls the main axial blocks.
FP32 feature sampling preserves physical coordinate interpolation; the final
survival projection and log-space accumulation preserve stopping precision.
Inductor's BF16 rounding setting is local to these compiled functions.

Logs include valid slabs per decision, historical age, actual slab-grid overlap
with the current crop, loader slab seconds, encoder seconds (CUDA events on GPU),
and historical encoder gradient norms. Loader seconds are summed worker time,
not end-to-end wall latency. Decision throughput and peak allocated VRAM are
reported separately. Current crops per update now equal supervised decisions.

### Measuring CPU data preparation

Use the saved run configuration to benchmark actual image, bank, target and
augmentation work without starting another model or modifying training data:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  ../../../../.venv/bin/python scripts/benchmark_loader.py \
  --config output/patch4_run3/config.json --workers 0 --warmup 2 --batches 20 \
  --hash-batches --out /tmp/loader.json
```

The report includes mean/p50/p95 latency and exact tensor hashes, excluding
timing fields. Hashing is outside the timed region. Add `--profile /tmp/loader.prof`
for cProfile hotspots, or use `--workers 10` to measure concurrent loader delivery.
It uses the config's initial replay paths, a fixed seed, and the live bank;
compare revisions with the same unchanged inputs. Worker profiling and concurrent
delivery are separate measurements. Neither measures GPU transfer or optimizer time.

CPU segment heatmaps fuse the original float32 distance arithmetic in Numba
(without fast math) and retain Torch's exponential. Bank masks compute coordinates
only at occupied cells. Bank draw weights are cached until the next published
manifest, and exact-distance queries defer the dense seed index until mining
actually requests it. These changes preserve sampling, crops, labels and model
precision; checkpoint formats and resume options are unchanged. As before,
resuming restarts loader RNG streams rather than reproducing uninterrupted draws.

The September 30, 2026 check against `3a7658e02` used `patch4_run3`'s
`neighbor_samples_r0_32_l80_160_v2` bank, S1 CT level 0 and configured fiber
annotations. Both versions ran alongside the original training process on an
Intel Core Ultra 7 270K Plus, with Python 3.14.4, PyTorch 2.12.1+cu130,
NumPy 2.4.6 and Numba 0.66.0 (normal cached JIT, `fastmath=False`). The command
above measured 20 microbatches of eight after two warmups, with seed 0:

| CPU preparation, seconds/batch | Before | After |
| --- | ---: | ---: |
| Mean | 3.077 | 1.746 |
| Median | 2.197 | 1.060 |
| p95 | 6.099 | 5.813 |

All 860 non-timing tensor hashes matched across those 160 decisions. These are
single-worker measurements under concurrent training load, including occasional
host-memory stalls; they are not end-to-end training throughput. Reports and
profiles are in `output/loader_speedup/`. In the initial ten-batch cProfile
comparison, unused dense seed-index construction consumed 13.0 seconds and
full-grid construction 6.2 seconds; both were removed from those training paths.
Segment rendering fell from 3.0 to 0.54 seconds before the additional draw-weight
cache. Validation: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/patch_loader_pytest`
reported 279 passed, 15 skipped, one deselected and ten passing subtests. GPU
tests were skipped in the sandbox; the running job provides the live GPU check.

`patch4_run3` resumed at step 1,000 with model/EMA/AdamW/RNG state intact and
the same 10 workers, microbatch 8, batch 16 and one refinement step. On the RTX
5090, the original updates 51–1,000 versus resumed updates 1,051–1,200 gave:

| Live measurement | Before | After |
| --- | ---: | ---: |
| Mean data wait / update | 786 ms | 130 ms |
| Median data wait / update | 780 ms | 106 ms |
| p95 data wait / update | 1,023 ms | 171 ms |
| Overall crops / second | 16.38 | 44.09 |

Wait statistics summarize 50-update logging windows (19 before, three after),
excluding the initial startup windows. Throughput includes the full logged wall
time. This is an 83.4% wait reduction and 2.69x throughput, with live sampling
variation and a shorter post-resume observation period. See
`output/loader_speedup/live_comparison.json` and `handover.json` for the measured
intervals and exact resume command. The original checkpoint is retained at
`output/patch4_run3/ckpt_001000.pt`.

### Second loader optimization round

The next round, measured against `a0d3a940b` with the same hardware and run
configuration, removes more repeated CPU work:

- Rotate the 65,536 possible direction-byte pairs once per crop, then interpolate
  through that lookup table without expanding the source block into six fields.
- Sample identical crop geometry once per batch. Matched decisions still receive
  independent output storage, historical slabs, augmentation and labels.
- Batch exact nearest-polyline queries, retaining the original arithmetic and
  unsorted tree traversal for identical tie breaking. Large queries are bounded
  to 256 points per batch.
- Cache resampled bank-path lengths alongside the shard geometry.

Sequential CPU-only runs of these algorithmic changes alongside training measured 40 microbatches of eight
after four warmups, with seed 0 and the config's initial (empty) replay list.
Both used one loader thread, normal cached Numba JIT without fast math, and no
profiler in the timed comparison:

| CPU preparation, seconds/batch | `a0d3a940b` | Optimized |
| --- | ---: | ---: |
| Mean | 1.887 | 1.139 |
| Median | 1.637 | 1.069 |
| p95 | 3.277 | 1.614 |

This is **1.66x loader throughput**, with all **1,720 non-timing tensor hashes
identical** across 320 decisions. No sampling probabilities, image precision,
augmentation or supervision rules changed. The baseline cProfile run attributed
20.2 of 55.1 seconds to direction crops, 8.0 to scalar interpolation, and 7.8 to
nearest-polyline queries. An intermediate profile reduced the latter to 1.8
seconds; use the unprofiled table above for overall timing, since profiling and
concurrent workloads affect the measurements.

From the fiber-follow directory, the CPU benchmark command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  ../../../../.venv/bin/python scripts/benchmark_loader.py \
  --config output/patch4_run3/config.json --workers 0 --warmup 4 --batches 40 \
  --hash-batches --out /tmp/loader-round2.json
```

Reports, profiles, unchanged baseline source snapshots and its benchmark harness
are retained in `output/loader_speedup_round2/`. The full suite passed 287
tests (15 skipped, one deselected, ten passing subtests). The final guard against
reusing mixed-dtype coordinates also passed all 14 direction/crop tests. Validation used
`NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/fiber-round2-pytest`.
GPU tests were skipped in the sandbox. These CPU measurements are specific to
this Ubuntu x86-64 host; the changes introduce no platform-specific dependencies.

The larger microbatch exposed additional allocation and IPC costs. Main images
and historical slabs now use the same direct shared-storage allocation as
PyTorch's default worker collator. NumPy views write into that storage, and the
original shared tensor is returned, eliminating the worker queue's full image
copy. Outside a worker, allocation remains ordinary CPU tensor storage. A real
one-worker loader also matched all 1,720 baseline hashes after this change.

`launch_memory.sh` defaults `NUMPY_MADVISE_HUGEPAGE=0` (overridable). On this
Linux host, large transient NumPy allocations were incurring huge-page compaction
stalls. This process-local setting changes allocation policy, not values or
system-wide settings. A CPU-only 20-batch transfer benchmark, after three warmups,
used the actual `(16,8,120,101,101)` float32 image shape: mean/median/p95 delivery
fell from 506/442/1,023 ms to 282/281/341 ms with direct shared storage. This measures
synthetic filling and IPC, not complete training; see `batch_transfer.json` and
`benchmark_batch_transfer.py` in the results directory. Run the latter with the
project Python and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, adding `--shared`
for direct allocation. The helper uses PyTorch's internal storage API, as its
collator does; storage-preservation and worker-delivery checks cover this boundary.

`patch4_run3` finally resumed from `ckpt_005000.pt` with model, EMA, AdamW and RNG state
restored (283 optimizer entries, no optimizer or LR reset). At the user's request,
the resumed run uses **batch 16 / microbatch 16**, with 10 workers and the other
training options retained. Both step-5,000 evaluations finished before stopping:
diagnostics took 31.98 seconds and recovery took 23.89 seconds; their reports and
images remain in the run directory. `output/loader_speedup_round2/handover_after_eval.json`
records completion evidence, the exact launch command, environment override and PID.
As with every existing resume, loader
random streams restart; changing microbatch grouping also changes sampled batches.
Live throughput therefore reflects both loader improvements and the requested
microbatch change; the fixed-microbatch CPU table isolates the loader changes.

The final live comparison uses 19 pre-work logging windows (950 updates ending
at steps 1,500–2,450, excluding the window containing the step-2,000 evaluation)
and nine post-restart windows (450 updates ending at steps 5,100–5,500). Startup,
periodic evaluation and active benchmarks are excluded. Wait percentiles describe
the 50-update window averages, not individual-update tail latencies.

| Live training measurement | Before this round | Final run |
| --- | ---: | ---: |
| Mean data wait / update | 151.8 ms | 31.0 ms |
| Median data wait / update | 146.9 ms | 29.3 ms |
| p95 data wait / update | 220.6 ms | 38.9 ms |
| Crops / second | 40.35 | 71.89 |
| Peak allocated GPU memory | 6.83 GiB | 13.30 GiB |

This is **79.6% less data wait and 1.78x live throughput**, including the requested
microbatch increase. GPU memory remains within the RTX 5090's capacity. Detailed
windows and the comparison are saved in `output/loader_speedup_round2/live_after.json`
and `live_comparison.json`. Training remained active beyond step 5,500.

## Source sampling and labels

The default requested shares remain 30% matched decisions, 20% bank following,
35% annotation-fresh, and 15% replay before fallback/rejection. Of fresh draws,
20% request covered-parent resampling and 30% request synthetic wrong turns.
75% of requested matched pairs teach recoverable geometry choices; the rest
contrast a departure with legitimate following. The retired stream-crop cap and
auxiliary historical decision budgets no longer alter these decision shares.

`--bank-hard-fraction .5` keeps half of bank proposals uniform. For the other
half, the loader chooses one of three geometry criteria uniformly and selects
the strongest of eight proposals: nearby paths with similar tangents and bending,
curved paths, or converging/diverging neighbors. Comparisons use four-voxel
arclength spacing; distances and curve similarity are independent of vertex
density and trace direction. Following draws still honor unique-path eligibility;
matched decisions and covered-parent draws use all valid relationships. This
changes exposure to the existing trusted bank; it does not change its annotations
or mine previously absent weak-signal examples.

New DAgger collections use the negative and near-negative banks to detect foreign
contact along the actual committed polyline, including between decision heads.
The first intersection with a `.75`-voxel bank-path tube outside the intended
annotation's `1.5`-voxel tube certifies a switch. Both radii are configurable
(`--bank-switch-tolerance`, `--bank-own-tolerance`). Overlapping tubes remain
ambiguous; absent bank coverage cannot establish fiber identity. Contact positions
are continuous segment/capsule intersections, not rounded voxel or head locations.
Once departed, the existing stopping-only supervision remains absorbing.

Replay stores `failure_kind`, `travelled`, `switch_pos`, `switch_distance`,
`switch_decision` (the within-trace decision that committed the contacting segment),
`switch_bank_path` (shard/path index), and `switch_bank_run`. The pre-switch window
retains geometry targets and shares the event metadata; confirmed switched states
have geometry masked. These labels never enter model inputs. Bank provenance is
saved with the collection. `--replay-failure-fraction .5` reserves half of replay
for available generic departures, bank switches, premature stops, endpoint
overshoots, and pre-switch states, equally by category and then by fiber. The other
half balances recoverable drift bands. If one side is empty, the other receives
its budget; fully empty replay falls back to annotation-fresh draws. Logs report
collected categories and realized training endpoint counts.

Replay v6 requires newly collected caches. It stores every committed vertex once per trace,
with causal prefix indices for each decision; old caches are rejected.

Each matched endpoint scores four shuffled paths: the two bank/annotation
continuations and two smooth transitions between them, with varied transition
onsets. Paired observations share candidate geometry and order. Candidate labels
use the same dense tolerance, recovery limit, foreign masks, endpoint semantics
and censoring as generated paths. They are not assigned all-positive/all-negative
labels by candidate identity. Logs count first-segment and later first failures.
The transitions are rejected-path proposals, never geometry targets or certified
bank following paths.

The loader no longer samples contrastive points or emits embedding query tensors.
Lateral resampling uses visible forward bank coverage directly. The obsolete
`--negative-near-fraction`, `--negative-near-distance` and `--negative-lateral-max`
options are removed; bank mining and crop support determine available foreign
geometry. Separate negative, following and continuation banks remain supported.

## Verification and diagnostics

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../../../../.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-slab-pytest

../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.history_learning \
  --device cuda --steps 250 --out /tmp/slab-learning.json

../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs \
  --warmup 2 --repeats 10 --decisions 6 --out /tmp/slab-benchmark.json

# Match the launcher's microbatch, including batched attention and candidate padding:
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs \
  --warmup 3 --repeats 10 --decisions 16 --microbatch 16 --out /tmp/slab-benchmark-b16.json
```

Use fresh output paths. The controlled learning check has two fixed identity
pairs with identical current crops and differing historical CT, one pair requiring
non-seed evidence. It requires correct geometry, acceptance of valid supplied
and generated paths, and rejection of wrong continuations. Rejecting everything
fails. It reports full history, seed only, no history, and shuffled CT with all
metadata and path heatmaps held fixed. This is a learning-capability check,
not evidence of held-out tracing quality.

For a frozen real paired tensor batch from `IdentityObservationBuilder`, run:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.history_diagnostic \
  --checkpoint output/RUN/last.pt --fixture /tmp/paired-batch.pt \
  --out /tmp/paired-history-report.json
```

The fixture must contain complete adjacent matched pairs, candidates and labels;
it can be reused unchanged across checkpoints. Existing held-out rollout and
monitor recovery evaluations remain part of training. `identity_preflight` checks
real-data independent batches, and `benchmark_feature_training` measures a bounded
fork of a compatible run using its actual data. The synthetic benchmark excludes I/O;
compare equal hardware, precision and supervised decision counts.

See [recorded validation](SLAB_VALIDATION.md) for measured results and limits.

## Mixed Paris 4 and AFV training

Run from `fiber_follow/`:

```bash
bash scripts/train_mixed_ct.sh
# Optional CPU check of real crops, supervision and forward/backward for all sources:
/home/sean/Documents/villa4/vesuvius/.venv/bin/python scripts/check_mixed_ct.py
```

The launcher uses [mixed_ct_datasets.json](../configs/mixed_ct_datasets.json):
10% Paris 4, 45% `0175A_5mm_v1.afv`, and 45% `1447_5mm_v1.afv`.
`0175A_SMALLMORE.afv` is excluded. Weights apply to microbatches, preserving
adjacent matched decision pairs. Source counts and actual shares are logged.
The new run uses one CT channel, patch4, three refinement steps, batch/microbatch
16, and initial LR 0.0003. Trailing command-line options override launcher defaults.
It starts a new model; CT-plus-prediction checkpoints have different input shapes.
Modes using presence/direction predictions still require those volumes.

AFV geometry is read lazily from SQLite in native L0 XYZ coordinates. Its embedded
scan identity and the file SHA256 must match the config. The CT sources are
PHerc0175A/20250521115057 and PHerc1447/20250521151220 (8.64 µm native voxels).
AFV trace coordinates use two native voxels (17.28 µm); the existing Paris 4 trace
grid is approximately 16 µm. Main crop spacing is half a trace voxel in both.

Remote CT chunks are fetched on demand into
`datasets/automated_fiber_volumes/ct_cache/uncompressed/`, namespaced by URL and
level. Local Zarr v2 metadata has `compressor: null` and `filters: null`.
Chunks are decoded once, written atomically without compression, then memory
mapped across workers and collectors. Reopening a cached region needs no network;
uncached regions still require access to the source. The disk cache grows with
visited regions and has no eviction policy; each worker has a bounded RAM/mapping
cache. Cache paths are relative to the dataset JSON unless absolute.

Validation reserves deterministic whole fiber IDs independently in each source;
training does not exclude a Z band. Each AFV source reserves exactly 406 fibers
(`validation.count`), for **1,000 held-out fibers total** across all sources. Paris 4
keeps its 124 previously reserved fibers and reserves another 10% of the remaining
641, leaving 577 training and 188 validation fibers. This preserves the existing
neighbor bank's parent compatibility. Its old `val_z` remains mining provenance
only. Validation geometry is excluded from AFV neighbor queries and filtered out
of Paris 4 mined paths, so it cannot reenter through intentional switches,
neighbor following, matched choice/departure pairs, or foreign labels.

Each source supports those neighbor tasks and its own rollout replay. Collection
cycles through all sources, running one subprocess at a time, with separate replay
caches. Only training fibers seed replay, and replay identity manifests must match
the source's training split. Monitor rollout metrics are evaluated and plotted
separately for each source. The legacy recovery fixture and optional long diagnostic
remain Paris 4 diagnostics. Held-out monitor, calibration and final groups are
distinct; seed generation is bounded to 32 fibers per group.

The CPU preflight writes the exact reserved IDs and evaluation manifests under
`datasets/automated_fiber_volumes/splits/`, plus `training_preflight.json`.
Training writes its manifests into the run directory and saves the resolved
dataset config and its digest in checkpoints. Resuming rejects a changed dataset
config or split. Editing weights or holdouts requires a separate experiment.

## Historical v9 checkpointing measurements

The following existing measurements are preserved from before the slab replacement.
They concern the retired recurrent implementation, not v10 configuration options.

A bounded RTX 5090 comparison (Torch 2.12.1, seed 194, two streams of 24
observations, two warmups and three measured updates, two refinement steps)
measured the following for the token-only model. Each update had 48 observations,
six supervised decisions, 14 historical encoder crops with gradients and 90
writer replay operations, exercising all three historical age bands.

| Historical encoder checkpointing | Peak allocated | Peak reserved | Mean update | p50 / p95 |
| --- | --- | --- | --- | --- |
| Enabled (default) | 1.310 GiB | 1.457 GiB | 913 ms | 910 / 921 ms |
| Disabled | 2.967 GiB | 3.195 GiB | 892 ms | 892 / 897 ms |

Disabling it used **1.658 GiB more live tensor memory** in this workload. The
measured time reduction was only 2.2%, from a short sample. This excludes I/O and
is not a tracing-quality comparison. The default remains enabled. Full local
results and exact commands are in `output/token_only_validation/comparison.json`.
