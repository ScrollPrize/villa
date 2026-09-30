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
loader workers, and a fresh `axial_patch4_tokens_slabs_v10_run1` destination.
The convolutional launcher defaults to `axial_survival_slabs_v10_run1`.
Existing output directories are never overwritten.

Architecture identifiers are `axial_fiber_slabs_v10`,
`axial_patch4_fiber_slabs_v10`, and `axial_patch4_tokens_fiber_slabs_v10`.
Fresh training and replay v6 are required. Recurrent checkpoints and old replay
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

Training compiles supervised prediction and supplied-candidate scoring. Valid
slabs are gathered and encoded before those fixed-shape compiler boundaries.
The current-image attention K/V projections are reused within the decision.
CUDA uses BF16 autocast with FP32 survival and coordinate policy; the slab encoder
and other parameters have independent gradient clipping (5 and 100).
`--activation-checkpointing` still controls the main axial blocks.

Logs include valid slabs per decision, historical age, actual slab-grid overlap
with the current crop, loader slab seconds, encoder seconds (CUDA events on GPU),
and historical encoder gradient norms. Loader seconds are summed worker time,
not end-to-end wall latency. Decision throughput and peak allocated VRAM are
reported separately. Current crops per update now equal supervised decisions.

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
fork of a v10 run using its actual data. The synthetic benchmark excludes I/O;
compare equal hardware, precision and supervised decision counts.

See [recorded validation](SLAB_VALIDATION.md) for measured results and limits.

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
