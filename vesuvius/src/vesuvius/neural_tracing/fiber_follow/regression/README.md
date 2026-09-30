# Observation-memory fiber regression

Regression uses a continuous follower with main-encoder observation memory,
shared-decoder refinement, and causal segment survival confidence. The image
encoder can use convolutions or 4x4x4 patches. The patch launcher uses only the
coarse patch tokens throughout, with no fine-feature reconstruction or output planes.
See [architecture and supervision](TRAJECTORY_MEMORY.md).

## Training

From `fiber_follow`, using the existing project environment:

```bash
bash scripts/launch_memory.sh
tail -F output/logs/axial_survival_memory_v9_run1.log
```

The default `--encoder conv` architecture is `axial_fiber_memory_v9`. The
launcher refuses an existing destination. Set `RUN_NAME` for another fresh run;
trailing arguments override `regression.train.build_parser` defaults.

To try the convolution-free encoder in a fresh run:

```bash
bash scripts/launch_patch4_memory.sh
tail -F output/logs/axial_patch4_tokens_memory_v9_run1.log
```

The patch launcher defaults to `--batch 1 --microbatch 16 --feature-sequence-length 2`:
a minimum of one supervised decision per update, with eight traces by two
observations per loader chunk. Whole chunks remain intact, so an update can have
more than one supervised decision. Edit `BATCH_SIZE`, `MICROBATCH_SIZE`, and
`WORKERS` (default 8 data-loader workers) near the top of the script, or override
them in the environment:

```bash
BATCH_SIZE=8 MICROBATCH_SIZE=16 WORKERS=8 bash scripts/launch_patch4_memory.sh
```

Decisions still run one at a time; `--batch` controls gradient accumulation.
Both encoders always compile their training operations. `RUN_NAME`, `BANK_PATH`,
and trailing arguments work as in `launch_memory.sh`.

The patch launcher selects `--encoder patch4 --token-only`. It embeds ordered
4x4x4 patches and applies four axial transformer blocks at width 128. The default
crop produces a 30x26x26 grid (20,280 tokens). Generator and survival attention
read this grid plus references and historical memory: 23,545 positions before
masking. Path queries, refinement, segment evidence and memory detail tokens all
sample the same coarse lattice at its physical patch centers. Predictions remain
continuous coordinates. No reconstructed fine volume, fine-feature stencils or
output-plane tokens are used. There are no spatial convolutions.

Token-only runs save architecture `axial_patch4_tokens_fiber_memory_v9`. This is a
fresh-training architecture; it cannot resume the older fine-feature patch model.
Existing `axial_patch4_fiber_memory_v9` and convolutional v9 checkpoints still load
with their original feature layouts. Loading infers the layout from the checkpoint;
explicit conflicting `--encoder` or `--token-only` options are rejected on resume.
Use `--no-token-only` to explicitly train the original reconstruction-based patch
model. v1-v8 checkpoints remain unsupported. Tracing quality needs evaluation.

Attention K/V projections and their compacted valid CUDA rows are reused across
refinement attempts and supplied candidate paths within each decision. They remain
differentiable and are never cached across optimizer updates.

`--activation-checkpointing` controls checkpointing inside axial blocks. Separately,
selected historical encoders are checkpointed by default. The optional
`--no-history-encoder-checkpointing` retains their activations instead of recomputing
them during backward, trading additional VRAM for less computation. The launcher
keeps historical checkpointing enabled; compare memory before disabling it.

Every historical crop is encoded into spatially located appearance tokens. Most
crops use an observation-only, no-grad encoder path: they skip the trajectory
generator, confidence scorer, and refinement. The convolutional encoder reuses
stem features; token-only patches supply coarse features directly. Legacy dense
models reconstruct fine features after reading historical memory. The
endpoint and at most `--feature-history-decisions 2` uniformly sampled earlier
positions receive full supervised predictions. Sampling happens once per stream
in the loader, independently of the model's confidence and label observability.
Unknown supervision contributes zero and is not redistributed.

Each selected decision reconstructs its entire preceding memory history at current
writer weights. All writer transitions are differentiable. Up to three preceding
images are re-encoded with gradients, one per available age band (1–4, 5–16,
17+ observations before that decision). Other observations use detached cached
features, potentially produced at older weights. This selectively trains the
visual encoder from later losses; it is not full-history encoder backpropagation.
Only selected images are retained on the CPU, and each is released after its last
planned use. Finished streams are evicted. No writer graph crosses optimizer
updates and no decision can see future observations.

There is **one task prediction and one loss per selected decision**, including
endpoints. There is no second endpoint replay loss or replay-weight option.
The loss combines geometry, generated-path survival, and eligible candidate-path
survival. By default endpoints receive .75 of a stream's task-loss weight and
selected historical decisions share .25. A single-observation stream, zero
history decisions, or zero history-loss fraction assigns weight one to its
endpoint. Sampling weights preserve the expected direct auxiliary history-loss
budget for fixed model/state; they do not make sparse encoder gradients unbiased.

`--batch 8` now counts **supervised decisions per optimizer update**, not image
observations. The loader gathers at least that many decisions without dropping
rows. Weighted loss sums divide by the actual decision count in the update,
never by observation count or realized loss-weight sum. Thus the loss scale and
optimizer cadence differ from v8. Observation-only collection does not advance
AdamW/EMA or the learning-rate schedule. Logs report both observation and decision
throughput, supervised endpoints, writer replay observations, and selected
historical encoder crops.

`--microbatch 4` and `--feature-sequence-length 2` configure loader chunks of two
traces by two observations. These chunk boundaries do not truncate memory
gradients: each supervised decision reconstructs its prefix independently.
Independent partial chunks can be packed within an update. Observation encoding
is batched; gradient-bearing decisions currently reconstruct one stream at a time.

Workers receive the decision mask before building each image batch. Every row
keeps its original augmentation and stream/holdout checks, but annotation tensors,
foreign-mask rasterization and candidate labels are built only for supervised
rows. Observation-only batches omit those targets; mixed batches zero-fill unused
target rows. Fresh observation rows still query bank geometry to preserve the
coverage feedback used by subsequent lateral sampling.

The trainer assembles one complete CPU decision batch ahead while the current
update runs on the GPU. This overlaps loader waiting without changing chunk order,
update boundaries, loss normalization, or optimizer/EMA cadence. It can retain one
additional update's crops in host memory, on top of the DataLoader's worker queue.

Memory writing and the scorer's final normalization/confidence projection run in
FP32; other CUDA model computation uses BF16 autocast. Sampling coordinates,
survival accumulation, and proposal comparisons remain FP32. Gradient clipping
remains separate for memory (5) and the rest of the model (20).

`train.py` is the sole always-compiled trainer. Its model methods compile in place
with full-graph boundaries for observation collection, supervised prediction,
optional candidate scoring, selected historical encoding, and memory transitions.
The no-grad collection and checkpointed gradient-bearing encoding use distinct
boundaries. B=1 decisions avoid padded prediction rows. Inductor
`emulate_precision_casts` remains enabled. CPU semantic tests can use an explicit
eager graph-capture backend; CUDA checks use real Inductor.

Every supervised decision uses fixed proposal slots, masking retries after
full-horizon acceptance. Inference compacts accepted rows and skips their retries.
Geometry gives the last attempted proposal 75% and shares 25% across earlier
attempts; a sole attempt gets 100%. Generated survival averages actual attempts.
Skipped refinement parameters retain `grad=None` for AdamW. Selection chooses
the longest acceptable prefix, then its endpoint confidence, then earlier attempt.

CUDA BF16/FP16 cross-attention removes padded keys per row and prefers Flash;
projection K/V is reused within each decision. This preserves mask semantics but
changes floating-point reductions and consumes gathered-K/V storage. No promise
of bitwise-identical training trajectories or unchanged trained accuracy is made.

Use actual training data for steady-state throughput. Fork a current v9 checkpoint
whose remaining schedule extends beyond the requested updates:

```bash
PYTHONPATH=../../.. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_feature_training \
  --checkpoint output/axial_survival_memory_v9_run1/last.pt \
  --out output/real_training_benchmark --updates 150 --warmup 50 --workers 8
```

The fork uses the checkpoint's real volumes, annotations, banks, and published
replay caches. Its primary rates include data waiting and normal logging between
updates. It measures the contiguous window after warmup and the final compilation
event; inspect `measured_updates` and extend the run if that window is too short.
Collector/diagnostic/checkpoint boundaries cannot be crossed. Report observations
and completed endpoints per wall-clock second; also compare similar observation
volumes and history lengths because v9 changes the meaning of an optimizer batch.

A separate synthetic structural benchmark measures complete streams, including their selected
historical encoder gradients and optimizer work, without loading a checkpoint:

```bash
PYTHONPATH=../../.. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_sparse_training \
  --out output/sparse_benchmark.json --length 16 --streams 2 --warmup 3 --repeats 10
```

Compare equal observation streams and endpoint throughput; milliseconds/update
alone is misleading when the supervised decision count changes. Synthetic speed
and gradient tests do not establish tracing quality or memory-use accuracy.

## Inputs and sampling

All physical coordinates are trace-grid voxels. The main crop is 120x101x101 at
spacing 0.5. The output has 16 points on forward planes 1 through 16.
Every generator decoder layer reads the complete deep lattice and the dense
features at every lateral pixel on all 16 output planes, during both initial
prediction and refinement. Each plane retains the full 101x101 resolution:
163,216 fine tokens join 39,015 deep tokens plus references and observation
memory. Fine tokens have learned channel projections and physical XYZ position
embeddings. Output planes between input slices interpolate only along depth.
Every scorer layer also reads all full-resolution output planes, alongside
segment-local dense samples and deep/reference/memory tokens. Plane sampling is
shared with the generator; the scorer has independent channel and XYZ projections
and reuses its own attention K/V across generated and candidate paths.
Observed history and the seed condition the crop; annotations only define losses.
Cold starts with a remote seed encode that seed crop once. Warm tracing encodes
only each new head crop and carries observation memory forward.

Direction inputs are `CT, presence, uu, vv, ff, uv, uf, vf`: the six independent
components of an unsigned direction second moment, expressed in each crop's own
frame. Missing directions are zero. Interpolate second moments, not encoded bytes
or signed vectors. CT/presence blur, brightness, noise and dropout do not change
the direction channels. Direction fields come from the selected presence grid.

Matched identity pairs and constructed wrong-fiber histories supply difficult
choice/departure states. Remote observations can establish identity even when the
seed is outside the current crop. Training streams retain observed geometry,
including wrong turns, without using membership labels to filter memory writes.
Unknown identity and censored annotation mask supervision.

Sampling revision 8 emits the causal sparse-decision plan, per-decision encoder
selection, and CPU crop-retention deadlines described above. Every admitted
observation still contributes visual evidence, including wrong turns and unknown
bridges. History losses use the same annotation and censoring rules as endpoints.

The launcher requests matched pairs for .3 of endpoint proposals, with .75 of
pair proposals being recoverable choices (`--decision-choice-fraction`). Both
sides of a choice receive geometry targets; an already-departed side receives
only stopping supervision. Unavailable or holdout-rejected pairs fall back to
ordinary draws. The launcher also caps whole synthetic-switch streams at .15 of
admitted crops. Logs report actual endpoint counts, choice counts, and supervision
weights separately from crop-source percentages, so fallback and stream expansion
are visible.

Bank following has its own endpoint-proposal budget: the launcher's
`--bank-following-probability .2` reserves 20% independently of the 30% matched
decisions. The remaining 50% uses `--fresh-fraction .7`: 35% annotation-fresh
and 15% recent replay overall, before fallback and rejection. Fresh targets come
from actual annotations; covered-parent resampling and synthetic switch histories
still use annotated targets. Generated bank paths enter through the dedicated
following budget or matched decisions. Missing or unsafe bank proposals fall back
to the annotation/replay mix; empty replay strata fall back to annotations.
These are requested endpoint shares, not guaranteed crop percentages.
Sampling revision 5 changes `--bank-following-probability` from a fraction of fresh
draws to an independent endpoint fraction, including when resuming old runs.
Its sum with `--decision-fraction` must not exceed one.

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

Old replay caches remain loadable with their original labels and default unknown
switch metadata. Recollect them to obtain continuous contact locations; older
caches did not store every committed segment. Sampling revision 6 enables these
budgets on resume and changes seeded draw order. It does not establish a tracing
accuracy improvement without a training/rollout comparison.

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

## Validation

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. \
../../../../.venv/bin/python -m pytest \
  tests/test_survival_confidence.py tests/test_detailed_memory.py \
  tests/test_trajectory_memory.py tests/test_recurrent_refinement.py \
  tests/test_feedback_refinement.py tests/test_fixed_training.py \
  tests/test_feature_correctness.py tests/test_sparse_memory_training.py tests/test_direction_inputs.py \
  tests/test_decision_training.py tests/test_sampling_balance.py \
  tests/test_bank_failures.py tests/test_sampling_ratios.py \
  tests/test_output_plane_features.py \
  -q -o cache_dir=/tmp/fiber-survival-pytest
```

Checks cover suffix independence, segment evidence, full observation access,
first-failure/censoring losses, generator gradient isolation, memory retention,
chronological replay, optimizer updates, checkpoint round trips and tracing.
Feedback checks cover early exit, shrinking active batches, absolute path
replacement, score/geometry gradient isolation, per-attempt losses and selection.
Fixed-training checks cover graph reuse across row counts and acceptance outcomes,
replay autograd contracts, packed stream gradients, and skipped AdamW updates.
Output-plane checks cover full lateral coverage, physical coordinates, fractional
depth interpolation, access through every generator layer and gradient isolation.
They do not establish trained tracing accuracy or confidence calibration.

## Bounded compute and VRAM comparison

This uses full-size synthetic images, complete observation streams, optimizer
updates and the normal compiled BF16/FP32 policy. It excludes data loading and
tracing quality. Use fresh result paths and run the variants sequentially:

```bash
python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_sparse_training \
  --encoder patch4 --token-only --length 24 --streams 2 --warmup 2 --repeats 3 \
  --out /tmp/token_checkpoint_on.json
python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_sparse_training \
  --encoder patch4 --token-only --no-history-encoder-checkpointing \
  --length 24 --streams 2 --warmup 2 --repeats 3 --out /tmp/token_checkpoint_off.json
```

Peak allocated memory measures live PyTorch tensors; peak reserved memory also
includes allocator-held space. Memory cost depends on historical crops retained
for each decision and is not a universal constant for all stream lengths.

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
