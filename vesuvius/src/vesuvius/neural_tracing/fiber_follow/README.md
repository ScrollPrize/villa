# fiber_follow

Layout:

- `flow_matching/`: the flow-matching follower (model, losses, `train`, and
  `collect`/`infer` entry points). Launch with `bash scripts/launch_single_path.sh NAME`.
- `regression/`: an independent directly supervised alternative
  ([README](regression/README.md)). It uses fine level-0 CT, live historical CT/path slabs,
  continuous path prediction with shared-decoder refinement, and causal survival confidence.
  Launch with `bash scripts/launch_memory.sh` (`RUN_NAME` selects the destination).
- `shared/`: volume access, sampling, data, tracing, evaluation, diagnostics and
  run-loop code used by both. `shared/` never imports `flow_matching/` at module
  load; the model families pass their checkpoint loaders to `shared.collect` and
  `shared.infer`.
- `output/` stays at this level for both trainers.
- `visualization/`: configurable interpretation atlases for the current patch4
  slab model ([usage and reproduction](visualization/README.md)).

The current slab follower and its training workflow are documented in
[regression/README.md](regression/README.md). The remaining model description
below documents the older `single_path_flow_v11` implementation.

## Model and tracing

- CT level 1 (`ct_grid_scale=8`), presence, and rendered observed history.
- One 176 × 96 × 96 crop at spacing 1: 128 voxels behind, 47 ahead,
  ±47.5 laterally. All 128 one-voxel history observations condition the model.
- Shared encoder-decoder widths (24, 64, 128), group norm, geometric-history
  FiLM. History tokens include coordinates, age, validity and local features;
  a separate support flag distinguishes out-of-crop observations.
- Six pre-norm blocks, width 256, eight heads, FFN width 1024. Bidirectional
  self-attention joins history and future tokens. Cross-attention reads spatial
  tokens pooled another factor of two from the deepest features of this crop.
- New training runs start each 16-point future curve from independent standard
  Gaussian normalized lateral residuals. The fitted per-plane residual scales
  convert those residuals to voxel coordinates. Four
  midpoint steps require eight velocity evaluations. A ninth evaluation at
  the final coordinates and `t=1` supplies confidence. Local patches refresh
  during refinement; static image/history features and image keys/values cache
  within a decision.
- The public output is `points [B,16,3]`, `confidence_logits [B,16]`, and
  monotone `confidence [B,16]`. By default each decision samples one curve, with no ranking
  or averaging. A private RNG stream per directed trace seed makes its initial
  noise reproducible independently of batching or other traces stopping. Numerical
  model outputs can still vary across devices and batch shapes. No forecast is
  carried into the next decision.
- Commit at most `n_commit` points per decision (default 8, at most the 16-point
  horizon), respecting the six-voxel first connection limit, bounds and loops. A
  rejected decision ends the trace; nothing is forced. History coordinates and
  previous commits remain fixed.

Flow matching retains 64 stratified time/noise draws per state. Residual scales
are fitted from 2,048 masked training states. Confidence labels describe the
actually generated curve at tolerance 1.5, using the same four
midpoint steps as inference, diagnostics and replay collection. The curve is
sampled once per training state and reused for its labels and confidence loss;
generated coordinates and labels are
detached, while the final evaluation trains the shared features and denoiser.
Confidence loss is half masked BCE over the commit window (prefixes 1 to
`--n-commit`, default 8) and half over all 16.
Its coefficient ramps from zero to one over 2,000 optimizer updates. Labels follow the
shared state contract (`shared/state_labels.py`): unknown annotation endings and
uncertified states are censored, terminal states have confidence negatives and no
localization loss. GT history is diagnostic data only.

## Data and first run

Both trainers sample by the shared task budget (`TaskBudget`, `--task-share`); this
model has no live continuation or synthetic failures, so their shares go to fresh
simulated traces (70% fresh, 30% DAgger replay classes). Fresh states are simulated
tracer decisions (`make_sample`); missing replay classes fall back to fresh states and
are logged. See the [direct follower](regression/README.md#aligned-training-and-tracing)
for the shared state contract, collection and replay rules.

Annotations are cleaned once at load (`shared/annotation_repair.py`, Paris 4 and AFV):
short kinks (a tick, V, hook or small step that turns more than 30 degrees within one voxel
and then resumes its direction) are replaced by the shortest smooth bridge; real corners and
hairpins, whose direction changes, are kept, and everything else is unchanged. Paris 4 fibers
whose annotation doubles back on itself are kept out of training (validation is unchanged).
On Paris 4 this repairs about 1,500 kinks (0.4% of annotated length) and keeps 14 fold-back
fibers out of training; the CT validation is in `output/trace_noise_realism_20261002/`.
Geometry hashes change accordingly, so replay collected before the repair is rejected. The
older frozen seed manifest (`output/single_path_v11_preparation/seeds.json`, used by
`launch_single_path.sh`, `launch_regression.sh` and `scripts/evaluate_single_path.py`) was
built on unrepaired geometry and no longer matches; `--dataset-config` runs build their
validation manifest from the loaded fibers.

Replay stores observed states, their trace facts and original-fiber correspondence,
independently of prediction shapes. Training uses the latest completed on-policy caches
(default: four), rotating older caches out as new ones arrive. Every training
replay draw is relabeled and checked again before recropping. The original 48-voxel holdout
position guard and complete crop/history/target exclusion remain in force.

From this directory, using an existing environment with the project dependencies:

```bash
export PYTHONPATH=../../..
bash scripts/launch_single_path.sh single_path_v11_run1
```

The frozen seed manifest (`output/single_path_v11_preparation/seeds.json`)
already exists in this workspace. A new run samples fresh states until current
replay is available, unless initial caches are supplied with `--onpolicy`.
Stop a run from either trainer with `bash scripts/stop.sh NAME`.
The launcher benchmarks
before starting training. If microbatch 2 exhausts GPU memory, rerun with
`MICROBATCH=1`; effective batch remains eight and the crop remains full sized.
The benchmark reports GPU peak allocated/reserved memory, training throughput and
complete tracing latency, including data preparation and final confidence.

To adapt existing v11 weights to the aligned sampler in a new run:

```bash
bash scripts/launch_single_path.sh v11_gaussian_20260925 \
  --init-from output/v11_compiled_20260925_150551/ckpt_004000.pt --lr 1e-4
```

`--init-from` loads EMA weights and fitted residual scales, verifies the volume,
geometry and evaluation split, and starts a fresh optimizer and schedule. New
runs default to `--sampler-mode gaussian`; the mode is saved in checkpoints.
Checkpoints without that field retain their historical zero initialization.
`--resume` preserves the checkpoint's mode and rejects a conflicting override;
use `--init-from` to change it. `--sampler-mode zero` remains available for
controlled comparisons. The launcher writes a separate matching preflight per run.

### Optional general passage scorer

To train from scratch with one deterministic proposal and four Gaussian alternatives:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/launch_single_path.sh flow_scorer_run1 \
  --sampler-mode zero --scorer passage --gaussian-candidates 4 \
  --n-commit 8 --steps 50000 --lr 1e-3 --workers 6
tail -f output/logs/flow_scorer_run1.log
```

The launcher runs the full-size GPU preflight in the foreground, then starts
training in the background. The existing preparation artifacts are reused.
Preflight and training raise the soft open-file limit before opening mmap
caches or starting loader workers, using the same shared helper as the direct
trainer. The system hard limit is unchanged; workers inherit the
raised limit. This avoids descriptor exhaustion from cached mapped chunks.
Both generator and scorer learn from scratch; the Gaussian flow-matching loss
retains its 64 time/noise draws and the image encoder is shared.

`PassageScorer` in [`passage_scorer.py`](passage_scorer.py) is independent of the
generator, encoder, candidate count, and horizon. It accepts path evidence,
context, geometry, and a frontier. It uses dedicated path attention and prefix
mean/max pooling. The flow adapter supplies a 27-location feature
stencil, deep image features/support, final history-conditioned flow tokens,
and path geometry. This uses the flow model's existing CT/presence/history
features.

The mixed pool always includes a zero-start candidate in slot zero. All five
generated paths receive masked prefix-correctness BCE, half over prefixes 1–8
and half over the full 16-point horizon, with the existing confidence ramp.
There are no teacher candidates or pairwise loss. Paths and labels are detached;
scoring gradients train the shared image/context features and dedicated scorer.
Selection uses prefix confidence through `--n-commit`, excludes candidates with
an invalid first connection, and prefers slot zero on ties. The selected path
then uses the existing confidence gate and partial-commit policy. Training logs
report deterministic/selected/oracle correctness and rescues versus spoiled
deterministic answers. Training refinement diagnostics follow candidate zero;
inference refinement plots follow the selected candidate.

`--scorer legacy` remains the default and preserves old checkpoints exactly.
`--scorer passage --gaussian-candidates 0` changes only the confidence head.
Extra Gaussian candidates require `--sampler-mode zero --scorer passage`.
`--candidate-selection stop_fallback` retains the eight-point winner whenever
it can advance at the tracing threshold. If it would stop, it selects another
eligible candidate with the longest acceptable prefix, then highest confidence
at that prefix (candidate zero wins remaining ties). It never lowers the gate
threshold or recovery limit. Old checkpoints default to `prefix`; the policy
can be explicitly changed on resume and is saved in subsequent checkpoints.
Training ranking metrics remain threshold independent. Gate metrics use the
actual fallback policy and report how often the alternate first point is correct.

Each collection traces one directed episode per distinct fiber (`--dagger-fibers`,
default 64) from a saved coverage cursor, without exploration. Replay publication logs
include the collected class/event supply, coverage and skipped seeds. Each resume writes
its effective options to `resume_STEP.json` alongside the original run configuration.

Scorer type, pool size, and selection horizon are saved in checkpoints and
verified by preflight/resume; changing scorer architecture requires fresh training.
Mixed-pool tracing has reproducible per-trace random streams and calibration
defaults to three sampling seeds, even though the primary candidate starts at zero.

Validation for this option: 117 CPU tests passed (four CUDA tests skipped in the
sandbox), plus a full-crop compiled RTX 5090 preflight and two real-data training
updates from scratch with a reloadable optimizer/RNG checkpoint. The mixed-pool
preflight measured 13.48 GiB peak allocated, 14.29 GiB reserved, and 19.42
samples/second over two measured effective-batch-eight updates after one warmup.
These are cached-batch timings, not end-to-end training throughput or an accuracy
result. Artifacts: `output/preflight/passage_scorer_check.json` and
`output/flow_scorer_smoke_20260926/`.

Training defaults: 50,000 optimizer updates; microbatch 2 with four accumulation
steps; AdamW, peak LR 1e-3, weight decay 1e-4, 1,000-update warmup/cosine decay,
gradient clipping at 1, EMA decay .999, CUDA BF16, contiguous convolutions and
compiled CUDA training methods. Compilation preserves eager RNG draws and the
existing backward precision; the first update can take about a minute.
Use `--no-compile` for eager training. CPU training, EMA diagnostics and tracing
remain eager. Preflight uses the selected compilation mode, and resumed runs
compile after restoring weights/optimizer state. See
[PERFORMANCE_COMPILE.md](PERFORMANCE_COMPILE.md) for the measured 32% reduction
in update time and roughly halved allocated GPU memory compared with eager
training using encoder reuse.
The v11 convolution layout was selected by full-crop CUDA measurements. See [PERFORMANCE.md](PERFORMANCE.md) for results,
numerical checks, and commands to compare layouts on another GPU.
A detached curve pass establishes masked-loss denominators across the effective
batch; each microbatch reuses its encoder graph for flow/confidence gradients. Censoring
and departures therefore do not change the objective when switching microbatch
size. Inference still encodes once per decision.
Encoder reuse is enabled by default. Repeated full-crop trials measured about
25% less update time in eager training, at roughly 24 GiB peak allocated GPU memory
(about 12 GiB with compilation). Use
`--no-cache-training-encoding` to re-encode instead when memory is tight.
The launcher benchmarks the selected
mode, and training requires a matching preflight result. See
[PERFORMANCE_ENCODING.md](PERFORMANCE_ENCODING.md) for measurements, correctness
checks and reproduction commands.
CPU batch preparation skips history-distance calculations outside a conservative
box where the float32 result already rounds to zero. Paired real batches were
bit-identical and prepared 2.23× faster; see [PERFORMANCE_BATCH.md](PERFORMANCE_BATCH.md).
Model-only benchmark throughput excludes loading; actual training throughput
also depends on this CPU pipeline and worker transfers.
Checkpoints save every 1,000 updates and at completion. `ckpt_*.pt` and
`last.pt` also hold the optimizer and RNG state; an interrupted run continues
in place with `--resume output/NAME/last.pt` and the same `--name` and options
(residual scales come from the checkpoint, the log is appended, and replay
continues from the caches the run had published). Collector snapshots under
`dagger/` are not resumable. Loader workers restart their own streams, so the
sampled states after a resume differ from an uninterrupted run. The EMA decay
ramps from .1 toward .999 over the first updates. One background collector
refreshes replay every 1,000 updates when idle, using EMA, 64 distinct training fibers,
a 768-voxel cap, the default confidence .5 and no exploration. Diagnostics use the
original monitor fibers at the diagnostic threshold .5 (`DIAGNOSTIC_THRESHOLDS`). Images show observed history,
GT, the final curve, and successive denoising updates.
Diagnostics have private RNG streams and do not advance training's noise stream.
Inference exposes `--sampling-seed`; collection uses its existing `--seed` for
sampling as well as seed selection.

Terminal logs show readable loss, throughput, confidence and refinement tables;
`log.jsonl` retains one complete JSON record per line for analysis. On logging
updates (`--log-every`, default 50, and the final update), `refinement.by_drift`
records commit-window (first `--n-commit` points) lateral Euclidean error at initialization and after
each midpoint update. It reuses the detached training rollout with no extra
model evaluations, and measures the full effective batch before the optimizer
update. These are sampled training states, not a fixed validation set.

Drift bands are `<1`, `1-1.5`, `1.5-2`, `2-3.5`, `>=3.5`, plus `unknown` and
an `all` aggregate. Drift uses the current position's original-fiber GT
correspondence. Departed states are excluded; error uses the same annotated,
crop-observable crossing mask at every step. Each band stores `state_count`,
`known_point_count`, and per-step `error_sum`, `point_count`, `error_mean`, and
`nonfinite_point_count`. Empty means are JSON null (terminal `--`). Pool error
sums and point counts across records before dividing; nonfinite predictions
are excluded from those sums/counts and reported separately. `improved_count`,
`worsened_count` and `comparison_count` have one entry per update relative to
the preceding curve, using per-state mean error on identical known points and
a 1e-6 voxel change tolerance. States with nonfinite errors in either curve
are excluded from that comparison. GT history is used only for diagnostics.

## Evaluation

The frozen manifest retains the original 32 monitor fibers and 48 assessment
fibers. The remaining 44 validation fibers form the final split; monitor fibers
are excluded as well as assessment fibers. Current seed counts are 32 monitor,
96 calibration and 176 final. Geometry hashes and source identities are stored.

```bash
python scripts/evaluate_recovery.py output/RUN/last.pt \
  --fixtures output/single_path_v11_preparation/calibration_recovery.npz \
  --out output/RUN/recovery.json
python scripts/evaluate_single_path.py calibrate --checkpoint output/RUN/ckpt_NNNNNN.pt --out output/RUN/calibration
python scripts/evaluate_single_path.py run --checkpoint output/RUN/ckpt_NNNNNN.pt \
  --policy output/RUN/calibration/selection.json --splits final --out output/RUN/final.json
python scripts/evaluate_single_path.py compare BASE.json output/RUN/final.json
```

Both followers use the shared protocol (`shared/evaluation.py`; the direct follower's
entry point is `regression/evaluate.py`). Calibration uses calibration seeds only and
selects the greatest coverage among thresholds reaching 95% scored precision, or reports
that none qualifies. Runs report the strict first-departure metrics (3-voxel tolerance,
sustained departure, sticky wrong length) with decision and geometric outcomes, by
source and split; `compare` pairs identical seeds and resamples fibers. Fixed-state
recovery fixtures must be regenerated in the current replay schema.

## Verification and current status

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../../../../.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-slab-pytest
```

The retained tests cover the historical slab model and its training loop,
including data preparation, causal replay, inference, diagnostics and CUDA
gradients. Tests for other model families and unrelated utilities have been
removed. See [slab validation](regression/SLAB_VALIDATION.md) for current results.

The original zero-start full-crop RTX 5090 benchmark passed: microbatch 2, effective batch 8,
9.20 GiB peak allocated / 10.02 GiB reserved, 8.85 samples/second (cached-batch
training, including accumulation and optimizer updates), and 82.7 ms per complete
tracing decision over a 64-voxel rollout. There are three measured updates after
warmup. `output/single_path_v11_preparation/benchmark_b2.json` holds raw timings.
GPU access requires execution outside this session's filesystem sandbox.

Sampler-alignment validation: 110 CPU tests passed (six CUDA tests skipped), and
the Gaussian compiled CUDA regression passed separately. The full-crop Gaussian
preflight passed with microbatch 2, effective batch 8 and 64 flow draws, including
complete tracing. Two compiled real-data updates initialized from step 4,000
produced a resumable Gaussian checkpoint; two short traces from that checkpoint
published nine replay states. These are implementation checks; no accuracy improvement from
Gaussian sampling is claimed. CT level 1 losing fine neighboring fiber detail
remains an experimental risk.
