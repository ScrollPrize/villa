# fiber_follow

An independent directly supervised alternative is in [`direct/`](direct/README.md).
It uses fine level-0 CT, coarse backward context, and a direct curve decoder with two bounded local corrections.
Train it separately with `bash scripts/launch_direct.sh NAME`.

The active follower is `single_path_flow_v11`: a jointly denoised future curve,
with optional mixed proposals and passage scoring, trained from scratch.
Historical v10 results and the pre-migration source remain
in `output/single_path_v11_baseline/`; old follower checkpoints are rejected by
training/inference. The independent beam model keeps its checkpoint names.

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
  horizon), respecting the six-voxel first connection limit,
  bounds, loops, stop patience, and bounded collection exploration. History
  coordinates and previous commits remain fixed.

Flow matching retains 64 stratified time/noise draws per state. Residual scales
are fitted from 2,048 masked training states. Confidence labels describe the
actually generated curve at tolerance 1.5, using the same four
midpoint steps as inference, diagnostics and replay collection. The curve is
sampled once per training state and reused for its labels and confidence loss;
generated coordinates and labels are
detached, while the final evaluation trains the shared features and denoiser.
Confidence loss is half masked BCE over the commit window (prefixes 1 to
`--n-commit`, default 8) and half over all 16.
Its coefficient ramps from zero to one over 2,000 optimizer updates. Unknown
annotation endings are censored; confirmed departures have confidence negatives
and no localization loss. GT history is diagnostic data only.

## Data and first run

The mixture is 50% fresh augmentation, 25% permanent recovery bank and 25%
recent replay. Each replay source reserves 10% of draws for confirmed departures;
other draws balance the drift bands <1, 1–1.5, 1.5–2, 2–3.5 and fibers within each
band. Missing strata fall back to fresh states. Logs contain realized source
fractions and stratum counts. A smooth accumulated displacement over a uniformly
sampled 16–64 history voxels augments existing drift, heading, wobble, missing
and truncated histories.

Replay v5 stores observed states and original-fiber correspondence, independently
of prediction shapes. The one-time importer discards predictions, relabels drift,
checks the larger crop/holdout footprint, preserves source provenance, deduplicates
states, and samples up to 20,000 unique eligible states. The fixed bank never
rotates out with the latest four completed caches. Every training replay draw is
relabeled and checked again before recropping. The original 48-voxel holdout
position guard and complete crop/history/target exclusion remain in force.

From this directory, using an existing environment with the project dependencies:

```bash
export PYTHONPATH=../../..
python scripts/prepare_single_path.py
python scripts/import_replay.py output/flow_v10_small_b8_d64/dagger/decisions_*.npz \
  output/replay_bootstrap/*.npz \
  --out output/single_path_v11_preparation/fixed_recovery.npz
bash scripts/launch_single_path.sh single_path_v11_run1
```

Preparation refuses to overwrite a bank and reuses a matching frozen seed manifest.
These preparation artifacts already exist in this workspace. The launcher benchmarks
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
and beam trainers. The system hard limit is unchanged; workers inherit the
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
The v11 convolution layout was selected by full-crop CUDA measurements; the beam
model retains channels-last. See [PERFORMANCE.md](PERFORMANCE.md) for results,
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
refreshes replay every 1,000 updates when idle, using EMA, 64 training seeds,
a 6,000-voxel cap, threshold .7, and eight exploration calls. Diagnostics use the
original monitor fibers at thresholds .5 and .85. Images show observed history,
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
python scripts/evaluate_single_path.py calibrate \
  --manifest output/single_path_v11_preparation/seeds.json --out output/RUN/calibration \
  --checkpoints output/RUN/ckpt_*.pt
python scripts/evaluate_single_path.py final \
  --manifest output/single_path_v11_preparation/seeds.json --out output/RUN/final \
  --selection output/RUN/calibration/selection.json
```

Calibration selects the greatest coverage among checkpoint/threshold pairs
reaching 95% scored precision, then locks the checkpoint hash, threshold and
sampling seeds. Gaussian checkpoints default to three repeats (`--sampling-seeds
0 1 2`), with both per-seed summaries and pooled metrics. Final evaluation uses
the same locked repeats. Zero-start checkpoints default to seed 0; specify the
same repeat set explicitly when preparing a paired baseline comparison.
Final evaluation enforces that choice and uses actual 6,000-voxel rollouts.
Reports include drift-band counts, four-plane correctness, false stops,
departure continuations, coverage, divergence, total wrong length and its
per-trace distribution. Fixed-state evaluation also measures subsequent recovery
toward the original fiber from identical observed histories. Existing rollout
scoring semantics (3-voxel tolerance, sustained departure) are preserved; dense
prefix confidence labels use 1.5 voxels. Unknown length receives no verified
continuation credit.

Pass paired final baseline rows via `--baseline-rows` to compute fiber bootstrap
intervals. Baseline and new rows must have identical physical and sampling seed
identities; bootstrap resampling groups all repeats of each fiber together. The target
is ≥20% relative coverage improvement at matched 95% precision with no worse
sustained wrong continuations; no improvement is assumed from confidence shifts.

`evaluate_recovery.py --baseline-archive output/single_path_v11_baseline/source_before_v11.tar.gz`
loads the archived v10 implementation only for baseline evaluation. Use
`collect_baseline_fixtures.py --help` to preserve its actual observed decisions.
The archived deterministic rollouts have a 1,200-voxel cap and do not substitute
for the locked final comparison. `evaluate_single_path.py` also accepts
`--baseline-archive` for baseline calibration and locked final rollouts; calibrate
the baseline separately, using the same frozen seeds. Both final reports are
needed before making a matched-precision improvement claim.

## Verification and current status

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q \
  -o cache_dir=/tmp/single-path-pytest
```

Tests cover attention across futures and full history, masked and unsupported
observations, GT isolation, endpoint censoring, departed states, confidence gradient
routes, recovery limits, coordinate transforms, seeded Gaussian inference, legacy
zero-start inference, independence from batch grouping and diagnostic RNG, midpoint
integration, replay import/rotation, checkpoint round trips, beam compatibility,
accumulation, collection and synthetic parallel-fiber identity recovery.

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

## Learned beam scoring (`beam/`)

The untrained beam model now replaces native candidate step costs before
heuristic pruning. It scores every valid cone proposal using a single level-1
CT crop (192 × 96 × 96; 128 voxels behind and 63 ahead), candidate-specific
recent paths, and observed history. C++ retains search, endpoint constraints,
and fusion. CT loading shares the direct model's tight-block, mmap-backed
sampler. There is no coordinate decoder or tube objective.

See [`beam/README.md`](beam/README.md) for the scoring contract, build and
training commands, validation, and limitations. New checkpoints use
`beam_step_cost_v3`; previous beam checkpoints are incompatible.
