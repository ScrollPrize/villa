# Direct curve follower

`direct_curve_v1` is an independent model and trainer. It predicts one future
curve in one decoder pass followed by two bounded local corrections, directly
supervised against annotated geometry.
The flow model and its training commands remain available separately.

## Design

The task is to continue the original fiber from an imperfect observed path.
The model combines detailed evidence near the current position with longer
backward context. All geometry remains in trace-grid voxels (eight base voxels).

| Component | Default |
| --- | --- |
| Fine image | CT level 0 plus presence; 80 × 48 × 48 samples, spacing 0.5 |
| Fine physical extent | 16 behind, 23.5 ahead, ±11.75 laterally |
| Coarse image | CT level 1 plus presence; 88 × 32 × 32 samples, spacing 2 |
| Coarse physical extent | 128 behind, 46 ahead, ±31 laterally |
| Image encoders | Separate three-stage CNNs, widths 24/48/96, GroupNorm |
| History | Latest eight points, then every fourth point from 128 supplied observations |
| History token inputs | Observed xyz, age, features and image-support flags at both scales |
| Curve decoder | Four transformer decoder blocks, width 128, four heads, FFN 256 |
| Local correction | Two shared learned refinements with sequence attention; at most one trace voxel laterally per step |
| Total parameters | 2,399,205 with correction enabled (previous default: 998,725) |
| Prediction | 16 lateral coordinate pairs at forward planes 1 through 16 |
| Path evidence | 3 × 3 fine-feature patches at longitudinal offsets −1/0/+1, deep fine/coarse features, support flags |
| Confidence | Sequence attention over final-path evidence, then cumulative prefix features |
| Rollout | Commit up to four confident points; reobserve and predict again |

Pooled deep image tokens include physical coordinates and a scale flag. The
history tokens are masked individually; image tokens remain available when all
history is missing. Older out-of-crop observations retain their geometry and
explicit support flags. No GT history enters the model. There is no rendered
history channel, geometric FiLM vector, inherited spatial encoder, flow objective,
noise distribution, ODE solver, or candidate ranker.

Each future query starts with a 3 × 3 feature patch on the forward axis, at
one-voxel lateral pitch. Joint decoding produces bounded lateral coordinates
(±10.75 by default, leaving room for the confidence patch). The prediction is
piecewise linear between the fixed forward planes. This assumes the short local
continuation is a graph along the current heading; tight reversals are outside
that representation. The correction head reads three longitudinal slices of
fine-feature patches, deep features from both scales, coordinates, support flags,
and the history-conditioned decoder features. A width-128 transformer encoder
block lets neighboring proposed points exchange evidence. The same head runs
`--correction-steps` times (default two), refreshing sampled features each time.
Every adjustment has Euclidean norm at most `--correction-limit` (default one
trace voxel), and every proposal stays within the observable lateral extent.
Deep sampling uses the actual stride-four convolution lattice at each scale.
Both image encoders and the main decoder run only once per decision.
Confidence samples the final corrected curve and uses its own sequence block.
`--no-correction` disables refinement while retaining the richer confidence head.

The localization objective is Smooth L1 on the interpolated curve at annotated
quarter-voxel forward intervals, in physical trace units. It backpropagates through
the actual output coordinates. Targets outside the observable fine crop censor
the remaining localization prefix. Unknown annotation endings and confirmed
departures supply no localization supervision. Each curve's loss is half the
masked mean through `--n-commit` and half the masked mean over all 16 points.
With correction enabled, geometry is 0.75 times the final corrected loss plus
0.25 times the average loss of all earlier proposals. Every stage receives
geometry gradients, including through its feature lookup. The one-pass model uses its single
curve with the same commit-window weighting and total geometry coefficient.

Confidence uses the existing dense original-fiber prefix labels at tolerance
1.5 and masked BCE. Labels and coordinates used to sample confidence evidence
are detached. The image/decoder representation can learn from both objectives;
the coordinate head learns from localization. Predictions are made monotone by
a cumulative minimum of prefix probabilities. The total objective is localization
plus 0.5 times confidence BCE. Confidence also uses half the commit-window mean
and half the full-horizon mean. Each term averages known points within a state,
then averages states; an entirely unknown state contributes zero. This keeps the
objective independent of microbatch boundaries without an extra model pass.

The established recovery policy is retained: the first connection is bounded
by six voxels, and correctness starts at the first predicted point. Confidence
does not certify every point on that initial recovery connection. Previous commits
remain fixed. This makes evaluation comparable, but does not solve retrospective
path repair or long-term identity ambiguity after all observed history has drifted.

## Training

From `fiber_follow/`, with the existing project environment:

```bash
bash scripts/launch_direct.sh direct_refined_run1 \
  --batch 64 --microbatch 16 --workers 10
```

The launcher runs in the background in a separate session and prints its PID
and log-following command. Standard output and errors are appended to
`output/logs/direct_refined_run1.log`; the PID is saved alongside it in
`direct_refined_run1.pid`. Python output is unbuffered. Watch progress with:

```bash
tail -f output/logs/direct_refined_run1.log
```

It uses the existing frozen seed manifest
and permanent recovery bank, writes to `output/direct_refined_run1`, and refuses to
overwrite a run. Set `PYTHON=/path/to/existing/python` to select an environment.
There is no benchmark/preflight requirement or automatic package installation.

Equivalent foreground entrypoint:

```bash
export PYTHONPATH=../../..
python -m vesuvius.neural_tracing.fiber_follow.direct.train \
  --name direct_refined_run1 --batch 64 --microbatch 16 --workers 10 \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
  --manifest output/single_path_v11_preparation/seeds.json \
  --fixed-bank output/single_path_v11_preparation/fixed_recovery.npz
```

Defaults are 50,000 updates, effective batch eight, microbatch two, AdamW at
3e-4 peak learning rate, 500-update warmup/cosine schedule, gradient clipping
at one, and EMA 0.999 with the existing early-update ramp. CUDA uses BF16;
CPU uses FP32. No compilation is required. `--batch` and `--microbatch` are
configurable; the latter must divide the former. The command above uses an
effective batch of 64 with four microbatches of 16. The new model defaults are
`--channels 24 --decoder-layers 4 --correction-steps 2 --n-commit 4`.

Fresh augmentation uses `--no-history-prob .15`. Conditional on history being
present, `--short-history-prob .4` chooses equally between uniformly sampled
1–8 and 9–32 observations. Remaining draws retain the existing full/randomly
truncated history policy. Thus roughly 34% of fresh states explicitly target
short startup histories; these percentages do not describe the replay mixture.
Replay observations are preserved. Flow training retains its existing sampler
defaults. Both probabilities are saved with the run.

The shared sampler provides 50% fresh augmentation, 25% fixed recovery and 25%
recent replay, with empty replay strata falling back to fresh samples. Replay
retains original-fiber correspondence; predictions from older models are unused.
Both rotated crop read footprints, observed history, and target geometry are
excluded from the holdout. Each item reads only the axis-aligned block its own
oriented crop touches (`read_tight_blocks`), which is bit-identical to the
rotation-invariant block and about a third of the voxels. The holdout guard still
uses the rotation-invariant footprint. The two CT levels use separate bounded
chunk caches; both scales share one presence reader with their combined budget.
`--worker-cache-gb` budgets the fine reader; the coarse reader adds approximately
the same cache capacity. With uncompressed arrays (see Throughput) the caches
hold memory maps, so the budget bounds live mappings rather than private memory.

Every 1,000 updates an idle background collector uses EMA weights and the direct
observation builder to collect up to 64 training seeds, with bounded exploration
after the gate closes. It publishes standard v5 replay for the live workers. Set
`--dagger-every 0` to disable it. Monitor diagnostics run at thresholds 0.5 and
0.85, with a 400-voxel cap by default, matching the flow trainer;
set `--diag-every 0` to disable them.
Monitor traces do not select the final checkpoint.

Every `--diag-every` updates (default 1,000), EMA diagnostics also write:

- `images/batch_STEP.png`: fine CT with observed history, GT, predicted curve,
  replay source and next-step confidence, in two lateral views.
- `images/batch_coarse_STEP.png`: the same geometry over the coarse CT crop,
  showing longer history context.
- `images/correction_STEP.png`: initial, intermediate and final proposals against
  GT (one proposal when correction is disabled).
- `images/rollout_STEP_c0.5.png` and `images/rollout_STEP_c0.85.png`: the evaluated
  monitor traces straightened along their GT fibers, with stop reasons.
- `curves.png`: geometry loss, confidence loss, dense coordinate error and
  monitor coverage/precision/divergence over training.

`STEP` is the six-digit update number, e.g. `001000`. Batch images use up to six
states from the last training microbatch. The rollout images reuse the evaluated
paths without additional tracing. Both trainers use the same monitor coverage:
`min(followed, cap) / min(available, cap)`, averaged over seeds, with the same
precision and departure scoring. The cap is `--diag-max-len`. Calibration/final
evaluation continues to use full available annotation lengths for both models.
New monitor records include `coverage_max_len`; older direct monitor records
used an uncapped denominator and are excluded from the new rollout curve plot.
Plotting adds one EMA batch forward and image-rendering work at
diagnostic updates. An already running trainer must restart from a checkpoint
with `--resume` to pick up code changes. Wall time of these periodic passes is
logged as `diagnostics_seconds` and `recovery_seconds` (about 27 s and 7 s at
the default 32 monitor seeds and 400-voxel cap), so `--diag-every` can be
raised on resume when updates are fast; `--recovery-every` must match the run.

On logging updates, `log.jsonl` also records `decisions.by_drift`: initial and
corrected commit-window error sums/counts, correction improvements/regressions,
first-point and committed-prefix correctness, positive confidence-label counts,
crop censoring, blocked recovery connections, and gate decisions at 0.5/0.85.
Accepted-wrong counts score the actual accepted prefix, including shorter commits;
unknown accepted prefixes are separate. Counts pool across the effective batch
before division. Empty error means are JSON null. These are training states,
not a fixed validation set.

The same statistics appear under `decisions.by_history` for 0, 1–8, 9–32 and
more than 32 observed points, including mean first-point confidence and gate
false stops. This measures the actual history supplied by both fresh and replay
states, making startup behavior visible alongside drift recovery.

Every `--recovery-every` updates (default 1,000), a separate fixed recovery
study runs on the first `--recovery-seeds` monitor seeds (default eight), with
one frozen augmented state per drift band: <1, 1–1.5, 1.5–2, and 2–3.5 voxels.
`monitor_recovery.npz` stores the exact observed states; its hash is saved in
config/checkpoints and checked on resume. A private RNG constructs them, and
fresh/resumed evaluation uses identical float32 geometry. Calibration and final
fibers are excluded. These synthetic fixtures contain no confirmed departures;
departure rejection is measured on training/replay states and rollout audits.
Diagnostics trace up to `--recovery-length` (default 32) at thresholds 0.5/0.85,
report endpoint recovery against the original fiber, and save full rows under
`recovery/monitor_STEP.json`. Endpoint recovery alone does not certify the whole
rollout. Set `--recovery-every 0` to disable this study independently of ordinary
monitor rollouts. No confidence ramp or smoothness penalty is enabled.

Training also logs loss, actual coordinate error, prefix correctness and realized
data-source fractions. Atomic checkpoints contain model/EMA weights,
optimizer/RNG state, geometry identities, source settings and training options.
Historical checkpoints still load their original one-pass or single-correction
architecture for inference and collection. Missing `rich_path_context` and
`correction_steps` fields mean the legacy heads and one correction, respectively.
They cannot resume into the changed training architecture; start a fresh run.
New checkpoints record all architecture and sampling options. Inference and
collection default to the checkpoint's commit count, with an explicit
`--n-commit` override available. Resume a new run with the same options and name:

```bash
bash scripts/launch_direct.sh direct_refined_run1 \
  --resume output/direct_refined_run1/last.pt --batch 64 --microbatch 16 --workers 10
```

`--batch` and `--microbatch` may change on resume; both must be positive and
the microbatch must divide the effective batch. For example, add
`--batch 16 --microbatch 8` for two accumulation steps. The optimizer state is
restored and the learning rate is not automatically scaled when batch changes.
The schedule remains based on optimizer updates, so a larger effective batch
processes more samples over the remaining updates.

The learning schedule and other data options must match the saved run. Loader workers
restart at a new deterministic seed on resume; their sample stream is not an
exact continuation of an uninterrupted run. Collector snapshots are inference
checkpoints, not resumable training checkpoints.

For a paired architecture comparison, keep data, seed, commit window, and training
budget identical:

```bash
bash scripts/launch_direct.sh direct_corrected --seed 0
bash scripts/launch_direct.sh direct_onepass --seed 0 --no-correction
```

The baseline has the new loss weighting too, so this comparison isolates the
correction stage rather than changing its objective at the same time.

## Throughput

Data loading, not the GPU, bounded the original trainer: every 128³ CT chunk
is decoded from the volpkg's rANS codec in about 3.5 ms while holding the GIL,
each coarse crop read a 200³ block touching dozens of chunks, and per-worker
caches thrashed. Three changes remove that, all with bit-identical inputs:

1. **Uncompressed arrays.** Decode the CT pyramid and the presence levels in
   place once (about 440 GB for `s1_ds2` levels 0 and 1 plus presence level 3;
   the remaining levels are small):

   ```bash
   python scripts/decode_store.py \
     /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/[0-5] \
     /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs/*_presence.ome.zarr/[0-9]
   ```

   Each chunk is decoded to a `.raw` sibling first; only when every chunk has
   one are the originals replaced and `.zarray` rewritten with
   `compressor: null`, so the array stays readable until the final swap and an
   interrupted run resumes. A random sample of chunks is re-decoded and compared
   byte for byte before the swap. Stop readers first; the compressed source is
   not kept. `ChunkedArray` memory-maps uncompressed chunks, so all loader
   workers, the tracer and the collector share decoded data through the page
   cache and the worker cache budget bounds live mappings rather than private
   memory. Each mapping holds a file descriptor, so the trainer raises its
   soft open-file limit to the hard limit at startup (`raise_open_file_limit`);
   other entrypoints under a 1024 soft limit need `ulimit -n` raised first.
2. **Tight per-item blocks** (see the training section above).
3. **Channels-last convolutions.** On CUDA the model and EMA keep
   `channels_last_3d` parameters (`conv_memory_format`); with NCDHW tensors
   cuDNN used a slow direct backward kernel for these narrow 3-D convolutions
   (74 ms to 49 ms per 24-state update on an RTX 5090, BF16 autocast). Resumed
   optimizer moments are re-laid out to match. Loader batches are pinned and
   copied asynchronously. Kernel selection changes floating-point rounding at
   the level of any cuDNN algorithm change; precision, loss and data are unchanged.
4. **Compiled training forward.** `--compile` (default on CUDA; `--no-compile`
   disables) wraps the training model in `torch.compile`, fusing the
   GroupNorm/SiLU/cast chains around the convolutions (49 ms to 36 ms per
   update). The EMA model, diagnostics, recovery evaluation and the collector
   stay eager, so their batch shapes never trigger recompilation and checkpoint
   keys are unchanged. Inductor fuses and reorders reductions, so the training
   forward's rounding differs slightly from eager; the objective is the same.
   The first update after launch or resume includes compilation time.

## Evaluation and tracing

Use the same frozen calibration/final fiber split and the same rollout scoring:

```bash
python scripts/evaluate_direct.py calibrate \
  --manifest output/single_path_v11_preparation/seeds.json \
  --out output/direct_corrected_run1/calibration \
  --checkpoints output/direct_corrected_run1/ckpt_*.pt
python scripts/evaluate_direct.py final \
  --manifest output/single_path_v11_preparation/seeds.json \
  --out output/direct_corrected_run1/final \
  --selection output/direct_corrected_run1/calibration/selection.json
python -m vesuvius.neural_tracing.fiber_follow.direct.infer \
  --checkpoint output/direct_corrected_run1/last.pt \
  --seed 18529.9,13044.9,51234.1 --out output/direct_traces
```

Calibration selects maximum coverage at ≥95% scored precision and locks the
checkpoint hash and threshold. Final rollouts use the held-out final seeds and
a 6,000-voxel cap. `--baseline-rows` supports paired fiber bootstrap comparisons.
For fixed calibration-state recovery (after training), use:

```bash
python scripts/evaluate_direct_recovery.py output/direct_corrected/last.pt \
  --fixtures output/single_path_v11_preparation/calibration_recovery.npz \
  --out output/direct_corrected/calibration_recovery.json
```

Physical sources and seed identities must match the frozen manifest; CT resolution
may differ intentionally. Reports record the actual volume settings and architecture.
This comparison measures the complete design, including finer inputs; it does
not isolate the effect of replacing flow matching.

## Verification and limitations

The combined refinement upgrade passed 132 CPU tests (six CUDA-only skips),
two compiled CUDA/BF16 updates at effective batch 64 / microbatch 16, an exact
EMA checkpoint reload check on CUDA, and two CPU updates using the real CT,
annotations and permanent recovery bank. The full-crop GPU smoke peaked at
3.02 GiB allocated / 3.62 GiB reserved on the RTX 5090. This is a memory and
correctness smoke check, not a throughput benchmark or evidence of improved
held-out accuracy. The model has 2,399,205 parameters.

The CPU suite uses an existing environment containing pytest; the training
environment need not install it:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_CACHE_DIR=/tmp/direct-numba \
MPLCONFIGDIR=/tmp/direct-mpl PYTHONPATH=../../.. \
/home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/direct-pytest
```

The existing store-decoding test needs local multiprocessing sockets, which
some execution sandboxes disable. No model/data changes are needed to run it
outside that restriction.

Tests cover gradient routes, missing/invalid history, annotation censoring,
departure supervision, physical feature coordinates, observation sampling,
microbatch equivalence, checkpoint/optimizer/RNG restoration, both crop exclusions,
and learning identity recovery between two visually identical parallel fibers.
The synthetic recovery test is a learnability test, not a performance claim.

Validation on 2026-09-25 after adding correction: the full sandboxed suite passed
118 tests with six CUDA skips. The direct model's CUDA/BF16 gradient test passed
separately with GPU access. Tests additionally cover the correction bound, actual
proposal/final feature lookups, auxiliary geometry gradients, confidence isolation,
commit-window weighting, legacy one-pass checkpoint loading, fixed monitor states,
and diagnostic RNG/update invariance.

A full-size two-update CPU smoke on real level-0 CT and the permanent recovery
bank passed, including periodic fixed monitor diagnostics and checkpoint resume.
The standalone calibration-recovery adapter evaluated two states; the replay
collector published six states from two training seeds. These are implementation
checks, not evidence of improved tracing accuracy. Reproduce the bounded smoke
with a new name:

```bash
PYTHON=/home/sean/Documents/villa/vesuvius/.venv/bin/python \
NUMBA_CACHE_DIR=/tmp/direct-numba bash scripts/launch_direct.sh direct_smoke \
  --out-root /tmp/direct-follower-validation --device cpu --steps 2 \
  --batch 1 --microbatch 1 --workers 0 --threads 2 \
  --diag-every 0 --dagger-every 0 --log-every 1 --ckpt-every 1 \
  --recovery-every 1 --recovery-seeds 1 --recovery-length 2
```

Full CUDA memory/throughput and final held-out accuracy require measurement.
The model remains a small deterministic predictor: regression can still average
ambiguous continuations, and a confidence head can share the predictor's blind
spots. Further refinement or multiple hypotheses should be motivated by measured
failures.
