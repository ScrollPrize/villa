# Direct curve follower

`direct_curve_v1` is an independent model and trainer. It predicts one future
curve in one decoder pass followed by one bounded local correction, directly
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
| Image encoders | Separate three-stage CNNs, widths 16/32/64, GroupNorm |
| History | Latest eight points, then every fourth point from 128 supplied observations |
| History token inputs | Observed xyz, age, features and image-support flags at both scales |
| Curve decoder | Two transformer decoder blocks, width 128, four heads, FFN 256 |
| Local correction | History-conditioned MLP reading fine patches at the proposal; at most one trace voxel laterally |
| Total parameters | 998,725 (963,139 with correction disabled) |
| Prediction | 16 lateral coordinate pairs at forward planes 1 through 16 |
| Confidence | Sample fine-image patches on the predicted curve; score cumulative prefix features |
| Rollout | Commit up to eight confident points; reobserve and predict again |

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
that representation. A small correction head then reads 3 × 3 fine-feature
patches at the proposed points, together with their coordinates and the decoder
features that encode history. Its lateral adjustment has Euclidean norm at most
`--correction-limit` (default one trace voxel); the result stays within the fine
crop's observable lateral extent. The encoders and main decoder run once.
Confidence samples fresh patches at the corrected curve. `--no-correction`
retains the original one-pass architecture for paired comparisons.

The localization objective is Smooth L1 on the interpolated curve at annotated
quarter-voxel forward intervals, in physical trace units. It backpropagates through
the actual output coordinates. Targets outside the observable fine crop censor
the remaining localization prefix. Unknown annotation endings and confirmed
departures supply no localization supervision. Each curve's loss is half the
masked mean through `--n-commit` and half the masked mean over all 16 points.
With correction enabled, geometry is 0.75 times the corrected loss plus 0.25
times the initial proposal loss; both paths receive geometry gradients, including
through the proposal's feature lookup. The one-pass baseline uses its single
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
bash scripts/launch_direct.sh direct_corrected_run1
```

The launcher runs in the foreground. It uses the existing frozen seed manifest
and permanent recovery bank, writes to `output/direct_corrected_run1`, and refuses to
overwrite a run. Set `PYTHON=/path/to/existing/python` to select an environment.
There is no benchmark/preflight requirement or automatic package installation.

Equivalent explicit entrypoint:

```bash
export PYTHONPATH=../../..
python -m vesuvius.neural_tracing.fiber_follow.direct.train \
  --name direct_corrected_run1 \
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
configurable; the latter must divide the former.

The shared sampler provides 50% fresh augmentation, 25% fixed recovery and 25%
recent replay, with empty replay strata falling back to fresh samples. Replay
retains original-fiber correspondence; predictions from older models are unused.
Both rotated crop read footprints, observed history, and target geometry are
excluded from the holdout. The two image scales use separate bounded volume caches.
`--worker-cache-gb` budgets the fine reader; the coarse reader adds approximately
the same cache capacity.

Every 1,000 updates an idle background collector uses EMA weights and the direct
observation builder to collect up to 64 training seeds, with bounded exploration
after the gate closes. It publishes standard v5 replay for the live workers. Set
`--dagger-every 0` to disable it. Monitor diagnostics run at thresholds 0.5 and
0.85, with a 1,200-voxel cap by default; set `--diag-every 0` to disable them.
Monitor traces do not select the final checkpoint.

On logging updates, `log.jsonl` also records `decisions.by_drift`: initial and
corrected commit-window error sums/counts, correction improvements/regressions,
first-point and committed-prefix correctness, positive confidence-label counts,
crop censoring, blocked recovery connections, and gate decisions at 0.5/0.85.
Accepted-wrong counts score the actual accepted prefix, including shorter commits;
unknown accepted prefixes are separate. Counts pool across the effective batch
before division. Empty error means are JSON null. These are training states,
not a fixed validation set.

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
Old checkpoints without a correction field still load with the original one-pass
architecture for inference. They cannot resume into the changed training objective;
start a new run. New corrected and uncorrected checkpoints both record their
architecture options. Resume with the same options and run name:

```bash
bash scripts/launch_direct.sh direct_corrected_run1 \
  --resume output/direct_corrected_run1/last.pt
```

The learning schedule and data options must match the saved run. Loader workers
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

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_CACHE_DIR=/tmp/direct-numba \
MPLCONFIGDIR=/tmp/direct-mpl PYTHONPATH=../../.. \
/home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/direct-pytest
```

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
