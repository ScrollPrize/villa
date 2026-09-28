# Direct curve follower

`direct_curve_v1` is an independent model and trainer. It predicts one future
curve in one decoder pass followed by two bounded local corrections, directly
supervised against annotated geometry.
The flow model and its training commands remain available separately.

## Explicit identity decisions

Run the full identity experiment from fresh weights with:

```bash
bash scripts/launch_identity_decisions.sh
tail -f output/logs/direct_identity_decisions_run1.log
```

This launcher retains the full dataset, live neighbor bank and 100,000-step
schedule of `launch_identity_shared_bank.sh`. Its new options are
`--persistent-seed --decision-fraction 0.25 --candidate-weight 1.0`.
It disables the older replay-only wrong-continuation replacement because
matched departure pairs now supply that supervision directly.

Matched pairs share current position, frame, observed recent history and two
candidate continuations. Changing the original seed changes which continuation
belongs to the target. Choice pairs supervise both the appropriate geometry and
candidate ownership. Departure pairs put recent history entirely on B: a B seed
permits continuing B, while an A seed requires stopping. Wrong-tail lengths span
8–128 trace voxels. Candidate order and pair order are randomized. Independent
photometric augmentation still applies to each state.

Candidate prefix BCE uses the existing prediction confidence head, including its
identity attention. No additional decision head or inference candidate search is
introduced. Loss is ordinary geometry + 0.5 prediction-confidence BCE + 0.5
InfoNCE + 1.0 candidate BCE, averaged per observed state. Unsupported candidates
are masked. About 25% of all states are requested as complete pairs; unavailable
or held-out bank examples fall back to ordinary samples. Logs report both the
requested and realized fractions, candidate acceptance/rejection by case and
tail length, and InfoNCE averaged over eligible states as well as all states.

The persistent reference is one CT patch at the observed seed, active from the
first decision. Synthetic states use their observed path start; paired states
explicitly simulate different original seeds. Traces and newly collected replay
preserve the same seed position, direction and age. Old replay without a recorded
seed keeps it absent instead of substituting annotation-derived history.
The reference occupies the first existing anchor slot; other anchor slots are
masked. Older checkpoints retain their old anchor behavior. Start fresh when
switching to this convention. The new losses and logging test the intended
learning signal; improved held-out tracing remains an experimental result to
measure after training.

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
bash scripts/launch_regression.sh direct_refined_run1 \
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
python -m vesuvius.neural_tracing.fiber_follow.regression.train \
  --name direct_refined_run1 --batch 64 --microbatch 16 --workers 10 \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
  --manifest output/single_path_v11_preparation/seeds.json \
  --fixed-bank output/single_path_v11_preparation/fixed_recovery.npz
```

Defaults are 50,000 updates, effective batch 24 in one microbatch, six loader workers, AdamW at
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
`--dagger-every 0` to disable it. Monitor diagnostics run at threshold 0.5,
with a 400-voxel cap by default;
set `--diag-every 0` to disable them.
Monitor traces do not select the final checkpoint.

Every `--diag-every` updates (default 1,000), EMA diagnostics also write:

- `images/batch_STEP.png`: fine CT with observed history, GT, predicted curve,
  replay source and next-step confidence, in two lateral views.
- `images/batch_coarse_STEP.png`: the same geometry over the coarse CT crop,
  showing longer history context.
- `images/correction_STEP.png`: initial, intermediate and final proposals against
  GT (one proposal when correction is disabled).
- `images/rollout_STEP_c0.5.png`: the evaluated
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
crop censoring, blocked recovery connections, and gate decisions at the diagnostic threshold 0.5 (`DIAGNOSTIC_THRESHOLDS`).
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
Diagnostics trace up to `--recovery-length` (default 32) at threshold 0.5,
report endpoint recovery against the original fiber, and save full rows under
`recovery/monitor_STEP.json`. Endpoint recovery alone does not certify the whole
rollout. Set `--recovery-every 0` to disable this study independently of ordinary
monitor rollouts. No confidence ramp or smoothness penalty is enabled.

Training also logs loss, actual coordinate error, prefix correctness and realized
data-source fractions. Terminal output (including `output/logs/NAME.log`) uses
formatted metric blocks and drift/history tables.
The complete metrics remain in `RUN/log.jsonl` for analysis and plotting.
Running trainers pick up formatting changes after restarting from a checkpoint.
Atomic checkpoints contain model/EMA weights,
optimizer/RNG state, geometry identities, source settings and training options.
New checkpoints record all architecture and sampling options. Inference and
collection default to the checkpoint's commit count, with an explicit
`--n-commit` override available. Resume a new run with the same options and name:

```bash
bash scripts/launch_regression.sh direct_refined_run1 \
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
bash scripts/launch_regression.sh direct_corrected --seed 0
bash scripts/launch_regression.sh direct_onepass --seed 0 --no-correction
```

The baseline has the new loss weighting too, so this comparison isolates the
correction stage rather than changing its objective at the same time.

## Visual identity (`direct_identity_v1`)

`--identity` trains the design in `IDENTITY_REPAIR_PLAN.md`. The existing
architecture is unchanged and its checkpoints load as before (the retired
`rich_path_context: true` entry is accepted).

| Component | Default |
| --- | --- |
| History patches | Fine CT, 9 × 41 × 41 samples (±2 along, ±10 lateral trace voxels) in each point's own path frame |
| Fine crop | 80 × 81 × 81 samples (±20 laterally); prediction lateral limit ±19; training query patches can extend beyond this crop |
| Patch layout | 32 recent patches every 4 voxels over 128; four seed-segment anchors every 4 voxels once the head is ≥144 voxels past the seed |
| Patch tokens | Embedding plus centre, tangent and u axis in the head frame, log age, anchor flag |
| Appearance encoder | CT only; stem and three pre-activation residual blocks (lateral dilations 1–6), width 32, per-voxel channel LayerNorm, 32-d L2-normalized output |
| Where identity enters | Decoder memory and queries; correction and confidence heads cross-attend from appearance at proposed points to the tokens (null token when none) |
| Total parameters | 2,951,845 |

The encoder's receptive field is exactly one patch (9 × 41 × 41), and its norm never pools
space. As a result, a patch's embedding equals the dense appearance map over
the fine crop at the same place (a test checks this). Candidate, positive and
negative embeddings in new runs come from separate, identical CT patches
centered at each query (`--identity-query-patches`, enabled by default).
Both classes use the head frame, the same 9 × 41 × 41 patch, encoder, and
photometric augmentation. The main fine crop and transformer token sizes stay
unchanged. Prediction evidence still uses the dense map; committed-path
patches are read once per decision. Training anchors come from annotation
144–2048 voxels behind (probability 0.75), while tracing uses the actual
seed segment.

New models save `appearance_version: 2`. Older checkpoints without that field
retain version 1's 7 × 33 × 33 patches and their saved fine crop when loaded,
resumed, or used with `--init-tracer`. Start a fresh identity run to use the
larger appearance encoder. The patch array contains about twice as many samples,
and the fine crop about 55% more; rerun preflight for the intended microbatch.

New runs sample smooth wrong-fiber tails over 4–128 trace voxels with
`--bank-wrong-continuation-tail 4 128`; equal bounds fix the length. Use the
shared bank `output/neighbor_samples_r0_32_l80_160_v2` for all three
bank roles. Bridges span 16–24 voxels for nearby paths; for paths more than
12 voxels away, the bridge is at least twice the maximum separation.
A stored path must fit the bridge, drawn tail, and margin: the 80–160 range
does not guarantee a 128-voxel tail on every draw. Seed anchors stay on the original prefix; recent
history may be entirely on the wrong fiber for long tails. New runs force original
seed anchors when fewer than two recent on-target patches remain, and reject a
synthetic replacement if neither reference is available. Long-continuation draws
use stored path lengths to skip paths that cannot fit the requested tail.
Existing checkpoints resume with their saved range (4–12 for checkpoints made
before this option). See [bank generation](neighbor_mining.md) for the launcher.

Identity objective (`--identity-weight .5`, `--identity-temperature .1`):
InfoNCE between the mean of the recent on-fiber history embeddings (at least
two; on-fiber means within 1.5 voxels of the annotation) and four annotated
positives 1–20 voxels ahead. Each positive competes only with up to eight
negatives beside it (within two voxels along the fiber). With
`--seed-anchor-fallback`, at least two valid seed patches provide the reference
when recent history is insufficient. This replaces the recent reference; it does
not add another loss. Each state still has one InfoNCE term and the effective
batch, including ineligible states, remains the loss denominator.

Identity training requires the validated native-traced neighbors: add
`--negative-bank output/neighbor_samples_r0_32_l80_160_v2`. Persistent loader workers
discover completed new shards every 30 seconds, so the bank can grow while
training runs. Unknown coverage stays masked, and held-out candidates are
excluded. Cached geometry uses at most 64 MiB per worker by default. See
[live bank usage and validation](neighbor_mining.md#live-training-integration),
including attaching the bank when resuming an existing identity checkpoint.

Pair sampling version 4 uses exact centerline points for both classes. Target
annotations and bank paths are interpolated at quarter-voxel arclength intervals,
sampled uniformly, and checked against the same interpolated presence threshold
(`--negative-threshold .7`) and full appearance receptive-field bounds. New
runs infer `--negative-lateral-max` from the bank (32 for this shared bank).
Four of eight negative slots are reserved for distances up to 12 voxels, four
for outer paths. Within each band, visit distinct stored paths before reusing
one. Missing slots remain masked; surplus outer points cannot consume near
slots. Configure the split with `--negative-near-fraction` and
`--negative-near-distance`. An optional `--near-negative-bank` adds a second
source of certified relationships for primary states.
A wider presence-only search crop locates candidates; CT is read in the small
query patches. The full read footprint is checked against the holdout before
volume I/O. Paths are never expanded into tubes and neither class is snapped or phase-shifted
off its line. Both use the same CT interpolation and appearance encoder.
Fractional interpolation weights follow each line's
actual coordinates; they are not forced to match by displacing the negatives.
The pure-presence component sampler and arbitrary departed-head negatives have
been removed. Ranking values from earlier sampling versions are not directly
comparable.

Existing checkpoints retain their saved sampling mode; checkpoints made before
these options recover their radius and query mode from saved `identity_sampling`
(the earliest radius defaults to 10) and retain their old unstratified sampling.
To change those settings, start a new run, optionally with
`--init-tracer CHECKPOINT`. Embedding evaluation uses the checkpoint's saved mode.

The same bank supplies ordinary following supervision with
`--bank-following-probability .1`: 10% of fresh draws attempt a training-eligible
mined path (about 5% of all states when the 50/25/25 replay mixture is full).
These states receive the same augmentation and geometry/confidence loss weights
as primary annotations. Cut path ends are unknown, not stop labels; each state
also passes the usual crop/history/label holdout checks. Its own mined path is
the positive target, and the original annotated parent can supply negatives.
An empty bank falls back to primary annotations. Logs report the realized
`bank_following_fraction` (source 4).

New runs reserve another 20% of fresh attempts for primary annotation locations
covered by bank relationships, requiring at least two on-target recent patches
(`--bank-coverage-probability .2`). The ordinary holdout checks still apply;
missing coverage falls back to normal sampling, and image presence can still
mask InfoNCE pairs. `--following-bank` and `--continuation-bank` can override
the shared source for those roles. Following uses globally deduplicated draws;
primary negatives and continuations retain every validated parent relationship.
Logs break out eligible states, anchor source, distinct stored paths per
positive, geometry, confidence and InfoNCE by source and near/outer distance.
Distance-only InfoNCE is diagnostic and is not added to the training objective.

Confidence labels rasterize only cells containing validated centerline samples;
each cell must entirely clear the original annotation's exclusion tube. The
continuous query coordinates remain unchanged. Distance metrics keep their
names; identity metrics sit under `identity` in `log.jsonl`.

Bank paths also supply synthetic wrong-fiber continuations: an annotated prefix
smoothly blends onto a verified neighbor, followed by a requested 4–128
voxels on that neighbor. The bridge is observed history only; departed states
have zero positive geometry supervision and teach confidence rejection.
`--bank-wrong-continuation-probability .75` attempts this replacement for 75% of
**recent DAgger departure draws**. Other drift strata, fixed replay and fresh
allocations are unchanged. Unsafe or unavailable proposals retain the original
replay sample. `bank_wrong_continuation_fraction` reports the actual batch share.

Launch a fresh run using this single bank with:

```bash
bash scripts/launch_neighbor_bank_shared.sh
# Once the miner has written run.json; bank generation continues concurrently:
bash scripts/launch_identity_shared_bank.sh --init-tracer output/PREVIOUS_RUN/last.pt
```

This uses effective batch 16, microbatch 4 and 12 loader workers while the
five-worker miner runs at nice 19. It refuses to overwrite its run directory. Additional
training options can be appended; `RUN_NAME` selects a different fresh run name
(default `direct_identity_shared_bank_run2`), and `BANK_PATH` selects its bank.

Earlier real-data verification on 2026-09-27 used the outer-only v1 bank and the CT/presence volumes
listed in the launcher. `output/shared_bank_training_probe.py` exercised all
three roles in one GPU optimizer update, including 23 primary-state negatives
outside the main crop. Both following states had 61 geometry targets, and the
wrong-continuation state had none and received known rejection labels.
The report is `output/shared_bank_training_probe.json`. Its snapshot audit found
519 stored paths in the 80–160 length range and zero duplicates under the
configured overlap rule. This is a plumbing check, not a training-quality result.

The complete available target history remains the shared anchor for every
positive and negative in a comparison. Neighbor candidates are local patches;
they do not require their own traced histories. A neighbor with little or no
known history can therefore remain a negative without shortening the target's
history or inventing padding evidence. Unavailable candidates are masked, and
states without two valid target-history patches or any valid negatives receive
no identity loss. Geometry and confidence training continue on those states.

The v2 fixes were verified with 209 passing training/data/mining tests (three
CUDA skips and one deselection in the CPU sandbox), including persistent-worker
bank refresh, cross-parent relationship retention, role selection, near/outer
slots, forced departure references, seed-reference gradients, legacy sampling
restoration and additive microbatch diagnostics. A real CT/presence-backed CPU
optimizer update exercised covered primary, bank following and 128-voxel
departure states. The departure had zero recent on-target patches and four seed
anchors; it received InfoNCE and confidence supervision with zero geometry
targets. Script and metrics: `output/mined_bank_review_20260927/verify_fixes.py`
and `verify_fixes.json`. These checks validate training plumbing, not eventual
model quality.

Data: presence is zeroed in both crops for `--presence-dropout .25` of
states, after identity targets are computed. Photometric contrast (×1/1.4–1.4),
brightness (±0.1) and noise (σ ≤ 0.03) are drawn independently for the
appearance copy of the fine crop and for the history patches; the
query patches share the appearance crop's augmentation parameters.
The localization crops are not augmented. Primary fresh draws are 20% near mined contact
episodes (`--contacts`), 10% on hard spans (`--hard-spans`), and 10% near
earlier fresh states that had bank negatives. Both files are checked
against the current fiber geometry. Every patch footprint and the identity
label geometry pass the holdout check.

Before the first run, `regression/identity_preflight.py` measured the previous
patch/crop sizes and negative-sampling defaults, in
`output/identity_preflight_20260926/`. These results predate the larger patches:

| Check | Result |
| --- | --- |
| Contact states with a scored positive | 94% (254 states, 144 episodes; 95% of episodes yield) |
| Negatives per contact state | 23.6; 27% within 2 voxels of the other annotated fiber, 0.5% within 1.5 of the traced fiber's own annotation |
| Ordinary fresh states with a scored positive | 91% (most neighbors unannotated: 2.2% near another annotation) |
| Read cost per decision | patches +12 ms over 41 ms fine+coarse crop reads; +1.64M source voxels over 1.16M (32 patches) |
| GPU, microbatch 16, eager BF16 | 17.1 GiB allocated / 18.8 GiB reserved; about 0.24 s per update |
| Single-decision forward (eager) | 10.7 ms, against 6.3 ms for `direct_curve_v1` |

`negatives.png` holds the visual spot-checks.

```bash
bash scripts/launch_regression.sh direct_identity_run1 --identity \
  --negative-bank output/neighbor_negatives_bulk_v1 \
  --contacts "$PWD/output/direct_ct_spatial_run1/contacts.json" \
  --hard-spans "$PWD/output/hard_spans_8a0bb01095fa.json" \
  --batch 64 --microbatch 16 --workers 10
```

Evaluation (`regression/identity_eval.py`) freezes an ambiguous subset once:
seeds whose annotated continuation within 400 voxels passes within 6 voxels of
another annotated fiber. That is 42 of 96 calibration and 75 of 176 final
seeds (`output/identity_eval_20260926/ambiguous.json`). `rollouts` reports all
seeds and the subset: identity switches (a departure followed by ≥8 of the next
32 points within 3 voxels of another annotated fiber), incorrect length and
its quantiles, per-decision on-track vs departed confidence AUC (the stage-B
definition), false stops and positions, latency, patch-read time and peak
memory. It also writes a coverage-risk curve over `--thresholds` and paired
fiber-bootstrap deltas against `--baseline-rows`. For identity checkpoints,
`embedding --negative-bank BANK` reads evaluation-only paths and reports held-out ranking by history age, plus the prediction and
confidence shifts when appearance tokens are masked or shuffled.

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
python scripts/evaluate_regression.py calibrate \
  --manifest output/single_path_v11_preparation/seeds.json \
  --out output/direct_corrected_run1/calibration \
  --checkpoints output/direct_corrected_run1/ckpt_*.pt
python scripts/evaluate_regression.py final \
  --manifest output/single_path_v11_preparation/seeds.json \
  --out output/direct_corrected_run1/final \
  --selection output/direct_corrected_run1/calibration/selection.json
python -m vesuvius.neural_tracing.fiber_follow.regression.infer \
  --checkpoint output/direct_corrected_run1/last.pt \
  --seed 18529.9,13044.9,51234.1 --out output/direct_traces
```

Calibration selects maximum coverage at ≥95% scored precision and locks the
checkpoint hash and threshold. Final rollouts use the held-out final seeds and
a 6,000-voxel cap. `--baseline-rows` supports paired fiber bootstrap comparisons.
For fixed calibration-state recovery (after training), use:

```bash
python scripts/evaluate_regression_recovery.py output/direct_corrected/last.pt \
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
commit-window weighting, fixed monitor states,
and diagnostic RNG/update invariance.

A full-size two-update CPU smoke on real level-0 CT and the permanent recovery
bank passed, including periodic fixed monitor diagnostics and checkpoint resume.
The standalone calibration-recovery adapter evaluated two states; the replay
collector published six states from two training seeds. These are implementation
checks, not evidence of improved tracing accuracy. Reproduce the bounded smoke
with a new name:

```bash
PYTHON=/home/sean/Documents/villa/vesuvius/.venv/bin/python \
NUMBA_CACHE_DIR=/tmp/direct-numba bash scripts/launch_regression.sh direct_smoke \
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
