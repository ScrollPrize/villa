# fiber_follow — how it works and what we've tried

Status as of 2026-09-24. See `README.md` for file layout and commands.

> **Historical results: pre-`controlled_spans_v2`.** The experiments below used
> extrapolated tails as labels, treated annotation ends as stop targets, and
> permitted cached on-policy states inside the spatial holdout. Their precision
> also omitted continuation after reaching an annotation endpoint. The current
> loader/scorer fixes these issues; see `README.md` for the new semantics and
> fresh-training commands. The numbers below are not a controlled-span benchmark.

## Uniform-cube performance pass (2026-09-24)

On NVIDIA GB10 / PyTorch 2.12.1+cu130, a CUDA profile of the 64-cube showed
convolution backward dominating (about 65% of device time), including slow
backward kernels and layout conversions. Channels-last 3D convolution storage
and cuDNN training kernel tuning reduce this overhead without changing crop,
model widths, batch size, targets, or precision.

Synthetic batch 32; full forward, supervised losses, backward, gradient clipping
and AdamW; BF16 autocast; three warmup and five measured steps per variant:

| execution | mean step | median | min–max | peak allocated GPU memory |
|---|---:|---:|---:|---:|
| Original layout | 2.182 s | 2.163 s | 2.154–2.222 s | 10.41 GiB |
| Channels-last | 1.275 s | 1.271 s | 1.223–1.329 s | 10.80 GiB |
| Channels-last + kernel tuning | 1.081 s | 1.082 s | 1.068–1.088 s | 12.50 GiB |

These exclude data loading, rollouts, and DAgger contention. The larger tensor
still does substantially more work than the historical 52×32×32 input; this is
about a 2× improvement over the same 64-cube workload, not a return to the old
crop's timings. Tuning remains disabled for changing rollout batch sizes.

Reproduction artifacts: `/tmp/fiber-follow-perf64/bench_training.py` with
`--mode baseline --profile`, `--mode channels_last --profile`, and
`--mode autotune`, run using the project `.venv/bin/python` and codec
`LD_LIBRARY_PATH`. GPU profiles and JSON timings are saved beside the script.

A paired real-volume BF16 check using identical weights, inputs, and teacher
candidates produced identical evaluation outputs on the checked batch. In
training mode, total loss differed by 0.0064%, gradient cosine similarity was
0.999975, and relative gradient L2 difference was 0.71%. Small heatmap rounding
differences changed some nearly tied decoded proposals; execution is not
bit-identical. All parameter gradients were finite. CPU layout regression checks
cover outputs and supervised gradients; 59 regression tests pass.

An eight-step end-to-end training smoke used a real controlled fiber, batch 32,
two persistent loader workers, and disabled diagnostics/DAgger. After startup,
the last five step intervals averaged 1.173 s (median 1.170 s), including loading,
transfers, loss logging, and optimization. This small cached workload does not
measure full-dataset I/O or concurrent collection. It completed checkpoint
saving normally. Output: `/tmp/fiber-follow-perf64/runs/optimized_smoke`.

Command, with the project codec library path set:
```bash
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.train \
  --fiber-zarrs /home/sean/Documents/volpkgs/s1_2um.volpkg/20260411134726-fibers-20260915212757-L1_masked \
  --fibers /tmp/fiber-follow-supervised/smoke_fibers \
  --out-root /tmp/fiber-follow-perf64/runs --name optimized_smoke \
  --steps 8 --batch 32 --workers 2 --diag-every 0 --dagger-every 0 \
  --log-every 1 --ckpt-every 8
```

## Uniform 64-cube (2026-09-24)

Current architecture: `spatial_candidates_v3`. All crop sampling now uses
uniform metric spacing. Defaults are 64×64×64 at one trace-grid voxel spacing,
16 behind / 47 ahead, with 61×61 heatmaps covering ±30 on each future plane.
The configurable crop expansion and its sampling transforms have been removed.
The full-prefix GRU, 16-plane / 32-voxel forecast, 128-voxel history, and shared
0.7 confidence threshold remain. Parameter count remains 966,435.

A fixed-seed geometry audit used 4,000 fresh augmented training states passing
the spatial holdout filter, plus 3,597 saved states from step 3,000 of the
preceding run. Among states with known future crossings, 3.59% / 3.67% had some
GT outside the previous crop; 18.16% / 17.07% left its fixed ±12 prediction
support. The uniform cube reduces crop exits to 1.26% / 1.62%, and ±30
prediction support reduces unrepresentable states to 1.61% / 1.82%.
These are geometry measurements over the 32-voxel target horizon, not evidence
of improved trained accuracy. Fresh sampling was length-weighted, with endpoint
oversampling, seed 452; replay was read without loading an old checkpoint.

Diagnostics now distinguish history, the current point, full proposals,
prediction limits and out-of-crop GT. Training logs dense crossing fractions
outside the crop/prediction support and near the crop edge. Annotation policy
is unchanged; leaving the field of view is not labeled as a fiber endpoint.

Validation: 58 CPU regression tests passed; two real-volume GPU optimizer steps
had finite gradients, followed by a checkpoint roundtrip and tracing smoke test.
A synthetic batch-32 BF16 forward/backward/AdamW benchmark on the GB10, with
three warmup and five measured steps, took mean 2.119 s (median 2.106,
min 2.071, max 2.180), peak allocated GPU memory 10.42 GiB. The preceding
52×32×32 setup measured mean 0.408 s (median 0.411, min 0.400, max 0.414),
2.18 GiB under the same protocol. These exclude data loading and DAgger.
Benchmark harness: `/tmp/fiber-follow-uniform64/bench.py`, invoked with the
existing project Python, BF16 autocast and no precision/batch/model changes.

## Full-prefix confidence correction (2026-09-24)

The preceding architecture was `spatial_candidates_v2`. The confidence head now receives
a GRU state accumulated over the candidate, before predicting each prefix's
confidence. Local scoring convolutions and the ranking head remain in place.
The GRU adds 55,872 parameters, bringing the default model to 966,435 parameters.
No checkpoint migration or partial weight-loading path is provided.
Validation: 54 regression tests pass, including full-prefix influence and
independent recurrent states. Two real-volume GPU optimizer steps using the
trainer's BF16 autocast produced finite gradients throughout the model, followed
by a checkpoint roundtrip and a tracing smoke test at the 0.7 default.

The previous confidence labels described an entire prefix, but its two temporal
convolutions could only observe a short window of candidate features. A pair of
paths differing at their first point had opposite targets but exactly identical
logits at the final 12 of 16 horizons. The regression test now verifies late
predictions and gradients depend on that early point, and candidate recurrent
states remain independent.

The shared default confidence threshold is now 0.7 for diagnostics, collection,
and inference. On 64 held-out fibers excluded from the original 16 diagnostic
seeds, using 400-voxel rollout limits:

| checkpoint (v1) | threshold | coverage | scored precision |
|---|---|---|---|
| 1,000 | 0.7 | 66.8% | 86.6% |
| 2,000 | 0.7 | 64.9% | 92.5% |
| 2,000 | 0.8 | 36.1% | 97.0% |

These results motivate the threshold, but do not demonstrate calibration or
quality of the untrained v2 head. On this broader set, lowering the threshold
also increased absolute wrong length. Full-prefix context removes a verified
expressiveness limitation; its effect on rollout quality requires retraining.

## Initial spatial implementation: dense supervision and continuous DAgger (2026-09-23)

The initial spatial implementation used `spatial_candidates_v1`: a history-conditioned 3D spatial
encoder-decoder, coherent heatmap proposals, a spatial candidate scorer, and
supervised prefix confidence. Greedy rollout commits only the confident prefix
and stops before an unsafe commit. See `README.md` for the architecture, limits,
and current commands; the architecture description and experiments below are
historical.

Every line point between the outer control points is equally valid GT for every
fiber. A `reviewed` tag has no effect. Every interior line vertex is preserved;
longer segments are subdivided without dropping annotation bends. Low-presence
regions retain all positional supervision. Unknown annotation ends are censored,
and explicitly tagged physical terminations supply negative continuation labels.

DAgger runs asynchronously during one training run: periodic policy snapshots
feed a background collector, completed caches enter live replay, and the optimizer
continues without restart. Collection records exact frames/histories/proposals,
marks pre-failure and premature-stop windows, and limits exploratory suffixes.
Candidate labels always come from the original dense GT, including when scoring
candidates proposed by an earlier checkpoint. Source checkpoints and collection
settings are recorded. No pseudo-labels or checkpoint conversion paths are used.

Validation: 51 regression tests pass. A real-volume forward/backward pass through
the default 910,563-parameter model produced finite gradients in every parameter.
A 250-step CPU integration run on one real annotated fiber collected from step 10
and step 100 snapshots while consuming 110 replay samples without restarting
training. A second 250-step run with two persistent loader workers consumed 36
new replay samples. These small runs establish execution and label semantics,
not tracing quality.
No full GPU training or controlled quality comparison has been completed for
this architecture. Evaluate proposal recall versus ranking error, confidence
threshold tradeoffs, full-fiber coverage, wrong length, and unknown continuation
before deciding whether rollout beam search is needed.

The earlier data audit found 765 fibers and 26,546 controlled spans and excluded
approximately 439,750 trace-grid voxels of exterior tails. The present split uses
an arclength-weighted centroid so point sampling density cannot change membership.
Seed caches must be rebuilt because retaining all interior vertices changes the
geometry identity.

## Historical implementation and experiments

Everything below describes earlier models and runs, including superseded
stop labels, resampling, and collection procedures.

## 1. Task and data

Trace papyrus fibers automatically from a seed point and write VC3D
`vc3d_fiber` v3 JSON. Stopping when unsure is preferred over guessing.

| input | path | used as |
|---|---|---|
| fiber prediction zarrs (presence, nx, ny), OME level 3 | `/home/sean/Documents/volpkgs/s1_2um.volpkg/20260411134726-fibers-20260915212757-L1_masked` | model input |
| GT fibers (765 VC3D JSONs, ~3.5M points) | `/mnt/bigpc/spiral_dataset_working/fibers` | labels, seeds |
| CT `s1_ds2.zarr` level 1 | `/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr` | tried as input (v6, ct1); not used now |

Everything runs on the **trace grid** = fiber zarr level 3 = 8 base voxels
(~19 µm). GT line points are resampled to 1 grid voxel spacing. Fibers are
2–3 grid voxels wide and neighbours are often only a few voxels apart, so
tracing needs ~1 voxel accuracy. nx/ny encode an unsigned fiber axis
(nz ≥ 0); in empty space the direction field is dense noise (presence is 0
in ~64% of crop voxels, but the direction is only masked in ~4% of those).

## 2. How tracing works

The tracer is autoregressive: it looks at a crop in the frame of its **own
current heading**, predicts the next stretch, commits part of it, and repeats.

**State:** position, heading frame (u, v, f; parallel-transported between
steps), and the trace so far.

**Model input (per step):**

- **Oriented crop** (`geometry.CropSpec`), historical default 52 deep
  (16 voxels behind, 35 ahead) × 32 × 32. It is **foveated**: 1-voxel spacing
  near the current point, and the lateral extent grows with distance ahead
  (half-width = max(15.5, 0.8 × distance)), reaching ±28 at 35 ahead
  (1.8-voxel spacing). Channels:
  1. presence (trilinear)
  2. 6 axis-tensor channels from nx/ny, rotated into the local frame (uu, vv,
     ff, uv, uf, vf; sign-free, nearest-voxel). Optionally multiplied by
     presence (`--gate-direction`).
  3. own-history channel: a Gaussian tube (σ = 1) around the last ~20 traced
     points.
- **History path vector:** the last 128 voxels of its own trace, every 4th
  point (32 points), in the local frame, into the head via a small MLP.

**Model** (`model.FollowNet`): a small 3D CNN (stem + 3 stride-2 stages) →
flatten (+ history MLP) → MLP head. Two output types:

- `points`: regress 10 future points at 2, 4, …, 20 voxels ahead (offsets from
  straight ahead), plus a stop logit.
- `heatmap`: for each forward plane c = 2, 4, …, 20, a 25 × 25 map over lateral
  (u, v) position. It is trained against a Gaussian (σ = 0.7) at the GT plane
  crossing; points are decoded with a sub-voxel soft-argmax around the peak.
  There is also a stop logit. It can represent two candidates at a crossing,
  and its peakiness is a confidence signal (not yet used for stopping).

**Rollout** (`trace.ModelTracer`): each call commits the first 4 predicted
points (8 voxels), densified to 1-voxel spacing. The new heading is the
tangent there. Traces are batched across seeds on the GPU. The tracer stops on:

- the model's stop logit > 0.5 (the last 8 voxels are then trimmed, since a
  stop only fires after drifting)
- low presence on the centre line for 3 calls (disabled for CT-only models)
- a loop, the volume bounds, or the maximum length

`infer.py` traces both directions from a seed (snapped to the local
presence maximum) and writes fiber JSON that passes `vc3d_fiber_format`'s
parser.

## 3. How training works

**Split:** fibers whose centroid lies in base z ∈ [45000, 48500) are held out:
123 val and 642 train. Training samples inside that band (±48 grid voxels)
are skipped, which removes 8.1% of training points (from 265 train fibers,
mostly V fibers).

**Fresh samples** (`data.make_sample`): pick a point on a training fiber
(random direction). Then build the training state:

- **Perturb the state:**
  - lateral offset with σ drawn from (0.5, 1.2, 2.5) voxels
  - heading error with σ drawn from (5°, 12°, 25°)
  - random roll
- **Make the history:** the GT points behind, with the offset ramping in, plus
  noise and a slow random wobble of up to 1 voxel. Sometimes the history is
  truncated or removed to simulate a fresh seed.
- **Targets:** the next GT points (points output) or the GT plane crossings
  (heatmap output), both relative to the perturbed state, so the model learns
  to return to the fiber.

**Breaks:** GT stretches where presence (the max over each point's 3×3×3
neighbourhood) is below 0.1 for ≥ 8 voxels are treated as physical breaks
that were bridged by hand: 1,557 stretches, 1.6% of GT. No position loss is
applied past a break, and the stop target fires within 8 voxels of a break
or fiber end. Approaches to ends and breaks are oversampled.

**On-policy states** (`collect.py`, DAgger-style): a checkpoint traces the
training fibers. States that are still within 2.5 voxels of GT get GT
continuation targets. States in the first 24 voxels after leaving the fiber
(> 3.5 voxels) get **stop** with no position target, so nothing ever trains
it to follow a wrong fiber. The v5 collection has 1.26M states, 24.8k of them
off-track, stored memory-mapped in `output/v5/onpolicy_mmap/`.

**Batch mix:** 60% fresh GT, 35% on-track on-policy, 5% off-track (stop).

**Loss:** smooth-L1 on points, or cross-entropy on heatmaps, plus 0.5 × stop
BCE.

**Optimisation:** AdamW, lr 2e-3, cosine schedule over `--steps` (1500
for the iteration runs), batch 256.

**Speed** (current fast path, `--norm batch --compile`, 8 workers):

- Workers build crops with a fused numba sampler (`fast_sample.py`) that
  matches the torch path within float rounding.
- The model runs with BatchNorm, channels-last layout, bf16 and `torch.compile`.
- b1's config went from 342 to ~775 samples/s (2.3×). It is GPU-bound at
  ~95% with ~20 GB RAM, which puts 1500 steps at ~8 min.
- The earlier bottleneck was GroupNorm, which forces fp32 casts under
  autocast, plus layout conversions; together about 60% of GPU time.

**Coverage of the data:** a 1500-step run draws ~230k fresh samples against
2.7M usable training points, so it sees only ~8.5% of them. All variants
plateauing suggests training length is a limiting factor.

## 4. How evaluation works

`scripts/eval_ckpt.py CKPT --tag NAME`

- **Seeds:** `output/eval/val_seeds.pkl` holds 2 high-presence (≥ 0.8)
  points on each of the 123 held-out fibers, traced in both directions, for
  **492 seed/directions**. The initial heading is the direction field at the
  seed, signed to agree with GT. Seeds and headings are identical for every
  tracer.
- **Scoring** (`evaluate.score_trace`): a trace counts as on the fiber until
  it is > 3 voxels from GT for 3 consecutive points. The metrics are:

| metric | meaning |
|---|---|
| coverage | GT length followed before leaving, divided by the GT length available in that direction (mean over traces) |
| coverage to break (`cov_nb`) | the same, but the available length stops at the first break (stopping there is fine) |
| precision | correct traced length / (correct + wrong continuation), per trace, averaged |
| **length precision** | the same, pooled over all traces; the headline number for "don't guess" |
| wrong length | traced length after leaving the fiber (mean voxels) |
| diverged | fraction of traces that left the fiber before reaching its end |

**Caveat:** until late in the session, evals used only the first 200 seeds,
which cover **50 of the 123** held-out fibers. On the full set, coverage is
0.03–0.07 lower for every run; precision is about unchanged; the ranking
held. Differences of ~0.02 are within noise.

Training also writes live diagnostics to `output/<run>/images/`:

- `batch_*.png` shows crops in the model frame (thin slab following GT;
  green = GT, orange = prediction, red = own history).
- `rollout_*.png` shows 16 held-out rollouts straightened along GT, with
  breaks in red and a distance-to-GT curve.

## 5. Experiments

Baseline = `FieldTracer`: integrate the direction field 1 voxel at a time
with inertia and presence re-centring, and stop after 12 low-presence steps.
It is a floor, **not** VC3D's native beam-search tracer.

### Full held-out set (492 seed/directions, 123 fibers)

| run | coverage | cov to break | precision | length precision | wrong length | diverged |
|---|---|---|---|---|---|---|
| baseline | 0.447 | 0.531 | 0.294 | 0.205 | 3725 | 0.89 |
| **v5** | **0.518** | **0.605** | 0.538 | **0.395** | 1659 | 0.74 |
| v8 | 0.504 | 0.589 | 0.536 | 0.383 | 1698 | 0.75 |
| g1 | 0.494 | 0.583 | 0.490 | 0.355 | 1940 | 0.81 |
| a1 | 0.505 | 0.592 | 0.499 | 0.357 | 1941 | 0.78 |
| **b1** | 0.482 | 0.568 | **0.542** | **0.395** | **1551** | 0.76 |
| b1f | 0.493 | 0.580 | 0.534 | 0.392 | 1612 | 0.79 |

### Small set only (200 seeds, 50 fibers; optimistic on coverage)

| run | coverage | cov to break | precision | length precision | wrong length | diverged |
|---|---|---|---|---|---|---|
| v1 | 0.580 | 0.644 | 0.502 | 0.335 | 2053 | 0.75 |
| v2 | 0.551 | – | – | – | 2299 | 0.81 |
| v3 | 0.561 | 0.626 | 0.444 | 0.300 | 2317 | 0.78 |
| v4 | 0.565 | 0.630 | 0.528 | 0.365 | 1709 | 0.81 |
| v6 | 0.564 | 0.630 | 0.539 | 0.364 | 1703 | 0.79 |
| v7 | 0.321 | 0.366 | 0.898 | 0.830 | 100 | 0.17 |

### What each run changed

| run | change (vs previous) | steps / init | params | train speed | lesson |
|---|---|---|---|---|---|
| v1 | first model: 32×20 box crop (8 behind, 23 ahead), 8 points to 16 ahead | 3500, scratch | 3.0M | 1140 sps | beats the baseline clearly |
| v2 | deeper box 48×20 (39 ahead), 12 points to 24 ahead | 1500, scratch | 3.9M | 1420 sps | no gain: the fiber left the ±9.5 crop in 40% of samples |
| v3 | **foveated** crop 40×24 (31 ahead, far ±25); fiber fully inside 95% | 1500, scratch | 3.5M | 1010 sps | ≈ v2 |
| v4 | **break-aware** labels (no loss past breaks, stop near breaks/ends), tracer stops on 1st stop | 1500, scratch | 3.5M | 960 sps | wrong length 2317 → 1709 |
| v5 | + **on-policy** states from v4 (50% of batch) | +1500 from v4 | 3.5M | 1020 sps | best overall so far |
| v6 | v5 recipe + **CT** channel (zero-init) | +1500 from v4 | 3.5M | 740 sps | no gain, 1.7× slower tracing; CT dropped |
| v7 | on-policy from v5 (off-track states 15% of batch), crop 44×32 | 1500, backbone from v5 | 6.0M | 570 sps | over-stops: precision 0.83 but coverage 0.32 |
| v8 | crop 16 behind, 128-voxel history vector, more perturbation + history wobble, off-track 5% | 1500, backbone from v7 | 6.8M | 490 sps | back at v5's operating point |
| ct1 | CT-only input (level 1) | stopped early | – | – | level 1 CT too blurry |
| g1 | v8 recipe + **gated direction** (× presence), from scratch: reference | 1500, scratch | 6.8M | 460 sps | gating: no visible gain |
| a1 | g1 + **uniform** 60×44 crop (43 ahead, ±21.5) | 1500, scratch | 15.5M | 240 sps | +0.01 coverage, no precision gain, 2× cost |
| b1 | g1 + **heatmap** output + bigger backbone (32-64-128-192, +1 conv/stage) | 1500, scratch | 18.4M | 340 sps | +0.04 length precision, −20% wrong length vs g1 |
| b1f | b1 + BatchNorm + channels-last + compile + fused loader | 1500, scratch | 18.4M | **775 sps** | same quality as b1 within noise at 2.3× speed; fast path validated |

## 6. Where things stand

- **Best model:** v5 on raw numbers. On equal budget (1500 steps from
  scratch), b1's recipe (heatmap + bigger model) is the only change that
  improved precision.
- **All variants plateau** around 0.48–0.52 coverage and 0.36–0.40 length
  precision on the full set. Each run sees ~8.5% of the training points,
  so under-training is the leading hypothesis.
- **Next:** train the b1 recipe much longer on the fast path.
- **Open levers:**
  - use the heatmap confidence as a stop signal
  - re-collect on-policy states with a newer model (so they carry the full
    128-voxel history)
  - an eval-time stop-threshold sweep, to compare models on
    coverage-vs-precision curves rather than single points
  - failure analysis of where traces leave the fiber (crossings, parallel
    neighbours, breaks, possible GT errors)
  - comparison against VC3D's native tracer on the same seeds


## Reproducing the pipeline smoke check

The integration checks used the existing environment, real fiber prediction
volume, and `anon_20260822T031659943_000148.json` copied into a temporary annotation
directory. They intentionally used a small model and a 12-voxel rollout budget.
For a new temporary output/run name:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vesuvius.neural_tracing.fiber_follow.train \
  --name pipeline_smoke --out-root /tmp/fiber-follow-checks \
  --fiber-zarrs /home/sean/Documents/volpkgs/s1_2um.volpkg/20260411134726-fibers-20260915212757-L1_masked \
  --fibers /tmp/fiber-follow-smoke-annotations --device cpu \
  --steps 250 --batch 2 --workers 2 --diag-every 0 \
  --dagger-every 10 --dagger-seeds 2 --dagger-batch 2 \
  --dagger-explore-calls 2 --dagger-trace-len 12 --replay-keep 2 \
  --crop-depth 20 --crop-width 12 --crop-behind 4 \
  --n-future 4 --n-history 8 --hist-points 4 --hist-stride 2 \
  --heat-bins 9 --n-candidates 2 --widths 8 16 --hidden 16 --norm group \
  --worker-cache-gb .1 --log-every 100
```

Completion timing and replay counts vary because collection is asynchronous.
Look for `dagger_states` events followed by increasing `replay_samples_seen`.
An unfinished final collection is explicitly discarded at training shutdown.
