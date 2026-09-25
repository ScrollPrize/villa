# fiber_follow — how it works and what we've tried

Status as of 2026-09-25. See `README.md` for file layout and commands.

> **Historical results: pre-`controlled_spans_v2`.** The early experiments in the Historical implementation section used
> extrapolated tails as labels, treated annotation ends as stop targets, and
> permitted cached on-policy states inside the spatial holdout. Their precision
> also omitted continuation after reaching an annotation endpoint. The current
> loader/scorer fixes these issues; see `README.md` for the new semantics and
> fresh-training commands. The numbers below are not a controlled-span benchmark.

## Deterministic-path gate fine-tune, frozen arm: negative result (2026-09-25)

Follow-up to the scorer ablation (`output/flow_v10_small_b8_d64/scorer_ablation_20260925/`),
which found selection solved by the `y0 = 0` flow path and the remaining failures in the
gate: at 1.5–2 voxels of drift the committed path is wrong 84% of the time and the gate
closes on 5.5% of those states. The question was whether that is a supervision problem
or a feature problem. `scripts/launch_det_gate.sh det_gate_frozen frozen` fine-tuned the
50k EMA checkpoint for 10k steps (batch 8, lr 1e-4, cosine, 5.2 steps/s) with the encoder,
history conditioning and flow frozen: rank loss off, slot 0 the `y0 = 0` path committed
alone, half the confidence loss on that slot over the commit window, replay stratified
50% fresh / 25% drift 1–2 vox / 15% other / 5% departed (gate-open ×3), collection at 0.7,
diagnostics at 0.85. Run: `output/det_gate_frozen/`; sweep: `eval_det_gate/gate_sweep.txt`.

**Training.** Losses were flat for the whole run (confidence BCE 0.29–0.44, committed-slot
term 0.17–0.40, no trend). The head recalibrated downward: false stops on correct commits
at 0.7 rose from 2% (source run) to 5–8% on training batches, the 32-seed rollout
diagnostic fell from 0.60 coverage at step 500 to 0.08 from step 2,000 on (precision
0.997), and 64-seed collections at 0.7 shrank from 4,045 states to 1,387–2,066 against
2,100–3,000 for the source run. Five percent departed states was not enough of a
reduction from v7's over-stopping recipe once the 25% drift stratum was added.

**Paired sweep.** Both checkpoints were rolled out with the gate off on the same 48
calibration traces (600-voxel cap, flow seed 0). Because the generator was frozen they commit
the identical path from identical states: 5,870 paired decisions, 5,801 on track, 69 departed.
At matched false-stop rates on correct sub-1-voxel states the fine-tuned head is no better:

| model | threshold | false stops (<1 vox, correct) | closed at 1.5–2 vox | closed off-track |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.85 | 0.6% | 5.1% | 34.8% |
| frozen | 0.30 | 0.7% | 7.0% | 30.4% |
| baseline | 0.90 | 1.0% | 7.0% | 37.7% |
| frozen | 0.50 | 1.6% | 15.3% | 40.6% |

AUC of first-plane confidence for four-plane correctness: 0.695 → 0.662 overall, 0.913 →
0.860 below 1 voxel, 0.648 → 0.613 at 1–1.5, 0.446 → 0.573 at 1.5–2 (157 states, both
near chance). On-track vs departed AUC 0.690 → 0.686. Confidence fell by 0.10 on correct
sub-1-voxel states and by 0.15–0.20 on drifted states whether the committed path was right
or wrong: the head learned that drifted states look different, not which continuations
fail, and even that does not move the operating curve.

**Conclusion.** With fixed features, targeted gate supervision does not improve stopping.
The gate is feature-limited: the flow's whole sample cloud leaves the fiber and the gate is
asked to detect the model's own error from the features that produced it. The joint arm
was not run. Deployable policy stays the 50k checkpoint with the deterministic path at
0.85–0.90 (or conf4 ranking at 0.7). Next compute goes to reducing drift, starting with
the backward-context crop from the assessment (45% of history tokens fall outside the
current crop). Caveats: one flow seed, 48 traces, 600-voxel cap; the 1.5–2 and off-track
bins hold 157 and 69 decisions.

## Normalized future flow and all-sample scoring (v10, 2026-09-24)

The current implementation adds per-plane residual standardization around a
straight-ahead prior. Scales are fitted from 2,048 training-loader states,
floored at one trace voxel, and saved with the run. Training uses 32
stratified time/noise draws, group norm, zero transformer dropout, and a
1,000-step warmup. Every future plane has an always-valid image observation
at the prior mean, with a 3 × 3 lateral patch at configurable voxel pitch.
The sampler uses four explicit midpoint steps (eight field evaluations).

All 16 generated paths are scored. Mode selection and its distinct-index
fallback are removed. Generated, teacher and replay candidates all carry
measured sample support; generated paths exclude their own support vote.
The ranking objective and partial-annotation behavior are unchanged. Live
weights train the scorer; an EMA ramps to decay 0.999 and supplies collection,
diagnostic and evaluation weights. Checkpoints save both copies. No
previous-architecture loading path is provided.

**Prior mean and crop censoring (same day).** The first v10 draft set the prior
mean by extrapolating a weighted observed-history tangent. Measured on 3,000
loader states under the launcher settings, per-axis residual RMS at planes
1/4/8/16/32/64 was 1.05/1.11/1.75/4.35/9.41/17.37 straight ahead and
1.08/1.46/2.36/4.80/9.63/17.04 with the tangent mean: no gain at the far
planes and extra spread at planes 2 to 16. The lateral slope of the backward
tangent had median 0.18 and 0.47 at the 90th percentile, where the far-plane
mean hit the crop-edge clamp. Wobble was not the cause; the same held with
wobble off. The prior mean is now straight ahead in the frame.

Prefix crop censoring is now enabled in the flow loss and in scale
calibration: known targets outside the crop half-width minus the flow patch
radius were about 10% of plane-64 targets and carried about 57% of that
plane's second moment. Restricting the residual to observable targets lowers
the far-plane RMS from about 17 to 11.8; the near planes are unchanged. Once
an annotated plane is outside that extent, every later plane is censored even
if the curve re-enters. `flow_censored_fraction` and `censored_counts` (in the
calibration record) report the effect. Probe: session scratchpad
`sigma_probe.py`; the numbers are loader statistics, not model results.

This is an implementation change, not a measured production-quality gain;
held-out rollout and GPU throughput comparisons still require a fresh run.

### Metrics only on logging steps and 16 training draws (2026-09-24)

The training loop now requests detailed loss metrics only on `--log-every`
steps and the final step. Other steps use the same loss with no diagnostic
candidate selection or scalar extraction. Direct `loss_fn` callers still
receive metrics by default. Model, CLI and launcher defaults now use 16 flow
draws instead of 32; the number of sampled candidates and midpoint steps stay
at 16 and 4. Fewer draws change the stochastic gradient estimate and random
number consumption, so the resulting training trajectory is not identical.
Existing processes retain their loaded code and configuration.

CPU loss-only benchmark: seed 73, single thread, batch-one curved synthetic
fixture from `tests/test_future_flow.py` (8 future planes, 6 generated paths,
5 teachers and 6 masked replay paths), 20 warmups and 200 timed calls. Model
forward and backward, data loading, and the draw-count change are excluded.
The CPU profiler found 52 `aten::item` calls with metrics, zero without; after
the change, logged calls remain dominated by tensor reductions and indexing.

| Loss call | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Before, always computing metrics | 0.958 | 0.958 | 0.966 |
| After, logging step | 0.958 | 0.959 | 0.968 |
| After, non-logging step | 0.192 | 0.191 | 0.197 |

Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. python /tmp/bench_fiber_metrics.py`
(session scratch benchmark). This is not an end-to-end GPU speed measurement;
CUDA was unavailable in the execution environment.

Validation: 52 passed, 11 CUDA-only tests skipped. Regression tests require
bit-identical losses and parameter gradients with metrics enabled/disabled for
known, unknown and off-track targets, and verify interval/final-step logging
through the training entrypoint. Run from this directory:

```bash
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. \
MPLCONFIGDIR=/tmp/fiber-metrics-mpl NUMBA_CACHE_DIR=/tmp/fiber-metrics-numba \
python -m pytest tests -q -o cache_dir=/tmp/fiber-metrics-pytest-cache
```

## Future-only flow with fixed observed history (2026-09-24)

`future_flow_v9` removes anchor/history reconstruction. The flow reads the
actual current point and supplied recent history as fixed conditioning tokens,
with image features sampled there. Only future-plane lateral coordinates are
generated. The scorer also reads observed history only and measures the first
step from the actual current point. GT history is used for observed-drift
diagnostics only. `--recent-history-points` replaces `--clean-points`; the
production preset keeps 32 recent history points and a 64-point forecast.

Candidate selection now excludes already selected indices from its fallback.
Full-path distance breaks ties between identical commit prefixes, preserving
different tails for the scorer. Even identical samples occupy different indices;
this does not claim that their geometry is distinct. A batched regression with
six equal prefixes and different tails now selects indices beginning `[0, 5]`
instead of repeatedly selecting index zero.

Absent history is excluded from transformer attention in both training and
inference. Partial future annotations train the available subpath: unannotated
tokens are excluded as attention keys and from the velocity loss. Their padding
cannot influence known points, and is not interpolated toward a fabricated
target. At inference the requested full horizon is generated without an
annotation mask. This removes the padding dependency; held-out rollouts still
need to establish the quality of full-horizon generation near annotation ends.

History correction outputs, metrics, audit fields and plot overlays have been
removed. Observed-history drift metrics remain. Curve plots now consume flow
training records. Follower checkpoints carry the new architecture identifier
and require fresh training. Beam checkpoint compatibility remains deferred;
its current trainer only follows the renamed configuration argument.

Validation (existing environments, no installs):

- 38 focused regression tests passed, including candidate ties, masked-token
  value/gradient isolation, independence from GT history, observed-history
  conditioning, gradient separation, checkpoint/tracer/audit integration,
  diagnostic rendering and curve fitting. Command from this directory:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_CACHE_DIR=/tmp/fiber-flow-review-numba MPLCONFIGDIR=/tmp/fiber-flow-review-mpl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src python -m pytest -q -p no:cacheprovider tests/test_future_flow.py tests/test_history_confidence.py`.
- Escalated CUDA/BF16 validation on the RTX 5090, PyTorch 2.12.1+cu130,
  exercised the production input `[2,3,208,128,128]`, 16 future samples,
  six candidates, eight integration steps and eight training draws. One row
  had missing history and a partially annotated future. Two AdamW steps had
  finite losses 4.612677 and 4.351247 and finite gradients for every parameter.
  Model size: 2,718,821 parameters. Peak allocated memory in this synthetic
  check: 9.79 GiB; this is not a controlled performance comparison with v8.
- CUDA probes verified masked future coordinates cannot influence known
  velocities, including NaN padding; unique candidate indices; pinned forward
  planes; exact repeated seeded sampling; and exact sampling after checkpoint
  reload. A separate CUDA/BF16 synthetic curve fit took 450 steps: first/last
  five-step mean loss 0.9775/0.0894, best candidate mean point error 0.3601
  voxels. These checks establish execution and synthetic fitting, not trained
  accuracy on real fibers.

CUDA reproduction (run with driver access):

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python /tmp/future_flow_cuda_check.py
```

The CUDA harness and `/tmp/future-flow-v9-cuda/smoke.pt` are local temporary
artifacts. The tracked regressions are `tests/test_future_flow.py` and
`tests/test_history_confidence.py`. No full training run was launched.

## Joint flow-matching polylines replace the recurrent decoder (2026-09-24)

Implemented `joint_flow_v8` before any `direct_paths_v7` run was trained, so
there is no v7 result to compare against and no checkpoint adapter. The v7
decoder was a per-candidate GRU unrolled over 64 planes with six mode
embeddings, a whole-path winner-takes-all coordinate loss and a separate
conv1d history cleaner joined to the decoder by a fitted tangent. Its known
weaknesses were structural rather than measured: a fixed mode count under
WTA (dead modes), a mean-over-modes trunk term that pulls every mode onto the
average of two plausible branches when the near-term continuation is
ambiguous, and 64 recurrent updates between distant evidence and the first
step. A relaxed-WTA trunk (`--trunk-loser-weight`) was added briefly and then
removed with the decoder.

What changed (`model.py`, `supervision.py`, `train.py`, `trace.py`):

- `PathFlow`: a non-causal transformer velocity field over the joint polyline
  (anchor, 32 past points, 64 future planes = 97 tokens). Anchor/past tokens
  are free in three axes; future tokens are pinned to their forward plane and
  move laterally. Prior: Gaussian (σ 4 voxels) around observed history where
  supplied, straight behind otherwise, straight ahead for the future. Tokens
  carry their coordinate, prior mean, image stencil at the coordinate,
  observation flag, time and the history context. Rectified-flow MSE in prior
  units, 8 (time, noise) draws per example against one image encoding.
- Candidates are modes of 16 samples: density-peak selection with suppression
  at 1.5 voxels RMS over the commit window, leftover slots filled by
  farthest-point. The cleaned history is the sample mean of the past tokens.
  Per-candidate support (sample agreement) is logged as `candidate_support`
  and `selected_support`; it is not a scorer input yet.
- The scorer, labels, teacher candidates, DAgger replay, tracer and
  diagnostics are unchanged. Candidates reach the scorer without gradient.
  `TraceParams.seed` seeds one generator per trace call.
- Removed: `PathDecoder`, `predict_clean_history`/`clean_head`,
  `coordinate_loss`, `bounded_vector`, the clean-history loss term and CLI
  options `--clean-weight --clean-tangent-points --max-history-correction
  --max-lateral-slope --path-sample-radius --trunk-loser-weight`. Added
  `--flow-layers --flow-heads --flow-steps --flow-samples --flow-draws
  --prior-scale --candidate-separation --flow-weight`. The launcher is now
  `scripts/launch_ct_flow.sh`. The beam re-ranker keeps sharing the encoder
  (`encode` now returns features and the history context).

Validation (existing environment, no installs):

- `tests/test_joint_flow.py` (new) and `tests/test_history_confidence.py`:
  30 passed, 1 skipped (`vc.fiber_trace` unavailable). Covered: future planes
  pinned in samples and candidates, seeded reproducibility, flow gradients
  into flow/image/history parameters while ranking and confidence leave the
  flow untouched, zero loss and gradient for unknown/departed states, unknown
  tails and absent history censored, prior construction, synthetic curved
  continuation fitted in 150 steps (loss below 0.6× initial, best mode within
  1.5 voxels), mode selection with support, config validation, beam subclass,
  training entrypoint and checkpoint roundtrip.
- Broader `tests/neural_tracing/test_fiber_follow*.py`: 59 passed and the same
  14 pre-existing failures as before this change (stale heatmap-era tests);
  one stub policy in that file now accepts the `generator` argument.
- Production-shape BF16 GPU smoke on the RTX 5090 with synthetic data,
  `[2,3,208,128,128]`, six candidates, 16 samples, 8 steps, 8 draws:
  2,723,173 parameters (880,899 in the flow). Five optimizer steps had finite
  losses 4.42 → 4.18; steady train step 0.19 s; peak allocated/reserved
  20.91/22.06 GiB, matching the v7 footprint. Batch-one inference with
  sampling took 0.028 s per decision and 2.50 GiB peak; a repeated seeded
  forward reproduced the samples bit for bit. Script:
  the session scratchpad `flow_smoke.py`. These are execution checks, not
  evidence of improved tracing.

Fresh run (not launched by this change):

```bash
bash scripts/launch_ct_flow.sh ct0_presence_flow_v8
tail -F output/logs/ct0_presence_flow_v8.log
```

Judge it on commit-horizon `oracle_recall`, `oracle_error`,
`clean_current_error` and the gate metrics, which isolate the swapped modules.
Operating parameters that did not exist before: `--flow-samples`,
`--flow-steps` and `--candidate-separation` at inference, alongside the
confidence threshold.

## Direct history-anchored coordinates; Gaussian prediction removed (2026-09-24)

Implemented `direct_paths_v7` after the stopped
`ct0_presence_h32_f64_w128_v6` run showed large errors at the first point after
history. In the old proposal mechanism, eight global heatmap peaks were chosen
before applying the history-distance preference. A synthetic reproduction with
an accurate central peak just below eight remote peaks left all six candidates
19.8 voxels away at their first point. Better confidence thresholds cannot
recover a continuation that never reaches the candidate set.

The follower now generates six trainable coordinate paths from the cleaned
current point, conditioned on observed/cleaned history and locally sampled image
features at every future plane. Gaussian prediction heads, targets, losses,
peak selection, their CLI options and the tube launcher have been removed.
Observed history is still rendered as a smooth input channel. No checkpoint
adapters were added. The independent beam method owns its optional tube head
and target rendering under `beam/` (checkpoint schema `beam_rerank_v2`).

Key choices:

- Keep the wide 208×128×128 CT + presence crop, 32 voxels visual history,
  64 one-voxel forecast planes, 32-point cleaner, and bidirectional scorer.
- Bound each clean correction to four voxels and each lateral decoder step to
  two voxels. The first-point distance from the observed anchor is at most
  6.08 voxels at the preset spacing. Local image samples use a 3×3 stencil of
  radius two voxels. The bound is a geometric assumption; log
  `target_step_limit_fraction` to detect targets it cannot represent.
- Coordinate Smooth-L1 selects one whole-trajectory winner, adds 0.5 times
  the first-four-point loss for all candidates, and 0.25 times step-vector
  supervision. Unknown suffixes and departed states have no coordinate loss;
  out-of-crop but annotated targets remain supervised.
- Detach proposal coordinates at the scorer input and in label construction.
  Ranking/confidence train the scorer and shared encoders; direct coordinate
  supervision trains path placement. Teacher candidates only train scoring.
- Clean-history loss now excludes history points that were not supplied.
- Monitor first-plane error, anchor error, maximum first jump, coordinate loss
  components and endpoint spread. Mode diversity is learned, not forced.

Validation: 31 focused tests passed, including bounded first steps under extreme
head outputs, coordinate gradients into decoder/history/image features,
synthetic curved-path fitting, whole-path assignment, unknown-target censoring,
teacher independence, batch independence, history masks and checkpoints.
The native `vc.fiber_trace` binding is unavailable in the test environment; the
separate beam model/head was checked without an end-to-end native beam run.

Production-size BF16 smoke used a real annotated fiber
`anon_20260902T164827563_000384.json` and real CT/presence, repeated to batch two:
2,137,331 parameters; input `[2,3,208,128,128]`; all 64 planes known;
two optimizer steps had losses 4.084775 and 3.998077 and finite gradients for
every parameter. Peak allocated/reserved GPU memory was 20.94/22.10 GiB.
A batch-one inference process ran with the training allocations retained
(2.38/3.31 GiB allocated/reserved for inference). A subsequent independent
collection from the new checkpoint completed two short traces and wrote four
replay states. These are execution checks, not evidence of improved tracing.

Reproduction artifacts: `/tmp/fiber-follow-v7-smoke.py`,
`/tmp/fiber-follow-v7-collector-memory.py`, `/tmp/fiber-follow-v7-smoke.log`,
`/tmp/fiber-follow-v7-smoke/decisions.npz`, and
`/tmp/fiber-follow-v7-tests.log`. These temporary artifacts are local to this
validation session. The tracked regression tests are
`tests/test_direct_paths.py` and `tests/test_history_confidence.py`.

Fresh run (not launched by this change):

```bash
bash scripts/launch_ct_paths.sh ct0_presence_paths_v7
tail -F output/logs/ct0_presence_paths_v7.log
```

Preset remains batch two for 50k steps (100k examples), collector/diagnostic
batch one and 32 diagnostic seeds. Compare at matched examples and held-out
rollout length precision/coverage. Re-sweep confidence after paths improve;
the earlier threshold sweep remains historical evidence for its own checkpoint.

## Correcting lateral clipping in the 64-voxel forecast (2026-09-24)

The running `ct0_presence_h32_f64_v6` exposed a crop-support problem. At step
2700, the last 20 logged training batches averaged 19.6% of known dense future
crossings outside the image crop and 21.7% outside the heatmap; the previous
16-voxel-horizon run had roughly 0.4–0.5%. Batch images show genuine clipping
and proposals continuing on unrelated visible structures after GT leaves view.
The initial long-horizon preset retained a 64-sample lateral width and
4°/10°/20° heading-noise scales while quadrupling the forecast horizon.

A geometry-only audit used seed 9301, 1024 length-weighted fresh-state requests
(935 passed the original crop's holdout check), and 1002 sampled on-track states
from the active replay caches. All widths were evaluated on the same existing
frames/labels; no CT reads, network scores, or retraining enter these results.
The audit does not simulate the wider crop's revised state-admission footprint.

| Lateral samples | Heading-noise scales | Fresh crossings OOB | On-track replay crossings OOB |
|---|---|---:|---:|
| 64 | 4/10/20° | 12.94% | 23.77% |
| 64 | 2/5/10° | 10.23% | unchanged frames |
| 64 | 0/0/0° | 9.30% | unchanged frames |
| 96 | 4/10/20° | 5.59% | 12.48% |
| 128 | 4/10/20° | 2.71% | 7.19% |
| **128** | **2/5/10°** | **2.06%** | **7.19%** |

Reducing fresh augmentation alone cannot address curvature or policy-heading
errors in replay. The corrected launcher therefore uses crop **208×128×128**
at spacing 0.5, still 32 voxels behind and a 64-voxel forecast. Lateral image
support is ±31.75; `heat_bins=125` covers ±31 at the same 0.5 spacing. Heading
noise becomes 2/5/10°. Architecture and target semantics are unchanged; known
geometry outside the crop remains known geometry. This reduces, but does not
eliminate, clipping (about 15.2% of replay crossings in the final 16 voxels are
still outside the wider crop in this audit).

The wider crop has 4× the voxels of the initial v6 crop. Preset batch sizes are
2 training / 1 collection / 1 diagnostic; all 32 diagnostic seeds remain.
There is no gradient accumulation. At 50k steps this means 100k examples, versus
400k for the initial v6 preset; `--steps 200000` matches that example budget.
These GPU batch sizes are provisional until a memory check can run after the
current narrow job is stopped. The running job is not altered by this preset.

Validation: a production-size CPU float32 forward/backward optimizer step
with native CT/presence from `anon_20260902T164827563_000384.json`, input shape
`[1,3,208,128,128]`, all 64 planes annotated, loss 3.25553, finite gradients for
every parameter, and a v6 checkpoint roundtrip. This is a shape/data check,
not a quality result or GPU throughput measurement. Existing focused tests
and launcher shell syntax are also checked.

```bash
# After stopping the narrow run, launch a fresh run with the corrected preset:
bash scripts/launch_ct_tube.sh ct0_presence_h32_f64_w128_v6
# Geometry audit and full-size CPU validation used the existing environment:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/sean/Documents/villa4/vesuvius/.venv/bin/python /tmp/fiber-follow-context-audit.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/sean/Documents/villa4/vesuvius/.venv/bin/python /tmp/fiber-follow-v6-wide-smoke.py
```

Audit script/results: `/tmp/fiber-follow-context-audit.py` and
`/tmp/fiber-follow-context-audit.log`. CPU smoke script/results:
`/tmp/fiber-follow-v6-wide-smoke.py` and `/tmp/fiber-follow-v6-wide-smoke.log`.
The fresh audit is repeatable with the fixed seed; replay results depend on
which caches were active when the audit ran.

## 32-voxel visual history, 64-voxel forecast, joint scoring (2026-09-24)

Fresh architecture `spatial_candidates_v6`, requested after the v5 run reached
8k. V5 cleaning was improving while local proposal metrics had plateaued:
selected error averaged 0.975 over 3k–5k and 0.969 over 6.5k–8k. Fixed-32-seed
rollout diagnostics at threshold 0.7 remained variable: 6k coverage/precision
40.2%/91.9%, 6.5k 63.7%/82.5%, 8k 23.1%/85.5%. These observations motivate
the experiment; they do not establish the cause of the variation.

Configuration: native CT + presence + own-history tube; crop 208×64×64,
spacing 0.5, behind 64 samples (=32 trace voxels), forward image extent 71.5.
Predict 64 future points at spacing 1; clean and explicitly score against
32 past points plus current. Keep 128-coordinate geometric history, six
proposals, widths 24/64/128, hidden 128, six-point tangent, and at most four
committed points. Lateral support stays ±15.75; monitor crop/heatmap OOB rates.

Both ranking and prefix confidence now compare history with the entire
proposed continuation. History uses an ordered GRU plus masked token pooling;
future tokens use a history-initialized forward GRU and a backward GRU. A
fusion head combines both directions, pooled future evidence, and explicit
history at each point. This provides short gradient paths from older history
and distant future points. Prefix correctness labels, unknown-end censoring,
and per-candidate state isolation are unchanged. A wrong distant suffix does
not label an otherwise correct early prefix negative. The beam model removes
these spatial-only modules and retains its existing scoring design.

This is a fresh run with no earlier-checkpoint adapters. Launch:

```bash
bash scripts/launch_ct_tube.sh ct0_presence_h32_f64_v6
```

The crop has 3.25× as many voxels as v5. On the available RTX 5090 (32 GB),
a production-shape batch-32 forward ran out of memory. Batch 16 passed two
forward/backward optimizer steps but peaked at 22.37 GiB allocated / 25.89 GiB
reserved, leaving too little margin for simultaneous collection. The preset
therefore uses batch 8 and collection batch 4, without gradient accumulation.
This halves/quarters examples per step relative to batch 16/32; compare runs
by examples and elapsed time as well as steps. The 50k-step budget gives 400k
examples. No learning-rate or normalization changes were made.

Validation of the final model (1,902,846 parameters):

- 19 focused tests passed: both histories affect ranking/confidence; history
  beyond eight points affects relative ranks; a changed distant suffix affects
  first-point confidence; candidate permutation/isolation and masked-history
  behavior; BF16 backward; long-horizon prefix labels and unknown-end censoring;
  native CT/presence alignment; checkpoint/tracing/training integration; batched
  diagnostics preserve all seeds and metrics.
- Broader suite: 88 passed, the same 15 failures recorded before this change,
  and one beam module skipped because `vc.fiber_trace` is unavailable here.
- Two GPU optimizer steps with the production shape `[8,3,208,64,64]`, native
  CT and presence from `anon_20260902T164827563_000384.json` (one real sample
  repeated for capacity testing). All parameter gradients finite; loss
  3.27232 then 3.21446; all 64 planes known; v6 checkpoint roundtrip passed.
  Peak training allocation/reservation: 20.97/22.11 GiB. A separate batch-4
  scorer passed while the training allocations remained resident, using
  2.38/3.30 GiB allocated/reserved. This checks simultaneous residency, not
  end-to-end asynchronous collection throughput.
- Validation rollouts are now chunked (`--diag-batch 4` in the preset), retaining
  all 32 diagnostic seeds. This avoids a 32-seed forward at the first diagnostic.

Exact verification commands from this directory (the existing shell Python
has pytest; the training virtualenv supplies the GPU environment):

```bash
PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  tests/test_history_confidence.py
SMOKE_BATCH=8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/sean/Documents/villa4/vesuvius/.venv/bin/python /tmp/fiber-follow-v6-smoke.py
```

Logs: `/tmp/fiber-follow-v6-final-tests.log` and
`/tmp/fiber-follow-v6-smoke-final.log`. The smoke script and collector memory
probe are `/tmp/fiber-follow-v6-smoke.py` and
`/tmp/fiber-follow-v6-collector-memory.py`.
No training-quality or throughput improvement is claimed by the smoke tests.

## CT + presence and observed/cleaned history confidence (2026-09-24)

Implemented for the next fresh run as `spatial_candidates_v5`; no training
quality result is claimed yet. The stop-policy sweep below motivates stronger
history evidence for confidence, because changing rank alone seldom rescues
the examined wrong selections.

- Confidence reads a masked sequence of observed and cleaned recent history,
  with image features at both positions and their correction vectors, before
  processing each candidate. Ranking still pools candidate features.
- Initial proposal tangent is fitted over up to six corrected points; it can
  be compared with `--clean-tangent-points 2`. First-plane extrapolation uses
  the corrected anchor's forward coordinate. Stored rollout geometry is not
  rewritten by the cleaner.
- `launch_ct_tube.sh` now supplies native CT, independently sampled fiber
  presence, and the history tube (`--inputs ct+presence`). Presence is a learned
  input only. Neither direction arrays nor presence-derived labels are used.
- Position/tangent improvement and correction magnitude are logged with valid
  counts. `eval_ckpt.py --history-audit` stratifies held-out decisions by gate
  errors, drift, and departure, without changing the trace. Unknown boundaries
  censor the audit; departed clean geometry is not scored as supervised truth.

Next comparison: fresh V5 CT + presence versus V5 CT-only, followed by the
two-point/six-point tangent comparison. Keep ranking temperature, replay
settings, and evaluation seeds fixed. Sweep confidence and compare coverage at
matched wrong length. Historical checkpoint results do not establish the new
head's calibration. V5 has no checkpoint adapters.

Validation: 15 new CPU tests pass, including independent CT/presence sampling
under rotation, both crop-sampler paths, history masking and recurrent-state
isolation, BF16 confidence forward/backward with native CPU kernels, corrected
anchor extrapolation, audit censoring, a checkpoint roundtrip, matching training
and rollout inputs, and a two-step training-entrypoint smoke. The broader
follower/CT/history suites report 84 passed and the same 15 failures observed
before this change (stale clean-head interfaces and the removed replay helper).
Pytest was loaded from the existing uv cache; no packages were installed.

A CPU forward/backward pass with the actual 64-cube configuration
(`widths=24,64,128`, hidden 128, batch 1, six candidates) used a real sample from
`anon_20260902T164827563_000384.json`, native level-0 CT, and the presence zarr.
The input shape was `[1,3,64,64,64]`, presence ranged from 0 to 0.9995, and all
gradients were finite. Model size is 1,705,214 parameters. This is a pipeline
smoke check with random weights, not a training-quality or throughput result.
Local logs: `/tmp/fiber-history-verified-tests.log` and
`/tmp/fiber-history-real-smoke.log`.

## Stop policy sweep on the live CT run (2026-09-24)

`ct0_tube_64_clean_history`, checkpoint 24,000 of 50,000. 32 held-out fibers,
one seed each, both directions (64 rollouts), 400-voxel limit, `rollout_diag`
scoring (coverage capped at 400). Precision is length precision. Harness:
`ModelTracer` with `TraceParams`; the new `stop_patience` / `commit_floor`
fields commit a single point on a would-stop call until `stop_patience`
consecutive would-stops, unless the top candidate's first-point confidence is
below `commit_floor`. Defaults reproduce the immediate stop.

| confidence | patience / floor | coverage | precision | diverged | wrong len | followed median | ends by stop |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.7 | 1 / - (current default) | 0.122 | 0.864 | 0.06 | 6.6 | 12.6 | |
| 0.6 | 1 / - | 0.167 | 0.902 | 0.09 | 6.5 | 20.8 | |
| 0.5 | 1 / - | 0.202 | 0.851 | 0.16 | 13.4 | 30.2 | |
| 0.4 | 1 / - | 0.247 | 0.850 | 0.23 | 16.7 | 32.8 | |
| **0.3** | 1 / - | **0.323** | **0.852** | 0.28 | 21.7 | 41.2 | |
| 0.2 | 1 / - | 0.390 | 0.837 | 0.30 | 29.1 | 78.9 | 46/64 |
| 0.7 | 2 / - | 0.164 | 0.847 | 0.09 | 10.7 | 24.3 | |
| 0.7 | 3 / 0.3 | 0.207 | 0.822 | 0.13 | 16.4 | 30.9 | |
| 0.5 | 2 / - | 0.256 | 0.814 | 0.23 | 21.9 | 41.7 | |
| 0.5 | 3 / 0.2 | 0.318 | 0.859 | 0.20 | 19.7 | 47.0 | |
| 0.4 | 2 / - | 0.342 | 0.843 | 0.27 | 24.2 | 50.6 | 53/64 |
| 0.3 | 2 / - | 0.382 | 0.799 | 0.39 | 37.5 | 72.9 | 45/64 |
| 0.3 | 3 / 0.15 | 0.386 | 0.817 | 0.33 | 33.8 | 72.1 | 46/64 |
| 0.3 | 4 / 0.1 | 0.390 | 0.758 | 0.38 | 48.5 | 72.9 | 42/64 |

Reading: lowering the threshold from 0.7 to 0.3 multiplies coverage by 2.6 at
the same length precision; below 0.3, or with patience on top, coverage gains
are small and precision drops. Even at 0.2 about 70% of rollouts still end by
a confidence stop with a median of 75 voxels followed: per-decision false
stops of a few percent compound over dozens of decisions, so the head's
discrimination, not the threshold, is the remaining limit. Recommended
operating point for this checkpoint: `confidence 0.3`, default patience.

Gate calibration on the run's own replay cache (`decisions_023500`, 1,215
on-policy states, 44% exploratory, 12% off-track; labels recomputed with
`candidate_labels` for the rank-selected proposal, first prefix point):

| threshold | false stop P(stop \| correct) | false continue P(go \| wrong) |
|---:|---:|---:|
| 0.3 | 0.03 | 0.59 |
| 0.5 | 0.05 | 0.51 |
| 0.7 | 0.09 | 0.40 |

The selected proposal is wrong in 25% of these (failure-enriched) states, and
in only 12% of those is another proposal correct, so ranking is not the
problem there; the confidence head is. Of the 947 hard replay states, 16% are
off-track (negatives only), 18% are on-track would-stops, 98% of which carry
a known GT continuation (the intended corrective positives), and 67% are the
48-voxel pre-departure window.

Changes made for the next run (no effect on the running process):
`--rank-temperature` (default 20, was a fixed 5) keeps the listwise ranking
target peaked once proposals are good, with `rank_target_entropy` logged
(1 = uniform target); `gate{0.3,0.5,0.7}_{first,commit}_{false_stop,false_go}`
and `gate_*_negatives` are logged every step; `TraceParams.stop_patience` and
`commit_floor` are available to rollouts, collection and `eval_ckpt --params`.

## Beam re-ranker method: bindings and pipeline validation (2026-09-24)

New training method in `beam/` (see `README.md`, "Beam re-ranker"). The
volume-cartographer beam tracer gained a per-request beam hook
(`FiberTraceBeamHookOptions` on the one-way, segment, whole-fiber and
extrapolation requests) and nanobind bindings `vc.fiber_trace`. With no hook
the C++ search is unchanged; all 51 `test_fiber_trace3d` and 9
`test_fiber_trace_review` cases pass, including new cases that check an
observe-only or identity hook reproduces the plain trace exactly, that
replacement losses drive the prune, and that `stop` ends the trace.

Real-data checks with the dev build tree (`build-dev`, RelWithDebInfo), the
`PHercParis4-...-7ff0ce6c` prediction manifest and `las_008` normals:

| check | result |
|---|---|
| derived scales | VC trace voxel = 2 base voxels; 4 VC trace voxels per fiber_follow grid voxel |
| `whole_fiber_metric` on held-out `anon_20260815T020414697_000008.json` (9 spans) | 0 restarts from Python (3.3 s, 1 thread) and from `vc_fiber_trace_metric` (0.8 s wall, OpenMP) |
| `BeamDataset`, one process, 64-cube CT states, pool 32 | 15 states/s after a 0.5 s first chunk |
| label sanity, unperturbed starts, no hard oversampling (80 states) | hand-best candidate on-fiber in 100% of pools; 98% of labeled candidates on-fiber; hand-best max error median 0.56 grid voxels, all below 2 |
| 8-step GPU smoke, batch 16, 2 forkserver workers, shared GPU | 1,620,852 parameters, finite losses, span diagnostics with the model hook ran (1 fiber, 10 spans, 0 restarts hand and model) |

Hand-beam baseline on all 124 held-out fibers
(`beam.evaluate_spans hand --threads 6`, error threshold 20 base voxels,
200 s):

| spans | length (grid voxels) | restarts per 1000 grid voxels | span success | fibers with a restart |
|---:|---:|---:|---:|---:|
| 4,497 | 533,901 | 0.451 | 94.6% | 75 of 124 |

So the hand beam fails on about one span in twenty, and uniformly sampled
training traces rarely contain an off-fiber hand choice (about 2% of pool
entries). `beam/mine.py` therefore runs the same metric once over the
training fibers, caches the failing span indices, and the dataset traces a
mined span with `--hard-span-prob` (default 0.5). No model quality claims are
made yet; the comparison to run is `evaluate_spans` hand versus a trained
checkpoint on these 124 fibers.

Existing follower suites: the 15 failures in `test_fiber_follow*.py` are
identical with and without the `runloop.py` extraction and come from the
in-progress `clean_history` model change (`clean_points`, `replay_dict`), not
from this work. The new `test_fiber_follow_beam.py` (5 cases) and the VC
`python/tests/test_fiber_trace.py` (7 cases) pass against the build tree.

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


## 2026-09-25: single_path_flow_v11 implementation and first run

The active follower was replaced by a 16-point history-conditioned denoiser:
176×96×96 level-1 CT/presence/history crop, 128 supplied history tokens,
(24,64,128) spatial widths, six 256-wide / eight-head denoising blocks,
coarse image cross-attention, four midpoint updates from zero, and a final
confidence evaluation. No candidate/ranking/support/teacher path remains in the
active follower. The shared encoder was extracted; a saved beam state dictionary
loads strictly and reproduces its saved outputs exactly on CPU.

The before-change source archive, original 50k EMA checkpoint hashes, original
96-seed deterministic rollout results (thresholds .70/.85, cap 1200), and geometry
identities are preserved under `output/single_path_v11_baseline/`. A CPU baseline
fixture smoke evaluated four frozen augmented states at .5/.85; it is explicitly
not the final comparison. Another fixture contains 48 actual observed baseline
decisions from four calibration seeds with a 48-voxel cap. That initial fixture
used fixed per-decision support RNG; the archived rollout adapter now resets the
legacy sampling RNG once per trace, matching v10's original tracing behavior.
A subsequent GPU fixture with the original per-trace RNG preserved 52 decisions
from the same four seeds (`observed_recovery_trace_rng.npz`). The 384 augmented
recovery states are independent of either model's outputs.

`output/single_path_v11_preparation/` contains the frozen seed manifest and permanent
bank. Import scanned 131,470 training decisions, rejected 2,017 for larger-footprint
holdout overlap, removed six duplicates, and selected 20,000 unique states from
129,447 eligible states. Each stored state retains original-fiber correspondence
and source-cache/row provenance. Fixed/recent pools validate independently; the
published collection smoke contains six decisions from two training seeds.

Frozen evaluation groups: original 32 monitor fibers (32 seeds), original 48
assessment fibers (96 seeds), and remaining 44 final fibers (176 seeds).
The former deterministic evaluation script excluded assessment fibers alone;
v11 also excludes monitor fibers from final evaluation. The manifest SHA256 is
`1ae303f787f1ce611614c421b57ee44afc77bd759d192a0d277ff61239c62aaf`.

Verification command (existing environment, current checkout on PYTHONPATH):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 AGENTS_AGENT_MODE=1 \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/single-path-cuda-pytest
```

34 tests pass with GPU access; sandboxed CPU execution reports 33 passed and one
CUDA skip. Tests cover masked/unsupported history, bidirectional dependencies,
fixed coordinates, GT isolation, final-curve labels/gradient routes, exact masked
accumulation across differing censoring/departure patterns, midpoint integration,
replay import/rotation, checkpoint rejection/roundtrip, beam compatibility,
collection and synthetic parallel-fiber recovery. The synthetic test reduces
lateral MSE below .25 and below 15% of its initial value after 160 small-model
updates; it is a learnability check, not evidence of held-out improvement.

GPU preflight (requires access outside the sandbox):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. AGENTS_AGENT_MODE=1 \
  /home/sean/Documents/villa4/vesuvius/.venv/bin/python scripts/benchmark_single_path.py \
  --manifest output/single_path_v11_preparation/seeds.json \
  --out output/single_path_v11_preparation/benchmark_b2.json
```

RTX 5090, full crop, BF16, channels-last, 64 flow draws, microbatch 2 × four:
peak allocated 9,873,314,304 bytes (9.20 GiB), reserved 10,756,292,608 bytes
(10.02 GiB). Update times after warmup: .9131, .8988, .8999 seconds; cached-batch
training throughput 8.85 states/second. A complete 64-voxel trace used 16 decisions
in 1.3224 seconds (82.65 ms/decision), including crop I/O, history rendering,
eight velocity evaluations and final confidence. This is not a controlled speed
comparison to v10: architecture, crop, resolution and training objective changed.

The fresh run `output/single_path_v11_run1` was launched using `scripts/launch.sh`
with CT `/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr` and absolute
paths to the prepared bank, manifest and successful benchmark JSON. All remaining
training defaults apply (50,000 optimizer updates, effective batch eight, 2,048
calibration states, EMA .999, periodic replay and .5/.85 diagnostics). Its source
snapshot and per-file hashes are saved in the run directory. Residual-scale calibration completed and the run reached optimizer update 50
with finite losses at 7.94 sampled states/second including loading. The full
training run and locked calibration/final acceptance study remain pending; no
coverage or wrong-continuation improvement is claimed.
