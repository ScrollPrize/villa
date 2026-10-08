# Qual supplied fibers — Paris4 crop (2026-09-23)

## Input and protocol

Input: `/home/hendrik/test_datasets/2026-09-23_quals_fibers/Paris4_EXTENDED_min10_L0.vc3d.json`.
Producer metadata: automatic extraction/merge-gap/EXTEND, model
`50M_MULTISCROLL_8000`, dataset `PARIS4_EXTENDED` revision613; no manual review.
275 fibers (149H/126V supplied tags, not used as ground truth).
Coordinates already absolute native L0 XYZ, scale1,2.4um; the producer already
multiplied L2 coordinates by4. Do not apply that transform again.

Same half-open crop `[10240,22016,6144] .. [11264,23040,7168)`, reference tag
`hendrik_crop1` in `/home/hendrik/test_datasets/2026-09-01_fiber_stack2`, and
Lasagna normal manifest `PHercParis4-20260411134726-las-sd2-5b17ff6c` as earlier runs.
The supplied fibers extend outside the crop: clipping keeps each contiguous
in-crop piece separately, without inventing connections across outside gaps.

Replay evaluates supplied geometry, not a new prediction-guided tracer.
At each reference-derived start/restart, locally acquire a nearby aligned
polyline, then follow it in the selected direction until failure/end. No
future-reference candidate ranking, warping or switching between fibers.
The shared reference replay driver supplies the same acceptance ellipse,
matching/reset and summary metric `100/(failures+1)`.

Acquisition is nearest Euclidean segment projection within the existing
20-normal/80-tangential-base-voxel ellipse and absolute tangent angle<=25°.
Ties follow source/segment order. The actual projected offset is evaluated;
sampling follows the fixed polyline in8-base-voxel arc steps and includes its
terminal endpoint. Reference resets use the shared8-base-voxel minimum advance.
Source vertex density therefore does not set the evaluation step. No prediction
field or supplied H/V tag influences candidate selection. Supplied full fibers
are retained for replay; evaluation references are clipped to the crop.

BP consumes all clipped candidates with no graph-quality filtering (there is
no comparable model/path cost for imported geometry). Same piece512,
dominance0.9, fixed-orientation sum-product-mixed and reference-assisted
oracle-inliers protocol as the beam comparison. Metrics:

- Constraint score: `-100 * problematic / retained_fulfilled`; closer to0 better.
- Oracle round0 exact-reference score: `100 * exact / (exact+wrong+missing)`,
  all26 references. Round0 follows conditioned-inlier closure; this is a
  reference-assisted diagnostic, not held-out accuracy.

## Results

| Predictions / route | Replay score ↑ | Constraint score ↑ | Exact references ↑ |
|---|---:|---:|---:|
| Classic / Fiberlet control | 14.286% | -84.42% | 84.62% |
| New u22000 / Fiberlet | 2.500% | -91.79% | 88.46% |
| Classic / VC3D beam8 | 50.000% | -240.90% | 88.46% |
| New u22000 / VC3D beam8 | 33.333% | -178.90% | 88.46% |
| Qual / supplied fibers | 0.0464% | -55.31% | 84.62% |

Qual raw results:

- Import275 fibers/129983 vertices into279 separate in-crop runs.
- Replay48/48 cases complete,24 reference runs,101.035764mm directed length;
  2154 failures, of which1832 no-aligned-candidate,284 candidate-end,38 distance
  violations.9/48 cases failure-free (18.75%);60.1385% reference length credited.
- BP graph369 pieces/2252 constraints.802 problematic /1450 retained fulfilled.
  Oracle round0:22 exact/3 wrong/1 missing, all26 denominator.
- Import0.97s, replay209.90s, BP6.31s. Replay time is exhaustive local segment
  acquisition and normal sampling, not neural inference. BP initial solve hit
  the established message limit; final conditioned oracle round0 converged.

Interpretation: sparse imported geometry gives the lowest problematic/fulfilled
ratio in these controls but has many acquisition gaps in replay. Repeated
8-voxel restart attempts in an uncovered interval each count as a failure, so
2154 failures do NOT mean2154 independent wrong branch choices. Of these only38
were distance exits after acquisition. This is not an end-to-end comparison of
callable tracing algorithms; it evaluates the provided geometry with the stated
local acquisition policy. No missing geometry is filled in and no GT-guided
matching is used to rescue gaps. Likewise BP sees far fewer constraints than
the beam cohorts, so its improved constraint ratio must be read with coverage
and exact-reference results, not by itself.

Earlier Fiberlet controls use their graph quality filter; Qual and beam do not.
The Classic rerun remains in this comparison table but is omitted from progress
plots as requested. Historical best exact-reference result92.31% remains in
those plots; the table above lists recent matched prediction comparisons.

All three plots updated:
[replay](../fiber_benchmarks/imgs/fiber_reference_replay_progress.svg),
[constraints](../fiber_benchmarks/imgs/fiber_crop_pruning_progress.svg),
[references](../fiber_benchmarks/imgs/fiber_crop_reference_accuracy.svg).
Imported geometry OBJ: `/ephemeral/hendrik/fiber_crop_bench/20260923_qual_v1/qual-crop.obj`.

## Reproduction and validation

Use approved `run_fiber3d_native_build` and `run_classic_fiber_evaluation.py
crop-benchmark --config ...` runners. Native regression tests and exact input,
binary/source hashes, raw counts and timings will accompany measured results.

Configurations under `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/`:
`crop_benchmark_qual_20260923.json`, `crop_benchmark_qual_tests_20260923.json`,
and `crop_benchmark_qual_publish_20260923.json`.
Output root: `/ephemeral/hendrik/fiber_crop_bench/20260923_qual_v1`.

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_qual_20260923.json
```

Independent native review passed. Native tests:21 reference-benchmark,
83 crop-trace,86 production trace and23 replay cases passed. Python runner6
passed. Tests include actual-offset acquisition, non-step-multiple endpoints,
missing/wrong-direction/short fibers, frozen selection despite a better future
alternative, deterministic ties, scale rejection and crop reentry.
RelWithDebInfo build; one measurement, up to128 requested workers (48 replay
cases). No inference or GPU usage required.
