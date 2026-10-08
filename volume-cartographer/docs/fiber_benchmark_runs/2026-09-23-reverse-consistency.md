# Optional forward/backward beam consistency

Same Paris4 2048³ crop and new inference as the preceding40/80,45° experiment.
Production beam8/lookahead2/step4trace voxels(8base), canonical anchor seeds,
128workers. No GT is used for candidate tracing or reverse checks.

## Method

- Trace both halves from a seed as before.
- From each endpoint trace towards that seed with the same beam and reversed
  endpoint tangent. Check each committed point against the original half.
- Use20normal/40in-sheet base-voxel ellipse radii. Project locally backwards
  on the original arc, over at most twice each reverse step's travel.
- On ellipse failure, stalled progress or early termination, retain the
  successfully verified original interval. Skip40base voxels further towards
  the seed and retry. Multiple failed attempts can create longer gaps.
- Save original geometry, not a new reverse geometry. Merge through the original
  seed only if both adjacent pieces reach it. Never bridge a skipped interval.
- Only retained pieces suppress subsequent anchor seeds, using40/80 and45°.
  Consequently another seed may supply a valid trace near an earlier gap.
- Per-piece cost integrates original step losses. Artifact provenance retains
  the original seed key/position when a piece requires a new local storage point.

The check is at beam-step cadence, not a proof of continuous containment.
Immediate missing reverse-start predictions are failed checks, not whole-run
errors. Other sampler/config errors still propagate. Very short intervals get
at least two steps of reverse budget. All distances are base voxels.

## Reproduce

Build (no installs):

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
```

Run the tracing and unchanged BP removal benchmark using the approved runner:

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_beam2048_reverse_20260923.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_bp2048_reverse_20260923.json
```

Outputs use root`/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1`,
prefixes`beam-crop-new2048-coverage40x80-angle45-reverse-v2` and
`bp-beam-new2048-coverage40x80-angle45-reverse-v2`.
The unversioned reverse pilot used an earlier build and is not the final comparison.
Runner records full commands, binary/shared-library/input hashes, patch, walltime
and peak RSS. RelWithDebInfo, one run per configuration; timings are not controlled
repeated performance measurements.

## Validation

86native crop tests and20CLI tests pass. Includes unchanged reversible geometry
and costs at non-unit scale, partial divergence, repeated backoff, immediate
missing reverse start, tighter versus looser acceptance, actual multiworker
equivalence, parameter validation, and split artifact provenance roundtrip.
At measurement time reverse checks were opt-in. Following the result, the user
requested promotion: CLI dense beam crop tracing now enables them by default.
Shared C++ config, replay and fiberlet defaults remain unchanged.

## Results

|Metric|40/80,45° baseline|Plus reverse20/40,backoff40|
|---|---:|---:|
|Output pieces|1484|2778|
|Stored length(base voxels)|2301369.24|2225819.91|
|Tracing walltime(seconds)|181.13|147.42|
|Single best piece has≥95%reference coverage AND≥95%containment|88.46%(23/26)|76.92%(20/26)|

Final reverse run:1473attempted anchor seeds,4206reverse attempts,1315failed
attempts,51445.2986base voxels skipped. Retained+skipped=2277265.2115base voxels:
**2.2591%** of the checked forward geometry removed. This denominator is not the
old run length, because changed suppression changes which anchors are accepted.
3353candidate computations included1880speculative candidates discarded on
ordered commit; reverse counters exclude those discarded candidates.

Single-piece matching is deliberately sensitive to splitting and is not union
coverage across fragments. In particular reference5.5 loses its full-span match,
and5/9 become partial pieces. Do not equate the23→20drop with lost total
coverage. All three closest candidates per reference, means/medians/coverage,
are saved in`2026-09-23-reverse-consistency-matches.json`, using the existing
length-weighted source-polyline diagnostic and40/80 evaluation ellipse.
Neither validation nor BP uses those diagnostic matches for tracing.

## BP removal comparison

Same reference-conditioned oracle-inliers protocol, weights and iteration
limits as the preceding45° baseline. This is NOT independent reference accuracy
from unassisted initial BP.

|Metric|40/80,45° baseline|Plus reverse20/40,backoff40|
|---|---:|---:|
|Initial constraints|163183|161136|
|Removed incident constraints|59560|39199|
|Retained infringed constraints|43105|45452|
|Retained defect constraints|0|0|
|Total problematic constraints|102665|84651|
|Retained fulfilled constraints|60518|76485|
|**Score:−100×problematic/retained fulfilled**|**−169.6437%**|**−110.6766%**|
|Removed/initial|36.4989%|24.3267%|
|Problematic/initial|62.9140%|52.5339%|
|Reference-conditioned correct|100%(26/26)|100%(26/26)|
|BP walltime(seconds)|339.22|419.13|

This is a substantial improvement on this crop's removal score. It does not
prove all retained fragments are correct or that the unassisted solve improves:
the graph and suppression decisions change, and this metric uses GT-assisted
conditioning/pruning. Initial unconditioned BP reached the500-message cap.
Conditioned oracle reports converged and accepted. Fresh retained-graph solve
also reaches its iteration cap; no claim of global optimality.

No full-volume benchmark performed. The new option is dense-crop-only;
imported fibers and fiberlet crop tracing reject it. The
reported timings are single uncontrolled runs, not evidence of a speedup.

## Default promotion

Follow-up unassisted result:10/26correct(38.46%),versus7/26(26.92%)before reverse
checking. See[full independent ladder](2026-09-23-reverse-consistency-independent.md).

CLI dense beam crop tracing now defaults to20/40reverse radii and40base backoff;
`--no-reverse-consistency` restores the old control. Historical beam-tracing
benchmark configs have this explicit opt-out so reruns preserve their algorithm.
New reverse experiment configs remain explicitly enabled. Coverage thresholds
and45°suppression are still specified by the chosen benchmark config.

Smoke config:`crop_benchmark_reverse_defaults_smoke_20260923.json` runs one
anchor with implicit defaults and one with opt-out on the same cached crop.
Validation:22CLI tests pass. Default smoke3.38s, one retained trace,two reverse
checks; metadata confirms20/40radii and40backoff. Opt-out smoke2.88s, one retained
trace,zero reverse checks,no reverse config in metadata. Both preserve the same
1662.1715base-voxel reversible line.86native tests passed before this CLI-only
promotion. No full benchmark rerun was needed.
