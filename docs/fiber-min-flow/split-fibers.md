# Split-fiber flow experiment

Variation of accepted fractional baseline `5122bf36b`, not a replacement default.
Run settings: existing `fiber_uv_layout.split_bounds` eligible-midpoint splitter,
minimum piece/tail length1000 base voxels, minimum4 distinct perpendicular
observations per piece, no continuation penalty. Short or insufficiently
constrained original fibers remain intact. All pieces and original measurements
are retained subject to the same20 ambiguous unsigned exclusions per case.

Cases correspond to villa2's free/joint/selectable input caches. The latter two
are array-for-array identical. The third case is explicitly **joint, GT
splittable**, not selectable/disabling: flow has no piece-disabling variables.
Joint intact-GT protects all seven GT inputs from splitting; the third allows
those inputs to split. Frozen reference GT remains unsplit in every evaluation.
All pieces retain100% of original arc length; GT labels never enter splitting
or solving. Input GT identity comes from AFV metadata.

## Requested GT metrics

Same frozen reference, coverage and component alignment as accepted baseline.
Perpendicular L1 is mean absolute integer winding error over575 reference events;
parallel signs use4343 events. GT7 remains excluded from evaluation.

| Case | Original fibers → pieces | Perp L1 | Perp GT sign violations | Parallel GT sign violations | Flow time |
|---|---:|---:|---:|---:|---:|
| Free | 2053 → 2273 | 0.142609 | 4/575 | 0/4343 | 0.212s |
| Joint, intact GT | 2060 → 2291 | 0.135652 | 3/575 | 0/4343 | 0.241s |
| Joint, GT splittable | 2060 → 2421 | 0.133913 | 4/575 | 0/4343 | 0.268s |

Free and intact-GT results match their whole-fiber baseline metrics. GT-splittable
reduces perpendicular L1 by1/575, but adds one missing-next-winding violation
at event43223 (ordinary424227 versusGT1). The prior three missing-next errors
remain at events44084,44138,44940 (ordinary424632,424646,425073 versusGT1).
Free additionally retains its one actual reversal at event31689
(ordinary291441 versusGT2). All remaining deficits are one winding step.
Across all4918 events, mean absolute error is .041684/.033550/.034567 respectively;
third-case parallel L1 worsens from .020032 to .021414. This is not an overall
accuracy improvement.

There are no new disconnected components: free5, both joint2; every original
component stays connected. Thus results are not inflated by extra independently
GT-aligned gauges. Under one global gauge, free perpendicular L1 is .173913,
signs5/575 perpendicular and27/4343 parallel; both joint rows are unchanged.
Component sizes and event coverage per gauge are in metrics.json.

## What splitting did

184 ordinary fibers split in free;194 in intact-GT joint;201 total in
GT-splittable joint. The latter represents seven GT fibers as137 pieces.
No winding discontinuity is used on any ordinary fiber, even though pieces
are independent and no continuation penalty discourages breaking.

The third case uses five one-step breaks, all within connected components:

| Original GT input | Source ID | Original arc (vx) | Winding jump |
|---|---:|---:|---:|
| GT1 | 471177 | 3298.582 | -1 |
| GT2 | 471178 | 47379.397 | +1 |
| GT2 | 471178 | 48709.406 | -1 |
| GT2 | 471178 | 63229.992 | +1 |
| GT2 | 471178 | 66192.890 | -1 |

These are integer assignment changes in a common original-start chart, not
coordinate-seam crossings. They lower the optimized measurement objective from
4108.967315 to4102.354677, but do not establish that the GT traces are erroneous.
Free and intact-GT objectives remain3232.192059 and4108.967315.

The ordinary fibers involved in remaining violations do not qualify for this
split policy:291441,424632,424646 have lengths1570,1053,639vx (less than two
1000vx pieces), and425073 has only4 perpendicular observations in free /6 in
joint, fewer than the8 needed for two eligible pieces. Thus this run does not
test arbitrary breaks on those specific fibers. More aggressive splitting
would be a separate experiment.

Internal measured-constraint sign violations (distinct from requested GT signs):
free3/4283 perpendicular,0/10772 parallel; intact-GT6/4924,0/15428;
GT-splittable3/4924,0/15428.

## Implementation, validation and reproduction

`fiber_min_flow_split.py` reuses the existing splitter and records piece parent,
original-source ID and arc bounds. Each piece winding is expressed at its original
fiber start; remap endpoint indices only, preserving full-fiber and connector
seam transport. The shared `large_reference` accepts an optional piece lookup,
selecting by original source plus event arc rather than collapsing repeated
source IDs. Exact cuts belong to the right-hand piece for both solving and scoring.

```bash
/home/hendrik/.venv_las/bin/python global-fiber-min-flow/run_split_large.py --config global-fiber-min-flow/run_split_large.json
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=vesuvius/src global-fiber-min-flow/test_audit_signs.py vesuvius/tests/test_fiber_min_flow.py vesuvius/tests/test_fiber_min_flow_split.py vesuvius/tests/test_fiber_uv_bucket_split.py vesuvius/tests/test_fiber_uv_layout.py
```

Outputs: `global-fiber-min-flow/data/split_large_v1/`. Includes hashed input
manifest, configs, all solved arrays, piece maps, certificates, per-case metrics,
all scored GT events in CSV, violation details, and comparison/break records.
Original AFC remains unavailable as in baseline; use unchanged saved tensors.
No installation, GPU use, learned rejection, geometry optimization or continuation
weight tuning. Forty tests pass. Every run verifies zero-split topology and
piece-aware reference scoring against accepted whole-fiber results on actual data.
Certificates pass with largest absolute dual gap5.46e-12, no capacity violation.
Per-case total times11.61/14.69/14.88s include preparation, solving, evaluation and
an additional whole-fiber scoring equivalence check; exclude initial input hashing.
Flow times above are single-run CPU measurements, not repeated benchmarks.
