# Case31 skeleton geometry: what the prediction constrains

Date: 2026-09-19. Backend: `skeleton`. Reference: the annotated case31 fibre,
refined with the skeleton cost and its endpoints held fixed. All distances are
base voxels.

## Question

The skeleton backend traced case31 without failures, but its worst deviation
from the refined reference was 21.17, while the thresholded presence contains
the reference to within 2.00 everywhere. Where does the difference come from,
and is it recoverable?

## Method

The thresholding input was dumped beside the other artifacts
(`dump_crop_presence`) and the fibre's cross-section measured in the plane
normal to the refined reference at 401 samples along it, restricted to the
connected piece of the mask containing the reference point so a fibre passing
nearby does not count.

## What the section looks like

| quantity | median | p90 | max |
|---|---:|---:|---:|
| width along the sheet, one sigma | 3.6 | 8.1 | 9.7 |
| width across the sheet, one sigma | 1.3 | 1.6 | 3.2 |
| reference offset from the centre, along the sheet | 2.0 | 8.8 | 17.6 |
| reference offset from the centre, across the sheet | 0.3 | 0.8 | 4.0 |

The fibre presence is a ribbon, not a tube, and it is saturated across the
ribbon. The reference lies on the ribbon's centre line across the sheet and
can lie anywhere across its width along the sheet, so presence places the
fibre in one direction only.

## Where the deviation lives

Over the 48 reference samples whose nearest graph point is more than 8 away,
the median along-sheet width is 8.0 and the median along-sheet offset is 9.5,
while the across-sheet offset stays at 0.3. Those are the stretches where a
neighbouring fibre has merged laterally into the same ribbon and the thinning
voxels run between the two fibres. The deviation of the committed trace
follows the width sample for sample.

## What was done

The lateral position is left to the tracing. Each port carries the centre and
the two extremes of its section as sub-ports, each centred by a small presence
step; every line variant and region connection between sub-ports is
initialised by a cheapest path through the thresholded prediction and refined
with Ceres, and all are kept; and each sub-port records its cheapest variant
into the line and its cheapest connection into the region, so the tracer walks
sub-port to sub-port with nothing chosen ahead of it. See the method document
for detail.

## Result

Case31, scoped, interior of the route, against the refined reference:

| configuration | failures | coverage | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| start of task | 0 | 100% | 2.52 | 13.75 | 21.17 |
| per-sub-port routing, seam centre excluded | 0 | 100% | 2.02 | 5.77 | 8.69 |

Case31 traces as one segment from start to end without a restart. The
unrestricted control reproduces the numbers. The two-case run goes from three
failures to two, both in its second case, which leaves the fibre entirely and
is a separate matter.

With the seam centre kept as a routing target (`route_through_centre` true)
the trace runs along the seam at 71–74% of the reference and the maximum is
16.66; with it excluded the maximum is 8.69, at the reference start.

## Rejected on measurement

| variant | outcome |
|---|---|
| recentring the graph with a local centroid fit | every vertex on its drift bound; pulls to the merged axis |
| bridging wide stretches with straight chords | chords leave the mask and put ports in sub-threshold gaps |
| lower section threshold, lobe detection, lobe-following inits | workarounds for the bridging defect; removed with it |
| one committed matching per region, sub-ports assigned per line afterwards | the two sides disagree on which sub-ports carry a fibre; dead ends |
| reconciling the two with end costs, stop-first ranking, alternating passes | bad fixed points, or pairing the reference with the wrong fibre to avoid ends |
| plain shortest path as the variant init | luck-dependent; the metric-cost search replaced it |

## Commands

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_skeleton_20260918_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_skeleton_unrestricted_20260918_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_explicit_choices_skeleton_20260918_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_skeleton_vis_20260918_v1.json
```

Strips for the current configuration:
`/ephemeral/hendrik/fiber3d_train2/evaluations/contrast_case31_skeleton_vis_20260919_r3/scrolls/00/cases/0/rendered/`
