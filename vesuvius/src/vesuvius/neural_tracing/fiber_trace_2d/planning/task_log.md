# Task log

Task: construct route and connection geometry from the prediction rather than
from the skeleton's own voxels.

## 2026-09-19 — starting measurements, carried over from the previous task

Route-based metrics on case31, all in base voxels against the refined
reference. Only route-based numbers are used from here on; nearest-point
figures let geometry from unrelated fibers count and are misleading.

| stage | GT to route, max | route to GT, max |
|---|---:|---:|
| thresholded voxels (nearest point, a bound) | 2.00 | — |
| best connected route in the skeleton graph | 17.62 | 46.18 |
| routable network after the choice | no connected route | — |
| oracle over all candidate connections | 21.57 | 46.18 (2.85 median, 17.94 p95) |
| committed trace | 22.27 | 26.07 |

Findings that set the direction:

- Port optimization is not the lever. Disabling it leaves the committed trace
  at 25.71, identical; changing its Gaussian sigma from 6 to 2 also leaves it
  at 25.71.
- Ports are already a median 7.52 from the reference at their skeleton
  position, while the skeleton overall is a median 2.37, so the error is
  concentrated where ports live.
- Of 27 refined lines near the reference deviating more than 8, all 27 have
  their worst point in the outer 20% of the line and none in the middle.
- The oracle route is built from three long lines, the same three graph edges
  that carry 76% of the reference, joined by region connections. Its two worst
  elements are the entry and exit; every interior element is between 4.2 and
  21.7.

## 2026-09-19 — what the prediction actually constrains

Dumped the thresholding input for case31 (`dump_crop_presence`) and measured
the fibre's cross-section in the plane normal to the refined reference, at 401
samples along it. All figures in base voxels, one sigma of the blob containing
the reference point.

| quantity | median | p90 | max |
|---|---:|---:|---:|
| section width along the sheet | 3.6 | 8.1 | 9.7 |
| section width across the sheet | 1.3 | 1.6 | 3.2 |
| reference offset from the section centre, along the sheet | 2.0 | 8.8 | 17.6 |
| reference offset from the section centre, across the sheet | 0.3 | 0.8 | 4.0 |

The thresholded fibre presence is a ribbon, not a tube. The reference sits on
the ribbon's centre line across the sheet, and it can sit anywhere across the
ribbon's width along the sheet. Presence therefore fixes the fibre in one
direction only.

Where the skeleton is far from the reference, the ribbon is wide: over the 48
reference samples whose nearest graph point is more than 8 away, the median
along-sheet width is 8.0 and the median along-sheet offset is 9.5, while the
across-sheet offset stays at 0.3. Those stretches are where a neighbouring
fibre has merged laterally into the same ribbon, so the blob's axis, which is
all thinning can report, runs between the two fibres.

The deviation of the committed trace tracks that width sample for sample: at
55% of the reference the width is 3.7 and the trace is 3.0 away; at 57% the
width is 9.0 and the trace is 17.3 away.

## 2026-09-19 — what was tried

- **Recentring by a local field fit (the previous method).** Every vertex sat
  on its drift bound, median shift 1.97 against a bound of 2.00. The fit is a
  weighted centroid, so inside a ribbon it pulls to the ribbon's centre, which
  is the wrong direction. Replaced.
- **Carrying the along-sheet position through the merge.** Continuity from the
  previous vertex, clamped into the blob. Applied to the committed route
  offline it moved the maximum only from 21.17 to 20.56, because the route
  enters the merge already on the blob's axis.
- **Bridging the merged stretches.** Replacing a stretch whose section is too
  wide to place the fibre with a straight bridge between the nearest vertices
  that can place it, then moving every vertex across the sheet onto the
  prediction. Offline on the committed route this took the maximum from 21.17
  to 9.5. In the graph, where the bridge is anchored at junction nodes that are
  themselves on a merged axis, it takes the committed trace from 21.17 to
  16.81. Kept.
- **Bridging region connections as well**, before the solve, after it, and
  both. Every variant introduced a `skeleton_terminal` failure and pushed the
  maximum to between 22.9 and 24.8. Not kept.
- **Requiring a bridge to stay above threshold at every point.** Rejects the
  bridges that help: the graph bound went back from 16.24 to 17.62 and the
  failure returned. Not kept.
- **Placing the split point by section width** instead of at a fixed 32 base
  voxels, pushing it out to 64 or 128 until the section narrows. Before the
  bridging work it moved ports near the reference from a median 7.87 to 6.19
  but cost a failure; after it, it is a small regression (16.81 to 17.30) with
  no failure. Removed again, so the split stays at the fixed offset.

## 2026-09-19 — result

Case31, scoped, against the refined reference, interior of the route only:

| configuration | failures | coverage | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| before (field-fit recentring) | 0 | 100% | 2.52 | 13.75 | 21.17 |
| after (bridged recentring) | 0 | 100% | 2.39 | 12.89 | 16.81 |

The committed trace now matches the best route the candidate network contains
(16.81 for both), so the remaining error is in the graph geometry and not in
the routing. The graph's own bound is 16.24, at the junction nodes that sit on
a merged axis and anchor the bridges.

Control run (unrestricted crop): 0 failures, maximum 16.81. Two-case
`explicit_choices`: 3 failures and 94.89% coverage, unchanged from before;
its case 0 improves from 21.17 to 16.81 and its case 1 is unaffected, being a
separate routing failure with a maximum of 111.

## 2026-09-19 — correction: bridging was wrong, and why

The 16.81 result above came from bridging: replacing wide stretches of the
recentred graph with straight chords. The chords leave the thresholded mask
(the "support guard" experiment had shown that and I misread it), so a port
placed on such a chord can sit in a sub-threshold gap between two fibres, its
section then sees one fibre only, and no candidate reaches the reference.
Recentring and bridging are removed. Ports are placed at thinning voxels,
which are inside the mask by construction.

## 2026-09-19 — lateral candidates as a routing choice

The user's design, implemented in this order:

1. Ports stay at the fixed 32 base voxels into a line. At each, the section of
   the thresholded prediction in the plane normal to the edge, restricted to
   the piece of the mask the thinning voxel lies in, gives three candidates:
   the centre and the two extremes along the section's long axis, an extreme
   being kept only if it is more than `candidate_min_offset_base` (6) from the
   centre. Each candidate is centred by a few steps of a small presence
   centroid in that plane (`candidate_refine_radius_base` 4), which walks an
   extreme in from the mask's edge onto the fibre it belongs to.
2. Every line is refined once per pair of end candidates; every region
   connection once per pair of candidates at its two ports.
3. In the region choice a port with k candidates is k items, so two fibres
   that share a merged edge are both routed, one per candidate. The end cost
   is charged once per port none of whose candidates is used. A connection's
   cost includes what each attached line can do at best with that candidate.
4. After the choice, a candidate is active where a matching routes a fibre
   through it; a line variant is active when both its end candidates are and
   it is the cheapest variant for each of them among the active ones. The
   route network joins segments by port and candidate together.
5. The initial geometry of every line and connection variant is the shortest
   path through the thresholded prediction between its two candidates, in a
   corridor of `search_radius_base` (16) around the skeleton piece, with
   length weighted slightly by presence. The skeleton stretch with blended
   ends is the fallback when the mask does not connect the two candidates;
   on case31 it was never needed (255 line variants, 199 connections).

Two findings on the way, both measured on case31:

- With one candidate per port and the far end of a line free, the two regions
  a line joins each assumed the other would follow and chose opposite sides
  (port 9 took the wrong fibre at 24.34). Making candidates items of the
  matching removed the conflict without any iteration.
- The unexplained-support term is sampled along the region's own skeleton,
  the merged axis, so a middle route explains it best and it pushed the
  choice to the middle candidate (region 5, port 10). With it at zero the
  choice follows the connection and line costs. `max_ports` is raised to 14
  because a region now counts items; one region on case31 still exceeds it.

## 2026-09-19 — result

Case31, scoped, interior of the route, against the refined reference:

| configuration | failures | coverage | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| start of task | 0 | 100% | 2.52 | 13.75 | 21.17 |
| lateral candidates, items, path-search init | 0 | 100% | 1.77 | 4.77 | 6.19 |

Every port near the reference now has an active candidate within 4.5, and at
the previous worst spot (58%) the chosen candidate is 1.21 away. The
unrestricted control reproduces the numbers. The two-case run is unchanged
at three failures, all in its second case, which is a separate routing
failure (maximum 111).

Rejected on measurement in this second half: a lower section threshold to
reconnect the lobes (a workaround for the bridging defect), lobe detection at
two thresholds, and a lobe-following walk for variant inits. All three were
removed when the cause was found.

## 2026-09-19 — the committed choice was the defect

The user's reading of the four line-129 strips: the variants that exit on the
reference side are right whatever they enter on, and the difference between
the two full assignments is 0.3 cost units over 796 vx, which presence and
direction cannot decide. Every remaining restart was an `unmatched_port`
stop: a sub-port used by the region's matching but not served by the line's,
or the reverse. Two independent brute-force matchings over the same sub-ports
were being reconciled with penalties and alternating passes, and that layer
produced dead ends, bad fixed points and a stop-first rule that paired the
reference with the wrong fibre to avoid ends.

The user's design has no such layer. Nothing is chosen ahead of tracing:

- Every line variant and every region connection is refined and kept.
- Per sub-port, the cheapest variant into the line from that side and the
  cheapest connection into the region: with three entry and three exit
  sub-ports a segment carries six routes, one per sub-port.
- The tracer walks sub-port to sub-port along those entries, so no sub-port
  can be a dead end while any variant or connection exists.
- The seed follows the entry route of the sub-port it enters through rather
  than the nearest of several overlapping variants, which fixed the start.
- The seam centre of a wide section, kept as a candidate, is not a routing
  target (`route_through_centre` false in the committed configs): with it
  on, the trace ran along the seam at 71–74% and the maximum was 16.7.

Removed: matchings, end costs, the unexplained-support term, activation, the
consistency passes, the forced-variant diagnostic and the sliced per-variant
strips. Kept: `render_line_variants`, which writes every variant of a line as
a named polyline.

## 2026-09-19 — result

Case31, scoped, interior of the route, against the refined reference:

| configuration | failures | coverage | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| start of task | 0 | 100% | 2.52 | 13.75 | 21.17 |
| per-sub-port routing, centre excluded | 0 | 100% | 2.02 | 5.77 | 8.69 |

Case31 traces as one segment from start to end without a restart; the
unrestricted control reproduces it; the two-case run is at two failures, both
in its second case, which leaves the fibre entirely and is a separate matter.
