# Reference winding comparison (100 sweeps)

Run through the existing approved CPU benchmark entrypoint:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_reference_comparison100.json
```

Output: `data/reference_comparison100_v1/`. No inference or solve is run.

## Winding colors and projection

Adjacent half-open winding rectangles overlap in columns. The continuous
midpoint `(max(begin)+min(end))/2` is assigned `k+0.5`; linear interpolation
between these landmarks defines continuous reference winding q. Extrapolate
linearly at ends; discrete decisions are bounded to actual IDs10..135.
This intentionally approximate labeling is not the missing winding-ids.tif.

Every integer winding has a deterministic random RGB color (seed20261011),
with linear RGB interpolation at fractional positions. Fitted q uses actualU
and inverse cumulative winding widths, including offsets and signed fiber arc;
it is NOT merely the largest-probability bucket index. One global integer offset
per solution aligns arbitrary numbering (weighted median reference-q minus
fitted-q). No scale, sign reversal, per-fiber adjustment or optimization.

Reference XYZ is multiplied by4 into AFV native2.4um voxels. A spatial index
on stride2 valid surface cells supplies four nearby cell candidates. Two
triangles per cell provide approximate closest surface coordinates, including
edge projection. All original vertices inside a cell must be valid; suspicious
long edges are rejected. This is an approximation to bilinear geometry, not a
guaranteed global closest-point solution. It is suitable for requested rough
comparison, not precise anatomical ground truth.

Sample original AFV polylines at their vertices plus <=200vx arc intervals.
Samples within200nativevx of the reference supply evidence. Their arc-length
weights determine ONE median discrete reference winding per original source
fiber, shared by any optimized pieces of it. Then reproject every point onto
that winding only. This restriction is independent of our fitted winding.
Unsupported samples remain gaps; the rendered reference curves also reject
large UV jumps and samples along connectors crossing invalid reference cells.
This is a sampled hole guard, not exact cell traversal at diagonal corners. Cells that
straddle approximate winding boundaries can lose a little support because cell
selection uses its center label. The nearest and restricted distances are saved.

## Views

- `reference_own.tif`: actual valid reference grid, own interpolated winding colors.
- `ccw/fit_own.{tif,svg}` (and `cw/`): our UV fibers, own fitted winding colors.
- `ccw/fit_reference.{tif,svg}`: our UV fibers, colors sampled from the chosen
  reference winding; gray where unsupported.
- `ccw/reference_fit.{tif,svg}`: the actual projected curves on the reference
  UV layout, each on its source-fiber median winding, colored by our fitted q.
  This does NOT paint surface pixels by nearest fiber, and is not a filled mesh.

All images use equal physical horizontal/vertical scale and2000px height;
compressed LZW TIFF. SVG is vector and contains no tooltips. Fiber opacity is
joint probability of the selected fitted winding. The reference-own image has
no fiber confidence opacity. The UV-fit views show retained pieces under the
saved solve's rejection rule; projection/median evidence is shared independently
of CCW/CW selection. A projected source can have gaps where the reference lacks
height coverage or where that single winding cannot represent its whole curve.

`projection.npz` records original sample geometry/arcs, nearest projection,
chosen winding, restricted projection and distances. `fibers.csv` gives source
and piece IDs, arcs, chosen reference winding, fitted anchor winding and joint
probability. `summary.json` records configuration, global offsets, length
coverage/error summaries and hashes. Projection cache is reused only with
matching reference XYZ/metadata and AFV hashes, geometry parameters and piece
identity. Solves and reference files are never modified.

## Validation

Synthetic tests cover rectangle landmarks, color interpolation, plane/edge
projection, degenerate triangles, weighted source median across split pieces,
missing matches, actual-U inverse, endpoint labels and hole-crossing rejection.
See `reference_comparison_results.md` for measured results when complete.
