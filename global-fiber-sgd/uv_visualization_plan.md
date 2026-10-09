# Full solution visualization and sweep costs

Committed optimizer state c01fb1ecd before starting this task.

## Plan
1. Read immutable saved CCW/CW solutions and histories; no optimizer updates.
2. Align history pre_update_energy at sweep s to state s-1. Compute missing
   final-state expected global energy with existing conditional_costs on managed
   GPU, adding frozen offset/width constants and reporting target entropy term
   separately. Never mislabel costs as belonging to the following sweep.
3. Render every retained piece as its actual model-space H or V line using
   hard_geometry and node_ds endpoints. Opacity=P(selected winding), colors H/V.
   No random subset. Omit q_broken>0.5 pieces explicitly; retain source/piece IDs
   and coordinate/confidence tooltips. SVG vector lines, plus lightweight PNG
   overview for inspection; histogram below of conditional max winding probability
   and joint max probability among retained pieces. Independent U/V axis scaling
   stated, coordinates in native base-voxel units.
4. Produce sweep CSV/JSON/Markdown and progression SVG with violations and global
   cost; both convention layouts. CPU rendering, no inference or optimization.
5. Tests: signed endpoint directions, conditional confidence, opacity, histogram
   counts, SVG XML/line counts, history cost alignment; independently review.

## Spec update
Latest override: equal U/V voxel scale (no independent scaling), remove all
tooltips. Raster output becomes LZW TIFF, total height1000px (revised from4000), proportional plot
width with annotation margins/histograms retained. Tile-wise rendering avoids
allocating the multi-gigabyte full raster. Keep every fiber and boundary. CPU
rerender saved10-sweep plots only; continuing optimization runs are untouched.
Test identical scale, TIFF height/LZW/pixels, SVG completeness and no titles.

Follow-up: overlay winding boundaries at U=[0,cumsum(fitted widths)] in both
SVG and PNG, clipped to visible U bounds. Dashed neutral lines with boundary
indices/tooltips; sparse overview labels to avoid crowding. CPU-only rerender;
validate cumulative positions and unchanged fiber/histogram counts.

Readout only. No geometry/probability changes or posthoc filtering beyond existing
broken rule. Conditional confidence is not calibrated correctness. Expected-cost
curves are distinct from the old mixture objective; constants identified.

## Docs updates
Task/log/status, model doc, report/config, exact approved runner command and
results paths. Full vector files may be large; provide raster overview, not a
replacement for zoomable SVG. No dependency install.
