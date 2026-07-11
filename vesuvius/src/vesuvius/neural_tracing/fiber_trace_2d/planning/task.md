# Trace2CP Center-Biased Closest Point Metric

User request:

- Add a second Trace2CP visualization column that does not use TTA and shows
  just the reference inference.
- Tweak the Trace2CP closest-point metric so the chosen closest point prefers
  the center between the two control points.
- Penalize candidate distances by horizontal distance from the CP-pair center.
- At either CP x location, the considered distance should be `2x` the actual
  vertical trace-to-trace distance.

Interpretation:

- The actual trace-to-trace distance remains the absolute y separation between
  the two traces at a candidate x.
- The considered metric distance is:
  `actual_distance * (1 + abs(x - center_x) / half_cp_x_span)`, clipped so the
  multiplier is `1x` at the center and `2x` at either CP.
- Trace2CP chooses the x with the smallest considered distance. The public
  `trace2cp_score` uses the considered distance. The actual distance remains
  available for diagnostics.
- When `--trace2cp-vis --med-tta` is used, the JPG should show the selected
  median-TTA result and a reference-only result side by side. Without
  `--med-tta`, the reference-only result is already the only result shown.
