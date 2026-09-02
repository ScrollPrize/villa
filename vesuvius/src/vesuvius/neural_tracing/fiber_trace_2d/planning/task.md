# Task

Make reference winding conflict diagnostics class-complete and prevent an
unmappable reference output-layer estimate from aborting a regular run.

- Split the existing fixed-reference-to-BP conflict summary into the usual
  five magnitude classes plus separate perpendicular and parallel sign rows:
  perpendicular 0.5, perpendicular 1.5+, parallel 0, parallel 1, and
  parallel 2+.
- Add an equivalent conflict table for constraints extracted only between
  tagged reference fibers, with both endpoint windings fixed to their known
  filename-ordered winding labels.
- Keep sign and magnitude factors separate and report factor count, conflict
  count/fraction, hard violations, and weighted loss for each class.
- Hide the long per-reference, per-piece constraint listings by default and
  expose them through an explicit diagnostic flag. Keep the new aggregate
  reference-to-reference table enabled whenever reference fibers are loaded.
- If the final calibrated reference estimate is incompatible with its unique
  orientation component and therefore cannot identify an integer published
  winding layer, print `NA` for `raw_w` rather than terminating the run.
