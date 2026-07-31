# Trace Native Extrapolation By Length

VC3D native fiber extrapolation must trace only until the configured
extrapolation distance is reached.

- Do not use target planes for extrapolation.
- Do not multiply the extrapolation distance by `max_step_factor`.
- Treat the requested distance as the hard trace-length/step budget.
- Clip the final step so the returned polyline has exactly the requested traced
  arc length.
- If prediction directions become invalid first, retain the existing partial
  native edge-truncated tail.
- Keep target-plane behavior unchanged for CP-to-CP tracing.
