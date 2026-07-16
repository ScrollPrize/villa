# Trace2CP Compact Geometry CP-Span Coverage

Fix Trace2CP segment-source compact geometry so CP-to-CP line spans cannot
contain unsampled line points.

Requirements:

- Compact geometry preload must sample all line points needed by CP source
  windows and by consecutive CP-to-CP Trace2CP spans.
- A Trace2CP segment must not fail because an interior line point was simply
  not sampled by compact geometry preload.
- If a sampled line point is invalid, keep failing loudly and report the real
  Lasagna data reason.
- Defensive diagnostics should identify unexpected unsampled invalid points and
  include a direct Lasagna probe of the point values.
- Do not synthesize, propagate, or invent missing Lasagna normals.
