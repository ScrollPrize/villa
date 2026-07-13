# Trace2CP DP Local-Angle Semantics

- Remove the side Trace2CP DP's candidate-angle-derived vertical move cap.
- Candidate angle limits must be relative to the local sampled/predicted
  direction field, not relative to global horizontal slope.
- Keep only a broad compute search band for DP vertical moves, independent of
  the candidate angle setting, so steep local fiber directions above 45 degrees
  can be represented.
- Set the side DP horizontal transition step to 4 px.
- Reduce the default second-order DP smoothness penalties so the path is less
  over-smoothed.
