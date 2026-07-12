# Regular Combined Trace2CP Candidate-Point Direction Scoring

Make regular non-z `--trace2cp-combined` tracing evaluate the direction term
the same way as z-search: average agreement with the current/last point's
oriented direction and agreement with the predicted direction sampled at the
candidate point.

Requested behavior:

- For each non-z combined candidate, sample the candidate point direction.
- Mark candidates invalid if candidate-point direction sampling fails.
- Keep embedding terms and weights unchanged.
- Preserve z-search behavior.
