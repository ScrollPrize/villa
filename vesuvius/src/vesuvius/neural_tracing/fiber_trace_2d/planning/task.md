# Trace2CP Side-DP Z Smoothness

- Disable the regular per-step side-DP z transition penalty.
- Use dz smoothness instead, so constant z motion is not penalized but abrupt
  changes in z velocity are discouraged.
- Keep the change scoped to the side/joint Trace2CP DP backend. Do not change
  regular stepwise Trace2CP tracing or the side/top-z experiment.
