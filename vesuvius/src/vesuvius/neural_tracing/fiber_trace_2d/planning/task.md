# Training-integrated reviewed failure benchmark

Integrate the optimized native/live-inference reviewed-fiber benchmark into the
fiber training test cycle.

- Freeze a persistent test dataset containing the union of truncated reviewed
  pieces that failed in any canonical reviewed-model evaluation.
- Use that smaller locked dataset for the existing deterministic dense test
  losses and cross-section diagnostics.
- At each configured trace-test interval, evaluate the current raw training
  model with the existing highly parallel live-inference/native replay runner.
- Report and log forward, reverse, and combined restart counts plus the implied
  aggregate segment length: total directed reference length divided by total
  restarts plus one.
- Preserve exact membership, hashes, trace settings, and per-step summaries.
