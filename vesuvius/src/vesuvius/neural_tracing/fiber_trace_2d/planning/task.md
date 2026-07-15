# 3D Trace2CP Metric Wiring Fix

Finish the explicitly requested 3D Trace2CP metric wiring that was only
partially implemented in the previous 3D follow-up.

Required behavior:

- 3D training test evaluation must be able to run the public Trace2CP metric by
  projecting dense 3D model outputs onto the existing 2D Trace2CP side-strip
  geometry.
- Best checkpoint selection must use `test/trace2cp_error` when this metric is
  enabled.
- TensorBoard and stdout must log the 3D Trace2CP metric clearly.
- Add the minimal required 3D config keys rather than silently inferring
  missing 2D Trace2CP geometry settings.
- Add a 3D CLI inspection path that can run the same projection metric for a
  checkpoint and export a compact visualization.

The 3D training path must still load ordinary CP-centered 3D blocks for
training. Trace2CP evaluation may reuse the existing 2D loader only for metric
geometry and visualization.
