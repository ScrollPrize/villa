# 3D Single-Output Presence Visualization Simplification

Simplify `fiber_trace_3d` TensorBoard sample-sheet presence visualization for
single-output models. In single-output mode, show only:

- raw predicted presence,
- predicted presence modulated by the absolute cosine between estimated
  orientation and the displayed slice normal,
- predicted presence modulated by the absolute cosine between estimated
  orientation and the GT CP tangent.

The multi-output / conditioned visualization should keep the existing branch
presence summary columns.
