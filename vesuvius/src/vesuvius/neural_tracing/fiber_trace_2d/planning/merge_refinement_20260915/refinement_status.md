- [x] Commit previous implementation.
- [x] Update and independently review plan.
- [x] Implement fixed-crop local test state and EMA metric parity.
- [x] Independent code review and selected-tangent correction.
- [x] Final CPU/manager CUDA regression checks (15 refinement,57 managed,2 CUDA).
- [x] Resume managed training at update13300; first test completed and both EMA
  presence/direction tags verified directly in TensorBoard events.
- [x] Correct user-specified restart origin: stop that continuation and launch
  original non-refinement update11800 with fresh geometry (root3545749).
- [x] Remove missed test taper and endpoint locks; preserve uniform ellipse cap,
  omit outside joins, validate17CPU+57managed+2CUDA, restart original11800 root3826686.
