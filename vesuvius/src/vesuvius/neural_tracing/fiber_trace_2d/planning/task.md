# Task

Extend winding diagnostics and visualization for tagged reference fibers.

- Keep the aggregate `<base>_reference.obj` layer.
- Export one additional OBJ layer for every filename-ordered reference fiber
  as `<base>_reference_hs_<source_index>.obj`, whose virtual winding is
  `source_index / 2`.
- Load those layers in `view_fiber_windings` and rotate their complete
  visibility mask with the existing Next, Previous, and animation actions.
  References at half-steps `n` and `n+0.5` belong to solver winding slot `n`.
- Provide mutually exclusive Aggregate, Selected, and Hidden reference display
  modes so aggregate and indexed geometry cannot be double-rendered.
- Correct the BP `raw_w` column so it is not independently inferred. It must
  be the exact final `est_w` candidate inverse-mapped through the selected
  global sign and gauge offset, converted from the latent half-step coordinate
  to integer `mapWinding` with the reference H/V class, component phase sign,
  and selected phase, then shifted into the indices used by the generated
  `<base>_w_<index>_*.obj` layers. It therefore tells the user which inferred
  winding layer to inspect without permitting a separate half-step choice.
- Report `NA` when the final estimate cannot be inverse-mapped to one
  unambiguous solver gauge and compatible orientation component. The Ceres
  table remains unchanged.
- Do not change Ceres result-layer generation or thresholding.
