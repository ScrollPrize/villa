# Task: Napari winding-fiber viewer

Add a focused Napari visualization tool for the per-winding crop-trace OBJ
artifacts emitted by `vc_fiber_trace_chunk direction-ablation`.

- Discover every `<base>_w_<N>_{h,v,err,tie}.obj` artifact from one output
  base.
- Load each nonempty winding/state artifact as a separate line layer and give
  every layer a distinct bright color.
- Present joint visibility controls for H, V, Broken, All, and None. The
  existing `err` and exact `tie` outputs both belong to the Broken viewer
  category so ambiguous fibers are not silently omitted.
- Provide previous/next winding navigation that shows the H and V layers for
  one winding at a time.
- Keep OBJ parsing, discovery, color assignment, and visibility selection
  independently testable without starting a GUI.
