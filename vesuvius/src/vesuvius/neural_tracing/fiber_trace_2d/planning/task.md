# Native 3D Trace2CP Render Brightness And Blocking Guard

Fix the basic native 3D Trace2CP rendering/loading behavior before working on
the strip geometry itself.

Requirements:

- Native 3D Trace2CP exported strip images must display raw volume brightness,
  clipped to `0..255`, not per-panel percentile or min/max normalized
  brightness.
- Report what scaling the 3D training input uses. For the current fast config
  this is `image_normalization: "zscore"` before model inference.
- Trace2CP strip rendering must use blocking coordinate sampling and must not
  silently render mixed fine/coarse fallback data when fine chunks fail.
- If the VC3D sampler reports chunk errors during Trace2CP side/top strip
  rendering, fail loudly instead of producing a misleading image.
- The strip geometry/orientation problems are explicitly out of scope for this
  task and will be handled after the basic loading/rendering path is correct.
