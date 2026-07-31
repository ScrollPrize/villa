# Per-Segment Interpolation Goals And Cubic-Spline Fallback

- Give every CP-to-CP segment a persisted `interp_goal` and `interp_mode`.
- `interp_goal` is one of `global`, `cspline`, `lasagna`, or `trace` and records
  the requested policy. `interp_mode` is one of `cspline`, `lasagna`, or
  `trace` and records the algorithm that produced the stored geometry.
- Resolve goals through `trace -> lasagna -> cspline` and
  `lasagna -> cspline`; `cspline` has no lower fallback.
- For a `global` segment, resolve the first attempted mode from the fiber-wide
  Lasagna/trace mode. If its endpoint distance is below 100 base voxels, use
  `cspline` directly. Explicit per-segment goals do not use this distance rule.
- Interpolate adjacent `cspline` segments jointly. Use neighboring traced or
  Lasagna geometry as hard boundary directions, derive shared internal CP
  tangents without normal or trace inference, and favor a smooth path close to
  the shortest CP polyline.
- Decide Lasagna fallback per span from its initialization/rollout result, then
  jointly refine connected successful Lasagna spans while protecting trace,
  `cspline`, and unrelated manually selected geometry.
- Persist a mode-dependent `metric` and a compact `msg` on each segment.
  Lasagna stores its final maximum normal-alignment error, trace stores its
  minimum meeting-plane error, and `cspline` stores no metric. Fallback messages
  preserve the reason for each rejected higher-priority attempt.
- Add a Ctrl-right-click segment menu for selecting the interpolation goal.
  Changing a goal, the global mode, or relevant neighboring geometry must
  reoptimize the affected segment groups and update each actual `interp_mode`.
- Show every visible span's metric and message below the strip. Keep a label in
  the viewport while any part of its span remains visible and resolve label
  collisions deterministically by pushing neighboring labels.
- Prefix each span label with `C`, `L`, or `T` for its actual `interp_mode`.
