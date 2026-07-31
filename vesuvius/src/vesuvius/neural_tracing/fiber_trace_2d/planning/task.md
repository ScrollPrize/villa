# Strict VC3D Fiber V3 Parsing And Consumer Completion

- Make every C++ and Python consumer that accepts `vc3d_fiber` version 3
  validate the complete v3 contract before using it.
- A v3 fiber must have a valid top-level `optimization_mode`, object control
  points, one complete `segment_to_next` descriptor on every non-final control
  point, and no descriptor on the final control point.
- Invalid or incomplete v3 data must fail without synthesizing descriptors,
  defaulting missing v3 fields, rewriting the source, or offering a schema
  repair path.
- Preserve legacy version-1 support. Missing `optimization_mode` on v1 means
  Lasagna, and v1 numeric control points acquire v3 descriptors only when a
  user-initiated save writes a new canonical v3 file.
- Complete v3 control-point support in the remaining Lasagna atlas, Atlas
  constraints/inspection, and Spiral consumers.
- Make `vc_lasagna_line_probe --output` write geometry and v3 segment metadata
  coherently after Lasagna reoptimization instead of retaining stale
  descriptors from the input.
- Repair the NML loader and all direct `Vc3dFiber` constructors after the
  addition of `control_point_segments`, without making that field optional.
- Do not change coordinate scaling. Persisted trace thresholds are historical
  values expressed in base voxels and are already correct.
