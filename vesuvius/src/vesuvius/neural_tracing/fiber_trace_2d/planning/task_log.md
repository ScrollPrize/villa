# Task Log: Bright material colors for BP state lines

- The normalized vertex RGB extension imported as vertex attributes in MeshLab
  and did not provide the requested bright line primitive colors. It will be
  removed completely from this exporter/API.
- Selected standard OBJ `mtllib`/`usemtl` plus sibling MTL output so the
  material applies directly to line and point primitives.
- Independent review required reuse of the existing texture MTL serializer,
  atomic two-file publication with the OBJ last, explicit opacity, empty-layer
  coverage, and a real MeshLab visual check. The plan was updated before code.
- Extracted shared OBJ token, material-reference, and MTL serialization helpers.
  Existing textured output now calls the same serializer without changing its
  record ordering or material values.
- Added an atomic material-backed polyline bundle writer. It writes the local
  same-stem MTL first and the referencing OBJ last; the existing plain writer
  remains unmaterialed.
- Direct BP layers now use saturated cyan, magenta, orange, and lime materials
  on line/point primitives, with plain XYZ vertices.
- The initial build found that `cv::Vec3d` is neither range-iterable nor a
  literal type under this toolchain. Indexed validation and a runtime-constant
  palette preserve the intended behavior portably.
- Built `vc_fiber_trace_chunk` and the focused test with `-j32`; all 65
  `test_fiberlet_crop_trace` cases passed. This includes exact preservation of
  the pre-existing textured material serialization, plain-writer compatibility,
  empty state bundles, token rejection, local MTL references, publication
  paths, record ordering, and the four material definitions.
- Broader opportunistic suites are not green on this current build for
  unrelated existing assertions: `test_fiberlet_paths` reports 295 bit-exact
  local-metric comparison failures at line 414, and `test_fiber_replay` reports
  four strip-dimension failures. Neither failure path uses the OBJ-material
  changes; the focused compatibility test covers the shared serializer change.
- Regenerated the existing centered-1024 state artifacts. Direct inspection
  confirmed plain XYZ vertices, local `mtllib`, `usemtl` before geometry, and
  matching MTL `Ka`, `Kd`, `Ks`, `d`, and `illum` records for all four states.
- Headless validation cannot confirm MeshLab's actual wireframe presentation;
  user visual confirmation remains required and is not claimed complete.
- `git diff --check` passed.
