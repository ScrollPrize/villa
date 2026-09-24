# Strip regression correction

Removed the analytic LineRibbonSurface and restored ordinary QuadSurface rendering, projection, depth normals and geometry consumers. Restored support resampling and roll smoothing. Manual corrections are applied as angular offsets to the baseline frames, not replacement raw normals. Cross-view centers are linear and display tangents use the previous +/-4-point window, shared with normal editing. CP displacement and normal provenance remain.

No independent reviewer tool is available. Live GUI artifact and frame-rate verification is not available in this session; no performance claim is made from unit tests.

Validation:

- `cmake --build volume-cartographer/build --target VC3D test_line_annotation_generated_views test_lasagna_line_optimizer test_lasagna_line_view_surfaces test_quadsurface_basics -j32`: passed.
- `ctest --test-dir volume-cartographer/build -R '^(test_fiber_trace3d|test_line_annotation_generated_views|test_fiber_slice_geometry|test_lasagna_line_optimizer|test_lasagna_line_view_surfaces|test_quadsurface_basics)$' --output-on-failure`: six suites passed.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest volume-cartographer/scripts/tests/test_fiber_merge.py`: 127 passed.
- `git diff --check`: passed.

Regression coverage verifies exact ordinary QuadSurface runtime type, shifted-origin rendering/projection, off-center depth sampling equivalence, zero-correction equivalence despite noisy normal inputs, and manual rotation of both ribbons. No commit made.
