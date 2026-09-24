# Unified normal correction pipeline

Earlier sign-only fix was insufficient: independent baseline-relative angle interpolation and post-construction rotations could retain half turns. Extracted existing central-chord tangent and minimal frame transport for shared use. Target CP axes interpolate in a common transported reference, and uncorrected spans retain sampled normals. Replaced post-frame corrections with sampled-normal substitution before all standard strip construction/alignment. Tracing and stored vectors unchanged.

Regression coverage includes short curved supports comparing corrected input against ordinary injected sampled normals, and equal/opposite CP axes over a 170-degree baseline. No independent reviewer available; actual user fiber/live GUI reproduction is not yet verified.

Validation: six C++ suites passed using
`ctest --test-dir volume-cartographer/build -R '^(test_fiber_trace3d|test_line_annotation_generated_views|test_fiber_slice_geometry|test_lasagna_line_optimizer|test_lasagna_line_view_surfaces|test_quadsurface_basics)$' --output-on-failure`.
Python: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest volume-cartographer/scripts/tests/test_fiber_merge.py`, 127 passed.
Build: `cmake --build volume-cartographer/build --target VC3D test_line_annotation_generated_views test_lasagna_line_view_surfaces -j32`.
Final rebuild after header updates: `cmake --build volume-cartographer/build --target VC3D -j32`.
No commit requested for this follow-up.
