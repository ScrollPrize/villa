# Task log: vc_reader migration

## Implementation and scope

- Completed audit items 1, 2, 3, 7 and 8. Deleted prepare_scroll.py and the orphaned data/volume.py and tifxyz/writer.py modules, plus stale references.
- Winding phase samples use resident.sample_array; winding surface points use tifxyz.Surface.points_at. Surface-coordinate sampling uses resident.GridField. Torch callers use vc_reader.torch.sample_array with explicit index order and application masks.
- Raw CT rendering uses ZarrReader.render_points, a thin adapter to the existing native grid renderer. Added prepared per-pixel geometry input to that renderer in C++ with CPU and CUDA implementations. The Python ZarrReader adapter uses its CPU array; the C++ request exposes GPU selection.
- Removed the duplicate Python surface-transfer rasterizer; all rasterization uses vc_reader.SurfaceTransfer.
- Items 4–6 (Zarr metadata/chunk interpretation, tifxyz metadata parsing, projection) remain unchanged at the user's explicit request.

## Review and behavior

- Independent plan review identified boundary, nearest-tie, and precision differences. The user explicitly directed use of vc_reader defaults, so no compatibility handling was added.
- Adopted float32 resident interpolation, half-up nearest ties, default bicubic surface interpolation and shared quad validity. Rendering uses array fill outside bounds and the default composite cutoff of zero; no accepted layer produces NaN, displayed as black.
- The initial spec.md lookup found no file; final review located specs.md (plural), read the relevant fused-map requirements, and obtained independent review against it. Updated its sampling ownership wording. No conflict with fused-map construction or batched line/CP lookup was found. local_development.md is absent.
- No dependency installation or environment bootstrap was performed. Initial Python checks selected stale installed extensions; the documented load_build helper selects the freshly built modules without installation. The wider vesuvius tests require its directory on PYTHONPATH for tests.zarr_utils.

## Validation

Run from the repository root:

```bash
AGENTS_AGENT_MODE=1 CCACHE_DISABLE=1 cmake --build vc_reader/build-native-cuda --target _reader _transforms _catalog test_native_grid test_native_grid_cuda -j 2

AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_READER_BUILD_DIR="$PWD/vc_reader/build-native-cuda" vesuvius/.venv/bin/python -m pytest -q -p no:cacheprovider vc_reader/tests/test_render_points.py vesuvius/tests/neural_tracing/test_shared_sampling_migration.py vesuvius/tests/tifxyz_label_transfer

AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_READER_BUILD_DIR="$PWD/vc_reader/build-native-cuda" PYTHONPATH="$PWD/vc_reader/tests:$PWD/vesuvius" vesuvius/.venv/bin/python -c 'from build_support import load_build; load_build(); import pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py", "vesuvius/tests/neural_tracing/test_fiber_trace.py", "vesuvius/tests/neural_tracing/test_rowcol_triplet_merge.py"]))'

AGENTS_AGENT_MODE=1 ctest --test-dir vc_reader/build-native-cuda -R '^test_native_grid(_cuda)?$' --output-on-failure
git diff --check
```

- Native extension and test builds passed.
- Rendering, transfer and migration checks: 149 passed, including 14 subtests.
- Existing fiber tracing and row/column merge suites: 346 passed.
- Native grid CPU test passed. The sandbox skipped the CUDA test; rerunning with GPU access passed, including CPU/CUDA prepared-geometry comparison.
- Synthetic Zarr and tifxyz fixtures cover fractional sampling, nearest ties, singleton axes, anisotropic scales, isolated valid pixels, negative and positive fill, composite cutoff, native tile boundaries, gradients, coordinate ordering, and default surface validity/interpolation.
- Numeric checks include the quadratic surface's default bicubic value 11.5625 at column 1.25 (the previous bilinear value was 11.75), exact linear field/phase expectations, and matching CPU/CUDA render values. Different boundary/tie behavior is intentional, not a numerical-equivalence claim.
- No performance claim or benchmark was made. No real-scroll end-to-end workload or macOS/arm64 run was performed.
