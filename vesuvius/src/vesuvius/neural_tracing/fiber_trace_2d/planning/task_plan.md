# 3D Default VC3D Cache Budget Plan

## Goal

Prevent each 3D DataLoader worker from inheriting VC3D's 8 GiB per-volume
decoded chunk cache when `volume_cache_memory_mib` is unset/null. The Python
3D config layer should default to 512 MiB per loader/worker.

## Implementation

1. Add a `DEFAULT_VOLUME_CACHE_MEMORY_MIB = 512.0` constant in
   `fiber_trace_3d.loader`.
2. Add a small parser helper for `volume_cache_memory_mib`.
   - Missing/null returns the 512 MiB default.
   - Explicit positive values are preserved.
   - Non-positive or non-finite values still raise.
3. Use the helper in both `load_config` and `config_from_mapping`.
4. In `fiber_trace_3d.train._make_trace2cp_geometry_loader`, pass the same
   512 MiB default to the generated 2D geometry loader when the 3D raw config
   leaves `volume_cache_memory_mib` missing/null.
5. Do not edit batch-size or worker-count config values.

## Tests

- Add tests that missing/null 3D cache config resolves to 512 MiB.
- Add a test that explicit positive cache config is preserved.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Spec Update

- Document that 3D `volume_cache_memory_mib: null` means Python-side default
  512 MiB per loader/worker, not VC3D's internal default.

## Docs Updates

- Update `docs/code_structure.md` around 3D loading/cache behavior.

## Changelog

- Add a 2026-07-15 entry for the 512 MiB 3D VC3D cache default.

## Non-Goals

- Do not change 2D standalone loader defaults.
- Do not change batch size or DataLoader worker count.
