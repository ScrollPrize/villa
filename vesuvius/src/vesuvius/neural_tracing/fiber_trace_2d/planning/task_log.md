# 3D Default VC3D Cache Budget Task Log

## Notes

- User corrected the target default to 512 MiB, not 1 GiB.
- The change should be Python-side so existing JSON `null` no longer falls
  through to VC3D's internal 8 GiB default.
- `load_config(...)`, `config_from_mapping(...)`, and the generated 2D
  Trace2CP geometry loader path now resolve missing/`null`
  `volume_cache_memory_mib` to `512.0`.
- Explicit positive `volume_cache_memory_mib` values are preserved.

## Deviations Or Deferrals

- None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 27 tests.
- Config smoke check showed `loader_example.json`, `train_s1a_nml_all.json`,
  and `train_s1a_nml_all_64_sd2.json` resolve to `512.0` MiB /
  `536870912` bytes.
