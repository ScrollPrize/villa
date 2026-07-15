# 3D Streaming Prefetcher Task Log

## Notes

- Inspected the 2D and 3D prefetch implementations.
- Confirmed 3D currently generates all chunk requests serially before creating
  download futures.
- Confirmed the current `workers` value in the 3D startup message only controls
  the later download pool, not dependency generation.
- Replaced the task and plan with a focused requirement to mirror the 2D
  streaming producer/download prefetcher.
- Implemented 3D streaming prefetch with bounded dependency producers,
  bounded download futures, ordered raw-sample producer consumption, global
  chunk de-duplication, cache-hit / `.empty` classification before download,
  earliest-raw-sample download priority, safe-prefix `idx` tracking, live
  progress, sample skip accounting, fatal cancellation, and temporary PyTorch
  CPU thread pinning.
- Added `prefetch_sampler_workers` to the 3D config path. The code fallback
  remains `2`, matching the 2D code fallback; checked-in runnable 3D configs
  now set `4`, matching the checked-in 2D configs.
- Added focused tests that use fake 3D prefetch request producers/downloaders
  to verify ordered streaming, de-duplication, cache-hit classification,
  `.empty` classification, and sample-skip accounting without remote I/O.

## Deviations Or Deferrals

- Remote/cache smoke with the real S1A volume was not run in this task. The
  focused tests cover the scheduler behavior without network access.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed with `21 passed`.
- Config parse smoke:
  `PYTHONPATH=vesuvius/src:. python - <<'PY' ... load_config(...) ... PY`
  confirmed `loader_example.json`, `train_s1a_nml_all.json`, and
  `train_s1a_nml_all_64_sd2.json` all parse with
  `prefetch_workers=16` and `prefetch_sampler_workers=4`.
