# Task Log: Truly Rolling Shared 3D Tiled Inference

## Implemented

- Replaced the full-output-Z `_RollingZBand` with `_CircularZBand`, whose mmap
  depth is planned from canonical Z tile writes and chunk-aligned flush
  frontiers and is independent of full volume depth.
- Added logical-to-physical generation checks, wrap-aware chunk reads,
  chunk-region clearing, explicit mmap close/unlink cleanup, and backing-size
  reporting.
- Added `run_tiled_inference_3d` as the sole neural runner. Both Lasagna
  predict3d and Fiber 3D inference now use it; the legacy `_infer_tiled_3d` and
  `_infer_tiled_products_3d` loops were removed.
- Added explicit `OutputProductSpec.inference_scaledown` semantics so the
  runner can feed Lasagna fine/coarse products in one model traversal while
  retaining base-relative manifest scales.
- Changed normalization/finalization from full XY bands to globally anchored
  output chunks. Each scale owns one geometric weight ring shared by its raw
  products; already complete product chunks receive no raw accumulation.
- Kept external `pred_dt` outside neural scheduling and moved it to an
  independently resumable post-inference stage.
- Changed progress rendering to carriage-return updates only on a TTY and
  durable per-Z-row lines otherwise; ring depth/backing and flush boundaries
  are reported explicitly.

## Compatibility decisions

- Tile traversal remains canonical Z/Y/X order and each scheduled tile is run
  once.
- The numerical order remains prediction times full-resolution blend weight,
  followed by `_pyrdown3d`, followed by float32 accumulation.
- Coherent products are written as complete sibling bundles through the
  existing atomic output adapter.
- GPU pause, optional InstanceNorm calibration, Fiber normalization/autocast,
  pyramid creation, manifests, and output encodings remain adapter or caller
  responsibilities and were preserved.

## Validation

- Python bytecode compilation passed for the shared engine, Lasagna wrapper,
  Fiber adapter/CLI, and regression tests.
- Circular wrap/overwrite/backing-size, one-pass multi-scale, and adapter
  protocol/schema tests passed (`5 tests`).
- The existing Zarr-backed crop/resume tests could not complete in this Python
  3.14 environment: a standalone `zarr.open(..., mode="w")` blocks in Zarr's
  synchronous wrapper before entering the changed inference code. A timed
  faulthandler trace confirmed the wait is in `zarr.core.sync.sync` at test
  setup, not in the circular runner.
- A standalone 10-second `zarr.open(..., mode="w")` smoke test reproduced the
  same pre-inference hang and exited through `timeout` with status 124.
- `pytest` is not installed in `/home/hendrik/.venv_las`, so the Fiber pytest
  module was not run; no dependency installation was authorized for this task.

## Deviations / remaining validation

- A representative full Fiber volume run was not started because it is a
  long-running GPU/data workload and the exact output path is user-owned.
  Ring/backing/RSS/throughput measurements therefore remain operational
  validation for the next real inference run.
- Cross-platform macOS execution was not available; the implementation uses
  portable mmap close/unlink behavior and no longer depends on Linux
  `madvise` or hole punching.
