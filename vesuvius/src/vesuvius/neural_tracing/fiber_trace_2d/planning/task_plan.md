# VC3D Requested-Level Blocking Coordinate Sampling Plan

## Problem

The current VC3D Python `Volume.sample_coords(..., blocking=True)` path waits
for dependency fetches, but then renders with `sampleCoordsFineToCoarse`.
Sampling itself uses `tryGetChunk`, so requested-level chunks can still be
uncovered at render time if they were evicted, requeued, or only present in the
persistent cache but not decoded. The sampler can then show coarser zarr levels
or invalid black pixels. That is viewer-style progressive behavior, not the
strict requested-level blocking behavior required by training/debug rendering.

## Implementation

- Add a strict requested-level coordinate sampling path in VC3D:
  - collect the exact requested-level chunk dependencies for the supplied
    coordinate grid and valid mask;
  - fetch/decode every dependency with `getChunkBlocking`;
  - store the returned `ChunkResult` objects in a local pinned map/vector before
    sampling starts, so `shared_ptr` chunk bytes stay alive even if the global
    decoded cache evicts entries;
  - sample only from those pinned requested-level chunks;
  - never call `sampleCoordsFineToCoarse` in this strict path.
- Treat chunk statuses explicitly:
  - `Data` and `AllFill`: valid requested-level data;
  - `Missing`: write black/fill and mark the pixel covered only for pixels that
    depend on genuinely missing requested-level chunks;
  - `Error`: fail loudly with the chunk key/error;
  - `MissQueued`: impossible after blocking; fail loudly if observed.
- Wire Python `Volume.sample_coords(..., blocking=True)` to the strict
  requested-level sampler. Keep non-blocking sampling on the existing
  progressive/fallback path for viewer-style callers.
- Extend returned sampler stats so debug code can verify semantics:
  - `requested_level_only: true` for blocking strict sampling;
  - `fallback_levels: 0`;
  - `missing_chunks` count for genuinely missing requested-level chunks;
  - keep existing `covered_pixels`, `requested_chunks`, and `error_chunks`.
- Update `Vc3dCoordinateSampler.sample_coords` to preserve and pass through the
  new stats without reinterpretation.
- Update Python debug callers:
  - 2D Trace2CP strip rendering should require `requested_level_only` when the
    sampler is a VC3D sampler and `blocking=True`;
  - native 3D Trace2CP `_sample_block_volume` should reject `error_chunks > 0`
    and any nonzero fallback stat;
  - native 3D volume panels and prediction/presence sampling should therefore
    share the same fixed semantics.
- Add concise debug details to failures:
  - requested level;
  - first failing chunk key/status;
  - sampler stats;
  - coordinate grid shape/context where available.

## Spec Update

- Update `planning/specs.md` to state that VC3D blocking coordinate sampling is
  strict requested-level sampling: all dependencies are decoded and pinned
  before sampling, scale fallback is disabled, and only true requested-level
  missing chunks may produce black fill.
- Update native 3D Trace2CP specs to require the strict sampler for both raw
  strip volume panels and inferred-block/presence sampling.
- Clarify that sampler `valid_mask` is not proof of requested-level data; it is
  only sample coverage/geometry validity.

## Docs Updates

- Update `docs/code_structure.md`:
  - describe the strict blocking path in `fiber_trace_2d/sampling.py` and the
    VC3D binding;
  - correct the native 3D Trace2CP section so it no longer says only
    `error_chunks` protects against fallback;
  - document the new stats used to confirm requested-level-only behavior.

## Testing Plan

- Add VC3D/C++ focused tests where practical:
  - blocking coordinate sampling returns requested-level values when a coarser
    level has different values;
  - blocking coordinate sampling does not call or use coarser levels;
  - requested-level missing chunks render black/fill without scale fallback;
  - requested-level errors fail loudly.
- Add Python regression tests:
  - `Vc3dCoordinateSampler.sample_coords(... blocking=True)` exposes
    `requested_level_only` and zero fallback stats;
  - native 3D `_sample_block_volume` rejects non-strict/fallback/error stats;
  - 2D Trace2CP blocking strip rendering rejects fallback/error stats.
- Run focused checks:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp"`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "trace2cp or sampler"`
  - rebuild/reinstall the local VC3D Python bindings per
    `planning/local_development.md`;
  - run the user's native 3D Trace2CP command on the reported segment and
    verify no low-res fallback panels are produced.

## Changelog

- Add a changelog entry that blocking VC3D coordinate sampling is now strict
  requested-level sampling and native 3D Trace2CP uses it for both volume and
  prediction rendering.

## Deviations Or Deferrals

- No trace scoring, fusion, model inference, normalization, or strip geometry
  changes are planned.
- Persistent `.empty` markers will still represent genuinely missing chunks.
  If a stale `.empty` marker exists for a chunk that is present remotely, that
  is a separate cache-corruption problem; the strict path will make such cases
  visible as requested-level missing black rather than hiding them with scale
  fallback.
