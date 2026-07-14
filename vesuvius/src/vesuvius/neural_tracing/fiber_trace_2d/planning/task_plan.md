# Startup Compact Geometry Acceleration Plan

## Findings

- Current startup geometry construction is correct but serial:
  - `_initialize_fiber_line_geometry_store()` loops over records;
  - `_build_geometry_for_record()` loops over every line point;
  - each line point calls `_lasagna_normal_at_zyx()`;
  - `_lasagna_normal_at_zyx()` performs three tiny 2x2x2 channel reads
    (`grad_mag`, `nx`, `ny`), trilinear interpolation, Lasagna ambiguous normal
    decoding, and a small principal-axis solve.
- Measured baseline with `loader_example.json`:
  - startup compact geometry build: `8m53s`;
  - records: `464`;
  - valid/skipped CPs: `14773/184`;
  - compact store memory: `63.5 MiB`;
  - hot path after startup: `compact_geometry=0.012 ms/patch`.
- The startup build has three clear waste sources:
  - record construction is independent but currently sequential;
  - normal sampling is point-by-point Python work with many tiny zarr reads;
  - normals are sampled for line points that may never be inside any configured
    CP source window.

## Goals

- Implement all four acceleration directions together:
  1. parallelize by record;
  2. vectorize per-record normal sampling/decoding;
  3. preprocess only CP-relevant line points;
  4. combine and benchmark the full path.
- Preserve deterministic record order, sample order, CP validity semantics, and
  compact store sharing.
- Do not add a new worker-count config key. Reuse existing `loader_workers` for
  startup compact geometry construction:
  - `loader_workers=1`: serial/debug path;
  - `loader_workers>1`: parallel record build.

## Implementation Plan

### 1. CP-Relevant Line Range Filtering

- Add a helper such as `_required_line_ranges_for_record(record, source_shape_hw)`.
- For every control point in the record:
  - compute its existing source-window line-index bounds with
    `_line_window_bounds_for_control_point()`;
  - retain the CP's exact `control_point_line_index()`;
  - record the CP-to-required-range mapping for later validity checks.
- Merge overlapping or adjacent line-index ranges into compact intervals.
- Only sample Lasagna normals for line indices inside those merged ranges.
- Keep arrays full-record sized where downstream code expects full line-index
  addressing:
  - `normals_xyz`: full `[num_line_points,3]`, with `NaN` outside relevant
    ranges;
  - `normal_valid`: full `[num_line_points]`, false outside relevant ranges.
- A CP remains valid only if its entire source-window range lies inside one
  valid normal interval.
- This keeps behavior strict: no fabricated normals, no nearest-normal
  propagation, no valid CP if required data is missing.

### 2. Vectorized Lasagna Normal Sampling

- Add a batched normal sampler, for example
  `_lasagna_normals_at_zyx_batch(record, points_zyx_base, line_indices)`.
- The batch sampler should preserve `_lasagna_normal_at_zyx()` semantics:
  - `grad_mag` must be present and trilinearly positive;
  - `nx` and `ny` must be present;
  - Lasagna normals must be decoded through
    `lasagna.omezarr_pyramid._decode_normals`;
  - the two-sign ambiguity is handled by the same tensor/principal-axis logic.
- Avoid thousands of tiny zarr reads:
  - process each merged relevant line interval, or sub-ranges of it;
  - compute channel-space bounding boxes with one-voxel interpolation margin;
  - read `grad_mag`, `nx`, and `ny` blocks once per sub-range;
  - if a bounding box is too large, split the interval and retry.
- Use NumPy vectorization inside each loaded block:
  - compute base indices and interpolation fractions for all points;
  - gather the 8 interpolation corners for `grad_mag`, `nx`, and `ny`;
  - compute trilinear weights in arrays;
  - decode all 8 normal corners with `_decode_normals`;
  - build `tensor[N,3,3]` and `hint[N,3]` with vectorized weighted sums.
- Vectorize the principal-axis solve:
  - run the current 16-step power iteration over all valid tensors with
    `np.einsum`/batched matrix-vector multiplication;
  - sign-align against the vectorized hint, matching scalar behavior.
- Return:
  - `normals[N,3]`;
  - `valid[N]`;
  - compact invalid reason metadata sufficient to produce useful CP skip
    diagnostics.
- Keep the scalar `_lasagna_normal_at_zyx()` for focused tests, debugging, and
  fallback if the vectorized path encounters an unsupported array/indexing
  case.

### 3. Parallel Record-Level Startup Build

- Modify `_initialize_fiber_line_geometry_store()` to use a
  `ThreadPoolExecutor` when `loader_workers > 1`.
- Use `worker_count = min(loader_workers, total_records)`.
- Each worker builds exactly one record geometry using the CP-relevant,
  vectorized path.
- The main thread collects futures and stores results by original
  `record_index`; final `by_record_index` order must be identical to serial.
- Progress output remains in the main thread:
  - records done/total;
  - valid/skipped CP totals;
  - resident memory;
  - ETA;
  - optionally worker count in the start line.
- Do not let progress output interleave from workers.
- Do not duplicate the compact geometry store:
  - workers return `_FiberLineGeometry` plus memory/valid/skipped counts;
  - the main thread updates the single shared store.
- Preserve clone behavior: cloned loaders still receive the already-built store
  by reference.

### 4. Combined Integration

- Update `_build_geometry_for_record()` so it uses:
  - CP-relevant line ranges;
  - vectorized normal sampling for those ranges;
  - existing interval construction and `build_side_strip_frame_arrays()` over
    valid contiguous ranges.
- Keep `build_side_strip_frame_arrays()` and source-grid reconstruction
  unchanged unless a bug is uncovered.
- Keep the exact hot-path compact lookup shape from the previous task.
- Keep invalid CP behavior deterministic:
  - record geometry may be built out of order;
  - final store order and CP validity arrays are ordered by original record/CP;
  - training/prefetch sample stream sees the same ordered dataset.

## Spec Update

- Add to `planning/specs.md`:
  - startup compact geometry construction may parallelize by record using
    `loader_workers`;
  - startup normal sampling may vectorize channel reads and normal decoding but
    must preserve Lasagna decoding, ambiguity handling, and strict invalid-data
    semantics;
  - startup preprocessing should only require line points that can affect the
    configured CP source windows;
  - `loader_workers=1` is the serial deterministic debug path.
- No new config key is planned.

## Docs Updates

- Update `docs/code_structure.md`:
  - describe record-level parallel compact geometry build;
  - describe CP-relevant range filtering and vectorized normal sampling;
  - document the `loader_workers` interaction for startup geometry.
- Update `planning/local_development.md` with the benchmark command and the
  baseline/result table.
- Update `planning/changelog.md` after implementation.
- Keep `planning/task_log.md` scoped to this task only.

## Testing

- Unit tests:
  - vectorized normal sampling matches scalar `_lasagna_normal_at_zyx()` on
    local fake/zarr arrays within tight tolerance;
  - CP-relevant range filtering does not sample irrelevant line endpoints;
  - invalid `grad_mag == 0` inside a required CP window still invalidates that
    CP;
  - invalid data outside all CP source windows does not invalidate unrelated
    CPs;
  - serial (`loader_workers=1`) and parallel (`loader_workers>1`) startup
    produce identical CP validity arrays and source geometry for representative
    samples;
  - cloned loaders share the same built geometry store.
- Regression command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Benchmark / Validation

- Reuse the exact benchmark command already used for this task:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`
- Compare against baseline:
  - startup: `8m53s`;
  - compact store memory: `63.5 MiB`;
  - hot path: `76.76 patches/s`, `compact_geometry=0.012 ms/patch`.
- Report:
  - startup total wall time;
  - valid/skipped CP counts;
  - compact store MiB;
  - hot-path benchmark summary;
  - profile breakdown;
  - worker count used.

## Risks / Checks

- Threading may not scale if zarr/NumPy work holds the GIL too often. The
  vectorized block sampler is included to remove much of that Python-loop work.
- Bounding-box block reads can over-read if a fiber interval spans a large
  region. Split intervals by max block size to avoid memory spikes.
- Vectorized power iteration may differ by tiny floating-point roundoff from
  scalar per-point logic. Tests should compare with tolerances and source-grid
  regressions, not exact bit identity.
- Reading larger channel blocks can increase transient memory. Keep block size
  bounded and release block arrays after each interval/sub-range.
