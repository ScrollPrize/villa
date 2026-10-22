# Task log: rolling live OME-Zarr input cache for shared 3D inference

## Planning findings

- Bulk auto-download currently completes before inference. TensorStore's
  existing bounded read-ahead only reads already-local chunks and is not a
  remote materialization cache.
- The shared runner currently uses transient local chunk existence both to
  skip model tiles and to classify supported output chunks. Live caching must
  make these checks authoritative against remote inventory or an initially
  empty/evicted cache can incorrectly suppress valid inference.
- Canonical traversal is Z-major with lazily materialized axis lattices. The
  live window can therefore remain bounded at 10,000 events and a safe eviction
  frontier can advance only after a complete canonical model Z row commits.
- At the current 512/96/32 defaults, PHercParis4 scale 0 has 6,241 tiles per
  model Z row. A 10,000-tile window spans at most three model Z positions and
  approximately 7--12 128-voxel input Zarr planes. The configured 10 TiB cache
  target is much larger than the immediate forward working set.
- A conservative target requires no ahead/LRU deletion: when over target,
  delete oldest whole planes strictly behind the safe boundary; if none are
  available, report and temporarily remain over target.
- `preprocess_cos_omezarr.py` still contains a duplicated automatic-download
  wrapper even though Fiber imports the shared runner helper. This task will
  remove that divergence rather than add another backend-specific cache path.
- Independent review required remote inventory to remain active-plane bounded,
  clarified sparse-plane eviction, preserved `.noremote` as advisory only,
  separated the 10,000 descriptor/materialization window from the much smaller
  TensorStore tile-array window, and strengthened local accounting, metadata
  validation, shared/exclusive locking, and fatal transport-error behavior.
- Current uncommitted downloader retry/progress/scanner-failure changes are the
  baseline for this task and must be preserved rather than rewritten away.

## Deviations

- Live fetching initially manages only the primary selected input scale.
  Lasagna `pred_dt` is an independent post-inference external source; live mode
  rejects a separately remote `pred_dt` rather than silently materializing a
  second unbounded cache. Already-local `pred_dt` behavior remains unchanged.
- S3 list responses are consumed as keys, not object-size projections. Progress
  therefore reports exact completed resident/downloaded bytes, current transfer
  rate, and in-flight chunk count, but does not invent a projected-byte total.
  Actual completed bytes remain the sole eviction trigger as required.

## Implementation

- Added `lasagna/live_omezarr_cache.py` with selected-level metadata validation,
  authoritative lazy per-Z remote inventory, atomic retrying raw-chunk transfer,
  aggregate local accounting, shared/exclusive advisory locking, and
  conservative whole-plane eviction.
- Kept canonical event generation lazy while separating the 10,000-item live
  materialization window from the existing bounded TensorStore/full-tile read
  window. Durable completed outputs are skipped before fetch. Serial and
  multi-device paths call the same shared scheduler.
- Changed live source-support checks to use authoritative remote inventory,
  never transient local cache contents or `.noremote` state.
- Removed the duplicate Lasagna downloader implementation and routed Fiber and
  Lasagna through the shared automatic-download helper. Bulk download takes the
  same exclusive selected-level lock used by live mutation; ordinary tiled
  readers take a shared lock.
- Added Fiber, Lasagna, and manager CLI flags, manager completion, source-link
  initialization without bulk prefetch, portable provenance, and final manager
  cache statistics. Normal non-live prefetch and no-prefetch behavior remains
  unchanged.
- Made progress refresh once per second through both terminals and manager
  pipes, with newline history checkpoints at most once per minute while work
  advances. Remote-missing accounting now counts each absent chunk once when
  its Z-plane inventory is first listed instead of counting overlapping tile
  requests repeatedly.
- Updated shared inference specs, code-structure documentation, Lasagna README,
  manager documentation, and changelog.

## Validation

- `python -m py_compile` passed for the new cache, shared runner, downloader,
  both inference entry points, and manager modules.
- `pytest lasagna/tests/test_live_omezarr_cache.py
  lasagna/tests/test_live_tiled_predict3d.py
  lasagna/tests/test_download_omezarr.py lasagna/tests/test_manager.py -q`:
  **77 passed**.
- Two focused legacy shared-runner tests covering normal Python-Zarr/TensorStore
  serial equivalence and lazy canonical events: **2 passed**.
- Fiber CLI/signature tests including live options and invalid combinations:
  **7 passed, 199 deselected**.
- Fiber, Lasagna, and manager `--help` smoke checks expose all three live flags.
- `git diff --check` passed.
- Focused progress/missing-accounting regression validation: **11 passed**.
- No real S3/full-volume performance run was made, so no throughput or peak-cache
  claim is reported. The focused synthetic tests verify window ordering,
  pre-fetch resume rejection, selected-scale isolation, exact accounting, and
  safe-plane deletion. A broader local suite attempt encountered an existing
  Zarr 3/Python 3.14 synchronous `open_group` stall in an unrelated atomic-output
  test; the focused normal-runner tests complete successfully.
