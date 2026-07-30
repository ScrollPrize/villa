# Task Log: VC3D Lasagna Attachment And Generic Remote File Cache

## Findings And Decisions

- VC projects already had taggable `lasagna_datasets`, while the first native
  fiber integration added a separate `fiber_inference_datasets` JSON field.
  The canonical representation is now one collection with reserved role tag
  `vc-lasagna-fiber`; untagged entries remain regular Lasagna data.
- `fiber_inference_datasets` is a legacy project-schema field only. It is read,
  tagged, merged, and deduplicated for compatibility, but is no longer written.
  The fiber manifest schema and learned tracer are not classified as legacy.
- Both supported remote-origin forms remain distinct: an explicit adjacent
  `lasagna-remote.json` controls a materialized local manifest, while a direct
  remote manifest resolves relative groups from its own parent URL.
- Generic `Volume`, remote volume specs, project storage, and VC3D remain
  strictly 3D. History shows the CZYX path came from the original May 2026
  Lasagna preprocessing/inference artifact contract, not generic VC volume
  support. Current inference writers emit separate per-channel 3D OME-Zarr;
  older flat CZYX preprocessing/fit intermediates require conversion and are
  rejected by VC3D attachment and normal sampling.
- Derived manifest/group/channel provenance is project bookkeeping for stable
  deduplication, reload reconstruction, stale-channel reconciliation, role
  changes, and detach ownership. It is not generic 4D grouping.

## Implementation

- Added a reusable exact-byte arbitrary single-file cache with cache-first and
  refresh policies, built-in HTTP/S3 transport, custom fetchers, atomic
  replacement with failed-refresh rollback, sidecar identity/size validation,
  in-process request coalescing, invalidation, and managed/unmanaged disk
  accounting.
- Integrated direct remote Lasagna manifest materialization into
  `LasagnaDataset::openLocation()`, including cache hit/download diagnostics,
  one retry for corrupt cached JSON, remote auth propagation, and reuse by
  `vc_fiber_trace_metric` without CLI changes.
- Ported Open Data manifest publication to the generic cache.
- Added canonical tagged project entries, old-field migration, Lasagna-owned
  3D channel preparation, provenance, atomic manifest-plus-volume attachment,
  reload reconciliation, role reclassification, and ownership-aware detach.
- Added VC3D File menu actions for local and remote manifests, regular/fiber
  role selection, background validation/preparation, atomic GUI-thread commit,
  and typed Detach entries.
- Updated Line Annotation resolution to use selected canonical tagged entries,
  the project cache root, and the original persisted locator for session
  identity. Tracer scoring and interaction behavior were not changed.
- Removed CZYX projection from the project-volume adapter, restricted Open Data
  validation and the VC3D normal sampler to 3D arrays, and migrated affected
  sampler fixtures to the current per-channel 3D representation. Generic
  volume and project loading behavior was not changed.

## Validation

- Configured dependencies were reused; no install or bootstrap command ran.
- Built with:

  ```bash
  cmake --build volume-cartographer/build/ci-tests-clang-systemdeps \
    --target test_remote_file_cache test_lasagna_manifest \
    test_lasagna_project_volumes test_volume_pkg \
    test_persistent_zarr_cache_budget test_open_data_manifest VC3D \
    vc_fiber_trace_metric -j 8
  ```

  Result: all targets built successfully. The existing Qt code emitted only
  deprecation warnings.

- Focused and broader related tests:

  ```bash
  ctest --test-dir volume-cartographer/build/ci-tests-clang-systemdeps \
    --output-on-failure \
    -R '^(test_remote_file_cache|test_lasagna_manifest|test_lasagna_project_volumes|test_volume_pkg|test_persistent_zarr_cache_budget|test_open_data_manifest|test_open_data_volume_prefill|test_zarr_chunk_fetcher|test_volume_pkg_full|test_volume_pkg_more)$'
  ```

  Result: 10/10 passed in 14.82 seconds. An intermediate run exposed that
  generic project re-resolution discarded a still-shared prepared Lasagna
  runtime volume after detach. `resolveAll()` now preserves Lasagna-owned 3D
  runtime objects, and the new shared-ownership detach regression passes.

- CLI surface check:

  ```bash
  volume-cartographer/build/ci-tests-clang-systemdeps/bin/vc_fiber_trace_metric --help
  ```

  Result: succeeded; `--remote-cache-dir` remains the shared remote-manifest
  cache option for both fiber and normal manifests.

- Follow-up CZYX removal validation:

  ```bash
  cmake --build volume-cartographer/build/ci-tests-clang-systemdeps \
    --target test_lasagna_project_volumes test_lasagna_normal_sampler \
    test_open_data_manifest -j2
  volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_lasagna_project_volumes
  volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_open_data_manifest
  python -m py_compile lasagna/lasagna_volume.py
  ```

  Result: all targets compiled; project-volume tests passed 4/4 and Open Data
  tests passed 32/32. The normal-sampler run passed the new CZYX rejection and
  migrated 3D cases but still fails two pre-existing assertions expecting 24
  batch-prefetch reads; the current committed direct sampling path reports 0.
  That unrelated counter-test mismatch was left unchanged.

## Deviations And Deferred Work

- The active no-delegation runtime policy prevented the nested `AGENTS.md`
  independent-agent plan review. A direct consistency audit was performed; no
  conflicting task/spec/plan requirements were found.
- No focused Qt interaction test was added. Core attachment/reconciliation is
  covered automatically and the VC3D target compiles; interactive local/remote
  attachment and Line Annotation tracing remain the explicitly requested next
  usage-test phase.
- The planned Qt work was kept in `MenuActionController` with the existing
  `VolumeAttachmentController` auth/cache methods made reusable, rather than
  adding a separate one-use Lasagna attachment controller. Validation and
  preparation remain in core, and all network/descriptor work still runs in
  the background task.
- Project-based atlas command-line tools still use local selected-manifest
  paths and have no remote-cache option. VC3D and `vc_fiber_trace_metric` are
  locator-aware; extending the separate atlas CLIs remains outside this task.
- Automatic TTL/ETag refresh, recursive directory caching, cross-process fetch
  locking, and remote atlas `init_shell_dir` listing/materialization remain out
  of scope. Cache-first treats a locator as stable until explicit refresh or
  invalidation.
