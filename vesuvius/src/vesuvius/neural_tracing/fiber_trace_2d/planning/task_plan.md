# Plan: rolling live OME-Zarr input cache for shared 3D inference

## Scope and invariants

1. Preserve `lasagna.tiled_predict3d.run_tiled_inference_3d` as the sole tiled
   neural inference scheduler for Fiber and Lasagna. Product adapters retain
   model/output semantics; live source materialization belongs to shared input
   scheduling.
2. Preserve normal behavior exactly when live mode is disabled: the existing
   blocking auto-download path, crop handling, output resume, TensorStore and
   Python-Zarr readers, numerical preprocessing, accumulation, flushing, and
   output formats remain unchanged.
3. Live cache mutation is authorized only by `--live-fetch`. The disk target is
   configured by `--live-cache-gib` and defaults to `10240` GiB (10 TiB).
   `--live-fetch-ahead-tiles` defaults to `10000`. `--download-workers` remains
   the transfer-concurrency control. These disk settings are distinct from
   TensorStore's RAM `--input-cache-gib`. Explicit live-cache/lookahead CLI
   overrides without `--live-fetch` are errors rather than silently ignored.
4. Reject live mode before mutation unless all of these hold:
   - inference has no crop;
   - `--no-download` is not active (reject this conflict before metadata or
     cache mutation);
   - the input path is a local numeric OME-Zarr-v2 array level;
   - its group root has valid `_download` metadata and an S3 source;
   - the configured cache target, lookahead, and worker count are positive.
   Initially reject a separately remote `pred_dt` live source; an already local
   post-stage remains unchanged.
5. Only paths that parse as valid 3-D chunk keys under the exact resolved input
   level are cache data. Never delete `.zarray`, `.zattrs`, group metadata,
   `.dl_cache`, temporary state outside the selected level, another numeric
   level, symlink targets, or unrecognized files.
6. Before any chunk mutation, compare the local selected-level Zarr-v2 metadata
   with the selected remote level metadata: shape, chunks, dtype, order,
   compressor, filters, fill value, and dimension separator must agree. Refuse
   stale/mismatched local metadata rather than copying raw chunks into an
   incompatible array.

## Shared cache component

7. Add one reusable Lasagna module for selected-level source resolution,
   chunk-coordinate/path handling, atomic S3 chunk transfer, lazy per-Z-plane
   remote inventory, local selected-level inventory/accounting, and the live
   cache controller. Reuse existing downloader S3/retry and raw-byte transfer
   behavior rather than copying another fetch implementation.
8. Remove the remaining duplicated Lasagna predict3d `_auto_download` wrapper
   and make Fiber and Lasagna call the same shared source-resolution/download
   helper. Keep `download_omezarr.py` as the bulk-prefetch CLI/programmatic
   entry point.
9. Use one advisory lock per selected level. Ordinary shared inference readers
   acquire a shared lock; live mutation and bulk downloader writes acquire the
   exclusive lock. Refuse conflicting cooperating processes. Lock files live
   under `.dl_cache`, not inside the level. Document that unrelated external
   readers which ignore the advisory lock remain outside this protection.
10. At live startup, scan only the selected local level for valid chunk files,
    aggregate count/actual bytes by input Z-chunk index, and discard individual
    paths. This gives exact current-scale accounting without retaining millions
    of path objects, enumerating inference jobs, or touching other scales.
    Ignore metadata, temp files, malformed entries, and every symlink leaf or
    directory component. Rescan only a selected plane when evicting it. Support
    both Zarr-v2 `.` and `/` dimension separators.

## Lazy fetch and source support

11. Keep the existing canonical axis lattices and lazy nested Z/Y/X event
    generator. Add a separate bounded live-materialization window of at most
    `live_fetch_ahead_tiles`; do not form the global Cartesian tile list.
12. Before downloading a tile, determine whether any output product work
    remains without consulting transient local input presence. Completed output
    tiles are skipped before fetch/read submission.
13. Map each candidate tile's clipped input read bounds to exact selected-level
    Zarr chunk coordinates. Lazily list each required remote Z-chunk prefix
    authoritatively, regardless of advisory `.noremote` contents. Retain only
    per-plane present-key sets for the active/materialization window, derive
    missing keys from a successfully completed listing, and drop plane
    inventory after it is behind the safe frontier with no active operation.
    Relisting after process restart is acceptable; never grow an eventual
    full-scale Python set of chunk keys.
14. Materialize only remotely present chunks needed by the bounded tile window.
    Deduplicate concurrent requests by chunk key, reuse already valid local
    files, and write downloads through unique temporary files plus atomic
    replacement. Definitively absent remote chunks remain absent and count as
    supported fill/masked space rather than transfer errors.
15. A tile may enter TensorStore/Python-Zarr read submission only after all its
    remotely present chunks are local. At that point source-support checks use
    authoritative live inventory, not the accidental current filesystem state;
    otherwise an initially empty or subsequently evicted cache could falsely
    classify valid output as unsupported.
16. Keep fetch composition and downloads outside GPU workers. The coordinator
    polls bounded futures and continues GPU/result/flush work; it must not wait
    for all lookahead tiles to materialize before starting the first ready
    canonical tiles. The 10,000-item window contains cheap tile descriptors and
    chunk-materialization futures only; it must never create 10,000 full tile
    arrays. Only materialized work enters the independently bounded existing
    TensorStore window of `prefetch_tiles_per_gpu * device_count` reads.
17. Treat remote listing, GET, and atomic-write failures as fatal after the
    existing bounded SDK/application retries. Preserve completed chunks for
    resume, but never reinterpret transport failure or a listed-then-404 object
    as masked fill. `.noremote` remains advisory and eviction never adds to it.

## Conservative Z-plane eviction

18. Advance a monotonic safe input-Z boundary only after every tile/read in a
    canonical model Z row has completed and committed. Convert the next row's
    clipped input lower bound to a Zarr-plane boundary. A plane is eligible only
    when its exclusive voxel end is at or before that boundary.
19. The relevant/protected band includes every active or future read from the
    oldest uncommitted row through the end of the 10,000-tile materialization
    window. Never evict a plane intersecting this band or a plane with an active
    download/read.
20. After actual completed selected-level chunk bytes exceed the target, select
    the oldest eligible Z plane and synchronously/atomically sweep every
    currently cached valid chunk file having that `iz`; in a masked sparse
    volume, "complete plane eviction" means no eligible cached files for that
    `iz` remain, not that every theoretical Y/X chunk was present. Prune empty
    chunk directories, update accounting, and repeat plane-by-plane until
    actual usage is at or below target or no eligible plane remains. Do not
    evict merely because projected/in-flight bytes cross the target, choose Y/X
    subsets, use LRU, or delete far-ahead planes.
21. The target is deliberately conservative rather than hard. Allow temporary
    overshoot from protected data, downloads already in flight, filesystem
    accounting delay, or lack of an eligible old plane. Do not stall inference
    or violate the safe boundary merely to meet the target. Retry eviction when
    the next Z-row commit advances the boundary.
22. On interruption/failure, stop scheduling, settle/cancel fetch work, retain
    every atomically completed chunk, clean unique temporary files where safe,
    and leave the sparse cache directly resumable. Evicted chunks are not
    recorded as remote-missing.

## CLI, manager, progress, and provenance

23. Add the three live options to both Fiber `infer.py` and Lasagna
    `predict3d`, validate them consistently, and pass one shared configuration
    into `run_tiled_inference_3d`.
24. Add explicit manager `inference run` live options. `--live-fetch`
    suppresses the manager's separate full-prefetch phase, initializes the
    `_download` source link, and forwards the target/lookahead settings.
    Preserve the existing `--no-prefetch` behavior for non-live runs and reject
    conflicting combinations, including configured/passed `--no-download`,
    before initializing source metadata.
25. Report selected scale, target/current/projected cache bytes, lookahead,
    transfer rate/counts, remote-missing chunks, safe Z boundary, evicted
    planes/chunks/bytes, and conservative over-target state without emitting an
    idle stream of repeated newline records.
26. Record live-fetch enablement, target, lookahead, and final
    downloaded/reused/missing/evicted/peak-resident statistics in portable
    inference provenance and manager run metadata. Do not add these runtime
    cache details to Lasagna manifests or Atlas data-entry schemas.

## Tests and validation

27. Unit-test source/level metadata validation, current-scale-only inventory, malformed
    and symlink exclusion, remote Z-plane inventory, request deduplication,
    missing remote chunks, atomic downloads, and advisory lock contention.
28. Unit-test eviction with synthetic levels: only whole safe Z planes are
    removed; protected/current/future planes and other scales/metadata survive;
    deletion repeats until under target; insufficient eligible space leaves a
    reported overshoot without failure; eviction never creates noremote state.
29. Exercise the shared scheduler with a fake remote/local Zarr: initially
    absent valid chunks must not be skipped, fully masked tiles must remain
    absent, resumed output avoids fetch, lookahead stays bounded/lazy, and
    delayed fetch overlaps other ready work where canonical ordering permits.
    Add a stale `.noremote` case whose remotely present chunk is still listed
    and fetched. Assert the 10,000 materialization and smaller TensorStore read
    windows are bounded independently.
30. Run equivalent small-volume inference with full-prefetch and live-fetch
    inputs through both direct Lasagna and Fiber adapters and compare persisted
    output chunks/metadata. Cover TensorStore plus Python-Zarr fallback and
    serial plus multi-device scheduler logic where existing fake workers allow.
31. Test manager argument generation and lifecycle: live mode skips the bulk
    prefetch request, initializes source metadata, records settings, and launches
    the same shared backend flags.
32. Run focused Lasagna downloader, shared predict3d, manager, and Fiber 3D
    tests; compile/import smoke tests; and `git diff --check`. For performance,
    report only measured synthetic/representative startup, fetch lead, GPU input
    starvation, peak cache, and eviction figures. A real full-volume run is not
    required for correctness validation.
33. Preserve and rerun the pending downloader retry/progress/scanner-failure
    regression work already present in the worktree; live-cache extraction must
    not overwrite or weaken that baseline.

## Spec update

Extend the Shared 3D Tiled Inference specification with the opt-in rolling
selected-scale disk cache, exact defaults, full-volume-only validation,
authoritative source support, bounded lazy tile materialization, current-scale
isolation/locking, and conservative whole-Z-plane eviction semantics. Clarify
that 10 TiB is a target ceiling and lack of safe obsolete planes permits
temporary overshoot. Preserve the existing sole-runner, no-global-job-list,
resume, and numerical-output requirements.

## Docs updates

Update `docs/code_structure.md`, `lasagna/README.md`, and the manager docs with:

- live-fetch architecture and its relationship to TensorStore read-ahead;
- Fiber, Lasagna, and manager command examples;
- 10 TiB/10,000-tile defaults and disk-vs-RAM cache distinction;
- full-volume-only and selected-scale deletion safety restrictions;
- Z-boundary eviction/temporary-overshoot behavior;
- interruption/resume and concurrent-mutator locking behavior.

## Changelog

Add a concise dated entry for shared rolling live OME-Zarr input caching after
implementation and validation. Put detailed findings, deviations, commands,
and results in `planning/task_log.md`.
