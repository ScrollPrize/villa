# Changelog

- Routed Open Data prefill and Settings redownload through persistence-only
  maintenance demand on the process chunk-cache service. Exact source payloads
  are stored without decode, share keyed transfers with rendering, and run
  behind interactive and ordinary background work. Removed Volume-level
  private-cache factories and migrated in-process Lasagna corner sampling to
  canonical process-cache sources.

## 2026-08-15

- Completed process-global regular chunk-cache ownership. Normal volumes now
  discover the core-owned service internally, VC3D no longer injects services,
  and C++/Python cache policy callers use global configuration APIs.
- Moved decoded RAM capacity fully into the shared cache service and made
  runtime capacity changes preserve sources, queues, and in-flight work;
  reductions now evict only globally oldest decoded entries.
- Preserved every distinct control-point bend in generated annotation ribbons
  and changed intermediate supports to closest-target segment-local spacing,
  with explicit nonuniform support arclength mapping in both directions.
- Restored view-independent background ownership for exact SurfaceCache tile
  dependencies so frame-demand replacement cannot publish false incomplete
  tiles by cancelling a fill's normal-band chunks.
- Corrected remote bandwidth and adaptive admission to measure received HTTP
  payload bytes from request issue through completion, including connection and
  TTFB latency. Removed the obsolete four-completions-per-worker fallback and
  isolated local/custom fetches from remote statistics and persisted state.
- Missing and failed sparse-array requests now end only their own measurements
  instead of repeatedly resetting clean-start adaptive probing.
- Completion-paced admission ramps use every terminal request to release the
  next permit while retaining payload-only bandwidth evidence.

## 2026-08-14

- Replaced completion-span remote bandwidth estimates with service-wide
  streamed HTTP payload measurement. VC3D status, the Zarr download benchmark,
  and adaptive admission now share a five-active-second estimator.
- Made `ChunkCacheService` the sole source factory and scheduler owner. Source
  acquisition can no longer change global I/O policy, and runtime concurrency
  updates modify admission on the existing scheduler without cancelling,
  restarting, or duplicating work. Explicit batch caches remain isolated
  through their own services.
- Restored render-owned chunk priority: mouse input now marks only the active
  view, accepted renders publish captured-focus distances, and viewport
  occurrences deduplicate by declared projected chunk footprint. Compatible
  source reopen now adopts refreshed fetchers without evicting decoded chunks,
  while inactive different-source overlays clear only their own demand and
  reject obsolete queued callbacks.
- Added 100 to interactive priority for each source's terminal pyramid level,
  ensuring the best available whole-view fallback loads before ordinary
  relative levels while retaining relative ordering between terminal demands.
- Corrected generated-view scale declarations: line ribbons now arclength-
  resample to a uniform 50-base-voxel target, declare exact along/cross grid
  density, and retain original line semantics through a bidirectional strip
  mapping. Plane and generated views now share analytic source-level and
  fallback selection in base-volume units.

## 2026-08-13

- Removed the obsolete implicit `beginViewRequest()` epoch API, dead private
  decoded-cache routing hooks, and write-only surface-view generation state.
  Context-free chunk calls remain explicit background work, while interactive
  ownership continues through versioned per-view demand snapshots.
- Added per-view generation ownership for interactive chunk work. Superseded or
  closed-view probe/download/decode tasks are canceled while shared-view and
  explicit background requests are retained.
- Prevented stale running probes and downloads from entering another queue
  stage, and reject late asynchronous requests from a cleared view generation.
- Added service-wide adaptive remote download admission from two to 64 fetches.
  Completion-paced bracketed probes compare doubled and halved concurrency by
  encoded goodput and p90 latency. Stability requires five minutes of saturated
  observations; a 2x bandwidth change shortens exploration toward one minute,
  and underfilled queue tails retain the last saturated capacity estimate. The
  initial search uses 4x probes, then continuously refines at 2x until five
  direction reversals or retained-center brackets confirm a local optimum. The
  status bar uses the same rolling encoded-bandwidth samples.
- Persisted the adaptive remote-download operating point across clean VC3D
  restarts. Startup immediately uses the previous admission limit and capacity
  model while resetting stability history for frequent initial re-probing.
- Corrected interactive fallback-range selection after generated surfaces were
  given explicit base-volume parameter units.
- Split regular chunk loading into independent 32-worker persistent-cache
  classification, source download/read, and CPU decode queues so cached decode
  work no longer delays discovery and admission of remote misses.

## 2026-08-12

- Added per-scale unresolved-fetch counts to VC3D's existing cache status bar
  during active remote downloads.
- Corrected the shared RAM/disk GiB display and merged Z-scroll sensitivity into
  the same status label.
- Unified VC3D regular decoded chunks behind one source-qualified application
  cache service, retaining warm data across volume switches and sharing base,
  overlay, Spiral, and surface-filler source reads.
- Added a reduced-resolution viewport dependency pre-pass and focus-aware,
  multi-view chunk scheduling. Pending GUI work is ordered by active view,
  coarse level, and pointer distance while background requests receive bounded
  fair service; direct and SurfaceCache rendering reuse their existing geometry
  paths.
- Expanded interactive fallback demand to as many as five coarser levels,
  bounded by average chunk-to-viewport coverage, and retained that demand during
  refinement renders.
- Added the opt-in `--debug-download-queue` VC3D overlay, which colors pixels
  belonging to actively fetched remote chunks by pyramid level in all shared
  slice viewers.

## 2026-08-08

- Added a synthetic Valgrind rendering benchmark with native replay scoring and
  a one-sided performance-only CI regression gate.
