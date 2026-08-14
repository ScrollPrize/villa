# Changelog

## 2026-08-14

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
- Added service-wide adaptive remote download admission from two to 64 fetches,
  based on recent successful encoded-chunk bandwidth and chunk size; the status
  bar now uses the same bandwidth estimate.
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
