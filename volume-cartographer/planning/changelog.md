# Changelog

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
