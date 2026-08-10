# Task log: anchor-seeded fiberlet over-segmentation

## 2026-08-10 implementation

- Added `FiberPredictionField::storedGridInfo` and
  `sampleStoredGridBatch`. The boundary binds only exact canonical
  `presence/nx/ny`, returns the decoded unoriented axis without a fabricated
  reference direction, accepts different channel chunk layouts, and retains
  the existing local/remote Lasagna and decoded-chunk caches. Existing tracer
  support for prefixed prediction options is unchanged.
- Added reusable `FiberAnchors` types and a deterministic two-component
  non-orthogonal directional PCA mixture. The fit uses exclusive squared-dot
  assignment, fixed Gaussian/presence weights, compensated reductions,
  deterministic multistart and ties, aligned-support centroids, independent
  empty/degenerate/support rejection, and fixed-origin whole-cell crops.
- Added `vc_fiberlets anchors`, sparse version-1 `anchors.json`, and
  base-coordinate line-glyph `anchors.obj`. The source identity is a
  credential-free local/remote manifest locator plus an `fnv1a64:` hash of the
  materialized manifest bytes. JSON stores each component once rather than
  duplicating anchor data. Timing and worker count remain stdout diagnostics
  and are intentionally absent from artifacts so thread count cannot change
  bytes.
- Extracted the private project-save atomic string writer into shared
  `vc::core::util::atomicWriteString`; `VolumePkg` and fiberlet artifacts now
  use the same Windows/POSIX replacement behavior.
- Added 13 focused tests covering empty and straight cells, arbitrary axis-sign
  flips, 15/30/45/60/90-degree pairs, weak-component rejection, three-mode
  selection, inclusive support, clipped edge cells, whole-cell crop identity,
  thread-independent artifacts, canonical manifest binding, mixed chunk
  layouts, explicit scale requirements, shape mismatch, and prefixed-only
  rejection.

## Validation

- Configure: `cmake -S . -B build` in `volume-cartographer` using the existing
  cache. Result: Release, GCC 16.1.1, configuration succeeded.
- Build: `cmake --build build --parallel 32 --target vc_fiberlets
  test_fiber_anchors`. Result: succeeded.
- Tests: `ctest --test-dir build --output-on-failure -R
  '^(test_fiber_anchors|test_fiber_trace3d|test_lasagna_manifest|test_volume_pkg)$'`.
  Result: 4/4 executables passed; anchor suite contains 13 cases and existing
  native tracer contains 46 cases.
- Compile coverage: `cmake --build build --parallel 32 --target vc_cli_all`.
  Result: succeeded, including the registered `vc_fiberlets` target.
- Determinism: ran the same real 8-cubed prediction crop at 1 and 32 threads,
  then compared both artifacts with `cmp`. JSON and OBJ were byte-identical.
- Diagnostic inspection: projected the size-4 OBJ glyphs in XY. The sparse
  cell-local one/two-line topology and spatial distribution were coherent; no
  malformed primitives were visible.
- `git diff --check`: passed.

## Representative calibration

Release command shape, repeated for cell sizes 2, 4, and 8:

```bash
/usr/bin/time -f 'wall=%e peak_kib=%M' build/bin/vc_fiberlets anchors \
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json \
  /tmp/vc_fiberlet_anchor_bench_sN \
  --cell-size N \
  --crop-prediction-xyzwhd 1696,2528,2264,32,32,32 \
  --threads 32 --cache-gib 0.5
```

| Cell | Side/diagonal base vx | Cells | Anchors | Zero/one/two | Solver s | Wall s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16.0 / 27.7 | 4096 | 4079 | 1646 / 821 / 1629 | 0.196 | 0.25 | 51.1 |
| 4 | 32.0 / 55.4 | 512 | 542 | 148 / 186 / 178 | 0.272 | 0.29 | 47.5 |
| 8 | 64.0 / 110.9 | 64 | 79 | 7 / 35 / 22 | 0.076 | 0.09 | 51.1 |

These are single warm-cache calibration runs, not performance benchmarks.
No production cell size was selected: that requires overlaying the 3D OBJ on
the volume and comparing the side and diagonal against an agreed minimum
sustained sheet/fiber separation. The diagnostic XY projection is not a
substitute for that domain decision.

## Reuse and deferred work

- Direct remote manifest caching and `lasagna-remote.json` sidecars are not
  reimplemented. The CLI calls the existing `LasagnaDataset::openLocation`,
  whose direct-remote cache/refresh and sidecar behavior remains covered by
  `test_lasagna_manifest`; the new canonical stored-grid binding is covered
  with local tiny Zarr arrays.
- Connection search, directed DP/CUDA work, path-quality filtering,
  deduplication, and extension remain intentionally deferred to the next
  fiberlet stage, as specified in the task.
