# Task Log

## 2026-09-03

- Confirmed the source combined dataset records a 512-base storage chunk, 16
  coordinate units per chunk, and maximum endpoint reach of 4 units: 128 base
  voxels per axis.
- Converted the available input extent from ZYX to XYZ:
  `[7168,17408,3072) .. [13824,24064,9728)`.
- Confirmed the target crop is
  `[10240,22016,6144) .. [11264,23040,7168)` and therefore has source margins
  `(3072,4608,3072)` below and `(2560,1024,2560)` above in XYZ.
- The final 512 lattice needs global offset `(256,256,256)` for its boxes to
  straddle the target crop boundaries; the source margins are sufficient for
  lattice overhang and endpoint dependency closure.
- Independent review fixed final-stage selection to the target crop rather
  than the padded search box. This produces exactly 27 final 512 boxes; graph
  materialization remains padded separately and falls through to unfiltered
  source data outside the overlay.
- Sparse absence is valid dataset semantics but cannot prove that a partial
  local mirror contains every intended remote chunk. The supplied encoded
  extent is geometrically sufficient; completeness still depends on the
  producer/inventory guarantee.
- Implemented exact box-driven planning. For the requested crop, the three
  passes contain 512, 343, and 27 boxes. Their unions are respectively
  `[9728,21504,5632)..[11776,23552,7680)`,
  `[9856,21632,5760)..[11648,23424,7552)`, and
  `[9984,21760,5888)..[11520,23296,7424)` in base XYZ.
- Extracted the existing transient reduction pass into reusable core code and
  layered it directly over a combined stored Fiberlet dataset. Crop tracing
  now accepts repeatable `--stage` values and records their complete policy in
  the trace artifact.
- The first requested run exposed compact-direction requantization across
  transient layers: one source-valid route no longer matched the curved-domain
  lattice after repeated decode/encode cycles. Transient rewritten chunks now
  use the lossless float32 cache profile, while untouched chunks continue to
  fall through to the compact source. A cross-profile overlay regression was
  added.
- A smoke run of all three stages over a 128-base crop passed filtering, graph
  reconstruction, tracing, artifact publication, and artifact reread.
- The Release 1024-crop run completed with stage totals:
  `5903905 -> 4011983`, `2082011 -> 1292725`, and
  `632703 -> 448512` Fiberlets. Filtering took approximately 2m45s, 1m04s,
  and 11s; final graph materialization took 4.58s and tracing 2.0s.
- The final output contains 2,000 accepted traces at
  `data/workdir3/fiber-crop-1024-staged/crop_traces.zarr`; visualization is
  `data/workdir3/fiber-crop-1024-staged/traces.obj`.
- Validation: Release build of `vc_fiber_trace_chunk`, `vc_fiberlets`,
  `test_fiberlet_storage`, and `test_fiberlet_crop_trace`; 41 storage cases and
  81 crop-trace cases passed. The subproject has no current `planning/spec.md`,
  so the behavioral contract was documented in `docs/fiber_chunk_tracing.md`.
