# Task Log

## Recovered Baseline

- Output crop, base XYZ half-open: `[10240,22016,6144) ->
  [11264,23040,7168)`, side 1024.
- Default trace lookahead: 384 base voxels on every face.
- Required search range, base XYZ half-open: `[9856,21632,5760) ->
  [11648,23424,7552)`, side 1792.
- Fiber source: manager Fiber 3D run
  `s1a_128_1_single_8x8_20260801_084232`, checkpoint `best91_5k.pt`,
  requested source group 1.
- Normal source: manager Lasagna run
  `20260419_180421_conthr_1e-5_warp_2um_noss_dist`, checkpoint
  `model_current.pt`, requested source group 2.
- Fiberlet prediction spacing: 8 base voxels; storage chunk side: 512 base
  voxels; encoding: compact directions and nonlinear uint16 costs.
- Stored trace baseline: 1998 traces with 2000 maximum attempts.
- Evaluation cohort: best 25% by trace cost density, 512-base-voxel pieces,
  26 `hendrik_crop1` reference fibers.

## Decisions

- Document managed full-volume prediction and Fiberlet preprocessing because
  the manager deliberately rejects cropped prediction bundles.
- State the smaller expanded range separately as the minimum graph support
  needed by the crop trace, not as a substitute for the recorded full-volume
  source artifacts.

## Implementation

- Added `volume-cartographer/docs/fiber_pruning_benchmark.md` with the complete
  managed generation and evaluation pipeline.
- Linked it from `volume-cartographer/docs/fiber_chunk_tracing.md`.
- Removed fixed-crop tuning history, benchmark formulas, population results,
  and timings from the general crop-tracing reference; it now links to the
  standalone guide for those workload-specific details.
- Recorded the requested, search-expanded, and storage-aligned regions
  separately to avoid XYZ/ZYX or crop/support ambiguity.

## Independent Review

- Distinguished the manager Lasagna prediction used for Fiberlet construction
  from the `las008_s1_full` normals used by tracing and evaluation.
- Added source run UUIDs, manifest and model hashes, reference inventory hash,
  and the differing source/Fiber/normal base shapes.
- Marked the staged Fiberlet dataset as a partial local mirror and the capped
  1998-trace dataset as the intentional frozen benchmark cohort.
- Separated fresh trace generation from repeated evaluation so benchmark runs
  reuse validated preprocessing artifacts.

## Validation

- Parsed every Bash code block after substituting documented placeholders and
  checked it with `bash -n`.
- Confirmed all documented `vc_fiber_trace_chunk` options against `--help` and
  the manager snapshot-selector form against `lasagna/docs/manager.md`.
- Recomputed the 26-file reference inventory digest as
  `1a2a5c0d608f8b5b6cf9ceb361a78ff163eea640422662d669d89a33eeca3b90`.
- `git diff --check` passed.
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 97 test cases
  passed.
