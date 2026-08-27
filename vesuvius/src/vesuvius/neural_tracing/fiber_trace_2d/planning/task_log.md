# Task Log: Split-piece fiber BP

## Findings

- Constraint extraction already produces piece-local geometry and canonical
  same-source continuity links before spatial constraint search.
- BP currently maps every piece back to a unique source trace and therefore
  rejects both multiple pieces and continuity links.
- The CLI currently passes full source fibers to BP visualization and assumes
  each report vector has source-fiber cardinality; graph construction alone is
  therefore insufficient.
- Existing labeling semantics retain continuity as strong parallel evidence,
  not a mathematical equality. BP must preserve that behavior.

## Plan Review

- Independent review approved piece-node BP after requiring exact dense
  source-geometry clipping, full continuity-topology validation, explicit
  source-versus-piece cardinality, deterministic exact seed mapping, per-piece
  unary/balance semantics, complete CSV/OBJ identity, and stronger no-split
  equivalence coverage. The plan now includes those corrections.

## Deviations

- None.

## Validation

- Built `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` from the existing
  `volume-cartographer/build` configuration with `-j32`.
- `volume-cartographer/build/bin/test_fiberlet_crop_trace`: 72 test cases
  passed after adding split exactness, topology rejection, dense clipping,
  finite-continuity, unary, and no-split coverage.
- The Paris4 1024 crop with ordinary `--piece-length 512` produced 1,298 BP
  pieces from 500 source fibers and selected 26,402 full-orientation factors.
  Mixed-state sum-product converged in 0.745 seconds.
- The matching split `--perpendicular-only` smoke selected 15,017 factors and
  converged; a `--constraints-per-fiber 5` smoke selected 1,747 factors and
  converged, confirming both selector paths retain continuity.
- The split CSV contains one header plus 1,298 piece rows with global/source/
  local/arc identity. The ten orientation-band OBJs and four direct-state OBJs
  each partition all 1,298 pieces.
- `git diff --check` passed.
