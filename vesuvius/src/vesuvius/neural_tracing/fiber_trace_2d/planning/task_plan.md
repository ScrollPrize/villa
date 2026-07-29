# Native C++ Trace2CP Python-Parity Fixes Plan

## Implementation

1. Match Python beam selection:
   - score pruning as `cumulative_loss + depth * 1e-12`
   - preserve original child generation order on ties
   - use squared-distance pruning with the same `>= distance**2` keep rule
   - select reached-target states by cumulative loss only
   - remove `tracedLength` from search ordering decisions
2. Match Python compact normal principal axes:
   - replace the hint-seeded 16-step power iteration with a symmetric
     eigensolver for the largest eigenvector
   - keep existing hint/no-hint sign orientation semantics
3. Preserve candidate-loss math:
   - keep the active `candidate_substeps=1` all-pairs product unchanged
   - add focused regression coverage that verifies the C++ public trace path
     uses the all-pairs penalty rather than only presence or candidate/output
     alignment

## Spec Update

- Add native Trace2CP parity requirements for beam pruning/reached-state
  ordering, compact normal principal-axis decoding, and all-pairs candidate
  loss.

## Docs Updates

- No user-facing command documentation changes are needed.
- Update planning status, task log, and changelog.

## Tests

- Extend `test_fiber_trace3d` with focused regression tests.
- Build and run:
  - `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`
  - `volume-cartographer/build/bin/test_fiber_trace3d`
  - `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
