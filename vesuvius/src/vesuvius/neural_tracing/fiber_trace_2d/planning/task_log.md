# Native 3D Trace2CP Vectorized Beam Lookahead Log

## Findings

- Candidate output sampling and scoring currently run on `cache.device`, so
  normal CLI runs can use GPU for `grid_sample`, direction decoding, branch
  reduction, and candidate scoring.
- Current beam lookahead still has Python loops over frontier states and child
  node creation.
- Current `_trace_candidate_directions` builds NumPy candidates for one axis at
  a time, which blocks fully batched GPU candidate expansion.

## Planned Work

- Move beam-mode candidate generation, current-point lookup, candidate scoring,
  target-plane crossing, and lookahead frontier expansion to batched torch
  tensors on `cache.device`.
- Keep final path reconstruction in Python because only one selected path is
  reconstructed.

## Validation

- Pending implementation.

## Deviations / Deferred

- None so far.
