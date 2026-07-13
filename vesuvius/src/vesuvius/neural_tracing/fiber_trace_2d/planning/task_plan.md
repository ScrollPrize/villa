# Torch-Vectorized Trace2CP DP Backend Plan

## Implementation

- Add a torch backend for `_trace2cp_top_monotone_direction_path_z`.
- Keep the existing NumPy/Python DP as fallback when no torch device is passed.
- Reuse the existing input validation and preprocessing where practical:
  direction fields, valid masks, presence fields, candidate-angle penalty,
  move grids, columns, and progress labels.
- Implement the torch backend as a sequential loop over DP columns, with each
  column vectorized over:
  - current move chunks,
  - all z layers,
  - all image rows,
  - all previous moves,
  - all sampled pixel columns along a transition.
- Store backpointers in CPU NumPy arrays to avoid keeping the large
  backpointer tensor on GPU, while keeping active DP cost tensors on the torch
  device.
- Route Trace2CP CLI side/z/top DP calls through the torch backend using the
  existing model/device.
- Preserve progress output and final timing rows.

## Spec Update

- Document that Trace2CP CLI DP runs use a torch-vectorized backend when a
  torch device is available, falling back to the NumPy/Python backend for
  direct calls.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP section to describe the torch
  backend and the remaining sequential column recurrence.

## Tests

- Add a parity test comparing torch and NumPy DP outputs on a small deterministic
  problem.
- Run focused Trace2CP tests and the full `test_fiber_trace_2d_loader.py`.

## Changelog

- Add a dated changelog entry for the torch-vectorized Trace2CP DP backend.
