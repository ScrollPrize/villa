# Native C++ Trace2CP Python-Parity Fixes Task Log

## Notes

- Re-checked Python pruning:
  - `_prune_native_beam_tensor_indices()` scores states as
    `cumulative_loss + depth * 1e-12`.
  - spatial pruning repeatedly takes the current tensor-order `argmin`, then
    masks endpoints whose squared distance is below `distance**2`.
  - reached-target selection takes the first reached state with minimum
    cumulative loss only.
- Updated C++ pruning to sort state indices stably by that same score and to
  preserve generation order on score ties. Removed `tracedLength` from search
  ordering decisions.
- Updated C++ reached-target selection to use the same minimum-loss-only rule
  and removed the previous `std::partition` plus comparator path.
- Replaced compact normal principal-axis power iteration with `cv::eigen` on
  the accumulated symmetric tensor. Existing hint/no-hint sign orientation is
  preserved.
- The active C++ candidate-loss formula for `candidate_substeps=1` already
  matched Python's all-pairs direction product, so it was preserved and covered
  with a public trace-path regression test.

## Deviations

- No command-line or manifest behavior changed.
- The focused candidate-loss work is a regression test, not a formula rewrite,
  because the active formula was already aligned with Python for the current
  native metric command.

## Validation

- `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`
  passed.
- `volume-cartographer/build/bin/test_fiber_trace3d` passed: 23 test cases.
- `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
  passed.
- `git diff --check` passed.
