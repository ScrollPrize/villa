# Merge `fiber2d-tweaks`

User request:

- Merge the `fiber2d-tweaks` branch into the current `fiber-2d-training`
  branch in the same way as the previous merge.
- Run the merge first, inspect conflicts and spec changes, then resolve the
  conflicts while preserving all still-relevant current specs.

Current merge state:

- `git merge --no-commit fiber2d-tweaks` has been run.
- `runner.py`, `docs/code_structure.md`, and `planning/specs.md` auto-merged.
- Conflicts were produced in the per-task planning docs and
  `vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`.
- The incoming branch adds `--dir-vis` pixel-perfect image-space diagnostic
  variants: identity, flip-x, flip-y, rot90, rot180, rot270, plus optional
  `--dbg-dirs` pasted-center debug row.
- The current branch keeps strict coordinate-only geometric augmentation for
  training, augment-vis, line tracing, Trace2CP, labels, and TTA. The merge
  resolution must keep that rule while allowing the narrow `--dir-vis`
  diagnostic exception.
