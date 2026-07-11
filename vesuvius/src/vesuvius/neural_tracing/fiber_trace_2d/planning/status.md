# Direction Visualization Cell Spacing Status

- [x] Capture current user task in `planning/task.md`.
- [x] Create focused visualization task plan.
- [x] Update dir-vis overlay stride/segment rendering.
- [x] Update docs/specs and task log.
- [x] Compile-check changed Python.
- [x] Run focused dir-vis overlay test.

Result: `--dir-vis` now samples directions every fourth source pixel on the 2x
display image, so direction samples occupy 8x8 display-pixel cells. The segment
length remains proportional to cell size, yielding 6-display-pixel anti-aliased
segments with a visible border between neighboring samples.
