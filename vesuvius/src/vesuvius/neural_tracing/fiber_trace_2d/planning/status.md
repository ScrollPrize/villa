# Direction Visualization Image-Space Augmentations Status

- [x] Capture current user task in `planning/task.md`.
- [x] Create focused visualization task plan.
- [x] Add dir-vis image-space augmentation variants.
- [x] Concatenate augmented dir-vis panels into one natural-size strip output.
- [x] Add optional `--dbg-dirs` pasted-center debug row.
- [x] Update docs/specs and task log.
- [x] Compile-check changed Python.
- [x] Run focused dir-vis tests.

Result: `--dir-vis` now writes one labeled contact-sheet `dir_vis.jpg` with
identity, flip-x, flip-y, rot90, rot180, and rot270 panels. Each panel runs the
checkpointed model on the corresponding pixel-perfect image-space variant and
shows that augmented image with the existing native-resolution inference plus
nearest-neighbor 4x display upsampling, 8x8 display-cell, and 6-pixel
anti-aliased direction overlay. Dir-vis center-crops non-square center patches
to the largest native square with no rescaling, and the valid mask no longer
blacks out display pixels. `--dbg-dirs` adds a second row with raw/no-arrow
reference first, then half-image-sized unaugmented center pasted into
transformed contexts.
