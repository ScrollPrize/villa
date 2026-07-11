# Direction Visualization Image-Space Augmentations Plan

## Implementation

- Add a small dir-vis-only helper that materializes identity, flip-x, flip-y,
  rot90, rot180, and rot270 image-space variants for the loaded center strip
  patch and valid mask.
- Use contiguous arrays for the variants so PyTorch can consume flipped/rotated
  NumPy data safely.
- Run the checkpointed direction model once per variant and draw each variant's
  direction overlay in that variant's image coordinates.
- Write all variant overlays into one labeled `dir_vis.jpg` contact sheet and
  include per-variant drawn counts in `dir_vis_summary.txt`.
- Add `--dbg-dirs` for a second debug row that keeps the normal augmented row,
  then adds raw/no-arrow reference plus half-image-sized center-pasted
  variants inferred and rendered separately.

## Spec Update

- Document the explicit dir-vis image-space augmentation variants, the debug
  pasted-center row, and the single output image.

## Docs Updates

- Update `docs/code_structure.md` dir-vis description.
- Keep `planning/task_log.md` limited to this visualization task.

## Testing

- Compile-check `runner.py`.
- Run focused dir-vis overlay, augmentation, pasted-center, and layout tests.
