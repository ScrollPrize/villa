# Direction Visualization Cell Spacing Plan

## Implementation

- Update `_direction_field_overlay_rgb` to use a default stride of 4 source
  pixels while retaining the 2x image scale.
- Keep the segment-length formula tied to the display cell size so stride 4
  yields 6-display-pixel direction segments.
- Render the overlay on a small supersampled RGBA canvas and downsample it so
  segment edges are anti-aliased.
- Update the dir-vis export summary to report stride 4, 8-pixel cells, and
  6-pixel segments.

## Spec Update

- Document the 8x8 display-pixel cell, 6-display-pixel segment length, and
  anti-aliased direction drawing.

## Docs Updates

- Update `docs/code_structure.md` dir-vis description.
- Keep `planning/task_log.md` limited to this visualization task.

## Testing

- Compile-check `runner.py`.
- Run the focused dir-vis overlay test.
