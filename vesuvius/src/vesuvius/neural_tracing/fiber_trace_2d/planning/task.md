# Direction Visualization Cell Spacing

Adjust the 2D fiber-trace `--dir-vis` overlay so direction samples are spaced on
8x8 display-pixel cells instead of 4x4 cells.

Requirements:

- Keep the existing 2x nearest-neighbor patch display scale.
- Draw one direction segment every fourth source pixel so each sample owns an
  8x8 display-pixel cell.
- Draw 6-display-pixel segments, leaving a visible border between neighboring
  samples.
- Anti-alias the drawn direction segments.
