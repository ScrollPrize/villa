# Direction Visualization Image-Space Augmentations

Add explicit pixel-perfect image-space augmentations to the 2D fiber-trace
`--dir-vis` output so direction predictions can be inspected under simple
domain-shift probes.

Requirements:

- Use identity plus flip-x, flip-y, rot90, rot180, and rot270 variants.
- Apply these variants in image space; these transforms are allowed here because
  they are pixel-perfect.
- Run dir-vis inference for each variant and concatenate the resulting
  direction visualizations into one `dir_vis.jpg` output.
- Keep the existing 8x8 display-cell, 6-pixel anti-aliased direction segment
  rendering.
- Add `--dbg-dirs` for an extra debug row: raw unaugmented patch in the first
  cell, then transformed patches with their half-image-sized center overwritten
  by the unaugmented center crop and inferred/rendered separately.
