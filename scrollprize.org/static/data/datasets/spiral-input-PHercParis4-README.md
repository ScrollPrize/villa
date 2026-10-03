# `spiral-input` — PHercParis4

Manual annotations of the spiral (winding) structure of scroll **PHercParis4**
(Scroll 1), used as ground-truth inputs for fitting a global winding solution.

In cross-section, a scroll is a single papyrus sheet wound into a spiral around
a central axis (the *umbilicus*). The annotations here record which parts of
the surface belong to the same wrap (*winding*) of the sheet, and how windings
relate to one another, so that a global solution for the sheet's path through
the volume can be fit and evaluated.

## Contents

| Path | Description |
| --- | --- |
| `verified_patches/` | Manually verified surface patches. Each patch is a grid-topology quad mesh sampled on the papyrus surface (~27,399 items). |
| `unverified_patches/` | Candidate surface patches not yet manually verified (~203,900 items). |
| `tracks/` | Line annotations: curves traced across the surface, stored as sequences of `(z, y, x)` points. |
| `fibers/` | Fiber annotations. |
| `outer_shell/` | Geometry of the scroll's outer shell. |
| `lasagna_inputs/` | Volume data consumed by the fitting pipeline. |
| `umbilicus.json` | The scroll's central axis: points defining the spiral center as a function of `z` (depth). |
| `same_windings.json` | Same-winding annotations — which annotations lie on the same wrap of the sheet. |
| `relative_windings.json` | Relative winding relationships (how many wraps apart two annotations are). |
| `abs_winding.json` | Absolute winding-number annotations. |

**Total:** ~49.6 GB across ~905,000 files.

## Conventions

- Coordinates are in **9.6 µm voxels**, level 2 of the 2.4 µm volume `20260411134726-2.400um-0.2m-78keV-masked.zarr`
  (multiply by 4 for that volume's level-0 voxels), matching `voxel_size_um: 9.6` in `spiral-scroll.json`.
  Checked for `verified_patches/`, `outer_shell/`, `umbilicus.json`, `same_windings.json`, `relative_windings.json`
  and `abs_winding.json`.
- Exception: `eval_fibers/` are in level-0 (2.4 µm) voxels, as their `vc_open_data_*` fields state.
- Axis order: `umbilicus.json` uses `x`/`y`/`z` keys; points in the winding files are `[x, y, z]`; patches are
  tifxyz meshes with separate `x`, `y`, `z` images.

## License

See <https://dl.ash2txt.org/LICENSE.txt>.
