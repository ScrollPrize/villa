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

- Coordinates are `(z, y, x)`, in full-resolution scroll-volume voxels.

## License

See <https://dl.ash2txt.org/LICENSE.txt>.
