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
| `verified_patches/` | Manually verified surface patches. Each patch is a grid-topology quad mesh sampled on the papyrus surface (84,316 items). |
| `unverified_patches/` | Candidate surface patches not yet manually verified. Announced as ~203,900 items; not published at this path yet (it returns 404). |
| `tracks/` | Line annotations: curves traced across the surface, stored as sequences of `(z, y, x)` points. |
| `fibers/` | Fiber annotations. Published as a separate dataset, [`fiber-skeletons`](https://dl.ash2txt.org/datasets/fiber-skeletons/); this path returns 404. |
| `outer_shell/` | Geometry of the scroll's outer shell. |
| `lasagna_inputs/` | Volume data consumed by the fitting pipeline. |
| `umbilicus.json` | The scroll's central axis: points defining the spiral center as a function of `z` (depth). |
| `same_windings.json` | Same-winding annotations — which annotations lie on the same wrap of the sheet. |
| `relative_windings.json` | Relative winding relationships (how many wraps apart two annotations are). |
| `abs_winding.json` | Absolute winding-number annotations. |

**Total:** ~49.6 GB across ~905,000 files.

## Conventions

- Coordinates are `(z, y, x)`, in full-resolution scroll-volume voxels — **except the
  JSON point collections**. `same_windings.json`, `relative_windings.json` and
  `abs_winding.json` store `(x, y, z)` in the voxel grid of **level 2** of the scan
  (18946 x 8174 x 8174), which is the grid the surface prediction is computed on.
  Read as `(z, y, x)` at full resolution, their points fall outside the papyrus.
- `umbilicus.json` names its axes per point (`x`, `y`, `z`) and is in that same
  level-2 grid.

## License

See <https://dl.ash2txt.org/LICENSE.txt>.
