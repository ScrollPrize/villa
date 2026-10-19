# `vc_objrefine` (alpha-comp refine)

`vc_objrefine` moves each vertex of a surface along its normal onto the
papyrus it finds in the CT. For every vertex it composites the volume front
to back along the normal and moves the vertex to the opacity-weighted offset
of the material it meets, plus `border_off`. VC3D's Alpha-comp refine dialog
runs this tool.

It can also keep those per-vertex offsets as a report instead of discarding
them (see [Offset report](#offset-report)).

## Usage

```sh
vc_objrefine <zarr-volume> <src-surface> <out-surface> params.json
```

The source is either an `.obj` with one `vn` line per `v` line, or a tifxyz
directory; the output has the same type.

| Key | Default | Meaning |
| --- | --- | --- |
| `refine` | `true` | Move the vertices. With `false` the surface is written unchanged |
| `start`, `stop`, `step` | `-2`, `30`, `2` | Scan range and sample spacing along the normal, in voxels |
| `low`, `high` | `118/255`, `165/255` | Opacity ramp: intensities below `low` are transparent, above `high` opaque. Values above 1 are read as 0 to 255 |
| `r` | `3` | Probe radius: a `(2r+1)`-square patch, Gaussian-blurred, read at its center pixel |
| `reader_scale` | `0.5` | The probe patch's pixel spacing is `1/reader_scale` voxels |
| `border_off` | `1` | Added to the offset when moving a vertex |
| `scale_group` (or `scale_level`) | `1` | Zarr level to read |
| `cache_bytes`, `cache_mb` or `cache_gb` | 10 GB | Decoded chunk cache size |
| `overwrite` | `true` | tifxyz output: replace an existing output directory (or an earlier report) |
| `gen_vertexcolor` | `false` | `.obj` output: write the sampled intensity as vertex colors |
| `offset_report` | `false` | tifxyz only: also save the offset report below |
| `offset_threshold` | `5.0` | Report summary: count vertices whose offset magnitude is above this, in voxels. Must be 0 or more |
| `offset_min_opacity` | `0.5` | Report summary: below this accumulated opacity a vertex counts as "no sheet found". Must be in [0, 1] |

Offsets and voxel distances are in the surface's own coordinates (full
resolution voxels) at every `scale_group`.

## Offset report

`"offset_report": true` keeps the offset `vc_objrefine` computes for each
vertex instead of discarding it. It is opt-in, needs tifxyz input (the maps
live on the segment grid; for `.obj` input the flag is ignored with a
warning) and never changes the surface written. It is set in the params JSON
on the command line; VC3D's dialog does not set it.

With `"refine": false` the run only measures, and the output path is a
report folder, not a surface: it gets the three files below and nothing
else.

```json
{"refine": false, "offset_report": true}
```

```sh
vc_objrefine volume.zarr paths/my_segment reports/my_segment params.json
```

The report folder must be new, empty, or an earlier report (replaced when
`overwrite` is true). The tool refuses the source segment, a folder inside
it, and any folder that holds other files, such as another segment.

With `"refine": true` the refined surface is written as usual and the report
goes into its `alphacomp_offset/` subfolder.

### Outputs

- `offset.tif`: float32 on the segment grid, the offset of each vertex along
  its normal in full-resolution voxels. NaN where no ray was cast, which are
  the vertices refine leaves in place: invalid points, the last row and
  column, vertices next to an invalid point, and vertices with no valid
  normal.
- `opacity.tif`: float32 on the same grid, the scan's accumulated opacity
  from 0 to 1. Near 0 means no papyrus was found between `start` and `stop`.
- `report.json`: the inputs (absolute volume and source paths, source UUID),
  the scan parameters, and a summary: counts of valid, measured, "no sheet"
  and summarized vertices; signed offset mean and 5th, 50th and 95th
  percentiles; offset magnitude 50th, 90th, 95th and 99th percentiles and
  maximum; the count and share of summarized vertices beyond
  `offset_threshold`. "No sheet" vertices (opacity below
  `offset_min_opacity`) are left out of the statistics, so
  `beyond_threshold_fraction` is a share of the vertices where papyrus was
  found and `no_sheet_fraction` a share of the measured ones.

The three files are written to a temporary folder and moved into place
together. If that fails after a refine run has saved the surface, the tool
exits with 1 and says the surface was saved without its report.

### Reading the map

- The offset is the opacity-weighted mean of the sample positions from
  `start` to `stop`, so it lies in that range and is resolved to about
  `step`.
- It is one-sided: it finds the first papyrus ahead along the normal. A
  vertex inside papyrus reads `start`; a trace past its sheet reads the next
  papyrus ahead, or "no sheet", never a negative distance. It is an offset
  readout, not a sheet-switch detector.
- Where nothing was found the offset reads 0, the same as a vertex on a
  papyrus face. Read it only where `opacity.tif` is at least
  `offset_min_opacity`, as the summary does.
- The maps describe the source vertex positions and use refine's own probe,
  whose patch starts at the vertex rather than centering on it (about 8.5
  voxels off with the defaults). With `"refine": true` each vertex moved by
  exactly `offset + border_off`.

A later `vc_objrefine` run into the same output removes an old
`alphacomp_offset/` folder, so a report never outlives the geometry it
measured.
