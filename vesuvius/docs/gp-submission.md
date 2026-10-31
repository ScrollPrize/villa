# Grand Prize submission meshes

The [Grand Prize rules](https://scrollprize.org/prizes#2027-grand-prize) ask
for flattened tifxyz meshes named by sequential column number
(`column_01.tifxyz`, `column_02.tifxyz`, ...), at most one per column of text
except for pieces where the papyrus is broken or detached, and one image per
column named after the mesh it was rendered from. `vesuvius.gp_submission`
cuts those meshes from a flattened surface (a tifxyz whose grid is already the
flattening, such as a segment exported by VC3D) and checks the folder before
it is handed in. It is a console script of the `vesuvius` package
(`pip install -e vesuvius` from a checkout), needs no GPU and never modifies
the source surface.

```bash
S=/path/to/submission
vesuvius.gp_submission cut-columns --submission $S --surface /path/to/flattened_surface --cuts cuts.csv
# render each mesh to $S/images/column_NN.tif (see Images below)
vesuvius.gp_submission check --submission $S
```

`cuts.csv` has the header `column,u0,u1` and an optional `kind` column, for
example:

```text
column,u0,u1,kind
1,100,340,column
2,340,600,column
2,600,650,break
```

`u0` and `u1` are grid columns of the surface as stored on disk (the width of
`x.tif`), `u1` exclusive. `column` runs from 1 to 999. `kind` is `column` (the
default), `break` or `patch`; break and patch pieces are written as
`column_NN_partK.tifxyz`, numbered from 1 per column. Every row is checked
before anything is written; a bad row, a duplicate column or a range with no
valid points stops the command with exit `2`, listing every bad CSV line (or,
for an empty range, the mesh and the surface's valid u range).
Gaps in the column numbers are left to `check`.
`cut-columns` refuses to write into a `meshes/` folder that is not empty;
`--overwrite` replaces the whole folder, and only once every new mesh has been
written. It does not touch `images/`, so re-render after a re-cut; `check`
flags images whose mesh is gone. Re-run `check` after a re-cut; the old report
stays until you do.

## Images

The rules require images generated programmatically from the CT volume, but
do not fix a tool or file format. One way is VC3D's `vc_render_tifxyz` (see
[installing VC3D](https://scrollprize.org/tutorial_VC3D#installing-vc3d)), as
in the [ink detection tutorial](https://scrollprize.org/tutorial5): render each
mesh through a volume and take the maximum over the slices. This renders the
4x downsampled level (`--group-idx 2`) of a volume on the same grid as the
scan the surface was traced on; `--volume` may be a local OME-Zarr or its
`https://` / `s3://` URL, and the max step needs `numpy` and `tifffile`:

```bash
mkdir -p $S/images
for M in $S/meshes/column_*.tifxyz; do
  N=$(basename $M .tifxyz)
  vc_render_tifxyz --volume /path/to/volume.zarr --group-idx 2 --cache-gb 4 \
    --scale 1 --scale-segmentation 1 --segmentation $M \
    --num-slices 16 --slice-step 0.5 --tif-output /tmp/render/$N
  python -c "import glob, sys, numpy as np, tifffile; \
tifffile.imwrite(sys.argv[2], np.max([tifffile.imread(p) for p in sorted(glob.glob(sys.argv[1] + '/*.tif'))], axis=0))" \
    /tmp/render/$N $S/images/$N.tif
  rm -r /tmp/render/$N
done
```

`--group-idx` already scales the mesh to the level it reads. Only a volume
whose level 0 is itself smaller than the scan, like the 4x downsampled
prediction in the ink tutorial, needs `--scale-segmentation 0.25`. Any file
named after its mesh (`column_01.png`, `column_01.tif`, ...) counts as its
image; `check` does not open it and ignores hidden files. Keep the banner
image outside `images/`.

## Gates

Required gates follow the rules and decide the exit code. Warnings cover
checks the rules do not state as a number or a requirement; they are reported
but do not fail the run.

| Gate | Kind | Passes when | Rule |
| ---- | ---- | ----------- | ---- |
| `tifxyz_format` | required | every `meshes/*.tifxyz` reads with `vesuvius.tifxyz` and has valid grid edges | "a set of meshes in tifxyz format" |
| `column_naming` | required | `meshes/` holds only `column_NN[_partK].tifxyz` (hidden files aside), numbered from 1 without gaps, two digits below 100 | sequential column numbers, as in `column_01.tifxyz` |
| `image_names_match_meshes` | required | `images/` is not empty, every file in it is named after a mesh, and no mesh has two | each image named after the mesh used to generate it |
| `every_mesh_has_image` | warning | every mesh has an image | the rules allow one image for a sequence of consecutive wraps, so a part or a joined column may share one |
| `mesh_isometry` | warning | each mesh's 3D grid-edge length over the grid step (`1 / scale`) lies within `--iso-tolerance` (default `0.10`, between 0 and 1) of 1.0 at p05 and p95 | the rules ask for a "low-distortion isometric" flattening with no number; the tolerance is ours |
| `no_overlapping_cuts` | warning | no two meshes cut from the same surface share grid columns | ours; follows from one mesh per column |

## Report

`check` writes `check_report.json` and `check_report.md` into the folder,
prints one summary line to stderr plus `FAIL name: message` or
`WARN name: message` for each failed gate, and exits `0` when every required
gate passes and `2` otherwise. A missing `meshes/` folder also exits `2` with
a single `error:` line. The gate records follow `vesuvius.surface_preflight`:
`name`, `required`, `passed`, `observed`, `threshold` and `message`, under a
`status` and `summary`. Leave the two report files out of the hand-in if you
do not want them there.

## What it does not check

It does not read the text or judge legibility, and it does not open the
images. It does not check the rest of the rules: that the meshes together
cover the whole scroll surface, that columns are numbered in winding order, that a `_partK` mesh really covers broken or detached papyrus,
scale bars, the scroll each image came from, the banner image,
the Docker image, or the written methodology. Isometry is measured on the stored
grid. Points with `z <= 0` are treated as missing, and a mesh whose
`meta.json` has no `scale` is read with the `vesuvius.tifxyz` default of 20,
as everywhere else in the package, which usually shows up as a large
`mesh_isometry` warning.
