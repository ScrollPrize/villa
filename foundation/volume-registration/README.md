# Volume registration

This directory contains a script (`find_transform.py`) to find a transform between a fixed Zarr volume and a moving source.
The moving source can be either another Zarr volume or a Neuroglancer-loadable VTK mesh.
It runs a local [neuroglancer](https://github.com/google/neuroglancer) instance to display the data and adds functionality to find a transform by manual alignment.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Example invocation:

```bash
python -i find_transform.py \
--fixed SCROLLS_HEL_4.681um_113keV_1.2m_binmean_2_PHerc_0500P2_HA_0001_masked.zarr/ \
--fixed-voxel-size 9.362 \
--moving PHerc500P2-0.5um_masked.zarr/ \
--output-transform output_transform.json \
--initial-transform initial_transform.json
```

### Example invocation with a moving mesh:

```bash
python -i find_transform.py \
--fixed SCROLLS_HEL_4.681um_113keV_1.2m_binmean_2_PHerc_0500P2_HA_0001_masked.zarr/ \
--fixed-voxel-size 9.362 \
--moving /absolute/path/to/MAN5_scaled.vtk \
--moving-type mesh \
--moving-source-unit-size 1000 \
--output-transform output_transform.json
```

If `--moving` points to a local `.vtk` file, the script automatically serves its directory over a local HTTP server with CORS enabled and passes the resulting `vtk://http://127.0.0.1:<port>/...` URL to Neuroglancer.
In mesh mode, the tool also overlays live intersection contours of the moving mesh in the orthogonal slice views using annotation lines in fixed-space coordinates.
Use `--mesh-slice-max-segments` to reduce the number of contour segments kept in that overlay if you want a lighter-weight slice rendering path. The default is `256`.
Live pan/slice tracking for that mesh slice overlay is enabled when `numba` is available.

### Overview

Typically one finds a transform by following these steps (details below):

- Performing a coarse initial alignment by rotating, translating, and flipping the moving source using keybinds until it roughly aligns with the fixed volume.
- Adding manual landmark points to each source based on visual features, refining the alignment.
- (Optional and not recommended at this time) Using SimpleITK to fit a transform. The current implementation uses low-resolution levels of the Zarr input volumes, and does not result in precise transforms.

[Overview video](https://drive.google.com/file/d/1d05znwDmNCJdOsLd8VlH0clRorNhtcKg/view?usp=drive_link)

#### Visualization

- `c` - Toggle volume color

#### Coarse initial alignment

First one roughly positions the moving source using the following commands:

- Step sizes can be customized via `--small-rotate-deg`, `--large-rotate-deg`, `--small-translate-voxels`, and `--large-translate-voxels`.
- `Alt + a` - Rotate +X (`+ Shift` for bigger step)
- `Alt + q` - Rotate -X (`+ Shift` for bigger step)
- `Alt + s` - Rotate +Y (`+ Shift` for bigger step)
- `Alt + w` - Rotate -Y (`+ Shift` for bigger step)
- `Alt + d` - Rotate +Z (`+ Shift` for bigger step)
- `Alt + e` - Rotate -Z (`+ Shift` for bigger step)
- `Alt + f` - Flip X
- `Alt + g` - Flip Y
- `Alt + h` - Flip Z
- `Alt + j` - Translate +X (`+ Shift` for bigger step)
- `Alt + u` - Translate -X (`+ Shift` for bigger step)
- `Alt + k` - Translate +Y (`+ Shift` for bigger step)
- `Alt + i` - Translate -Y (`+ Shift` for bigger step)
- `Alt + l` - Translate +Z (`+ Shift` for bigger step)
- `Alt + o` - Translate -Z (`+ Shift` for bigger step)

#### Adding landmark points

Next, landmark points are added to each source based on visual features.
These refine the transform.
After there are sufficient pairs of landmark points (4+ in unconstrained mode, 3+ in constrained mode), the transform is automatically fit to the landmark points each time a point pair is added.

- `Alt + 1` - Add landmark point to fixed volume at cursor position
- `Alt + 2` - Add landmark point to moving volume at cursor position

#### Refining landmark points

- Point perturb step can be customized via `--point-perturb-voxels`.
- `Alt + x` - Delete nearest landmark point
- `Alt + [` - Navigate to previous fixed point
- `Alt + ]` - Navigate to next fixed point
- `Shift + j` - Perturb fixed point +X
- `Shift + u` - Perturb fixed point -X
- `Shift + k` - Perturb fixed point +Y
- `Shift + i` - Perturb fixed point -Y
- `Shift + l` - Perturb fixed point +Z
- `Shift + o` - Perturb fixed point -Z

#### Reviewing landmark errors

After the transform is fit from landmarks, a **landmark errors** layer appears in the neuroglancer viewer showing the registration error for each landmark pair. The error is the distance (in voxels) between the fixed landmark and the corresponding moving landmark after transformation.

- Points are **color-coded** from green (low error) to red (high error), with marker size scaling with error magnitude.
- The neuroglancer **side panel** (click the `landmark_errors` layer) shows a table of all points with their error values. Click a column header to sort, and click a row to navigate to that point.
- A **summary table** is also printed to the terminal each time errors update, showing all landmarks sorted by error along with RMS and max error.
- `Alt + Shift + ]` - Navigate to the landmark with the largest error

This is useful for identifying landmarks that may need adjustment, or for spotting regions where a non-affine transform might be needed.

#### Constrained fit mode

By default, landmark fitting uses an unconstrained 12 DOF affine (requires 4+ point pairs). For volumes that share physical constraints — roughly aligned z-axis, isotropic scaling, rotation mainly around z — a **constrained fit mode** is available that fits only 5 continuous parameters: isotropic scale, z-rotation angle, and 3D translation. It additionally tries all 8 axis flip combinations (independent ±1 on each axis) and picks the best. This requires only 3+ point pairs.

- `m` - Toggle between constrained and unconstrained fit mode
- `--constrained-fit` - Start in constrained fit mode

#### Automatically refining the transform
> **_NOTE:_**  Not particularly recommended, as the current implementation uses low-resolution levels of the Zarr input volumes, and does not result in precise transforms.

The transform can be automatically refined using image registration via SimpleITK.
This is only available when the moving source is also a Zarr volume.
The registration method uses the lower resolution Zarr levels and the Mattes mutual information metric to register the volumes.

- `f` - Fit the transform to the landmark points using SimpleITK

#### Saving the transform

- `w` - Write the current transform to the output file. This also prints a shareable neuroglancer URL that can be used to view the volumes with the transform applied.

## Checking points against a second scan (`interface_distance.py`)

Once a `transform.json` exists between two scans of the same object, it can be
used in the other direction: take points you already have in one volume's
voxel frame (predicted ink voxels, the centre of a measured depth band on a
segment, hand-placed landmarks) and ask whether the second scan sees a sharp
intensity change at the same place. `interface_distance.py` maps each point
through one or more transform files, reads a short intensity profile along a
local normal in the reference volume, and reports the distance `D` from the
point to the nearest qualifying `|gradient|` peak, in voxels of the frame the
points were given in.

What `D` does and does not mean: a small `D` says the point sits at an
interface that an independent scan also sees. Papyrus backs, folds and dense
fibres all produce interfaces, so it does not say which side of a sheet the
point is on, and it says nothing about ink. It is a geometric consistency
check, useful for catching depth that is off by several voxels or that wanders
between neighbouring points.

```bash
# points in the canonical 2.4um PHerc. Paris 4 frame, reference is the 1.129um
# rescan; its transform.json registers 1.129um (moving) to canonical (fixed),
# so the chain is one inverse step
python interface_distance.py \
    --points w00_20231016151002_depth_anchors.csv \
    --step inv:/path/to/1.129um/transform.json \
    --reference https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260608103018-1.129um-0.2m-78keV-masked.zarr/0 \
    --normal points \
    --output w00_interface_distance.json --csv w00_interface_distance.csv
```

- `--step PATH` applies `p_fixed = M @ p_moving` as written in the file;
  `--step inv:PATH` applies the inverse. Steps compose in order, so a chain
  through a shared fixed volume is `--step a.json --step inv:b.json`.
- `--points` accepts a plain CSV with `x,y,z` (optional `nx,ny,nz`) columns, or
  the `*_depth_anchors.csv` written by `export_depth_anchors.py` in
  khj1222/vesuvius-challenge, in which case the point is the band centre.
- `--normal volume` (default) estimates the normal from the reference window's
  structure tensor and rejects points without a dominant direction;
  `--normal points` uses the normal supplied with each point.
- `--moving-offset-zyx dz dy dx` adds a constant shift before the chain, for a
  residual registration offset you have measured separately.
- Reading a window per point straight from the bucket is slow; point
  `--reference` at a local copy when scoring more than a few hundred points.

The output JSON holds the composed chain, the per-point records (profile,
normal, `D`, reasons a point was not evaluable) and a summary with the median
`D`, the fraction at or below `--threshold`, and its Wilson 95% interval.
Tests: `python -m pytest tests/test_interface_distance.py`.
