# Volume registration

This directory contains a script (`find_transform.py`) to find a transform between two volumes.
It runs a local [neuroglancer](https://github.com/google/neuroglancer) instance to display the volumes, and adds functionality to find a transform between them.
Live visual overlay allows the transform to be found by manually aligning the volumes.

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

### Overview

Typically one finds a transform by following these steps (details below):

- Performing a coarse initial alignment by rotating, translating, and flipping the moving volume using keybinds until it roughly aligns with the fixed volume.
- Adding manual landmark points to each volume based on visual features, refining the alignment.
- (Optional and not recommended at this time) Using SimpleITK to fit a transform. The current implementation uses low-resolution levels of the Zarr input volumes, and does not result in precise transforms.

[Overview video](https://drive.google.com/file/d/1d05znwDmNCJdOsLd8VlH0clRorNhtcKg/view?usp=drive_link)

#### Visualization

- `c` - Toggle volume color

#### Coarse initial alignment

First one roughly positions the moving volume using the following commands:

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

Next, landmark points are added to each volume based on visual features.
These refine the transform.
After there are 4+ pairs of landmark points, the transform is automatically fit to the landmark points each time a point pair is added.

- `Alt + 1` - Add landmark point to fixed volume at cursor position
- `Alt + 2` - Add landmark point to moving volume at cursor position

#### Refining landmark points

- `Alt + x` - Delete nearest landmark point
- `Alt + [` - Navigate to previous fixed point
- `Alt + ]` - Navigate to next fixed point
- `Shift + j` - Perturb fixed point +X
- `Shift + u` - Perturb fixed point -X
- `Shift + k` - Perturb fixed point +Y
- `Shift + i` - Perturb fixed point -Y
- `Shift + l` - Perturb fixed point +Z
- `Shift + o` - Perturb fixed point -Z

#### Automatically refining the transform
> **_NOTE:_**  Not particularly recommended, as the current implementation uses low-resolution levels of the Zarr input volumes, and does not result in precise transforms.

The transform can be automatically refined using image registration via SimpleITK.
The registration method uses the lower resolution Zarr levels and the Mattes mutual information metric to register the volumes.

- `f` - Fit the transform to the landmark points using SimpleITK

#### Saving the transform

- `w` - Write the current transform to the output file. This also prints a shareable neuroglancer URL that can be used to view the volumes with the transform applied.
