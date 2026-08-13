# Accessing Vesuvius Data

This guide shows how to discover the datasets bundled with the Vesuvius package and how to load them through the Python APIs.

## Importing the Package

```python
import vesuvius
```

Accept the data sharing terms before trying to access remote assets:

```bash
vesuvius.accept_terms --yes
```

## Listing Available Scrolls and Segments

Call `vesuvius.list_files()` to read the packaged `scrolls.yaml` file. The dictionary mirrors the YAML structure: scroll → energy → resolution with entries for the canonical `volume` and any published `segments`.

```python
from vesuvius import list_files

scans = list_files()
```

Example layout (edited for brevity):

```python
{
    "1": {
        "54": {
            "7.91": {
                "volume": "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/",
                "segments": {
                    "20230827161847": "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/20230827161847.zarr/"
                }
            }
        }
    },
    "5": {
        "53": {
            "7.91": {
                "volume": "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr/20241024131838.zarr/"
            }
        }
    }
}
```

To refresh the YAML files against the public repository, run:

```python
from vesuvius.utils import update_list

update_list(
    "https://dl.ash2txt.org/other/dev/",
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels/"
)
```

## Listing Annotated Cubes

`vesuvius.list_cubes()` reads `cubes.yaml` and returns the available instance-annotated cubes:

```python
from vesuvius import list_cubes

cubes = list_cubes()
```

The dictionary is keyed by scroll → energy → resolution → cube coordinates. Each cube entry points to the base URL that hosts the corresponding NRRD pair.

## Working with `Volume`

`Volume` wraps multiresolution scroll volumes or segment surfaces. It can open remote assets described in the YAML files or a local Zarr store.

```python
from vesuvius import Volume

# Canonical Scroll 1 (energy/resolution inferred from defaults)
scroll = Volume("Scroll1")

# Explicit metadata for a scroll volume
scroll = Volume(type="scroll", scroll_id=1, energy=54, resolution=7.91, normalization_scheme="instance_zscore")

# Segment volumes can be resolved from their timestamp
segment = Volume(type="segment", segment_id=20230827161847)

# Load a local OME-Zarr directly
local = Volume(type="zarr", path="/data/Scroll1.zarr", normalization_scheme="instance_minmax")

# Return PyTorch tensors with a specific dtype
tensor_volume = Volume(
    type="scroll",
    scroll_id=5,
    energy=53,
    resolution=7.91,
    return_as_tensor=True,
    return_as_type="np.float32"
)
```

### Key Constructor Arguments

- `type`: `'scroll'`, `'segment'`, `'zarr'`, a canonical label such as `'Scroll1'`, or a segment timestamp.
- `scroll_id`, `energy`, `resolution`: optional overrides when the type does not embed them.
- `segment_id`: integer timestamp for segment volumes.
- `path`: local or remote URI for direct Zarr access (required when `type='zarr'`).
- `normalization_scheme`: `'none'` (default), `'instance_zscore'`, `'global_zscore'`, `'instance_minmax'`, or `'ct'`.
- `global_mean`/`global_std` or `intensity_props`: required when using `global_zscore` or `ct` normalization.
- `return_as_type`: numpy dtype string such as `'np.uint16'` or `'np.float32'`.
- `return_as_tensor`: return PyTorch tensors instead of NumPy arrays.
- `download_only`: download metadata (and ink labels for segments) without opening the full volume.
- `verbose`: print detailed diagnostics during initialization and reads.

### Inspecting Metadata and Data

```python
scroll.meta()              # Human-readable summary
shape_l0 = scroll.shape()  # Shape of resolution level 0
subshape = scroll.shape(1) # Shape of a downsampled level
print(scroll.dtype)        # Original dtype
slice_zyx = scroll[20:200, 10:40, 50:90, 0]  # (z, y, x, level)
```

If the store is a single-resolution Zarr array, omit the level index (`scroll[20, 10, 50]`). When requesting tensors, the same indexing syntax applies and returns `torch.Tensor` objects.

### Segment Ink Labels

For segments, `Volume` automatically attempts to download the companion PNG ink label. The array is available as `segment.inklabel`. Supply `download_only=True` to fetch the metadata and ink label without loading the voxels.

## Working with `Cube`

`Cube` loads instance-annotated NRRD cubes that pair a mask with the matching sub-volume.

```python
from vesuvius.data.volume import Cube

cube = Cube(scroll_id=1, energy=54, resolution=7.91, z=2256, y=2512, x=4816, cache=True)
volume_patch, mask_patch = cube[10, :, :]
```

Constructor arguments:

- `scroll_id`, `energy`, `resolution`: identify the parent scan.
- `z`, `y`, `x`: starting coordinates encoded in the cube folder name.
- `cache`: cache downloads locally (default `False`).
- `cache_dir`: override the cache directory when caching is enabled.
- `normalize`: divide voxel intensities by the dtype max.

`Cube.activate_caching(cache_dir=None)` enables caching after construction, while `Cube.deactivate_caching()` disables it.

## Additional Notes

- Remote access requires the packaged configuration files, installed under `vesuvius/install/configs/`.
- The package never auto-refreshes the YAML files at import time; call `update_list` when you need the newest inventory.
- For authenticated endpoints (for example, private S3 buckets), ensure your environment provides valid credentials that `fsspec` can read.
