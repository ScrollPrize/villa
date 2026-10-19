# vc_reader

Tools for working with Vesuvius Challenge data: discover scans and annotations,
read and sample large CT volumes, and transform aligned image and label inputs on CPU
or CUDA. The package is built for large multidimensional images and PyTorch
training pipelines; its Zarr reader and transforms also work with your own data.

- **Discover data:** browse published metadata for samples, volumes, segments,
  ink labels and geometry, including scans at a requested resolution.
- **Read and sample:** fetch local, HTTP or S3 Zarr crops with bounded caches,
  batch reads, and sample rotated boxes or curved surfaces.
- **Align targets:** read CT with ink and supervision masks, sample registered
  labels into a volume grid, or project surface annotations into 3D.
- **Process images and labels:** apply CPU/CUDA geometry, intensity, noise and filter
  transforms to 2D or 3D inputs while keeping images and labels aligned.
  `CudaLoader` adds GPU transforms and prefetching to a PyTorch DataLoader.

## Quick start

See [installation and notebook setup](examples/README.md#run). This example
discovers two image/ink-label pairs on PHerc0009B and PHercParis4, then trains
one tiny ink classifier on two automatically selected tiles from each pair.
It requires PyTorch, a build with
`VC_READER_ENABLE_CUDA=ON`, and an accessible CUDA device. Reads use the CUDA
backend; DLPack shares the returned GPU buffers with PyTorch without downloading
them. A small CPU scan of the label masks finds annotated tiles before any CT
pixels are fetched. The 25 surface layers become input channels for a 2D CNN
with batch size 1.

A tile is a rectangular crop of the flattened surface image. Its `(y, x)` is
the top-left **row and column** in the selected image pyramid level, shared by
the paired image, label and masks. These are not coordinates in the original
3D scan. The example walks a grid of tiles and keeps the first two containing
supervised training pixels for each pair.

```python
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vc_reader import load, ScrollSource

catalog = load()
selections = [  # Sample and segment IDs from the catalog.
    ("PHerc0009B", "20250919125754"),
    ("PHercParis4", "20231210121321"),
]

# discover labeled regions using a lower resolution
LEVEL, TILE_SIZE = 3, 128 
pairs = []
for sample_id, segment_id in selections:
    scroll = ScrollSource(sample_id, catalog=catalog)
    label = scroll.segments[segment_id].labels.ink(label_set="20260918")
    surface_volume = label.surface_volumes.zarr()
    tiles = []
    # An empty CT layer slice reads only labels and masks while locating tiles.
    with label.open_paired(image=surface_volume, level=LEVEL, layers=np.s_[0:0]) as overview:
        height, width = overview.shape
        for y, x in product(range(0, height, TILE_SIZE), range(0, width, TILE_SIZE)):
            patch = overview[y:y + TILE_SIZE, x:x + TILE_SIZE]
            support = patch.label.valid & (patch.label.masks["supervision"] > 0)
            if "validation" in patch.label.masks:
                support &= patch.label.masks["validation"] == 0
            if support.any():
                tiles.append((y, x))  # Top-left image pixels, found from annotations.
                if len(tiles) == 2:
                    break
    if not tiles:
        raise ValueError(f"No supervised training tiles found for {sample_id}/{segment_id}")
    pairs.append((label, surface_volume, tiles))

# construct a tiny ex model
model = nn.Sequential(
    nn.Conv2d(25, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 1, kernel_size=1),
).cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
steps = 0

readers = [
    (label.open_paired(
        image=surface_volume, level=LEVEL, layers=np.s_[40:65], backend="cuda"), tiles)
    for label, surface_volume, tiles in pairs
]

# train 
for epoch in range(3):
    for reader, tiles in readers:
        for y, x in tiles:
            patch = reader[y:y + TILE_SIZE, x:x + TILE_SIZE]
            image = torch.from_dlpack(patch.image.data).float() / 255  # (1, 25, H, W)
            ink = (torch.from_dlpack(patch.label.data) > 0).float()    # (1, H, W)
            image_valid = torch.from_dlpack(patch.image.valid).bool().all(dim=1)
            label_valid = torch.from_dlpack(patch.label.valid).bool()
            supervision = torch.from_dlpack(patch.label.masks["supervision"]) > 0
            train_mask = image_valid & label_valid & supervision
            if "validation" in patch.label.masks:
                train_mask &= torch.from_dlpack(patch.label.masks["validation"]) == 0
            if not train_mask.any():
                continue

            optimizer.zero_grad(set_to_none=True)
            pixel_loss = F.binary_cross_entropy_with_logits(model(image)[0], ink, reduction="none")
            loss = (pixel_loss * train_mask).sum() / train_mask.sum()
            loss.backward()
            optimizer.step()
            steps += 1

for reader, _ in readers:
    reader.close()

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs_completed": epoch + 1,
    "steps": steps,
}, "ink_checkpoint.pt")
print(f"Completed {steps} optimizer updates; saved ink_checkpoint.pt")
```

Only supervised pixels with valid labels and valid CT in every input layer
contribute to the loss. Published validation masks are excluded when present;
PHercParis4 has no published validation mask in this selection. Readers stay
open across epochs; call `close()` when training is finished to release their
caches and resources. For full training and
evaluation with a spatial holdout, see the [2.5D ink training notebook](examples/01_ink_training_25d.ipynb).

Each `surface_volume` is CT sampled along its segment's curved surface; its
depth axis indexes surface layers. `label.volume` identifies the original 3D
scan, which can be read in ZYX voxel coordinates through `label.volume.open(level=0)`.

`.surface_volumes.zarr()` requires a unique matching render; multiple matches
raise an error. Use filters such as `artifact_id=...` to select among them.

`patch` keeps each target's data, geometric validity and companion masks
together.

## Performance 

**Zarr reading and sampling — median milliseconds per request:**

| Workload | zarr-python | TensorStore | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: | ---: | ---: |
| Batch of 8 × 128³ crops | 14.158 | 4.894 | **1.147** | 4.796 |
| Curved surface, 96 × 512 | 10.824 | 6.645 | **0.868** | 1.792 |
| Rotated box, 128³ | 69.288 | 65.259 | 13.792 | **1.629** |


**Transform pipeline — median milliseconds per 128³ crop:**

| Work | CPU throughout | CPU read → CUDA processing |
| --- | ---: | ---: |
| 12 transforms only, excluding transfers | 146.465 | **0.673** |
| Read + normalize + 12 transforms + CPU output | 149.723 | **1.833** |


## Examples and guides

The [notebooks](examples/README.md) cover 2.5D and 3D ink training, general Zarr
reading, CUDA transform, and mesh-segment rendering.

| Topic | Guide |
| --- | --- |
| Metadata, resolution queries and custom data | [Discovery and pairing](docs/discovery.md) |
| Source formats, storage and caches | [Source reading](docs/source-reading.md) |
| Registered labels and fields | [Aligned sampling](docs/aligned-sampling.md) |
| Surface layers and projecting ink into CT | [Surface sampling](docs/surface-sampling.md) |
| Measurements and reproduction commands | [Benchmarks](benchmarks/README.md) |

## Transforms

Image transforms and normalization support channel-first **2D `(C, Y, X)`
and 3D `(C, Z, Y, X)`** arrays on CPU and CUDA. Use
`from vc_reader import transforms as A`. Expand an entry for its measured
configuration, timings and previews; the recipes are examples, not constructor
defaults. Z/Y/X slice pairs show the original above the transformed output.

Use fixed parameters for processing, such as `A.GaussianBlur(sigma=1.0)`, or
sample parameters with `A.Uniform`, `A.Normal` and `A.Choice` for augmentation.
Several constructors have random defaults; pipeline calls take an explicit
`seed` for reproducibility, and `Plan` records sampled operations for replay.

| Operation family | Examples |
| --- | --- |
| Spatial | `Affine`, `Flip`, `CropPad`, `Elastic`, `ForegroundCrop` |
| Intensity | `Normalize`, `Gamma`, `CLAHE`, `HistogramMatching` |
| Filters | `GaussianBlur`, `MedianFilter`, `OrientedBlur`, `UnsharpMask` |
| Noise and acquisition effects | `GaussianNoise`, `BiasField`, `GibbsRinging`, `ThickSlice` |
| Masking and dropout | `MaskImage`, `GridDropout`, `ObjectDropout` |
| Labels and morphology | `Morphology`, `ReplaceLabel`, `Regions`, `OneHot` |
| Features | `DistanceTransform`, `StructureTensor`, `MedialSurface` |
| Layout and output construction | `ToPseudo2D`, `FromPseudo2D`, `DeepSupervision` |

Gallery timings use 128³ float32 CT crops, one CPU thread, and medians of five
calls after one warmup. **CUDA uploaded** includes upload and download;
**CUDA resident** starts and ends on the GPU. Reading is excluded.

<details>
<summary><strong>AdditiveNoise</strong></summary>

Add independent noise to each voxel. Supports uniform, Gaussian, Laplace and centered beta distributions; this example uses Laplace noise.

```python
A.AdditiveNoise(scale=0.05, distribution='laplace', mean=0)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 15.608 ms | 1.313 ms | 0.240 ms | 11.9× | 65.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![AdditiveNoise: original and transformed Z slices](docs/transforms/previews/AdditiveNoise-z.webp)](docs/transforms/previews/AdditiveNoise-z.webp) | [![AdditiveNoise: original and transformed Y slices](docs/transforms/previews/AdditiveNoise-y.webp)](docs/transforms/previews/AdditiveNoise-y.webp) | [![AdditiveNoise: original and transformed X slices](docs/transforms/previews/AdditiveNoise-x.webp)](docs/transforms/previews/AdditiveNoise-x.webp) |

</details>

<details>
<summary><strong>Affine</strong></summary>

Rotate, translate and scale the volume, applying the same geometry to images and labels.

```python
A.Affine(degrees=5, translate=2, scale=0.95)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 68.183 ms | 6.039 ms | 0.608 ms | 11.3× | 112.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Affine: original and transformed Z slices](docs/transforms/previews/Affine-z.webp)](docs/transforms/previews/Affine-z.webp) | [![Affine: original and transformed Y slices](docs/transforms/previews/Affine-y.webp)](docs/transforms/previews/Affine-y.webp) | [![Affine: original and transformed X slices](docs/transforms/previews/Affine-x.webp)](docs/transforms/previews/Affine-x.webp) |

</details>

<details>
<summary><strong>BiasField</strong></summary>

Multiply intensities by a smooth 3D gain field generated from a coarse Gaussian random lattice.

```python
A.BiasField(std=0.3, scale=0.025)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 11.655 ms | 1.442 ms | 0.323 ms | 8.1× | 36.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BiasField: original and transformed Z slices](docs/transforms/previews/BiasField-z.webp)](docs/transforms/previews/BiasField-z.webp) | [![BiasField: original and transformed Y slices](docs/transforms/previews/BiasField-y.webp)](docs/transforms/previews/BiasField-y.webp) | [![BiasField: original and transformed X slices](docs/transforms/previews/BiasField-x.webp)](docs/transforms/previews/BiasField-x.webp) |

</details>

<details>
<summary><strong>BlankRectangle</strong></summary>

Fill randomly placed cuboids with a chosen value, simulating missing regions.

```python
A.BlankRectangle(size=16, value=0, count=3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0.797 ms | 1.205 ms | 0.111 ms | 0.7× | 7.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BlankRectangle: original and transformed Z slices](docs/transforms/previews/BlankRectangle-z.webp)](docs/transforms/previews/BlankRectangle-z.webp) | [![BlankRectangle: original and transformed Y slices](docs/transforms/previews/BlankRectangle-y.webp)](docs/transforms/previews/BlankRectangle-y.webp) | [![BlankRectangle: original and transformed X slices](docs/transforms/previews/BlankRectangle-x.webp)](docs/transforms/previews/BlankRectangle-x.webp) |

</details>

<details>
<summary><strong>BrightnessAdd</strong></summary>

Add a constant offset to every voxel in the selected image channels.

```python
A.BrightnessAdd(shift=0.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.650 ms | 1.329 ms | 0.158 ms | 1.2× | 10.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BrightnessAdd: original and transformed Z slices](docs/transforms/previews/BrightnessAdd-z.webp)](docs/transforms/previews/BrightnessAdd-z.webp) | [![BrightnessAdd: original and transformed Y slices](docs/transforms/previews/BrightnessAdd-y.webp)](docs/transforms/previews/BrightnessAdd-y.webp) | [![BrightnessAdd: original and transformed X slices](docs/transforms/previews/BrightnessAdd-x.webp)](docs/transforms/previews/BrightnessAdd-x.webp) |

</details>

<details>
<summary><strong>BrightnessGradient</strong></summary>

Add a localized Gaussian brightness field. This example fixes its center at the middle of the volume; `loc=(0, 1)` samples a random position. The field is mean-centered by default.

```python
A.BrightnessGradient(scale=12, loc=0.5, strength=0.3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.587 ms | 1.380 ms | 0.184 ms | 1.9× | 14.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BrightnessGradient: original and transformed Z slices](docs/transforms/previews/BrightnessGradient-z.webp)](docs/transforms/previews/BrightnessGradient-z.webp) | [![BrightnessGradient: original and transformed Y slices](docs/transforms/previews/BrightnessGradient-y.webp)](docs/transforms/previews/BrightnessGradient-y.webp) | [![BrightnessGradient: original and transformed X slices](docs/transforms/previews/BrightnessGradient-x.webp)](docs/transforms/previews/BrightnessGradient-x.webp) |

</details>

<details>
<summary><strong>BrightnessMultiply</strong></summary>

Multiply every voxel in the selected image channels by a brightness factor.

```python
A.BrightnessMultiply(factor=1.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.632 ms | 1.359 ms | 0.119 ms | 1.2× | 13.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BrightnessMultiply: original and transformed Z slices](docs/transforms/previews/BrightnessMultiply-z.webp)](docs/transforms/previews/BrightnessMultiply-z.webp) | [![BrightnessMultiply: original and transformed Y slices](docs/transforms/previews/BrightnessMultiply-y.webp)](docs/transforms/previews/BrightnessMultiply-y.webp) | [![BrightnessMultiply: original and transformed X slices](docs/transforms/previews/BrightnessMultiply-x.webp)](docs/transforms/previews/BrightnessMultiply-x.webp) |

</details>

<details>
<summary><strong>ChannelMisalignment</strong></summary>

Independently shift, rotate and scale selected image channels to simulate registration errors. This example moves the second of two identical CT channels.

```python
A.ChannelMisalignment(degrees=5, translate=2, scale=1, channels=(1,))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 46.855 ms | 2.591 ms | 0.252 ms | 18.1× | 186.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ChannelMisalignment: original and transformed Z slices](docs/transforms/previews/ChannelMisalignment-z.webp)](docs/transforms/previews/ChannelMisalignment-z.webp) | [![ChannelMisalignment: original and transformed Y slices](docs/transforms/previews/ChannelMisalignment-y.webp)](docs/transforms/previews/ChannelMisalignment-y.webp) | [![ChannelMisalignment: original and transformed X slices](docs/transforms/previews/ChannelMisalignment-x.webp)](docs/transforms/previews/ChannelMisalignment-x.webp) |

Two input channels (second is a copy of the CT); only channel 1 moves. Both channels are included in timing.

</details>

<details>
<summary><strong>CLAHE</strong></summary>

Equalize local histograms in 3D tiles, limit their contrast, and interpolate between neighboring tiles to avoid seams.

```python
A.CLAHE(clip_limit=2, tile_shape=(16, 16, 16), bins=256)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 52.663 ms | 1.712 ms | 0.549 ms | 30.8× | 96.0× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![CLAHE: original and transformed Z slices](docs/transforms/previews/CLAHE-z.webp)](docs/transforms/previews/CLAHE-z.webp) | [![CLAHE: original and transformed Y slices](docs/transforms/previews/CLAHE-y.webp)](docs/transforms/previews/CLAHE-y.webp) | [![CLAHE: original and transformed X slices](docs/transforms/previews/CLAHE-x.webp)](docs/transforms/previews/CLAHE-x.webp) |

</details>

<details>
<summary><strong>ClipOutliers</strong></summary>

Clip intensities at selected percentiles. This example then restores the original standard deviation.

```python
A.ClipOutliers(lower=1, upper=99, p_retain_std=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 21.853 ms | 1.757 ms | 0.532 ms | 12.4× | 41.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ClipOutliers: original and transformed Z slices](docs/transforms/previews/ClipOutliers-z.webp)](docs/transforms/previews/ClipOutliers-z.webp) | [![ClipOutliers: original and transformed Y slices](docs/transforms/previews/ClipOutliers-y.webp)](docs/transforms/previews/ClipOutliers-y.webp) | [![ClipOutliers: original and transformed X slices](docs/transforms/previews/ClipOutliers-x.webp)](docs/transforms/previews/ClipOutliers-x.webp) |

</details>

<details>
<summary><strong>Contrast</strong></summary>

Scale intensity differences from the channel mean. The default range-preservation option clips results to the original range.

```python
A.Contrast(factor=1.2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 5.635 ms | 1.297 ms | 0.133 ms | 4.3× | 42.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Contrast: original and transformed Z slices](docs/transforms/previews/Contrast-z.webp)](docs/transforms/previews/Contrast-z.webp) | [![Contrast: original and transformed Y slices](docs/transforms/previews/Contrast-y.webp)](docs/transforms/previews/Contrast-y.webp) | [![Contrast: original and transformed X slices](docs/transforms/previews/Contrast-x.webp)](docs/transforms/previews/Contrast-x.webp) |

</details>

<details>
<summary><strong>CropPad</strong></summary>

Crop or pad all aligned targets to a requested ZYX shape. This example takes a centered 112³ crop from the 128³ input.

```python
A.CropPad(shape=(112, 112, 112))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0.911 ms | 1.932 ms | 0.369 ms | 0.5× | 2.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![CropPad: original and transformed Z slices](docs/transforms/previews/CropPad-z.webp)](docs/transforms/previews/CropPad-z.webp) | [![CropPad: original and transformed Y slices](docs/transforms/previews/CropPad-y.webp)](docs/transforms/previews/CropPad-y.webp) | [![CropPad: original and transformed X slices](docs/transforms/previews/CropPad-x.webp)](docs/transforms/previews/CropPad-x.webp) |

</details>

<details>
<summary><strong>Defocus</strong></summary>

Average intensities within a spherical 3D neighborhood to simulate out-of-focus acquisition.

```python
A.Defocus(radius=2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 9.619 ms | 1.944 ms | 0.776 ms | 4.9× | 12.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Defocus: original and transformed Z slices](docs/transforms/previews/Defocus-z.webp)](docs/transforms/previews/Defocus-z.webp) | [![Defocus: original and transformed Y slices](docs/transforms/previews/Defocus-y.webp)](docs/transforms/previews/Defocus-y.webp) | [![Defocus: original and transformed X slices](docs/transforms/previews/Defocus-x.webp)](docs/transforms/previews/Defocus-x.webp) |

</details>

<details>
<summary><strong>Elastic</strong></summary>

Warp the volume with a smooth random 3D B-spline displacement field, keeping images and labels aligned. The previews use interior slices so boundary padding does not dominate the view.

```python
A.Elastic(sigma=32, magnitude=8, border=True)
```

CPU and CUDA construct and store elastic fields in float32. Ordinary crop
geometry also uses float32, with wider arithmetic where coordinate range or
numerical conditioning requires it. Output dtypes are preserved. Small
coordinate differences can select a neighboring label at a nearest-neighbor
boundary. New replay plans use float32 fields; old plans must be regenerated.
See the [precision policy](docs/precision.md) and
[transform benchmarks](benchmarks/README.md#transform-benchmarks).

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 85.995 ms | 2.536 ms | 0.696 ms | 33.9× | 123.6× |

[▶ Watch Elastic animation (WebM)](docs/transforms/animations/Elastic.webm)

Strong cubic B-spline deformation: original → deformed → original. Fixed Z/Y/X slices, seed 125, achieved RMS 5.36 voxels.

[![Elastic animation poster](docs/transforms/animations/Elastic-poster.webp)](docs/transforms/animations/Elastic.webm)

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Elastic: original and transformed Z slices](docs/transforms/previews/Elastic-z.webp)](docs/transforms/previews/Elastic-z.webp) | [![Elastic: original and transformed Y slices](docs/transforms/previews/Elastic-y.webp)](docs/transforms/previews/Elastic-y.webp) | [![Elastic: original and transformed X slices](docs/transforms/previews/Elastic-x.webp)](docs/transforms/previews/Elastic-x.webp) |

Strong cubic B-spline example: 64-voxel knot spacing, requested RMS displacement 8 voxels, with border clamping.

</details>

<details>
<summary><strong>Equalize</strong></summary>

Equalize the histogram across the entire volume while retaining the channel intensity range.

```python
A.Equalize(bins=256)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 19.669 ms | 1.588 ms | 0.369 ms | 12.4× | 53.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Equalize: original and transformed Z slices](docs/transforms/previews/Equalize-z.webp)](docs/transforms/previews/Equalize-z.webp) | [![Equalize: original and transformed Y slices](docs/transforms/previews/Equalize-y.webp)](docs/transforms/previews/Equalize-y.webp) | [![Equalize: original and transformed X slices](docs/transforms/previews/Equalize-x.webp)](docs/transforms/previews/Equalize-x.webp) |

</details>

<details>
<summary><strong>FDA</strong></summary>

Blend low-frequency amplitudes from a reference volume into the 3D Fourier spectrum while preserving the source phase.

```python
A.FDA(reference=reference, blend=1, fraction=0.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 613.754 ms | 3.463 ms | 2.119 ms | 177.2× | 289.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![FDA: original and transformed Z slices](docs/transforms/previews/FDA-z.webp)](docs/transforms/previews/FDA-z.webp) | [![FDA: original and transformed Y slices](docs/transforms/previews/FDA-y.webp)](docs/transforms/previews/FDA-y.webp) | [![FDA: original and transformed X slices](docs/transforms/previews/FDA-x.webp)](docs/transforms/previews/FDA-x.webp) |

</details>

<details>
<summary><strong>Flip</strong></summary>

Mirror the volume along selected axes. This example always flips X.

```python
A.Flip(axes='x', p_axis=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.425 ms | 1.869 ms | 0.323 ms | 0.8× | 4.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Flip: original and transformed Z slices](docs/transforms/previews/Flip-z.webp)](docs/transforms/previews/Flip-z.webp) | [![Flip: original and transformed Y slices](docs/transforms/previews/Flip-y.webp)](docs/transforms/previews/Flip-y.webp) | [![Flip: original and transformed X slices](docs/transforms/previews/Flip-x.webp)](docs/transforms/previews/Flip-x.webp) |

</details>

<details>
<summary><strong>ForegroundCrop</strong></summary>

Choose a random crop containing a sampled foreground voxel. All aligned targets follow the crop; empty masks fall back to random cropping.

```python
A.ForegroundCrop(shape=(112, 112, 112))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.387 ms | 2.059 ms | 0.531 ms | 1.2× | 4.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ForegroundCrop: original and transformed Z slices](docs/transforms/previews/ForegroundCrop-z.webp)](docs/transforms/previews/ForegroundCrop-z.webp) | [![ForegroundCrop: original and transformed Y slices](docs/transforms/previews/ForegroundCrop-y.webp)](docs/transforms/previews/ForegroundCrop-y.webp) | [![ForegroundCrop: original and transformed X slices](docs/transforms/previews/ForegroundCrop-x.webp)](docs/transforms/previews/ForegroundCrop-x.webp) |

</details>

<details>
<summary><strong>ForegroundDropout</strong></summary>

Place cuboids around sampled foreground voxels and erase only foreground inside those cuboids.

```python
A.ForegroundDropout(size=(16, 16, 16), count=3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 6.501 ms | 1.885 ms | 0.452 ms | 3.4× | 14.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ForegroundDropout: original and transformed Z slices](docs/transforms/previews/ForegroundDropout-z.webp)](docs/transforms/previews/ForegroundDropout-z.webp) | [![ForegroundDropout: original and transformed Y slices](docs/transforms/previews/ForegroundDropout-y.webp)](docs/transforms/previews/ForegroundDropout-y.webp) | [![ForegroundDropout: original and transformed X slices](docs/transforms/previews/ForegroundDropout-x.webp)](docs/transforms/previews/ForegroundDropout-x.webp) |

</details>

<details>
<summary><strong>Gamma</strong></summary>

Apply a power-law intensity curve over the channel range, changing the relative brightness of dark and light structures.

```python
A.Gamma(gamma=0.8)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8.660 ms | 1.420 ms | 0.133 ms | 6.1× | 65.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Gamma: original and transformed Z slices](docs/transforms/previews/Gamma-z.webp)](docs/transforms/previews/Gamma-z.webp) | [![Gamma: original and transformed Y slices](docs/transforms/previews/Gamma-y.webp)](docs/transforms/previews/Gamma-y.webp) | [![Gamma: original and transformed X slices](docs/transforms/previews/Gamma-x.webp)](docs/transforms/previews/Gamma-x.webp) |

</details>

<details>
<summary><strong>GaussianBlur</strong></summary>

Smooth the volume with a separable 3D Gaussian filter.

`full_width=False, truncate=4` uses radius `floor(4*sigma + 0.5)`, matching
SciPy Gaussian filter support. The default `full_width=True` interprets
`truncate` as the full width. CPU and CUDA use the same convention.

```python
A.GaussianBlur(sigma=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 6.254 ms | 1.412 ms | 0.229 ms | 4.4× | 27.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GaussianBlur: original and transformed Z slices](docs/transforms/previews/GaussianBlur-z.webp)](docs/transforms/previews/GaussianBlur-z.webp) | [![GaussianBlur: original and transformed Y slices](docs/transforms/previews/GaussianBlur-y.webp)](docs/transforms/previews/GaussianBlur-y.webp) | [![GaussianBlur: original and transformed X slices](docs/transforms/previews/GaussianBlur-x.webp)](docs/transforms/previews/GaussianBlur-x.webp) |

</details>

<details>
<summary><strong>GaussianNoise</strong></summary>

Add independent zero-mean Gaussian noise with the specified standard deviation.

```python
A.GaussianNoise(std=0.05)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 9.104 ms | 1.255 ms | 0.125 ms | 7.3× | 73.0× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GaussianNoise: original and transformed Z slices](docs/transforms/previews/GaussianNoise-z.webp)](docs/transforms/previews/GaussianNoise-z.webp) | [![GaussianNoise: original and transformed Y slices](docs/transforms/previews/GaussianNoise-y.webp)](docs/transforms/previews/GaussianNoise-y.webp) | [![GaussianNoise: original and transformed X slices](docs/transforms/previews/GaussianNoise-x.webp)](docs/transforms/previews/GaussianNoise-x.webp) |

</details>

<details>
<summary><strong>GibbsRinging</strong></summary>

Truncate high frequencies in the 3D Fourier spectrum, creating ringing around sharp boundaries.

```python
A.GibbsRinging(cutoff=0.7)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 425.044 ms | 1.617 ms | 0.485 ms | 262.9× | 875.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GibbsRinging: original and transformed Z slices](docs/transforms/previews/GibbsRinging-z.webp)](docs/transforms/previews/GibbsRinging-z.webp) | [![GibbsRinging: original and transformed Y slices](docs/transforms/previews/GibbsRinging-y.webp)](docs/transforms/previews/GibbsRinging-y.webp) | [![GibbsRinging: original and transformed X slices](docs/transforms/previews/GibbsRinging-x.webp)](docs/transforms/previews/GibbsRinging-x.webp) |

</details>

<details>
<summary><strong>GridDropout</strong></summary>

Erase a periodic grid of cuboid holes with a randomly sampled offset.

```python
A.GridDropout(unit_size=(16, 16, 16), ratio=0.4)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 39.059 ms | 1.297 ms | 0.146 ms | 30.1× | 267.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GridDropout: original and transformed Z slices](docs/transforms/previews/GridDropout-z.webp)](docs/transforms/previews/GridDropout-z.webp) | [![GridDropout: original and transformed Y slices](docs/transforms/previews/GridDropout-y.webp)](docs/transforms/previews/GridDropout-y.webp) | [![GridDropout: original and transformed X slices](docs/transforms/previews/GridDropout-x.webp)](docs/transforms/previews/GridDropout-x.webp) |

</details>

<details>
<summary><strong>HistogramMatching</strong></summary>

Match the volume intensity histogram to a reference volume, with optional blending.

```python
A.HistogramMatching(reference=reference, blend=1, bins=256)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 86.937 ms | 3.000 ms | 1.899 ms | 29.0× | 45.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![HistogramMatching: original and transformed Z slices](docs/transforms/previews/HistogramMatching-z.webp)](docs/transforms/previews/HistogramMatching-z.webp) | [![HistogramMatching: original and transformed Y slices](docs/transforms/previews/HistogramMatching-y.webp)](docs/transforms/previews/HistogramMatching-y.webp) | [![HistogramMatching: original and transformed X slices](docs/transforms/previews/HistogramMatching-x.webp)](docs/transforms/previews/HistogramMatching-x.webp) |

</details>

<details>
<summary><strong>Illumination</strong></summary>

Apply a smooth multiplicative illumination field. Supports directional ramps, corner falloff and Gaussian illumination; this example uses a diagonal ramp.

```python
A.Illumination(strength=0.3, mode='linear', direction=(1, 1, 1))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 28.240 ms | 1.355 ms | 0.201 ms | 20.8× | 140.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Illumination: original and transformed Z slices](docs/transforms/previews/Illumination-z.webp)](docs/transforms/previews/Illumination-z.webp) | [![Illumination: original and transformed Y slices](docs/transforms/previews/Illumination-y.webp)](docs/transforms/previews/Illumination-y.webp) | [![Illumination: original and transformed X slices](docs/transforms/previews/Illumination-x.webp)](docs/transforms/previews/Illumination-x.webp) |

</details>

<details>
<summary><strong>Invert</strong></summary>

Reflect intensities around the channel mean, reversing bright and dark structures.

```python
A.Invert()
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.582 ms | 1.389 ms | 0.178 ms | 1.9× | 14.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Invert: original and transformed Z slices](docs/transforms/previews/Invert-z.webp)](docs/transforms/previews/Invert-z.webp) | [![Invert: original and transformed Y slices](docs/transforms/previews/Invert-y.webp)](docs/transforms/previews/Invert-y.webp) | [![Invert: original and transformed X slices](docs/transforms/previews/Invert-x.webp)](docs/transforms/previews/Invert-x.webp) |

</details>

<details>
<summary><strong>LocalContrast</strong></summary>

Change contrast around a Gaussian-weighted local mean inside a smooth region.

```python
A.LocalContrast(scale=12, loc=0.5, factor=1.2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4.167 ms | 1.297 ms | 0.171 ms | 3.2× | 24.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LocalContrast: original and transformed Z slices](docs/transforms/previews/LocalContrast-z.webp)](docs/transforms/previews/LocalContrast-z.webp) | [![LocalContrast: original and transformed Y slices](docs/transforms/previews/LocalContrast-y.webp)](docs/transforms/previews/LocalContrast-y.webp) | [![LocalContrast: original and transformed X slices](docs/transforms/previews/LocalContrast-x.webp)](docs/transforms/previews/LocalContrast-x.webp) |

</details>

<details>
<summary><strong>LocalGamma</strong></summary>

Blend a gamma-adjusted image into a localized Gaussian region.

```python
A.LocalGamma(scale=12, loc=0.5, gamma=0.8)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 12.362 ms | 1.308 ms | 0.161 ms | 9.5× | 76.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LocalGamma: original and transformed Z slices](docs/transforms/previews/LocalGamma-z.webp)](docs/transforms/previews/LocalGamma-z.webp) | [![LocalGamma: original and transformed Y slices](docs/transforms/previews/LocalGamma-y.webp)](docs/transforms/previews/LocalGamma-y.webp) | [![LocalGamma: original and transformed X slices](docs/transforms/previews/LocalGamma-x.webp)](docs/transforms/previews/LocalGamma-x.webp) |

</details>

<details>
<summary><strong>LocalSmoothing</strong></summary>

Blend Gaussian smoothing into a localized region, leaving distant voxels largely unchanged.

```python
A.LocalSmoothing(scale=12, loc=0.5, strength=0.7, sigma=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 7.336 ms | 1.695 ms | 0.414 ms | 4.3× | 17.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LocalSmoothing: original and transformed Z slices](docs/transforms/previews/LocalSmoothing-z.webp)](docs/transforms/previews/LocalSmoothing-z.webp) | [![LocalSmoothing: original and transformed Y slices](docs/transforms/previews/LocalSmoothing-y.webp)](docs/transforms/previews/LocalSmoothing-y.webp) | [![LocalSmoothing: original and transformed X slices](docs/transforms/previews/LocalSmoothing-x.webp)](docs/transforms/previews/LocalSmoothing-x.webp) |

</details>

<details>
<summary><strong>LowResolution</strong></summary>

Downsample and upsample the image to simulate reduced spatial resolution.

```python
A.LowResolution(scale=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 6.298 ms | 1.284 ms | 0.127 ms | 4.9× | 49.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LowResolution: original and transformed Z slices](docs/transforms/previews/LowResolution-z.webp)](docs/transforms/previews/LowResolution-z.webp) | [![LowResolution: original and transformed Y slices](docs/transforms/previews/LowResolution-y.webp)](docs/transforms/previews/LowResolution-y.webp) | [![LowResolution: original and transformed X slices](docs/transforms/previews/LowResolution-x.webp)](docs/transforms/previews/LowResolution-x.webp) |

</details>

<details>
<summary><strong>MedianFilter</strong></summary>

Replace each voxel with the median of its 3D neighborhood, suppressing isolated noise while retaining edges.

```python
A.MedianFilter(size=3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 15.783 ms | 1.310 ms | 0.149 ms | 12.1× | 105.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![MedianFilter: original and transformed Z slices](docs/transforms/previews/MedianFilter-z.webp)](docs/transforms/previews/MedianFilter-z.webp) | [![MedianFilter: original and transformed Y slices](docs/transforms/previews/MedianFilter-y.webp)](docs/transforms/previews/MedianFilter-y.webp) | [![MedianFilter: original and transformed X slices](docs/transforms/previews/MedianFilter-x.webp)](docs/transforms/previews/MedianFilter-x.webp) |

</details>

<details>
<summary><strong>Morphology</strong></summary>

Apply a morphological operation to labels. This example closes small gaps using dilation followed by erosion.

```python
A.Morphology(operation='close', radius=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 12.452 ms | 0.732 ms | 0.312 ms | 17.0× | 39.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Morphology: original and transformed Z slices](docs/transforms/previews/Morphology-z.webp)](docs/transforms/previews/Morphology-z.webp) | [![Morphology: original and transformed Y slices](docs/transforms/previews/Morphology-y.webp)](docs/transforms/previews/Morphology-y.webp) | [![Morphology: original and transformed X slices](docs/transforms/previews/Morphology-x.webp)](docs/transforms/previews/Morphology-x.webp) |

</details>

<details>
<summary><strong>MotionBlur</strong></summary>

Average samples along a 3D line to simulate motion during acquisition.

```python
A.MotionBlur(length=5, direction=(1, 1, 1), samples=9)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 240.809 ms | 2.594 ms | 1.356 ms | 92.8× | 177.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![MotionBlur: original and transformed Z slices](docs/transforms/previews/MotionBlur-z.webp)](docs/transforms/previews/MotionBlur-z.webp) | [![MotionBlur: original and transformed Y slices](docs/transforms/previews/MotionBlur-y.webp)](docs/transforms/previews/MotionBlur-y.webp) | [![MotionBlur: original and transformed X slices](docs/transforms/previews/MotionBlur-x.webp)](docs/transforms/previews/MotionBlur-x.webp) |

</details>

<details>
<summary><strong>MultiplicativeNoise</strong></summary>

Multiply each voxel by an independently sampled uniform random gain.

```python
A.MultiplicativeNoise(low=0.8, high=1.2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8.368 ms | 1.310 ms | 0.149 ms | 6.4× | 56.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![MultiplicativeNoise: original and transformed Z slices](docs/transforms/previews/MultiplicativeNoise-z.webp)](docs/transforms/previews/MultiplicativeNoise-z.webp) | [![MultiplicativeNoise: original and transformed Y slices](docs/transforms/previews/MultiplicativeNoise-y.webp)](docs/transforms/previews/MultiplicativeNoise-y.webp) | [![MultiplicativeNoise: original and transformed X slices](docs/transforms/previews/MultiplicativeNoise-x.webp)](docs/transforms/previews/MultiplicativeNoise-x.webp) |

</details>

<details>
<summary><strong>Normalize</strong></summary>

Normalize intensities per channel. This example uses z-score normalization of nonzero voxels, keeping zeros unchanged; its display window includes negative values.

```python
A.Normalize(mode='zscore')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 6.726 ms | 1.347 ms | 0.245 ms | 5.0× | 27.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Normalize: original and transformed Z slices](docs/transforms/previews/Normalize-z.webp)](docs/transforms/previews/Normalize-z.webp) | [![Normalize: original and transformed Y slices](docs/transforms/previews/Normalize-y.webp)](docs/transforms/previews/Normalize-y.webp) | [![Normalize: original and transformed X slices](docs/transforms/previews/Normalize-x.webp)](docs/transforms/previews/Normalize-x.webp) |

</details>

<details>
<summary><strong>ObjectDropout</strong></summary>

Randomly remove whole connected foreground objects, clearing their image voxels and labels by default.

```python
A.ObjectDropout(drop_probability=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 13.619 ms | 2.345 ms | 0.509 ms | 5.8× | 26.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ObjectDropout: original and transformed Z slices](docs/transforms/previews/ObjectDropout-z.webp)](docs/transforms/previews/ObjectDropout-z.webp) | [![ObjectDropout: original and transformed Y slices](docs/transforms/previews/ObjectDropout-y.webp)](docs/transforms/previews/ObjectDropout-y.webp) | [![ObjectDropout: original and transformed X slices](docs/transforms/previews/ObjectDropout-x.webp)](docs/transforms/previews/ObjectDropout-x.webp) |

</details>

<details>
<summary><strong>OrientedBlur</strong></summary>

Smooth along a specified 3D direction and two perpendicular directions with separate blur widths. This example is aligned with Z.

```python
A.OrientedBlur(sigma_along=2, sigma_across=0.5, axis=(1, 0, 0))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 7.649 ms | 1.874 ms | 0.719 ms | 4.1× | 10.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![OrientedBlur: original and transformed Z slices](docs/transforms/previews/OrientedBlur-z.webp)](docs/transforms/previews/OrientedBlur-z.webp) | [![OrientedBlur: original and transformed Y slices](docs/transforms/previews/OrientedBlur-y.webp)](docs/transforms/previews/OrientedBlur-y.webp) | [![OrientedBlur: original and transformed X slices](docs/transforms/previews/OrientedBlur-x.webp)](docs/transforms/previews/OrientedBlur-x.webp) |

</details>

<details>
<summary><strong>PixelDistributionAdaptation</strong></summary>

Transfer reference intensity statistics. The standard mode shown here blends the reference mean and standard deviation; minmax mode transfers the range.

```python
A.PixelDistributionAdaptation(reference=reference, blend=1, mode='standard')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 18.454 ms | 2.614 ms | 1.448 ms | 7.1× | 12.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![PixelDistributionAdaptation: original and transformed Z slices](docs/transforms/previews/PixelDistributionAdaptation-z.webp)](docs/transforms/previews/PixelDistributionAdaptation-z.webp) | [![PixelDistributionAdaptation: original and transformed Y slices](docs/transforms/previews/PixelDistributionAdaptation-y.webp)](docs/transforms/previews/PixelDistributionAdaptation-y.webp) | [![PixelDistributionAdaptation: original and transformed X slices](docs/transforms/previews/PixelDistributionAdaptation-x.webp)](docs/transforms/previews/PixelDistributionAdaptation-x.webp) |

</details>

<details>
<summary><strong>PlasmaBrightnessContrast</strong></summary>

Use a smooth multiscale 3D random field to vary brightness and contrast throughout the volume.

```python
A.PlasmaBrightnessContrast(brightness=0.2, contrast=0.3, scale=0.025, persistence=0.5, octaves=5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 31.285 ms | 1.831 ms | 0.677 ms | 17.1× | 46.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![PlasmaBrightnessContrast: original and transformed Z slices](docs/transforms/previews/PlasmaBrightnessContrast-z.webp)](docs/transforms/previews/PlasmaBrightnessContrast-z.webp) | [![PlasmaBrightnessContrast: original and transformed Y slices](docs/transforms/previews/PlasmaBrightnessContrast-y.webp)](docs/transforms/previews/PlasmaBrightnessContrast-y.webp) | [![PlasmaBrightnessContrast: original and transformed X slices](docs/transforms/previews/PlasmaBrightnessContrast-x.webp)](docs/transforms/previews/PlasmaBrightnessContrast-x.webp) |

</details>

<details>
<summary><strong>PlasmaShadow</strong></summary>

Darken the volume with a smooth multiscale 3D random shadow field.

```python
A.PlasmaShadow(strength=0.5, scale=0.025, persistence=0.5, octaves=5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 25.357 ms | 1.690 ms | 0.591 ms | 15.0× | 42.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![PlasmaShadow: original and transformed Z slices](docs/transforms/previews/PlasmaShadow-z.webp)](docs/transforms/previews/PlasmaShadow-z.webp) | [![PlasmaShadow: original and transformed Y slices](docs/transforms/previews/PlasmaShadow-y.webp)](docs/transforms/previews/PlasmaShadow-y.webp) | [![PlasmaShadow: original and transformed X slices](docs/transforms/previews/PlasmaShadow-x.webp)](docs/transforms/previews/PlasmaShadow-x.webp) |

</details>

<details>
<summary><strong>RandomBinaryOperator</strong></summary>

Apply a selected binary morphological operation to image channels. This example dilates a thresholded foreground volume.

```python
A.RandomBinaryOperator(channels=(0,), radius=1, operations=('dilate',))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10.243 ms | 1.404 ms | 0.241 ms | 7.3× | 42.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RandomBinaryOperator: original and transformed Z slices](docs/transforms/previews/RandomBinaryOperator-z.webp)](docs/transforms/previews/RandomBinaryOperator-z.webp) | [![RandomBinaryOperator: original and transformed Y slices](docs/transforms/previews/RandomBinaryOperator-y.webp)](docs/transforms/previews/RandomBinaryOperator-y.webp) | [![RandomBinaryOperator: original and transformed X slices](docs/transforms/previews/RandomBinaryOperator-x.webp)](docs/transforms/previews/RandomBinaryOperator-x.webp) |

</details>

<details>
<summary><strong>RandomToneCurve</strong></summary>

Apply a random monotone nonlinear intensity curve while preserving the channel range endpoints.

```python
A.RandomToneCurve(scale=0.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 13.061 ms | 1.232 ms | 0.194 ms | 10.6× | 67.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RandomToneCurve: original and transformed Z slices](docs/transforms/previews/RandomToneCurve-z.webp)](docs/transforms/previews/RandomToneCurve-z.webp) | [![RandomToneCurve: original and transformed Y slices](docs/transforms/previews/RandomToneCurve-y.webp)](docs/transforms/previews/RandomToneCurve-y.webp) | [![RandomToneCurve: original and transformed X slices](docs/transforms/previews/RandomToneCurve-x.webp)](docs/transforms/previews/RandomToneCurve-x.webp) |

</details>

<details>
<summary><strong>RemoveComponents</strong></summary>

Randomly remove connected foreground components below a size threshold. This example modifies the label mask.

```python
A.RemoveComponents(on='mask', max_fraction=0.25, p_fill=0)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 20.792 ms | 1.335 ms | 0.970 ms | 15.6× | 21.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RemoveComponents: original and transformed Z slices](docs/transforms/previews/RemoveComponents-z.webp)](docs/transforms/previews/RemoveComponents-z.webp) | [![RemoveComponents: original and transformed Y slices](docs/transforms/previews/RemoveComponents-y.webp)](docs/transforms/previews/RemoveComponents-y.webp) | [![RemoveComponents: original and transformed X slices](docs/transforms/previews/RemoveComponents-x.webp)](docs/transforms/previews/RemoveComponents-x.webp) |

</details>

<details>
<summary><strong>RicianNoise</strong></summary>

Combine the image with two independent Gaussian noise fields to simulate magnitude-image Rician noise.

```python
A.RicianNoise(std=0.05)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 21.196 ms | 1.422 ms | 0.185 ms | 14.9× | 114.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RicianNoise: original and transformed Z slices](docs/transforms/previews/RicianNoise-z.webp)](docs/transforms/previews/RicianNoise-z.webp) | [![RicianNoise: original and transformed Y slices](docs/transforms/previews/RicianNoise-y.webp)](docs/transforms/previews/RicianNoise-y.webp) | [![RicianNoise: original and transformed X slices](docs/transforms/previews/RicianNoise-x.webp)](docs/transforms/previews/RicianNoise-x.webp) |

</details>

<details>
<summary><strong>RingingOvershoot</strong></summary>

Apply a separable windowed-sinc filter in 3D, retaining edge overshoot and ringing.

```python
A.RingingOvershoot(cutoff=0.7, radius=4)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 7.994 ms | 5.418 ms | 4.292 ms | 1.5× | 1.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RingingOvershoot: original and transformed Z slices](docs/transforms/previews/RingingOvershoot-z.webp)](docs/transforms/previews/RingingOvershoot-z.webp) | [![RingingOvershoot: original and transformed Y slices](docs/transforms/previews/RingingOvershoot-y.webp)](docs/transforms/previews/RingingOvershoot-y.webp) | [![RingingOvershoot: original and transformed X slices](docs/transforms/previews/RingingOvershoot-x.webp)](docs/transforms/previews/RingingOvershoot-x.webp) |

</details>

<details>
<summary><strong>Rotate90</strong></summary>

Rotate aligned targets by multiples of 90 degrees. This example rotates the YX plane by one quarter turn.

```python
A.Rotate90(planes=('yx',), turns=(1,))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.052 ms | 2.296 ms | 0.587 ms | 0.9× | 3.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Rotate90: original and transformed Z slices](docs/transforms/previews/Rotate90-z.webp)](docs/transforms/previews/Rotate90-z.webp) | [![Rotate90: original and transformed Y slices](docs/transforms/previews/Rotate90-y.webp)](docs/transforms/previews/Rotate90-y.webp) | [![Rotate90: original and transformed X slices](docs/transforms/previews/Rotate90-x.webp)](docs/transforms/previews/Rotate90-x.webp) |

</details>

<details>
<summary><strong>SaltAndPepper</strong></summary>

Replace selected voxels with the channel minimum or maximum, producing dark and bright impulse noise.

```python
A.SaltAndPepper(amount=0.02, salt_fraction=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 9.965 ms | 1.250 ms | 0.206 ms | 8.0× | 48.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![SaltAndPepper: original and transformed Z slices](docs/transforms/previews/SaltAndPepper-z.webp)](docs/transforms/previews/SaltAndPepper-z.webp) | [![SaltAndPepper: original and transformed Y slices](docs/transforms/previews/SaltAndPepper-y.webp)](docs/transforms/previews/SaltAndPepper-y.webp) | [![SaltAndPepper: original and transformed X slices](docs/transforms/previews/SaltAndPepper-x.webp)](docs/transforms/previews/SaltAndPepper-x.webp) |

</details>

<details>
<summary><strong>Sharpen</strong></summary>

Enhance local edges with a discrete Laplacian sharpening filter. This example keeps overshoots instead of clipping them.

```python
A.Sharpen(strength=0.25, p_clamp=0)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4.557 ms | 1.449 ms | 0.174 ms | 3.1× | 26.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Sharpen: original and transformed Z slices](docs/transforms/previews/Sharpen-z.webp)](docs/transforms/previews/Sharpen-z.webp) | [![Sharpen: original and transformed Y slices](docs/transforms/previews/Sharpen-y.webp)](docs/transforms/previews/Sharpen-y.webp) | [![Sharpen: original and transformed X slices](docs/transforms/previews/Sharpen-x.webp)](docs/transforms/previews/Sharpen-x.webp) |

</details>

<details>
<summary><strong>SheetCompression</strong></summary>

Use the CT intensity profile to compress sheet-like structures along a selected axis while transforming aligned targets together.

```python
A.SheetCompression(strength=0.2, intensity_sigma=1, smoothing=2, axes='z')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 41.401 ms | 2.545 ms | 0.887 ms | 16.3× | 46.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![SheetCompression: original and transformed Z slices](docs/transforms/previews/SheetCompression-z.webp)](docs/transforms/previews/SheetCompression-z.webp) | [![SheetCompression: original and transformed Y slices](docs/transforms/previews/SheetCompression-y.webp)](docs/transforms/previews/SheetCompression-y.webp) | [![SheetCompression: original and transformed X slices](docs/transforms/previews/SheetCompression-x.webp)](docs/transforms/previews/SheetCompression-x.webp) |

</details>

<details>
<summary><strong>ShotNoise</strong></summary>

Sample Poisson photon counts from nonnegative intensities and rescale them, producing signal-dependent shot noise.

```python
A.ShotNoise(photons=100)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 61.419 ms | 2.374 ms | 1.268 ms | 25.9× | 48.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ShotNoise: original and transformed Z slices](docs/transforms/previews/ShotNoise-z.webp)](docs/transforms/previews/ShotNoise-z.webp) | [![ShotNoise: original and transformed Y slices](docs/transforms/previews/ShotNoise-y.webp)](docs/transforms/previews/ShotNoise-y.webp) | [![ShotNoise: original and transformed X slices](docs/transforms/previews/ShotNoise-x.webp)](docs/transforms/previews/ShotNoise-x.webp) |

</details>

<details>
<summary><strong>SliceIllumination</strong></summary>

Reduce brightness across selected slices and wider bands to simulate slice-dependent acquisition defects.

```python
A.SliceIllumination(defects=3, width=4, reduction=0.5, base_p=0.1, base_reduction=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.115 ms | 1.277 ms | 0.156 ms | 0.9× | 7.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![SliceIllumination: original and transformed Z slices](docs/transforms/previews/SliceIllumination-z.webp)](docs/transforms/previews/SliceIllumination-z.webp) | [![SliceIllumination: original and transformed Y slices](docs/transforms/previews/SliceIllumination-y.webp)](docs/transforms/previews/SliceIllumination-y.webp) | [![SliceIllumination: original and transformed X slices](docs/transforms/previews/SliceIllumination-x.webp)](docs/transforms/previews/SliceIllumination-x.webp) |

</details>

<details>
<summary><strong>Smear</strong></summary>

Blend shifted copies of preceding slices into each slice to simulate directional smearing.

```python
A.Smear(shift=(3, 0), alpha=0.25)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 5.003 ms | 1.556 ms | 0.461 ms | 3.2× | 10.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Smear: original and transformed Z slices](docs/transforms/previews/Smear-z.webp)](docs/transforms/previews/Smear-z.webp) | [![Smear: original and transformed Y slices](docs/transforms/previews/Smear-y.webp)](docs/transforms/previews/Smear-y.webp) | [![Smear: original and transformed X slices](docs/transforms/previews/Smear-x.webp)](docs/transforms/previews/Smear-x.webp) |

</details>

<details>
<summary><strong>StripeMasking</strong></summary>

Erase periodic slabs along selected axes with random offsets; overlapping slabs form a stripe pattern through the volume.

```python
A.StripeMasking(unit_size=(16, 16, 16), ratio=0.25, axes='zyx')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 39.640 ms | 1.319 ms | 0.140 ms | 30.1× | 283.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![StripeMasking: original and transformed Z slices](docs/transforms/previews/StripeMasking-z.webp)](docs/transforms/previews/StripeMasking-z.webp) | [![StripeMasking: original and transformed Y slices](docs/transforms/previews/StripeMasking-y.webp)](docs/transforms/previews/StripeMasking-y.webp) | [![StripeMasking: original and transformed X slices](docs/transforms/previews/StripeMasking-x.webp)](docs/transforms/previews/StripeMasking-x.webp) |

</details>

<details>
<summary><strong>ThickSlice</strong></summary>

Downsample and upsample along selected axes to simulate thicker acquisition slices. Labels follow the same slice mapping.

```python
A.ThickSlice(scale=0.5, axes='z')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.337 ms | 5.592 ms | 0.443 ms | 0.4× | 5.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ThickSlice: original and transformed Z slices](docs/transforms/previews/ThickSlice-z.webp)](docs/transforms/previews/ThickSlice-z.webp) | [![ThickSlice: original and transformed Y slices](docs/transforms/previews/ThickSlice-y.webp)](docs/transforms/previews/ThickSlice-y.webp) | [![ThickSlice: original and transformed X slices](docs/transforms/previews/ThickSlice-x.webp)](docs/transforms/previews/ThickSlice-x.webp) |

</details>

<details>
<summary><strong>Transpose</strong></summary>

Permute spatial axes for all aligned targets. This example exchanges Y and X.

```python
A.Transpose(order='zxy')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.025 ms | 2.199 ms | 0.611 ms | 0.9× | 3.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Transpose: original and transformed Z slices](docs/transforms/previews/Transpose-z.webp)](docs/transforms/previews/Transpose-z.webp) | [![Transpose: original and transformed Y slices](docs/transforms/previews/Transpose-y.webp)](docs/transforms/previews/Transpose-y.webp) | [![Transpose: original and transformed X slices](docs/transforms/previews/Transpose-x.webp)](docs/transforms/previews/Transpose-x.webp) |

</details>

<details>
<summary><strong>UnsharpMask</strong></summary>

Add a scaled difference between the image and a Gaussian-smoothed copy to enhance fine structure.

```python
A.UnsharpMask(strength=0.5, sigma=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10.299 ms | 1.632 ms | 0.394 ms | 6.3× | 26.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![UnsharpMask: original and transformed Z slices](docs/transforms/previews/UnsharpMask-z.webp)](docs/transforms/previews/UnsharpMask-z.webp) | [![UnsharpMask: original and transformed Y slices](docs/transforms/previews/UnsharpMask-y.webp)](docs/transforms/previews/UnsharpMask-y.webp) | [![UnsharpMask: original and transformed X slices](docs/transforms/previews/UnsharpMask-x.webp)](docs/transforms/previews/UnsharpMask-x.webp) |

</details>

<details>
<summary><strong>Vignetting</strong></summary>

Attenuate intensities with a radial falloff in all three dimensions, darkening voxels farther from the center.

```python
A.Vignetting(strength=0.5, radius=0.8)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 28.360 ms | 1.411 ms | 0.173 ms | 20.1× | 164.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Vignetting: original and transformed Z slices](docs/transforms/previews/Vignetting-z.webp)](docs/transforms/previews/Vignetting-z.webp) | [![Vignetting: original and transformed Y slices](docs/transforms/previews/Vignetting-y.webp)](docs/transforms/previews/Vignetting-y.webp) | [![Vignetting: original and transformed X slices](docs/transforms/previews/Vignetting-x.webp)](docs/transforms/previews/Vignetting-x.webp) |

</details>

<details>
<summary><strong>ZoomBlur</strong></summary>

Average multiple 3D zoom states around a center point to create radial blur.

```python
A.ZoomBlur(strength=0.15, samples=9)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 257.941 ms | 3.091 ms | 1.939 ms | 83.5× | 133.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ZoomBlur: original and transformed Z slices](docs/transforms/previews/ZoomBlur-z.webp)](docs/transforms/previews/ZoomBlur-z.webp) | [![ZoomBlur: original and transformed Y slices](docs/transforms/previews/ZoomBlur-y.webp)](docs/transforms/previews/ZoomBlur-y.webp) | [![ZoomBlur: original and transformed X slices](docs/transforms/previews/ZoomBlur-x.webp)](docs/transforms/previews/ZoomBlur-x.webp) |

</details>

### 2D tracing companions

These operate on 128 × 128 inputs derived from the 128³ crop. Their timings are for 2D work and they have no Z/Y/X volume slice triplet.

<details>
<summary><strong>StripGeometry</strong></summary>

Warp a 2D strip with rotation and a smooth lateral offset. This companion example uses the central Z slice of the CT crop.

```python
A.StripGeometry(degrees=5, smooth_offset=2, stride=16)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.092 ms | 0.038 ms | 0.042 ms | 28.9× | 25.8× |

[![StripGeometry: original and transformed 2D views](docs/transforms/previews/StripGeometry.webp)](docs/transforms/previews/StripGeometry.webp)

2D-only companion: central Z slice, not a cubic-volume timing.

</details>

<details>
<summary><strong>SurfacePerturb</strong></summary>

Add localized smooth displacements to a 2D coordinate surface. This companion uses a synthetic surface with CT-derived height; the last panel shows displacement magnitude.

```python
A.SurfacePerturb(blobs=3, amplitude=1, points_affected=100, p=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0.527 ms | 0.287 ms | 0.269 ms | 1.8× | 2.0× |

[![SurfacePerturb: original and transformed 2D views](docs/transforms/previews/SurfacePerturb.webp)](docs/transforms/previews/SurfacePerturb.webp)

2D-only companion: synthetic coordinate surface with CT-derived height. Preview is displacement magnitude, not an anatomical surface.

</details>
