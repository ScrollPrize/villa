# vc_reader

## Pipeline benchmark

[`benchmarks/bench_pipeline.py`](benchmarks/bench_pipeline.py) exercises affine
reading/sampling, normalization, and a fixed augmentation chain (flip, brightness,
gamma, Gaussian blur). It uses the package's NumPy dependency and accepts a local,
HTTP, or S3 **3-D uint8 Zarr v2 array**. With no `--array`, it generates a temporary,
seeded synthetic input for smoke checks; use representative CT data for performance
decisions.

```bash
# CPU smoke check using synthetic input.
python benchmarks/bench_pipeline.py --backend cpu --shape 32 32 32 --crops 2 --reps 3

# CPU/CUDA comparison on the same real data and crop origins.
python benchmarks/bench_pipeline.py --array /path/to/volume.zarr --level 0 \
  --backend both --shape 128 128 128 --crops 4 --threads 8 \
  --warmup 2 --reps 10 --output /tmp/pipeline.json
```

Use matching Release reader/augmentation extensions; CUDA requires a build with
`VC_READER_ENABLE_CUDA=ON` and an accessible GPU. `--native-dir /path/to/cmake-build`
selects both extensions from an existing build without installing them.
`--backend cuda` measures CUDA alone. `--start Z Y X`, `--jitter`, `--angle`,
`--normalization`, and cache-budget options control the workload; see `--help`.

The report includes mean/p50/p95/min/max stage and total latency, crops/s,
Mvoxels/s, output GiB/s, and CPU/CUDA median speedups. JSON also records every
measurement, crop geometry, seeds, platform, extension hashes, cache counters,
and validation errors. Validation and warmup are excluded from timings. CPU/CUDA
outputs must agree within `atol=5e-5, rtol=1e-5`; single-backend runs check shape,
dtype, and finite values.

Reading and affine sampling are fused, so their time is reported together.
Normalization and augmentation use separate calls for stage measurement.
CUDA sampling uses device chunks without automatic CPU fallback. The default
includes the final download so both backends return host arrays;
`--cuda-output device` measures a GPU-consumer endpoint instead. Calls complete
before timing stops. Host/OS caches stay warm; `--cuda-cache clear` drops device
chunks before each crop to include uploads, while `warm` reuses the device cache
subject to its budget. This is a sequential latency/throughput benchmark, not a
cold-storage or overlapped training-loader benchmark.

## Augmentations

Each entry shows the **measured configuration**, not the constructor defaults. Use `from vc_reader import augment as A`.

Benchmarks use **128 × 128 × 128 float32** crops from PHercParis4 volume `20260411134726-2.400um-0.2m-78keV-masked.zarr`, **scale 2**, centered at ZYX `(9064, 3664, 3664)`. CT values are divided by 255. Timings are medians of 5 measured calls after 1 warmup, using one CPU thread and a Release build.

**CUDA uploaded** includes uploading the host input and downloading the output. **CUDA resident** starts and ends with device arrays. Speedup is CPU time divided by the corresponding CUDA time; values above 1× mean CUDA is faster.

**Images:** Z, Y and X slice pairs show the original above the augmented output. Planes are selected where the effect is visible, with the same display window 


Expand an augmentation below to see its parameters, timings and images. Elastic also includes a WebM animation.

<details>
<summary><strong>AdditiveNoise</strong></summary>

Add independent noise to each voxel. Supports uniform, Gaussian, Laplace and centered beta distributions; this example uses Laplace noise.

```python
A.AdditiveNoise(scale=0.05, distribution='laplace', mean=0)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 21.679 ms | 1.362 ms | 0.286 ms | 15.9× | 75.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![AdditiveNoise: original and augmented Z slices](docs/augmentations/previews/AdditiveNoise-z.webp)](docs/augmentations/previews/AdditiveNoise-z.webp) | [![AdditiveNoise: original and augmented Y slices](docs/augmentations/previews/AdditiveNoise-y.webp)](docs/augmentations/previews/AdditiveNoise-y.webp) | [![AdditiveNoise: original and augmented X slices](docs/augmentations/previews/AdditiveNoise-x.webp)](docs/augmentations/previews/AdditiveNoise-x.webp) |

</details>

<details>
<summary><strong>Affine</strong></summary>

Rotate, translate and scale the volume, applying the same geometry to images and labels.

```python
A.Affine(degrees=5, translate=2, scale=0.95)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 55.834 ms | 6.180 ms | 0.920 ms | 9.0× | 60.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Affine: original and augmented Z slices](docs/augmentations/previews/Affine-z.webp)](docs/augmentations/previews/Affine-z.webp) | [![Affine: original and augmented Y slices](docs/augmentations/previews/Affine-y.webp)](docs/augmentations/previews/Affine-y.webp) | [![Affine: original and augmented X slices](docs/augmentations/previews/Affine-x.webp)](docs/augmentations/previews/Affine-x.webp) |

</details>

<details>
<summary><strong>BiasField</strong></summary>

Multiply intensities by a smooth 3D gain field generated from a coarse Gaussian random lattice.

```python
A.BiasField(std=0.3, scale=0.025)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 12.392 ms | 2.508 ms | 1.465 ms | 4.9× | 8.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BiasField: original and augmented Z slices](docs/augmentations/previews/BiasField-z.webp)](docs/augmentations/previews/BiasField-z.webp) | [![BiasField: original and augmented Y slices](docs/augmentations/previews/BiasField-y.webp)](docs/augmentations/previews/BiasField-y.webp) | [![BiasField: original and augmented X slices](docs/augmentations/previews/BiasField-x.webp)](docs/augmentations/previews/BiasField-x.webp) |

</details>

<details>
<summary><strong>BlankRectangle</strong></summary>

Fill randomly placed cuboids with a chosen value, simulating missing regions.

```python
A.BlankRectangle(size=16, value=0, count=3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0.659 ms | 1.259 ms | 0.152 ms | 0.5× | 4.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BlankRectangle: original and augmented Z slices](docs/augmentations/previews/BlankRectangle-z.webp)](docs/augmentations/previews/BlankRectangle-z.webp) | [![BlankRectangle: original and augmented Y slices](docs/augmentations/previews/BlankRectangle-y.webp)](docs/augmentations/previews/BlankRectangle-y.webp) | [![BlankRectangle: original and augmented X slices](docs/augmentations/previews/BlankRectangle-x.webp)](docs/augmentations/previews/BlankRectangle-x.webp) |

</details>

<details>
<summary><strong>BrightnessAdd</strong></summary>

Add a constant offset to every voxel in the selected image channels.

```python
A.BrightnessAdd(shift=0.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.355 ms | 1.178 ms | 0.117 ms | 1.2× | 11.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BrightnessAdd: original and augmented Z slices](docs/augmentations/previews/BrightnessAdd-z.webp)](docs/augmentations/previews/BrightnessAdd-z.webp) | [![BrightnessAdd: original and augmented Y slices](docs/augmentations/previews/BrightnessAdd-y.webp)](docs/augmentations/previews/BrightnessAdd-y.webp) | [![BrightnessAdd: original and augmented X slices](docs/augmentations/previews/BrightnessAdd-x.webp)](docs/augmentations/previews/BrightnessAdd-x.webp) |

</details>

<details>
<summary><strong>BrightnessGradient</strong></summary>

Add a localized Gaussian brightness field. This example fixes its center at the middle of the volume; `loc=(0, 1)` samples a random position. The field is mean-centered by default.

```python
A.BrightnessGradient(scale=12, loc=0.5, strength=0.3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 9.871 ms | 1.567 ms | 0.475 ms | 6.3× | 20.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BrightnessGradient: original and augmented Z slices](docs/augmentations/previews/BrightnessGradient-z.webp)](docs/augmentations/previews/BrightnessGradient-z.webp) | [![BrightnessGradient: original and augmented Y slices](docs/augmentations/previews/BrightnessGradient-y.webp)](docs/augmentations/previews/BrightnessGradient-y.webp) | [![BrightnessGradient: original and augmented X slices](docs/augmentations/previews/BrightnessGradient-x.webp)](docs/augmentations/previews/BrightnessGradient-x.webp) |

</details>

<details>
<summary><strong>BrightnessMultiply</strong></summary>

Multiply every voxel in the selected image channels by a brightness factor.

```python
A.BrightnessMultiply(factor=1.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.221 ms | 1.068 ms | 0.112 ms | 1.1× | 10.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![BrightnessMultiply: original and augmented Z slices](docs/augmentations/previews/BrightnessMultiply-z.webp)](docs/augmentations/previews/BrightnessMultiply-z.webp) | [![BrightnessMultiply: original and augmented Y slices](docs/augmentations/previews/BrightnessMultiply-y.webp)](docs/augmentations/previews/BrightnessMultiply-y.webp) | [![BrightnessMultiply: original and augmented X slices](docs/augmentations/previews/BrightnessMultiply-x.webp)](docs/augmentations/previews/BrightnessMultiply-x.webp) |

</details>

<details>
<summary><strong>ChannelMisalignment</strong></summary>

Independently shift, rotate and scale selected image channels to simulate registration errors. This example moves the second of two identical CT channels.

```python
A.ChannelMisalignment(degrees=5, translate=2, scale=1, channels=(1,))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 50.230 ms | 2.857 ms | 0.539 ms | 17.6× | 93.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ChannelMisalignment: original and augmented Z slices](docs/augmentations/previews/ChannelMisalignment-z.webp)](docs/augmentations/previews/ChannelMisalignment-z.webp) | [![ChannelMisalignment: original and augmented Y slices](docs/augmentations/previews/ChannelMisalignment-y.webp)](docs/augmentations/previews/ChannelMisalignment-y.webp) | [![ChannelMisalignment: original and augmented X slices](docs/augmentations/previews/ChannelMisalignment-x.webp)](docs/augmentations/previews/ChannelMisalignment-x.webp) |

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
| 52.156 ms | 1.698 ms | 0.595 ms | 30.7× | 87.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![CLAHE: original and augmented Z slices](docs/augmentations/previews/CLAHE-z.webp)](docs/augmentations/previews/CLAHE-z.webp) | [![CLAHE: original and augmented Y slices](docs/augmentations/previews/CLAHE-y.webp)](docs/augmentations/previews/CLAHE-y.webp) | [![CLAHE: original and augmented X slices](docs/augmentations/previews/CLAHE-x.webp)](docs/augmentations/previews/CLAHE-x.webp) |

</details>

<details>
<summary><strong>ClipOutliers</strong></summary>

Clip intensities at selected percentiles. This example then restores the original standard deviation.

```python
A.ClipOutliers(lower=1, upper=99, p_retain_std=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 23.865 ms | 1.568 ms | 0.524 ms | 15.2× | 45.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ClipOutliers: original and augmented Z slices](docs/augmentations/previews/ClipOutliers-z.webp)](docs/augmentations/previews/ClipOutliers-z.webp) | [![ClipOutliers: original and augmented Y slices](docs/augmentations/previews/ClipOutliers-y.webp)](docs/augmentations/previews/ClipOutliers-y.webp) | [![ClipOutliers: original and augmented X slices](docs/augmentations/previews/ClipOutliers-x.webp)](docs/augmentations/previews/ClipOutliers-x.webp) |

</details>

<details>
<summary><strong>Contrast</strong></summary>

Scale intensity differences from the channel mean. The default range-preservation option clips results to the original range.

```python
A.Contrast(factor=1.2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 5.554 ms | 1.177 ms | 0.135 ms | 4.7× | 41.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Contrast: original and augmented Z slices](docs/augmentations/previews/Contrast-z.webp)](docs/augmentations/previews/Contrast-z.webp) | [![Contrast: original and augmented Y slices](docs/augmentations/previews/Contrast-y.webp)](docs/augmentations/previews/Contrast-y.webp) | [![Contrast: original and augmented X slices](docs/augmentations/previews/Contrast-x.webp)](docs/augmentations/previews/Contrast-x.webp) |

</details>

<details>
<summary><strong>CropPad</strong></summary>

Crop or pad all aligned targets to a requested ZYX shape. This example takes a centered 112³ crop from the 128³ input.

```python
A.CropPad(shape=(112, 112, 112))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0.765 ms | 1.902 ms | 0.555 ms | 0.4× | 1.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![CropPad: original and augmented Z slices](docs/augmentations/previews/CropPad-z.webp)](docs/augmentations/previews/CropPad-z.webp) | [![CropPad: original and augmented Y slices](docs/augmentations/previews/CropPad-y.webp)](docs/augmentations/previews/CropPad-y.webp) | [![CropPad: original and augmented X slices](docs/augmentations/previews/CropPad-x.webp)](docs/augmentations/previews/CropPad-x.webp) |

</details>

<details>
<summary><strong>Defocus</strong></summary>

Average intensities within a spherical 3D neighborhood to simulate out-of-focus acquisition.

```python
A.Defocus(radius=2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 13.262 ms | 1.949 ms | 0.808 ms | 6.8× | 16.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Defocus: original and augmented Z slices](docs/augmentations/previews/Defocus-z.webp)](docs/augmentations/previews/Defocus-z.webp) | [![Defocus: original and augmented Y slices](docs/augmentations/previews/Defocus-y.webp)](docs/augmentations/previews/Defocus-y.webp) | [![Defocus: original and augmented X slices](docs/augmentations/previews/Defocus-x.webp)](docs/augmentations/previews/Defocus-x.webp) |

</details>

<details>
<summary><strong>Elastic</strong></summary>

Warp the volume with a smooth random 3D B-spline displacement field, keeping images and labels aligned. The previews use interior slices so boundary padding does not dominate the view.

```python
A.Elastic(sigma=32, magnitude=8, border=True)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 91.666 ms | 3.527 ms | 1.796 ms | 26.0× | 51.0× |

[▶ Watch Elastic animation (WebM)](docs/augmentations/animations/Elastic.webm)

Strong cubic B-spline deformation: original → deformed → original. Fixed Z/Y/X slices, seed 125, achieved RMS 5.36 voxels.

[![Elastic animation poster](docs/augmentations/animations/Elastic-poster.webp)](docs/augmentations/animations/Elastic.webm)

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Elastic: original and augmented Z slices](docs/augmentations/previews/Elastic-z.webp)](docs/augmentations/previews/Elastic-z.webp) | [![Elastic: original and augmented Y slices](docs/augmentations/previews/Elastic-y.webp)](docs/augmentations/previews/Elastic-y.webp) | [![Elastic: original and augmented X slices](docs/augmentations/previews/Elastic-x.webp)](docs/augmentations/previews/Elastic-x.webp) |

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
| 23.191 ms | 1.448 ms | 0.406 ms | 16.0× | 57.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Equalize: original and augmented Z slices](docs/augmentations/previews/Equalize-z.webp)](docs/augmentations/previews/Equalize-z.webp) | [![Equalize: original and augmented Y slices](docs/augmentations/previews/Equalize-y.webp)](docs/augmentations/previews/Equalize-y.webp) | [![Equalize: original and augmented X slices](docs/augmentations/previews/Equalize-x.webp)](docs/augmentations/previews/Equalize-x.webp) |

</details>

<details>
<summary><strong>FDA</strong></summary>

Blend low-frequency amplitudes from a reference volume into the 3D Fourier spectrum while preserving the source phase.

```python
A.FDA(reference=reference, blend=1, fraction=0.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 608.138 ms | 4.508 ms | 3.091 ms | 134.9× | 196.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![FDA: original and augmented Z slices](docs/augmentations/previews/FDA-z.webp)](docs/augmentations/previews/FDA-z.webp) | [![FDA: original and augmented Y slices](docs/augmentations/previews/FDA-y.webp)](docs/augmentations/previews/FDA-y.webp) | [![FDA: original and augmented X slices](docs/augmentations/previews/FDA-x.webp)](docs/augmentations/previews/FDA-x.webp) |

</details>

<details>
<summary><strong>Flip</strong></summary>

Mirror the volume along selected axes. This example always flips X.

```python
A.Flip(axes='x', p_axis=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.216 ms | 2.390 ms | 0.871 ms | 0.5× | 1.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Flip: original and augmented Z slices](docs/augmentations/previews/Flip-z.webp)](docs/augmentations/previews/Flip-z.webp) | [![Flip: original and augmented Y slices](docs/augmentations/previews/Flip-y.webp)](docs/augmentations/previews/Flip-y.webp) | [![Flip: original and augmented X slices](docs/augmentations/previews/Flip-x.webp)](docs/augmentations/previews/Flip-x.webp) |

</details>

<details>
<summary><strong>ForegroundCrop</strong></summary>

Choose a random crop containing a sampled foreground voxel. All aligned targets follow the crop; empty masks fall back to random cropping.

```python
A.ForegroundCrop(shape=(112, 112, 112))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.324 ms | 2.018 ms | 0.691 ms | 1.2× | 3.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ForegroundCrop: original and augmented Z slices](docs/augmentations/previews/ForegroundCrop-z.webp)](docs/augmentations/previews/ForegroundCrop-z.webp) | [![ForegroundCrop: original and augmented Y slices](docs/augmentations/previews/ForegroundCrop-y.webp)](docs/augmentations/previews/ForegroundCrop-y.webp) | [![ForegroundCrop: original and augmented X slices](docs/augmentations/previews/ForegroundCrop-x.webp)](docs/augmentations/previews/ForegroundCrop-x.webp) |

</details>

<details>
<summary><strong>ForegroundDropout</strong></summary>

Place cuboids around sampled foreground voxels and erase only foreground inside those cuboids.

```python
A.ForegroundDropout(size=(16, 16, 16), count=3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 12.382 ms | 1.959 ms | 0.417 ms | 6.3× | 29.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ForegroundDropout: original and augmented Z slices](docs/augmentations/previews/ForegroundDropout-z.webp)](docs/augmentations/previews/ForegroundDropout-z.webp) | [![ForegroundDropout: original and augmented Y slices](docs/augmentations/previews/ForegroundDropout-y.webp)](docs/augmentations/previews/ForegroundDropout-y.webp) | [![ForegroundDropout: original and augmented X slices](docs/augmentations/previews/ForegroundDropout-x.webp)](docs/augmentations/previews/ForegroundDropout-x.webp) |

</details>

<details>
<summary><strong>Gamma</strong></summary>

Apply a power-law intensity curve over the channel range, changing the relative brightness of dark and light structures.

```python
A.Gamma(gamma=0.8)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8.721 ms | 1.325 ms | 0.236 ms | 6.6× | 36.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Gamma: original and augmented Z slices](docs/augmentations/previews/Gamma-z.webp)](docs/augmentations/previews/Gamma-z.webp) | [![Gamma: original and augmented Y slices](docs/augmentations/previews/Gamma-y.webp)](docs/augmentations/previews/Gamma-y.webp) | [![Gamma: original and augmented X slices](docs/augmentations/previews/Gamma-x.webp)](docs/augmentations/previews/Gamma-x.webp) |

</details>

<details>
<summary><strong>GaussianBlur</strong></summary>

Smooth the volume with a separable 3D Gaussian filter.

```python
A.GaussianBlur(sigma=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 5.919 ms | 1.478 ms | 0.316 ms | 4.0× | 18.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GaussianBlur: original and augmented Z slices](docs/augmentations/previews/GaussianBlur-z.webp)](docs/augmentations/previews/GaussianBlur-z.webp) | [![GaussianBlur: original and augmented Y slices](docs/augmentations/previews/GaussianBlur-y.webp)](docs/augmentations/previews/GaussianBlur-y.webp) | [![GaussianBlur: original and augmented X slices](docs/augmentations/previews/GaussianBlur-x.webp)](docs/augmentations/previews/GaussianBlur-x.webp) |

</details>

<details>
<summary><strong>GaussianNoise</strong></summary>

Add independent zero-mean Gaussian noise with the specified standard deviation.

```python
A.GaussianNoise(std=0.05)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 9.287 ms | 1.245 ms | 0.122 ms | 7.5× | 76.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GaussianNoise: original and augmented Z slices](docs/augmentations/previews/GaussianNoise-z.webp)](docs/augmentations/previews/GaussianNoise-z.webp) | [![GaussianNoise: original and augmented Y slices](docs/augmentations/previews/GaussianNoise-y.webp)](docs/augmentations/previews/GaussianNoise-y.webp) | [![GaussianNoise: original and augmented X slices](docs/augmentations/previews/GaussianNoise-x.webp)](docs/augmentations/previews/GaussianNoise-x.webp) |

</details>

<details>
<summary><strong>GibbsRinging</strong></summary>

Truncate high frequencies in the 3D Fourier spectrum, creating ringing around sharp boundaries.

```python
A.GibbsRinging(cutoff=0.7)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 397.072 ms | 1.585 ms | 0.487 ms | 250.5× | 815.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GibbsRinging: original and augmented Z slices](docs/augmentations/previews/GibbsRinging-z.webp)](docs/augmentations/previews/GibbsRinging-z.webp) | [![GibbsRinging: original and augmented Y slices](docs/augmentations/previews/GibbsRinging-y.webp)](docs/augmentations/previews/GibbsRinging-y.webp) | [![GibbsRinging: original and augmented X slices](docs/augmentations/previews/GibbsRinging-x.webp)](docs/augmentations/previews/GibbsRinging-x.webp) |

</details>

<details>
<summary><strong>GridDropout</strong></summary>

Erase a periodic grid of cuboid holes with a randomly sampled offset.

```python
A.GridDropout(unit_size=(16, 16, 16), ratio=0.4)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 39.331 ms | 1.415 ms | 0.336 ms | 27.8× | 117.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![GridDropout: original and augmented Z slices](docs/augmentations/previews/GridDropout-z.webp)](docs/augmentations/previews/GridDropout-z.webp) | [![GridDropout: original and augmented Y slices](docs/augmentations/previews/GridDropout-y.webp)](docs/augmentations/previews/GridDropout-y.webp) | [![GridDropout: original and augmented X slices](docs/augmentations/previews/GridDropout-x.webp)](docs/augmentations/previews/GridDropout-x.webp) |

</details>

<details>
<summary><strong>HistogramMatching</strong></summary>

Match the volume intensity histogram to a reference volume, with optional blending.

```python
A.HistogramMatching(reference=reference, blend=1, bins=256)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 97.854 ms | 4.232 ms | 3.148 ms | 23.1× | 31.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![HistogramMatching: original and augmented Z slices](docs/augmentations/previews/HistogramMatching-z.webp)](docs/augmentations/previews/HistogramMatching-z.webp) | [![HistogramMatching: original and augmented Y slices](docs/augmentations/previews/HistogramMatching-y.webp)](docs/augmentations/previews/HistogramMatching-y.webp) | [![HistogramMatching: original and augmented X slices](docs/augmentations/previews/HistogramMatching-x.webp)](docs/augmentations/previews/HistogramMatching-x.webp) |

</details>

<details>
<summary><strong>Illumination</strong></summary>

Apply a smooth multiplicative illumination field. Supports directional ramps, corner falloff and Gaussian illumination; this example uses a diagonal ramp.

```python
A.Illumination(strength=0.3, mode='linear', direction=(1, 1, 1))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 35.146 ms | 1.314 ms | 0.204 ms | 26.7× | 172.0× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Illumination: original and augmented Z slices](docs/augmentations/previews/Illumination-z.webp)](docs/augmentations/previews/Illumination-z.webp) | [![Illumination: original and augmented Y slices](docs/augmentations/previews/Illumination-y.webp)](docs/augmentations/previews/Illumination-y.webp) | [![Illumination: original and augmented X slices](docs/augmentations/previews/Illumination-x.webp)](docs/augmentations/previews/Illumination-x.webp) |

</details>

<details>
<summary><strong>Invert</strong></summary>

Reflect intensities around the channel mean, reversing bright and dark structures.

```python
A.Invert()
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2.170 ms | 1.127 ms | 0.180 ms | 1.9× | 12.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Invert: original and augmented Z slices](docs/augmentations/previews/Invert-z.webp)](docs/augmentations/previews/Invert-z.webp) | [![Invert: original and augmented Y slices](docs/augmentations/previews/Invert-y.webp)](docs/augmentations/previews/Invert-y.webp) | [![Invert: original and augmented X slices](docs/augmentations/previews/Invert-x.webp)](docs/augmentations/previews/Invert-x.webp) |

</details>

<details>
<summary><strong>LocalContrast</strong></summary>

Change contrast around a Gaussian-weighted local mean inside a smooth region.

```python
A.LocalContrast(scale=12, loc=0.5, factor=1.2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4.171 ms | 1.423 ms | 0.427 ms | 2.9× | 9.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LocalContrast: original and augmented Z slices](docs/augmentations/previews/LocalContrast-z.webp)](docs/augmentations/previews/LocalContrast-z.webp) | [![LocalContrast: original and augmented Y slices](docs/augmentations/previews/LocalContrast-y.webp)](docs/augmentations/previews/LocalContrast-y.webp) | [![LocalContrast: original and augmented X slices](docs/augmentations/previews/LocalContrast-x.webp)](docs/augmentations/previews/LocalContrast-x.webp) |

</details>

<details>
<summary><strong>LocalGamma</strong></summary>

Blend a gamma-adjusted image into a localized Gaussian region.

```python
A.LocalGamma(scale=12, loc=0.5, gamma=0.8)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 15.140 ms | 1.494 ms | 0.449 ms | 10.1× | 33.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LocalGamma: original and augmented Z slices](docs/augmentations/previews/LocalGamma-z.webp)](docs/augmentations/previews/LocalGamma-z.webp) | [![LocalGamma: original and augmented Y slices](docs/augmentations/previews/LocalGamma-y.webp)](docs/augmentations/previews/LocalGamma-y.webp) | [![LocalGamma: original and augmented X slices](docs/augmentations/previews/LocalGamma-x.webp)](docs/augmentations/previews/LocalGamma-x.webp) |

</details>

<details>
<summary><strong>LocalSmoothing</strong></summary>

Blend Gaussian smoothing into a localized region, leaving distant voxels largely unchanged.

```python
A.LocalSmoothing(scale=12, loc=0.5, strength=0.7, sigma=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8.806 ms | 1.687 ms | 0.578 ms | 5.2× | 15.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LocalSmoothing: original and augmented Z slices](docs/augmentations/previews/LocalSmoothing-z.webp)](docs/augmentations/previews/LocalSmoothing-z.webp) | [![LocalSmoothing: original and augmented Y slices](docs/augmentations/previews/LocalSmoothing-y.webp)](docs/augmentations/previews/LocalSmoothing-y.webp) | [![LocalSmoothing: original and augmented X slices](docs/augmentations/previews/LocalSmoothing-x.webp)](docs/augmentations/previews/LocalSmoothing-x.webp) |

</details>

<details>
<summary><strong>LowResolution</strong></summary>

Downsample and upsample the image to simulate reduced spatial resolution.

```python
A.LowResolution(scale=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 7.724 ms | 1.531 ms | 0.684 ms | 5.0× | 11.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![LowResolution: original and augmented Z slices](docs/augmentations/previews/LowResolution-z.webp)](docs/augmentations/previews/LowResolution-z.webp) | [![LowResolution: original and augmented Y slices](docs/augmentations/previews/LowResolution-y.webp)](docs/augmentations/previews/LowResolution-y.webp) | [![LowResolution: original and augmented X slices](docs/augmentations/previews/LowResolution-x.webp)](docs/augmentations/previews/LowResolution-x.webp) |

</details>

<details>
<summary><strong>MedianFilter</strong></summary>

Replace each voxel with the median of its 3D neighborhood, suppressing isolated noise while retaining edges.

```python
A.MedianFilter(size=3)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16.054 ms | 1.267 ms | 0.211 ms | 12.7× | 75.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![MedianFilter: original and augmented Z slices](docs/augmentations/previews/MedianFilter-z.webp)](docs/augmentations/previews/MedianFilter-z.webp) | [![MedianFilter: original and augmented Y slices](docs/augmentations/previews/MedianFilter-y.webp)](docs/augmentations/previews/MedianFilter-y.webp) | [![MedianFilter: original and augmented X slices](docs/augmentations/previews/MedianFilter-x.webp)](docs/augmentations/previews/MedianFilter-x.webp) |

</details>

<details>
<summary><strong>Morphology</strong></summary>

Apply a morphological operation to labels. This example closes small gaps using dilation followed by erosion.

```python
A.Morphology(operation='close', radius=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 11.810 ms | 0.701 ms | 0.263 ms | 16.8× | 44.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Morphology: original and augmented Z slices](docs/augmentations/previews/Morphology-z.webp)](docs/augmentations/previews/Morphology-z.webp) | [![Morphology: original and augmented Y slices](docs/augmentations/previews/Morphology-y.webp)](docs/augmentations/previews/Morphology-y.webp) | [![Morphology: original and augmented X slices](docs/augmentations/previews/Morphology-x.webp)](docs/augmentations/previews/Morphology-x.webp) |

</details>

<details>
<summary><strong>MotionBlur</strong></summary>

Average samples along a 3D line to simulate motion during acquisition.

```python
A.MotionBlur(length=5, direction=(1, 1, 1), samples=9)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 215.162 ms | 2.470 ms | 1.397 ms | 87.1× | 154.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![MotionBlur: original and augmented Z slices](docs/augmentations/previews/MotionBlur-z.webp)](docs/augmentations/previews/MotionBlur-z.webp) | [![MotionBlur: original and augmented Y slices](docs/augmentations/previews/MotionBlur-y.webp)](docs/augmentations/previews/MotionBlur-y.webp) | [![MotionBlur: original and augmented X slices](docs/augmentations/previews/MotionBlur-x.webp)](docs/augmentations/previews/MotionBlur-x.webp) |

</details>

<details>
<summary><strong>MultiplicativeNoise</strong></summary>

Multiply each voxel by an independently sampled uniform random gain.

```python
A.MultiplicativeNoise(low=0.8, high=1.2)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 13.039 ms | 1.192 ms | 0.184 ms | 10.9× | 70.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![MultiplicativeNoise: original and augmented Z slices](docs/augmentations/previews/MultiplicativeNoise-z.webp)](docs/augmentations/previews/MultiplicativeNoise-z.webp) | [![MultiplicativeNoise: original and augmented Y slices](docs/augmentations/previews/MultiplicativeNoise-y.webp)](docs/augmentations/previews/MultiplicativeNoise-y.webp) | [![MultiplicativeNoise: original and augmented X slices](docs/augmentations/previews/MultiplicativeNoise-x.webp)](docs/augmentations/previews/MultiplicativeNoise-x.webp) |

</details>

<details>
<summary><strong>Normalize</strong></summary>

Normalize intensities per channel. This example uses z-score normalization of nonzero voxels, keeping zeros unchanged; its display window includes negative values.

```python
A.Normalize(mode='zscore')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 6.590 ms | 1.338 ms | 0.179 ms | 4.9× | 36.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Normalize: original and augmented Z slices](docs/augmentations/previews/Normalize-z.webp)](docs/augmentations/previews/Normalize-z.webp) | [![Normalize: original and augmented Y slices](docs/augmentations/previews/Normalize-y.webp)](docs/augmentations/previews/Normalize-y.webp) | [![Normalize: original and augmented X slices](docs/augmentations/previews/Normalize-x.webp)](docs/augmentations/previews/Normalize-x.webp) |

</details>

<details>
<summary><strong>ObjectDropout</strong></summary>

Randomly remove whole connected foreground objects, clearing their image voxels and labels by default.

```python
A.ObjectDropout(drop_probability=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 14.822 ms | 2.138 ms | 0.500 ms | 6.9× | 29.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ObjectDropout: original and augmented Z slices](docs/augmentations/previews/ObjectDropout-z.webp)](docs/augmentations/previews/ObjectDropout-z.webp) | [![ObjectDropout: original and augmented Y slices](docs/augmentations/previews/ObjectDropout-y.webp)](docs/augmentations/previews/ObjectDropout-y.webp) | [![ObjectDropout: original and augmented X slices](docs/augmentations/previews/ObjectDropout-x.webp)](docs/augmentations/previews/ObjectDropout-x.webp) |

</details>

<details>
<summary><strong>OrientedBlur</strong></summary>

Smooth along a specified 3D direction and two perpendicular directions with separate blur widths. This example is aligned with Z.

```python
A.OrientedBlur(sigma_along=2, sigma_across=0.5, axis=(1, 0, 0))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 12.713 ms | 4.846 ms | 4.078 ms | 2.6× | 3.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![OrientedBlur: original and augmented Z slices](docs/augmentations/previews/OrientedBlur-z.webp)](docs/augmentations/previews/OrientedBlur-z.webp) | [![OrientedBlur: original and augmented Y slices](docs/augmentations/previews/OrientedBlur-y.webp)](docs/augmentations/previews/OrientedBlur-y.webp) | [![OrientedBlur: original and augmented X slices](docs/augmentations/previews/OrientedBlur-x.webp)](docs/augmentations/previews/OrientedBlur-x.webp) |

</details>

<details>
<summary><strong>PixelDistributionAdaptation</strong></summary>

Transfer reference intensity statistics. The standard mode shown here blends the reference mean and standard deviation; minmax mode transfers the range.

```python
A.PixelDistributionAdaptation(reference=reference, blend=1, mode='standard')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 32.552 ms | 5.949 ms | 7.677 ms | 5.5× | 4.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![PixelDistributionAdaptation: original and augmented Z slices](docs/augmentations/previews/PixelDistributionAdaptation-z.webp)](docs/augmentations/previews/PixelDistributionAdaptation-z.webp) | [![PixelDistributionAdaptation: original and augmented Y slices](docs/augmentations/previews/PixelDistributionAdaptation-y.webp)](docs/augmentations/previews/PixelDistributionAdaptation-y.webp) | [![PixelDistributionAdaptation: original and augmented X slices](docs/augmentations/previews/PixelDistributionAdaptation-x.webp)](docs/augmentations/previews/PixelDistributionAdaptation-x.webp) |

</details>

<details>
<summary><strong>PlasmaBrightnessContrast</strong></summary>

Use a smooth multiscale 3D random field to vary brightness and contrast throughout the volume.

```python
A.PlasmaBrightnessContrast(brightness=0.2, contrast=0.3, scale=0.025, persistence=0.5, octaves=5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 34.523 ms | 1.916 ms | 0.817 ms | 18.0× | 42.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![PlasmaBrightnessContrast: original and augmented Z slices](docs/augmentations/previews/PlasmaBrightnessContrast-z.webp)](docs/augmentations/previews/PlasmaBrightnessContrast-z.webp) | [![PlasmaBrightnessContrast: original and augmented Y slices](docs/augmentations/previews/PlasmaBrightnessContrast-y.webp)](docs/augmentations/previews/PlasmaBrightnessContrast-y.webp) | [![PlasmaBrightnessContrast: original and augmented X slices](docs/augmentations/previews/PlasmaBrightnessContrast-x.webp)](docs/augmentations/previews/PlasmaBrightnessContrast-x.webp) |

</details>

<details>
<summary><strong>PlasmaShadow</strong></summary>

Darken the volume with a smooth multiscale 3D random shadow field.

```python
A.PlasmaShadow(strength=0.5, scale=0.025, persistence=0.5, octaves=5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 27.610 ms | 1.812 ms | 0.742 ms | 15.2× | 37.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![PlasmaShadow: original and augmented Z slices](docs/augmentations/previews/PlasmaShadow-z.webp)](docs/augmentations/previews/PlasmaShadow-z.webp) | [![PlasmaShadow: original and augmented Y slices](docs/augmentations/previews/PlasmaShadow-y.webp)](docs/augmentations/previews/PlasmaShadow-y.webp) | [![PlasmaShadow: original and augmented X slices](docs/augmentations/previews/PlasmaShadow-x.webp)](docs/augmentations/previews/PlasmaShadow-x.webp) |

</details>

<details>
<summary><strong>RandomBinaryOperator</strong></summary>

Apply a selected binary morphological operation to image channels. This example dilates a thresholded foreground volume.

```python
A.RandomBinaryOperator(channels=(0,), radius=1, operations=('dilate',))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16.719 ms | 1.314 ms | 0.253 ms | 12.7× | 66.0× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RandomBinaryOperator: original and augmented Z slices](docs/augmentations/previews/RandomBinaryOperator-z.webp)](docs/augmentations/previews/RandomBinaryOperator-z.webp) | [![RandomBinaryOperator: original and augmented Y slices](docs/augmentations/previews/RandomBinaryOperator-y.webp)](docs/augmentations/previews/RandomBinaryOperator-y.webp) | [![RandomBinaryOperator: original and augmented X slices](docs/augmentations/previews/RandomBinaryOperator-x.webp)](docs/augmentations/previews/RandomBinaryOperator-x.webp) |

</details>

<details>
<summary><strong>RandomToneCurve</strong></summary>

Apply a random monotone nonlinear intensity curve while preserving the channel range endpoints.

```python
A.RandomToneCurve(scale=0.1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 13.544 ms | 1.279 ms | 0.159 ms | 10.6× | 85.2× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RandomToneCurve: original and augmented Z slices](docs/augmentations/previews/RandomToneCurve-z.webp)](docs/augmentations/previews/RandomToneCurve-z.webp) | [![RandomToneCurve: original and augmented Y slices](docs/augmentations/previews/RandomToneCurve-y.webp)](docs/augmentations/previews/RandomToneCurve-y.webp) | [![RandomToneCurve: original and augmented X slices](docs/augmentations/previews/RandomToneCurve-x.webp)](docs/augmentations/previews/RandomToneCurve-x.webp) |

</details>

<details>
<summary><strong>RemoveComponents</strong></summary>

Randomly remove connected foreground components below a size threshold. This example modifies the label mask.

```python
A.RemoveComponents(on='mask', max_fraction=0.25, p_fill=0)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 26.932 ms | 1.299 ms | 0.941 ms | 20.7× | 28.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RemoveComponents: original and augmented Z slices](docs/augmentations/previews/RemoveComponents-z.webp)](docs/augmentations/previews/RemoveComponents-z.webp) | [![RemoveComponents: original and augmented Y slices](docs/augmentations/previews/RemoveComponents-y.webp)](docs/augmentations/previews/RemoveComponents-y.webp) | [![RemoveComponents: original and augmented X slices](docs/augmentations/previews/RemoveComponents-x.webp)](docs/augmentations/previews/RemoveComponents-x.webp) |

</details>

<details>
<summary><strong>RicianNoise</strong></summary>

Combine the image with two independent Gaussian noise fields to simulate magnitude-image Rician noise.

```python
A.RicianNoise(std=0.05)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 18.819 ms | 1.272 ms | 0.178 ms | 14.8× | 105.6× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RicianNoise: original and augmented Z slices](docs/augmentations/previews/RicianNoise-z.webp)](docs/augmentations/previews/RicianNoise-z.webp) | [![RicianNoise: original and augmented Y slices](docs/augmentations/previews/RicianNoise-y.webp)](docs/augmentations/previews/RicianNoise-y.webp) | [![RicianNoise: original and augmented X slices](docs/augmentations/previews/RicianNoise-x.webp)](docs/augmentations/previews/RicianNoise-x.webp) |

</details>

<details>
<summary><strong>RingingOvershoot</strong></summary>

Apply a separable windowed-sinc filter in 3D, retaining edge overshoot and ringing.

```python
A.RingingOvershoot(cutoff=0.7, radius=4)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 15.109 ms | 5.878 ms | 4.299 ms | 2.6× | 3.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![RingingOvershoot: original and augmented Z slices](docs/augmentations/previews/RingingOvershoot-z.webp)](docs/augmentations/previews/RingingOvershoot-z.webp) | [![RingingOvershoot: original and augmented Y slices](docs/augmentations/previews/RingingOvershoot-y.webp)](docs/augmentations/previews/RingingOvershoot-y.webp) | [![RingingOvershoot: original and augmented X slices](docs/augmentations/previews/RingingOvershoot-x.webp)](docs/augmentations/previews/RingingOvershoot-x.webp) |

</details>

<details>
<summary><strong>Rotate90</strong></summary>

Rotate aligned targets by multiples of 90 degrees. This example rotates the YX plane by one quarter turn.

```python
A.Rotate90(planes=('yx',), turns=(1,))
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.959 ms | 2.451 ms | 0.829 ms | 0.8× | 2.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Rotate90: original and augmented Z slices](docs/augmentations/previews/Rotate90-z.webp)](docs/augmentations/previews/Rotate90-z.webp) | [![Rotate90: original and augmented Y slices](docs/augmentations/previews/Rotate90-y.webp)](docs/augmentations/previews/Rotate90-y.webp) | [![Rotate90: original and augmented X slices](docs/augmentations/previews/Rotate90-x.webp)](docs/augmentations/previews/Rotate90-x.webp) |

</details>

<details>
<summary><strong>SaltAndPepper</strong></summary>

Replace selected voxels with the channel minimum or maximum, producing dark and bright impulse noise.

```python
A.SaltAndPepper(amount=0.02, salt_fraction=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 13.376 ms | 1.665 ms | 0.231 ms | 8.0× | 57.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![SaltAndPepper: original and augmented Z slices](docs/augmentations/previews/SaltAndPepper-z.webp)](docs/augmentations/previews/SaltAndPepper-z.webp) | [![SaltAndPepper: original and augmented Y slices](docs/augmentations/previews/SaltAndPepper-y.webp)](docs/augmentations/previews/SaltAndPepper-y.webp) | [![SaltAndPepper: original and augmented X slices](docs/augmentations/previews/SaltAndPepper-x.webp)](docs/augmentations/previews/SaltAndPepper-x.webp) |

</details>

<details>
<summary><strong>Sharpen</strong></summary>

Enhance local edges with a discrete Laplacian sharpening filter. This example keeps overshoots instead of clipping them.

```python
A.Sharpen(strength=0.25, p_clamp=0)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 7.270 ms | 1.390 ms | 0.252 ms | 5.2× | 28.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Sharpen: original and augmented Z slices](docs/augmentations/previews/Sharpen-z.webp)](docs/augmentations/previews/Sharpen-z.webp) | [![Sharpen: original and augmented Y slices](docs/augmentations/previews/Sharpen-y.webp)](docs/augmentations/previews/Sharpen-y.webp) | [![Sharpen: original and augmented X slices](docs/augmentations/previews/Sharpen-x.webp)](docs/augmentations/previews/Sharpen-x.webp) |

</details>

<details>
<summary><strong>SheetCompression</strong></summary>

Use the CT intensity profile to compress sheet-like structures along a selected axis while transforming aligned targets together.

```python
A.SheetCompression(strength=0.2, intensity_sigma=1, smoothing=2, axes='z')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 84.264 ms | 3.213 ms | 1.557 ms | 26.2× | 54.1× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![SheetCompression: original and augmented Z slices](docs/augmentations/previews/SheetCompression-z.webp)](docs/augmentations/previews/SheetCompression-z.webp) | [![SheetCompression: original and augmented Y slices](docs/augmentations/previews/SheetCompression-y.webp)](docs/augmentations/previews/SheetCompression-y.webp) | [![SheetCompression: original and augmented X slices](docs/augmentations/previews/SheetCompression-x.webp)](docs/augmentations/previews/SheetCompression-x.webp) |

</details>

<details>
<summary><strong>ShotNoise</strong></summary>

Sample Poisson photon counts from nonnegative intensities and rescale them, producing signal-dependent shot noise.

```python
A.ShotNoise(photons=100)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 62.775 ms | 2.459 ms | 1.269 ms | 25.5× | 49.5× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ShotNoise: original and augmented Z slices](docs/augmentations/previews/ShotNoise-z.webp)](docs/augmentations/previews/ShotNoise-z.webp) | [![ShotNoise: original and augmented Y slices](docs/augmentations/previews/ShotNoise-y.webp)](docs/augmentations/previews/ShotNoise-y.webp) | [![ShotNoise: original and augmented X slices](docs/augmentations/previews/ShotNoise-x.webp)](docs/augmentations/previews/ShotNoise-x.webp) |

</details>

<details>
<summary><strong>SliceIllumination</strong></summary>

Reduce brightness across selected slices and wider bands to simulate slice-dependent acquisition defects.

```python
A.SliceIllumination(defects=3, width=4, reduction=0.5, base_p=0.1, base_reduction=0.5)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.243 ms | 1.227 ms | 0.180 ms | 1.0× | 6.9× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![SliceIllumination: original and augmented Z slices](docs/augmentations/previews/SliceIllumination-z.webp)](docs/augmentations/previews/SliceIllumination-z.webp) | [![SliceIllumination: original and augmented Y slices](docs/augmentations/previews/SliceIllumination-y.webp)](docs/augmentations/previews/SliceIllumination-y.webp) | [![SliceIllumination: original and augmented X slices](docs/augmentations/previews/SliceIllumination-x.webp)](docs/augmentations/previews/SliceIllumination-x.webp) |

</details>

<details>
<summary><strong>Smear</strong></summary>

Blend shifted copies of preceding slices into each slice to simulate directional smearing.

```python
A.Smear(shift=(3, 0), alpha=0.25)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4.304 ms | 1.796 ms | 0.681 ms | 2.4× | 6.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Smear: original and augmented Z slices](docs/augmentations/previews/Smear-z.webp)](docs/augmentations/previews/Smear-z.webp) | [![Smear: original and augmented Y slices](docs/augmentations/previews/Smear-y.webp)](docs/augmentations/previews/Smear-y.webp) | [![Smear: original and augmented X slices](docs/augmentations/previews/Smear-x.webp)](docs/augmentations/previews/Smear-x.webp) |

</details>

<details>
<summary><strong>StripeMasking</strong></summary>

Erase periodic slabs along selected axes with random offsets; overlapping slabs form a stripe pattern through the volume.

```python
A.StripeMasking(unit_size=(16, 16, 16), ratio=0.25, axes='zyx')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 39.608 ms | 1.350 ms | 0.274 ms | 29.3× | 144.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![StripeMasking: original and augmented Z slices](docs/augmentations/previews/StripeMasking-z.webp)](docs/augmentations/previews/StripeMasking-z.webp) | [![StripeMasking: original and augmented Y slices](docs/augmentations/previews/StripeMasking-y.webp)](docs/augmentations/previews/StripeMasking-y.webp) | [![StripeMasking: original and augmented X slices](docs/augmentations/previews/StripeMasking-x.webp)](docs/augmentations/previews/StripeMasking-x.webp) |

</details>

<details>
<summary><strong>ThickSlice</strong></summary>

Downsample and upsample along selected axes to simulate thicker acquisition slices. Labels follow the same slice mapping.

```python
A.ThickSlice(scale=0.5, axes='z')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.996 ms | 2.593 ms | 0.996 ms | 0.8× | 2.0× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ThickSlice: original and augmented Z slices](docs/augmentations/previews/ThickSlice-z.webp)](docs/augmentations/previews/ThickSlice-z.webp) | [![ThickSlice: original and augmented Y slices](docs/augmentations/previews/ThickSlice-y.webp)](docs/augmentations/previews/ThickSlice-y.webp) | [![ThickSlice: original and augmented X slices](docs/augmentations/previews/ThickSlice-x.webp)](docs/augmentations/previews/ThickSlice-x.webp) |

</details>

<details>
<summary><strong>Transpose</strong></summary>

Permute spatial axes for all aligned targets. This example exchanges Y and X.

```python
A.Transpose(order='zxy')
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1.936 ms | 2.529 ms | 0.854 ms | 0.8× | 2.3× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Transpose: original and augmented Z slices](docs/augmentations/previews/Transpose-z.webp)](docs/augmentations/previews/Transpose-z.webp) | [![Transpose: original and augmented Y slices](docs/augmentations/previews/Transpose-y.webp)](docs/augmentations/previews/Transpose-y.webp) | [![Transpose: original and augmented X slices](docs/augmentations/previews/Transpose-x.webp)](docs/augmentations/previews/Transpose-x.webp) |

</details>

<details>
<summary><strong>UnsharpMask</strong></summary>

Add a scaled difference between the image and a Gaussian-smoothed copy to enhance fine structure.

```python
A.UnsharpMask(strength=0.5, sigma=1)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 21.419 ms | 1.636 ms | 0.526 ms | 13.1× | 40.7× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![UnsharpMask: original and augmented Z slices](docs/augmentations/previews/UnsharpMask-z.webp)](docs/augmentations/previews/UnsharpMask-z.webp) | [![UnsharpMask: original and augmented Y slices](docs/augmentations/previews/UnsharpMask-y.webp)](docs/augmentations/previews/UnsharpMask-y.webp) | [![UnsharpMask: original and augmented X slices](docs/augmentations/previews/UnsharpMask-x.webp)](docs/augmentations/previews/UnsharpMask-x.webp) |

</details>

<details>
<summary><strong>Vignetting</strong></summary>

Attenuate intensities with a radial falloff in all three dimensions, darkening voxels farther from the center.

```python
A.Vignetting(strength=0.5, radius=0.8)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 30.211 ms | 1.199 ms | 0.167 ms | 25.2× | 180.8× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![Vignetting: original and augmented Z slices](docs/augmentations/previews/Vignetting-z.webp)](docs/augmentations/previews/Vignetting-z.webp) | [![Vignetting: original and augmented Y slices](docs/augmentations/previews/Vignetting-y.webp)](docs/augmentations/previews/Vignetting-y.webp) | [![Vignetting: original and augmented X slices](docs/augmentations/previews/Vignetting-x.webp)](docs/augmentations/previews/Vignetting-x.webp) |

</details>

<details>
<summary><strong>ZoomBlur</strong></summary>

Average multiple 3D zoom states around a center point to create radial blur.

```python
A.ZoomBlur(strength=0.15, samples=9)
```

| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| 235.889 ms | 3.011 ms | 1.942 ms | 78.4× | 121.4× |

| Z slice · YX plane | Y slice · ZX plane | X slice · ZY plane |
| :---: | :---: | :---: |
| [![ZoomBlur: original and augmented Z slices](docs/augmentations/previews/ZoomBlur-z.webp)](docs/augmentations/previews/ZoomBlur-z.webp) | [![ZoomBlur: original and augmented Y slices](docs/augmentations/previews/ZoomBlur-y.webp)](docs/augmentations/previews/ZoomBlur-y.webp) | [![ZoomBlur: original and augmented X slices](docs/augmentations/previews/ZoomBlur-x.webp)](docs/augmentations/previews/ZoomBlur-x.webp) |

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
| 1.008 ms | 0.048 ms | 0.050 ms | 21.0× | 20.2× |

[![StripGeometry: original and augmented 2D views](docs/augmentations/previews/StripGeometry.webp)](docs/augmentations/previews/StripGeometry.webp)

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
| 0.512 ms | 0.280 ms | 0.268 ms | 1.8× | 1.9× |

[![SurfacePerturb: original and augmented 2D views](docs/augmentations/previews/SurfacePerturb.webp)](docs/augmentations/previews/SurfacePerturb.webp)

2D-only companion: synthetic coordinate surface with CT-derived height. Preview is displacement magnitude, not an anatomical surface.

</details>
