# Native augmentation

`vc_reader.augment` runs CPU augmentation in C++20. Python constructs the
pipeline and converts NumPy arrays or CPU Torch tensors; parameter sampling,
voxel operations, target routing, and replay run natively with the GIL released.
Transforms are flat configuration structs and overloaded functions, sharing
channel, interpolation, and random-sampling helpers. There is no transform
inheritance hierarchy or string-based operation dispatcher.

```python
from vc_reader import augment as A

augment = A.Pipeline([
    A.Flip(axes="yx"),
    A.Affine(degrees=(-10, 10), scale=(.9, 1.1), p=.2),
    A.GaussianNoise(std=(0, .05), p=.15),
    A.Gamma(gamma=(.7, 1.5), p=.3),
])

# patch is a dict: image=(C,Z,Y,X), mask=(C,Z,Y,X), optional padding_mask.
result = augment(patch, seed=A.seed(42, epoch, patch_index, draw_index))
```

Construct once and reuse. A number fixes a parameter; a two-element tuple
samples uniformly. `A.Uniform`, `A.Normal`, `A.Choice`, and `A.BalancedContrast`
are explicit distributions. `p` controls each transform; image transforms also
accept `p_channel`, `channels`, `on`, and `shared`. `shared=True` shares sampled
parameters across channels within each image. Noise has a separate
`shared_noise=True` option for identical noise realizations across channels.
`OneOf([...], weights=[...], p=...)` selects a child; that child's own `p` still
applies. Arguments and defaults are listed in
[`__init__.pyi`](../src/vc_reader/augment/__init__.pyi).

## Reusing reader buffers

A fresh reader crop can be handed directly to augmentation without another
input copy. The kernels still run in C++; the NumPy view only carries the buffer
and its lifetime between the two modules:

```python
image = reader.read(start, shape, dtype=np.float32)[None]  # add channel axis
result = augment({"image": image}, seed=patch_seed, inplace=True)
image = result["image"]
```

`inplace=True` permits modification of the supplied arrays and lets outputs
share their storage. Always use the returned dictionary: geometric transforms
and shape changes can allocate new buffers. Inputs must be writable, contiguous,
aligned, and nonoverlapping. Keep the default for cached samples or arrays needed
elsewhere. Output views retain their reader buffer, including after the reader
closes, so the output pool cannot recycle a buffer still in use.

For dataset pipelines returning fresh crops:

```python
dataset = AugmentedDataset(dataset, augment, seed=42, inplace=True)
```

The default also avoids unnecessary copies before geometry: inputs are borrowed
for reading and copied on first write. Unchanged outputs are copied on export to
preserve the existing independent-output contract.

## Targets and containers

Arrays are channel-first `(C,Y,X)` or `(C,Z,Y,X)`. Continuous targets require
float32 or float64. Labels retain their integer/bool dtype; continuous outputs
retain their float dtype. NumPy stays NumPy and Torch stays Torch. CUDA tensors,
autograd tensors, non-native byte order, and nonfinite continuous values are
rejected. Noncontiguous inputs are made contiguous. By default, inputs are
preserved and outputs have independent storage. Metadata passes through by reference.

The default schema declares `image`, `mask`, and `padding_mask`; fields may be
absent or `None`. Declare other arrays explicitly so labels and regression
targets cannot accidentally receive intensity augmentation:

```python
augment = A.Pipeline([
    A.Flip(), A.Gamma(p=.3),
], targets={
    "image": A.Image(valid="padding_mask"),
    "ink": A.Label(ignore_value=-1),
    "distance": A.ScalarField(),
    "padding_mask": A.ValidMask(),
    "points": A.Points(order="zyx", outside="keep"),
    "normals": A.NormalField(components="zyx"),
})
```

| Target | Geometry |
| --- | --- |
| `Image` | Linear interpolation; intensity transforms apply here |
| `Label` | Nearest interpolation and exact integer copying |
| `ScalarField` | Linear interpolation, no intensity transforms |
| `ValidMask` | Conservative validity through interpolation |
| `Points` | `(N,D)` coordinates; outside points kept, dropped, or rejected |
| `Vectors` | `(N,D)` directions; affine direction mapping |
| `VectorField` | Dense directions mapped with the inverse sampling Jacobian |
| `NormalField` | Dense normals mapped with the transpose sampling Jacobian and normalized |
| `CoordinateField` | Dense coordinates; `frame="external"` resamples values without changing their coordinate frame |

Spatial dimensions share one grid. A `(3,Y,X)` external coordinate field is
allowed for a surface sampled in 3D. Axis/component strings describe the actual
coordinate order (`yx`, `zyx`, or `xyz` as appropriate). Nonlinear transformations
of sparse vectors are rejected because no positions are available for their
Jacobians. Nonlinear point inversion raises if it fails to converge.

An image's validity mask prevents intensity augmentation from changing padding.
Geometry fills out-of-bounds labels with `ignore_value` (or `fill`), and images
with their configured fill. Geometry preserves int64/uint64 IDs exactly.
Value-based label operations require IDs within ±(2**53 − 1), and reject larger
IDs rather than silently rounding them.

## Replay and reproducibility

```python
geometry = A.Pipeline([A.Flip(), A.Elastic(sigma=5, magnitude=.5, p=.2)])
plan = geometry.sample(patch, seed=123)
aligned = plan.apply(patch)
student = A.Pipeline([A.GaussianNoise(std=.05)])(aligned, seed=456)
```

A plan contains resolved choices, noise seeds, and data-dependent displacement
fields. Replay requires the same array keys, shapes, and dtypes. Use the plan's
`apply` method or the pipeline that created it. Pipelines and plans support
pickle for worker startup and experiment debugging; pickle/plan bytes are
trusted application data, not an untrusted interchange format. `sample` executes
the pipeline on an owned copy to resolve data-dependent operations.

The explicit seed controls a native RNG; NumPy/Torch global RNG state is untouched.
Random sequences differ from the old backend. Identical inputs and seeds replay
exactly within the same build/platform; transcendental math can differ across
platforms. There are no internal worker threads: DataLoader workers provide
parallelism. `profile=True` adds `_aug_perf`, a dict of transform indices to
elapsed seconds.

## Vesuvius datasets

`ZarrDataset` accepts an explicit pipeline. Its existing behavior is unchanged
when `augmentation` is omitted. Target names must match the configured dataset.
The supplied pipeline also owns target processing, including medial surfaces
when a loss needs them. Supply a separate deterministic pipeline for validation.

```python
from torch.utils.data import DataLoader, RandomSampler
from vesuvius.models.datasets import ZarrDataset
from vesuvius.models.augmentation.native import AugmentationSampler

dataset = ZarrDataset(mgr, augmentation=augment, augmentation_seed=42)
sampler = AugmentationSampler(RandomSampler(dataset))
loader = DataLoader(dataset, sampler=sampler, batch_size=2,
                    num_workers=4, persistent_workers=True)
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        train_step(batch)
```

`AugmentationSampler` also wraps `DistributedSampler` and forwards `set_epoch`.
Epoch, sample index, draw occurrence, and optional `view` travel with each index,
so persistent workers receive epoch updates and repeated samples get distinct
draws. Worker scheduling does not affect augmentation for a fixed index stream.
Seed the underlying random/distributed sampler too when reproducing sample order.
Plain integer indexing is deterministic for that index; use the sampler wrapper
to vary augmentation across epochs. To adapt another dictionary-returning
dataset, disable its augmentation and use
`AugmentedDataset(dataset, augment, seed=42)` with the same sampler wrapper.

## Transform coverage and migration

| Existing transforms | Native API |
| --- | --- |
| Compose, Random, OneOf, timing wrappers | `Pipeline`, `p=`, `OneOf`, `profile=True` |
| Brightness, contrast, gamma, inversion, random clipping | `BrightnessMultiply`, `BrightnessAdd`, `Contrast`, `Gamma`, `Invert`, `ClipOutliers` |
| Gaussian/Rician noise, blur, median, sharpening | `GaussianNoise`, `RicianNoise`, `GaussianBlur`, `MedianFilter`, `Sharpen` |
| Local brightness, contrast, gamma, smoothing | `BrightnessGradient`, `LocalContrast`, `LocalGamma`, `LocalSmoothing` |
| Blanking, illumination, smear | `BlankRectangle`, `SliceIllumination`, `Smear` |
| Mirroring, rotation, transpose, cropping | `Flip`, `Rotate90`, `Transpose`, `CropPad` |
| Spatial affine/elastic, sheet compression | `Affine`, `Elastic`, `SheetCompression` |
| Low resolution, thick slice | `LowResolution`, `ThickSlice` |
| Binary operators, connected components | `RandomBinaryOperator`, `RemoveComponents` |
| Closing, remove label, regions, one-hot, image masking | `Morphology`, `ReplaceLabel`, `Regions`, `OneHot`, `MaskImage` |
| Deep supervision, pseudo-2D, skeleton targets | `DeepSupervision`, `ToPseudo2D` / `FromPseudo2D`, `MedialSurface` |
| Strip geometry, surface perturbation, crop translation | `StripGeometry`, `SurfacePerturb`, `translate_crop_box` |

`DeepSupervision` must be last; it returns a list of levels. Pair pseudo-2D
conversion and restoration in one pipeline. `translate_crop_box` operates before
reading a crop and takes only valid supervised `(N,3)` points.

This is a new backend, not a bitwise replacement for the old training pipeline.
Construct the desired probabilities and ranges explicitly. Linear interpolation
is used for continuous targets; no cubic interpolation mode is supplied. Kernels
use double intermediates and cast to the input dtype. Specific conventions:

- `GaussianBlur(sigma=..., truncate=6, boundary="mirror")` uses an odd filter
  width near `truncate * sigma`: radius is `floor((truncate*sigma + .5)/2)`.
  Here truncate describes the **full width**, following the existing blur, not
  SciPy's radius. `mirror` excludes the repeated edge pixel (Torch's reflect
  padding); `reflect` includes it. Small arrays support repeated reflection.
- `MedianFilter` uses a sliding histogram when a channel has at most 256 distinct
  values, including normalized uint8 CT. It stores the exact values; there is no
  quantization. Channels with more distinct values (for example, after noise or
  interpolation), or both signs of zero, use the general selection kernel.
- `LowResolution` uses nearest-exact downsampling and linear upsampling;
  `ThickSlice` uses legacy nearest sampling in both directions.
- `Morphology` defaults to a Euclidean ball. To reproduce the existing 3D
  closing footprint use `Morphology("close", radius=structure_size,
  footprint="repeated_cross")`. For existing 2D closing use a ball with
  `neutral_border=True`. Closing is binary and restores the channel maximum.
- `MedialSurface` uses native Zhang–Suen thinning per YX slice with optional
  morphology. Exact equivalence to every scikit-image version has not been
  established; review these targets before changing an existing experiment.

## Build and validation

The normal package build includes `_augment`. It needs C++20, Python development
headers, CMake, and nanobind. No SciPy, Torch, or image library is needed by the
augmentation extension. NumPy is the Python adapter's runtime dependency.
To build augmentation alone using an existing environment, without fetching the
reader's codec dependency:

```sh
cmake -S vc_reader -B /tmp/vc-augment-build \
  -DVC_READER_BUILD_READER=OFF -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/vc-augment-build --target _augment -j 2
cp /tmp/vc-augment-build/_augment*.so vc_reader/src/vc_reader/
PYTHONPATH=vc_reader/src:vesuvius/src python -m pytest \
  vc_reader/tests/test_augment.py vesuvius/tests/test_native_augmentation.py -q
```

Use a Python interpreter with a matching ABI. Torch/SciPy reference checks skip
when those optional test dependencies are absent. Multiprocess tests need local
IPC sockets. No dependency installation is performed by these commands.

[`validate_augment.py`](../benchmarks/validate_augment.py) reads a real local Zarr
crop through `ZarrReader`, checks geometry against NumPy, checks exact replay and
input ownership, and optionally checks a matching label crop. `--output` saves
before/after arrays as an NPZ. Example from the repository root:

```sh
PYTHONPATH=vc_reader/src python vc_reader/benchmarks/validate_augment.py \
  /path/to/image.zarr /path/to/labels.zarr \
  --start 200 400 --shape 128 128 --output /tmp/augmentation-check.npz
```

Omit the label path for an unlabeled CT volume. Supply three start/shape values
for 3D. The harness is a correctness check, not a throughput benchmark.
