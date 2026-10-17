# Native augmentation

`vc_reader.augment` runs native augmentation in C++20, with a CPU default and an
optional CUDA backend. Python constructs the
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

## Normalization

`A.Normalize(mode="zscore")` is available on CPU and CUDA. It replaces the
removed `normalization=` argument on the four Zarr crop-read methods.
Read/cast first, add a channel dimension, then use normalization anywhere in
an augmentation pipeline:

```python
pipeline = A.Pipeline([
    A.Normalize("minmax"),
    A.GaussianNoise(.02),
    A.MedianFilter(3),
    A.Normalize("zscore"),
], backend="auto")
image = reader.read(start, shape, dtype=np.float32)[None]
result = pipeline({"image": image}, seed=123, inplace=True)
```

`"minmax"` uses `(x-min)/(max-min)`, including zeros and fill values.
`"zscore"` uses `(x-mean)/std` over nonzero voxels, with population standard
deviation (`ddof=0`), and preserves all zeros. Constant channels and channels
with no nonzero values become zero. Both modes retain protection against
extreme finite float64 values; they introduce no epsilon or clipping.
Nonfinite image values raise, including values produced by an earlier step.

Further modes cover the vesuvius trainer normalizations (all on CPU and CUDA):

| Call | Definition |
| --- | --- |
| `Normalize("zscore", nonzero=False, eps=1e-8)` | `(x-mean)/max(std, eps)` over all voxels (vesuvius `normalize_zscore`, tracer `crop_zscore_v1` with `eps=1e-6`) |
| `Normalize("robust", lower=1, upper=99, clip=True)` | clip to percentiles, `(x-median)/(1.4826 MAD)` with vesuvius `normalize_robust` fallbacks (ink `robust_mad`) |
| `Normalize("percentile", lower=, upper=)` | clip to percentiles, `(x-p_lower)/(p_upper-p_lower)`; zero span gives zeros (ink percentile modes; `lower=0, upper=100` is ink `minmax`) |
| `Normalize("percentile", median=True)` | clip, `(x-median)/((p_upper-p_lower)/2)` (ink `robust_percentile_span`) |
| `Normalize("fixed", center=, scale=, bounds=(lo, hi))` | `(clip(x, lo, hi)-center)/scale` with constants: vesuvius `ct`, ink `clip_zscore`/`divide`, `/255*2-1` as `center=scale=127.5` |

`masked=True` (all modes except `fixed`) restricts statistics to the image's
validity mask, like vesuvius `use_mask` or statistics computed before padding.
Without a mask, or with no valid voxel, every voxel is used. All voxels are
transformed; padding is then refilled from the mask as usual. Percentiles use
NumPy's default linear interpolation; the robust, percentile and fixed modes
replace nonfinite results with zero (NumPy `nan_to_num`), and every mode
rejects nonfinite inputs. An option belonging to another mode raises.

These modes compute statistics in float64 rather than NumPy's float32 pairwise
sums, so results agree with the vesuvius functions within float32 rounding,
not bitwise. CUDA computes percentiles from a device radix sort of the selected
values with no host readback; a 128³ float32 crop takes about 1 ms (robust 1.3 ms)
on the RTX 5090, versus 15–60 ms on one CPU core.

Statistics use the current values of each selected channel independently;
`on`, `channels`, `p` and `p_channel` follow the usual image-transform rules.
Labels and other unselected targets are preserved. Validity-mask restoration
uses the same target fill rules as other intensity transforms. Input images
must be channel-first float32/float64, with nonempty 2D or 3D spatial grids.
The reader retains support for arbitrary-rank and empty crops independently.

The CPU implementation retains the reader's fixed segment boundaries and
ordered double-precision reductions, without a reader thread-pool dependency.
It decodes any CPU palette representation before computing statistics. CUDA
keeps images and normalization statistics resident through the chain, using
fixed parallel reductions and a single deferred finite-status check at the
pipeline boundary. CUDA FMA/reduction rounding is tolerated; precision and
zero-mask semantics are unchanged. Pickled pipelines and replay plans work
on both backends; replay recalculates statistics from the supplied image.

## Compact intensity and median chains

`Pipeline(..., compact=True)` is the default. When Gamma or median encounters
a float32 image whose channels each contain at most 256 distinct values, the
pipeline can keep uint8 indices and a float32 palette for each channel.
No rounding or clipping is needed to encode those values. Subsequent median,
Flip, Rotate90, Transpose, BrightnessMultiply, BrightnessAdd, Contrast, Gamma,
and Invert can continue in that representation. Transform order, probabilities,
channel selection and random parameter streams are unchanged.

Gaussian blur, interpolation, noise, CropPad, ThickSlice, slice illumination
and other unsupported transforms cause a decode before sampling/execution.
The public input and output remain float32; this does not add uint8 image input
or change normalization. Float64 images and images with an associated validity
mask use the ordinary path. Standalone Contrast, Invert, brightness and
permutations do not pay the conversion cost: they use compact storage only if
an earlier Gamma or median has already created it.

`MedianFilter(..., quantize=True)` can also start a compact chain after
deformation. That existing opt-in quantization still applies only to selected
channels and only when median is actually applied. The default median remains
exact. Median reorders the palette before rank selection, so inversion and
negative brightness factors are supported.

Mean and sample standard deviation are weighted by the number of voxels at
each palette entry. This changes reduction order and can produce small
floating-point rounding differences; it is not a universal bit-exact promise.
Use `compact=False` to retain the ordinary reduction order or compare timings.
Both modes support replay, pickling, NumPy/Torch containers and `inplace=True`.
When profiling, conversion is charged to the entering transform, and decoding
to the following ordinary transform or the final transform at pipeline exit.

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

Binary labels can already use bool or uint8 (one byte per voxel). Use a signed
dtype such as int8 when a `-1` ignore value must be represented, and ensure all
label/fill values fit the chosen dtype. There is no automatic narrowing or bit
packing. Smaller label storage did not improve Elastic latency in the
[measured dtype comparison](../benchmarks/cpu_optimization_results.md#follow-up-high-cost-kernels-and-binary-label-dtypes).

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

GaussianNoise and RicianNoise now consume both Box–Muller outputs. Their seeded
noise differs from builds using one output per draw; old noise plans store seeds
and therefore also produce the new sequence when applied by this build. Repeated
calls, shared-channel noise, and saved/reloaded plans remain deterministic within
the build. Parameter sampling and Elastic control-field draws retain their prior
RNG sequence.

Float32 images use float32 buffers and convolution arithmetic for GaussianBlur,
LocalSmoothing and Sharpen, float32 selection buffers for MedianFilter, and
float32 powers for Gamma/LocalGamma. Gamma also uses a float32 working buffer;
its statistics still accumulate in double. Noise's Box–Muller math follows image
precision. Float64 images retain double-precision computation. Median filtering
selects exact input values by default; other float32 paths introduce small rounding
differences from the previous double intermediates.

Double remains where it has a numerical purpose: Gaussian coefficient
normalization, intensity ranges and statistics, local-field construction,
geometric coordinates/interpolation and Elastic normalization. Rician squared
magnitudes use double to avoid float32 intermediate overflow. Filters fall back
to double for extreme image values/strengths, and gamma powers do so when
conversion of a normalized base or exponent risks underflow, or a large exponent
amplifies rounding near one. See the
[CPU optimization measurements](../benchmarks/cpu_optimization_results.md) for
real-CT errors, timings and reproduction commands.

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

### Elastic deformation

`Elastic` now uses a mean-centered cubic B-spline displacement field on CPU.
The dense Gaussian implementation has been removed from the production transform.
It supports 2D and 3D crops, including non-cubic shapes, using the same field for
images, labels, points, and normals. Image resampling remains linear; cubic
B-splines describe the displacement field, not the image interpolation.

The constructor keeps its argument names, with these new meanings:

- `sigma`: half the control-knot spacing in voxels; spacing is `max(1, 2*sigma)`.
  Larger values give broader, smoother bends. The default range remains `(5, 10)`.
- `magnitude`: requested RMS **vector** displacement in voxels after removing
  global translation. The default range remains `(1, 5)`. Zero gives identity.
- Each range is sampled once per transform. A conservative displacement-gradient
  bound of 0.8 reduces strength when necessary to keep the mapping injective
  inside the crop. Requested strength is consequently an upper bound.
- `border` and `p` retain their existing meanings.

For the strongest B-spline setting shown in the Paris 4 animated comparison:

```python
A.Elastic(sigma=32, magnitude=8)
```

This intentionally changes the distribution and seeded outputs of existing
Elastic configurations: the old magnitude was a per-component maximum scale,
not RMS vector displacement. Review strength settings when resuming training.
Previously saved replay plans contain their actual displacement fields and
continue to replay without regenerating a field.

See the [CPU measurements and validation](../benchmarks/elastic_production_results.md).

### Coverage

| Existing transforms | Native API |
| --- | --- |
| Compose, Random, OneOf, timing wrappers | `Pipeline`, `p=`, `OneOf`, `profile=True` |
| Normalization | `Normalize` (`minmax`, zero-preserving population `zscore`) |
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
is used for continuous targets; no cubic interpolation mode is supplied. Working
precision follows the rules above. Specific conventions:

- `GaussianBlur(sigma=..., truncate=6, boundary="mirror")` uses an odd filter
  width near `truncate * sigma`: radius is `floor((truncate*sigma + .5)/2)`.
  Here truncate describes the **full width**, following the existing blur, not
  SciPy's radius. `mirror` excludes the repeated edge pixel (Torch's reflect
  padding); `reflect` includes it. Small arrays support repeated reflection.
- `MedianFilter` encodes channels with at most 256 distinct values, including
  normalized uint8 CT, using an exact palette. 3×3×3 filters select ranks with
  a generated odd-even merge selection network, processing 64 byte codes or
  32 continuous values at a time. Maximum padding values are eliminated during
  schedule generation, and only comparisons needed for the median are kept.
  Other palette windows use a sliding histogram; other continuous
  windows use the general selector. `MedianFilter(size=(2, 8), quantize=True)`
  additionally lets higher-cardinality channels use palette rank selection
  after rounding to 256 evenly spaced levels between that channel's current
  minimum and maximum. Codes stay
  uint8 internally; outputs retain the input float32/float64 dtype and range.
  This is opt-in and can be used after Elastic without moving the filter.
  Monotone quantization commutes with rank selection (also for even windows):
  error versus the exact median is at most `(max - min) / 510`, plus output
  floating-point rounding. Already-small palettes remain exact; size 1 is an
  identity. Unrepresentable ranges use the exact fallback. Outliers widen the
  quantization step because the range is not clipped. The option does not
  change parameter sampling, and replay remains deterministic within a build.
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

## Optional CUDA backend

All 24 primary transforms from `bench_augment.py` have CUDA implementations:
BrightnessMultiply, BrightnessAdd, Contrast, Gamma, Invert, GaussianNoise,
RicianNoise, GaussianBlur, MedianFilter, Sharpen, BrightnessGradient,
LocalContrast, LocalGamma, LocalSmoothing, BlankRectangle, SliceIllumination,
Smear, Flip, Rotate90, Transpose, Elastic, LowResolution, ThickSlice, and
RandomBinaryOperator. `Normalize` also supports both backends. Each has its own `.cu` file under `cpp/augment/cuda/`.
Small shared files provide interpolation, convolution, noise generation,
channel adaptation, and device-buffer ownership. `OneOf` can compose these
transforms. Other transforms remain available on CPU; selecting an unsupported
transform with CUDA raises at pipeline construction.

```python
from vc_reader import augment as A
from vc_reader.augment.cuda import is_available

pipeline = A.Pipeline([A.GaussianNoise(.02), A.MedianFilter(3), A.Elastic(3, 1)],
                      backend="auto")
print(is_available(), pipeline.backend)  # Resolved backend: "cpu" or "cuda".
output = pipeline({"image": image, "mask": labels}, seed=123)
```

- `backend="cpu"` remains the default and does not query CUDA.
- `backend="cuda"` requires a CUDA-enabled build and visible device.
- `backend="auto"` chooses CUDA when available and CPU otherwise. It is an
  availability choice, **not a prediction of which backend is faster**.
- Inputs and outputs are still NumPy arrays or CPU Torch tensors. CUDA tensors
  are not accepted. Calls are synchronous and preserve inputs by default.
  Profiling includes host/device transfers, allocation, and synchronization.
- CUDA uses dense buffers, regardless of `compact`. Each call uploads images
  and labels once, keeps them on the GPU across the entire chain, and
  downloads the final outputs once. Images keep their float32/float64 dtype
  throughout; the CPU uint8 palette optimization does not split CUDA chains.
  Integer labels retain their dtype, and binary morphology uses uint8 scratch.
  Local fields, Elastic field expansion, statistics, and validity restoration
  run on the GPU. Scratch buffers use
  CUDA's stream-ordered memory pool where supported, with a portable allocator
  fallback. Buffers belong to the invocation; concurrent calls do not share data.
- CPU parameter draws preserve the existing RNG sequence. Small coefficients
  and control lattices are uploaded as needed. Overflow protection, explicit
  median quantization, and integer-label validation can read small control
  values back; they do not download images. Calling `pipeline.sample()` also
  downloads sampled dense fields so replay plans remain serializable.
  Dense vector, normal and local coordinate fields are mapped on the GPU with
  the CPU's Jacobian/inverse rules (central differences of the displacement,
  Newton inversion for coordinates). Sparse points and vectors supplied as host
  arrays keep the CPU geometry; supplied as device arrays they stay resident.
- Seeds, probabilities, channel selection, target routing, interpolation rules,
  float32/float64 precision, integer label IDs, and replay are preserved.
  Replay is exact within the same build/backend/device. CUDA transcendental
  functions can round differently from CPU libm, so cross-backend image outputs
  need not be bitwise identical. Regression checks use float32 `atol=3e-7`,
  `rtol=2e-6`, and float64 `atol=2e-13`, `rtol=2e-12`; integer outputs are exact.
  CUDA permits FMA contraction and uses deterministic parallel reductions;
  fast math and reduced precision are not enabled. Validity interpolation
  checks every nonzero-weight contributor directly on both backends, avoiding
  false holes caused by comparing a rounded sum with exactly one.
- The 3D size-3 median shares the CPU selector schedule. Other sizes use a
  general exact rank filter with bounded scratch storage, supporting sizes
  1–63 and even windows. Wider windows are not performance-tuned. Equal-valued
  signed-zero ties may select a different zero sign between backends.
- CUDA errors are propagated; a failed call is never silently retried on CPU.
  macOS and other CPU-only systems retain the existing build and execution path.

Build with an **already installed** CUDA toolkit, C++20 compiler, nanobind,
NumPy, and Python development headers. CUDA is off by default, and CPU builds
have no CUDA toolkit/runtime dependency. From the repository root:

```bash
export AGENTS_AGENT_MODE=1
cmake -S vc_reader -B /tmp/vc-augment-cuda \
  -DVC_READER_BUILD_READER=OFF -DVC_READER_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/vc-augment-cuda --target _augment -j 4
```

`120` is the architecture used for the tested RTX 5090; choose the appropriate
CMake CUDA architecture for the target GPU. No packages are installed by these
commands. To test this build without installing it (the paths below match this
workspace's Python 3.14 environment; adjust for your interpreter):

```bash
export PYTHONPATH=vc_reader/src:vc_reader/.venv/lib/python3.14/site-packages
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
  /tmp/vc-augment-cuda/_augment.cpython-314-x86_64-linux-gnu.so \
  -m pytest vc_reader/tests -q
```

GPU regression tests skip on CPU-only builds/devices; backend selection and CPU
fallback remain tested. Tests cover every benchmark transform in float32 and
float64, replay/pickle, 2D/3D and noncubic grids, multi-channel inputs, noncontiguous
arrays, validity masks, integer label widths (including IDs above 2**53),
probabilities, compositions, and overflow-protected float32 intermediates.
See [public CUDA measurements](../benchmarks/README.md#public-cuda-augmentation-benchmark)
for the real CT workload, comparison commands, and performance limitations.
