# Optional CUDA CT reader

The CUDA backend supports **3-D uint8 CT**: axis-aligned `read`/`read_many`,
integer `read_points`, explicit-point nearest/trilinear `sample`, and single or
batched nearest/trilinear `read_affine`. Outputs are owned device arrays, ready
for CUDA normalization/augmentation or a DLPack consumer such as PyTorch.

```python
import numpy as np
from vc_reader.reader import ZarrReader
from vc_reader import augment as A

pipeline = A.Pipeline([
    A.Normalize("minmax"),
    A.GaussianNoise(.01),
], backend="cuda")

with ZarrReader.open("/path/to/volume.zarr/2") as reader:
    with reader.cuda(cache_bytes=512 << 20, device=0) as gpu:
        patch = gpu.read_affine(
            (128, 128, 128), (.25, .5, .75), np.eye(3, dtype=np.float32),
            coordinate_origin=(9000, 3600, 3600),
        )
        result = pipeline({"image": patch.reshape(1, 128, 128, 128)},
                          seed=123, inplace=True)["image"]
        # No image download between sampling, normalization and augmentation.
        import torch
        tensor = torch.from_dlpack(result)  # shares CUDA storage and ownership
        # result.numpy() is an explicit host copy, if one is needed.
```

`reader.cuda()` requires a CUDA-enabled build and a visible device. The ordinary
reader API remains available without CUDA. `vc_reader.cuda.is_available()` checks
reader availability; `vc_reader.augment.cuda.is_available()` checks augmentation.
No Torch/CuPy runtime dependency is required by the backend.

## Requests and routing

The adapter methods mirror the corresponding reader operations, returning
`DeviceArray` (or a list of arrays for batches). They accept `source=`:

- `"auto"` (default) samples from cached/uploaded source chunks. For axis-aligned
  reads it instead crops uint8 on CPU and uploads the crop if missing source
  chunks occupy more bytes than the raw crop output. Any request exceeding the
  chunk-cache budget uses CPU sampling/cropping followed by upload. Floating box
  conversion happens on GPU. Execution errors are propagated, never retried.
- `"chunks"` requires GPU sampling and sufficient cache budget for the request's
  chunk union. It raises before source loading if that working set exceeds the
  budget. Use this to warm the cache deliberately.
- `"cpu"` explicitly samples/crops on CPU and uploads its result. The returned
  device-array contract is identical.

The routing rule compares transferred bytes; it is not a universal latency
predictor. Fresh, sparse requests can favor CPU processing even when they fit
in the device cache. Use explicit routes when the workload is known.

The private per-adapter LRU cache is bounded by `cache_bytes`, including full
padded source chunks. Native planning deduplicates chunks across crop batches.
Only misses are read through the existing reader (local mmap/slab, codecs,
remote/disk/memory caches) and uploaded. Host staging uses batches of at most
64 MiB, or one chunk if a chunk is larger. Outputs, coordinate buffers and lookup
scratch are additional memory, outside the source-cache budget. CUDA's allocator
may also retain freed memory. No GPU decompression or direct storage I/O is claimed.

`gpu.stats` reports resident cache bytes/budget, cache hits, chunk uploads,
uploaded bytes, and CPU/GPU route counts. `gpu.clear_cache()` drops cached chunks.
Source volumes are treated as immutable; clear the GPU cache after changing data.
Closing the adapter frees its cache and scratch, but leaves returned outputs valid.
The source reader must remain open while issuing new requests.

Examples:

```python
boxes = gpu.read_many([((10, 20, 30), (64, 64, 64)),
                       ((20, 30, 40), (32, 64, 64))], dtype=np.float32)
crops = gpu.read_affine_many([
    ((64, 64, 64), (10.25, 20.5, 30.75), np.eye(3)),
    ((32, 64, 64), (11.25, 21.5, 31.75), np.eye(3)),
])
values = gpu.read_points(np.array([[10, 20, 30], [11, 21, 31]], dtype=np.int32))
interpolated = gpu.sample(np.array([[.25, .5, .75]], dtype=np.float32),
                          coordinate_origin=(9000, 3600, 3600))

# Repeated explicit requests: plan once and keep coordinates on device.
request = gpu.prepare_points(points, order=1)  # integer=True for int32/int64 gathers
values = request.read()
```

Prepared requests own a snapshot of host/device coordinates and the native chunk
plan. They reuse indices even after source eviction; subsequent reads reload only
missing source chunks. Regular planes/crops should use affine geometry instead
of storing dense coordinates. Shape-changing or heterogeneous crop batches return
nonoverlapping views into one contiguous device output allocation. Empty crops,
missing chunks, out-of-volume neighbors and fill values follow CPU semantics.
Integer gathers never convert their indices to floating point.

## Precision

The default **3-D uint8 CPU path and CUDA path use float32** geometry, weights
and interpolation. Absolute caller coordinates are split into an integer origin
and local float32 coordinates before narrowing. Supplying `coordinate_origin`
explicitly avoids constructing large absolute floating coordinates at all.
Already-rounded absolute float32 inputs cannot recover lost precision.

Affine coordinates use separate float32 multiply/add rounding on both backends;
interpolation permits ordinary CUDA FMA. Nearest uses ties-to-even in global
integer space, including odd origins. Compared with the previous float64 default,
coordinates near half-voxel ties can select a different voxel. This is a discrete
sampling change, not a small intensity perturbation. Explicit `dtype=np.float64`
on the CPU preserves the legacy reference for linear interpolation/affine reads;
CUDA accepts uint8 nearest/crop output or float32 output, with linear output
always float32. Other source dtypes keep their existing CPU implementation.

Source dimensions and explicit integer coordinate origins are limited to 2**30;
individual output-axis extents are limited to 2**24, keeping output indices exactly
representable in float32. Integer point values may use the full int32/int64 range;
outside values return fill. Nonfinite explicit coordinates produce NaN for linear
sampling and fill for nearest. Nonfinite affine definitions are rejected.

## Device arrays, pipelines and streams

`DeviceArray.shape`, `.dtype`, `.device`, `.size`, `.nbytes` and `.ndim` describe
storage. `.reshape(...)` returns a view. `.astype(np.float32)` converts uint8 on
GPU. `vc_reader.cuda.asarray(numpy_array, device=0)` uploads host arrays.
`.numpy()` explicitly downloads; implicit NumPy conversion raises.

CUDA pipelines accept a sample whose array targets are all `DeviceArray` values
on the same device. Image/scalar targets must be float32/float64, label and valid
mask dtypes are preserved, and metadata passes through. Use `.astype(np.float32)`
for raw uint8 reader output before normalization. Inputs are preserved by default
using a device copy; `inplace=True` permits modification and avoids that copy.
Always use returned arrays, since spatial operations can replace storage.
Replay/plan recording also accepts device samples. Sparse/directional/local
coordinate targets retain the existing host API and are rejected by the resident
API, which prevents hidden full-image downloads.

Calls use the selected device's default stream and **complete before returning**.
There is no arbitrary producer-stream/asynchronous-call API in this version.
DLPack exports can be consumed on any stream because production is complete.
External writes through shared DLPack views must finish before handing those
arrays back to vc_reader or calling `.numpy()`. Final destruction of exported
storage synchronizes the device to protect work queued by external consumers.
Small validation/statistics flags may transfer to host inside augmentation;
image/label volumes remain resident. GPU outputs remain valid after their reader,
cache or pipeline has been released.

CUDA adapters, prepared requests and arrays cannot be pickled or used after fork.
Construct them inside each worker; the ordinary `ZarrReader` retains its existing
fork/spawn support. Concurrent calls on one CUDA reader serialize cache mutation
and sampling. Different adapters have independent caches.

## Build and validate

Use the existing development environment. `VC_READER_ENABLE_CUDA=OFF` is still
the default; CPU builds require neither the CUDA toolkit nor runtime. CUDA builds
now compile both reader and augmentation support. The build dependency is
nanobind >= 2.10.2 for reusable array-API/DLPack wrappers
([nanobind changelog](https://nanobind.readthedocs.io/en/latest/changelog.html#version-2-10-1-dec-8-2025));
this implementation was tested with nanobind 3.1.0.

Commands used from the repository root, reusing already-present dependencies:

```bash
export AGENTS_AGENT_MODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=vc_reader/src
cmake -S vc_reader -B /tmp/vc-reader-cuda \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DVC_READER_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D="$PWD/vc_reader/build/cp314-cp314-linux_x86_64/_deps/vc_delta3d-src" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/vc-reader-cuda -j 3
vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_augment.cpython-314-x86_64-linux-gnu.so \
  -m pytest -q vc_reader/tests
```

Adjust Python ABI paths, cached codec location and CUDA architecture for the
local environment. Tests that use local HTTP fixtures require localhost socket
access; CUDA tests require a GPU and skip in CPU-only builds. Performance and
memory-check commands/results are in the public-backend benchmark report.
