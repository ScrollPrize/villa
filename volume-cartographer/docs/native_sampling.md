# Lasagna channel sampling

Lasagna datasets (normals, fiber predictions, widths, embeddings) are read
through vc_reader. Each channel group's Zarr array is opened once with
`vcr::zarr::openArray` (`openLasagnaGroupArray`); the groups a sampler binds
share one `vcr::MemoryCache`, whose size is the sampler's `maxCachedBytes`.
Remote groups are cached on disk by vc_reader under the dataset's remote cache
root, laid out by URL (`<root>/<scheme>/<bucket>/<path>`), so every tool and
dataset using that root shares the cached chunks. Pooled prediction stores
(`.native_pool.json`) are `vcr::computed` arrays.

Sampling reads the eight trilinear corners of each point with one
`Array::read_points` call per channel and applies the Lasagna decoding in VC:
compact-normal axis tensors, validity-masked corner weights and grad_mag
decoding. Chunks that are not stored read as the array's fill value.

## Python

```python
import numpy as np
from vc.volume import NormalSampler

normal_sampler = NormalSampler("/path/to/dataset.lasagna.json")
points = np.array([[128, 256, 512]], dtype=np.float32)
normals, valid = normal_sampler.sample(points, threads=1)
```

Normal points are base-volume **XYZ**, and returned normals are unit XYZ axes
(sign-ambiguous), with a boolean validity array. Interpolation uses the native
axis-tensor decoder, not interpolation of encoded nx/ny bytes. Zero grad_mag
marks invalid samples. The opt-in manifest setting `mask_invalid_normal_corners`
also excludes invalid corners from interpolation; the default remains unchanged.
The binding owns its dataset and releases the GIL for native sampling.

## Live inference

`FiberPredictionField::setReadinessGate` makes each channel group a computed
array: a chunk is read from the store only after the gate has returned for it,
once per chunk while it stays in the memory cache. The gate runs on the
sampling thread, so a case queue on that thread can switch cases and cancel
the wait. A new gate starts a new session; chunks acknowledged under an
earlier gate are not reused.

## Cache location and authentication

`global_cache_root()` resolves without creating directories:

1. Explicit `{"version": 1, "root": "/absolute/path"}` in
   `$VC3D_CONFIG_DIR/cache.json` (default `$HOME/.VC3D/cache.json`).
2. Existing `/volpkgs`, then `/ephemeral`, using their `remote_cache` child.
3. Caller suggestion, then `$HOME/.VC3D/remote_cache`.

Malformed explicit configuration fails rather than silently falling back.
Remote groups with explicit keys sign `s3://` requests; otherwise public
buckets are read anonymously and other buckets with discovered AWS
credentials when the dataset allows it. A remote manifest is downloaded once under
the cache root, at the path `vcr::zarr::decodedCachePath` gives it.
