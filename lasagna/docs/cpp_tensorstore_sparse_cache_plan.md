# C++ TensorStore Sparse Cache Plan

## Goal

Replace Lasagna's current Python-side sparse zarr chunk reader with a high
performance C++/CUDA extension that owns the full cold-read path:

1. determine missing chunks from a GPU query mask,
2. batch TensorStore reads in C++,
3. use TensorStore's in-memory chunk cache and C++ concurrency,
4. stage results in pinned host memory,
5. upload batches to GPU,
6. update the existing CUDA `chunk_table` pointer layout used by the sampler.

The Python optimizer should keep doing optimization/model work, but should not
loop over missing chunks and submit one Python future per chunk.

## Current Implementation

Current code:

- `lasagna/sparse_cache.py`
- `lasagna/sparse_grid_sample_3d_u8.py`
- `lasagna/sparse_grid_sample_3d_u8_kernel.cu`
- `lasagna/sparse_grid_sample_3d_u8_diff.py`
- `lasagna/sparse_grid_sample_3d_u8_diff_kernel.cu`

Current cache structure:

- Logical cache chunk size: `32^3`.
- Stored GPU chunk size: `34^3`, with a 1-voxel margin on all sides.
- One `SparseChunkGroupCache` per zarr group.
- `chunk_table`: CUDA `int64[cZ,cY,cX]`.
- Each entry is a raw device pointer to `uint8[C,34,34,34]`, or `0` if not
  resident.
- Loaded chunk batches are kept alive by Python in `self._batches`.
- Sampling kernels already consume only `chunk_table`, channel count, grid,
  origin, and inverse scale.

Current bottleneck:

- Missing chunk discovery is mostly GPU-side and fine.
- Cold reads are Python `ThreadPoolExecutor` tasks calling `zarr`.
- Each chunk is handled as a separate Python future.
- 4D channel groups loop over channels in Python.
- CPU read/decode/cache behavior is delegated to zarr and OS page cache.

## TensorStore Cache Pool Semantics

TensorStore's `cache_pool` is an in-memory cache pool controlled through a
`tensorstore::Context`.

Important behavior:

- `cache_pool.total_bytes_limit` sets the memory budget.
- To actually reuse cached data, `recheck_cached_data` must not force a fresh
  validation on every read. Use `"open"` for this workload unless live-mutating
  zarrs must be observed during a run.
- `data_copy_concurrency` and `file_io_concurrency` are C++-side concurrency
  controls and should be configured in the same shared context.
- The context must be shared by all TensorStore arrays belonging to the same
  volume/run. Creating a separate context per read defeats the cache.

Example context shape:

```json
{
  "cache_pool": {
    "total_bytes_limit": 8589934592
  },
  "data_copy_concurrency": {
    "limit": 8
  },
  "file_io_concurrency": {
    "limit": 16
  }
}
```

Example zarr spec option:

```json
{
  "driver": "zarr",
  "kvstore": {
    "driver": "file",
    "path": "/path/to/group.zarr"
  },
  "recheck_cached_data": "open"
}
```

## Proposed C++ Extension

Create a new PyTorch extension, for example:

- `lasagna/sparse_tensorstore_cache.py`
- `lasagna/sparse_tensorstore_cache_ext.cpp`
- `lasagna/sparse_tensorstore_cache_ext.cu`

The extension should expose an opaque C++ cache object through pybind11.

### Python Surface API

Target Python usage:

```python
cache = TensorStoreSparseChunkGroupCache(
    channels=["grad_mag", "nx", "ny"],
    zarr_path="/path/to/group.zarr",
    channel_indices={"grad_mag": 0, "nx": 1, "ny": 2},
    is_3d_zarr=False,
    vol_shape_zyx=(Z, Y, X),
    device=torch.device("cuda:0"),
    chunk_size=32,
    cache_pool_bytes=8 << 30,
    file_io_threads=16,
    data_copy_threads=8,
)

cache.prefetch(xyz_fullres, origin, spacing)
cache.sync()
sampled = cache.grid_sample(xyz_fullres, origin_t, inv_scale_t, diff=False)
```

This should preserve the existing `SparseChunkGroupCache` interface closely
enough that `fit_data.py` and `optimizer.py` need minimal changes.

### C++ Object State

Each C++ cache object owns:

- TensorStore context.
- One opened TensorStore array for the group.
- Channel list and channel index mapping.
- Volume shape and chunk grid.
- CUDA `chunk_table` tensor or equivalent device allocation.
- Host-side resident bitset for chunks.
- Host-side pending/in-flight bitset for chunks.
- GPU-side missing chunk mask scratch buffers if needed.
- Pinned host staging buffers.
- Device chunk batches retained for lifetime of cache.
- Thread pool / async queue for TensorStore reads if TensorStore futures alone
  are not enough for scheduling.

Keep the existing `chunk_table` ABI if possible so the current CUDA sampling
kernels can stay unchanged.

## Batched Prefetch Design

### Step 1: Determine Needed Chunks

Preferred implementation:

- Run a CUDA kernel that maps `xyz_fullres` points to chunk IDs.
- Write a dense bool/byte needed mask `needed[cZ,cY,cX]`.
- Apply 26-neighborhood dilation on GPU.
- Compare against resident and pending masks.
- Produce a compact missing chunk list.

Options for compaction:

1. Use `torch.nonzero` in Python initially, passing the compact list into C++.
   This is acceptable only if it is not a hot bottleneck.
2. Better: implement CUB/Thrust compaction in the C++/CUDA extension and return
   only a count to Python.

Recommendation: implement compaction in C++/CUDA in the first C++ version. The
point of this change is to remove Python per-chunk overhead.

### Step 2: Submit Batched TensorStore Reads

For each missing logical chunk `(cz,cy,cx)`, read a padded region:

- logical origin:
  - `z0 = cz * 32 - 1`
  - `y0 = cy * 32 - 1`
  - `x0 = cx * 32 - 1`
- padded size: up to `34x34x34`, clamped at volume boundaries.

For 3D zarr:

- read `[z0:z1, y0:y1, x0:x1]`
- place into channel `0` of `uint8[C,34,34,34]`

For 4D zarr:

- prefer one TensorStore read over all needed channel indices if the channels
  are contiguous.
- if channel indices are not contiguous, either:
  - read the channel span and gather selected channels in C++, or
  - issue one read per selected channel into the same pinned chunk buffer.

Recommendation: read the channel span when possible. Most Lasagna groups should
be compact channel groups, and one read is better than multiple small reads.

### Step 3: Use TensorStore Cache Pool

Open all TensorStore arrays with a shared context:

- one context per Lasagna volume/run,
- not one context per group if groups share the same backing store and IO pool
  budget should be global,
- at minimum one context per `TensorStoreSparseChunkGroupCache`.

Config knobs:

- `cache_pool_bytes`, default initially `8 GiB`.
- `file_io_threads`, default `16` or hardware-concurrency bounded.
- `data_copy_threads`, default `8`.
- `recheck_cached_data`, default `"open"`.

Expose these in the Lasagna JSON under a sparse cache section, e.g.

```json
{
  "sparse_cache": {
    "backend": "tensorstore_cpp",
    "cache_pool_bytes": 8589934592,
    "file_io_threads": 16,
    "data_copy_threads": 8,
    "chunk_size": 32
  }
}
```

### Step 4: Batch Upload to GPU

Current Python code stacks all completed chunks into one pinned CPU tensor and
then one GPU tensor. Keep that behavior in C++:

- allocate pinned host batch `uint8[N,C,34,34,34]`,
- fill chunks as TensorStore reads complete,
- allocate CUDA tensor `uint8[N,C,34,34,34]`,
- async copy pinned batch to GPU on a dedicated stream,
- update `chunk_table[cz,cy,cx]` with `base_ptr + i * chunk_bytes`,
- retain the GPU batch in C++ object state so pointers remain valid.

Avoid allocating one CUDA tensor per chunk.

### Step 5: Synchronization Contract

Preserve current optimizer contract:

- `prefetch(...)` starts async work and returns quickly.
- `sync()` waits for reads/uploads submitted by previous prefetches.
- `grid_sample(...)` assumes needed chunks are resident.

The optimizer already calls:

- sync at start of iteration,
- sample/model/loss,
- prefetch for next iteration.

That overlap pattern should remain.

## Sampling Kernels

Keep the existing kernels initially:

- `sparse_grid_sample_3d_u8_kernel.cu`
- `sparse_grid_sample_3d_u8_diff_kernel.cu`

They only require:

- CUDA int64 `chunk_table`,
- `C`,
- grid,
- offset,
- inv_scale.

The new C++ cache object can expose `chunk_table` to Python or expose
`grid_sample(...)` directly and internally call the existing kernel binding.

Recommendation:

1. First version: expose `chunk_table` and reuse current Python wrappers.
2. Second version: move `grid_sample` method into the C++ object to reduce
   Python glue further.

## Missing/Pending Correctness

The new cache must avoid duplicate work:

- resident chunk: skip,
- pending chunk: skip,
- missing chunk newly submitted: mark pending immediately,
- read failure: clear pending, optionally mark failed/empty depending on error.

For local zarr, failure should be loud. Silent zeros are dangerous because they
look like valid empty volume.

For boundary chunks:

- zero-fill the whole padded buffer first,
- copy the clamped TensorStore read into the correct offset.

## Memory Budgeting

There are two independent memory budgets:

1. TensorStore CPU cache pool:
   - compressed/decompressed backend chunks depending on driver behavior,
   - configured by `cache_pool_bytes`.
2. Lasagna GPU sparse cache:
   - append-only `uint8[N,C,34,34,34]` batches,
   - currently no eviction.

Do not confuse these. TensorStore cache pool does not cap GPU memory.

Add logging:

```text
[sparse_cache_cpp] group=normals chunks=... gpu_loaded=... gpu_mib=...
[sparse_cache_cpp] tensorstore cache_pool=8192MiB file_io=16 data_copy=8
[sparse_cache_cpp] iter new=... pending=... read_ms=... upload_ms=...
```

## Build Integration

Likely dependencies:

- PyTorch C++ extension
- CUDA
- TensorStore C++ library

Implementation options:

1. Build with `torch.utils.cpp_extension.load`.
   - Fast iteration.
   - But TensorStore C++ dependency discovery may be awkward.
2. Add a CMake-built extension under `lasagna/`.
   - Better for TensorStore linkage and repeatable builds.
   - Prefer this for the final implementation.

Recommendation: use a CMake-built extension if TensorStore C++ is already
available on target machines. Otherwise create a minimal prototype with
`cpp_extension.load` only after confirming include/library paths.

Do not add install/bootstrap scripts as part of this change. Document the
TensorStore dependency and let environment setup happen separately.

## Benchmark Protocol

Use the existing Lasagna service workload that showed slow startup/prefetch.

Measure:

- initial prefetch wall time,
- first `pred_dt` initial loss time,
- per-iteration cache sync time,
- number of missing chunks per iteration,
- TensorStore read time,
- GPU upload time,
- CUDA sample time,
- total optimizer iteration time.

Report:

- command/config,
- input volume,
- GPU model,
- CPU core count,
- cache pool size,
- file/data copy thread counts,
- cold run and warm run.

Minimum before/after table:

| Metric | Python zarr sparse cache | C++ TensorStore sparse cache |
|--------|---------------------------|------------------------------|
| initial_prefetch ms | | |
| initial_eval.loss.pred_dt ms | | |
| steady sync ms p50/p95 | | |
| steady iteration ms p50/p95 | | |
| chunks loaded total | | |
| GPU cache MiB | | |

## Implementation Steps

1. Add a new backend class without deleting `SparseChunkGroupCache`.
2. Add config selection: `sparse_cache.backend = "python_zarr"` or
   `"tensorstore_cpp"`.
3. Implement C++ object construction and TensorStore open.
4. Implement CUDA needed-mask + compaction.
5. Implement batched TensorStore read into pinned host memory.
6. Implement batched GPU upload and `chunk_table` update.
7. Reuse existing sparse CUDA sampling kernels.
8. Add timings and cache stats.
9. Validate outputs against Python backend on the same query points.
10. Benchmark cold and warm runs.

## Validation

Functional checks:

- For a fixed set of query points, compare sampled channels from:
  - old Python zarr sparse cache,
  - new C++ TensorStore sparse cache.
- Exact equality is expected for uint8 trilinear output after rounding if the
  same chunk data and interpolation are used.
- Test 3D and 4D zarr groups.
- Test boundary chunks at all volume faces.
- Test repeated prefetch does not reload resident chunks.

Performance checks:

- Confirm Python no longer submits one future per chunk.
- Confirm C++ file IO threads are active during cold prefetch.
- Confirm warm prefetch benefits from TensorStore cache pool and/or OS cache.
- Confirm GPU memory growth matches loaded chunk count.

## Open Questions

- Should GPU sparse cache remain append-only, or do we need eviction for long
  runs over large surfaces?
- Should all channel groups share one TensorStore context/cache pool?
  Recommendation: yes, but start with per-cache object if shared ownership is
  simpler.
- Are Lasagna zarr groups always local files for solver use, or do we need S3 /
  HTTPS kvstores in the C++ backend too?
- Is TensorStore C++ already available in the deployment environment, or only
  the Python package?

