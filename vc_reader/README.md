# vc_reader

Find and locate Vesuvius Challenge open data from the published
[`metadata.json`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
(schema: [`metadata.schema.json`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.schema.json)).

The catalog fetches and parses the metadata and answers "which volume / level
/ labels / predictions, and where are they". It uses only the standard library.
`vc_reader.reader` then reads the data. Zarr v2 arrays are supported now
(volumes and volume predictions). Zarr v3 (ink labels), tif and png are next.

## Optional CUDA CT sampling

`reader.cuda()` provides device outputs for affine crops/batches, nearest and
trilinear point sampling, integer gathers, and axis-aligned reads. It reuses a
bounded GPU chunk cache and can feed `augment.Pipeline(..., backend="cuda")`
without an intermediate image download. See the [CUDA reader guide](docs/cuda_reader.md)
for the API, routing, float32 precision contract, DLPack ownership, build commands
and validation. CPU-only builds remain the default.

## Usage

```python
import vc_reader

cat = vc_reader.load()                        # fetched once, cached in ~/.cache/vc_reader

scroll = cat.sample("pherc paris 4")          # also "PHercParis4", "Scroll 1", "paris4"
scroll.find_volumes(resolution=2.399)         # both 2.4um volumes (78 keV and 137 keV)
vol = scroll.volume(resolution=2.399, energy_kev=78)   # or scroll.volume("20260411134726")
vol.url()                                     # https://…/20260411134726-2.400um-0.2m-78keV-masked.zarr/

lvl = vol.level(um=9.6)                       # pyramid level 2
lvl.url()                                     # …masked.zarr/2/
lvl.expected_shape_zyx                        # (18946, 8174, 8174)

for p in lvl.predictions():                   # volume predictions, aligned to this scale
    print(p.artifact.type, p.sublevel, p.url())

for m in vol.ink_labels():                    # labels on this volume, or mappable onto it
    print(m.segment.id, m.relation, m.labels.url(), m.labels.array_url("supervision"))
```

### Ink labels on, or mappable onto, a volume

`Volume.ink_labels()` returns `InkLabelMatch` objects:

- `relation == "direct"`: the labels were published on this volume.
- `relation == "transform"`: the labels are on another volume of the same sample
  that is registered to this one. `m.transform` maps level-0 voxels of the
  labels' volume to level-0 voxels of this one. Lift label pixels to 3D with
  the labels' own mesh (`m.labels.mesh`), then apply `m.transform`.
  `m.coverage` is the segment's recorded overlap with this volume.
  `m.same_frame` is true when the two volumes share a voxel grid (a
  superseded volume and its replacement).

By default each (segment, label set) appears once. The order of preference is:
direct, then same frame, then natively annotated, then lowest volume id. Pass
`dedupe=False` to get every candidate, or `mappable=False` to get direct
matches only. `Sample.ink_labels()` lists every published label set without
any mapping.

### Coordinates and scales

Points are `[x, y, z]` in level-0 voxels. Level N is a 2^N downsample.

```python
lvl.to_level0 / lvl.from_level0               # Affine scale by 2^N / 2^-N
vol.level(2).transform_to(other_vol.level(1)) # voxel at L2 of vol -> voxel at L1 of other
scroll.transform(a, b)                        # level-0 to level-0, from volume_transforms
```

`Affine` is plain Python. `.apply(p)`, `@` (composition), `.inverse()` and
`.to_numpy()` are available.

### Segments on a volume

```python
for view in vol.segments(min_coverage=0.5):   # segments with a mesh on this volume
    view.mesh, view.layers, view.ink_detections, view.ink_3d_projections, view.ink_labels
    view.render_pixel_size_um(view.layers[0]) # pixel_size_um * 2^render_level
```

### Lookup rules

- **Samples**: matching ignores case, punctuation and zero padding. The
  `PHerc` prefix is optional, and the community names `Scroll 1`–`Scroll 5`
  work. A failed lookup suggests close names.
- **Volumes**: look up by id or long id, or by `resolution` (µm). A volume
  whose pixel size is exactly the requested value wins. Otherwise any volume
  within 0.5% relative tolerance (`rel_tol=`) matches. Narrow the result with
  `energy_kev=` or `scan=` (scan id, long id or `scan.name`). Superseded volumes are skipped
  unless you name them or pass `include_superseded=True`.
- **Several volumes at one resolution**: volumes of different scans usually
  cover different regions (Scroll 1's stitched parts 1 and 2, PHerc0139's
  ROI2 and B), so `volume()` never chooses between scans. Pass `scan=` instead.
  `scan.name` is the facility's name for the acquisition, which usually says
  the region: the ESRF `scanRadix` (`SCROLLS_HEL_2.403um_77keV_0.22m_PHerc_0139_ROI2_HA`)
  or the DLS `scan_name` (`PHercParis4 54keV stitched_part_2, …`). It is None
  when the metadata records neither. One
  scan can also have several exports. `vol.is_secondary` marks the
  ones the pipeline no longer renders segments on (`render_segments: false`),
  such as experimental, ROI or redundant exports. `volume()` drops those when
  the same scan has a main export, so PHerc0139 at 9.362 µm gives the main
  volume rather than its `pag0`/`pag50` variants. If more than one volume is
  left, `AmbiguousMatchError` is raised. It lists the candidates with their
  scans and says which argument narrows the match. `find_volumes()` always returns every match.
- **URLs**: `artifact.url()` returns HTTPS and prefers the open-data bucket.
  `url("s3")` returns the `s3://` form. EduceLab-licensed volumes are only on
  `data.aws.ash2txt.org` / `s3://vesuvius-challenge`.

## Reading zarr arrays

`vc_reader.reader` reads zarr v2 arrays and groups (OME-Zarr volumes and their
levels: CT volumes, predictions) from a local directory, `http(s)://` or
`s3://` (public, or with credentials), optionally
through persistent disk caches and an in-memory chunk cache. The reading is
done by a C++ extension (`vc_reader._native`, built by the pyproject with
scikit-build-core and nanobind) on native threads without the GIL. Python only
finds the caches, checks them against the remote `.zarray` and holds VC3D's
lease; optional prefetch scheduling also runs in Python. It needs the `reader`
extra (numpy).

```python
from vc_reader.reader import ZarrReader, MemoryCache

lvl = cat.sample("Scroll 1").volume(resolution=2.4, energy_kev=78).level(um=9.6)
# reuse the RAID cache and VC3D's cache; misses are fetched and appended
r = lvl.open(cache_dir=["/mnt/raid_nvme/volume_cache", "vc3d"], memory_cache=8 << 30)
crop = r.read((8192, 3584, 3584), (128, 128, 128))   # out-of-bounds parts read as fill_value
crops = r.read_many([(start, (64, 64, 64)) for start in starts])  # one parallel read; shared chunks read once
crops = r.read_many(regions, out=crops)               # reuse arrays: no first-touch page faults
block = r[8192:8704, 3584:4096, 3584:4096]
layers = r[[8192, 8194, 8198], 3584:4096, 3584:4096]  # selected layers and spatial slices
r.attrs                                             # array .zattrs, loaded on first access

ZarrReader.open("/data/volume.zarr/2")                # local array
ZarrReader.open("s3://vesuvius-challenge-open-data/…/volume.zarr", level=2)
shared = MemoryCache(16 << 30)                        # one budget for several readers

vals = r.read_points(zyx)                             # (..., 3) ints -> stored values; outside: fill_value
vals = r[z_idx, y_idx, x_idx]                         # numpy integer-array indexing, same path
vals = r.sample(zyx_float)                            # trilinear, float32 (order=0: nearest, as stored)
```

### Oriented crops and oblique slices

`read_affine` samples a regular 3-D output grid in the opened array's voxel
coordinates. Native code generates coordinates as needed and samples the
source chunks directly, without assembling an axis-aligned source block or
a full coordinate array.

```python
import numpy as np

angle = np.deg2rad(30)
c, s = np.cos(angle), np.sin(angle)
axes = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
shape = (64, 128, 128)
center = np.array([9000, 3600, 3600], dtype=np.float64)
origin = center - axes @ ((np.asarray(shape) - 1) / 2)
crop = r.read_affine(shape, origin, axes)             # float32, trilinear
r.read_affine(shape, origin, axes, out=crop)          # reuse final storage
plane = r.read_affine((1, 128, 128), origin, axes)    # singleton depth is a plane
crops = r.read_affine_many([
    (shape, origin, axes),
    (shape, origin + [32, 0, 0], axes),
])
```

The mapping is **`source_zyx = origin + axes @ [output_z, output_y, output_x]`**.
Each column of `axes` is the source-space step for one output axis, including
spacing. `origin` is the first sample, not the centre. Reflections, shears and
anisotropic spacing are supported; axes need not be orthonormal. Coordinates
are relative to the opened pyramid level, not physical units or level 0.
There is no antialiasing filter when downsampling.

For 3-D uint8 CT, default geometry/interpolation uses float32 with integer
global origins; `coordinate_origin=` accepts already-local geometry. Explicit
`dtype=np.float64` retains the CPU reference path. Other source dtypes retain
their existing float64 interpolation. `order=0` uses nearest sampling (ties to even) and retains
the stored dtype by default; an explicit float32/float64 dtype converts it. Outside neighbours and
missing chunks use the array's stored `fill_value`. This does **not** imply
zero padding for arrays with nonzero fill values. Normalize the returned image
with an augmentation pipeline, as described below.

`out` must have the requested dtype and shape and be writable, aligned and
C-contiguous. `read_affine_many` accepts differing shapes and an optional
sequence of output arrays (or `None` entries); buffers must not overlap.
It processes crops in order within one native call, using native threads for
each crop. Chunk loads are shared within each point batch (up to 2²⁰ samples);
reuse across batches and crops depends on the caches. It does not perform
joint chunk planning across all crops.

For tracer frames whose columns are `(u, v, forward)` in **XYZ**, and output
axes `(depth, height, width) = (forward, v, u)`, reorder both rows and columns:

```python
# pos_xyz/frame_xyz and crop offsets are in trace-grid units;
# input_scale converts trace-grid voxels to the opened array's voxels.
axes = input_scale * spacing * frame_xyz[::-1, ::-1]
first_local = np.array([lateral_start, vertical_start, forward_start])
origin = input_scale * (pos_xyz + frame_xyz @ first_local)[::-1]
image = r.read_affine((depth, height, width), origin, axes)
```

### Casting and normalization

`read`/`read_many` return stored values in the stored dtype by default. Affine
reads return float32 for trilinear sampling and the stored dtype for nearest
sampling. All four methods accept explicit `dtype=np.float32` or `np.float64`
conversion; `out` never opts into conversion implicitly. Source data, cached
chunks and `reader.dtype` remain unchanged.

**Normalization belongs to `vc_reader.augment`.** The `normalization=` reader
argument has been removed from `read`, `read_many`, `read_affine`, and
`read_affine_many`. Migrate a normalized read to an explicit floating read
followed by a reusable pipeline:

```python
from vc_reader import augment as A

normalize = A.Pipeline([A.Normalize("zscore")], backend="auto")
image = r.read(start, size, dtype=np.float32)[None]  # (C, Z, Y, X)
standard = normalize({"image": image}, seed=0, inplace=True)["image"][0]

unit = A.Pipeline([A.Normalize("minmax")], backend="cpu")
oriented = r.read_affine(shape, origin, axes, dtype=np.float64)[None]
result = unit({"image": oriented}, seed=0)["image"][0]
```

`inplace=True` permits reusing a freshly read buffer on CPU. A CUDA pipeline
uploads the image once, keeps normalization and subsequent augmentations on
device, and downloads the final result once. Use `backend="cpu"` for host
execution, `"cuda"` to require CUDA, or `"auto"` for availability-based fallback.
Add normalization at the desired position in an existing augmentation chain.

- `Normalize("minmax")` maps each channel's minimum and maximum to 0 and 1,
  including zero and fill values in its statistics.
- `Normalize("zscore")` (the default) uses the population mean and standard
  deviation (`ddof=0`) of **nonzero voxels only**. Input zeros remain zero;
  nonzero fill values participate like other image values.
- Statistics are independent for each selected image channel and use float64
  after reading/casting/interpolation. Images retain their float32/float64
  dtype. Constant/all-zero channels become zero; nonfinite inputs raise.
- `Normalize` accepts the augmentation library's channel-first 2D/3D images.
  Empty reader crops remain valid reader outputs; skip augmentation for empty
  crops. CPU and CUDA can differ in floating-point rounding, while replay is
  deterministic within a backend/build/device.

Block reads still convert directly from source chunks into floating output.
Nearest affine reads that change dtype use a temporary stored-dtype crop
before conversion. These dtype options do not change indexing, `read_points`,
or `sample` behavior. See [normalization details](docs/augmentation.md#normalization)
for channel/target selection and migration examples.

### Groups

Groups open their arrays with shared settings and one memory cache:

```python
from vc_reader.reader import ZarrGroup, S3Credentials

g = ZarrGroup.open("s3://private-bucket/scroll.zarr", cache_dir="/data/cache", memory_cache=8 << 30)
g.levels, g.attrs                                     # ("0", ..., "5") from OME multiscales, else "0", "1", ...
r = g.level(2)                                        # ZarrReader; g["labels/ink"], "mask" in g, g.keys()
g.array_keys(), g.group_keys()                        # immediate children, filtered by node type
vol.open_group()                                      # a catalog Volume's OME-Zarr root

ZarrGroup.open(url, cache_dir="/data/cache", offline=True)   # only the disk caches, never the network
ZarrReader.open("https://host/v.zarr/0", auth=("user", "password"))
ZarrReader.open("s3://bucket/v.zarr/0", auth=S3Credentials(key_id, secret, region="us-east-1"))
```

> [!NOTE]
> **Readers work in forked processes**, such as PyTorch `DataLoader` workers
> on Linux. A reader does its work on background threads, and a forked process
> inherits the reader but not those threads. So a reader notices when it is
> used in a process other than the one that opened it, and rebuilds its native
> engine there from its saved settings. The parent keeps its own engine and
> finishes its own queued cache writes. In each child, a shared `MemoryCache`
> starts empty and retains one budget shared by its readers. Workers started
> with `spawn` or `forkserver` preserve that sharing when the readers are
> pickled together, as in a dataset. Separate pickle streams create separate
> caches. Cache contents are process-local, so multiply the budget by the
> number of workers when planning total memory.

### Prefetching

Use the same `(start, size)` regions as `read_many`. Prefetch loads whole chunks
through the native reader without allocating crop outputs. Normal remote reads
join/promote outstanding work automatically; no separate readiness gate is needed.

```python
request = r.prefetch(regions)
# Do other work, then read normally.
crops = r.read_many(regions)
request.wait(timeout=30)  # optional: raises loading/publication errors
request.done(), request.exception()
request.cancel()         # withdraw queued work only this request needs

with r.prefetch_window() as ahead:
    for current_regions, future_batches in plans:
        # Ordered nearest first; replaces this window's previous plan.
        ahead.update([current_regions, *future_batches])
        crops = r.read_many(current_regions)
        process(crops)
```

Windows retain interest even when the work queue is full. Updates drop obsolete
queued work, preserve overlapping work, and leave other callers' requests alone.
An unchanged, completed window does not repeatedly reload evicted chunks. Reads
resolve cache misses normally. Closing a window withdraws its outstanding interest;
already-running loads may finish. A discarded one-shot request still runs to completion.

Without a service, the reader lazily creates an in-process scheduler, closed with
the reader. Local prefetch warms file pages and, when configured, `MemoryCache`.
Remote prefetch needs a disk cache or enough memory cache for a chunk. Offline
prefetch only uses existing caches and reports cache misses without downloading.

For multiple loader workers, explicitly create one coordinator in the parent:

```python
from vc_reader.reader import PrefetchService, ZarrReader

def worker(client, source, cache_dir):
    with ZarrReader.open(source, level=2, cache_dir=cache_dir,
                         prefetcher=client, threads=4) as reader:
        with reader.prefetch_window() as ahead:
            # Plan/update/read here, inside the worker.
            ahead.update([[((8192, 3584, 3584), (128, 128, 128))]])
            return reader.read((8192, 3584, 3584), (128, 128, 128))

if __name__ == "__main__":
    with PrefetchService(connections=48, max_pending_chunks=512,
                         max_inflight_bytes=256 << 20) as service:
        # Pass service.client into datasets before starting their DataLoaders.
        # Stop/join the workers before leaving this context.
        pass
```

The picklable client works with spawn, forkserver and fork. Readers can also be
passed to workers; create requests/windows in the worker itself. `ZarrGroup.open`
accepts `prefetcher` and passes it to its arrays. The service requires writable disk
caches for remote arrays and rejects `offline=True`; workers never independently
download missing chunks. Group discovery and attribute access retain their normal
synchronous metadata behavior.

Scheduling and cache contracts:

- Foreground chunk misses outrank queued speculation and share its in-flight
  loads. Compatible readers share one service reader per array/configuration.
  One connection and one chunk's byte capacity are reserved for demand when the
  configured limits permit; running transfers are not forcibly interrupted.
- `connections` limits concurrent chunk loads, `max_pending_chunks` bounds queued
  plus active jobs, and `max_inflight_bytes` bounds the sum of their decoded chunk
  sizes. The latter is **not a total RAM cap**: HTTP/IPC buffers, output arrays,
  memory caches and file pages also consume memory. A chunk exceeding it is rejected.
- `max_plan_chunks` (default 65,536) bounds total retained plan entries;
  `max_requests` (default 1,024) bounds live requests/windows. Oversized plans raise
  instead of silently losing work. Release completed request handles when finished.
- Cross-process completion waits for publication of cache files or missing-chunk
  markers. Existing disk limits still apply. A request whose data cannot be
  published reports an error; foreground reads can receive an owning chunk over
  IPC, including when the disk budget is smaller than a chunk. Completion does
  not pin data or promise permanent residency.
- `r.prefetch_stats()` / `service.stats()` report queued/active work, active
  decoded bytes, completed loads, errors, promotions and deduplication. Speculative
  requests retain completion status, not a second unbounded decoded-data cache.

Focused verification uses tiny local arrays and a loopback HTTP fixture:
`python -m pytest tests/test_prefetch.py`. No external volumes or network are needed.

### Reader options

- **Credentials** (`auth=`): the default `"auto"` signs `s3://` requests to
  buckets without a known public https form with AWS credentials from the
  environment (`AWS_ACCESS_KEY_ID`…, `AWS_PROFILE`, `~/.aws`; with botocore
  installed also SSO and instance roles) when there are any, and reads
  everything else anonymously. `"aws"` requires those credentials, which are
  refreshed as they near expiry. `S3Credentials(...)` gives fixed keys (and
  optionally `region`, `endpoint_url` for S3-compatible servers).
  `(user, password)` is HTTP basic auth; `load_basic_auth(path)` reads the
  JSON file vesuvius uses for `volume_auth_json`. Requests are signed with AWS
  Signature V4 in the native code. 401/403 raise `PermissionError`: S3
  answers 403 rather than 404 for an absent chunk unless the credentials may
  list the bucket (`s3:ListBucket`), so sparse arrays in private buckets need
  that permission.
- **Offline** (`offline=True`, needs `cache_dir`): nothing is fetched. Array
  and group metadata come from the disk caches (any supported layout); a chunk
  none of them holds raises `vc_reader.CacheMissError`, and chunks the caches
  mark as absent read as `fill_value`. Groups opened online with a writable
  cache store their `.zgroup`/`.zattrs` there for later offline use.
- **Array metadata**: `r.attrs` lazily reads source `.zattrs`; `r.compressor`,
  `r.filters` and `r.order` expose the array's storage metadata. Attribute
  dictionaries are in-memory snapshots; changing them does not write the
  source. Attributes fetched with a writable cache are stored in
  `.vc_reader_source_attrs.json`, separate from the cache's own `.zattrs`.
  Accessing `.attrs` offline raises `CacheMissError` if it has never been
  cached; a recorded absence of source `.zattrs` returns `{}`.
- **Indexing**: integer arrays can be mixed with unit-step slices, with NumPy
  broadcasting, axis ordering and negative-index handling. These selections
  use batched block reads. `np.asarray(r)` materializes the entire array;
  use slices for large volumes. Boolean indexing, new axes and non-unit
  slice steps are not supported.
- **Points and sampling**: `read_points` gathers stored values at integer
  indices; `sample` interpolates at real coordinates (voxel `i` at coordinate
  `i`; `order=1` uses float32 for default 3-D uint8 CT sampling, or the legacy
  float64 path for other source types/explicit float64 output, with
  neighbours outside the array counting as `fill_value`, like
  `scipy.ndimage.map_coordinates(mode="grid-constant")`; `order=0` nearest
  with ties to even). Both run natively on the reader's threads and read each
  chunk once per batch of 2^20 points, through the same caches as `read`.
- **Groups**: `levels` follows the OME-NGFF `multiscales` datasets, else the
  arrays `"0"`, `"1"`, … `keys()` lists local directories, consolidated
  `.zmetadata`, S3 prefixes (ListObjectsV2) or, offline, the caches; plain
  HTTP servers cannot be listed, so open children by name there.
- **Codecs**: uncompressed, blosc, standalone zstd, LZW and vc-delta3d chunks
  in C order. LZW supports TIFF-style MSB-first byte streams under
  [`imagecodecs_lzw`](https://github.com/cgohlke/imagecodecs/blob/master/imagecodecs/numcodecs.py)
  or the alias `lzw`; it does not open TIFF containers. Other
  arrays (numcodecs filters, Fortran order, other compressors) raise
  `ValueError` when opened.
- **Disk caches** (`cache_dir=` one directory or a list; `"vc3d"` means
  VC3D's configured remote cache root): every existing cache of the array is
  found and read, and misses are appended to the first one in its own format.
  Supported layouts:
  - vc_reader / vesuvius `RemoteChunkedArray`: decoded chunks under
    `<root>/s3/<bucket>/<path>/<level>/`, a plain uncompressed zarr v2 array.
  - VC3D mirror: `<root>/<name>-<fnv1a64>/<level>/`, holding the source's
    bytes verbatim.
  - VC3D Delta3D: the same, with chunks encoded as `vc-delta3d`.

  VC3D caches are used under VC3D's shared lease, so VC3D never replaces a
  cache vc_reader has open. If none exists, a new decoded cache is created
  under the first directory. `compress_cache=True` makes that new cache store
  its chunks with lossless Delta3D (3-D uint8/uint16 arrays). It is the same
  layout, its `.zarray` names `vc-delta3d`, and any reader uses it afterwards.
  `cache_writes=False` makes all caches read-only. A cache that disagrees with
  the remote `.zarray` raises instead of mixing data. Quantized (lossy) Delta3D
  chunks are refetched exactly, and replaced, unless `allow_lossy_cache=True`.
- **Disk limits**: `cache_max_bytes=100 << 30` sets one shared budget for
  decoded chunk payloads under the first explicit `cache_dir` (not `"vc3d"`).
  With a limit, misses are written there; other cache directories and VC3D
  caches remain readable but are not modified. Existing decoded chunks in
  that directory's `http/`, `https/` and `s3/` trees count toward the budget.
  Oldest writes are evicted first, using unlink so active memory maps remain
  valid. A chunk larger than the limit is returned without being cached;
  zero disables chunk storage. Missing-chunk markers, metadata, the SQLite
  index and filesystem overhead are outside the payload budget.
  Temporary writes and unlinked files held by active memory maps can use
  additional space until those writes and readers finish.

  SQLite serializes accounting and eviction across readers and processes.
  Each process scans the decoded caches on its first bounded open to
  reconcile existing files; later writes update the index incrementally.
  Configure the same cap on every writer sharing a directory; changing it
  updates the shared limit. Unbounded writers and other programs do not
  participate in accounting until a later reconciliation. Offline and
  `cache_writes=False` opens never evict. Call `flush()` to finish queued
  writes and their eviction before checking usage. The index requires a
  local filesystem with SQLite locking support.
- **Write-behind**: a fetched chunk is decoded in memory and returned at once.
  Its cache file is written, and encoded if the cache is compressed, by
  background writer threads. Until then, rereads are served from the pending
  write, never refetched. `r.flush()` waits for the queued writes, and
  `close()` (or dropping the reader) finishes them.
- **Compressed vs uncompressed caches**: on real CT (PHercParis4 2.4 µm,
  level 2), Delta3D stores chunks in 68% of the space (1.46× smaller). Reading
  them costs a decode: a 256 × 128³ batch took 75 ms cold and 59 ms warm,
  against 70 and 41 ms from an uncompressed cache. A memory cache pays that
  decode once per chunk.
- **Memory cache** (`memory_cache=` bytes, or a shared `MemoryCache`): an LRU
  of decoded chunks, keyed by array, so one budget serves many readers.
- **Local reads** of uncompressed chunks go through persistent memory maps
  (`read_mode="mmap"`, the default) or `pread` (`read_mode="slab"`), on
  `threads` threads (default 16 for local arrays, 32 for remote ones).
  Passing reused arrays as `out=` avoids the first-touch page faults of new
  memory.
- **Remote reads**: GETs with libcurl, at most `connections=`
  (default 16) in flight, a 10 s stall timeout and retries with backoff. A 404
  means fill. A chunk wanted by several reads at once is fetched once.
- Readers pickle as their location plus metadata, so workers reopen them
  without a network round trip (see the note above for forked workers).
  `r.stats()` counts memory and disk hits, remote fetches and cache write
  errors. Values are exactly the stored values.

Limits:

- Memory-mapped chunk files must only ever be replaced by rename (vc_reader,
  vesuvius and VC3D all write that way). Truncating a mapped file in place
  would crash the process; use `read_mode="slab"` if something does that.
- Each reader keeps up to 8192 mappings, against Linux's default
  `vm.max_map_count` of 65530.

Building needs a C++20 compiler, CMake, libcurl, c-blosc, zstd and SQLite development files
(`apt install libcurl4-openssl-dev libblosc-dev libzstd-dev libsqlite3-dev`;
on macOS `brew install c-blosc zstd sqlite`). VC-Delta3D v0.1.1 is fetched at a pinned commit; its decoder runs
without the GIL.

Benchmarks against zarr-python and TensorStore are in
[`benchmarks/`](benchmarks/README.md).

## Native augmentation

`vc_reader.augment` provides C++ transforms for channel-first 2D/3D NumPy arrays
and CPU Torch tensors, with explicit seeds, replay, and shared geometry across
images and targets. See the [API and dataset integration guide](docs/augmentation.md)
for examples, transform coverage, numerical conventions, and build/test commands.

## CLI

```
vc-reader samples paris
vc-reader volumes "Scroll 1"
vc-reader labels "pherc paris 4" --resolution 2.399 --energy 78 --um 9.6
vc-reader labels "Scroll 1" --resolution 7.91 --scan 20230206171837
vc-reader --metadata ./metadata.json labels PHerc0139 --volume 20260930100252
```

Without installing: `PYTHONPATH=src python -m vc_reader …`.

## Development

```
cd vc_reader
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python scikit-build-core nanobind
uv pip install --python .venv/bin/python --no-build-isolation -e '.[bench,test]'   # rerun after C++ changes
.venv/bin/python -m pytest              # offline: catalog fixture + reader tests (local files, local HTTP server)
.venv/bin/python -m pytest -m network   # parse the live metadata.json end to end
```

The catalog tests also run on a bare Python 3.10+ without building
(`PYTHONPATH=src python -m pytest`); the reader tests are then skipped. They
need the built extension, plus numcodecs and `vc-delta3d` (the `test` extra)
to write their fixtures.

The LZW tests use checked-in raw strips encoded independently by libtiff:
repetitions exercise dictionary self-references, and deterministic noise
exercises 9–12-bit code widths and dictionary resets. Regenerate them with
`python tests/make_lzw_fixtures.py` when libtiff is installed; libtiff is not
a runtime dependency of the reader or its tests.

`tests/data/metadata_subset.json` holds real entries trimmed from the published
file. Regenerate it with `python tests/make_fixture.py path/to/metadata.json`.

The cache directory can be overridden with `VC_READER_CACHE_DIR`. A cached copy
younger than `max_age` (default 1 h) is used as is. After that the copy is
revalidated with the ETag, and it is used offline if the server cannot be
reached.
