# Reading label sources

`Source.open()` reads local, HTTP(S), and S3 Zarr arrays/groups, PNG, JPEG,
TIFF, and explicitly ordered stacks of 2D files. It exposes `shape`, `dtype`,
`axes`, NumPy-style indexing, `companions`, `resolved`,
`stats()`, and `close()`. The source descriptor stays immutable.

Zarr sources use the existing native reader, authentication, memory cache,
chunk planning, disk cache, prefetch support, and worker lifecycle. Remote image
files use that same HTTP/S3 transport and directory budget. No installation
occurs at runtime. Image decoders come from the optional `labels` extra
(NumPy, Pillow, tifffile); local image reads do not load the native extension,
PyTorch, or CUDA. Unsupported TIFF compression identifies the required codec.

```python
from vc_reader import Encoding, Source, Storage
from vc_reader.reader import DecodedCache

source = Source(
    'annotation', Storage('/data/annotation.png', 'png'),
    role='label', axes=('y', 'x'),
    encoding=Encoding('integer', ((0, 0), (255, 1)), 'uint8'),
)
cache = DecodedCache(max_bytes=256 << 20, max_decode_bytes=256 << 20)
with source.open(decoded_cache=cache) as reader:
    labels = reader[100:164, 200:296]
```

Published ink-label descriptors select pyramid level zero, including their
separately named supervision and validation arrays:

```python
source = volume.labels.ink().source
with source.open(cache_dir='/data/label-cache', cache_max_bytes=1 << 30,
                 memory_cache=64 << 20, threads=1) as reader:
    labels = reader[100:228, 200:328]
    supervision = reader.companions['supervision'][100:228, 200:328]
```

Companion names do not choose training splits or merge masks with validity.
An explicit `same_grid` binding checks spatial shape and available transforms
before inheriting the parent's placement. Changing a parent selection does not
silently change independently declared companion selections: declare matching
companion grids when choosing a different pyramid level.

## Values and selection

Indexing follows `reader.axes` and `reader.shape`, including the channel axis:
CYX sources use `reader[channel, y0:y1, x0:x1]`, and YXC sources use
`reader[y0:y1, x0:x1, channel]`. Integers remove axes; slices preserve them.
Unit-step slices, negative indices, omitted trailing axes, and `...` follow
NumPy bounds rules. Slice bounds clip; out-of-range integers raise `IndexError`.
Stepped slices, boolean/array indexing, and `None` are not supported.

Opening-time channels/time selection uses
`source.with_selection(channels=(2, 0), time=0)`. Channel names are supported when
metadata supplies them. Time selection removes the time axis; singleton
channels remain. `with_selection` replaces the selection, so include an existing
level/series selection when changing channels or time.

Zarr group children and ambiguous TIFF series/levels require explicit selection.
For example, `source.with_selection(level=0)` or
`source.with_selection(array_path='labels', series='main')`. Anonymous TIFF axes
require declarations. Explicit stacks keep their supplied order, inspect the
first plane at open, and inspect other planes only when requested.

Reader properties describe the selected, encoded output. Integer lookup keeps
axes; exact color lookup removes the channel axis. Unknown values raise unless
an explicit fallback is supplied. Raw palette PNG reads preserve indices and
expose palette/transparency metadata through `reader.palette`. RGB, alpha,
16-bit grayscale, float32 samples, and uint64 label IDs retain their values.
JPEG returns decoded samples without thresholding or assigning classes.

For padded crops, `read(start, size)` accepts spatial YX/ZYX origins and extents;
`read_many(regions)` batches those region requests. These explicit region reads
keep the requested size outside the array, without NumPy clipping or
negative-index wrapping. `Source.fill_value` belongs to the
output dtype and bypasses encoding. Otherwise Zarr uses its storage fill,
encoded when applicable, and image files use zero. Missing Zarr chunks are
in-bounds fill data and undergo normal encoding. Failed downloads, corrupt
images, and missing stack planes raise. Returned arrays remain valid after
subsequent reads, eviction, or close.

## Direct Zarr array selections

`ZarrReader` uses NumPy-style selections for indexing, dtype conversion,
destination buffers and batched reads:

```python
import numpy as np

crop = np.s_[9000:9128, 3600:3728, 3600:3728]
block = reader[crop]
float_block = reader.read(crop, dtype=np.float32)
reader.read(crop, out=buffer, dtype=np.float32)
blocks = reader.read_many([
    crop,
    np.s_[9032:9160, 3632:3760, 3632:3760],
])
```

`np.s_` builds slice selections without reading data, using the same colon
syntax as direct indexing. These methods accept integers, unit-step slices and ellipsis.
Negative indices wrap, slice bounds clip, and integer indices remove axes.
Destination buffers must match the selected shape and requested dtype;
conversion writes directly into those buffers. Batched selections share chunk
reads and support one optional output buffer per selection.

`reader.cuda()` accepts the same selections through indexing, `read` and
`read_many`. Its arrays and optional `valid=True` masks stay on the device;
batching shares chunk uploads. CPU destination buffers are supported by
`ZarrReader`. Explicitly padded low-level geometry uses `read_region` and
`read_regions`, whose bounds are not clipped or wrapped.

## Storage metadata and placement

`reader.resolved` records the actual location/selection, selected output layout,
array-to-storage transform and units, and resolved registration. Raw reads can
remain unregistered. Contradictory axes, shapes, dtypes, frames, transforms, or
companion bindings fail with the source ID.

NGFF scale/translation metadata and available TIFF physical spacing resolve
selected grids. Unit conversion uses micrometers internally. Storage-frame
placement requires explicit `VolumeRegistration.storage_units` for dimensional
metadata. Catalog fallback placement records `volume_spacing_um` so physical
metadata can supersede an estimated scale. Unknown units and unknown
relationships between different grids/series fail rather than inventing
placement. Explicit array placement keeps its crop origin and does not acquire
an extra scale on the same grid. Corresponding-level requests require one exact
resolution match, including a published cross-volume transform when applicable.

Small affine compositions use the existing Python floating-point matrices.
The regression with origin `2**26 + .25` demonstrates why float32 would lose a
required subvoxel origin. Label IDs and shard offsets use exact integers; no
bulk float64 coordinate arrays are introduced.

## Bounded I/O, decoding, and workers

Remote images stream into seekable disk files. The default encoded-file payload
budget is 1 GiB. `cache_max_bytes` configures the shared directory budget for
both image files and decoded Zarr chunks; omitting it reuses a previously set
limit. Existing Zarr-only defaults are preserved. SQLite coordinates atomic
publication and eviction across processes. Temporary materialization without
a writable cache also has a finite budget. An image larger than the limit raises
with instructions to increase it. Metadata/index files and filesystem overhead
are separate from the payload budget.

ETag/Last-Modified revalidation occurs once per image location and reader/process.
Decoded keys include content identity. Repeated reads use pinned file handles
and avoid rehashing unchanged content. `offline=True` serves cached files/chunks
or raises `CacheMissError`; it does not fetch credentials for image-cache hits.
`cache_writes=False` uses existing files or a bounded temporary cache online.
Open descriptors remain valid if an eviction unlinks their files. Physical
blocks for an actively pinned inode can remain until its read finishes.

`decoded_cache_bytes` and `max_decode_bytes` default to 256 MiB. They control
retained image/tile data and concurrent image decode/conversion working space.
Pass one `DecodedCache` to share both limits across readers in a process;
configure limits on that object, without also passing per-reader byte limits.
Zarr retains its native `memory_cache` and bounded thread/connections controls;
source selection/encoding scratch is additionally checked against
`max_decode_bytes`. Caller-owned output arrays have separate accounting.

PNG/JPEG may decode the complete image, even for a tiny crop. TIFF maps
uncompressed pages when possible and selectively decodes intersecting compressed
tiles/strips, including planar channels and separate channel pages. Dimensions
and conservative decoder/scratch estimates are checked before allocation.
Opaque codec internals prevent these estimates from being a hard OS RSS limit.
Small outputs do not retain uncached whole-image allocations.

Shared decode work is serialized, including conversions. Source readers default
to one decode/read thread per worker. Pickle stores configuration and descriptors;
fork/spawn workers reopen resources and caches lazily. Closing is idempotent and
preserves explicitly shared cache ownership.

## Zarr v3 support

The shared `ZarrReader` and `ZarrGroup` support v2 and the v3 layouts used by the
published ink labels: regular grids, default/v2 chunk keys, bytes serialization,
optional Blosc or Zstd compression, and `sharding_indexed`. Unsupported codec
pipelines/storage transformers fail explicitly. Shard indexes support
little-endian bytes with optional CRC32C and either index location, following the
[Zarr sharding specification](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html).

Only the index and requested compressed inner chunks are fetched. HTTP ranges
are validated; ignored ranges are rejected before accepting a whole-shard body.
A strong ETag binds payload ranges to the index version. Missing inner chunks
retain the normal fill/cache behavior. Decoded inner chunks use the existing
v2 disk-cache representation, including offline reads and shared budgets.

Index working storage is limited to 64 MiB. One remote index of at most 8 MiB
can be retained per array; larger indexes are reread within the working limit.
Encoded payloads are capped at twice the decoded inner-chunk size plus 1 MiB.
Arrays requiring larger indexes or unusual oversized encoded payloads raise.
These limits never materialize an entire shard. Native chunk sizes and worker
counts still determine the native decoder's working memory.

## Run and validate

With existing dependencies and configured build directories:

```bash
AGENTS_AGENT_MODE=1 cmake --build build/cp314-cp314-linux_x86_64 --target _native -j 2
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _native -j 2
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda .venv/bin/python -m unittest discover -s tests -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=build/cp314-cp314-linux_x86_64 .venv/bin/python -m unittest discover -s tests -p 'test_source*.py' -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=build/cp314-cp314-linux_x86_64 .venv/bin/python -m unittest discover -s tests -p test_zarr_v3.py -v
```

HTTP tests use loopback servers. No dependencies were installed. Tests cover
exact image/label values, selection and registration, pre-decode limits,
shared budgets, eviction, atomic population, authentication/retries, offline
hits/misses, v3 range/checksum failures, and process reopening. The configured
CUDA build also runs sampling/augmentation/training tests. Aligned label
sampling and its CPU/CUDA equality checks are documented in
[aligned sampling](aligned-sampling.md) and its
[validation report](aligned-sampling-benchmarks.md).

The complete configured suite passed 94 tests, including existing CUDA tests.
The CPU-only build separately passed 36 source/descriptor tests and seven v3
tests. Validation was performed on Linux amd64. macOS and arm64 runtime checks were
not available; the implementation uses portable CPU code without new SIMD or
GPU codec dependencies.

See [label-reader benchmark results](source-reading-benchmarks.md) for exact
inputs, reproduction commands, latency, memory, I/O, and profiling limits.
