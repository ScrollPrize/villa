# vc_reader

Find and locate Vesuvius Challenge open data from the published
[`metadata.json`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
(schema: [`metadata.schema.json`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.schema.json)).

The catalog fetches and parses the metadata and answers "which volume / level
/ labels / predictions, and where are they". It uses only the standard library.
`vc_reader.reader` then reads the data. Zarr v2 arrays are supported now
(volumes and volume predictions). Zarr v3 (ink labels), tif and png are next.

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
  `energy_kev=`. Superseded volumes are skipped unless you name them or pass
  `include_superseded=True`. If more than one volume matches,
  `AmbiguousMatchError` is raised with the candidates listed.
- **URLs**: `artifact.url()` returns HTTPS and prefers the open-data bucket.
  `url("s3")` returns the `s3://` form. EduceLab-licensed volumes are only on
  `data.aws.ash2txt.org` / `s3://vesuvius-challenge`.

## Reading zarr arrays

`vc_reader.reader` reads zarr v2 arrays (OME-Zarr levels: CT volumes,
predictions) from a local directory, `https://` or public `s3://`, optionally
through a persistent disk cache and an in-memory chunk cache. It needs the
`reader` extra (numpy, urllib3, numcodecs, and vc-delta3d on Python 3.12+).

```python
from vc_reader.reader import ZarrReader, MemoryChunkCache

lvl = cat.sample("Scroll 1").volume(resolution=2.4, energy_kev=78).level(um=9.6)
# reuse the RAID cache and VC3D's cache; misses are fetched and appended
r = lvl.open(cache_dir=["/mnt/raid_nvme/volume_cache", "vc3d"], memory_cache=8 << 30)
crop = r.read((8192, 3584, 3584), (128, 128, 128))   # out-of-bounds parts read as fill_value
crops = r.read_many([(start, (64, 64, 64)) for start in starts])  # overlapping reads, one pool
crops = r.read_many(regions, out=crops)               # reuse arrays: no first-touch page faults
block = r[8192:8704, 3584:4096, 3584:4096]

ZarrReader.open("/data/volume.zarr/2")                # local array
ZarrReader.open("s3://vesuvius-challenge-open-data/…/volume.zarr", level=2)
```

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
  under the first directory. `cache_writes=False` makes all caches read-only.
  A cache that disagrees with the remote `.zarray` raises instead of mixing
  data. Quantized (lossy) Delta3D chunks are refetched exactly unless
  `allow_lossy_cache=True`. Decoding `vc-delta3d` needs `vc-delta3d>=0.1.1`
  (in the `reader` extra on Python 3.12+), and `volcomp` needs the `volcomp_zarr` binding from
  SuperOptimizer/volume-compressor.
- **Memory cache** (`memory_cache=` bytes, or a shared `MemoryChunkCache`): an
  LRU of decoded chunks, keyed by array, so one budget serves many readers.
- **Local reads** of uncompressed chunks (`read_mode=`):
  - `slab` (default): `pread` the contiguous rows a region needs into a
    per-thread buffer.
  - `pread`: read whole chunks.
  - `mmap`: memory maps with `MADV_WILLNEED`, as in vesuvius `ChunkedArray`.
  
  Reads under 1 MiB (`inline_bytes=`) run in the caller after queueing
  `POSIX_FADV_WILLNEED` readahead for every piece. Larger reads run on
  `workers` threads (16 by default for local arrays). When their pieces are
  small, as in crop batches, the pieces are split into a few hinted groups per
  worker. Passing reused arrays as `out=` to `read`/`read_many` avoids the
  first-touch page faults of new memory (warm 128³ batches: about 25%
  faster).
- **Remote reads**: anonymous GETs over a urllib3 pool, with at most
  `connections=` (default 16) in flight and a 10 s stall timeout. Bodies are
  read straight into the chunk buffer, 404 means fill, and other failures are
  retried with backoff. A chunk wanted by several reads at once is fetched
  once.
- Readers pickle as their location plus metadata, so DataLoader workers reopen
  them without a network round trip. Values are exactly the stored values.

### Native engine

`engine="native"` reads through a C++ extension (`vc_reader._native`, built by
the pyproject with scikit-build-core and nanobind). It does the whole read on
native threads without the GIL:

- local chunks, through persistent memory maps by default (`read_mode="slab"`
  reads with `pread`);
- HTTP fetches with libcurl, at most `connections` at once, with stall
  timeouts and retries;
- raw, blosc and vc-delta3d decoding;
- the disk cache layers above, read and appended in their own formats;
- an in-memory LRU of decoded chunks.

Python still finds the caches, checks them against the remote `.zarray` and
holds VC3D's lease, so both engines see the same caches and return identical
values.

```python
r = lvl.open(engine="native", cache_dir=["/mnt/raid_nvme/volume_cache", "vc3d"], memory_cache=8 << 30)
from vc_reader._native import MemoryCache
shared = MemoryCache(16 << 30)                      # one budget for several native readers
```

Limits:

- Arrays with numcodecs filters, Fortran order or other codecs need
  `engine="python"`; opening them with the native engine raises `ValueError`.
- Memory-mapped chunk files must only ever be replaced by rename (vc_reader,
  vesuvius and VC3D all write that way). Truncating a mapped file in place
  would crash the process.
- Each native reader keeps up to 8192 mappings, against Linux's default
  `vm.max_map_count` of 65530.

Building it needs a C++20 compiler, CMake, libcurl and c-blosc development
files (`apt install libcurl4-openssl-dev libblosc-dev`; on macOS `brew install
c-blosc`). VC-Delta3D is fetched at a pinned commit.

Benchmarks against zarr-python and TensorStore are in
[`benchmarks/`](benchmarks/README.md).

## CLI

```
vc-reader samples paris
vc-reader volumes "Scroll 1"
vc-reader labels "pherc paris 4" --resolution 2.399 --energy 78 --um 9.6
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

The catalog tests also run on a bare Python 3.10+ (`PYTHONPATH=src python -m
pytest`); the reader tests are skipped without numpy and urllib3, and the
Delta3D cache test without `vc-delta3d`.

`tests/data/metadata_subset.json` holds real entries trimmed from the published
file. Regenerate it with `python tests/make_fixture.py path/to/metadata.json`.

The cache directory can be overridden with `VC_READER_CACHE_DIR`. A cached copy
younger than `max_age` (default 1 h) is used as is. After that the copy is
revalidated with the ETag, and it is used offline if the server cannot be
reached.
