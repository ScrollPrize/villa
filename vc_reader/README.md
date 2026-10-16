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
lease. It needs the `reader` extra (numpy).

```python
from vc_reader.reader import ZarrReader, MemoryCache

lvl = cat.sample("Scroll 1").volume(resolution=2.4, energy_kev=78).level(um=9.6)
# reuse the RAID cache and VC3D's cache; misses are fetched and appended
r = lvl.open(cache_dir=["/mnt/raid_nvme/volume_cache", "vc3d"], memory_cache=8 << 30)
crop = r.read((8192, 3584, 3584), (128, 128, 128))   # out-of-bounds parts read as fill_value
crops = r.read_many([(start, (64, 64, 64)) for start in starts])  # one parallel read; shared chunks read once
crops = r.read_many(regions, out=crops)               # reuse arrays: no first-touch page faults
block = r[8192:8704, 3584:4096, 3584:4096]

ZarrReader.open("/data/volume.zarr/2")                # local array
ZarrReader.open("s3://vesuvius-challenge-open-data/…/volume.zarr", level=2)
shared = MemoryCache(16 << 30)                        # one budget for several readers

vals = r.read_points(zyx)                             # (..., 3) ints -> stored values; outside: fill_value
vals = r[z_idx, y_idx, x_idx]                         # numpy integer-array indexing, same path
vals = r.sample(zyx_float)                            # trilinear, float32 (order=0: nearest, as stored)
```

Groups open their arrays with shared settings and one memory cache:

```python
from vc_reader.reader import ZarrGroup, S3Credentials

g = ZarrGroup.open("s3://private-bucket/scroll.zarr", cache_dir="/data/cache", memory_cache=8 << 30)
g.levels, g.attrs                                     # ("0", ..., "5") from OME multiscales, else "0", "1", ...
r = g.level(2)                                        # ZarrReader; g["labels/ink"], "mask" in g, g.keys()
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
> finishes its own queued cache writes. In the child, each reader starts with
> an empty memory cache of the same size: a `MemoryCache` shared by several
> readers is not shared there. Workers started with `spawn` or `forkserver`
> receive a pickled reader, which is reopened the same way.

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
- **Points and sampling**: `read_points` gathers stored values at integer
  indices; `sample` interpolates at real coordinates (voxel `i` at coordinate
  `i`; `order=1` n-linear in float64, returned as float32 or float64, with
  neighbours outside the array counting as `fill_value`, like
  `scipy.ndimage.map_coordinates(mode="grid-constant")`; `order=0` nearest
  with ties to even). Both run natively on the reader's threads and read each
  chunk once per batch of 2^20 points, through the same caches as `read`.
- **Groups**: `levels` follows the OME-NGFF `multiscales` datasets, else the
  arrays `"0"`, `"1"`, … `keys()` lists local directories, consolidated
  `.zmetadata`, S3 prefixes (ListObjectsV2) or, offline, the caches; plain
  HTTP servers cannot be listed, so open children by name there.
- **Codecs**: uncompressed, blosc and vc-delta3d chunks in C order. Other
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

Building needs a C++20 compiler, CMake, libcurl and c-blosc development files
(`apt install libcurl4-openssl-dev libblosc-dev`; on macOS `brew install
c-blosc`). VC-Delta3D v0.1.1 is fetched at a pinned commit; its decoder runs
without the GIL.

Benchmarks against zarr-python and TensorStore are in
[`benchmarks/`](benchmarks/README.md).

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

`tests/data/metadata_subset.json` holds real entries trimmed from the published
file. Regenerate it with `python tests/make_fixture.py path/to/metadata.json`.

The cache directory can be overridden with `VC_READER_CACHE_DIR`. A cached copy
younger than `max_age` (default 1 h) is used as is. After that the copy is
revalidated with the ETag, and it is used offline if the server cannot be
reached.
