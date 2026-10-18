# Aligned volume sampling

`vc_reader.sampling.open_sample` opens a reusable reader for a `VolumeLevel` or
registered volumetric image `Source`. Target names are caller-owned; `image`
is reserved for the reference. Discovery and importing `vc_reader.sampling`
remain independent of NumPy, native extensions, decoders, and PyTorch. Opening
requires the native reader and NumPy; image formats additionally require their
existing optional decoders.

```python
from vc_reader.sampling import open_sample

# reference is a selected VolumeLevel or registered image Source.
# label and field are explicitly chosen Source descriptors.
with open_sample(reference, {"label": label, "field": field}) as reader:
    sample = reader[100:132, 200:328, 300:428]
    image = sample.data["image"]
    labels = sample.data["label"]
    supported = sample.valid["label"]
    source_masks = sample.masks["label"]
    # A named target uses CZYX indexing and returns just that target's array.
    component = reader["field"][1, 100:132, 200:328, 300:428]
```

Index the shared reference grid in ZYX order. `reader.shape` is that grid's
spatial shape. Unit-step slices, integers, negative indices, omitted trailing
axes, and `...` follow NumPy rules: slice bounds clip to the array, negative
indices count from its end, and out-of-range integers raise `IndexError`.
Stepped slices, boolean/array indexing, and `None` are not supported.

Slice-only selections return CZYX data, boolean `(1, Z, Y, X)` validity,
and separately named single-channel companion masks. Integer indices remove
the corresponding spatial axes from every data, validity, and mask array;
`reader[z, :, :]` therefore returns CYX arrays. Channel axes remain present.

`reader["field"]` exposes one target on the same reference grid with CZYX
indexing and a CZYX `shape`. `reader["field"][1, ...]` selects one channel,
removes that channel axis, and returns an array. Use `1:2` to retain the axis.
This reads only the named target; geometric components are transformed together
before channel selection. The returned CUDA array is a device view, with no
host download. Raw `Source.open()` readers instead index their reported storage
axes, such as YXC or CYX. Selected channel order is preserved.

For padded regions extending outside the reference grid, `read(start, size)`
uses explicit ZYX origins and extents without NumPy clipping or negative-index
wrapping. `read_many(regions)` batches those region requests and returns
a list in request order; callers own batching and training policies. Empty
regions and empty request lists are supported. Buffers survive subsequent reads,
cache eviction, and `close()`.

`VolumeLevel` supplies its catalog automatically. For standalone sources,
`catalog=` supplies missing published relationships. Compatible query context
on either descriptor can establish one relationship, in either direction;
otherwise missing relationships fail at open. No catalog is downloaded
implicitly. Storage placement and geometric component frames are resolved
separately, and contradictory declarations fail instead of selecting a transform.
A standalone source can use `ValueFrame(..., affine=...)`; a nonzero value-frame
level request needs a catalog to resolve its actual storage grid.

## Sampling rules

- Labels and discrete masks use nearest neighbor, ties to even, and exact
  stored integer copies, including uint64 values above `2**53`. Encodings run
  in the source reader before sampling. CPU categorical results retain their
  declared dtype; CUDA arrays use native byte order with the same integer type.
- Images and scalar fields use linear interpolation with float32 output.
  Integer-aligned direct copies retain the stored dtype. No normalization or
  antialiasing is implicit.
- Vectors use the value-frame linear map; normalization requires `normalize=True`.
  Normals use its inverse transpose and normalize. Undefined normals become zero
  with false validity. Geometric fields return float32.
- Coordinate fields use the full value-frame map. Their output positions are
  absolute in the selected reference array, independent of the read's crop
  origin. External coordinate frames preserve their declared frame.
- A 2D `Plane` has inclusive normal support `[-0.5, 0.5]` in units of its
  declared thickness vector. Sampling is nearest/bilinear in-plane, constant
  across that footprint. A surface registration requires the separate surface
  operations; it is not implicitly projected into a volume crop.

Unsupported neighbors use the source reader's resolved output-domain fill.
Partial linear samples blend that fill with supported values but have false
validity. Zero-weight neighbors do not affect values or validity. Fully outside
samples retain fill, including geometric fields; partial geometric samples are
transformed after interpolation. Missing Zarr chunks are supported fill data;
transport, decoding, and unknown-encoding errors propagate.

Companion masks use their own resolved placement or explicit `same_grid`
binding. Their values do not change parent validity. Normal validity also
requires a defined direction; other validity masks describe geometric support.

Geometric targets populate `sample.metadata['value_frames']`, including
component order, reference array placement/selection, or external name/units.
For geometric fields, `component_order` describes the selected channels;
update that declaration when changing their selection/order. Metadata otherwise defaults to `None`. Omit metadata from default DataLoader
collation or convert the fields explicitly.

## CUDA, ownership, and budgets

Pass `backend='cuda', device=0` to return existing `vc_reader.cuda.DeviceArray`
objects. Use `.numpy()` for an explicit download or DLPack to share storage
with a consumer. Operations complete on the existing native reader stream
before returning. No PyTorch or GPU image decoder dependency is added.
Create CUDA readers in the process owning the device; they reject fork reuse,
pickling, and creation inside an active PyTorch DataLoader worker.

CPU readers serialize descriptors/options and reopen lazily after unpickling
or fork. Read calls on one reader serialize; native I/O uses the configured
reader threads, defaulting to one. Explicit shared `MemoryCache` and
`DecodedCache` objects retain their ownership semantics. Otherwise the sampler
owns one shared cache of each kind across its source readers and clears them
on close.

Additional sampling controls:

| Option | Default | Bound |
| --- | ---: | --- |
| `sampling_cache_bytes` | 64 MiB | Retained native-endian CPU tiles or CUDA tiles, shared across targets |
| `sampling_work_bytes` | 64 MiB | Conservative simultaneous source/canonical/device tile buffers per request block |
| `plan_cache_bytes` | 8 MiB | Retained native affine plan storage |
| `device` | 0 | CUDA device ordinal |

All other reader options are passed to `Source.open()`. Native source chunk
caches, image decode scratch, plan-building scratch, Python object overhead,
and caller-owned result arrays have separate accounting. Plans contain at most
32768 output voxels' tile requests; they never retain a dense coordinate grid.
Oversized tile requests split into smaller output blocks. If one stencil cannot
fit the working budget, the read raises rather than exceeding it. Repeated
selections of the same storage/encoding reuse tile values and compatible plans.
The existing bounded uint8 CPU affine reader remains the fast path where it
covers the requested semantics.

Spatial sizes are limited to `2**24` per output axis; crop starts and source
extents are limited to `2**40`. Tile-index overflow and nonfinite transforms
raise. Bulk geometry and interpolation use float32 with integer global origins.
Small affine composition/inversion and origin splitting retain Python double:
the regression at origin `2**26 + 1.5` shows that narrowing before splitting
changes nearest-neighbor decisions. There are no bulk float64 coordinate arrays.

## Build and verification

Using the existing environments and configured build directories:

```bash
AGENTS_AGENT_MODE=1 cmake --build build/cp314-cp314-linux_x86_64 --target _native _augment -j 2
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _native _augment -j 2
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=build/cp314-cp314-linux_x86_64 .venv/bin/python -m unittest discover -s tests -p test_sampling.py -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda .venv/bin/python -m unittest discover -s tests -v
```

See [sampling measurements](aligned-sampling-benchmarks.md) for fixed real
inputs, baseline comparisons, profiling, and remaining platform limitations.
