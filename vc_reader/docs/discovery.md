# Discovering registered sources

Discovery describes published and user-supplied data using only catalog
metadata. It does not open files, resolve Zarr metadata, decode images, or
sample arrays. The catalog and discovery modules use only the Python standard
library. Unknown storage, axes, and placement remain explicit unknowns.

```python
from vc_reader import Catalog

catalog = Catalog.from_file("metadata.json")  # Also accepts gzip snapshots.
volume = catalog.sample("PHercParis4").volume("20260411134726")
labels = volume.labels()
representations = volume.level(2).representations(mappable=False)
related_volumes = volume.registered_volumes()
```

`labels()` and `representations()` return deterministic lists of immutable
`Source` values. `registered_volumes()` returns existing `Volume` objects;
select their `.level(...)` explicitly. A relationship establishes coordinate
connectivity and does not promise spatial overlap.

`mappable=False` requires registration directly to the queried volume ID.
`mappable=True` follows published transforms, including the catalog's existing
same-frame relationships. Ink labels preserve the specialized API's preference,
deduplication, supersession, and coverage rules. Generic sources retain distinct
IDs and use only explicit supersession links for `latest_only=True`.
`labels(label_set=...)` restricts to that recorded annotation set.

Level-specific representation queries record a corresponding-level request in
`source.storage.selection.corresponding`. For a prediction whose base grid is
volume level 2, a query at level 3 requests sublevel 1. A query at level 1 records
sublevel -1, indicating that the required finer array is unavailable under the
catalog convention. Mapped-volume requests retain the requested reference
frame and leave the sublevel unresolved. Opening in the subsequent reader
increment will validate actual metadata and reject unavailable/ambiguous levels.
There is no silent fallback to a coarser level.

Use `source.with_selection(level=0)` or
`source.with_selection(array_path="prediction", series="main")` to select an
explicit array instead. These return a copy with the same source ID. The
registration retains the grid for which it was declared; changing array
selection never silently reinterprets an affine. Label queries retain the
labels' native grid at every queried volume level.

## Adding a local or remote source

```python
import json
from vc_reader import Affine, Source, SourceCollection, Storage, VolumeRegistration

crop = Source(
    "annotation",  # Normalized to the user:annotation namespace.
    storage=Storage("/data/annotation.zarr", "zarr"),
    role="label",
    axes=("z", "y", "x"),
    registration=VolumeRegistration(
        volume.sample.id, volume.id,
        affine=Affine.from_rows([
            [1, 0, 0, 120],
            [0, 1, 0, 80],
            [0, 0, 1, 40],
        ]),
    ),
)
sources = SourceCollection(catalog, [crop])
labels = sources.labels(volume=volume.level(2))
restored = Source.from_dict(json.loads(json.dumps(crop.to_dict())))
assert restored == crop
```

The affine maps array-local XYZ voxel centers to volume level-0 XYZ. Crop
starts/sizes and declared storage axes use ZYX/YX. A `VolumeRegistration` can
instead declare `mode="storage"`, a named `storage_frame`, and a
storage-frame-to-volume affine. That mode requires composition with the array's
storage metadata when opening; a direct array affine must not acquire a second
copy of its scale.

Use `Plane(origin, basis_x, basis_y, thickness)` for an embedded 2D slice, passed
as `VolumeRegistration(..., plane=...)`. All four vectors are in level-0 XYZ;
`thickness` is a perpendicular vector spanning one source voxel along the
normal. An explicit stack uses `Storage(format="stack", locations=(...))` in
caller-supplied order. No filenames are sorted or directories scanned.

Surface labels use `SurfaceRegistration(surface, pixel_to_uv=...)`. A `Surface`
declares its geometry storage, axes/component order, XYZ-to-volume transform,
and geometry-pixel-to-UV transform. UV maps are 2x3 XY affines, with U increasing
along columns and V along rows. UV scale is independent of geometry XYZ units.
Published labels may lack these storage-level details; their descriptors retain
known target identity while reporting the missing declarations.

Geometric roles (`vector`, `normal`, `coordinate`) require explicit
`component_order` and `ValueFrame` declarations before sampling. Coordinate
values may use a named external frame with explicit units. Vector normalization
is an explicit boolean. `Encoding` supports integer or exact color-tuple
lookups, an integer output dtype, and an optional fallback for unknown values.

`Companion(name, source, same_grid=True)` explicitly binds a single-channel mask
to its parent's selected grid. Otherwise give the mask its own registration.
Companion values never become geometric validity or training policy.
`coverage` and `conflict` are reserved projection mask names.

## Inspection and lifecycle

`list(sources)` includes unknown, unregistered, and superseded assets.
`sources.get(id)` retrieves a descriptor or raises `NotFoundError`.
Unknown semantics are excluded from label/representation queries. Inspect
`source.unresolved_fields`, `source.provenance`, and `source.query` for missing
declarations, published metadata, and the queried relationship/coverage.
An empty `unresolved_fields` tuple is not proof that storage can be opened.

Catalog IDs include the sample/owner and published artifact ID. Missing IDs use
the artifact's metadata position and canonical catalog SHA-256. Those fallback
IDs are stable within a snapshot; record `sources.snapshot_sha256` alongside
experiments. User entries use a separate namespace. Duplicate IDs and generic
supersession cycles raise instead of replacing or merging assets.

Descriptors are deeply immutable and support versioned JSON and pickle without
storage handles. Collection entries and catalog relationships are snapshots;
construct a new collection/catalog after changing metadata. Entity convenience
queries reuse `catalog.sources`. A collection indexes artifacts once and retains
up to 128 query results; each query returns a fresh list. Selected copies can
share an ID, so downstream read/cache keys must include selection and placement.

`SampleResult(data, valid, masks, metadata=None)` defines the later sampling
contract without importing an array library. Volume data will use `(C,Z,Y,X)`,
surface labels `(C,Y,X)`, and sampled surface layers `(C,D,Y,X)`. Each target has
singleton-channel validity and a dictionary of named masks. `read_many` will
return an ordered list of unbatched results; batching belongs to the caller.

Aligned sampling uses these descriptors directly:

```python
from vc_reader.sampling import open_sample
with open_sample(volume.level(2), targets={"annotation": crop}) as reader:
    result = reader[:32, :32, :32]
```

## Run and verify

No native build or dependency installation is needed. From `vc_reader/`:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src python3 -S scripts/discovery_examples.py \
  --metadata /path/to/metadata.json --sample PHercParis4 --volume 20260411134726
AGENTS_AGENT_MODE=1 PYTHONPATH=src python3 -S -m unittest discover \
  -s tests -p test_sources.py -v
AGENTS_AGENT_MODE=1 PYTHONPATH=src python3 -S benchmarks/bench_discovery.py \
  --metadata /path/to/metadata.json --sample PHercParis4 \
  --output /tmp/discovery.json --profile /tmp/discovery.pstats
```

`-S` excludes installed editable-import hooks, making these commands useful in
an isolated worktree too. The benchmark uses six fixed volumes, two warmups, and
ten measured repetitions per query. It checks specialized/new label agreement,
records source IDs and snapshot hashes, and separates index time and traced
Python allocation from warm-cache latency. Asset bytes read are zero. These
measurements do not exercise file decoding, sampling, or CUDA.

### Recorded validation (2026-10-17)

On Linux amd64 / Python 3.14, the cached published catalog contains 45 samples
and 4,495 artifacts. Its canonical snapshot SHA-256 is
`de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d`.
The measured command was:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src /usr/bin/python3.14 -S benchmarks/bench_discovery.py \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --output /tmp/discovery-benchmark-final.json \
  --profile /tmp/discovery-profile-final.pstats
```

Values below are warm-cache **mean / p50 / p95 microseconds**, with two warmups
and ten measurements per case. The comparison is the existing specialized
`ink_labels()` API versus indexed `SourceCollection.labels()` in this change;
both return the same selected annotations. Catalog loading is excluded.

| PHercParis4 volume | Specialized query | Indexed source query |
| --- | ---: | ---: |
| 20230205180739 | 36.98 / 36.81 / 38.45 | 1.76 / 1.56 / 2.58 |
| 20230206171837 | 33.27 / 32.92 / 35.61 | 1.57 / 1.53 / 1.81 |
| 20260310170716 | 36.99 / 36.87 / 37.99 | 1.60 / 1.55 / 1.85 |
| 20260310173927 | 33.27 / 33.22 / 34.24 | 1.65 / 1.58 / 1.90 |
| 20260323153942 | 36.49 / 36.15 / 37.92 | 1.59 / 1.55 / 1.82 |
| 20260411134726 | 35.47 / 35.00 / 37.18 | 1.60 / 1.52 / 1.89 |

Index construction takes 0.351 seconds. Separately instrumented initialization
and queries retain 32.58 MiB of Python allocations, with a 32.58 MiB peak.
The profile attributes most startup work to canonical JSON hashing and immutable
descriptor adaptation/freezing; those costs are outside the repeated-query path.
Metadata values and existing affine matrices are retained without introducing
bulk numeric arrays or changing sampling precision.

The five permanent discovery tests pass with Python 3.11 and 3.14 without
optional packages. Real-snapshot validation round-tripped all 4,495 descriptors,
checked 296 discovery queries against specialized matching, and confirmed 2,156
specialized-query cases were unchanged from the original checkout. The existing
Release CUDA-enabled build passes the full 45-test suite, including its CPU
backend cases. The separately cached CPU-only build reproduces five existing
augmentation failures in both checkouts (compact local-field plans and signed
zero in low-resolution sampling); no native code was changed here. macOS and
arm64 were not runtime-tested.
