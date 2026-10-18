# Scroll-centered discovery and paired reading

The catalog and public views use the standard library. `load()` reads catalog
metadata; navigating an already loaded catalog does not request arrays or
import NumPy, native extensions, Pillow, or PyTorch.

## Collections and relationships

```python
from vc_reader import load, ScrollSource
catalog = load()
scroll = ScrollSource('PHerc0009B', '20250820154339', catalog=catalog)
label = scroll.segments['20250919125754'].labels.ink(label_set='20260918')
image = label.images.zarr()
assert image.labels[label.id].id == label.id
assert label.volume.id == '20250820154339'
```

`catalog.samples` includes scrolls and fragments. `catalog.scrolls` filters on
the published sample type. Both return `ScrollSource` views. PHerc0009B is a
fragment in the acceptance snapshot; its classification is preserved.
`ScrollSource` also resolves the catalog's common sample-name aliases.

Collections iterate entities in stable ID order and support `[id]`,
`.filter(attribute=value, ...)`, and `.get(...)`. Filters use exact values;
relationship filters also accept entity IDs, e.g. `.get(volume='...')`.
Resolution queries have their own explicit tolerance. `.get()` returns the
single matching item. It raises `NotFoundError` if nothing matches or
`AmbiguousMatchError` if multiple items match, with candidates in the error.
With no criteria, the collection must contain exactly one item.
A missing ID also raises `NotFoundError`.

`labels.ink(label_set='20260918')` selects an ink label directly, with the same
missing/ambiguous match errors. `labels.ink()` works when exactly one ink label
is present. To browse multiple ink labels, use
`labels.filter(artifact_kind='ink-labels')` and iterate the returned collection.

`images.zarr()` selects the single Zarr image. Pass criteria such as
`images.zarr(artifact_id='...')` to choose among multiple images; missing or
ambiguous matches raise the same errors. Use `images.filter(format='zarr')`
to browse all Zarr images.

An omitted volume exposes the complete sample. `.on_volume(id)` shows direct
associations only. A segment qualifies if any published asset targets that
volume, including labels or images without a mesh. Views share entity records
and immutable `Source` descriptors; navigation never rewrites registration.

- `volumes`, `segments`, `labels`, `images`, `representations`, and
  `geometry` expose distinct kinds. `.artifacts` retains unknown types too.
- `label.segment`, `label.volume`, `label.images`, `label.masks`, and
  `image.labels` follow the indexed relationships. Image variants remain
  separate. Explicit published image references constrain the segment/target
  canvas convention. Equal shapes do not establish a relationship.
- `label.mappable_volumes` contains volume, transform, coverage (or `None`), and
  `required_operations`. Surface labels still need `surface_projection` for a
  3D destination. Connectivity does not promise complete spatial coverage.
- Explicit supersession hides assets and volumes by default. ID lookup finds
  historical entries; `.filter(include_superseded=True)` includes them in
  queries. Distinct alternatives without supersession links remain visible.
- `asset.source.unresolved_fields` and `.source.provenance` expose missing
  semantics and published metadata. Raw opening requires supported storage;
  aligned operations additionally validate the required placement.

## CT resolution candidates

```python
matches = catalog.scrolls.volumes.at_resolution(8, rel_tol=.02)
paris = catalog.scrolls['PHercParis4'].volumes.at_resolution(8, rel_tol=.02)
match = paris.get(volume='20230206171837')
resolved = match.resolve()
print(resolved.selection, resolved.spacing_um, resolved.shape)
with match.open() as reader:
    patch = reader[100:104, 200:204, 300:304]
```

All native and catalog-convention pyramid matches within tolerance are returned,
including nearby matches when another volume matches exactly. Ordering is
resolution difference, scroll ID, volume ID, level, source ID. Tolerance
is 0.5% by default. Candidates start `verified=False`; `.resolve()` checks
storage and returns a confirmed copy. `.open()` runs the same checks. Missing
levels, contradictory spacing, and contradictory shapes fail explicitly.
Neither method resamples CT. `volume.level(n).open()` selects a stored level
explicitly, using its actual storage transform for aligned sampling.

## Flattened paired reads

```python
with label.open_paired(layers=slice(40, 65), backend='cpu') as reader:
    patch = reader[18576:18704, 7984:8112]
    image = patch.image.data       # C,D,H,W
    ink = patch.label.data         # C,H,W
    valid = patch.label.valid      # 1,H,W geometric support
    masks = patch.label.masks      # published supervision/validation values
    one_layer = reader['image'][0, -1, :32, :32]
```

The default is image level zero and all stored layers. `image=` can be omitted
only for exactly one paired image. Shared indexing is the selected image's
YX grid. Named indexing is CDYX for images and CYX for labels. An integer
removes its axis; negative integers and unit-step slices follow NumPy rules.
A `layers` integer removes the depth axis. `.read(start_yx, size_yx)` permits
out-of-bounds requests with fill values and separate geometric validity;
`.read_many(...)` returns results in request order.

Published storage transforms resolve pixel grids. Labels and companion masks
select matching stored grids independently; a missing matching grid uses the
known affine and exact categorical nearest-neighbor sampling (ties to even).
Level numbers across different pyramids are not assumed equivalent. Unknown
transforms fail when required to relate grids. Layer depth is never interpreted
as original-volume Z and 2D labels are never repeated through it.

Companion values stay separate from `valid`: supervision and validation do not
become an eligibility rule. The authoritative artifact `properties.arrays`
list determines published companions. Creation reports can describe staging
outputs not copied to the published package. In the acceptance snapshot that
list names supervision and validation; a creation report also mentions an
unavailable mapping-validity output, which is not attached as a published mask.

Paired reads and volume-aligned reads share bounded CPU/CUDA affine execution,
chunk/decoded caches, tile and plan budgets, and output ownership. Configure
`memory_cache`, `decoded_cache_bytes`, `sampling_cache_bytes`,
`sampling_work_bytes`, and `plan_cache_bytes` when opening. `stats()` reports
I/O, cache bytes, and CUDA upload bytes. Results survive later reads and close.
CPU readers serialize descriptors and reopen in workers; CUDA readers cannot
be pickled or used after fork. CUDA belongs in the device-owning process.

## Local and external sources

`scroll.with_source(source, kind=..., volume=..., segment=...)` returns an overlay
view. Kinds are `volume`, `image`, `label`, `representation`, `geometry`, and
`artifact`. Sources use `user:` IDs, as do newly created volume/segment IDs.
Published metadata is unchanged and published descriptors are shared.

For a custom label/image pair supply `image=image.id` when attaching the label
(or `label=label.id` when attaching an image), plus exactly one of:

- `same_grid=True`: a declared shared base pixel grid, validated at opening.
- `pixel_affine=((a,b,tx),(c,d,ty))`: label base XY pixels to image base XY pixels.

Owning the same segment or having the same shape does not pair custom sources.
Unregistered sources remain inspectable and raw-readable. Volume-aligned
annotations use `VolumeRegistration`; `volume.open_aligned({'label': asset},
level=0)` uses the existing volume sampling engine. Surface projection remains
future work.

```python
from vc_reader import Affine, Source, Storage, VolumeRegistration
volume = catalog.scrolls['PHercParis4'].volumes['20230205180739']
source = Source('crop', Storage('/data/labels.zarr', 'zarr'), role='label',
                axes=('z', 'y', 'x'),
                registration=VolumeRegistration(volume.sample_id, volume.id,
                    affine=Affine.from_rows(((1,0,0,120), (0,1,0,80), (0,0,1,40)))))
view = volume.scroll.with_source(source, kind='label', volume=volume)
with volume.open_aligned({'label': view.labels['user:crop']}) as reader:
    result = reader[40:44, 80:84, 120:124]
```

`Source.to_dict()` / `Source.from_dict()` serialize individual descriptors.
`ScrollSource.to_dict()` / `ScrollSource.from_dict()` include the catalog,
overlays, relationships, and volume context without live readers. Pickling
views also preserves those relationships.

## Run and verify

Run [`scripts/paired_reading.py`](../scripts/paired_reading.py) with `--metadata`
and optionally `--native-dir` pointing to an existing build. It discovers the
acceptance pair, reads the fixed crop, verifies all arrays independently, and
asserts 9,926 nonzero ink pixels. `--backend cuda` repeats on CUDA.
The CLI supports `samples`, `scrolls`, `volumes`, `labels`, `segments`, `images`,
and `representations`; `volumes --resolution 8 --rel-tol .02 --resolve` validates
header selections. See [exact test/benchmark commands and results](scroll-reading-benchmarks.md).
