# vc_reader

Find and locate Vesuvius Challenge open data from the published
[`metadata.json`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json)
(schema: [`metadata.schema.json`](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.schema.json)).

This first version covers the catalog only. It fetches and parses the metadata
and answers "which volume / level / labels / predictions, and where are they".
Reading the zarr, tif and png data itself comes next. The catalog layer uses
only the standard library.

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
python -m pytest              # offline, against tests/data/metadata_subset.json
python -m pytest -m network   # parse the live metadata.json end to end
```

`tests/data/metadata_subset.json` holds real entries trimmed from the published
file. Regenerate it with `python tests/make_fixture.py path/to/metadata.json`.

The cache directory can be overridden with `VC_READER_CACHE_DIR`. A cached copy
younger than `max_age` (default 1 h) is used as is. After that the copy is
revalidated with the ETag, and it is used offline if the server cannot be
reached.
