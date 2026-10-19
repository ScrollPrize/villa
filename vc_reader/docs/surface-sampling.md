# Surface layers and label-patch projection

`volume.open_aligned(...)` and `vc_reader.sampling.open_sample(...)` support
registered surface targets on CPU and CUDA. Projection starts from a **2D label
patch** and derives the corresponding 3D volume crop.

## Project a label patch into CT

```python
from vc_reader.sampling import open_sample

# reference is a registered image Source or VolumeLevel.
# ink is a label Source with a SurfaceRegistration and explicit pixel_to_uv.
surface = ink.registration.surface
with open_sample(reference, {"ink": ink}, backend="cpu") as reader:
    sample = reader.project_surface(
        surface, start=(100, 200), size=(128, 128), label_target="ink",
        thickness_um=8, padding_um=32,
    )
    ct = sample.image.data                 # (C, Z, Y, X)
    labels = sample["ink"].data               # (C, Z, Y, X), stored integer type
    coverage = sample["ink"].masks["coverage"] # (1, Z, Y, X)
    conflicts = sample["ink"].masks["conflict"]
    origin = sample.metadata["projection"]["origin_zyx"]
```

`start` and `size` select YX pixels in the **opened label grid**, including its
selected pyramid level. `label_target` selects which named target defines that
patch. All configured surface annotations must register to the selected surface;
they remain separate named outputs with independent pixel registrations.

The reader maps the patch footprint through UV into connected 3D geometry,
clips geometry to that footprint **to calculate the enclosing reference-volume
voxel bounds**, then adds `padding_um` of physical CT context on every side.
Within that fixed crop, it projects the complete registered surface, including
sheet outside the selecting rectangle and folds that return from distant UV
regions. Other surface IDs are not included. The selecting rectangle determines
the crop, not an artificial boundary in the sheet. `thickness_um` sets the
**total** physical label-band width. Output shapes depend on surface geometry.

`sample['ink'].masks['coverage']` is geometric sheet coverage, independent of
label availability and values. `sample['ink'].valid` additionally requires valid
label support. A zero label with valid support is a negative; a fill value where
label validity is false is unknown, not background. Source masks such as
`supervision` retain their values at the winning labeled surface point and do not
alter geometric coverage. For ink training, intersect label validity, image
validity, supervision and the complement of `conflict` to form the loss mask. CT validity independently records volume support. Context beyond the
volume boundary receives the ordinary reader fill and false image validity.
No supervision policy, ignore index, loss mask, dataset split, fixed training
shape, or collation rule is selected by the library.

Metadata includes the selecting label and pixel patch, surface ID, UV mapping,
destination reference/level, crop shape and ZYX origin, crop-to-reference affine,
reference-to-level-0 affine, thickness, padding and geometry preparation time.
The public Source descriptors retain their native frames.

## Supply explicit geometry and registration

```python
from vc_reader import Affine, Source, Storage, Surface, SurfaceRegistration

surface = Surface(
    "my-surface", Storage("/data/surface", "tifxyz"), "sample-id", "volume-id",
    axes=("y", "x"), component_order=("x", "y", "z"),
    xyz_to_volume=Affine.scale(4),  # stored XYZ values are level-2 voxels
    spacing_um=(9.6, 9.6, 9.6),    # physical spacing of those coordinate values
)
ink = Source(
    "my-ink", Storage("/data/ink.zarr", "zarr"), role="label", axes=("y", "x"),
    registration=SurfaceRegistration(surface, pixel_to_uv=((1, 0, 0), (0, 1, 0))),
)
```

The identity pixel-to-UV map in this example is an explicit declaration; use
the actual render registration for your data. Shape, filename, matching render
resolution, or segment membership cannot establish that relationship.
Discovery's `mappable_volumes` continues to report `surface_projection` until
projection is explicitly requested. A transformed tifxyz geometry Asset can be
passed directly when discovery provides its native Surface registration;
other geometry IDs must uniquely resolve to a configured target's descriptor.

Tifxyz opens `x.tif`, `y.tif`, `z.tif`, and `meta.json` through the shared storage
readers. Its `scale` is **XY**: stored column/row coordinates equal native UV
coordinates multiplied by those respective scales. Coordinate-value units are
independent of UV scale. An explicitly supplied surface `pixel_to_uv` must
agree with that metadata. Nonfinite/negative tifxyz coordinates and an optional
`mask.tif` mark invalid geometry. Set `tifxyz_mask=False` when the bundle is
known to have no mask file, particularly for offline reopening: a missing cache
entry does not prove that a remote mask does not exist.

Equivalent coordinate grids use a registered Surface with `geometry=Storage(...)`,
three coordinate channels, explicit axes/component order and `pixel_to_uv`.
The coordinates remain in the frame declared by `xyz_to_volume`; normals and
micrometer offsets use `spacing_um` in that coordinate-value frame. Surface
masks and explicit normals are Sources with their own SurfaceRegistrations.
Explicit normals also require a volume `ValueFrame` and component order;
cross-volume values require a known frame transform on the aligned reader. Unknown
geometry frames and target UV registrations fail when opening aligned targets.
Raw browsing and Source reads remain available for unresolved assets.

Use `scroll.with_source(...)` for custom overlays. Surface descriptors, including
`tifxyz_mask`, serialize through the existing Source/overlay machinery.

## Sample physical surface layers

```python
with open_sample(reference, {"ink": ink}, backend="cuda") as reader:
    sample = reader.render_surface(
        surface, start=(100, 200), size=(128, 128),
        offsets_um=(-19.2, -9.6, 0, 9.6, 19.2),
    )
```

Here `start` and `size` are YX in **native surface UV pixels**, rather than the
label grid. Volume outputs are `(C, D, H, W)` in the supplied offset order;
labels and masks remain `(C, H, W)`. Coordinate grids are bilinearly sampled
with valid quad support and interpolation halos. There is no extrapolation
across holes or beyond the final stored row/column. Normals follow
`dP/du cross dP/dv`, with increasing columns defining u. Degenerate normals are
invalid. Offsets are applied in the surface's physical frame before mapping to
each volume. Geometric target components retain their declared reference-volume
basis, with value-frame metadata in the result.

`vc_reader.surface.SurfaceGeometry` exposes the same bounded UV interpolation
without reading a volume. Its `read_points(...)` returns local XYZ float32
coordinates, geometry validity, and an integer XYZ origin in the stored
coordinate-value frame. The benchmark uses this helper too.

## Projection geometry and determinism

Projection splits each valid coordinate-grid quad along its top-left to
bottom-right diagonal. It clips triangles to the selected label footprint only
when deriving crop bounds; rasterization uses complete triangles intersecting
the resulting 3D crop.
This piecewise-planar surface differs from bilinear layer sampling on curved
quads; they agree on planar geometry.

Zero thickness includes closed voxel cells intersecting a triangle, including
exact face contacts. Positive thickness includes cells within half the requested
width of a triangle in the reference physical metric, including edge and corner
caps. Reference physical spacing is required. Invalid or degenerate triangles
contribute nothing. A patch without valid nondegenerate geometry raises
`ValueError` because no spatial crop can be derived.

For each contributing triangle/voxel, the closest surface point to the voxel
center supplies barycentric UV coordinates. Labels use nearest-neighbor
sampling with ties-to-even and exact integer comparisons, including multichannel
values. Invalid label support contributes no label validity, but geometric
coverage remains true where the surface intersects a voxel. Overlapping contributions
select the smallest physical distance, then the stable triangle ID; each named
target has a single source identity. Disagreeing values set `conflict` while
retaining the selected value. Even coplanar neighboring triangles can disagree
near categorical boundaries under conservative cell coverage.

## Bounded output, caching and workers

```python
with open_sample(reference, {"ink": ink}, surface_output_bytes=128 << 20) as reader:
    for sample in reader.iter_project_surface(
        surface, (100, 200), (1024, 1024), label_target="ink",
        tile_size=(64, 64, 64), thickness_um=8, padding_um=32,
    ):
        consume(sample)  # each tile carries its own origin and full crop bounds
```

The dense result is checked against `surface_output_bytes` (default 512 MiB)
before allocating its data. The iterator emits deterministic ZYX tiles,
including empty context tiles; its results reproduce dense projection.
Output accounting is conservative and separate from working/cache budgets.

Crop selection visits the selected UV region and halos. The first projection
also scans the registered coordinate grid in bounded blocks to build a spatial
index; this is necessary to find sheet regions that fold back into a crop. It
reads geometry, not the complete CT or label image. The index is reused across
patches while retained by `plan_cache_bytes`. Only intersecting leaves are
triangulated, and triangle bounds are filtered before reading label pixels.
Index metadata and coordinate blocks obey `sampling_work_bytes`; an index too
large for that budget fails explicitly. If the index exceeds the cache budget,
it is rebuilt for the next projection. Decoded coordinate and label tiles share
`sampling_cache_bytes`; candidate batches obey `sampling_work_bytes`. Extremely small budgets that cannot hold a geometry tile
or a single sampling stencil fail explicitly. TIFF decoding retains its existing
separate decoded-cache and maximum-decode limits.

CPU output uses NumPy; CUDA output uses the existing owned `DeviceArray`/DLPack
interface. Coordinate/UV planning metadata can cross the CPU/device boundary;
volume and categorical sampling, normal/offset interpolation, rasterization,
and collision selection have CUDA kernels. File decoding and spatial indexing
remain host work. CUDA kernels synchronize before returning, as in the existing
reader. No PyTorch or SciPy runtime dependency is introduced.

CPU readers reopen in workers after serialization/fork. CUDA readers must live
in the device-owning process. Returned arrays survive later reads and reader
close. `stats()` reports shared cache usage, geometry payload reads/transfers,
candidate uploads and preparation/index/raster/output-allocation wall times.
Raster-call wall time includes dispatch and uploads. Use a CUDA profiler to
measure resident kernel execution separately.
