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
    ct = sample.data["image"]                 # (C, Z, Y, X)
    labels = sample.data["ink"]               # (C, Z, Y, X), stored integer type
    coverage = sample.masks["ink"]["coverage"] # (1, Z, Y, X)
    conflicts = sample.masks["ink"]["conflict"]
    origin = sample.metadata["projection"]["origin_zyx"]
```

`start` and `size` select YX pixels in the **opened label grid**, including its
selected pyramid level. `label_target` selects which named target defines that
patch. All configured surface annotations must register to the selected surface;
they remain separate named outputs with independent pixel registrations.

The reader maps the patch footprint through UV into connected 3D geometry,
clips geometry to that footprint, and calculates the enclosing reference-volume
voxel bounds. It reads CT and configured volume targets over those bounds.
`padding_um` adds physical CT context on every side; `thickness_um` sets the
**total** physical label-band width. Padding never pulls unrelated surface
regions into the annotation. Output shapes depend on the surface geometry.

`valid['ink']` equals coverage. A covered zero label is a valid negative;
uncovered voxels receive the source fill value. Source masks such as
`supervision` retain their values at the winning surface point and do not alter
coverage. CT validity independently records volume support. Context beyond the
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
bottom-right diagonal and clips those triangles to the selected label footprint.
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
values. Invalid label support contributes no coverage. Overlapping contributions
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

Geometry preparation visits only the selected UV region and halos. A cached BVH
indexes conservative tile bounds when its metadata fits `plan_cache_bytes`;
otherwise bounds are streamed over the selected region. Decoded coordinate and
label tiles share `sampling_cache_bytes`; candidate batches obey
`sampling_work_bytes`. Extremely small budgets that cannot hold a geometry tile
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
Raster-call wall time includes dispatch and uploads; use the separate CUDA
profile in the [validation report](surface-sampling-benchmarks.md) for resident
kernel timings.
