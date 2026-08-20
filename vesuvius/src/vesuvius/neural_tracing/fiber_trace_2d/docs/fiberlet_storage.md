# Precomputed Fiberlet Storage Proposal

This is a proposed compressed structure-of-arrays format. It is not implemented
and is not a compatibility contract. Cost and field compression must be
validated on representative extracted data before the format is fixed.

## Quantization experiment result

The final 2026-08-19 Paris4 benchmark traced the approximately 46,148-base-voxel
reference interval from its first control point to the end. Baseline extraction
produced 223,483 successful fiberlets over 22,155 graph anchors in 43.64 seconds
with 32 threads. Seven additional position/direction geometries reran the normal
DP. The complete 16-scenario command took 409.15 seconds. `P1/P2/P4` mean
position quanta of 1/2/4 base voxels, `D` means compact fitted direction, and
`C8/C16` mean per-chunk 8/16-bit total costs:

| Scenario | Failures | Delta | Max line | Max normal | Max tangential |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1 | 0 | 0.00 vx | 0.00 vx | 0.00 vx |
| P1 | 1 | 0 | 34.00 vx | 8.09 vx | 33.93 vx |
| P2 | 1 | 0 | 11.20 vx | 8.95 vx | 11.18 vx |
| P4 | 1 | 0 | 14.07 vx | 12.71 vx | 11.35 vx |
| D | 1 | 0 | 10.89 vx | 7.01 vx | 10.66 vx |
| P1+D | 1 | 0 | 34.08 vx | 7.39 vx | 33.99 vx |
| P2+D | 1 | 0 | 13.88 vx | 8.94 vx | 13.63 vx |
| P4+D | 1 | 0 | 14.07 vx | 12.74 vx | 11.28 vx |
| C8 | 1 | 0 | 8.76 vx | 4.89 vx | 8.76 vx |
| C16 | 1 | 0 | 0.00 vx | 0.00 vx | 0.00 vx |
| P1+D+C8 | 1 | 0 | 34.08 vx | 7.57 vx | 34.00 vx |
| P1+D+C16 | 1 | 0 | 34.08 vx | 7.39 vx | 33.99 vx |
| P2+D+C8 | 1 | 0 | 13.88 vx | 8.93 vx | 13.63 vx |
| P2+D+C16 | 1 | 0 | 13.88 vx | 8.94 vx | 13.63 vx |
| P4+D+C8 | 1 | 0 | 15.04 vx | 12.75 vx | 14.87 vx |
| P4+D+C16 | 1 | 0 | 14.07 vx | 12.74 vx | 11.28 vx |

All scenarios completed the interval and none increased the baseline failure
count. `C16` reproduced the baseline line exactly. The other representations
changed the selected geometric line without creating another threshold failure.
The position effect is non-monotonic because quantization changes the DP and
beam choices, not merely individual coordinates.

Compact direction stores the anchor's unoriented fitted axis as the existing
Lasagna `nx/ny` bytes. The axis is normalized and sign-canonicalized to the +Z
hemisphere, X/Y are rounded as `component * 127 + 128`, and nonnegative Z is
reconstructed from the unit-length constraint. It costs 2 bytes instead of
three float32 components (12 bytes). Across 22,519 retained full-fiber anchors,
its mean/max unoriented angular errors were 0.709/5.672 degrees.

Coordinate variants remained within the two-variant limit. With 512-base
chunks, `P1` needs 16-bit local positions and endpoint deltas; `P2` and `P4`
fit 8-bit fields. `P2` and `P4` therefore have the same raw record size, though
`P4` may compress differently. Costs require 1 byte for `C8`, 2 bytes for
`C16`, or 4 bytes when left float32, plus one 8-byte offset/scale pair per
populated chunk.

For the current whole-volume Paris4 density estimate, the complete proposed
raw layouts with compact directions are approximately:

| Layout | Raw estimate before block compression |
| --- | ---: |
| P1+D, float32 cost | 149.5 GiB |
| P2/P4+D, float32 cost | 121.8 GiB |
| P1+D+C8 | 136.3 GiB |
| P1+D+C16 | 140.7 GiB |
| P2/P4+D+C8 | 108.6 GiB |
| P2/P4+D+C16 | 113.0 GiB |

Position-only rows retain a 12-byte float32 direction instead of the 2-byte
compact direction, adding about 4.3 GiB to those whole-volume estimates.
Direction-only and cost-only rows do not select a position/reference layout,
so they have no standalone complete-format total: compact direction saves 10
bytes per anchor, `C8` saves 3 bytes per fiberlet versus float32 cost, and `C16`
saves 2 bytes per fiberlet. Baseline likewise is a numerical reference rather
than a proposed persisted layout.

Position quantization therefore needs further quality investigation before it
can be selected. The experiment intentionally reruns the normal DP rather than
forcing a baseline lattice route onto changed Hermite planes.

Line distance is a symmetric sampled Hausdorff-style measurement over the
disconnected replay polylines. Samples are at most one base voxel apart and are
projected onto the other line's actual segments in both directions. Normal and
tangential maxima use the Lasagna normal at the projected target point; their
maxima need not occur at the same sample as the Euclidean maximum.

## Design rules

- Anchors and fiberlets are separate records. Fiberlets reference anchors.
- This is one fiberlet dataset format with selectable numeric encoding profiles.
  The float32 cache and compact quantized forms use the same logical anchor,
  fiberlet, route, chunk, and graph schema. A dataset declares one profile;
  readers never infer it from field widths.
- Anchor chunks may be embedded in fiberlet chunk payloads or stored in a
  separate Zarr array. The adaptive float32 cache uses separate anchor chunks so
  anchors can be completed once and fiberlet chunks can be filled or replaced
  independently.
- Fiberlets form one logical list. Every field is stored as a parallel list.
- The logical lists are split into independently compressed spatial chunk blocks
  so tracing can load a local neighborhood.
- Fields use ordinary `uint8`, `int8`, `uint16`, `uint32`, and `float32` values.
- Sorting, delta transforms, byte shuffling, and block compression remove
  redundancy.
- Compact-profile endpoints use quantized coordinates plus a variant index;
  float32-profile endpoints use stable cell/component identity. In both cases the
  second endpoint is stored as a signed coordinate delta from the first.
- Scalar widths are fixed per dataset and have no per-value escape
  representation.
- Anchor positions may use a one-, two-, or four-base-voxel storage quantum.
- The outer store is a Zarr v2 group and uses ordinary Zarr spatial chunk keys.
  Chunk values are custom object payloads, not numeric Zarr tensor bytes.
- The custom payload codec and sample format are mandatory metadata. A generic
  Zarr implementation may inspect the JSON metadata and copy, delete, or move
  raw chunk objects, but cannot decode the arrays without the VC codec.

## Encoding profiles

### Float32 cache profile

`float32_cache` is a lossless cache of the current non-quantized anchor
and successful-fiberlet results. Its purpose is bounded-memory extraction,
replay, and quantization experiments; it is not the compact production layout.

- Anchor identity is the existing stable `(cell_z, cell_y, cell_x,
  component_index)` identity. `component_index` is strictly `0` or `1`.
- Anchor position and fitted axis are stored as the exact float32 prediction-
  space values produced by extraction. They are not rounded to base voxels and
  are not passed through the compact direction codec.
- The successful candidate is identified by its ordered pair of stable endpoint
  identities; no separate global candidate index is stored. The accumulated
  float32 path length and float32 total edge cost are stored directly.
- The DP route uses the same entry/exit/middle `(u,v)` lattice arrays as the
  compact profile. Those integer lattice choices are already the exact
  DP result and do not become more exact by expanding them into redundant XYZ
  samples. XYZ is reconstructed from the float32 endpoints/axes and the
  fingerprinted path configuration.
- Extraction diagnostics and rejected candidate records are build output, not
  graph data, and are not persisted in the cache.

### Compact quantized profile

`compact_quantized` is the proposed space-efficient profile described
by the quantized `AnchorArrays`, `FiberletSearchArrays`, and cost encoding
below. Its anchor key is quantized global position plus variant, its fitted axis
uses the compact two-byte encoding, and its total cost is `uint8` or `uint16`.
It remains experimental until the wide-corridor evaluation is complete.

Both profiles use the same Zarr envelope, spatial ownership rule, chunk header,
logical field IDs, field-descriptor mechanism, route-array ordering, source/
configuration fingerprints, reader, and graph construction. Profile metadata
selects the physical scalar and anchor-reference encodings for the same logical
fields. Mixing profiles inside one dataset is an error; it does not create a
second storage format or second loader.

The profile differences are limited to these physical encodings:

| Logical field | `float32_cache` | `compact_quantized` |
| --- | --- | --- |
| Anchor identity/reference | Stable cell plus component, integer delta for second endpoint | Quantized position plus variant, integer delta for second endpoint |
| Anchor position | `float32[3]` prediction coordinates | `uint8`/`uint16[3]` base-coordinate bins |
| Fitted axis | `float32[3]` | Existing two-byte direction encoding |
| Route count and entry/exit/middle U/V | Shared lossless integer lattice fields | Shared lossless integer lattice fields |
| Path length | `float32` | `float32` |
| Total edge cost | `float32` | Per-chunk `uint8` or `uint16` affine code |

All other logical fields and access behavior are shared. The field descriptor
maps each logical field ID to the profile-selected physical scalar type.

## Dataset structure

```text
dataset.zarr/
|- .zgroup
|- .zattrs                         # DatasetHeader
|- anchors/                        # present for separate-anchor layout
|  |- .zarray
|  |- .zattrs
|  `- <z>.<y>.<x>                  # custom AnchorChunk payload
`- fiberlets/
   |- .zarray
   |- .zattrs
   `- <z>.<y>.<x>                  # custom FiberletChunk payload
```

In the inline-anchor layout the `anchors/` array is absent and every committed
`fiberlets/<z>.<y>.<x>` payload contains `AnchorArrays` before its fiberlet
arrays. An anchor-only inline chunk is therefore not header-only. A chunk is
header-only only when both `anchor_count` and `fiberlet_count` are zero. In the
separate-anchor layout, fiberlet payloads contain no anchor arrays and reference
the corresponding independently loaded anchor chunks. Resolving a cross-chunk
second endpoint in inline layout loads the destination fiberlet chunk because
that chunk owns the destination anchor arrays.

Inline layout is valid only for a finite, complete dataset: readers reject
inline `partial` or `building` datasets. Adaptive and on-demand caches therefore
use separate layout, where all required anchor chunks complete before any
fiberlet chunk is published. This prevents a committed edge from depending on
an unavailable destination-anchor chunk without duplicating that anchor.

Each custom chunk contains a header followed by independently compressed field
blocks. A codec can therefore use the transform appropriate for each field
without requiring padded C++ records.

`FiberletPointArrays` are physically last in every chunk. The chunk header and
field descriptors precede the search arrays, so a range reader can load anchors,
connectivity, costs, lengths, and endpoint geometry without reading or
decompressing the complete interior paths.

## Zarr envelope

The initial envelope is Zarr v2 because that is the store layout already used
by the surrounding VC tooling. The root is a normal group:

```json
{"zarr_format": 2}
```

Each logical `anchors` or `fiberlets` array is a sparse three-dimensional
object array over the spatial chunk grid. For example, a grid containing
`128 x 256 x 256` spatial chunks uses:

```json
{
  "zarr_format": 2,
  "shape": [128, 256, 256],
  "chunks": [1, 1, 1],
  "dtype": "|O",
  "fill_value": null,
  "order": "C",
  "filters": [
    {
      "id": "vc-fiberlet-chunk",
      "codec_version": 1,
      "sample_format": "fiberlet-anchor-v1"
    }
  ],
  "compressor": null
}
```

`sample_format` describes logical chunk kind and is exactly one of
`fiberlet-anchor-v1`, `fiberlet-edge-v1`, or `fiberlet-inline-v1`. The dataset's
`encoding_profile` and per-field descriptors select float32 versus compact
physical encodings without changing the sample format.
The named object codec owns the complete custom payload, including its internal
per-field transforms and compression. The outer Zarr compressor is `null` to
avoid recompressing an already block-compressed payload.

Each array `.zattrs` repeats its `encoding_profile`, `chunk_kind`,
`sample_format`, and `dataset_fingerprint`, and declares
`_ARRAY_DIMENSIONS: ["chunk_z", "chunk_y", "chunk_x"]`. These values must
agree with the root attributes and custom chunk headers.

This is a Zarr extension, not a numeric/object array that stock readers can
decode. A reader without `vc-fiberlet-chunk` may parse `.zgroup`, `.zattrs`, and
`.zarray`, enumerate sparse chunk keys, and transport their raw bytes. It must
fail clearly if asked to decode an element. It must not interpret a payload as
one byte, a dense tensor, or a standard variable-length-byte codec.

The chunk key is the Zarr array index in `z.y.x` order. No separate coverage or
completion tensor duplicates chunk state:

- a missing chunk is not cached;
- a present validated header-only chunk was computed and is empty;
- a present validated nonempty chunk contains cached records.

In inline layout a chunk with anchors and zero fiberlets is nonempty and retains
its anchor blocks. For a finite whole-volume, crop, or reference-tube build, the
root `build_domain` metadata defines the expected chunk keys deterministically.
Resume recomputes those keys and tests final-file presence. A cache used purely
on demand may omit `build_domain` and remain intentionally partial.

The current scope is a local filesystem cache. A writer encodes to a sibling
temporary file, validates it, calls `fsync`, renames it to the final key, and
calls `fsync` on the parent directory.
Only one writer owns a dataset build. A crash before rename leaves only a
temporary file; a crash after rename leaves a complete reusable final chunk.
Resume never accepts partial bytes.

The raw custom chunk object begins with the uncompressed fixed header and field
descriptor table, followed by independently compressed field blocks. Those
bytes are exactly the encoded value owned by `vc-fiberlet-chunk`; no outer codec
may wrap or reorder them. Store-level range readers can therefore inspect the
fixed prefix without invoking a Zarr decoder. A Zarr implementation that does
not know the object codec may reject opening the array; a generic filesystem or
Zarr-metadata tool can still enumerate and copy its raw keys.

## DatasetHeader

The header is stored in root `.zattrs`, using JSON types rather than a binary
record. All coordinate order and units are explicit.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `vc_format` | string | `fiberlet_dataset` |
| `format_version` | integer | Exact storage-format version |
| `encoding_profile` | enum | `float32_cache` or `compact_quantized` |
| `anchor_layout` | enum | `separate` or `inline` |
| `base_shape_zyx` | integer[3] | Base-volume bounds |
| `spatial_chunk_side_base` | integer | Cubic spatial chunk side, initially `512` |
| `chunk_grid_shape_zyx` | integer[3] | Zarr array shape; ceil-divided from base shape |
| `build_domain` | object or null | Optional whole grid, crop, or reference-tube request from which expected chunk keys are reproduced |
| `build_state` | enum | `partial`, `building`, or `complete`; complete requires a finite `build_domain` |
| `prediction_to_base` | JSON number / binary `float64` | Base voxels per prediction voxel at source precision |
| `prediction_to_base_ratio` | `uint64[2]` | Reduced numerator/denominator used for exact ownership arithmetic |
| `prediction_shape_zyx` | integer[3] | Bounds of the sampled prediction grid |
| `prediction_grid_origin_base_zyx` | integer[3] | Base-coordinate origin of prediction index zero; initially required to be integral |
| `coordinate_space` | string | Float32 profile uses `prediction_xyz_float32`; compact uses `base_xyz_quantized` |
| `anchor_position_quantum_base` | integer or null | Compact base voxels per stored anchor-position unit: `1`, `2`, or `4`; null for float32 profile |
| `anchor_position_type` | enum or null | Compact `uint8`/`uint16`, derived from chunk side and quantum; null for float32 profile |
| `anchor_delta_type` | enum or null | Compact `int8`/`int16`, derived from search range and quantum; null for float32 profile |
| `float_cell_type` | enum or null | Float32-profile chunk-local cell scalar; null for compact profile |
| `float_cell_delta_type` | enum or null | Float32-profile endpoint cell-delta scalar; null for compact profile |
| `route_count_type` | enum | `uint8` or `uint16`, validated from maximum layer count |
| `route_lattice_type` | enum | Signed integer type validated from transverse radius/step |
| `fiber_manifest` | string | Dense presence and direction source |
| `normal_manifest` | string | Lasagna normal source |
| `fiber_manifest_hash` | string | Content identity used for cache invalidation |
| `normal_manifest_hash` | string | Content identity used for cache invalidation |
| `algorithm_fingerprint` | string | Exact extractor/DP semantics used to produce the cache |
| `producer_numeric_fingerprint` | string | Compiler, architecture, floating-point mode, and relevant library identity for bit-exact cache validation |
| `anchor_cell_side_prediction` | `uint8` | Anchor cell side |
| `anchor_cell_radius_prediction` | `float64` | Cell selection radius used by anchor extraction |
| `anchor_neighborhood_margin_prediction` | `float64` | Extra sampled support around selected anchor cells |
| `max_anchors_per_cell` | `uint8` | Fixed extraction invariant, initially `2` |
| `maximum_endpoint_reach_cells_zyx` | integer[3] | Checked maximum candidate endpoint displacement in anchor cells |
| `maximum_owner_halo_chunks_zyx` | integer[3] | Checked maximum chunks needed to resolve candidate endpoints and anchors |
| `maximum_anchor_displacement_base` | `float64` | Checked bound from owning cell to refined anchor position for area-query expansion |
| `maximum_endpoint_angle_degrees` | `float64` | Candidate endpoint-axis acceptance bound |
| `maximum_prediction_deviation_degrees` | `float64` | Per-step dense-direction deviation bound |
| `graph_max_join_angle_degrees` | `float64` | Graph transition eligibility bound |
| `longitudinal_step_prediction` | `float64` | DP layer spacing |
| `transverse_step_prediction` | `float64` | DP transverse spacing |
| `corridor_radius_prediction` | `float64` | DP transverse bound |
| `loss_config` | fixed structure | Alignment and smoothness weights |
| `cost_encoding` | enum | `float32`, `uint8`, or `uint16` |
| `field_codec` | object | Per-field block compressor and transform version |

Dense presence, prediction directions, and Lasagna normals remain external
source volumes. They are sampled on demand and are not copied into every anchor
or fiberlet.

The complete serialized `FiberAnchorConfig`, `FiberletPathConfig`, and graph
configuration are included at their original scalar precision in addition to
the summary fields above. This includes candidate shell distance/radius,
sampling margins, endpoint and per-step angular limits, lattice bounds, all
loss weights, and join-angle settings. Unknown configuration fields and any
summary/config mismatch invalidate the cache; no reader fills defaults.

The float32 cache preserves produced float32 fields and route geometry, but graph
join arithmetic is still executed by the consumer. A consumer may claim
bit-exact equivalence only when `producer_numeric_fingerprint` matches. A
different build may explicitly opt to consume the stored geometry as data, but
its recomputed join costs are a new evaluation and must not be reported as a
float32 cached baseline.

There is no separate `SpatialChunkIndex` with file offsets. Zarr chunk keys are
the spatial index, filesystem metadata supplies encoded byte length, and the
chunk header supplies integrity and semantic validation. Only cached keys are
materialized. A finite build domain defines which keys are expected at
completion; completed empty spatial chunks use validated header-only payloads
according to the inline/separate rule above.

## Portable codec contract

The `*-v1` sample-format names in this proposal are provisional; they must not
be emitted by code until a byte-level appendix is frozen. That appendix must:

- assign the exact eight-byte magic, numeric values for every enum/scalar/
  transform/compressor/field ID, and the complete fixed-header and fixed-width
  descriptor offsets;
- use little-endian integers, explicit two's-complement signed integers, and
  IEC 60559/IEEE-754 binary32/binary64 bit patterns; never serialize a native
  C++ struct, padding byte, `size_t`, `bool`, or compiler enum layout;
- define `header_bytes` as the first field-block offset and make every offset,
  encoded length, decoded length, and element count uint64 with checked
  arithmetic and non-overlap validation;
- select one checksum algorithm and define its exact byte scope, including how
  the checksum field itself is treated; and
- provide checked-in golden payloads for every sample format, including empty,
  anchor-only inline, cross-chunk, and multi-block cases. Ubuntu/macOS and
  amd64/arm64 tests must encode the same bytes and decode the same field bit
  patterns.

Changing any of those definitions before implementation replaces the draft
identifier rather than adding a compatibility decoder. Once implementation is
accepted, the frozen appendix and golden bytes define `v1`.

## ChunkHeader

| Field | Type | Meaning |
| --- | --- | --- |
| `magic` | bytes | Distinguishes VC anchor/fiberlet payloads from ordinary Zarr chunks |
| `payload_version` | `uint32` | Exact custom payload version |
| `header_bytes` | `uint32` | Byte offset of the first field block; bounds the prefix range read |
| `descriptor_count` | `uint32` | Number of fixed-width field descriptors in the prefix |
| `chunk_kind` | enum | `anchors`, `fiberlets`, or `fiberlets_with_anchors` |
| `encoding_profile` | enum | Must match the root encoding profile |
| `chunk_zyx` | `uint32[3]` | Must match the owning Zarr chunk key |
| `owned_cell_origin_zyx` | `int64[3]` | Global anchor-cell index represented by local cell coordinate zero |
| `dataset_fingerprint` | fixed bytes | Hash of sources, algorithm, and numeric configuration |
| `anchor_count` | `uint64` | Anchors in this chunk |
| `fiberlet_count` | `uint64` | Fiberlets stored in this chunk |
| `middle_point_count` | `uint64` | Total interior points stored in the trailing point arrays |
| `field_descriptors[]` | fixed entries | Type, transform, uint64 offset, compressed size, and element count for each field array |
| `payload_checksum` | fixed bytes | Integrity check over header descriptors and encoded field blocks |

Every field descriptor is explicit. Unknown or inconsistent field lengths are
format errors rather than cases handled through escape values.

The payload is complete only when all descriptors, byte ranges, counts,
fingerprint, checksum, and owning key agree. An invalid final chunk is a hard
cache error; it is never treated as a partially usable prefix.

## Float32 profile arrays

### Float32AnchorArrays

Float32-profile anchors are sorted by `(cell_z, cell_y, cell_x, component_index)` within
each spatial chunk. All arrays have `anchor_count` elements.

An anchor's owning spatial chunk is derived from its canonical global cell
index, not from its refined float32 position. Let `c` be one component of the
global cell index, `S` the integer anchor-cell side in prediction voxels,
`N/D` the reduced positive `prediction_to_base_ratio`, `O` the integral base
origin, and `B` the spatial chunk side in base voxels. Ownership is computed
without floating point:

```text
cell_origin_base_numerator = O * D + c * S * N
owner_chunk = floor_div(cell_origin_base_numerator, D * B)
```

`floor_div` has mathematical floor semantics for signed inputs. The
`owned_cell_origin_zyx` header field is the smallest global cell index owned by
that chunk on each axis. Stored cell coordinates are the checked difference
from that origin. The writer proves that converting the metadata float64 scale
to `N/D` is exact and that all arithmetic fits its specified signed/unsigned
integer intermediates. Unsupported non-rational or non-integral-origin mappings
fail format creation. Refinement therefore cannot move an identity between
cache chunks, and ownership is identical across platforms. Compact anchors
retain the later rule that assigns ownership from their decoded quantized
global position.

| Array | Element type | References or meaning |
| --- | --- | --- |
| `cell_x[]` | `uint8` or `uint16` | Cell X relative to the spatial chunk |
| `cell_y[]` | `uint8` or `uint16` | Cell Y relative to the spatial chunk |
| `cell_z[]` | `uint8` or `uint16` | Cell Z relative to the spatial chunk |
| `component[]` | `uint8` | Stable component index, strictly `0` or `1` |
| `position_x[]` | `float32` | Exact extracted prediction-space X |
| `position_y[]` | `float32` | Exact extracted prediction-space Y |
| `position_z[]` | `float32` | Exact extracted prediction-space Z |
| `fitted_axis_x[]` | `float32` | Exact fitted unoriented axis X |
| `fitted_axis_y[]` | `float32` | Exact fitted unoriented axis Y |
| `fitted_axis_z[]` | `float32` | Exact fitted unoriented axis Z |

The cell-coordinate width is selected dataset-wide from the number of anchor
cells in one spatial chunk. Position and axis bit patterns are preserved; the
cache writer does not renormalize, sign-canonicalize, or recompute them.

### Float32FiberletSearchArrays

A float32-profile fiberlet references anchors by stable cell/component identity rather
than by rounded position. The first endpoint is the lexicographically smaller
identity and owns the fiberlet chunk.

| Array | Element type | References or meaning |
| --- | --- | --- |
| `first_cell_x[]` | float cell scalar | First cell X within the storage chunk |
| `first_cell_y[]` | float cell scalar | First cell Y within the storage chunk |
| `first_cell_z[]` | float cell scalar | First cell Z within the storage chunk |
| `first_component[]` | `uint8` | First component, `0` or `1` |
| `second_cell_dx[]` | `int8`, `int16`, or `int32` | Second cell X minus first cell X |
| `second_cell_dy[]` | `int8`, `int16`, or `int32` | Second cell Y minus first cell Y |
| `second_cell_dz[]` | `int8`, `int16`, or `int32` | Second cell Z minus first cell Z |
| `second_component[]` | `uint8` | Second component, `0` or `1` |
| `interior_point_count[]` | `uint8` or `uint16` | Exact number of non-endpoint route points |
| `entry_u[]` | route lattice scalar | First interior transverse U, or zero for no interior point |
| `entry_v[]` | route lattice scalar | First interior transverse V, or zero for no interior point |
| `exit_u[]` | route lattice scalar | Last interior transverse U when present, otherwise zero |
| `exit_v[]` | route lattice scalar | Last interior transverse V when present, otherwise zero |
| `path_length_prediction[]` | `float32` | Exact accumulated path length |
| `cost[]` | `float32` | Exact total edge loss used by graph ranking |

The float32 profile chooses the smallest fixed integer width that represents the
configured cell reach and transverse lattice, then validates every value. This
integer width selection is lossless and is not one of the numeric quantization
variables under evaluation.

### Float32FiberletPointArrays

The float32 profile reuses `FiberletPointArrays` unchanged. Entry/exit values remain
in the search arrays and the trailing `middle_u[]`/`middle_v[]` arrays store the
remaining lattice choices. The route geometry is reconstructed exactly as for
the live DP result from the stored float32 anchor geometry and fingerprinted
configuration. It does not store redundant expanded XYZ points.

### Float32 graph equivalence

Loading a float32 cache resolves each endpoint identity through the anchor
array, reconstructs the route, and builds the ordinary in-memory graph. Tests
must compare endpoint-pair edge identities, float32 total costs and lengths,
integer lattice routes, reconstructed points, join eligibility/costs, replay
failures, and replay geometry against the same uncached extraction. Cache
mismatch is an error, not permission to repair or silently rerun one chunk with
different settings.

The shared in-memory graph path must accept the format's minimal logical edge:
stable endpoint pair/`FiberletId`, authoritative scalar total cost, float32 path
length, endpoint tangents, and lazy route reference. Before the cache lands,
refactor live extraction to construct that same graph edge and rank directly by
the scalar total it already computes. Candidate indices and individual
`FiberletPathCost` components remain optional transient extraction diagnostics,
not required graph fields. Cache-backed replay/artifacts use `FiberletId` and do
not synthesize missing candidate indices or component costs. Regression tests
must prove this graph refactor leaves uncached edge ordering, beam choices,
failures, and replay geometry unchanged.

## Anchor identity

The remaining anchor and fiberlet array definitions describe the compact
physical encodings of the same logical fields.

An anchor is identified by:

```text
AnchorKey = (quantized_global_x,
             quantized_global_y,
             quantized_global_z,
             variant)
```

Position alone is not a sufficient identity. Two direction-specific anchors may
occupy one cell, and position quantization may collapse anchors from neighboring
cells onto the same stored coordinate. `variant` distinguishes all anchors that
share one quantized position.

Variants are assigned deterministically after position quantization by sorting
the colliding anchors by fitted-axis encoding and then by their stable extraction
identity. A position may have at most two anchors, so only variant values `0` and
`1` are valid. The field is physically `uint8` to keep the schema regular. A
quantization setting that collapses more than two anchors onto one stored
coordinate fails validation; there is no larger variant or escape form.

## AnchorArrays

All arrays have `anchor_count` elements.

| Array | Element type | References or meaning |
| --- | --- | --- |
| `position_x[]` | `AnchorPositionScalar` | Quantized X within the chunk |
| `position_y[]` | `AnchorPositionScalar` | Quantized Y within the chunk |
| `position_z[]` | `AnchorPositionScalar` | Quantized Z within the chunk |
| `variant[]` | `uint8` | Anchor variant, strictly `0` or `1` |
| `fitted_axis_x[]` | `uint8` | First byte of Lasagna direction encoding |
| `fitted_axis_y[]` | `uint8` | Second byte; the third component is reconstructed |

`AnchorPositionScalar` is selected dataset-wide:

```text
position_bins = spatial_chunk_side_base / anchor_position_quantum_base

position_bins <=   256: uint8
position_bins <= 65536: uint16
otherwise: configuration error
```

The chunk side must be divisible by the position quantum. Anchor positions are
quantized in global base coordinates before chunk assignment, then stored as
chunk-local units. Decoding is:

```text
position_base = chunk_origin_base
              + stored_position * anchor_position_quantum_base
```

The candidate quantization settings are:

| Stored scale relative to base | Quantum | Maximum per-axis rounding error |
| --- | ---: | ---: |
| `1/1` | 1 base voxel | 0.5 base voxel |
| `1/2` | 2 base voxels | 1 base voxel |
| `1/4` | 4 base voxels | 2 base voxels |

The fitted axis comes from anchor refinement and reconstructs the curved DP
frame. It is not interchangeable with one sample from the dense direction
volume.

No fiberlet count or adjacency field is stored under the anchor. Adjacency is
derived from the explicit endpoint references in `FiberletSearchArrays`.

`AnchorDeltaScalar` is also selected dataset-wide. The encoder calculates the
largest possible absolute quantized coordinate difference from the configured
maximum fiberlet endpoint distance, including the position-rounding bound:

```text
maximum_delta_units <= 127:   int8
maximum_delta_units <= 32767: int16
otherwise: configuration error
```

Every encoded delta is checked against the selected type. Chunk distance is not
stored separately; it follows naturally from the decoded global position.

## Fiberlet identity and ordering

A fiberlet is an undirected edge with two explicit anchor references:

```text
Fiberlet
|- first_anchor:  AnchorKey
`- second_anchor: AnchorKey
```

The lexicographically smaller endpoint is placed in `first_anchor`. The
fiberlet is stored in the spatial chunk containing that first anchor. Its ID is:

```text
FiberletId = (storage_chunk_xyz, fiberlet_index)
```

Within each chunk, fiberlets are sorted by:

```text
(first_z, first_y, first_x, first_variant,
 second_dz, second_dy, second_dx, second_variant)
```

This creates long repeated first-anchor runs and keeps similar second-endpoint
deltas together.

## FiberletSearchArrays

All arrays have `fiberlet_count` elements and row `i` across the arrays describes
one fiberlet.

| Array | Element type | References or meaning |
| --- | --- | --- |
| `first_x[]` | `AnchorPositionScalar` | First anchor X within the storage chunk |
| `first_y[]` | `AnchorPositionScalar` | First anchor Y within the storage chunk |
| `first_z[]` | `AnchorPositionScalar` | First anchor Z within the storage chunk |
| `first_variant[]` | `uint8` | First anchor variant, `0` or `1` |
| `second_dx[]` | `AnchorDeltaScalar` | Second anchor X minus first anchor X |
| `second_dy[]` | `AnchorDeltaScalar` | Second anchor Y minus first anchor Y |
| `second_dz[]` | `AnchorDeltaScalar` | Second anchor Z minus first anchor Z |
| `second_variant[]` | `uint8` | Second anchor variant, `0` or `1` |
| `interior_point_count[]` | `uint8` | Number of interior DP route points |
| `entry_u[]` | `int8` | First interior U coordinate, or zero if there is no interior point |
| `entry_v[]` | `int8` | First interior V coordinate, or zero if there is no interior point |
| `exit_u[]` | `int8` | Last interior U coordinate when there are at least two points, otherwise zero |
| `exit_v[]` | `int8` | Last interior V coordinate when there are at least two points, otherwise zero |
| `path_length_prediction[]` | `float32` | Exact accumulated fiberlet length in prediction voxels |
| `cost[]` | `uint8` or `uint16` | Quantized complete edge loss |

Endpoint keys are reconstructed in quantized global coordinate units:

```text
first.position = storage_chunk_origin_quantized
               + (first_x[i], first_y[i], first_z[i])
first.variant = first_variant[i]

second.position = first.position + (second_dx[i], second_dy[i], second_dz[i])
second.variant = second_variant[i]
```

The second position determines its spatial chunk by integer division and its
chunk-local position by the corresponding remainder. No explicit destination
chunk code or anchor-array index is stored.

The entry and exit transverse coordinates reconstruct the actual first and last
segments in the curved DP frames. They provide the endpoint tangents and segment
lengths used for join scoring without loading the complete interior point
arrays. They are moved out of the trailing geometry rather than duplicated
there. `path_length_prediction` is used directly by beam loss-density ranking.

## FiberletPointArrays

The endpoint anchor positions, fitted axes, and dataset lattice parameters
recreate the curved Hermite centerline and transported transverse frames.

Only interior points between the stored entry and exit points contribute to the
flattened trailing arrays:

| Array | Element type | Meaning |
| --- | --- | --- |
| `middle_u[]` | `int8` | Absolute transverse U lattice coordinate |
| `middle_v[]` | `int8` | Absolute transverse V lattice coordinate |

The geometry for a fiberlet with `N = interior_point_count[i]` is:

```text
N = 0: no interior points
N = 1: entry
N > 1: entry, N - 2 middle points, exit
```

It therefore consumes `max(N - 2, 0)` consecutive elements in each middle-point
array. Prefix-summing that count after a chunk is loaded gives every fiberlet's
range. No persisted point offsets or escape records are required.

The longitudinal DP layer is implicit from the point's position in the
fiberlet. For a middle layer `k`, `(middle_u[offset + k], middle_v[offset + k])`
selects one point in the reconstructed transverse frame. The corresponding
base-space point is:

```text
centerline(k)
+ transverse_frame_u(k) * middle_u[offset + k] * transverse_step
+ transverse_frame_v(k) * middle_v[offset + k] * transverse_step
```

The point arrays store absolute lattice coordinates for clarity. Their
pre-compression transform takes per-fiberlet deltas, resetting at every
fiberlet boundary, so the compressed stream still mostly contains `-1`, `0`,
and `1`.

These arrays contain the actual DP-selected geometry. They are not consulted
when ranking beam candidates. They are loaded only when a selected fiberlet's
polyline must be emitted, visualized, or evaluated point by point.

### Interior path resolution

The current defaults are:

```text
longitudinal step = 2.0 prediction voxels
transverse step   = 0.5 prediction voxels
```

For Paris4 fiber level `/3`, one prediction voxel is 8 base voxels. Therefore:

```text
longitudinal layer spacing = 16 base voxels
transverse lattice spacing =  4 base voxels
```

The stored entry, middle, and exit U/V values choose positions at this transverse
lattice resolution. The reconstructed XYZ points remain floating point because
the Hermite centerline and transported frame are not axis aligned. The final
fiber is a polyline through points spaced roughly 16 base voxels along the fiber,
not a one-base-voxel sample at every step.

Reversing a fiberlet swaps its endpoints and reconstructs the same route in
reverse; geometry is not duplicated.

## Cost encoding

For `compact_quantized`, `cost[i]` stores the quantized total loss of fiberlet
`i`:

```text
invalid_prediction
+ alignment
+ isotropic_smoothness
+ tangent_smoothness
+ normal_smoothness
```

The decoded value is:

```text
decoded_cost = cost_offset + cost_scale * cost[i]
```

The finite chunk minimum and maximum map to the full integer range. No values
are clipped. If every cost is equal, `cost_scale` is zero and `cost_offset` is
that common value.

`cost_offset` and `cost_scale` are parameters of the compact cost field
descriptor. They are absent from the float32 profile rather than occupying
unused common-header fields.

Only the total is stored. Changing loss weights requires regenerating or
rescoring fiberlets. Join costs are computed and cached on demand from the
reconstructed edge tangents, dense fiber prediction, and Lasagna normals.
Explicit pairwise graph transitions are not persisted.

For `float32_cache`, `cost[i]` is the original float32 total used by the graph
and `path_length_prediction[i]` is its original float32 denominator. There is
no offset/scale or decode arithmetic. Individual cost components are not needed
by replay and are not stored.

## Adjacency construction

Neither anchor stores nor owns a fiberlet list. When a chunk neighborhood is
loaded, the consumer scans the explicit first and second endpoint arrays and
builds an in-memory adjacency map:

```text
AnchorKey -> [FiberletId, ...]
```

Only endpoint fields need to be decompressed for this operation. Route, cost,
and dense source data are not touched. The adjacency map is cached with the
loaded neighborhood.

A separate persisted adjacency index can be considered only if measurements
show this scan is material; it is not part of the current proposal.

## Area-local loading

The spatial chunk grid is the on-demand index. A base-coordinate bounding box
maps directly to a small `z.y.x` key range; no dataset-wide index or scan is
required.

- Anchor queries expand by the declared maximum within-cell refinement
  displacement, open the resulting `anchors` chunks, decode only their anchor
  field blocks, and filter decoded positions to the requested box.
- Graph queries expand the requested key range by the declared maximum endpoint
  halo, decode endpoint/cost/length/entry/exit blocks, and filter rows to the
  requested anchors. This includes edges owned by a neighboring first endpoint.
- The trailing middle-route blocks are not read while constructing adjacency or
  ranking the beam. They are decoded only for selected fiberlets whose complete
  geometry is requested.
- A geometric “curve intersects box” query expands by the declared maximum path
  corridor as well as endpoint reach before filtering reconstructed routes.

Parsing cost is therefore the fixed header/descriptor prefix plus only the
requested independently compressed field blocks. Decompression is proportional
to records in overlapping spatial chunks, not the complete dataset. The spatial
chunk side remains a measured tuning parameter: validation must benchmark dense
`128`, `256`, and `512` base-voxel chunks for anchor-only load, graph-prefix
load, selected-route load, decoded bytes, latency, and peak memory. The initial
format adds no finer persisted index unless those measurements show that a
single dense spatial chunk is too expensive to parse.

## Adaptive cache publication and access

The float32 cache is built and consumed spatially; it never requires a complete
`PreparedCandidate[]` or complete fiberlet result list in memory.

1. Extract anchors in bounded spatial batches and publish committed
   `anchors/z.y.x` chunks. Neighbor context may be recomputed while fitting,
   but each canonical anchor is written once to its owning chunk.
2. A fiberlet work batch selects one or more first-anchor owner chunks and
   loads the separate anchor chunks in the configured cell-radius halo.
   Candidate ownership is always the chunk containing the lexicographically
   first endpoint, so cross-chunk edges are generated exactly once.
3. Prepare, sample, and solve candidates under an explicit memory budget.
   Successful results append to per-field segmented spools rather than a
   monolithic owner-chunk vector. Candidate identity and canonical sort keys are
   retained in those spools; a bounded external merge produces deterministic
   field order. The final block compressor streams each sorted field into a
   temporary payload. This remains bounded even when one persisted owner chunk
   is larger than RAM. A checked dataset-wide spatial chunk size and uint64
   descriptor/count limits are used; exceeding a declared format/configuration
   bound fails before publication rather than introducing an escape record.
4. Dense prediction/normal access continues to use the existing volume chunk
   cache. A build-scoped, coordinate-keyed scoring-page cache deduplicates
   interpolation corners across all work batches and may spill completed pages
   to disk. Each unique corner is sampled once per cache build even though
   prepared candidate geometry is released batch by batch.
5. Resume derives expected keys from a finite `build_domain`, or accepts any
   subset for an on-demand partial cache. A present final key with matching
   dataset fingerprint and valid checksum is reused; a missing expected key is
   computed; a conflicting or malformed key is an error. Empty completed chunks
   are explicit header-only objects only where their layout permits it. No
   old-format repair or mixed-profile reuse is attempted.
6. Replay uses a memory-bounded LRU of decoded anchor prefixes, fiberlet search
   prefixes, and selected route blocks. Beam states and reset/seed queries hold
   stable `AnchorKey`/`FiberletId` values, never pointers into evictable chunks.
   Before expanding an anchor, the reader loads every possible first-endpoint
   owner chunk in the declared maximum endpoint-reach halo, then resolves the
   endpoint anchor prefixes. This makes the incident-edge set complete even for
   edges owned by the other endpoint's chunk. Seed lookup loads every chunk
   intersecting its exact spatial query plus that halo. Chunk load/eviction
   order cannot change sorted edge order, costs, beam ties, resets, or results;
   an evicted stable ID is transparently reloaded. Complete-dataset counts are
   streamed from chunk headers without loading route arrays.

The memory budget determines work-batch and spool-sort run sizes, not persisted
chunk size. A Zarr spatial chunk remains the stable unit of ownership,
invalidation, transfer, and reuse even when several chunks are processed
together for throughput or one chunk is assembled from many bounded runs.
Writer and reader budgets are independent explicit settings. The replay reader
may not load a complete reference tube merely because it lies inside one build
domain.

In separate layout, the `anchors` array reaches `complete` before fiberlet
publication begins. The `fiberlets` array may remain `building` and be consumed
for explicitly committed neighborhoods. Root `build_state` becomes `complete`
only after every key derived from the finite `build_domain` exists and validates
in both required arrays. An open on-demand cache remains `partial`. Metadata
completion uses the same temp-file, `fsync`, rename, and directory-`fsync`
protocol. The custom header is fixed-size and field descriptors precede payload
blocks, so a local range reader may fetch the search prefix without the trailing
route geometry.

## Per-field compression

Each array is transformed and compressed independently in blocks. The initial
transform candidates are:

| Field | Pre-compression transform |
| --- | --- |
| Exact cell coordinates | Integer delta after identity sorting, then byte shuffle when wider than one byte |
| Exact anchor positions/axes | Byte shuffle only; float32 bits are unchanged |
| Anchor positions | Integer delta, then byte shuffle |
| Anchor variants | None; values are only zero or one |
| Anchor axes | None |
| First X/Y/Z | Integer delta after fiberlet sorting, then byte shuffle |
| First variant | None; long repeated runs after sorting |
| Second DX/DY/DZ | Byte shuffle for `int16`; none for `int8` |
| Second variant | None |
| Interior counts | None |
| Entry/exit U/V | None |
| Path length | Byte shuffle |
| Float32 total cost | Byte shuffle |
| Middle U/V | Per-fiberlet integer delta, resetting at each fiberlet boundary |
| `uint16` costs | Byte shuffle |
| `uint8` costs | None |

Transforms are whole-array operations with defined resets at chunk or
first-anchor-run boundaries. They do not use per-value escape markers. Each
field descriptor records the transform version needed to decode its block.

The general block compressor remains to be selected by measurement. Zstd is a
reasonable initial candidate, but the schema does not depend on it.

The Zarr object codec is an envelope for these internal transforms and blocks;
it is not the block compressor itself. Changing an internal compressor changes
the sample-format/field-codec metadata and dataset fingerprint, not the Zarr
spatial key convention.

## Data deliberately not stored

- dense presence, prediction direction, or Lasagna normal samples;
- expanded XYZ path points in either encoding profile;
- individual edge-loss components in either encoding profile;
- explicit pairwise graph transitions;
- persisted anchor-to-fiberlet adjacency.

## Paris4 storage estimate

The estimate extrapolates density measured in a fiber-enriched reference tube
over every populated prediction chunk:

```text
populated prediction chunks: 166,032
estimated anchors:           465,290,241
estimated fiberlets:       4,734,024,437
mean interior points:              5.03
```

These are planning estimates, not counts from a completed whole-volume
extraction.

### Uncompressed field sizes

| Field group | Bytes per item | Estimated size |
| --- | ---: | ---: |
| Anchors with `uint8` positions | 6 per anchor | 2.6 GiB |
| Anchors with `uint16` positions | 9 per anchor | 3.9 GiB |
| Endpoints with `uint8` positions and `int8` deltas | 8 per fiberlet | 35.3 GiB |
| Endpoints with one-byte/two-byte mixed coordinates | 11 per fiberlet | 48.5 GiB |
| Endpoints with `uint16` positions and `int16` deltas | 14 per fiberlet | 61.7 GiB |
| Interior point count | 1 per fiberlet | 4.4 GiB |
| Entry/exit U/V | 4 per fiberlet | 17.6 GiB |
| Exact path length | 4 per fiberlet | 17.6 GiB |
| Middle U/V coordinates | about 6.1 per fiberlet | 26.7 GiB |
| `uint16` cost | 2 per fiberlet | 8.8 GiB |
| `uint8` cost | 1 per fiberlet | 4.4 GiB |

### Uncompressed totals

| Position scalar | Delta scalar | `uint16` costs | `uint8` costs |
| --- | --- | ---: | ---: |
| `uint8` | `int8` | about 113.0 GiB | about 108.6 GiB |
| `uint8` | `int16` | about 126.2 GiB | about 121.8 GiB |
| `uint16` | `int8` | about 127.5 GiB | about 123.1 GiB |
| `uint16` | `int16` | about 140.7 GiB | about 136.3 GiB |

### Float32 cache estimate

With one-byte chunk-local cell coordinates, one-byte endpoint-cell deltas, and
one-byte route coordinates, the float32 cache has these approximate raw costs:

| Field group | Bytes per item | Estimated size |
| --- | ---: | ---: |
| Float32 anchors: identity, position, fitted axis | 28 per anchor | 12.1 GiB |
| Stable endpoint identities | 8 per fiberlet | 35.3 GiB |
| Route count and entry/exit U/V | 5 per fiberlet | 22.0 GiB |
| Float32 path length and total cost | 8 per fiberlet | 35.3 GiB |
| Middle U/V coordinates | about 6.1 per fiberlet | 26.7 GiB |
| **Float32 cache total** | | **about 131.4 GiB** |

This is the planning estimate for a complete whole-volume cache before
per-field transforms and compression. It is not the peak working memory: the
adaptive writer retains only a bounded set of owner chunks and publishes each
chunk before moving on. Wider cell/delta/count fields increase the raw total
and are selected from the actual configuration rather than assumed.

For a `512`-base chunk and an approximately 128-base maximum per-axis endpoint
delta, the expected selections are:

| Position quantum | Position scalar | Delta scalar |
| ---: | --- | --- |
| 1 base voxel | `uint16` | `int16` |
| 2 base voxels | `uint8` | `int8` |
| 4 base voxels | `uint8` | `int8` |

The encoder still derives these choices from the exact configured bound rather
than hard-coding this table.

These raw totals are intentionally larger than the earlier bit-packed design.
They are simple upper bounds before sorting, delta transforms, byte shuffling,
and block compression.

The coordinate widths have a large raw effect, but repeated sorted first
coordinates and bounded second-coordinate deltas are expected to compress
strongly. No compressed-size range is assigned until the quantization layouts
have been selected and real field arrays can be measured.

For comparison, Paris4 fiber prediction level `/3` occupies:

| Dense source | Stored size |
| --- | ---: |
| Presence | 32.0 GiB |
| Direction `nx` | 37.1 GiB |
| Direction `ny` | 37.8 GiB |
| Both direction channels | **75.0 GiB** |

## Validation before implementation

Validation is deliberately split into three stages.

### Stage 0: float32 cache and Zarr envelope

Before using the cache for quantization conclusions:

- round-trip separate and inline anchor layouts and verify identical decoded
  anchor/edge identity, float32 bits, route points, costs, lengths, joins,
  graph replay, and failure metrics against uncached extraction;
- verify bounded peak memory while writing and replaying a corridor larger than
  RAM, including a synthetic single owner chunk larger than RAM, and verify
  resume reuses only complete matching chunks;
- force repeated LRU eviction during seed lookup and beam expansion and verify
  incident-edge completeness, stable tie ordering, resets, failures, and replay
  geometry remain identical to uncached execution;
- verify cross-chunk edges are owned once and graph queries load the required
  source-chunk halo;
- verify local filesystem publication uses temp-file, `fsync`, and rename and
  parent-directory `fsync`, and never exposes a partial final chunk;
- verify finite build domains reproduce exact expected anchor/fiberlet key sets,
  while missing, header-only, and nonempty files distinguish uncached, computed-
  empty, and computed-present without a duplicate state array;
- verify anchor-only inline chunks retain their anchor arrays and cross-chunk
  inline endpoint resolution loads the destination inline chunk;
- verify rational cell ownership at exact boundaries, non-integral scales,
  nonzero integral origins, negative floor-division cases, and scalar overflow;
- encode checked-in golden payloads on supported OS/architectures and require
  byte identity plus strict endian, padding, offset, checksum, and overflow
  rejection;
- verify a generic metadata reader can inspect/copy the group and chunk objects
  while a stock decoder fails explicitly on the unknown VC object codec;
- corrupt headers, descriptors, fingerprints, checksums, and field lengths and
  require hard failures rather than repair or fallback;
- compare separate-anchor and inline-anchor encoded bytes after logical decode,
  and use the separate layout for the adaptive wide-corridor cache.

### Stage 1: numeric quantization

Generate uncompressed arrays from representative existing extractions and sweep:

- anchor position quantum: `1`, `2`, and `4` base voxels;
- cost width: `uint8` and `uint16`;
- every `uint8`/`uint16` position and `int8`/`int16` delta layout selected by
  representative chunk-size and search-range configurations.

Measure:

- anchor position displacement and collisions, with a strict maximum of two
  variants at any quantized coordinate;
- reconstructed fiberlet endpoint and interior displacement;
- endpoint tangent and join-cost changes;
- absolute and relative decoded-cost error;
- pairwise ordering inversions and top-k agreement;
- beam-state ordering and final-route agreement;
- restart counts and tracing error across representative fibers;
- strict endpoint-key round-trip and scalar-bounds validation for every layout.

Choose the position quantum, cost width, and allowed reference configurations
from these results. Use `uint8` costs only if route-level behavior is acceptable;
otherwise use `uint16`.

### Stage 2: field compression

Only after numeric layouts are selected, apply candidate transforms and block
compressors without further numeric changes. Measure:

- compressed and uncompressed bytes for every field;
- compression and decompression throughput;
- random chunk-prefix and trailing-geometry load latency;
- adjacency construction time and peak memory;
- complete byte-identical decode results before and after compression.
