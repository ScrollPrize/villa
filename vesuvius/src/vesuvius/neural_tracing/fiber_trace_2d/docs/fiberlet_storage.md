# Precomputed Fiberlet Storage Proposal

This is a proposed compressed structure-of-arrays format. It is not implemented
and is not a compatibility contract. Cost and field compression must be
validated on representative extracted data before the format is fixed.

## Design rules

- Anchors and fiberlets are separate records. Fiberlets reference anchors.
- Fiberlets form one logical list. Every field is stored as a parallel list.
- The logical lists are split into independently compressed spatial chunk blocks
  so tracing can load a local neighborhood.
- Fields use ordinary `uint8`, `int8`, `uint16`, `uint32`, and `float32` values.
- Sorting, delta transforms, byte shuffling, and block compression remove
  redundancy.
- Fiberlet endpoints use quantized coordinates plus a variant index. The second
  endpoint is stored as a signed coordinate delta from the first.
- Scalar widths are fixed per dataset and have no per-value escape
  representation.
- Anchor positions may use a one-, two-, or four-base-voxel storage quantum.

## Dataset structure

```text
FiberletDataset
|- DatasetHeader
|- SpatialChunkIndex[]
`- SpatialChunk[]
   |- ChunkHeader
   |- AnchorArrays
   |- FiberletSearchArrays
   `- FiberletPointArrays
```

Each field array is an independent compressed block. A codec can therefore use
the transform appropriate for that field without requiring padded C++ records.

`FiberletPointArrays` are physically last in every chunk. The chunk header and
field descriptors precede the search arrays, so a range reader can load anchors,
connectivity, costs, lengths, and endpoint geometry without reading or
decompressing the complete interior paths.

## DatasetHeader

| Field | Type | Meaning |
| --- | --- | --- |
| `format_version` | `uint32` | Exact storage-format version |
| `base_shape_zyx` | `uint32[3]` | Base-volume bounds |
| `chunk_side_base` | `uint16` | Spatial chunk side, initially `512` |
| `prediction_to_base` | `float32` | Base voxels per prediction voxel |
| `anchor_position_quantum_base` | `uint8` | Base voxels per stored anchor-position unit: `1`, `2`, or `4` |
| `anchor_position_type` | enum | `uint8` or `uint16`, derived from chunk side and quantum |
| `anchor_delta_type` | enum | `int8` or `int16`, derived from search range and quantum |
| `fiber_manifest` | string | Dense presence and direction source |
| `normal_manifest` | string | Lasagna normal source |
| `anchor_cell_side_prediction` | `uint8` | Anchor cell side |
| `max_anchors_per_cell` | `uint8` | Fixed extraction invariant, initially `2` |
| `longitudinal_step_prediction` | `float32` | DP layer spacing |
| `transverse_step_prediction` | `float32` | DP transverse spacing |
| `corridor_radius_prediction` | `float32` | DP transverse bound |
| `loss_config` | fixed structure | Alignment and smoothness weights |
| `cost_encoding` | enum | `uint8` or `uint16` |
| `field_codec` | enum | Block compressor and transform version |

Dense presence, prediction directions, and Lasagna normals remain external
source volumes. They are sampled on demand and are not copied into every anchor
or fiberlet.

## SpatialChunkIndex

Only populated chunks have entries.

| Field | Type | Meaning |
| --- | --- | --- |
| `chunk_xyz` | `uint16[3]` | Chunk coordinate |
| `file_offset` | `uint64` | Start of the encoded chunk block |
| `encoded_size` | `uint64` | Compressed chunk size |
| `checksum` | implementation choice | Optional integrity check |

The index allows range reads for the active chunk and its neighbors.

## ChunkHeader

| Field | Type | Meaning |
| --- | --- | --- |
| `anchor_count` | `uint32` | Anchors in this chunk |
| `fiberlet_count` | `uint32` | Fiberlets stored in this chunk |
| `middle_point_count` | `uint32` | Total interior points stored in the trailing point arrays |
| `cost_offset` | `float32` | Affine cost decoder offset |
| `cost_scale` | `float32` | Affine cost decoder scale |
| `field_descriptors[]` | fixed entries | Type, transform, offset, compressed size, and element count for each field array |

Every field descriptor is explicit. Unknown or inconsistent field lengths are
format errors rather than cases handled through escape values.

## Anchor identity

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
position_bins = chunk_side_base / anchor_position_quantum_base

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

`cost[i]` stores the quantized total loss of fiberlet `i`:

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

Only the total is stored. Changing loss weights requires regenerating or
rescoring fiberlets. Join costs are computed and cached on demand from the
reconstructed edge tangents, dense fiber prediction, and Lasagna normals.
Explicit pairwise graph transitions are not persisted.

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

## Per-field compression

Each array is transformed and compressed independently in blocks. The initial
transform candidates are:

| Field | Pre-compression transform |
| --- | --- |
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
| Middle U/V | Per-fiberlet integer delta, resetting at each fiberlet boundary |
| `uint16` costs | Byte shuffle |
| `uint8` costs | None |

Transforms are whole-array operations with defined resets at chunk or
first-anchor-run boundaries. They do not use per-value escape markers. Each
field descriptor records the transform version needed to decode its block.

The general block compressor remains to be selected by measurement. Zstd is a
reasonable initial candidate, but the schema does not depend on it.

## Data deliberately not stored

- dense presence, prediction direction, or Lasagna normal samples;
- expanded XYZ path points;
- individual edge-loss components;
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

Validation is deliberately split into two stages.

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
