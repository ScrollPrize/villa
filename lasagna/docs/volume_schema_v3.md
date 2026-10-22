# Lasagna volume schema v3: embedded manifest and resolution groups

Status: format proposal; writer and reader support are not implemented by this
document. Existing volumes and running jobs are unchanged.

This replaces the external `.lasagna.json` manifest and separate per-channel
Zarr stores with one Zarr store containing resolution-based channel groups.
It changes storage organization, not channel encodings, interpolation,
coordinate conventions, inference resolution, or pyramid reduction methods.

## Scope and compatibility

- New-schema volumes MUST embed the manifest in the root group's `lasagna`
  attribute, with `version: 3`. A sidecar manifest is neither required nor the
  authority for a v3 volume.
- New-schema volumes MUST group channels by their finest stored resolution,
  not by channel names or semantic categories.
- Each resolution group MUST contain multichannel `CZYX` arrays, including
  when it contains only one channel. Chunks MUST span the whole channel axis.
- The embedded manifest MUST name the channel whose stored zero value marks
  invalid data; validity is not inferred from a hardcoded channel name.
- Future readers must continue accepting existing v1/v2 `.lasagna.json`
  manifests and their existing layouts. Supporting v3 does not require
  rewriting old volumes. Old readers are not expected to read v3.
- This is **Lasagna schema version 3**, not Zarr format 3. The initial storage
  profile remains Zarr v2 with OME-NGFF 0.4 metadata, matching existing writers.
  Switching the underlying Zarr/NGFF version is separate work.

The immediate scope is Lasagna volumes. The same layout can accommodate
Fiber prediction channels without defining new Fiber channel encodings here.

## Layout and naming

`sd<N>` identifies a group whose finest stored level has `scaledown: N`.
As in the existing manifest, `N` is an exponent: its sampling interval is
`2**N` source voxels, or `source_to_base * 2**N` original base-volume voxels.
Thus `sd3` means an 8x reduction when `source_to_base` is 1, not a 3x reduction.

All channels with the same finest stored `scaledown` MUST be packed into the
same group. Their channel order is declared explicitly in the manifest.
Group names MUST be canonical `sd0`, `sd1`, etc.; names such as `nx`, `cos`,
or `normals` are not used for resolution groups.

Each group holds its own pyramid, with a constant channel set and order across
all of its levels. Numeric array names retain the existing absolute level
number: the finest array in `sd3` is `sd3/3`, not `sd3/0`.

Example preserving the current Paris4 resolutions:

```text
prediction.zarr/
  .zgroup
  .zattrs                  # { "lasagna": { "version": 3, ... } }
  sd3/
    .zgroup
    .zattrs                # OME multiscales, axes CZYX
    3/                     # [cos], 8x from base
    4/                     # [cos], 16x from base
    5/                     # [cos], 32x from base
  sd4/
    .zgroup
    .zattrs                # OME multiscales, axes CZYX
    4/                     # [grad_mag, nx, ny], 16x from base
    5/                     # [grad_mag, nx, ny], 32x from base
```

Groups are classified by their **finest** resolution. The derived `cos` array
at level 4 remains in `sd3`; it is not merged into `sd4`. This keeps each OME
image's channel identity stable across its pyramid. Finer levels that were
never generated MUST NOT be advertised as empty arrays.

## Embedded manifest

The root `.zattrs` contains this object (example values):

```json
{
  "lasagna": {
    "version": 3,
    "source_to_base": 1.0,
    "base_shape_zyx": [75784, 32693, 32693],
    "validity_channel": "grad_mag",
    "grad_mag_encode_scale": 1000.0,
    "grad_mag_factor": 0.25,
    "groups": {
      "sd3": {
        "zarr": "sd3/3",
        "scaledown": 3,
        "channels": ["cos"]
      },
      "sd4": {
        "zarr": "sd4/4",
        "scaledown": 4,
        "channels": ["grad_mag", "nx", "ny"]
      }
    },
    "provenance": "inference.json"
  }
}
```

### Complete manifest field reference

The following are all of the defined manifest fields. Defaults below describe
absence, not permission to discard an existing value during conversion.

| Field | JSON type / requiredness | Meaning and units |
|---|---|---|
| `version` | Integer, required | Lasagna schema version; exactly `3` in this format. Independent of Zarr's and OME-NGFF's version numbers. |
| `source_to_base` | Positive finite number, required | Base-volume voxels per source-coordinate voxel. Multiply a source XYZ coordinate by this scalar to obtain its base XYZ coordinate. Existing readers default an absent legacy value to `1.0`; v3 writes it explicitly. |
| `base_shape_zyx` | Three positive integers, required | Original reference-volume extent `[Z, Y, X]` in base voxels. This is not the model input, prediction, or crop shape. It remains unchanged when outputs are downsampled. Some legacy manifests omit it; that is not allowed for a v3 writer. |
| `groups` | Nonempty object, required | Maps each native-resolution name `sd<N>` to the channel-group object defined below. Object ordering has no semantic significance. |
| `validity_channel` | Nonempty string, required | Name of a channel declared in exactly one group. At that channel's stored voxels, a value exactly equal to zero means invalid/no data; a nonzero value is the valid-data signal. This names an existing signal channel, not a separate mask. No default or hardcoded `grad_mag` fallback applies to v3. |
| `grad_mag_encode_scale` | Positive finite number; required when `grad_mag` is present | Encoding denominator for winding density; decoded value is `stored * grad_mag_factor / grad_mag_encode_scale`. Legacy default is `1000.0`. This is not a spatial scale factor. |
| `grad_mag_factor` | Positive finite number; required when `grad_mag` is present | Multiplier applied when decoding gradient magnitude/winding density. Legacy default is `1.0`; an existing value such as `0.25` MUST be preserved, not replaced by the default. |
| `crops` | Optional array of six-integer arrays; default `[]` | Recorded processing regions, each `[x, y, z, width, height, depth]` in base voxels. Origins are nonnegative and sizes positive. Regions are half-open, for example X covers `[x, x + width)`. Multiple regions may overlap. They record crop coverage, not a per-voxel validity mask or a guarantee that every chunk in a region exists. |
| `umbilicus_json` | Optional string; absent or empty means unset | Path to the scroll's umbilicus/axis control-point JSON. It contains a nonempty `control_points` list of objects with finite `x`, `y`, `z` coordinates in base voxels. General volume reading does not require this artifact; a fitting operation that needs the umbilicus may require it. |
| `init_shell_dir` | Optional string; absent or empty means unset | Path to the shell-direction crop initialization directory used by surface fitting. It is an auxiliary fitting artifact, not a volume channel or part of the channel-coordinate transform. Readers that do not initialize fits need not access it. |
| `provenance` | Optional string; absent or empty means unset | Path to a JSON artifact recording how inference was produced, commonly `inference.json`. It is a reference, not an embedded provenance object, model identifier, or snapshot filename. Its contents do not change channel decoding. |

Auxiliary paths (`umbilicus_json`, `init_shell_dir`, `provenance`) resolve
relative to the Zarr root when relative; existing absolute paths or remote
references retain their meaning where supported by the consuming tool.
Writers should use root-relative paths for portable bundles. Moving the
manifest therefore requires rebasing relative paths if an artifact is not
moved along with it. This does not embed auxiliary artifact contents into the
manifest or make them mandatory for basic volume reading.

Unrecognized manifest fields are extension metadata, just as with the existing
manifest's extra fields: preserve their JSON values during read/write and
migration, but do not infer decoding or coordinate semantics from them.
Extensions MUST NOT override the defined fields. Their producer must document
their meaning separately; arbitrary application extensions are not secretly
part of the core schema defined here.

### Complete channel-group field reference

| Field | JSON type / requiredness | Meaning |
|---|---|---|
| `zarr` | String, required | Store-root-relative path to the finest array, exactly `sd<N>/<N>`. Unlike the legacy layout, v3 channel arrays do not refer to external stores or absolute paths. |
| `scaledown` | Nonnegative integer, required | Native-resolution exponent `N`, not the reduction factor itself. Array voxel spacing is `source_to_base * 2**N` base voxels. It must match the group name and finest array level. |
| `channels` | Nonempty array of distinct strings, required | Channel names in C-axis order. The list length equals the array's first dimension; index zero names `array[0, ...]`. The order must remain identical at every pyramid level in this group. |

A channel occurs in exactly one manifest group; its coarser pyramid copies do
not constitute additional channel bindings. All group fields are explicit;
there is no inferred channel order or default scale.

### Legacy aliases and exact changes from v2

Legacy readers accept `crop_xyzwhd`, a single six-integer crop with the same
order and units as an entry in `crops`, when a `crops` list is absent. A v3
writer emits the equivalent `crops` list, not that obsolete alias. If both
exist in legacy input, the `crops` list takes precedence, matching current
loading behavior. Python object member names such as `provenance_json`,
`zarr_path`, and `extra_fields` are implementation details, not extra JSON
fields.

The deliberate format changes are the version number, manifest location,
resolution-based group names, internal array paths, and packed CZYX layout.
V3 also makes the source scale and original base shape explicit rather than
relying on legacy defaults or missing metadata, and explicitly declares the
validity channel. The remaining field names,
values, and scientific meanings are retained. Conversion MUST preserve
encoding factors, crop records, auxiliary references, provenance, and
extension metadata; regrouping channels must not drop them.

### Coordinate and validity conventions

Array indexing is CZYX; geometric coordinates and crop origins are XYZ.
There is no implicit half-voxel offset or crop-origin translation introduced
by v3: array index `(z, y, x)` maps to base position
`(x * spacing, y * spacing, z * spacing)`, with the spacing given above.
Cropping remains sparse coverage in the full reference frame, not an
implicitly rebased crop array.

Missing chunks use zero fill. Validity is encoded in the signal channel named
by `validity_channel`, with the fixed rule **stored value == 0 means invalid**.
The comparison is against stored values, before decoding, scaling, or spatial
interpolation; it is exact, not an epsilon or configurable probability cutoff.
There is no separate validity array, boolean mask channel, or required new
per-voxel storage.

- Lasagna normal volumes declare `"validity_channel": "grad_mag"`.
- Fiber prediction volumes declare `"validity_channel": "presence"`.
  A stored presence of zero is invalid; a low but nonzero presence is not
  invalidated by this format rule. A tracer may independently apply its own
  confidence threshold, but that is not storage validity.
- Other prediction families may name another declared channel without changing
  the rule or schema. A missing declaration or an unknown channel name is a
  malformed v3 manifest, not permission to guess a channel.

The declaration applies to the whole volume, including its resolution groups
and their pyramid levels. Resolve the channel through `groups` and
`channels`, not an assumed channel index. Its pyramid copies use the same zero
rule. Where a consumer combines different resolution groups, it must use
their declared base-coordinate mappings to align the validity signal; equal
array indices do not imply equal physical positions. This declaration does
not prescribe a new mask-resampling method or change existing valid-corner
selection, normal interpolation, or pyramid reduction algorithms. In
particular, do not interpolate a signal first and then reinterpret its result
as the validity of every contributing corner.

An absent validity-channel chunk and a present chunk containing zeros have
the same invalid-data meaning. Coordinates outside the reference extent are
invalid. The encoded `nx`/`ny` byte values alone are not a validity mask.
Channel-specific no-data conventions still apply (for example, `pred_dt == 0`
has no distance value); they do not replace the declared volume-wide signal.
A missing chunk is not a missing resolution level; only genuinely available
arrays belong in the pyramid's dataset list.

Legacy v1/v2 Lasagna volumes retain their existing `grad_mag` validity behavior.
Conversion to v3 records that explicitly as `validity_channel: "grad_mag"`;
Fiber producers use `presence`. No legacy reader behavior changes merely
because the new format introduces this field.

### Existing Lasagna channel encodings

The current Lasagna inference channels are unsigned 8-bit values. Packing
them together does not change these encodings:

- `cos`: the model's scalar cosine/phase-related signal in `[0, 1]`, decoded
  as `stored / 255`. It is not a component of the sheet normal.
- `grad_mag`: sheet winding density, decoded by the formula in the field
  table. It is the magnitude of the winding-position gradient, not a
  probability. V3 introduces no rescaling by the stored pyramid level.
- `nx`, `ny`: the X and Y components of a sign-ambiguous sheet normal,
  decoded as `(stored - 128) / 127`. Reconstruct
  `nz = sqrt(max(0, 1 - nx*nx - ny*ny))`, then normalize the vector.
  The nonnegative-Z hemisphere represents the unoriented axis; `n` and `-n`
  denote the same normal. Decode and sign-align neighboring vectors before
  interpolation, then normalize; do not interpolate encoded bytes across a
  hemisphere/sign discontinuity. In the Lasagna normal profile, the declared
  `grad_mag` validity channel invalidates normals where winding density is zero.
- `pred_dt` (optional): the existing auxiliary signed-distance **encoding**,
  with outside values `[80, 127]`, inside values `[128, 175]`, and `0` for
  no data. These are encoded values, not a new distance measurement in base
  voxels. Preserve them without inventing a scale factor or interpreting
  this channel as presence.

Other prediction families may declare additional channel names; their
encodings must be documented by that family. This document does not invent
new Fiber prediction channel meanings.

## Arrays, chunks, and OME metadata

For level `L`, every resolution-group array has axes `[c, z, y, x]` and a
spatial sampling interval `source_to_base * 2**L` in base voxels. Its shape
uses the existing full-extent rounding convention: for each base dimension
`B`, `max(1, ceil(B / (source_to_base * 2**L)))`.

For an array with `C` channels, chunks have shape `[C, cz, cy, cx]`, with
positive spatial chunk dimensions chosen by the writer. This means one chunk
read/decode retrieves the corresponding spatial block of all grouped
channels. Spatial chunk size and compressor choice are not fixed by v3.
All channels in an array necessarily share its dtype; no conversion of channel
encodings is implied. A writer must not silently quantize incompatible
channel representations to pack them together.

Each `sd<N>` group is an OME-NGFF multiscale image. Its metadata lists only
the arrays actually provided, finest to coarsest. The root is a container;
it MUST NOT advertise the differing channel sets as one image pyramid.

Example `sd4/.zattrs`:

```json
{
  "multiscales": [
    {
      "version": "0.4",
      "name": "sd4",
      "axes": [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"}
      ],
      "datasets": [
        {
          "path": "4",
          "coordinateTransformations": [
            {"type": "scale", "scale": [1, 16, 16, 16]}
          ]
        },
        {
          "path": "5",
          "coordinateTransformations": [
            {"type": "scale", "scale": [1, 32, 32, 32]}
          ]
        }
      ]
    }
  ]
}
```

This example uses `source_to_base: 1` and base-voxel coordinates, not physical
micrometres; no physical unit is claimed. The embedded manifest supplies
channel semantics. Optional OME channel display metadata must agree with its
channel order. The channel transform remains 1 at every level.

The OME fields used above have the following meanings: `multiscales` is the
list of image-pyramid descriptions; `version` there is the OME-NGFF version
(`0.4`, not Lasagna's `3`); `name` identifies this resolution group; `axes`
defines dimension order using each axis's `name` and `type`; `datasets` lists
the available level arrays. Each dataset's `path` is relative to its `sd<N>`
group, unlike a manifest group's root-relative `zarr`. Its
`coordinateTransformations` contains one `type: "scale"` transform whose
`scale` is `[1, spacing, spacing, spacing]`. No translation is used for this
full-reference layout. These values MUST agree with the embedded manifest;
conflicts are errors, not a reason to silently pick one coordinate system.

Zarr's `.zgroup` declares `zarr_format: 2`. Each array's `.zarray` declares its
`shape`, `chunks`, `dtype`, `compressor`, `filters`, `order`, `fill_value`,
and chunk-key `dimension_separator` using the ordinary Zarr-v2 definitions.
The logical shape/order of dimensions is always CZYX regardless of the byte
storage order. No Lasagna-specific codec or implicit channel interleaving is
added: the declared shape, chunk shape, storage order, and codec determine the
bytes. The fill value is zero. Codec and chunk-key choices remain explicit
Zarr metadata rather than manifest fields.

Scalar and normal-aware pyramid reduction remain as currently implemented:
in particular, encoded `nx`/`ny` MUST NOT be independently averaged as scalar
bytes. Grouping them in a chunk does not change their joint decoding,
sign-ambiguous interpolation, or normal-aware pyramid construction.

See the [OME-NGFF 0.4 specification](https://ngff.openmicroscopy.org/0.4/)
for the underlying image, axis, and multiscale metadata conventions.

## Later implementation plan (not part of this documentation change)

1. Add shared v3 manifest validation/serialization while retaining legacy
   manifest loading; do not change the v1/v2 interpretation.
2. Update writers to assemble products by native resolution, write full-channel
   chunks, and embed the manifest at the root. Preserve values, scales, sparse
   coverage, encodings, and existing pyramid math.
3. Extend consumers, including the accelerated C++ sampler, to resolve a root
   store into the manifest, array path, and channel index. Reuse the existing
   decoding/interpolation implementations rather than duplicating them.
4. Test round trips, legacy compatibility, declared `grad_mag` and `presence`
   validity (including absent chunks, explicit zeros, and nonzero values),
   rejection of missing/unknown validity-channel declarations, packed-channel decoding, normal
   interpolation, metadata/shape agreement, and base-volume coordinate mapping.
   Benchmark I/O and decoding on equivalent spatial requests before claiming
   an acceleration.

No writer default, loader, manager, Atlas publication behavior, or training
setting is changed until that implementation is explicitly undertaken.
