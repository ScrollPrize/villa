# Fiber approval layer

A fiber's approval is a layer on top of its stored geometry and normals. It
never changes the fiber. The layer holds *stretches* with a state per aspect,
*normal points* and the normals that were shown when it was saved. It is the
extension `fiber-approval` (format 1, `keySpace="objects"`; see
[extensions.md](extensions.md)); the C++ API is
`objects/FiberApprovalExtension.hpp`, and the Python module binds the same
functions.

## Positions

Every position is an index into the fiber's stored `line_points`, with a
fraction: `12.5` lies halfway between line points 12 and 13. Stretch ends and
normal points are independent of control points.

## Generations

Every save appends one whole generation of the layer: one header row and the
stretch and normal point rows of that generation. Rows are only appended. The
generation with the highest number is the fiber's layer; lower ones are
history. A generation without stretches and normal points is an empty layer,
which is how a layer is cleared. `readFiberApproval(reader, fiberUuid)`
returns the newest generation, `fiberApprovalGenerations(reader)` the newest
generation number of every fiber that has one. A store without the tables has
no layers.

## Tables

`ext_fiber_approval`, primary key (`fiber_uuid`, `generation`):

| Column | Type | Value |
| --- | --- | --- |
| `fiber_uuid` | BLOB, 16 bytes | The fiber |
| `generation` | INTEGER | Generation number, counted per fiber |
| `fiber_version` | INTEGER | Object version of the fiber the layer was saved on |
| `baseline_run` | TEXT | Scaffold solve run whose normals the layer was shown on; may be empty |
| `author`, `saved_at`, `host` | TEXT | Who saved it, when (ISO-8601 UTC) and where |
| `line_normals` | BLOB | The shown normal at every stored line point: 3 float32 little-endian per point; empty when nothing was shown |

`ext_fiber_approval_stretch`, primary key (`fiber_uuid`, `generation`, `item`):

| Column | Type | Value |
| --- | --- | --- |
| `fiber_uuid`, `generation` | | The generation the row belongs to |
| `item` | INTEGER | Index of the stretch in its generation |
| `aspect` | INTEGER | geometry=0, normals=1 |
| `state` | INTEGER | approved=1, broken=2 |
| `from_position`, `to_position` | REAL | Line positions, `from_position < to_position` |
| `fingerprint` | INTEGER | Fingerprint of the line points the stretch touches |

`ext_fiber_approval_normal_point`, primary key (`fiber_uuid`, `generation`, `item`):

| Column | Type | Value |
| --- | --- | --- |
| `fiber_uuid`, `generation` | | The generation the row belongs to |
| `item` | INTEGER | Index of the normal point in its generation |
| `position` | REAL | Line position |
| `normal_x`, `normal_y`, `normal_z` | REAL | The wanted normal, unit length; an axis, its facing means nothing |
| `fingerprint` | INTEGER | Fingerprint of the line points around the position |

## Stretches

A stretch gives a range of one aspect the state *approved* or *broken*.
*Undefined* is the absence of a stretch and is never stored. The stretches of
one aspect are sorted and never overlap; the two aspects are independent.
`setFiberStretchState` paints a state over a range: existing stretches of that
aspect are cut back to the part outside it, a stretch touching the new one in
the same state is joined with it, and undefined only clears. The range is
clamped to the line; an empty or reversed range is refused.

## Normal points

A normal point states the wanted normal at one position. `setFiberNormalPoint`
normalizes it and replaces an existing normal point within `tolerance` (0.5
line points by default) of the position, keeping that point's position;
otherwise it inserts a new one. Normal points are sorted by position.

## Fingerprints and staleness

A fingerprint is FNV-1a 64 over the coordinates (IEEE doubles, little-endian,
`-0` as `+0`) of the line points `floor(from)` to `ceil(to)`, clamped to the
line (`fiberLineFingerprint`). A stretch is fingerprinted over its range, a
normal point over the one or two line points around its position. The column
holds the 64 bits as a signed integer.

`fiberApprovalStatus(layer, line)` compares the layer with the fiber's current
line points. A stretch or normal point is *current* when it lies within the
line and its fingerprint still matches; otherwise it is *stale* and states
nothing. Editing a fiber therefore invalidates only the items on the changed
line points. The stored line normals are usable only while there is exactly
one per current line point.

## Approved line normals

`approvedFiberLineNormals(layer, status, linePoints)` is what a consumer of
approved normals reads: one normal per line point, zero where nothing is
approved, empty when there is none at all.

- On every current stretch approved for normals: the stored line normal of
  each line point from `ceil(from)` to `floor(to)`.
- At the line point nearest every current normal point: the normal point's
  own normal, unless the position lies in a stretch whose normals are broken.

`fiberStretchRanges` lists the current stretches of one aspect and state as
position ranges.

## Python

`fiber_approval_tables()` returns the three table definitions,
`fiber_approval_rows(layer)` the typed rows of one generation (`header`,
`stretches`, `normal_points`) and `store.read_fiber_approval(fiber_uuid)` the
newest generation as a layer dict or `None`. A layer dict has the keys
`fiber_uuid`, `generation`, `fiber_version`, `author`, `saved_at`, `host`,
`baseline_run`, `stretches` (`aspect`, `state`, `from`, `to`, `fingerprint`),
`normal_points` (`position`, `normal`, `fingerprint`) and `line_normals`.
`set_fiber_stretch_state`, `fiber_stretch_state_at`, `set_fiber_normal_point`,
`fiber_approval_status`, `approved_fiber_line_normals` and
`fiber_line_fingerprint` are the functions above; the two setters return the
changed list instead of changing their argument.
