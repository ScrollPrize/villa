# Fiber piece approval

Approval of *parts* of a fiber, stored as rows of the immutable extension table
`ext_fiber_piece_approval` (extension `fiber-piece-approval`, format 1; see
[extensions.md](extensions.md)). The native API is
`objects/FiberPieceApprovalExtension.hpp`; the Python module binds the same
helpers (`fiber_piece_approval_table`, `approve_fiber_piece`,
`crop_fiber_piece_approvals`, `fiber_piece_approval_states`, ...). VC3D's Line
Annotation is the only writer today; the Fiber Map and Spiral inputs do not
consume approvals yet.

## Model

- A **piece** is the range of one fiber between two **lock control points**, or
  from one lock control point to the fiber end (an *end piece*). Pieces are
  referenced by stable control-point ids (`control_point_ids`), never by index.
- A **lock control point** is an ordinary control point carrying the tag
  `lock` in its `tags` annotation and an explicit `direction`. Editors keep it
  fixed in position and direction, so tracing against it is reproducible.
- An **approval row** records one approval of one range for one **aspect**:
  `geometry` (0) or `normals` (1). Rows are immutable; revoking or cropping an
  approval appends a *superseding* row (`supersedes` = the replaced
  `row_uuid`). A crop keeps the original `approval_uuid`, author and time and
  records who cropped it in `changed_by` / `changed_at`.
- Invariant: **live rows of one aspect never overlap on a fiber.** When a piece
  is opened next to an approved one and edited, the editor crops the neighbour's
  row to the part outside the new piece (a row entirely inside is revoked).
- Validity is **per span** (owner control index i, control i to i+1): a span is
  *approved* for an aspect when the one live row covering it is *current* there,
  i.e. the span's stored fingerprint and both of its controls' fingerprints
  still match the fiber. A row some of whose spans no longer match is *stale*.
- Extrapolation (the open tails past the end controls) is never covered. An end
  piece covers up to the last control at approval time; growth past it is
  uncovered until the piece is approved again.

## Table

| column | type | meaning |
| --- | --- | --- |
| `fiber_uuid` | BLOB, key | owner fiber |
| `row_uuid` | BLOB, key | this row |
| `approval_uuid` | BLOB | the approval event; shared by crop rows and by copies made on split/merge |
| `aspect` | INTEGER | 0 geometry, 1 normals |
| `state` | INTEGER | 1 approved, 2 revoked (fingerprint blobs empty) |
| `supersedes` | BLOB, nullable | `row_uuid` this row replaces |
| `lock_cp_a` | INTEGER | stable id of the lock control the fingerprints walk from |
| `lock_cp_b` | INTEGER, nullable | second lock control; NULL for an end piece |
| `inside_cp` | INTEGER | stable id of the control next to `lock_cp_a` inside the piece (orientation) |
| `fiber_version` | INTEGER | object version the fingerprints were taken at |
| `author`, `approved_at`, `host` | TEXT | the approval stamps |
| `changed_by`, `changed_at` | TEXT, nullable | who wrote a crop or revocation row |
| `span_fingerprints` | BLOB | per covered span: int64 id_low, int64 id_high, uint64 hash (24 bytes) |
| `cp_fingerprints` | BLOB | per covered control, walking order: int64 id, uint64 hash (16 bytes) |

Index `{fiber_uuid, aspect}`. Provisioning:
`{"fingerprint":"fnv1a64-v1","aspects":{"geometry":0,"normals":1},"states":{"approved":1,"revoked":2}}`.

### Approved normals: `ext_fiber_piece_normals`

A normals approval approves the normals that were shown on the stretch, so it
stores them: one row per approval row that carries normals, in a second table
of the same extension.

| column | type | meaning |
| --- | --- | --- |
| `fiber_uuid` | BLOB, key | owner fiber |
| `row_uuid` | BLOB, key | the approval row these normals belong to |
| `normals` | BLOB | uint64 span count; per span uint64 point count, then 3×float64 per line point |

One list per covered span, in the row's walking order. Inside a span the
normals run from the control with the lower stable id to the one with the
higher id, both controls included, one per line point. A normal is a unit
**axis**: the annotator approves the cross-section they see and cannot know
which of its two sides faces outward, so its facing means nothing and every
consumer resolves it itself. `readFiberPieceApprovals` returns each row with
its normals attached (`FiberPieceApproval::spanNormals`);
`fiberPieceApprovalRows` gives both tables' rows for a list of approvals.

A normals row without stored normals (written before they were stored, or
whose normals row is missing) approves nothing: its spans read as stale until
the stretch is approved again.

## Fingerprints

FNV-1a 64 over canonical little-endian bytes (`fnv1a64-v1`); the store
(`fiberPieceContent(reader, revision)`) and VC3D's save snapshot
(`pieceContentOfStoredFiber`) produce identical bytes.

- **Span hash**, normals: the span's line points and then its approved
  normals, both walked from the control with the lower stable id to the one
  with the higher id (count first, 3×float64 each). A moved line point makes
  the span's normals approval stale, as it does the geometry approval.
- **Span hash**, geometry: the span's line points, walked from the
  control with the lower stable id to the one with the higher id (3×float64
  each, count first), then the canonical JSON of the owner's `segment_to_next`.
  Keyed by the unordered id pair, so reversing the fiber changes nothing.
  A control that is not on the line (`controlLineIndex` −1) hashes the two
  control positions instead.
- **Control hash**, geometry: position (3×float64), `direction` canonicalized up
  to sign (first nonzero component made positive; an absence flag otherwise),
  `tags` sorted with `lock` excluded. Normals: `display_normal` (or absence)
  and `display_normal_source`.
- Inserting a lock control exactly on a line point with the line's tangent as
  direction and the enclosing span's descriptor copied to both halves changes
  no existing fingerprint.

Fiber width and gap fraction are fiber-level and part of no fingerprint.

## Helpers

- `fiberPieceApprovalTable()`, `fiberPieceApprovalRow` / `decodeFiberPieceApprovalRow`.
- `FiberPieceContent`: ids, positions, annotations, line points and the control
  → line index map; from a revision through any `StoreReader`, or from a
  `geometry::Fiber` plus its control annotations.
- `fiberPieceRange(content, lockA, lockB|none, insideCp)`: the inclusive control
  index range and walking direction; `openEnd` for an end piece.
- `fingerprintFiberPiece`, `approveFiberPiece`, `revokeFiberPieceApproval`.
- `cropFiberPieceApprovals(content, live, openedFirst, openedLast, who, now)`:
  the superseding rows for every live row sharing a span with the opened
  control range. A row cut on both sides yields two parts.
- `liveFiberPieceApprovals`, `fiberPieceApprovalStates`: per aspect and span the
  covering live row and its currency, the approved spans, per-row currency,
  the lock ids live rows reference and the *releasable* ones (referenced only by
  superseded rows: the editor drops their `lock` tag at its next save).
- `readFiberPieceApprovals(store, txn, fiber)` / `(reader, fiber)`.

## Transactions

`CreateExtensionTableOp{ExtensionTable}` and `AppendExtensionRowsOp{table, rows}`
(`Transactions.hpp`) register an extension table and append immutable rows
inside a request's checked transaction, under the request's version
expectations. VC3D writes crop rows in the same submit as the fiber save that
takes the overlap, and the approval row in a submit expecting the saved
version. Both operations have a JSON form (`ExtensionValue` as null, integer,
real, string or `{"blob": hex}`); the server applies them like any other
operation. Hosted *readers* have no extension access and throw.

Rows reference control ids, so VC3D copies the live rows of a fiber under the
child UUIDs when it splits or merges fibers; a row cut by a split is dropped.
