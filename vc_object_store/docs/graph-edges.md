# Graph edges (relationships)

Relationships are first-class revisioned records between two objects. They
carry a `kind`, an approval state, and an **anchor** at each end declaring
exactly what the edge depends on — point IDs or indices, surface channels,
metadata keys, or the whole geometry. Because anchors name their dependencies,
the store can say whether an edge is **applicable** to any pair of endpoint
versions, and can refuse a geometry edit that changes a dependency without an
explicit decision for that edge. Edges are added by
[source files](#edges-imported-from-sources), by the
[detectors](#discovering-edges-detectors), or directly with `add_edge`.

## The record

```python
edge = store.add_edge({
    "a": a, "b": b, "kind": "fiber_branch",
    "a_anchor": {"points": [{"array": "control_point_ids", "index": [10]}],
                 "position": [4, 5, 6], "direction": [1, 0, 0]},
    "b_anchor": {"points": [{"array": "control_point_ids", "index": [90]}],
                 "position": [1, 2, 3], "direction": [1, 0, 0]},
})
store.set_approval(edge, "approved")     # new edges start life "pending" unless stated
```

- Endpoint order in storage is canonical (`a` < `b`); put directional roles in
  metadata. A self-link is allowed and revisions its object once.
- Adding, removing, or rebinding an edge revises **both** endpoints in the
  same transaction — checked writes must expect both versions.
- Anchors declare dependencies through `points` (IDs/indices), `channels`,
  `metadata_keys` (top-level keys), or `whole_geometry`.
- A relationship version pins both endpoints' object versions
  (`a_version`/`b_version`) and their geometry. Resolve those pins; never
  substitute current heads when reading history.
- Lookups: `store.edge(uuid)`, `store.edges_of(uuid)`,
  `store.edges_by_kind(kind)`, `store.select_edges(...)`. Arbitrary UUID
  strings inside JSON metadata are **not** tracked references — application
  references should be ordinary `declared_reference` edges with explicit
  dependencies.

## Edge kinds

Kinds are names, not an enum; new kinds need no schema change. The kinds the
store and its tools write:

| Kind | Meaning | Written by |
| --- | --- | --- |
| `same-surface` | Two objects lie on the same winding/sheet: surface↔surface overlaps, fiber/track↔surface contacts. The detector default | Detectors (below), PCL ingestion |
| `point-surface` | Point-collection points lying on a surface. Anchors retain point IDs and source XYZ | Point-collection–surface detector, PCL source links |
| `crossing` | Two opposite-family tracks in one dataset share voxels; anchors use raw point indices and source XYZ | Track–track detector |
| `fiber_branch` | A fiber branches into another at anchored control points. `meta.adjacent=True` marks an **adjacent** (next-winding) branch; absent/false is an **ordinary** (same-winding) branch | Fiber JSON import/export |
| `attachment` | A data object describes another object (growth corrections, fit results on a surface). **Not a graph link** — see below | `attachmentEdge`, document attachment |
| `declared_reference` | An application-declared reference between objects | Application code |

Point-collection–surface edges also encode winding roles: a collection
ingested with an absolute-winding role links absolutely; a relative one
carries a delta along `wind_a` (same-winding is delta 0). The role comes from
ingested metadata, or the detector's `--pcl-absolute` / `--pcl-relative`
override.

## Attachments are not graph links

`attachment` edges tie a data object to the object it describes (VC3D growth
corrections, Lasagna fit results on a surface). `neighbours`, `component`,
`reachable_within`, `connected_to` selections, and `graph_read` **skip**
attachments unless the `attachment` kind is explicitly requested; `edges_of`,
`select_edges`, and exports list them. Such point collections carry a record
meta `role`; `link_point_collections_to_surfaces` never links them and
rejects them when named.

## Same-winding queries must pass `adjacent=False`

On `select_edges`, `neighbours`, `component`, `reachable_within`, and
`graph_read`, the `adjacent` filter selects which `fiber_branch` edges count:

| `adjacent=` | Selects |
| --- | --- |
| `False` | Ordinary (same-winding) branches, including branches without the flag |
| `True` | Adjacent branches only |
| `None` | No filter (both) |

Explicit filters exclude other edge kinds and malformed flags before
traversal, parallel edges included. Fiber imports reject contradictory
ordinary/adjacent classifications, reciprocal declarations included,
atomically.

## Applicability

`store.relationships_at_versions(refs)` (C++ `relationshipsAt`) reports each
edge with an `applicable` flag: true only when both selected endpoint
versions are the exact versions the edge pins, and their geometry, metadata,
and lifecycle dependencies still match what the anchors declare. Missing
endpoints are inapplicable. Unrelated point or metadata edits leave an edge
applicable; physical or domain validity (is this really the same sheet?)
remains the caller's judgement.

`applicable_only=True` on traversals and edge selections checks current
shared membership and dependencies; `include_retired=True` never overrides
applicability. Applicability queries need a read transaction, and edge
pagination applies after applicability filtering.

## Decisions when geometry changes

Editing a dependency an anchor declares requires a decision for that edge.
Preview with `store.affected_geometry_edit_edges(uuid, edits, version)` (or
`affected_geometry_edges` for full replacement), then supply `edge_edits` —
each `{edge_uuid, expected_generation, action, a_revision, b_revision}`
(geometry revision UUIDs or `None`; `update` also supplies `a_anchor` /
`b_anchor`) — and expect every validation endpoint in `expected_versions`:

| Action (`edge_edits`) | C++ | Effect |
| --- | --- | --- |
| `keep_recorded` | `keep_recorded_relationship` | Keep the edge with its historical pins. It may be inapplicable to the new geometry |
| `retain` | `retain_relationship` | Revalidate against the resulting endpoints and publish shared new pins — the edge now applies to the new geometry |
| `update` | `update_edge` | Publish caller-supplied anchors, after validation |
| `remove` | `remove_edge` | Remove the current relationship |

```python
edits = [{"array": "points", "element_offset": 0, "data": ...}]
for affected in store.affected_geometry_edit_edges(a, edits, version):
    ...  # build one edge_edit per affected edge_ uuid
revision = store.edit_geometry(a, edits, version, edge_edits=[...])
```

Shortcuts: `tx.retain_relationship(edge)`,
`tx.keep_recorded_relationship(edge)` and
`tx.keep_recorded_relationships_of(uuid)` (every relationship of an object,
as recorded) decide inside a transaction without a geometry op — e.g. after
retire/restore. In typed requests, `keepAffectedRecorded=True` on a geometry
operation (or `UpdateGeometryArraysOp` with `affected="keep_recorded"` /
`"retain"`) decides every otherwise-undecided affected edge the same way; the
server decides which edges are affected from the checked state. Decisions on
metadata or lifecycle changes must match the transaction's final state at
commit.

## Traversal and graph reads

```python
store.neighbours(a, adjacent=False)                      # direct neighbours
store.component(a, adjacent=False, applicable_only=True) # connected component
store.reachable_within(a, depth)
```

For bulk work, `graph_read` pins a graph view over current records **or** an
explicit numbered-version selection:

```python
with store.graph_read(versions=refs, adjacent=False, approval_states=[]) as graph:
    component = graph.component(seed)
    records = graph.objects()
    links = graph.edges()
    packed = graph.read_geometry_arrays("points", nodes=[0, 1])
```

- Default `approval_states` is approved-only; `[]` includes every state.
- A `versions=` graph always requires applicability, keeps isolated selected
  nodes, and cannot combine with current-object filters or `include_retired`.
- Nodes are in canonical UUID order; traversal is breadth-first in edge-UUID
  order. Current queries otherwise inspect current records.
- Graph views are thread-confined and must close before checkpointing.
  Returned NumPy arrays are read-only and stay valid after closure (their
  buffers are retained, not the transaction).
- C++: `neighbours`, `component`, and `graphReadViewAt(txn, versions)` on
  `ObjectStore`; the same through any `StoreReader` (local or hosted). Hosted
  graph traversal is a wire query — see [hosted](hosted.md).

Fiber consumers need no separate fiber layer: fibers use these same selection,
region, geometry, and relationship APIs. For transport,
`fiberNetworkJson(reader, versions, roots)` exports fibers pinned at numbered
versions together with their applicable `fiber_branch` edges — supply a
version for **every** endpoint, retired ones included; empty roots exports
all active selected fibers, otherwise their complete linked components. The
result is keyed by object UUID with `UUID.json` references, stable
control-point anchors resolved to positional indices, and reciprocal
ordinary/adjacent branches with approval state and edge UUIDs.

## Edges imported from sources

Ingestion imports the links a source declares, in the same transaction as the
objects:

- **Fiber branches** become `fiber_branch` edges: version-4 `target_uuid`
  endpoints (version 3 resolves `branch_file`; a link declared from both ends
  imports once, contradictions roll the batch back). Export writes them back
  reciprocally.
- **PCL point and winding links** resolve by document-scoped identity
  (`meta.source.origin`); duplicate active identities fail, and only links
  that resolve within the batch (or to already-stored, pinned endpoints)
  import. See [ingestion](ingestion.md#point-collections).
- Missing or ambiguous link targets fail the batch — there is no
  filename-based repair.

## Discovering edges: detectors

Detectors ([object_links/ObjectLinks.hpp](../include/vc_object_store/object_links/ObjectLinks.hpp))
work over **stored** geometry only — source files are never reopened (CT is
still an input to the line–surface detectors, for its normal field). They
compute outside write transactions and publish pending edges named
`meta.detector = {name, version, score?}`:

```python
report = store.link_surface_overlaps()
report = store.link_point_collections_to_surfaces()          # tolerance=..., absolute_winding=...
report = store.link_tracks_to_tracks()                       # dataset_ref=..., track_uuids=[...]
report = store.link_fibers_to_surfaces("/data/ct.zarr")      # ct_level=..., strip cache, scales
report = store.link_fibers_to_surfaces(normal_pool="/cache/normals_g4", ct_to_base_scale=4)  # lasagna pack
report = store.link_fibers_to_surfaces()                     # no normals: centerline-only
report = store.link_tracks_to_surfaces("/data/ct.zarr")
# report: {objects_considered, candidates_found, edges_created, edges_updated,
#          edges_skipped_existing, stale_removed, problems}
```

```sh
vc_object_links --db objects.vcobj --author "$USER" --link-surface-overlaps
vc_object_links --db objects.vcobj --author "$USER" --link-pcl-surfaces --pcl-tolerance 2.5
vc_object_links --db objects.vcobj --author "$USER" --link-track-tracks
vc_object_links --db objects.vcobj --author "$USER" --link-fiber-surfaces --ct /data/ct.zarr
vc_object_links --db objects.vcobj --author "$USER" --link-track-surfaces --ct /data/ct.zarr
```

#### Normal sources of the line–surface detectors

Fiber- and track-surface linking widens each centerline into a strip that
follows the sheet normal. The normal source is at most one of:

- CT (`--ct`, `ct_volume`): structure-tensor normals from uint8 Zarr data with
  complete 109³ support.
- Lasagna normals (`--normal-x/--normal-y`, `normal_x_volume`/`normal_y_volume`):
  paired uint8 nx/ny stores, read at `<store>/<ct_level>` and decoded as the
  spiral fitter does: `(v - 128) / 127`, `nz = sqrt(1 - nx² - ny²)`, with
  `nx = ny = 0` meaning no data. Nearest-voxel lookup; `ct_to_base_scale` is the
  store's downsampling (the spiral's `lasagna_scale`).
- Packed lasagna normals (`--normal-pool`, `normal_pool`): the same normals from
  a resident-pool pack (spiral-fitting's `pack_resident_pools.py`: `meta.json`,
  `table.npy`, `channel_0.u8`/`channel_1.u8`), sampled and decoded identically,
  bricks read on demand; a brick the pack dropped reads as no data. `ct_level`
  does not apply. The fiber strip cache keys it by the pack's recorded dataset
  identity and CT mask rather than its path. A pack is a specific directory,
  not a cache root: the detector reads only the pack named.
- None: no strip is built. Anchors compare the centerline points themselves
  with each surface, by distance only (same sheet within `same_sheet`, adjacent
  within `adjacent_low..adjacent_high`), under the same strip gate ratio/margin.

Naming several sources is a parameter error (`ValueError`), before any work.
A source that cannot be opened fails the run (standalone: `RuntimeError`; in
`link_objects`: a preparation failure, nothing written). No source is ever
substituted for an unavailable one. Edges record the choice in
`meta.normal_method` (`ct-…`, `lasagna-…` or `centerline`) and
`meta.normal_source` (the CT location, `{normal_x, normal_y}`,
`{normal_pool}`, or null). Native CT opening supports local and anonymous
HTTP/S3 arrays and pyramid levels; credentials and disk-cache configuration
are not exposed by this adapter. Codec support follows vc_reader.

### Linking as a checked request: `link_objects`

A `link_objects` operation ([object_links/LinkObjects.hpp](../include/vc_object_store/object_links/LinkObjects.hpp))
runs an ordered plan of detectors and publishes every link they found,
together with the store's **link marker**, in one commit — on a local store
and through `vc_object_server` alike, so a client of a served store never
runs a detector itself. The request is *prepared* before the writer lock is
taken (`ObjectStore::prepare`: the detection, over one read snapshot) and
committed by `submit` expecting every object the detectors read at the
version they read: a commit touching one of them in between is an ordinary
`TransactionConflict` and nothing is written; an unrelated commit does not
interfere. The writer is held only for the publication, as with the
standalone detectors. An identical retry replays the saved receipt.

```python
plan = {"detectors": [
            {"detector": "point-collection-surface", "params": {"tolerance": 2.5}},
            {"detector": "fiber-surface", "ct_volume": "/data/ct.zarr",
             "params": {"fiber_uuids": [...], "surface_uuids": [...], "ct_level": 1}},
            {"detector": "surface-overlap", "params": {}}],
        "marker": {"key": "..."}}                       # kept in the marker's metadata
request = {..., "operations": [{"op": vos.LINK_OBJECTS_OPERATION, "plan": plan}]}
receipt = store.submit(request, progress=lambda stage, done, total, detail: ...)   # local: detectors run here
receipt = hosted.submit(request)                                                  # served: they run on the service
marker = store.object_by_uuid(vos.link_marker_uuid())    # kind LINK_MARKER_KIND
```

`params` take the keyword names of the Python detector calls below; the
line–surface detectors name a CT normal source as the step's `ct_volume`,
and a lasagna source as the params `normal_x_volume`/`normal_y_volume` or
`normal_pool` (none of them: centerline-only linking). The link
marker is one object per store: its metadata is the plan's `marker` content
plus `commit_sequence` (the links' commit) and `detectors` (each step with
its `report`), so a client reads it like any object to decide whether the
links are current. Two link runs prepared over the same state conflict on
the marker. A detector that cannot run (an unreadable normal source, missing
inputs) fails the whole request, whether the detector reports the failure
or throws it: no step's links are published, the marker does not move, and
the error names the detector; candidates a detector
rejected are not failures and stay in its report's `problems`. A
preparation is bound to its request's digest and to the store it ran on:
`submit` refuses it for any other request or store.

On a served store the detectors read their CT or Lasagna normals where the
service runs: a `ct_volume`, `normal_x_volume`/`normal_y_volume` or
`normal_pool` must be a URL or a path readable on the service's host (a
client-local path fails the request as an unreadable source; nothing is
uploaded). The strip cache is the service's own
(`vc_object_server --strip-cache`), and a request naming a `cache_directory`
is refused. The reply comes when the detection is done, so two timeouts are
set together: the service's `--request-deadline` and the client's link
timeout (`HostedStoreOptions::linkTimeoutS`, Python `link_timeout_s`; both
default to 300 s) must exceed the longest link. A client that gives up
earlier retries with the same request UUID; the service keeps every link
request in flight by its UUID, so the retry waits for the detection already
running (or replays its receipt) instead of starting another, and a
same-UUID request with other contents is refused. `GET /v1/service` reports
`link_requests` (`in_flight`, `waiting_connections`, `detections`).

| Detector (`meta.detector.name`) | Pairs | Default edge kind | Notes |
| --- | --- | --- | --- |
| `surface-overlap` | surface ↔ surface | `same-surface` | 3D correspondences, robust UV-pose RANSAC/refit, dense overlap agreement. Key params: `tolerance=2.0`, `sample_spacing=4.0`, `uv_inlier_tolerance=3.0`, `min_inliers=32`, `min_major_spread=20`, `min_minor_spread=5`, `max_refit_rms=2.0`, `min_agreement_ratio=0.6` |
| `fiber-surface` | fiber ↔ surface | `same-surface` | CT normal-field strip around the centerline; anchor growth separates same-sheet from adjacent-sheet contacts. Only line points between the projected endpoint controls participate |
| `track-surface` | track ↔ surface | `same-surface` | Same discovery, strip, and gates as fiber–surface |
| `track-track` | track ↔ track | `crossing` | Opposite-family tracks in one dataset sharing voxels. `min_angle_degrees=30`, `tangent_radius_voxels=12`, `cluster_index_radius=4` |
| `point-collection-surface` | point collection ↔ surface | `point-surface` | `tolerance=2.5` working voxels; general collections prefer the largest valid-quad area then distance, `between_surfaces__A__B` collections prefer the nearest within that named pair |

Distance and scale parameters are in **working voxels**: `*_scale` maps
source XYZ to working XYZ, `working_to_base_scale` / `ct_to_base_scale`
address the selected CT level (the level's scale is included — no extra
pyramid scaling). An empty UUID selection means all objects of the relevant
kind; unknown or wrong-kind UUIDs are errors. Inputs that change during
discovery abort publication, and failed inputs are excluded from stale
removal. CT normals use uint8 Zarr data (local or anonymous HTTP/S3, any
pyramid level; codec support follows `vc_reader`; credentials and disk-cache
configuration are not exposed).

Re-running a detector is safe and scoped:

- Candidates matching an existing edge (kind, endpoints, anchor indexes)
  refresh **that detector's own pending** edges in place, preserving UUIDs
  and annotations; unchanged ones are skipped.
- With stale replacement on (default), the detector's own pending edges that
  are no longer found are removed — over the edge kinds in the batch, or an
  explicit stale scope. Approved, rejected, manually created, and other
  detectors' edges are never touched.
- C++ pipelines can commit their own candidates with `commitCandidates`
  (whole-store, or in a caller-owned transaction to publish new objects and
  their links atomically), under their own detector name/version.
