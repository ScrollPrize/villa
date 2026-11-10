# Writing

New content enters a store three ways: objects and geometry created directly
in code, native files ([ingestion](ingestion.md)), and files attached to an
object version ([artifacts](#artifacts)). All of it publishes through the same
checked transaction model: a write names the numbered versions it read, and a
stale expectation conflicts without writing anything. Exports and project
records are covered here too. Changing existing geometry is
[editing](editing.md).

## Checked transactions

```python
observed = {u: store.current_version(u) for u in (a, b)}
with store.write(expected_versions=observed) as tx:
    tx.set_meta(a, "review_note", "checked")
    tx.set_meta(edge, "confidence", 0.9)
print(tx.receipt["commit_sequence"], tx.receipt["objects"])
```

`store.write(...)` accepts:

| Argument | Meaning |
| --- | --- |
| `expected_versions` | `{uuid: version}` for every object changed; `None` expects absence (creation — preselect the UUID) |
| `expected_relationships` | Same, for relationships |
| `read_dependencies` | Versions of records read but not changed; checked, never versioned |
| `expected_commit_sequence` | Guards a query's complete membership/absence; any intervening commit conflicts |
| `request_uuid`, `author` | Retry identity and attribution (see [typed requests](#typed-requests-submit)) |

Every object a transaction changes — including both endpoints of any
relationship it touches — must be expected. One commit publishes at most one
revision per changed object, shared endpoints included. After the block,
`tx.receipt` holds the commit sequence and the new version of every expected
and changed record (`None` for deleted).

C++ uses `store.write(TransactionContext{...})`. Through a session, reads go
through `readStore` and writes are typed requests built with `newRequest` and
applied with `submit` — see [sessions](hosted.md#sessions-objectstoresession).
Plain `store.write()` with no expectations serializes read/modify/write under
the writer lock, but cannot protect a read made before it. Standalone Python
mutations (`store.set_meta`, `store.put_geometry`, ...) each commit their own
transaction.

For a loaded object in a collection, use `expectObject(request, selected.reference())`: it carries the complete store/lineage/object/version identity and checks it under the writer lock. Equal numeric versions from different owners are not interchangeable. Typed requests with these guards use command encoding 2 (`expected_object_references`); ordinary number-only requests retain encoding 1, including their retry digest. Both encodings are read, and an older server rejects encoding 2 rather than silently ignoring its guards. This is a command-format change, not a database-schema change.

Editable `AttachedCollection` values carry both the loaded reference and the coordinate mapping used to display their points. Preserve both through UI edits. `writeVolumeCollections` checks both before applying inverse coordinate edits; reloading is required if either the object or its registration changed. Unrelated store commits do not invalidate the mapping.

## Creating objects and geometry

```python
from vc_object_store import PointCollection, Track

uid = store.upsert_object({"kind": "point_collection", "name": "seeds",
                           "tags": ["review"]})
revision = store.put_geometry(uid, PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]]))

track = store.upsert_object({"kind": "track",
                             "track": {"external_key": "t0", "family": "vertical"}})
store.put_geometry(track, Track([[1, 2, 3], [4, 5, 6]]))
```

- `upsert_object(record)` creates (or replaces the record of) an object and
  returns its UUID. Pass `"uuid"` to preselect identity. Inside
  `store.write()` use `tx.upsert_object`. Per-kind detail dicts (`fiber`,
  `surface`, `track`) carry their summaries (counts, length, family, bbox).
- `put_geometry(uuid, geometry, expected_version=None)` publishes the first
  geometry revision. Python accepts an owned geometry value
  (`PointCollection`, `Fiber`, `Track`, `Surface`, `Umbilicus` — see
  [typed geometry](typed-geometry.md)) or an array dict
  (`{"kind": ..., "attributes": {...}, "arrays": [{"name", "dtype", "shape", "data"}]}`).
  It returns the new geometry revision UUID.
- Creation is one-way: sequence geometry (fibers, point collections, tracks)
  is created exactly once with `put_geometry`. Later changes are
  [edits](editing.md); once structural management has begun, full
  `put_geometry` replacement is rejected.
  `store.affected_geometry_edges(uuid, geometry)` previews the relationship
  decisions a full replacement with that geometry would need.
- Initial per-point annotations from source metadata are consumed into the
  first revision. Fiber controls and point collections have stable point IDs
  from creation; fiber JSON transfer preserves `control_point_ids` and
  allocates IDs when a new input omits them.
- A surface placeholder bound for later loading is an ordinary record with
  `storage_kind`/`storage_ref` — see
  [ingest_registered_geometry](ingestion.md#from-placeholders-ingest_registered_geometry).

Creating a surface from a grid, in a volume's voxels, with provenance:

```python
from vc_object_store import surfaces

uuid = surfaces.create_surface(
    "objects.vcobj", xyz,                    # (rows, columns, 3) in `volume` voxels
    name="fit 42", scale=1.0, volume="20260411134726",
    tool="lasagna", parameters={...}, sources=[seed_uuid],
    channels={"ink": ink_array}, tags=["fit"],
)
```

`surfaces.create_surface` / `surfaces.read_surface` take a store **location**
(a file path or a `vc_object_server` endpoint), so the same call works on a
served store; `locations.attached(location)` gives a short-lived session
attachment for other work. C++ uses `createSurfaceOperations`, which returns
the operations creating the record and its geometry in one checked request.

Copying an object between stores is a session operation:
`session.copy_object(from_attachment, to_attachment, uuid)` copies the
current record and geometry under a new UUID (`meta.copied_from` names the
source store, object, numbered version, and geometry revision; point
annotations kept). Relationships, history, and approval stay behind.

## Typed requests (`submit`)

C++ and hosted writes share one serializable request form
([objects/Transactions.hpp](../include/vc_object_store/objects/Transactions.hpp)):

```cpp
vc::objects::TransactionRequest r;
r.requestUuid = vc::objects::util::newUuid();
r.storeUuid = status.storeUuid; r.generation = status.generation; r.author = "reviewer";
r.expectedObjects = {{objectUuid, 3}};
r.operations = {vc::objects::SetMetaOp{objectUuid, "review_note", R"("checked")"}};
const auto receipt = store.submit(r); // receipt.objects[objectUuid] == 4
```

- A request carries a client-minted `requestUuid`, expected store and
  generation, author, expected object/relationship versions, read
  dependencies, an optional commit-sequence guard, and operations:
  `CreateObjectOp`, `UpdateObjectOp`, `SetMetaOp`, `RemoveMetaOp`,
  `SetApprovalOp`, `RetireObjectOp`, `RestoreObjectOp`, `DeleteObjectOp`,
  `AddEdgeOp`, `UpdateEdgeOp`, `RemoveEdgeOp`, `RetainRelationshipOp`,
  `KeepRecordedRelationshipOp`, `PutGeometryOp`, `EditGeometryOp`,
  `EditGeometryRectangleOp`, `SetSurfaceChannelsOp`, `UpdateGeometryArraysOp`,
  `IngestGeometryOp`, `RecordCatalogTransformsOp`, `CheckVolumeMappingOp`.
- **Retries are safe.** An identical retry returns the saved receipt
  (`replayed`) before any version check, so a lost reply never conflicts with
  its own changes. The same request UUID with different contents fails.
  Recover a lost outcome with `receipt(request_uuid)` — never resubmit under
  a new UUID.
- Stale preconditions throw `TransactionConflict` and write nothing.
  Geometry operations name each relationship decision, or set
  `keepAffectedRecorded` to keep every undecided affected relationship as
  recorded (the store decides which are affected from the checked state);
  either way the request must expect both endpoints.
- [objects/TransactionJson.hpp](../include/vc_object_store/objects/TransactionJson.hpp)
  is the canonical JSON form: binary payloads named by SHA-256 and sent
  separately. Python `store.submit(request, payloads)` takes that JSON as a
  dict or text with payloads keyed by SHA-256. Missing payloads raise/return
  their hashes: stage them and resubmit the identical request.
- Helpers fill in the expectations an edit depends on:
  `expectRelationshipEndpoints(reader, request)` (complete owner/lineage/version
  guards for far ends of an edited object's relationships), `editCurrentRequest` / session `editCurrent` (edit existing
  objects at current versions, relationships kept as recorded),
  `bindCoordinatesOperation`, `volumeTransformRecord`,
  `volumeTransformChoiceRecord`, `attachmentEdge`, `createSurfaceOperations`.
  Code holding an object keeps a `PinnedObject`
  (`pinOf(attachment, revision, activeGeneration)`); its `generation` names
  the published version for historical reads and `writeGeneration` the active
  history write requests expect — they can differ after a history reset, in
  which case reopen before editing.

## Export

Native export writes one object's revision back to its file format:

```python
ref = store.export_geometry(uuid, "/data/out")                    # current
ref = store.export_geometry(uuid, "/data/out", version=3)         # recorded version
ref = store.export_geometry(uuid, "/data/out", overwrite=True)
# CLI: vc_object_export --db DB --uuid UUID --output PATH [--version N] [--generation G] [--overwrite] [--valid-only]
```

C++ `exportGeometry` also has an overload taking the caller's `ReadTxn` (for
example inside `ObjectStoreSession::read`), so read-only stores export too and
the caller can report the same snapshot. Publication is atomic: Linux
`renameat2` or macOS `renamex_np`.

Export selects an immutable object revision — metadata and relationship
snapshot included — and stages output before publication; the return value is
the exported `{store, generation, object, version}` reference. Without
`overwrite`, an existing destination fails. Surfaces export each channel as a
sidecar `<name>.tif`, `valid` as a single-page `mask.tif` (255/0);
`--valid-only` instead writes invalid vertices as missing with no mask. Other
kinds reject auxiliary arrays. Every native exporter carries the authoritative
`meta.coordinates` (unknown values included) plus any `coordinate_binding`,
so export → import preserves the frame with no out-of-band context. Fibers
export as version-4 JSON, branches addressed by `target_uuid` with reciprocal
approval state and edge UUIDs (`fiberNetworkJson` in C++ exports whole linked
components pinned at numbered versions; `fiberVersionJson` is the
geometry-and-annotations codec).

`export_json(object_kind=None, fields=[...])` exports **metadata only**
(records, with filters and field projection) and `import_json` reads it back.
It is not a geometry backup.

## Artifacts

Artifacts are files that belong to an object but are not geometry — fitted
models, exports, logs. Their bytes are immutable and content-addressed beside
the database, at `<store>.artifacts/sha256/<2 hex>/<sha256>`; an object
version names them through a manifest in `meta.artifacts`: a sorted array of
`{name, sha256, length}`, names relative and `/`-separated. A manifest is
versioned and in the change feed like any metadata, and bytes are never
deleted while historical versions name them — back up the directory with the
database.

```cpp
const auto hash = store.putArtifact(file);   // durable (synced) before it returns
store.submit(request with SetMetaOp{object, kArtifactsMetaKey,
                                    artifactManifestJson({{"fit/model.pt", hash, size}})});
```

```python
import vc_object_store as vos

digest = store.put_artifact(path)                      # -> sha256 hex
store.artifact_length(digest); store.read_artifact(digest)
manifest = vos.artifact_manifest_json(
    [{"name": "fit/model.pt", "sha256": digest, "length": size}])   # for meta["artifacts"]
```

The shared commit path refuses a changed manifest that is malformed or names
bytes the store does not hold, whichever API wrote it. Through a session the
same works on any attachment: `artifacts(attachment, object, version)`,
`publish_artifacts(attachment, object, expected_version, {name: path}, author)`
(bytes stored durably, then named in one checked request),
`materialize_artifacts` (write a version's files under a local directory,
each atomic and verified), and `artifact_file` (one artifact as a cached local
file). CLI tools publish through `ToolObjects::publishArtifacts`, expecting
the version they read. Hosted clients get `putArtifact` / `fetchArtifact`
(verified, atomic). The older mutable per-object artifact directories are no
longer read or written, and their contents are not migrated.

All three session artifact readers accept `reference={store, generation, object, version}` (C++: the optional `ObjectVersion` argument). This selects the exact manifest through the shared version resolver, including a retained referenced version after an overlay edit or deletion. It cannot be combined with `version`/`generation` and must name the requested object. The older number-only selectors use the current owner's history; they do not identify a different source owner. Geometry and model-checkpoint consumers must carry the same complete reference to both reads.

## Derived geometry versus copying a store

Python writers of newly computed geometry can use `vc_object_store.derivation.derived_metadata(metadata, policy=..., reference=...)`. The default `portable` policy preserves application metadata but rejects managed `vcobj_extensions` and `artifacts` descriptors: JSON alone cannot copy their payloads. Explicit `geometry_only` removes those managed descriptors and records the policy and complete selected source reference in `meta.derivation`. It does not copy relationships or history. Use the shared collection-copy API when those data must be preserved. This policy is shared by the neural fiber writer/snapping workflow; callers must not independently strip or reinterpret managed descriptors.

## Projects

A store is also a project ([objects/Project.hpp](../include/vc_object_store/objects/Project.hpp)):
its resource records reference the volumes, normal grids, and Lasagna datasets
it uses, and each author keeps a `project_view` record of their own.

```python
import vc_object_store as vos

vos.create_project("project.vcobj", "scroll", {"open_data": {"sample": "PHerc0139"}},
                   author="reviewer")   # creates the store with its project record
with ObjectStore.open("project.vcobj", author="reviewer") as store:
    uuid = store.add_project_resource("volume", "/data/ct.zarr", tags=["main"])
    store.project_resources("volume", viewer="reviewer")  # shared + that author's own
    store.project_record()                                # None on a plain store
```

Resource kinds are `volume`, `normal_grid`, and `lasagna_dataset`; a resource
is shared, or one `owner`'s alone (`project_resources(kind, viewer=...)`).
`vos.project_resource_uuid(kind, location, owner="")` is the deterministic
identity `add_project_resource` returns.

Documents attached to objects (point-collection documents such as growth
corrections) go through `documents.attach_document(location, ...)` /
`attached_document` / `point_collections` — one checked request each, on a
file or endpoint alike; their edges are `attachment` edges, not graph links
(see [graph edges](graph-edges.md#attachments-are-not-graph-links)).
`copyProject` uses the shared [collection copy](collections.md#independent-import-shards) API: the mutable root, referenced/sharded databases and artifact payloads are copied as one portable bundle, not just the attached physical file.
