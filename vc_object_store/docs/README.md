# Object store

`ObjectStore` keeps objects, immutable geometry, metadata and relationships in one
SQLite file. Source files are needed for ingestion; subsequent queries, edits,
detection and export use stored revisions. CT remains an input to fiber/track
surface detection. See [build/CLI instructions](../README.md),
[typed geometry](typed-geometry.md) and [structural editing](structural-editing.md).
C++ callers link `vc_objects` for storage, `vc_object_geometry` for native I/O,
and `vc_object_links` for detectors. Public signatures live in
[ObjectStore.hpp](../include/vc_object_store/objects/ObjectStore.hpp).

## Ingestion and export

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("surface", "/data/surface")
    assert not report["problems"], report["problems"]
    report = store.link_surface_overlaps()
    assert not report["problems"], report["problems"]
    surfaces = store.objects_by_kind("surface")
    store.export_geometry(surfaces[0]["uuid"], "/data/exported-surface")
```

Ingestion is **one-time**. Existing UUIDs (even unchanged or retired), duplicates,
and missing/ambiguous source-link targets fail. Fiber directories and PCL
documents publish atomically; packed tracks use per-track transactions and can
report partial success. Track identity is dataset reference plus external key.
Always inspect `problems`.
`ingest_registered_geometry` loads placeholders only if they have no geometry
history, preserving unrelated metadata. Use store edits for subsequent changes.
A surface placeholder with `storage_kind="url"` and an `http(s)://` or `s3://`
(public bucket) `storage_ref` is a remote tifxyz: ingestion downloads its
`meta.json`, `x/y/z.tif` and, when present, `mask.tif`,
decodes them like a local directory and stores the geometry, so the remote is
read once. The placeholder, not the remote `meta.json`, supplies the identity.

PCL source links use document-scoped collection/point identities. The namespace
is `meta.source.origin`, the absolute lexically normalized import path; symlink
aliases and moved files are different namespaces. Import only new collections
at that path to extend a document. Only active owners resolve links; duplicate
active identities fail. Derived identity indexes are maintained through
edits, restoration and sync. Replacement can scan all affected IDs;
ordinary resolution reads only referenced chunks or metadata-snapshot buckets.

Native export selects an immutable object revision and stages output before publication;
`overwrite=False` protects existing destinations. Surfaces export their channels as
sidecar TIFFs; other kinds reject auxiliary arrays.
Directory export currently requires Linux `renameat2`. Metadata-only
`export_json` supports filters and field projection; it is not a geometry backup.
Typed detail counts, lengths, dimensions and boxes are import summaries: use
current geometry descriptors and spatial queries after edits.

## Revisions and transactions

C++ applications can share an `ObjectStoreSession` to manage attachments of
existing databases (local files or hosted stores), checked writes, worker
cancellation, and journal events. Reads go through a `StoreReader`
(`readStore`); writes are checked typed requests (`objects/Transactions.hpp`):

```cpp
vc::objects::ObjectStoreSession session;
vc::objects::OpenOptions options;
options.author = "reviewer";
const auto attachment = session.attach("objects.vcobj", options);
auto request = session.newRequest(attachment);
session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto& cancel) {
    if (cancel.load()) return;
    request.expectedObjects[objectUuid] = reader.currentObject(objectUuid)->version;
});
request.operations.push_back(vc::objects::SetMetaOp{objectUuid, "review_note", R"("checked")"});
session.submit(attachment, request); // TransactionConflict when the object moved on
```

Keep all work using a reader inside its callback and do not reenter the
session from a callback. A session holds any number of attachments (for example a
project store beside a hosted team store), each with its own access mode,
writer and events; a store is attached at most once. `attach()` adds one,
`replace(attachment, ...)` swaps one for another store, `detach(attachment)` removes
one, and `attachments()` lists them in attach order. Replacement validates the new
database first, drains the old attachment's accepted writes, cancels/waits for
its readers and delivers its last commits; events report `Detached` for the old
attachment before `Attached` for the new one. Include the attachment generation in
asynchronous requests and caches: even reattaching the same store invalidates old
requests. Objects whose geometry is published elsewhere (Open Data catalog
surfaces, `objects/CatalogStore.hpp`) are fetched by the fetcher registered for
their meta key (`setGeometryFetcher`), in any writable attachment, when a read
reaches them. A store is also a project (`objects/Project.hpp`): resource
records reference the volumes, normal grids and Lasagna datasets it uses, and
each author keeps a `project_view` record of their own. `takeEvents()` polls local and external commits of
every attachment for all object kinds, tagged by attachment; call it from the
application's event loop. Each change carries
its record's object or edge `kind` (kept for deletions too), so a consumer
filters by kind without looking the record up. Tools and editors that change existing objects
without deciding about each relationship build one checked request
(`newRequest`, then `submit`) read through `readStore`: they expect the objects
at the versions they started from, add every endpoint of their relationships
(`expectRelationshipEndpoints`), and let geometry operations keep affected
relationships as recorded (`keepAffectedRecorded`, or `UpdateGeometryArraysOp`
with `affected = "keep_recorded"`; "retain" when the relationships still apply,
as for point collections whose points keep their identity). `copyObject(from, to, uuid)` copies an
object's current record and geometry from one attachment into another as a new
object (new UUID, `meta.copied_from` naming the source store, object, numbered
version and geometry revision; point annotations kept); relationships, history and approval stay
behind. An editor checks that the version on screen is still current with
`requireCurrent(pin)` before it starts editing. `finalize()` waits
for accepted writes on every attachment; requests are already committed when `submit()` returns.

Fiber consumers use the same `selectObjects`, `geometryRegion`, geometry, and
relationship APIs. Update any catalog summaries used by the application in the
same checked transaction as the geometry edit, and explicitly decide how affected
relationships apply. There is no separate fiber repository or collection layer.

For native fiber transport, `fiberNetworkJson(reader, versions, roots)` (any
`StoreReader`, local or hosted) exports geometry pinned at numbered versions
together with its applicable `fiber_branch` edges. Supply one `ObjectVersion`
for every endpoint, including retired endpoints needed to resolve applicability. Empty roots exports all active selected fibers;
otherwise the exporter includes their complete linked components. The result is
keyed by object UUID, uses `UUID.json` references, and keeps display names separate.
It resolves stable control-point anchors to positional wire indices and emits
reciprocal ordinary/adjacent branches with approval state and edge UUIDs. Coordinate
identity and bindings accompany each object. `fiberVersionJson(reader, version)`
remains the geometry/annotation-only codec for consumers that read relationships
separately.

Object revisions capture metadata, geometry identity and relationship membership.
`current_object_revision`, `object_revision`, and `object_history` expose them;
`current_geometry`, `geometry_revision`, and `geometry_history` expose geometry.
Their UUIDs are distinct: metadata edits need not create geometry revisions.

Pin all modified objects and participating relationship endpoints when preparing
an edit outside the writer lock:

```python
observed = {u: store.current_version(u) for u in (a, b)}
with store.write(expected_versions=observed) as tx:
    tx.set_meta(a, "review_note", "checked")
    tx.set_meta(edge, "confidence", 0.9)
print(tx.receipt["commit_sequence"], tx.receipt["objects"])
```

C++ uses `store.write(TransactionContext{...})` (`expectedObjects`). `None` means
the object must not exist; preselect its UUID. `write()` also takes
`expected_relationships`, `read_dependencies` (versions of records read but not
changed) and `expected_commit_sequence`. Stale/missing expectations and caught geometry
revision conflicts abort the entire checked action. Plain `write()` serializes
read/modify/write under its lock but cannot protect an earlier external read.
Standalone Python mutations commit their own transaction. Read transactions pin
a snapshot; writer revision reads describe the last published state. Each commit
publishes at most one revision per changed object, including shared endpoints.

`published_at_us` is UTC Unix microseconds, shared by revisions in one action and
strictly increasing between actions even if the clock goes backward.
`versions_at(time)` (C++ `versionsAt`) returns the numbered version
`{store, generation, object, version}` of every object live at that time;
read them with `objects_at_versions` and `relationships_at_versions`. Objects
created later or physically deleted by then are excluded. Retired objects are
omitted unless `include_retired=True`. Public history references are always
numbered versions; revision UUIDs are internal storage keys. Restore historical
content by publishing a new checked action, not by moving old heads.

### Hosted attachments

`attachHosted(endpoint, HostedSessionOptions)` attaches a store served by
`vc_object_server` beside local attachments. The session never opens its
database; the session's operations work on every attachment:
`status`, `submit` (checked requests; conflicts throw `TransactionConflict`),
`receipt`, `currentObject`, `objectsAtVersions`, `geometryRevision`,
`readGeometryArray` and `volumeTransforms`. `selectObjects`, `geometryRegion` and
`nearestSurfaces` combine hosted results with local attachments and drafts in
attachment order. A hosted store answers them on the host: it resolves volume
targets through its recorded transforms, a region query is not cancellable, and
explicit transforms are refused.

`readStore(attachment, [](const StoreReader& reader, const auto& cancelled) {...})`
gives code one read interface for both backends (`objects/StoreReader.hpp`):
status, current and historical records, geometry descriptors, array ranges,
packed arrays and surface rectangles, point annotations, volume transforms,
complete object and edge listings (with projections), object counts and tags,
graph components and neighbours, and fiber/track lengths in a volume. Region
and proximity queries stay session operations (they combine attachments and
drafts). `GeometrySnapshot(reader)` builds the usual geometry views over it, transformed
views included. A local reader answers from one read transaction. A hosted one
answers each call from the attachment's cache or the host: answers about
numbered versions and geometry revisions are exact, while current heads may come
from different commits. `LocalStoreReader(store, txn)` wraps a transaction the
caller already holds. The reader and its views must not outlive the callback.

A background check every `pollInterval` asks for the store's status and, when
the commit sequence moved, reads the change feed. It delivers the commits through
`takeEvents()` as `Committed` events: `journalPosition` is the commit sequence,
and each change carries its record kind, or an empty kind when it cannot be
learned (`touches()` counts such a change). When the feed position is lost
(another history generation, a restored store), the event has `revalidate` set:
reread whatever is shown. `refresh()` checks now. `freshness()` reports the last
successful check and the last failure for the UI.

The attachment caches what it read, within `cacheBytes`: geometry descriptors
and arrays by revision, records at numbered versions, and current heads until
the feed reports them changed. Cached heads are therefore current as of the last
check. The session's own submissions invalidate theirs at once. Cached data
stays readable while the host is unreachable; anything else fails, and listings
and spatial queries always need the host.
Arrays too large for the cache are read in requested byte ranges; rectangular
surface reads fetch only their row ranges instead of downloading the whole array.

With a selection limit, `selectObjects` gives each attachment up to `limit`
records that earlier attachments do not hide. It reads further pages when some
are hidden, so a short result means there are no more.

Artifacts work the same way on every attachment. `artifacts(attachment, object,
version)` lists an object version's manifest. `materializeArtifacts` writes those
files under a local directory, each one atomic and verified.
`publishArtifacts(attachment, object, expectedVersion, files, author)` stores the
bytes durably, then names them in the manifest in one checked request (an empty
author is the attachment's). `artifactFile(attachment, object, name,
cacheDirectory)` returns one artifact as a local file kept by content
(`<cacheDirectory>/<sha256>/<file name>`, written unless it is there); Lasagna's
service reads fitted models this way. Command-line tools publish through
`ToolObjects::publishArtifacts`, expecting the version they read (or the
current version of an object they created). `copyProject` copies the
artifact directory with the database.

### Durable drafts

Drafts live in the session's memory. `useDraftDirectory(directory)` makes them
durable: `checkpointDrafts()` writes every draft (its grid, coordinates,
attachment store and `base`, the committed version it edits) and every pending
submission. Only drafts changed since the last checkpoint are rewritten, each to
a new file, and then `checkpoint.json` is replaced atomically and synced. A crash
or a failed write (`draftCheckpointError()`) therefore leaves the previous
checkpoint complete. The application schedules checkpoints; they are separate
from its saves. Call `useDraftDirectory` once, before making drafts: it installs
the last checkpoint's drafts and pending submissions, so edits made after that
checkpoint are lost. A directory serves one session at a time.

`submitDurably(attachment, request, drafts)` is a checked save that survives a
crash. The exact request and its payloads are written, and a checkpoint
completed, before it is sent. Success clears each listed draft that did not
change after the call; one that changed stays as a draft based on the committed
version. A conflict keeps the drafts and records it (`draftState().conflict`);
nothing is resent. Reapply by setting the draft again with a new `base`. When a
hosted store cannot be reached, the outcome is unknown: the submission stays
pending (`pendingSubmissions()`, also after a restart) until
`resolvePending()` finds its receipt or sends the identical request again.

Editors that keep a draft open while they save use the `DurableSave` form:
`drafts` names each draft with the revision whose grid the request holds (a save
copied on one thread while edits continue on another), and `keepDrafts` keeps
the drafts after success. A kept draft moves to the committed version and, when
it did not change after its saved revision, is marked saved. `markDraftSaved`
does the same for a draft that holds its base's committed geometry (a surface
just opened). Checkpoints leave saved drafts out, so opening large surfaces
writes nothing; `draftState().saved` reports it. `draftContents()` returns a
draft's grid, coordinates and attachment (for an application restoring a
recovered draft). A pending save recovered after a restart clears its drafts on
success: the editor that kept them is gone.

## Numbered history and checked requests

Schema 18 introduced numbered history; schema 19 stores it compactly, and schema 20 (current) drops the hub's replica identity table. Each object and relationship keeps its UUID; each
published state also has a version, starting at 1 and advancing once per
transaction that changes the record (deletions and relationship removals are
numbered too, so a deleted identity never looks newly absent). A historical
reference is `(store UUID, publication generation, record UUID, version)`;
`objectAtVersion`/`relationshipAtVersion` resolve it only in the generation it
was published in; `objectsAtVersions` reads many references in order in one
snapshot (versions that are still heads skip history replay). `startHistoryGeneration` (restore tooling) starts a new
generation for new writes without relabelling recorded versions.

Every transaction that publishes something advances the store's commit
sequence by one and appends its changes to an ordered feed:
`status()` returns store UUID, active generation, commit sequence and feed
start; `changesSince(cursor, maxChanges)` returns whole commits (continuing
inside a commit only when it alone exceeds the page). A cursor from another
generation, from the future or before the feed start comes back invalid:
revalidate cached heads and rerun queries.

Typed requests carry a client-minted request UUID, the expected store and
generation, an author (attribution, not authentication), expected object
versions (every changed object, `null` for creation), optional relationship
versions, read dependencies (checked, never versioned) and an optional commit
sequence guard for query membership/absence:

```cpp
vc::objects::TransactionRequest r;
r.requestUuid = vc::objects::util::newUuid();
r.storeUuid = status.storeUuid; r.generation = status.generation; r.author = "reviewer";
r.expectedObjects = {{objectUuid, 3}};
r.operations = {vc::objects::SetMetaOp{objectUuid, "review_note", R"("checked")"}};
const auto receipt = store.submit(r); // receipt.objects[objectUuid] == 4
```

An identical retry returns the saved receipt (`replayed`) before any version
check, so a lost reply never turns into a conflict with its own changes; the
same request UUID with different contents fails. Stale preconditions throw
`TransactionConflict` and write nothing. `objects/TransactionJson.hpp` gives the
canonical JSON form (binary payloads named by SHA-256, sent separately) and
the request digest. `write(TransactionContext)` offers the same checks around
direct `ObjectStore` calls; every committed write has `Txn::receipt()`.
`recordCatalogTransforms` publishes catalog registrations as versioned records
that every client then resolves from the store.

Geometry operations name each relationship decision explicitly, or set
`keepAffectedRecorded`: every relationship the change affects without its own
decision is then kept as recorded (its historical pins stay; the server decides
which relationships are affected from the checked state). `SetSurfaceChannelsOp`
replaces or removes surface channels. `UpdateGeometryArraysOp` makes the
current arrays equal whole desired arrays (`updateGeometryArrays`: sparse
deltas, sequences keep their stable point identities, optional replacement
per-point annotations) and decides every affected relationship one way,
`retain` or `keep_recorded`. Through a session,
`newRequest(attachment)` starts a request for the attachment's store, active
generation and author. Helpers over a `StoreReader` fill in what an edit
depends on: `expectRelationshipEndpoints` expects the far ends of an edited
object's relationships. Put a transformed view's `checkMappingOperation()`
(`CheckVolumeMappingOp`, wire name `check_volume_mapping`) before its inverse
geometry edits: the writer resolves the mapping again and conflicts if it is
unavailable, ambiguous or changed, including newly added registrations or
route choices. The request must also expect the object's numbered version.
`transformDependencies` guards existing registration records only; it does not
guard additions to that set. `editCurrentRequest` builds an edit of
existing objects at their current versions that keeps their relationships as
recorded, and the session's `editCurrent(attachment, uuids, operations)` and
`editRecords` read, build and submit one (on any attachment).
`bindCoordinatesOperation`, `attachmentEdge`, `volumeTransformRecord`,
`volumeTransformChoiceRecord` and `setVolumeUmbilicusRequest` give the
operations of the corresponding store writes; `registerVolumeTransform`,
`chooseVolumeTransform` and `importVolumeUmbilicus` submit them.
`createSurfaceOperations` gives the operations that create a surface record
and its geometry. Code holding an
object keeps a `PinnedObject`: its version and history generation (`pinOf`),
which requests expect.

Writable opens hold a host-local ownership lock on `<canonical path>-owner`:
shared between standalone writers, exclusive for `OpenOptions::serviceOwner`.

### Converting schema-16 to schema-19 stores

The runtime opens schema 20 only. Stop every client, settle hub replicas with
the old tooling, then convert into a new file (the source is never changed):

```sh
vc_object_convert old.vcobj new.vcobj   # writes new.vcobj.mapping.json and new.vcobj.report.json
```

The mapping gives each old object/relationship revision UUID its numbered
reference, for converting pins in exports and job configurations. The report
lists gaps rather than guessing: untimed baselines, relationship states without
a recorded publication, geometry revisions that do not select one object
version, metadata still holding old revision UUIDs, and an unversioned inventory
of external artifact files. Converted commits have author `unrecorded`.

A schema-16 source is first brought to schema 17 on the scratch copy: its
empty `patches` detail table becomes `surfaces` and the hub's `sync_baselines`
tables are dropped. Schema-16 stores holding surface (`patch`) records or sync
baselines are refused; no known store has them. A schema-17 source is numbered first (schema 18, on a scratch copy beside the
destination), then copied into a store built by the current schema
definition. Schema 19 keeps the same content in less space: history, version
and head keys are 16-byte blobs, a revision without a parent stores its whole
record rather than a patch from `{}` (later revisions stay RFC 6902 patches),
only heads with a parent keep a materialized record, and geometry payload and
manifest hashes are 32-byte blobs (still hex text in every API). A schema-18 source
is copied directly; its mapping is read from its recorded versions. Schema 20 is
schema 19 without the one-row `sync_replica` table (the hub's replica
identity); a schema-19 source is copied as it is, with that table dropped and
its mapping read back like schema 18's. The report
gives `source_bytes`, `destination_bytes` and the rows copied per table.
Content-addressed artifacts named by current or historical versions are copied
to the destination's artifact directory and verified against their manifest
hashes and lengths. `copied_artifacts` reports the number of distinct blobs.
A missing or corrupt referenced blob fails conversion and removes the partial
destination. An existing destination artifact directory is never overwritten.

### Artifacts

Artifacts are files that belong to an object but are not geometry, such as
fitted models, exports and logs (`objects/Artifacts.hpp`). Their bytes are
immutable and content-addressed beside the database, at
`<store>.artifacts/sha256/<2 hex>/<sha256>`, and read-only. An object version
names them through a manifest in its metadata, under the `artifacts` key: a
sorted array of `{name, sha256, length}`, with relative `/`-separated names.
A manifest is therefore versioned, historical and in the change feed like any
other metadata.

```cpp
const auto hash = store.putArtifact(file);  // copied (or moved) and synced to disk
store.submit(request with SetMetaOp{object, kArtifactsMetaKey,
                                    artifactManifestJson({{"fit/model.pt", hash, size}})});
```

`putArtifact` makes the bytes durable (file, rename and directory synced)
before it returns, so a transaction can only reference bytes that survive a
crash. The shared commit path refuses a changed manifest that is malformed or
names bytes the store does not hold. This check runs whichever API wrote the
manifest, typed request or direct `setMeta`. Bytes are never deleted while
the store is in use, because historical versions keep naming them. A backup
copies this directory with the database. The older mutable per-object
directories (`<store>.artifacts/<object uuid>/`) are no longer read or
written; files left in them by earlier versions are not migrated.

### Hosting a store (`vc_object_server`)

`vc_object_server` serves one store over HTTP to clients that have no
filesystem access to it. It opens the store as its service owner, so it will
not start while a standalone writer has the store open, and standalone
writable opens fail while it runs (read-only inspection still works). It has no
authentication: keep the default loopback address and reach it through an SSH
tunnel, or listen on a trusted private network.

```sh
vc_object_server /data/project.vcobj --port 8750          # on the storage host
ssh -N -L 8750:127.0.0.1:8750 storage-host                # on each client
```

C++ clients use `vc::objects::HostedStore` (`client/HostedStore.hpp`) with the
same request and result types as the local API: `status`, `submit`, `receipt`,
`changesSince`, `currentObjectRevision`, `currentVersion`, `objectAtVersion`,
`objectsAtVersions` (batches of up to the server's query limit),
`geometryRevision`, `readGeometryArray`, `putArtifact`, `artifactLength`,
`readArtifact`, `fetchArtifact` (verified, atomic local file),
`selectObjects`/`selectEdges` (one
page), `selectAllObjects`/`selectAllEdges`, `objectHistory`, `relationshipsAt`
(numbered object versions), `geometryRegion` and `nearestSurfaces`. Conflicts throw the same
`TransactionConflict`. Transport failures, 429 and 5xx are retried with the same
request UUID, so a lost reply is answered by the saved receipt. When retries run
out the outcome is unknown: recover it with `receipt()` or by resubmitting the
identical request, never under a new request UUID.

Wire protocol (version 1, JSON unless noted):

| Request | Answer |
| --- | --- |
| `GET /v1/status` | store, generation, commit sequence, feed start |
| `GET /v1/service` | the service's own state: connections, queue depths, staging use, unsent response bytes, WAL checkpoints |
| `PUT /v1/payloads/{sha256}` (bytes, `Content-Length` required) | stages one payload until it expires; 400 hash/length mismatch, 411 no length, 413 too large, 507 staging full |
| `POST /v1/transactions` (canonical request JSON) | receipt; 409 conflict; 422 `missing` payload hashes; 400 invalid |
| `GET /v1/receipts/{request}` | saved receipt or 404 |
| `GET /v1/changes?store&generation&sequence&offset&max` | feed page |
| `GET /v1/objects/{uuid}` | current revision and the status of the same snapshot |
| `GET /v1/objects/{uuid}/versions/{n}?store&generation` | that numbered revision or 404 |
| `GET /v1/geometry/{revision}` | geometry descriptor (immutable) |
| `GET /v1/geometry/{revision}/arrays/{name}?offset&count` | array bytes; 413 with `max_response_bytes` above the response limit |
| `GET /v1/geometry/{revision}/annotations/{name}` | per-point annotations of a sequence array (exact JSON text, immutable) |
| `PUT /v1/artifacts/{sha256}` (bytes, `Content-Length` required) | streamed like payloads, then kept durably by the store (`putArtifact`) |
| `GET /v1/artifacts/{sha256}/info` | `{sha256, length}` or 404 |
| `GET /v1/artifacts/{sha256}?offset&count` | artifact bytes; same response limit as arrays |
| `GET /v1/objects/{uuid}/history` | every recorded revision, or 404 |
| `POST /v1/query/objects` (`selection`, `fields`) | one page of records; `limit` required, at most `maxQueryResults` |
| `POST /v1/query/edges` (`selection`, `fields`) | one page of edges; same limit rule |
| `POST /v1/query/relationships` (`versions`: store, generation, object, version) | selected relationships with `applicable` |
| `POST /v1/query/versions` (`versions`, at most `maxQueryResults`) | one revision per reference, in order, `null` where not recorded |
| `POST /v1/query/region` (`slice`, `options`) | region runs, coverage stop reason and stats |
| `POST /v1/query/nearest_surfaces` (proximity query) | surface hits |
| `POST /v1/query/count` (`kind`, `include_retired`) | object count of a kind |
| `POST /v1/query/tags` (`kind` optional) | distinct tags of live objects, sorted |
| `POST /v1/query/graph` (`traversal`: `component` or `neighbours`, `object`, `edge_kinds`, `neighbour_kind`, `include_retired`) | object UUIDs reached from `object` |

Every query answer carries the `status` of the single read snapshot it came
from. Selections never carry transforms: volume targets resolve through the
transforms recorded in the store (a client catalog cache cannot change a shared
query), and a request that sets them is refused. Region budgets above the
server's limits are clamped, so the answer reports the budget it stopped at
instead of claiming completeness. Every selected record carries `version` and `versionGeneration`, the numbered
head it was read from, so a listing pins exactly what it returned (read
transactions only: rows read inside a write may not be published yet).
`selectAllObjects`/`selectAllEdges` page by UUID cursor without holding a
server snapshot. When commits land between pages, the client reads them from
the change feed and selects the touched records again, restricted to their
UUIDs. It repeats until the feed is quiet, so the result is exact as of one
commit sequence, whatever the store size.

A listing restarts only when a change can move membership beyond the records it
touched: a recorded transform during a volume listing, a relationship during a
`connectedTo` listing, an endpoint during an `applicableOnly` edge listing, or
an expired feed position. After a bounded number of restarts, or when the feed
never goes quiet (sustained writes far above interactive rates), it throws
`StoreBusy`. Length-ordered and unbound-volume listings page by offset and
restart on any commit. Deleted transform records count as transform changes.

### Selection records

A selection record (`objects/Selections.hpp`) keeps the exact result of an
object selection as of one commit, so a long job, export or training set can
read its inputs back later at the same versions while the store moves on. It
holds the query, the store, generation and commit sequence, every match in
listing order at its numbered version and, for volume and coordinate
selections, the versions of the recorded transforms and route choices in force.
Capture it locally with `recordSelection(store, readTxn, selection)` (one
snapshot) or hosted with `HostedStore::recordSelection(selection)` (the
catch-up listing, then the transforms, checked against the feed since the
listing's commit). Neither accepts explicit transforms or a limit, and the
local form refuses a volume selection that would use client catalog
registrations the store has not recorded. `encodeSelectionRecord` gives the
binary form (a JSON header, then 24 bytes per object: UUID, generation index,
version), meant to be kept as an artifact; `decodeSelectionRecord` validates
it. Read the objects back with `objectsAtVersions`.

`bench_object_store_selection` builds a synthetic store and times listings, as
in this run (1M objects, 6.6 GB, Release, P-cores, 3 iterations, medians):

| Listing of 500k fibers | Time |
| --- | --- |
| Local, UUIDs only, one snapshot | 0.99 s |
| Hosted, UUIDs only, 10k pages | 2.31 s |
| Hosted, with name/fiber/track fields | 5.29 s |
| Hosted, UUIDs only, 60 commits landing during the listing | 2.79 s |
| Selection record, local (`record` mode) | 1.00 s, 12.0 MB encoded |
| Selection record, hosted, 10k pages | 2.33 s |

The server answers a request whose payloads have expired from its saved
receipt when there is one. Otherwise it names the missing payloads, and the
client uploads them again and resubmits the identical request. The in-process
form is `ObjectStore::submit(requestJson, payloadLookup)`. Storage work never
runs on network threads. Checked transactions go to one writer queue. Reads,
feed pages and queries go to bounded read workers. Status and receipt checks
have their own lane. A full queue answers 503 with `Retry-After`, which
clients retry.

Payload uploads stream to files in the staging directory (`<store>.staging`
unless `ServerOptions::stagingDirectory` names another), one chunk at a time.
A dedicated transfer lane writes and hashes each chunk, and the connection reads
the next chunk only once the previous one is on disk. A slow disk therefore
slows the sender instead of filling memory. An upload reserves its declared
length up front. When staged plus reserved bytes would exceed `maxStagedBytes`,
the server answers 507 instead of queueing the upload. A client that
disconnects mid-upload leaves no partial file or reservation. Staging is not
durable: the server removes payload files left by a previous run when it
starts. Uploads are bounded in memory, but a transaction still loads its
payloads into memory when it runs.

A geometry array read is answered in one response of at most
`maxResponseBytes`. `HostedStore::readGeometryArray` reads larger arrays as
ranges within both its own `rangeBytes` and the server's limit. While unsent
response bytes exceed `maxResponseBytesInFlight`, new array reads get 503.
Together with the per-request deadline, this bounds the memory that slow
receivers can hold.

The writer lane also runs WAL upkeep between transactions. Every
`checkpointInterval` it runs a passive checkpoint
(`ObjectStore::checkpointWal`). When that copied everything and the WAL file is
larger than `walTruncateBytes`, it truncates the WAL, waiting at most
`truncateWait` for readers. A run of incomplete checkpoints means long-lived
readers are holding old snapshots and the WAL is growing; the server logs it,
and `/v1/service` reports it. SQLite's automatic checkpoints on commit still
run.

`ServerOptions` sets the queues, body, payload, staging and response limits,
the staging lifetime and directory, per-request deadlines, checkpoint
scheduling, query page size and region budgets. `test_object_store_hosted_limits`
checks these bounds. It also measures status latency while four clients upload
at once and submit latency for 2 and 4 concurrent writers, and it recovers a
transaction whose client hung up before the reply.

### Python access to numbered history and hosted stores

`ObjectStore` (local) and `HostedStore(endpoint)` (a `vc_object_server`) take
the same forms: requests are the canonical JSON as a dict or text with binary
payloads keyed by SHA-256 (`submit(request, payloads)`, `receipt`), queries
are the wire query form (`{"selection": {...}, "fields": [...]}`) and
numbered references are `{store, generation, object, version}` dicts
(`objects_at_versions`, `relationships_at_versions`). Selected rows carry
`version` and `version_generation`. `record_selection(query)` returns a
selection record dict; `encode_selection_record`/`decode_selection_record`
give the binary form. Artifacts: `put_artifact`, `artifact_length`,
`read_artifact`, `artifact_manifest_json(files)` for the `ARTIFACTS_META_KEY`
metadata value, and on `HostedStore` also `fetch_artifact(file, destination)`.
`HostedStore` queries take `with_status=True` to also return the status of the
server snapshot they read; `select_all_objects`/`select_all_edges` raise
`StoreBusy` as in C++. Conflicts raise `RuntimeError`, malformed requests
`ValueError`.
`ObjectStoreSession` has the same session operations as C++: `attach_hosted`,
`hosted`, `freshness`, `refresh`, `status`, `submit`, `receipt`,
`current_object`, `objects_at_versions`, `artifacts`, `materialize_artifacts`,
`publish_artifacts(attachment, object, expected_version, {name: path}, author)`,
`use_draft_directory`, `checkpoint_drafts`, `draft_checkpoint_error`,
`submit_durably`, `pending_submissions` and `resolve_pending`. `take_events()`
items carry `revalidate`.

A store *location* is a store file or a `vc_object_server` endpoint
(`http://` or `https://`): `ObjectStoreSession.attach_location(location,
author, read_only)` (C++ `attachLocation`) attaches either. The session reads
and writes surfaces and attached documents on any attachment:
`read_surface_grid` (a surface's current or numbered version as a whole grid
in a volume's voxels, `objects/SurfaceGrid.hpp`), `read_geometry_arrays`,
`select_objects` (wire query form, complete), `create_surface` and
`attach_document` (one checked request each), `attached_document` and
`point_collections`. The module helpers `surfaces.read_surface`/`create_surface`
and `documents.attach_document`/`attached_document`/`point_collections` take a
location (`locations.attached` gives a short-lived attachment), so a tool on
another machine works on a served store by its endpoint exactly as on a file.

## Relationships and graphs

Relationships are shared by both endpoints. Addition, removal or rebinding
revises both objects (once for a self-link). Relationship versions pin both
endpoints' object versions (`a_version`/`b_version`) and geometry. Resolve these
pins rather than substituting current heads.
`relationships_at_versions(refs)` (C++ `relationshipsAt(txn, versions)`) reports
`applicable` only when both selected endpoints share the same version and its
geometry, metadata and lifecycle dependencies match. Missing endpoints are
inapplicable.

Anchors declare point IDs/indices, `channels`, `metadata_keys` (top-level keys),
or `whole_geometry`. Editing a dependency requires an explicit decision:

| Decision | Effect |
| --- | --- |
| `keep_recorded_relationship` / `action="keep_recorded"` | Validate recorded identities and preserve historical pins; may be inapplicable now |
| `retain_relationship` / `action="retain"` | Revalidate against resulting endpoints and publish shared new pins |
| `update_edge` / `action="update"` | Publish caller-supplied anchors after validation |
| `remove_edge` / `action="remove"` | Remove the current relationship |

Geometry `edge_edits` specify `edge_uuid`, `expected_generation`, `action`,
`a_revision` and `b_revision` (geometry UUIDs or `None`); updates also supply
`a_anchor` and `b_anchor`. Include expected object revisions for all validation
endpoints in checked writes. Metadata/lifecycle decisions must match final state
at commit. Unrelated metadata/point edits leave relationships applicable; physical
or domain-specific validity remains the caller's responsibility.

**Attachments are not graph links.** Edge kind `attachment` ties a data object
to the object it describes (VC3D growth corrections, Lasagna fit results on a
surface). `neighbours`, `component`, `reachable_within`, `connected_to` and
`graph_read` skip attachments unless the `attachment` kind is requested;
`edges_of`, `select_edges` and exports list them. Such point collections carry
a record meta `role`; `link_point_collections_to_surfaces` never links them and
rejects them when named.

**Same-winding queries must pass `adjacent=False`.** On `select_edges`,
`neighbours`, `component`, `reachable_within` and `graph_read`, it selects ordinary
`fiber_branch` edges, including branches without the flag. `True` selects
adjacent branches; `None` does not filter. Explicit filters exclude other edge
kinds and malformed flags before traversal, including parallel edges.
`applicable_only=True` additionally checks current shared membership/dependencies;
`include_retired=True` does not override applicability. Edge pagination follows
applicability filtering. Fiber imports reject contradictory ordinary/adjacent
classifications, including reciprocal declarations, atomically.

```python
with store.graph_read(versions=refs, adjacent=False, approval_states=[]) as graph:
    component = graph.component(seed)
    records = graph.objects()
    links = graph.edges()
    packed = graph.read_geometry_arrays("points", nodes=[0, 1])
```

`graph_read` defaults to approved edges; `approval_states=[]` includes all states.
Graphs of explicit numbered versions (`versions=`, C++ `graphReadViewAt(txn,
versions)`) always require applicability, retain isolated selected nodes, and
cannot combine `versions` with current object filters or `include_retired`. Nodes are in canonical UUID order; traversal is breadth-first
in edge-UUID order. Current queries otherwise inspect current records.
Applicability queries require a read transaction.
Graph views are thread-confined and must close before checkpointing. NumPy arrays
are read-only and retain their buffers after closure, not the database transaction.

## Retirement and restoration

`retired` and `lifecycle_epoch` are library-managed revision fields. Use
`tx.retire_object(uuid)` or `tx.restore_object(uuid, source_version=None)` in a
checked write, with explicit decisions for every surviving incident edge after
endpoint edits. Retirement retains geometry, metadata and history. Restoration
uses the last published active content, or an explicit active version of that
object (`source_version`, optionally `source_generation`; the typed
`restore_object` operation takes a numbered `source`); it advances generation and resets approval while preserving creation
stamps. Already-retired retirement and already-active default restoration are
no-ops. Restoration does not recreate removed links or physically removed rows.

Lifecycle epochs prevent preserved links from silently reactivating after
restoration, even with identical geometry. Revalidate explicitly. Default current
catalogs, spatial queries, traversal and exports omit retired objects and dormant
edges; traversal cannot pass through retired nodes. `include_retired=True`
exposes them for inspection without asserting applicability. Direct UUID lookups,
geometry and historical reads remain available. Lifecycle-only changes reuse
payloads; restoring different geometry can rebuild spatial indexes.

Application references should be ordinary `declared_reference` edges with
explicit dependencies. Endpoint order is canonical, so put directional roles in
metadata. Arbitrary UUID strings in JSON are not tracked references.
`delete_object` retains history and still rejects managed geometry/current edge
references. There is no history/payload purge API.

## Volume identity and coordinate scale

Geometry objects use `meta.coordinates = {"volume_uuid": ..., "scale_factor": ...}`.
Despite the field name, `volume_uuid` is the volume's `id` from the open-data
`metadata.json`, stored verbatim as a nonempty string (for example
`"20260411134726"`). It is not hashed, lowercased, or converted to an object UUID.
Paths, levels, resolution and labels never infer it. Scale is a finite
positive JSON number with direction **stored XYZ = volume voxel XYZ × scale**;
it differs from physical resolution and surface grid scale. Values are metadata
only and never rescale payloads or detector parameters. Unknown/missing members
mean `null`, not scale 1. Setting coordinates replaces the whole field and
rejects unknown keys; positive 64-bit integers stay exact.

Native object/document/collection declarations, flat `volume_uuid`/`scale_factor`
fields and `ingest_geometry_source(..., coordinate_context={...})` must agree
where known. Packed tracks read declarations from `metadata.json` as well as
explicit context. Every native exporter carries authoritative `meta.coordinates`,
including unknown values, rather than resurrecting declarations in `meta.source`.
An explicit `coordinate_binding` travels alongside them. Export→import preserves
the frame without an out-of-band context argument. Set coordinates with `set_meta`;
dependent anchors declare `metadata_keys=["coordinates"]`.

C++ consumers share normalization, conflict checking, transfer metadata and exact
volume matching through `objects/GeometryCoordinates.hpp`. A display name or
matching grid dimensions cannot override a mismatched volume ID. Existing
bindings containing a derived UUID require an explicit metadata correction;
they are not silently treated as aliases of catalog IDs.

`ingestGeometryBatch` takes decoded objects, geometry and source annotations and
inserts them and their native relationships in the caller's transaction. Duplicate,
existing and retired identities are errors (`invalid_argument`, as are
inconsistent source relationships); this is not a refresh operation. The checked
form is `IngestGeometryOp`: `addIngestOperation(request, reader, batch,
coordinates)` adds it with the request's expectations (every new object absent,
every existing fiber a branch links to at its current version; point and winding
links must resolve within the batch), so ingestion works on hosted attachments
too. `ingestGeometrySource` (direct or through a session) submits one such
request per surface or track, or one for a fiber or point-collection source.
`updateGeometryArrays` accepts desired typed arrays, the expected geometry revision,
optional transformed annotations, and a caller-supplied relationship decision
callback. It uses sparse coordinate edits and stable-ID sequence operations for
fibers/PCLs/tracks, and sparse channel edits for unchanged surface grids. Surface
arrays are compared against the stored revision in 1 MiB windows, so surfaces of
any size update without reading the whole revision. Omitted
annotations follow stable IDs; resizing an annotated sequence without IDs requires
explicit transformed annotations. Grid/channel schema changes retain their dedicated
surface APIs. These helpers do not introduce another history or editing model.

## Arrays, surface channels and sparse edits

Generic readers support full arrays, byte ranges, surface rectangles and packed
homogeneous arrays. Packed results expose dtype, shape, data and row offsets;
trailing dimensions/dtypes must match and request order/duplicates are preserved.
Storage dtypes are `|u1`, `<u2`, `<i4`, `<i8`, `<u8`, `<f4`, `<f8`.

Surface `points` is float32 XYZ. Extra arrays are aligned channels of shape
`[rows,columns]` or `[rows,columns,components]` with positive component count.
A channel's rows and columns are each a whole multiple of the points grid
(1 for a per-vertex channel; for example 20 for ink rendered at 20 pixels per
grid step). The shapes imply the multiple. Rectangle reads and edits of a
channel use that channel's own pixel coordinates.
Absent and zero-filled channels differ.

`valid` is the per-vertex validity channel: `|u1` at grid resolution, nonzero
means valid (ingestion writes 255). A vertex is valid when its coordinates are
not the `(-1,-1,-1)` sentinel and `valid` is nonzero; without the channel every
vertex with coordinates is valid. Readers apply it (typed surfaces, surface
views, relationship discovery); spatial bounds still cover every vertex with
coordinates. `mask`, `x`, `y` and `z` are not channel names.

Ingestion reads `mask.tif` into `valid`: page 0, first sample, which may be a
whole multiple of the grid; a vertex stays valid only when every mask pixel it
covers is at least 255. Later pages (VC's render page) are ignored, and a mask
that is not a whole multiple of the grid is rejected. Ingestion recognizes other lowercase `.tif`
and `.tiff` sidecars by stem as channels, excluding XYZ; duplicate/reserved names, unaligned,
truncated or unsupported layouts fail before publication. Tiled/stripped and
contiguous/separate sample planes preserve TIFF sample order and exact values;
no color conversion, rescaling, NaN normalization or alpha premultiplication.
Multipage/volume/non-top-left TIFFs are rejected as channels. Native export
writes each channel back as `<name>.tif` with the same samples and `valid` as a
single-page `mask.tif` (255/0).

Surface overlaps are relationships made by the linking tools (`surface-overlap`
detector).

`update_surface_channels(uuid, channels, remove, expected_version, edge_edits=[])`
adds/replaces/removes channels atomically. Supply only changed arrays and exact
little-endian bytes. Removing absent channels, duplicate names or spatial
dimensions that are not a whole multiple of the grid fails; dtype/components may change. Use
[structural resizing](structural-editing.md#surface-growth-and-cropping) for shape
changes. Typed owned surfaces carry `validity` and `channels`.

```python
import struct

version = store.current_version(uid)
edits = [{"array": "points", "element_offset": 15,
          "data": struct.pack("<3f", 12, 24, 36)}]
affected = store.affected_geometry_edit_edges(uid, edits, version)
assert not affected  # otherwise supply explicit edge_edits
revision = store.edit_geometry(uid, edits, version)
raw = store.read_geometry_rectangle(revision, "points", 0, 0, 1, 1)
```

Sparse offsets count **scalars**, not bytes or points. Edits preserve shape,
dtype, names and attributes; overlaps apply in input order, last write wins.
Geometry mutations (`put_geometry(expected_version=)`, `edit_geometry`,
`edit_geometry_rectangle`, `update_surface_channels`, and the structural edits)
expect the object's numbered version; another version is a conflict
(`RuntimeError`). They return the new geometry revision, the payload identity
the geometry reads take. C++ `geometryAtVersion` resolves that expectation.
Net no-ops create no revision/journal entry, but stale expectations still fail.
Rectangle edits preserve all components and require exact byte lengths.
Geometry, bounds, approvals and relationship decisions publish atomically.

Surface tiles support 32/64/128 (default 128); size 0 uses linear 1 MiB
chunks. Sparse deltas retain exact bytes, including NaN payloads/signed zero,
with checksum verification and a 256 MiB decoded limit. Current heads use a
chunk overlay; historical reads reconstruct from snapshots or reverse a shorter
current suffix. Inherited channels reuse immutable array references. Only changed
payloads are rewritten, but manifest/bounds bookkeeping can scale with chunk
count and historical reads with history length. There are no automatic periodic geometry
checkpoints or pruning. Long-held SQLite readers can retain old WAL pages.
Structural changes publish chunk snapshots and restart the delta chain; subsequent
coordinate edits continue using deltas. Structural operations can read and assemble
the full geometry. Full `put_geometry` replacements have a separate contract.

## Bounded oriented region browsing

`geometry_region(RegionSlice, RegionOptions, cancel=None)` combines conservative
spatial bounds with exact double-precision segment/slab intersections. It returns
original unclipped XYZ and zero-based segment indices in contiguous runs (for
polyline kinds); the consumer handles rendering/trimming. `geometry/SegmentClip.hpp` shares the clipper.

```python
from vc_object_store import RegionSlice, RegionOptions

view = RegionSlice()
view.origin = [100, 200, 300]
view.u, view.v, view.normal = [1, 0, 0], [0, 1, 0], [0, 0, 1]
view.local_bounds = [-50, -50, -.5, 50, 50, .5]
options = RegionOptions()
options.kind = "fiber"
options.maximum_objects = 100
result = store.geometry_region(view, options)
```

Axes must be finite/orthonormal (tolerance `1e-12`); bounds are finite and ordered,
with zero thickness allowed. Segments can intersect with both endpoints outside.
`options.kind` selects one polyline kind: `fiber` (nonempty `line_points` take
precedence over controls) or `track` (`points`, widened exactly to float64), or
`surface`. Other kinds are rejected. `min_points` defaults to 2.

A surface returns the cut of its exact triangle mesh by the slab's mid-plane
(normal coordinate halfway between the slab's normal bounds), clipped to the
slice's u/v extent. Cells whose four vertices are valid (coordinates and the
`valid` channel) split along the diagonal into `(r,c),(r,c+1),(r+1,c)` and
`(r,c+1),(r+1,c+1),(r+1,c)`. A vertex on the plane counts as on its positive
side. Crossings are computed once per grid edge in double precision, so runs
are connected polylines through tile boundaries; invalid cells and the u/v
extent split them into separate runs. Only the spatial tiles that touch the slab
are read. Run points are the computed crossings (not stored vertices),
`array_name` is `points` and `first_segment` is the run's offset in its chain.
Surfaces have no family or length summary, so those filters are rejected.
Family/minimum length filter import annotations, not recomputed geometry.
Retired/external-only objects are excluded. UUID ordering is stable; `selected`
is attempted first when limited. Only intersecting objects consume the limit,
and each chosen object finishes unless a budget/cancellation stops it.
`maximum_objects=0` means unlimited and ignores `selected`. Runs merge across
storage boundaries without duplicate segments. Results own their data and carry
revision identity from one snapshot; each Python call starts a fresh snapshot.
`options.uuids` optionally restricts the query before selected-object priority and
limits. `None` is unrestricted; an explicit empty list matches nothing. Filter
setup participates in cancellation and work/processing budgets.

C++ `ObjectSelection::coordinates` and `RegionOptions::coordinates` accept
an exact JSON coordinate declaration (missing fields compare as null).

A volume target instead queries in one volume's voxels across every frame
declared in it. `ObjectSelection::volume` / `select_objects(volume=...)`,
`spatial_regions(volume=...)` and `RegionOptions::volume` take the volume ID
(the `volume_uuid` of `meta.coordinates`) and match objects declared in it with a
scale factor. Their bbox, minimum length, `length_desc` order and length cursor
use volume voxels, where volume length = stored length × (1 / scale_factor), the
factor `geometryScaleToVolume()` returns. Each frame is queried on its own index
and merged; equal volume lengths order by stored length, then UUID, so a cursor
of the last row's volume length and UUID resumes exactly. Region run points and
spatial region bounds are returned in volume voxels; records keep their stored
coordinates. `include_unbound` appends objects not bound to the volume (including
other volumes' frames) in UUID order after the bound ones, unless a positive
minimum length is set; a cursor into them carries `after_uuid` without
`after_length`. `volume` and `coordinates` are exclusive. Region queries drive the persisted R-tree and order only
spatial candidates by UUID. Geometry and intersection predicates are unchanged.

Volume targets never parse every object's metadata. The `objects` table carries
`coordinate_volume` and `coordinate_scale`, virtual generated columns equal to
`json_extract` of `meta.coordinates`, indexed as `objects_coordinates(kind,
coordinate_volume, coordinate_scale)` (schema version 17; earlier stores are
rejected and must be rebuilt). Version 16 also records object/edge kinds in the
journal, including deletions, for selective cache invalidation. A target finds its frames with one index seek per
distinct scale, and the unbound tail reads only the index ranges outside the
volume's binding. `unboundObjectCount(txn, kind, volume)` counts that tail the
same way. SQLite never answers a generated column from an index alone, so an
aggregate over these columns (a GROUP BY or count of a whole kind) still reads
every row; new queries should seek index ranges instead.

| Budget | Default | Charged work |
| --- | ---: | --- |
| `output_byte_budget` | 8 MiB | 24 bytes per appended XYZ; a new run needs 48 |
| `processing_byte_budget` | 64 MiB | Payload, verification, metadata, reconstruction and cache bytes |
| `candidate_work_budget` | 1,000,000 | Candidate/block visits and segment tests |
| `sql_step_budget` | 10,000,000 | SQLite VM instructions, including rejected probes |
| `max_unit_bytes` | 1 MiB | Largest admitted processing operation |

Zero budgets allow no work. Work is reserved before processing; whole cached
chunks are charged on every touch and verification/reconstruction separately.
Counters measure work, not unique bytes, disk traffic or memory. `stop` is
`exhausted`, `object_limit`, `output_budget`, `processing_budget`, `candidate_budget`,
`sql_budget` or `cancelled`; only `exhausted` means `complete`. Partial runs remain
valid but imply no spatial coverage. Invalid input/corruption raises.

`RegionCancellation.cancel()` may run from another Python thread; queries
release the GIL. C++ callbacks inspect stats, must not re-enter the store, and
propagate exceptions. Checks occur at SQL steps, candidate visits, admitted
units and segment tests. Hash/JSON/decompression units are indivisible; neither
SQLite instructions nor OS waits have wall-clock bounds. Budgets guarantee
neither latency nor allocator memory. There is no continuation/coverage cache.

## Tests

For an existing core build tree configured with `VC_OBJECT_STORE_BUILD_TESTS=ON`:

```sh
AGENTS_AGENT_MODE=1 ninja -C build-core
AGENTS_AGENT_MODE=1 ctest --test-dir build-core --output-on-failure
```

The C++ tests cover shared-session lifecycle, direct fiber API transactions,
bounded UUID-filtered spatial reads, numbered history (`object_store_versions`),
schema conversion (`object_store_convert`) and local/hosted parity through
`vc_object_server` (`object_store_hosted`, built with `VC_OBJECT_STORE_BUILD_SERVER`). To run the Python tests with an existing
Python-enabled build, use its Python interpreter and point to its module directory:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python -m pytest tests/python -q
```

`tests/python/test_hosted_client.py` runs only when `VC_OBJECT_SERVER` names a
`vc_object_server` binary. In a VC build tree, configure with
`-DVC_OBJECT_STORE_PYTHON=ON` and build the `vc_object_bindings` target; the
module directory is then `<build>/object_store/python`.

VC3D and CLI imports share `prepareGeometrySource` and the checked batch ingest.
Fibers accept current version-4 documents, VC3D version-1 and version-3 fiber
files and version-1 fiber collections. Version-4 branch endpoints require
`target_uuid`, and `branch_file` only locates a missing peer's transport file;
version-3 links resolve by `branch_file`. No older-format repair or database migration is performed.
Native export's `--version N [--generation G]` / Python `version=` (and
`generation=`) selects a recorded object version, including its metadata and
relationship snapshot, and returns the exported numbered version.

Native format clients link `vc_object_store::vc_object_geometry`. Set
`VC_OBJECT_STORE_BUILD_ADAPTERS=ON` and `VC_OBJECT_STORE_BUILD_LINKING=OFF`
for import/export without relationship detectors or their CT reader dependency.
VC3D uses this configuration; the standalone tools/Python build keeps detectors
on by default. The existing OpenCV/TIFF dependencies supply native formats.
