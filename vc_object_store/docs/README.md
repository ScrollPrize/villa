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
existing databases, checked writes, worker cancellation, and journal events.
Consumers call the existing `ObjectStore` APIs directly inside its callbacks:

```cpp
vc::objects::ObjectStoreSession session;
vc::objects::OpenOptions options;
options.author = "reviewer";
const auto attachment = session.attach("objects.vcobj", options);
vc::objects::RevisionSelection observed;
session.read(attachment, [&](const auto& store, const auto& txn, const auto& cancel) {
    if (cancel.load()) return;
    observed[objectUuid] = store.currentObjectRevision(txn, objectUuid)->uuid;
});
session.act(attachment, observed, [&](auto& store, auto& txn) {
    store.setMeta(txn, objectUuid, "review_note", R"("checked")");
});
```

Keep all work using a snapshot inside its callback and do not reenter the
session from a callback. A session holds any number of attachments (for example a
writable store beside read-only catalog stores), each with its own access mode,
writer, events and hub sync; a store is attached at most once. `attach()` adds one,
`replace(attachment, ...)` swaps one for another store, `detach(attachment)` removes
one, and `attachments()` lists them in attach order. Replacement validates the new
database first, drains the old attachment's accepted writes, cancels/waits for
its readers and delivers its last commits; events report `Detached` for the old
attachment before `Attached` for the new one. Include the attachment generation in
asynchronous requests and caches: even reattaching the same store invalidates old
requests. `takeEvents()` polls local and external commits of
every attachment for all object kinds, tagged by attachment; call it from the
application's event loop. Each change carries
its record's object or edge `kind` (kept for deletions too), so a consumer
filters by kind without looking the record up. `editKeepingRelationships(attachment, pins, edit)` is the checked edit for
tools and editors that change existing objects without deciding about each
relationship: it expects the pinned object revisions and every endpoint of
their relationships, holds the objects' leases, and gives the callback
`keep_recorded` decisions to pass to geometry edits (affected relationships keep
their historical pins). `copyObject(from, to, uuid)` copies an
object's current record and geometry from one attachment into another as a new
object (new UUID, `meta.copied_from` naming the source store, object and
revisions; point annotations kept); relationships, history and approval stay
behind. `artifactDirectory(attachment, uuid)` returns (and creates)
`<store>.artifacts/<object uuid>/` for an object's non-geometry files such as
fitted models or mesh exports; a hosted replica uses the directory beside its hub
so replicas and services share it (an `sftp://` hub has none). `finalize()` waits
for accepted writes on every attachment; edits are already committed by `act()`.

Fiber consumers use the same `selectObjects`, `geometryRegion`, geometry, and
relationship APIs. Update any catalog summaries used by the application in the
same checked transaction as the geometry edit, and explicitly decide how affected
relationships apply. There is no separate fiber repository or collection layer.

For native fiber transport, `fiberNetworkJson(store, txn, selection, roots)`
exports pinned geometry together with its applicable `fiber_branch` edges. Supply
explicit object revision pins for every endpoint, including retired endpoints
needed to resolve applicability. Empty roots exports all active selected fibers;
otherwise the exporter includes their complete linked components. The result is
keyed by object UUID, uses `UUID.json` references, and keeps display names separate.
It resolves stable control-point anchors to positional wire indices and emits
reciprocal ordinary/adjacent branches with approval state and edge UUIDs. Coordinate
identity and bindings accompany each object. `fiberRevisionJson` remains the
geometry/annotation-only codec for consumers that read relationships separately.

Object revisions capture metadata, geometry identity and relationship membership.
`current_object_revision`, `object_revision`, and `object_history` expose them;
`current_geometry`, `geometry_revision`, and `geometry_history` expose geometry.
Their UUIDs are distinct: metadata edits need not create geometry revisions.

Pin all modified objects and participating relationship endpoints when preparing
an edit outside the writer lock:

```python
observed = {u: store.current_object_revision(u)["uuid"] for u in (a, b)}
with store.write(expected_revisions=observed) as tx:
    tx.set_meta(a, "review_note", "checked")
    tx.set_meta(edge, "confidence", 0.9)
```

C++ uses `store.write(RevisionSelection{...})`. `None` means the object must not
exist; preselect its UUID. Stale/missing expectations and caught geometry
revision conflicts abort the entire checked action. Plain `write()` serializes
read/modify/write under its lock but cannot protect an earlier external read.
Standalone Python mutations commit their own transaction. Read transactions pin
a snapshot; writer revision reads describe the last published state. Each commit
publishes at most one revision per changed object, including shared endpoints.

`published_at_us` is UTC Unix microseconds, shared by revisions in one action and
strictly increasing between actions even if the clock goes backward.
`revision_selection_at(time)` selects the latest revisions at that time;
`revision_read_view_at(time)` returns their `objects`, `relationships`, and resolved
`selection` together. Objects created later or physically deleted by then are
excluded. Retired objects are omitted unless `include_retired=True`.
`revision_read_view({uuid: revision_or_None})` uses an explicit closed selection;
`None` resolves the latest head in that snapshot. Restore historical content by
publishing a new checked action, not by moving old heads.

## Relationships and graphs

Relationships are shared by both endpoints. Addition, removal or rebinding
revises both objects (once for a self-link). `recorded_relationships(object_revision)`
returns immutable edge versions with `a_revision`/`b_revision` object pins and
geometry pins. Resolve these pins rather than substituting current heads.
`relationships_at(selection)` reports `applicable` only when both selected
endpoints share the same version and its geometry, metadata and lifecycle
dependencies match. Missing endpoints are inapplicable.

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
with store.graph_read(revisions=selection, adjacent=False, approval_states=[]) as graph:
    component = graph.component(seed)
    records = graph.objects()
    links = graph.edges()
    packed = graph.read_geometry_arrays("points", nodes=[0, 1])
```

`graph_read` defaults to approved edges; `approval_states=[]` includes all states.
Explicit revision graphs always require applicability, retain isolated selected
nodes, and cannot combine `revisions` with current object filters or
`include_retired`. Nodes are in canonical UUID order; traversal is breadth-first
in edge-UUID order. Current queries otherwise inspect current records.
Applicability queries require a read transaction.
Graph views are thread-confined and must close before checkpointing. NumPy arrays
are read-only and retain their buffers after closure, not the database transaction.

## Retirement, restoration and synchronization

`retired` and `lifecycle_epoch` are library-managed revision fields. Use
`tx.retire_object(uuid)` or `tx.restore_object(uuid, source_revision=None)` in a
checked write, with explicit decisions for every surviving incident edge after
endpoint edits. Retirement retains geometry, metadata and history. Restoration
uses the last published active content, or an explicit active revision of that
object; it advances generation and resets approval while preserving creation
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

Establish a shared baseline with `synchronize(peer)` before independent edits.
Conflicts require a fresh `dry_run=True` preview and explicit `local`/`remote`
resolutions with its `expected_plan` token. Handles must be idle and writable;
call `fork_replica()` on a copied database before synchronizing it with its source.
Sync transfers current records, lifecycle bindings and retained geometry history,
not the peer's complete metadata/graph timeline. Use a database copy for that.
Missing historical pins receive incomplete, untimed baselines, never fabricated
history. A fresh retired replica may lack an active revision to restore: restore
on the originating replica and sync, or copy its full database. Metadata-only
JSON import rejects lifecycle-bearing exports.

## Hosted hub and replicas

A team shares one store by keeping a **hub** `.vcobj` on a network filesystem
(an sshfs mount, NFS or SMB) or on any server reachable over SSH, addressed as
`sftp://[user@]host[:port]/path/hub.vcobj` (`ssh://` works too). Nobody opens the hub to work. Each client clones
it into a local **replica** and opens that replica like any other store, so reads
and viewport queries stay on local disk. Hosting needs no server process. Public
signatures are in [Hub.hpp](../include/vc_object_store/objects/Hub.hpp).

```sh
vc_object_hub init objects.vcobj        # locally, with no other connections
cp objects.vcobj /mnt/share/hub.vcobj   # then place it on the share
vc_object_hub clone /mnt/share/hub.vcobj ~/objects-replica.vcobj
vc_object_hub lease acquire ~/objects-replica.vcobj UUID
#   ... edit UUID with any client of the replica ...
vc_object_hub sync ~/objects-replica.vcobj
# or, with no mount at all:
vc_object_hub clone sftp://alice@data.example.org/share/hub.vcobj ~/objects-replica.vcobj
```

**SFTP addresses.** Hub files addressed as `sftp://` or `ssh://` open through
the store's own SQLite VFS, which speaks SFTP v3 to the system OpenSSH `ssh`
program (`ssh -s host sftp`), so `~/.ssh/config`, keys, the agent, ProxyJump and
`known_hosts` apply as usual and nothing needs mounting. The protocol client is
the `vc_object_store::vc_sftp` library (`include/vc_object_store/sftp/Sftp.hpp`),
which VC3D also uses for remote volumes. It locks with the same `<hub>.lock`
directory as the `unix-dotfile` VFS, so clients on a mount and clients using an
address can share one hub. The server must offer OpenSSH's `fsync@openssh.com`
extension (stock OpenSSH does), or writes could not be made durable; the VFS
refuses to open a hub for writing without it.
- Without a registered authenticator, `ssh` runs in batch mode: keys and the
  agent only, and only hosts already in `known_hosts`.
- With one (`vc::sftp::setAuthenticator`, Python `set_sftp_authenticator`), every
  ssh prompt (password, key passphrase, one-time code, accepting a new host key)
  goes to the application through `SSH_ASKPASS`, with ssh's own wording. The
  helper relays prompts through FIFOs in a private 0700 directory, so secrets
  never appear in arguments or the environment. An accepted answer is reused
  when a later connection to the same destination asks the same question; a
  question repeated within one connection means the answer was wrong, so the
  user is asked again. VC3D shows dialogs, `vc_object_hub` asks on the terminal
  (Python: `use_terminal_sftp_prompts()`). Prompts need Linux or macOS; on
  Windows `ssh` always runs in batch mode.
- One ssh connection per destination serves every hub file and helper in a
  process. A broken connection reconnects on next use, unless writes were still
  unacknowledged; then the operation fails and SQLite rolls back from its journal.
- Writes are pipelined and collected before any read, sync or unlock, so a
  commit costs about one round trip plus the bytes. Each page read is still one
  round trip, as over sshfs.

The hub keeps SQLite's rollback journal and opens through the `unix-dotfile`
VFS (`OpenOptions::sharedFile`). That VFS locks by creating `<hub>.lock` with an
atomic `mkdir`, which works on network filesystems where WAL's shared memory and
byte-range locks do not. Only hub operations open the file, and each holds the
lock briefly. An ordinary open of a hub is refused, so nothing can switch it back
to WAL. Stores without the `hub_*` tables behave exactly as before. That includes
two-store `synchronize`, which hub sync does not use.

**Leases.** Changing the record or geometry of an object that exists on the hub
requires this replica to hold that object's lease. The check runs at commit, in
the library, so it covers every client of the replica. Objects created locally
need no lease until they are pushed, and a push grants their creator a lease.
`acquire_leases` is all-or-nothing. It refuses objects another replica holds,
and objects changed on the hub since this replica last pulled them
(`HubLeaseRefused::held` / `stale`); pull, then retry. Leases expire after their
duration (default ten minutes), so clients need roughly synchronized clocks.
Releasing a lease requires its changes to be pushed first. The hub enforces
unexpired leases on direct edits too.

Relationships are shared records, not leased. Linking to someone else's object,
or saving an edit that revalidates its links, does not need the other object's
lease. Two replicas changing the same relationship produce a conflict.

**Sync.** A replica tracks the hub's journal cursor, the hub version (latest
journal sequence) of each record it installed, and its own unpushed changes.
- `hubPull` installs records changed on the hub since the cursor, with their
  geometry history, in one replica transaction.
- `hubPush` publishes local changes in one hub transaction. Each record must
  still have the hub version it was edited from, and each changed hub object must
  be leased to this replica. The push copies geometry history, validates
  relationship dependencies like `synchronize`, and verifies the result.
- A retry after a push whose outcome was lost recognizes its own records.

**Applications** do not drive sync themselves. When an `ObjectStoreSession`
attaches a writable hosted replica, it starts a background worker. The worker
pulls and pushes every `HubSessionOptions::syncInterval`, and pushes shortly
after each `act()`; pulled changes arrive through `takeEvents()` like any other
external commit. Before editing, a caller takes
`holdLeases(attachment, uuids)`. The returned `LeaseHold` keeps the leases
renewed while it lives. Once no hold remains and the changes are pushed, the
worker releases the lease. A hold pulls once first if the hub is ahead, and it
is a no-op for ordinary stores, so callers need not check what they attached.
`hubStatus(attachment)` reports pending records, conflicts, the last error and
sync time for that replica, and a `StoreSessionEvent::Kind::Hub` event tagged with
the attachment announces each change. `syncHub(attachment)` and
`resolveHubConflicts(attachment, keep)` act asynchronously. Scripts that
do not use a session call `hub_sync` and the lease functions directly.

Python services use the same session: `ObjectStoreSession().attach(replica,
author)` starts background sync, `hold_leases(attachment, uuids)` is a context
manager whose leases renew until it exits, and `hub_status`, `sync_hub`,
`artifact_directory`, `copy_object` and `take_events` mirror the C++ calls. Data
is read and edited through `ObjectStore.open(replica)` connections to the same
file; the session pushes their commits.

```python
import vc_object_store as vos
vos.hub_clone(hub, replica, "lasagna")            # once per machine
session = vos.ObjectStoreSession()
attachment = session.attach(replica, "lasagna")
with session.hold_leases(attachment, [surface_uuid]):
    with vos.ObjectStore.open(replica, author="lasagna") as db:
        ...  # edit the surface, add objects
    model_dir = session.artifact_directory(attachment, surface_uuid)
```

Records changed both locally and on the hub become conflicts, and pushing stops
until they are resolved. `resolveHubConflict(uuid, "hub")` installs the hub's
record. `"local"` publishes this replica's record over it and requires the lease.
Object revision UUIDs are per replica, as with `synchronize`. Geometry revision
UUIDs and the store UUID are shared.

**Operations.**
- `clone` copies the hub file in large reads without holding the lock. It then
  checks, under the lock, that the header's change counter and the file size are
  unchanged; every commit changes the counter, so the copy is consistent. After
  three interrupted attempts it falls back to SQLite's backup API in locked steps.
  That fallback reads one page per request and is slow over a network.
- A client that dies while holding the hub lock leaves `<hub>.lock` behind, and
  every other client waits on it. Once no client is syncing, run
  `vc_object_hub unlock HUB --force` (a path or an address; library
  `hubLockAge`/`removeHubLock`); the next open rolls back the interrupted
  write from its journal.
- The share must apply `mkdir` atomically and must not serve stale cached data
  to SQLite. Mount NFS with `noac`. Mount sshfs (3.x) with
  `-o direct_io,dir_cache=no,entry_timeout=0,attr_timeout=0,negative_timeout=0`,
  so page reads and the existence of `<hub>-journal` always come from the
  server.
- A process keeps one connection per hub and author open. SQLite then keeps
  its page cache between operations while the hub is unchanged, instead of
  rereading the header and schema over the network each time.
- Hub operations wait up to 60 s for the lock. The lock is not fair, so heavy
  contention can still time out an operation; retry it.
- `detach` turns a replica back into an ordinary store.

Measured on 2026-10-23 with RelWithDebInfo on Ubuntu arm64, median of 10
iterations. "Local disk" uses a synthetic 20,000-fiber hub (354 MB). "Network"
uses 765 real fibers with 186 links (98 MB) on a remote host, mounted with the
sshfs options above at about 18 ms round-trip time.

| Operation | Local disk | Network |
| --- | ---: | ---: |
| Clone | 0.4 s | 3.8 s |
| Lease acquire | 13 ms | 460 ms |
| One-fiber push (metadata + geometry edit) | 19 ms | 520 ms |
| Pull of that change on another replica | 10 ms | 1,080 ms |
| Sync with nothing to do | 6 ms | 140 ms |

The first operation of a process also opens the hub, which takes about 2 s over
the network. A pull after any hub commit rereads the index pages it needs: in
rollback-journal mode, SQLite drops its page cache whenever the file changes.

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
existing and retired identities are errors; this is not a refresh operation.
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

Ingestion ignores tifxyz `overlapping.json`: surface overlaps are relationships
made by the linking tools (`surface-overlap` detector).

`update_surface_channels(uuid, channels, remove, expected_revision, edge_edits=[])`
adds/replaces/removes channels atomically. Supply only changed arrays and exact
little-endian bytes. Removing absent channels, duplicate names or spatial
dimensions that are not a whole multiple of the grid fails; dtype/components may change. Use
[structural resizing](structural-editing.md#surface-growth-and-cropping) for shape
changes. Typed owned surfaces carry `validity` and `channels`.

```python
import struct

current = store.current_geometry(uid)
edits = [{"array": "points", "element_offset": 15,
          "data": struct.pack("<3f", 12, 24, 36)}]
affected = store.affected_geometry_edit_edges(uid, edits, current["uuid"])
assert not affected  # otherwise supply explicit edge_edits
revision = store.edit_geometry(uid, edits, current["uuid"])
raw = store.read_geometry_rectangle(revision, "points", 0, 0, 1, 1)
```

Sparse offsets count **scalars**, not bytes or points. Edits preserve shape,
dtype, names and attributes; overlaps apply in input order, last write wins.
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
and bounded UUID-filtered spatial reads. To run the Python tests with an existing
Python-enabled build, use its Python interpreter and point to its module directory:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python -m pytest tests/python -q
```

VC3D and CLI imports share `prepareGeometrySource` and the checked batch ingest.
Fibers accept current version-4 documents and version-1 fiber collections;
branch endpoints require `target_uuid`. `branch_file` only locates a missing
peer's transport file. No older-format repair or database migration is performed.
Native export's `--revision` / Python `object_revision_uuid` pins an object
revision, including its metadata and relationship snapshot, and returns its
geometry revision UUID.

Native format clients link `vc_object_store::vc_object_geometry`. Set
`VC_OBJECT_STORE_BUILD_ADAPTERS=ON` and `VC_OBJECT_STORE_BUILD_LINKING=OFF`
for import/export without relationship detectors or their CT reader dependency.
VC3D uses this configuration; the standalone tools/Python build keeps detectors
on by default. The existing OpenCV/TIFF dependencies supply native formats.

## Large-store reads and volume-ID correction

`bench_volume_queries` times the volume-targeted reads VC3D issues (viewport
regions, a line-annotation strip's candidates, the unbound count, a catalog page
with its unbound tail, and a curved view's control-chain selection), five runs
each, read-only. Build and run it from this directory; the VC3D read-only
harness is documented in `docs/fiber-collections.md` in the sibling project:

```sh
AGENTS_AGENT_MODE=1 ninja -C build bench_volume_queries
build/bin/bench_volume_queries /path/to/paris4-fibers.vcobj 20260411134726
```

On the 931,104-fiber Paris 4 store (Ubuntu arm64, RelWithDebInfo, warm cache;
"before" is schema 13, "after" the same data with the schema 14 columns and index),
median milliseconds:

| Read | Before | After |
| --- | ---: | ---: |
| Region 300×200×6 | 810 | 32 |
| Region 1500×1000×6 | 826 | 52 |
| Strip candidates (`spatialRegions`) | 802 | 15 |
| Unbound count (was a per-binding count + `objectCount`) | 2,516 + 69 | 0.004 |
| Catalog page of 200 with unbound tail | 3,148 | 1.2 |
| Control-chain selection | 920 | 47 |

Query results were identical on both schemas (checked with
`test_object_store_volume_target`).

Older imports may contain a UUIDv5-derived volume binding. Correcting a known
binding is explicit and preserves prior revisions and all geometry. First preview:

```sh
python scripts/rebind_volume.py objects.vcobj --from-id OLD_VALUE --to-id CATALOG_ID
```

After closing editors, add `--apply` with `PYTHONPATH=build/python` to run checked,
batched metadata transactions. The tool is resumable and refuses stores with
relationships, whose coordinate dependencies need explicit review. Unknown
bindings are not inferred. Verify with `PYTHONPATH=build/python python -m pytest
tests/python/test_rebind_volume.py -q` and `PYTHONPATH=build/python python
scripts/import_afv.py --self-test`.

The approved Paris 4 correction on 2026-10-22 updated 931,104 fibers and one
collection from `2d2a9b21-9c82-5343-9443-57838a5fa5a1` to `20260411134726`.
All current bindings were checked; every scale factor remained 1. The SHA-256
fingerprint of all 931,104 sorted geometry-head pairs was unchanged, as were
1,862,208 control points and 344,875,721 line points. Historical metadata was
checked for 20 fibers and the collection. The final read-only VC3D check took
3 ms to attach, 2,931 ms to show the first page, and 2 ms to load the selected
fiber, with 293 UI heartbeats and just one resident editor. This was one
offscreen Ubuntu arm64 run, not a cross-platform rendering benchmark.

Validation passed: five targeted C++ tests, all 30 Python tests, the AFV
importer self-test, and VC3D's `--test-object-store` controller regression.
A fresh import of 1,000 real AFV fibers was also checked for byte-exact line
geometry and the verbatim catalog volume ID.
