# Hosted stores and sessions

A team shares one store by serving it: `vc_object_server` owns the database
on the storage host, and clients use the **same** checked-request, history,
geometry, and query APIs over HTTP without opening the file. In C++, an
`ObjectStoreSession` holds any mix of local and hosted attachments behind one
interface — reads through a `StoreReader`, writes as typed requests — and
adds change events, surface drafts, and crash-durable saves. Python exposes
the same session, plus a direct `HostedStore` client.

## Serving a store

```sh
vc_object_server /data/project.vcobj --port 8750          # on the storage host
ssh -N -L 8750:127.0.0.1:8750 storage-host                # on each client
```

The server opens the store as its service owner (exclusive write lock): it
will not start while a standalone writer has the store open, and standalone
writable opens fail while it runs; read-only inspection still works. There
is **no authentication** — keep the default loopback listen address behind an
SSH tunnel, or listen (`--listen`) only on a trusted private network.
`--read-workers`, `--max-payload-mib`, `--max-staged-mib`, and `--staging-dir`
bound its queues and staged uploads; a full queue answers 503 with
`Retry-After`, which clients retry. A `link_objects` request
([graph edges](graph-edges.md#linking-as-a-checked-request-link_objects))
runs its detectors on a read worker and queues only their commit to the
writer; it answers when the detection is done, so `--request-deadline`
(default 300 s) and the client's link timeout (`HostedStoreOptions::linkTimeoutS`,
Python `link_timeout_s`, default 300 s) must both exceed the longest link,
and `--strip-cache DIR` keeps the fiber strips between such requests. A
retry of a link request in flight (same UUID) waits for the running
detection instead of starting another. `GET /v1/service` reports connection,
queue, staging, link-request, and WAL-checkpoint state. In VC3D: Stores → Attach Hosted
Store...

## Pasting an SSH address in VC3D

With `vc_object_server` running on the storage host, use **File → Stores →
Attach Hosted Store...** and paste `ssh://dl.ash2txt.org:8750` or
`sftp://dl.ash2txt.org:8750`. Both mean the database service listening on the
remote host's loopback port **8750**. The SSH login port comes from `~/.ssh/config`
(normally 22); an optional `user@` selects the login name. These service
addresses do not accept a `.vcobj` file path. Select the attached store under
**Show Fibers From**, **Show Surfaces From**, etc. as needed.

VC3D manages a local loopback-only tunnel, uses the existing SSH key/agent and
host-trust/password prompts, and saves the pasted address in the project.
Reopening the project reconnects; detaching releases the connection. After a
network failure that terminates SSH, reopen the project to reconnect. No
manual `ssh -N -L` command is needed. `HostedStore` and
`ObjectStoreSession::attachLocation` accept these addresses in C++ and Python too.

The transport regression can use a closed real store, copied into a disposable
directory. It starts a local sshd with temporary keys and configuration, tests
both schemes and a write through the service, and checks listener cleanup.
`--gui` also checks GUI-thread authentication and persisted-project reopening.
It requires installed OpenSSH `ssh`, `sshd`, and `ssh-keygen`; installs nothing:

```sh
cmake --build volume-cartographer/build --target VC3D vc_object_server_app test_object_store_ssh -j 4
python3 vc_object_store/tests/run_ssh_test.py --build volume-cartographer/build \
  --store /path/to/closed-current-schema.vcobj --gui
```

## Clients

Python `HostedStore(endpoint)` mirrors the local store's forms: requests are
canonical request JSON (dict or text) with payloads keyed by SHA-256
(`submit(request, payloads)`, `receipt`), queries are the wire form
(`{"selection": {...}, "fields": [...]}`), and numbered references are
`{store, generation, object, version}` dicts (`objects_at_versions`,
`relationships_at_versions`, `record_selection`). Selected rows carry
`version` and `version_generation`; `with_status=True` also returns the
server snapshot's status. `put_artifact`, `artifact_length`, `read_artifact`,
and `fetch_artifact(file, destination)` (verified, atomic local file) cover
[artifacts](writing.md#artifacts). Conflicts raise `RuntimeError`, malformed
requests `ValueError`, and `select_all_objects` / `select_all_edges` can
raise `StoreBusy` (see [reading](reading.md#listing-and-selecting-records)).

C++ clients use `vc::objects::HostedStore`
([client/HostedStore.hpp](../include/vc_object_store/client/HostedStore.hpp))
with the same types as the local API: `status`, `submit`, `receipt`,
`changesSince`, `currentObjectRevision`, `currentVersion`,
`objectAtVersion`, `objectsAtVersions`, `geometryRevision`,
`readGeometryArray` (read in ranges past the server's response limit),
`putArtifact`, `artifactLength`, `readArtifact`, `fetchArtifact`,
`selectObjects`/`selectEdges` (one page, `limit` required),
`selectAllObjects`/`selectAllEdges`, `objectHistory`, `relationshipsAt`,
`recordSelection`, `geometryRegion`, `nearestSurfaces`. Conflicts throw the
same `TransactionConflict`. Transport failures, 429, and 5xx are retried with
the **same request UUID**, so a lost reply is answered by the saved receipt;
when retries run out the outcome is unknown — recover it with `receipt()` or
by resubmitting the identical request, never under a new UUID. A
`link_objects` request is sent with `linkTimeoutS` (default 300 s, the
server's default `--request-deadline`) in place of `stallTimeoutS`, since
its reply waits for the detection; a retry of one joins the detection the
server is already running. If a request's
payloads have expired from server staging and no receipt answers it, the
server names the missing hashes; upload them and resubmit identically.

Most tools never touch the client classes directly: a store **location** is
a file path or an `http(s)://` endpoint, and the Python helpers
(`surfaces.read_surface`, `surfaces.create_surface`,
`documents.attach_document`, `locations.attached(location)`) work on either.
`locations.open_store(location, author=, read_only=, link_timeout_s=,
cache_directory=)` yields the `ObjectStore` itself for a file and a
`HostedStoreHandle` for an endpoint: the same reading and request methods
(status, submit and receipts, the change feed, current and numbered records,
relationships at versions, geometry descriptors, arrays and annotations,
object and edge selections, pinned overlap correspondences) over one session
attachment, with `cache_directory` as its persistent cache, so one code path
reads, links and writes a store wherever it is.

## Sessions: `ObjectStoreSession`

```cpp
vc::objects::ObjectStoreSession session;
vc::objects::OpenOptions options;
options.author = "reviewer";
const auto attachment = session.attach("objects.vcobj", options);
auto request = session.newRequest(attachment);
session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto& cancel) {
    if (cancel.load()) return;
    request.expectedObjects[objectUuid] = reader.currentObject(objectUuid)->reference();
});
request.operations.push_back(vc::objects::SetMetaOp{objectUuid, "review_note", R"("checked")"});
session.submit(attachment, request); // TransactionConflict when the object moved on
```

- A session holds any number of attachments (a project file beside a hosted
  team store), each with its own access mode, writer, and events; a store is
  attached at most once. `attach`, `attachHosted(endpoint, options)`,
  `attachLocation(path-or-endpoint)` add one; `replace(attachment, ...)` swaps
  one store for another (validating the new database first, draining the old
  one's accepted writes, cancelling its readers); `detach` removes one;
  `attachments()` lists them in attach order. Keep all work using a reader
  inside its callback, and do not reenter the session from a callback. Include
  the attachment **generation** in asynchronous requests and caches — even
  reattaching the same store invalidates old requests.
- Requests are already committed when `submit()` returns; `finalize()` waits
  for accepted writes on every attachment. The session also offers
  `editCurrent(attachment, uuids, operations)` (read current, build, submit),
  `copy_object` between attachments, ingestion into any writable attachment,
  and artifact publish/read per attachment (see
  [writing](writing.md#artifacts)).
- `GeometrySnapshot(reader)` builds the usual geometry views — transformed
  ones included — over a local read transaction or a hosted attachment. A
  local reader answers from one snapshot; a hosted reader answers per call
  from the attachment cache or the host (numbered versions and geometry
  revisions are exact; current heads may come from different commits). The
  reader and its views must not outlive the callback. Region and proximity
  queries stay session operations, combining attachments and drafts in
  attachment order.
- Catalog geometry (Open Data surfaces) is fetched on read by the fetcher
  registered for its meta key (`setGeometryFetcher`,
  [objects/CatalogStore.hpp](../include/vc_object_store/objects/CatalogStore.hpp)),
  in any writable attachment.
- Python mirrors this: `ObjectStoreSession` with `attach_hosted` (its
  `cache_directory`, `disk_cache_bytes` and `min_free_bytes` are the
  persistent cache below), `hosted`, `attach_location`, `location`,
  `status`, `submit`, `receipt`, `current_object`, `objects_at_versions`,
  `relationships_at_versions`, `changes_since`, `select_objects`,
  `select_edges`, `read_surface_grid`, `read_geometry_arrays`,
  `geometry_revision`, `read_geometry_array`, `geometry_point_annotations`,
  `overlap_correspondences_at`, `create_surface`, `attach_document`,
  `attached_document`, `point_collections`, `artifacts`,
  `publish_artifacts`, `materialize_artifacts`, and the draft calls below.

### Change events

`takeEvents()` polls local and external commits of every attachment — call it
from the application's event loop. Each event is tagged by attachment; each
change carries its record's object or edge `kind` (kept for deletions), so a
consumer filters by kind without lookups. For hosted attachments a background
check every `pollInterval` reads the store's status and, when the commit
sequence moved, the change feed, delivering `Committed` events whose
`journalPosition` is the commit sequence. When the feed position is lost
(another history generation, a restored store), the event sets `revalidate`:
reread whatever is shown. `refresh()` checks now; `freshness()` reports the
last successful check and last failure. The attachment caches descriptors,
arrays by revision, records at numbered versions, and current heads until the
feed reports them changed, all within `cacheBytes` — cached data stays
readable while the host is unreachable, but listings and spatial queries
always need the host.

With history enabled, every publishing transaction appends to one ordered feed:
`status()` (`storeUuid`, `generation`, `commitSequence`, `feedStart`, `historyEnabled`) and
`changes_since(cursor, max_changes)` return whole commits (continuing inside
a commit only when it alone exceeds the page). A cursor from another
generation, the future, or before `feedStart` comes back invalid — revalidate
cached heads and rerun queries.

History-off stores keep numbered current versions and the commit sequence, but not a replayable feed. Sessions detect a lineage/sequence change and emit `revalidate` without fabricated change entries; hosted snapshots validate that the sequence stayed fixed during their reads. Explicit historical requests remain unavailable. A lineage UUID and commit sequence are distinct from the session-local attachment generation.

## What differs on a hosted attachment

- **Transforms resolve on the host.** Selections never carry transforms:
  volume targets use the store's recorded registrations, so a client catalog
  cache cannot change a shared query; a request setting explicit transforms
  is refused. Record shared registrations with `recordCatalogTransforms` —
  see [transforms](transforms.md#registrations-and-route-choices).
- **Region queries** run on the host, are not cancellable, and have their
  budgets clamped to server limits; the answer reports the budget it stopped
  at instead of claiming completeness.
- **Listings** page by UUID cursor without holding a server snapshot;
  `selectAllObjects` catches up through the change feed so the result is
  exact as of one commit sequence, and throws `StoreBusy` when writes never
  go quiet (length-ordered and unbound-volume listings page by offset and
  restart on any commit).
- **Large arrays** are read in requested byte ranges; rectangular surface
  reads fetch only their row ranges rather than whole arrays.
- Session `selectObjects` with a limit gives each attachment up to `limit`
  records not hidden by an earlier attachment, reading further pages as
  needed — a short result means there are no more.

## Durable drafts

Session surface drafts (see [editing](editing.md#editor-drafts)) live in
memory. `useDraftDirectory(directory)` makes them survive a crash:

- `checkpointDrafts()` writes every draft (grid, coordinates, attachment
  store, and `base` — the committed version it edits) and every pending
  submission, each changed draft to a new file, then replaces
  `checkpoint.json` atomically and syncs it. A crash or failed write
  (`draftCheckpointError()`) leaves the previous checkpoint complete; the
  application schedules checkpoints, separately from its saves. Call
  `useDraftDirectory` once, before making drafts: it installs the last
  checkpoint's drafts and pending submissions (later edits are lost). A
  directory serves one session at a time.
- `submitDurably(attachment, request, drafts)` is a checked save that
  survives a crash: the exact request and payloads are checkpointed before
  sending. Success clears each listed draft that did not change after the
  call (a changed one stays, rebased onto the committed version). A conflict
  keeps the drafts and is recorded (`draftState().conflict`); nothing is
  resent — reapply by setting the draft again with the new `base`. An
  unreachable host leaves the submission pending
  (`pendingSubmissions()`, also after restart) until `resolvePending()` finds
  its receipt or resends the identical request.
- The `DurableSave` form names each draft with the revision whose grid the
  request holds (a save copied on one thread while editing continues on
  another) and `keepDrafts` keeps drafts open after success, moved to the
  committed version and marked saved (`markDraftSaved` does the same for a
  draft holding its base's committed geometry). Checkpoints skip saved
  drafts, so opening large surfaces writes nothing. A pending save recovered
  after a restart clears its drafts on success — the editor that kept them is
  gone. `draftContents()` returns a draft's grid and attachment for
  restoration.

## Wire protocol (version 1)

JSON unless noted; every query answer carries the `status` of the single read
snapshot it came from. Payloads are staged by hash (`PUT /v1/payloads/{sha256}`,
bytes, `Content-Length` required; 400 hash/length mismatch, 411 no length,
413 too large, 507 staging full) and a transaction names them by hash.

| Request | Answer |
| --- | --- |
| `GET /v1/status` | store, generation, commit sequence, feed start |
| `GET /v1/service` | connections, queue depths, staging use, WAL checkpoints |
| `POST /v1/transactions` (request JSON) | receipt; 409 conflict; 422 `missing` payload hashes; 400 invalid. A request with a `link_objects` operation is prepared on a read worker first |
| `GET /v1/receipts/{request}` | saved receipt or 404 |
| `GET /v1/changes?store&generation&sequence&offset&max` | feed page |
| `GET /v1/objects/{uuid}` | current revision + snapshot status |
| `GET /v1/objects/{uuid}/versions/{n}?store&generation` | that numbered revision or 404 |
| `GET /v1/objects/{uuid}/history` | every recorded revision, or 404 |
| `GET /v1/geometry/{revision}` | geometry descriptor (immutable) |
| `GET /v1/geometry/{revision}/object` | latest retained object metadata naming that geometry, or 404; not immutable, use an exact object reference to pin metadata |
| `GET /v1/geometry/{revision}/arrays/{name}?offset&count` | array bytes; 413 above the response limit |
| `GET /v1/geometry/{revision}/annotations/{name}` | per-point annotations (exact JSON, immutable) |
| `PUT /v1/artifacts/{sha256}` (bytes) | kept durably by the store |
| `GET /v1/artifacts/{sha256}/info`, `GET /v1/artifacts/{sha256}?offset&count` | `{sha256, length}` or 404; artifact bytes |
| `POST /v1/query/objects`, `/query/edges` (`selection`, `fields`) | one page; `limit` required, at most `maxQueryResults` |
| `POST /v1/query/versions` (`versions`) | one revision per reference, in order, `null` where not recorded |
| `POST /v1/query/relationships` (`versions`) | selected relationships with `applicable` |
| `POST /v1/query/graph` (`traversal`, `object`, `edge_kinds`, ...) | object UUIDs reached from `object` |
| `POST /v1/query/region` (`slice`, `options`) | region runs, stop reason, stats |
| `POST /v1/query/nearest_surfaces` | surface hits |
| `POST /v1/query/count`, `/query/tags` | object count of a kind; distinct sorted tags |

`ServerOptions` sets the queues, body/payload/staging/response limits,
staging lifetime and directory, per-request deadlines, WAL checkpoint
scheduling, query page size, and region budgets; staged uploads stream to
disk and reserve their declared length up front, so a slow disk slows the
sender instead of filling memory.

## Persistent hosted reads and object downloads

`HostedSessionOptions::cacheDirectory` opts an attachment into persistent caching.
`diskCacheBytes` and `minFreeBytes` use the same `vc_reader` disk budget as volume
caches; leaving the directory empty preserves memory-only behavior. VC3D supplies
its configured remote-cache directory and limits automatically.

Numbered records, geometry descriptors, positional annotations, selected
relationship revisions, geometry blocks, and attached-file blocks persist across
restarts. Store UUID and history generation identify revision entries; SSH forward
ports are never part of their identity. Geometry and artifacts use shared 1 MiB
blocks for range reads and downloads. Atomic envelopes include SHA-256 checksums;
materialized attached files are also checked against their manifests. Cache write
failures leave interactive network reads usable. Heads, spatial queries, and
freshness checks still require the service; this is not offline editing.

```cpp
vc::objects::HostedSessionOptions options;
options.cacheDirectory = cacheRoot;
options.diskCacheBytes = int64_t(20) << 30;
options.minFreeBytes = int64_t(2) << 30;
auto attachment = session.attachHosted(endpoint, options);
auto job = session.prefetchObjects(attachment); // returns immediately
const auto progress = job->snapshot();         // thread-safe copy
// job->cancel();
```

One sequential prefetch runs per attachment, yielding between bounded transfers to
interactive payload reads. It captures the current non-retired objects at numbered
versions, their associated relationships, and store-held payloads. It does not fetch
external CT datasets or materialize catalog placeholders. `captured` identifies the
selection's store generation and commit sequence. Progress reports discovery,
loading and coverage checking, object counts, payload bytes downloaded, errors,
and cancellation. `complete` means the required entries remained in the disk cache
when coverage was checked; later use of the shared budget can evict them.

Jobs belong to the session, independently of UI rows. Detaching cancels a job
without waiting for its current network transfer. Completed blocks survive
cancellation. Re-running skips valid blocks, including after a restart. Missing
payloads, write failures, low free space, and eviction report incomplete coverage.
VC3D's Project tab exposes this as **Download Objects**, **Cancel**, and, after
new commits, **Update Cache**. Local store rows display **Local**.
