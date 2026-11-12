# Object store API docs

API documentation for `vc_object_store`, a SQLite object database for
revisioned geometry, metadata, and graph edges (see the
[package README](../README.md) for the overview). Each document opens with a
summary, then the operations for one kind of use, with Python, C++, and CLI
forms where they exist. Python examples assume
`from vc_object_store import ObjectStore` and an open `store`.

Current schema 21 also supports history-off stores, immutable store_references and overlays, independent shards and optional packed SQLite containers. Current versions and the store commit sequence remain available without archival history. See [store format](store-format.md) for version semantics and the [shared API contract](store-api-contract.md) for implementation rules.

| Document | Covers |
| --- | --- |
| [Ingestion](ingestion.md) | One-time import of surfaces, fibers, point collections, tracks, umbilici; registered placeholders; coordinate context; `vc_object_ingest` |
| [Reading](reading.md) | Opening a store, listing and selecting records, snapshot geometry views, bounded array reads, history reads, `geometry_region`, `nearest_surfaces`, selection records |
| [Writing](writing.md) | Creating objects and geometry, checked transactions and typed requests, receipts and conflicts, native export, artifacts, copying, project records |
| [Editing](editing.md) | Sparse coordinate and rectangle edits, surface channels, structural sequence edits, surface resize, metadata edits, retire/restore/delete, editor drafts |
| [Graph edges](graph-edges.md) | Relationship records and kinds, anchors and dependencies, decisions when geometry changes, applicability, traversal and `graph_read`, link detectors, `vc_object_links` |
| [Transforms](transforms.md) | `meta.coordinates` (volume identity and scale), volume registrations and route choices, transformed views and edits, volume-target queries |
| [Typed geometry](typed-geometry.md) | Owned geometry values (`PointCollection`, `Fiber`, `Track`, `Surface`, `Umbilicus`), snapshot views, encoding, validation |
| [Typed extensions](extensions.md) | Optional indexed result/measurement tables, extension provisioning, input-state references and versioning |
| [Hosted stores and sessions](hosted.md) | `vc_object_server`, `HostedStore` clients, `ObjectStoreSession` attachments, change events, hosted artifacts, durable drafts |

## Shared rules

These hold across every document:

- **Versions.** Every object and relationship keeps its UUID. Each published
  state also has a version (1-based, one per transaction that changes the
  record, deletions included). A historical reference is a
  `{store, generation, object, version}` dict (C++ `ObjectVersion`).
  Revision UUIDs seen in geometry APIs identify immutable geometry payloads;
  they are not the public way to name an object state — use numbered versions.
- **Checked writes.** A write expects the complete references it read
  (`expected_versions` / `expectedObjects`; `None` means "no live record").
  A stale expectation conflicts (`TransactionConflict` in C++, `RuntimeError`
  in Python) and writes nothing. Unchecked convenience calls
  (`store.set_meta(...)`, `store.put_geometry(...)`) run their own transaction
  under the writer lock, but cannot protect a read made earlier outside it.
- **Geometry edits revisit edges.** Any operation that changes geometry or
  lifecycle state must decide for each affected relationship: `retain`,
  `update`, `remove`, or `keep_recorded`. See [Graph edges](graph-edges.md).
- **Author is attribution.** Every store open and request carries an `author`
  string. It records who wrote; it is not authentication.
- **Errors.** Invalid arguments raise `ValueError` in Python
  (`std::invalid_argument` in C++); version conflicts raise `RuntimeError`
  (`TransactionConflict`); corrupt or unreadable stored data raises
  `RuntimeError` on read.
- **`published_at_us`** is UTC Unix microseconds, shared by all revisions of
  one action and strictly increasing between actions.

C++ signatures live under
[include/vc_object_store/](../include/vc_object_store/); the master header is
[objects/ObjectStore.hpp](../include/vc_object_store/objects/ObjectStore.hpp).
Link `vc_object_store::vc_objects` (storage), `vc_object_geometry` (native
formats), or `vc_object_links` (detectors). Build and test commands are in the
[package README](../README.md#build-and-test).

## Immutable and large-scale stores

- [Store format and policies](store-format.md): schema 21, history retention, sealing and database identities.
- [Offline schema conversion](schema-conversion.md): supported upstream inputs, validation and history maintenance.
- [Collections](collections.md): direct-key store_references and overlays, immutable dependencies and sharding.
- [Initial batch import](initial-import.md): shared source-independent, parallel ingestion.
- [Packed stores](packed-stores.md): optional single-file containers and their VFS.
