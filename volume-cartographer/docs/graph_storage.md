# Graph object identity and storage

Schema v3 also supports [managed geometry inside SQLite](graph_geometry.md),
including immutable revisions, chunked binary arrays, and source ingestion.
The external-file behavior below describes metadata-only objects and the
source-location hints retained for re-ingestion.

The graph identifies objects by UUID. Patch import preserves the UUID in
`meta.json`; fiber import preserves the UUID in the fiber JSON. PCL linking
preserves each collection's UUID in its point-collections JSON. Edges reference
those UUIDs. `name` is a display label, and `storage_ref` is a location hint,
normally relative to the graph file's directory. Until explicitly ingested, the graph stores metadata and
relationships and geometry remains in the external files.

Use `GraphDb::resolveStoragePath` in C++, or `resolve_storage_path` in Python,
to locate a file-backed patch, fiber or point collection:

```python
from vc.graph import GraphDb

with GraphDb.open("/data/graph.vcgr", read_only=True) as graph:
    patch_uuid = "a8fc18bb-7710-4143-908f-6cc6d3e03ecc"
    patch_dir = graph.resolve_storage_path(patch_uuid)
    # If geometry lives elsewhere, supply its search root:
    patch_dir = graph.resolve_storage_path(patch_uuid, dataset_root="/datasets/spiral")
```

Resolution first verifies the source UUID at the saved path. If that path is
missing, unreadable, or belongs to a different UUID, it searches the dataset
root recursively. By default the root is the graph file's directory; an
explicit root also rebases relative path hints. Absolute hints remain absolute.
The result is an absolute path (`pathlib.Path` in Python): the patch directory
or fiber/point-collections JSON file. For a point collection the resolver checks
the UUID inside `collections`; several objects may share the same file.
Missing or unsupported objects return `None`.

Discovery reads JSON metadata without loading TIFF geometry. It skips hidden
directories (including `.strip-cache`) and `.zarr` stores, and does not recurse
through directory symlinks. It raises an error on filesystem traversal failures
or multiple matching locations discovered during fallback, rather than choosing
an arbitrary copy. A valid saved hint is returned immediately without searching
for duplicates elsewhere. Fallback discovery is performed afresh on each call;
for bulk geometry loading, retain resolved paths for the duration of the load.

Renaming files or moving a dataset does not change object or edge UUIDs.
Resolution works with existing graphs and read-only handles and does not rewrite
stored hints, names, generations, or approval state. Callers loading geometry
should use this resolver instead of concatenating `storage_ref` themselves.
`sweep_external_data()` remains an audit of the stored path hints, so it can
report stale paths even when UUID discovery succeeds.

Track records use their existing dataset reference and external track key;
they are not individual UUID-bearing patch/fiber files and are outside this
resolver's scope.

## Record snapshots and approval history

JSON export version 2 preserves each object's and edge's UUID, generation,
creation/change stamps, approval state and approval stamps, along with its
metadata, tags, detail fields and storage references. Import restores these
fields exactly. Importing identical existing records is a no-op; a conflicting
UUID rejects the whole import, without overwriting local changes. The import
uses a savepoint, so a caller can catch a conflict without committing a partial
snapshot. Unknown export versions are rejected.

Each imported record receives a local journal entry attributed to the importer;
its original authors remain on the record. A snapshot contains current records,
not the source journal or external geometry, and is not a distributed merge or
a complete database backup. Use a database copy after checkpointing for history.

Legacy version 1 exports remain readable into fresh identities. Their missing
provenance cannot be recovered: records start at generation 1, positive
`data_generation` values become 1, and approval stamps remain unknown.

The journal now captures approval state, reviewer, time, host and reviewed
generation after each record mutation. Approval decisions remain visible after
another decision or a content edit. Changes to derived lengths and fills of
missing track families count as content edits: generation increases, change
stamps update, and approval returns to pending. Identical measurements do not
write a journal entry or invalidate approval.

## Schema migration and patch scale

Schema v1 databases migrate to v2 atomically on a writable open:

```python
with GraphDb.open("/data/graph.vcgr", author="reviewer"):
    pass
```

Read-only opens never migrate and report when a writable open is needed first.
Older binaries cannot open the migrated database for writing. Historical v1
journal entries have null approval snapshot fields because the old decisions
were not stored. Migration adds `migration_snapshot` entries preserving the
current approval of every surviving record; it does not invent past decisions.

Patch `scale` is now a two-element array in JSON, a pair in Python, and
`std::array<double, 2>` in C++, matching the column/row values written by
QuadSurface. Both axes survive import, export and linking. Legacy scalar input
is accepted as isotropic; migration copies the stored scalar to both axes.
Anisotropy previously discarded by v1 cannot be recovered from the database.
Patch import also reads the nested bounding box in `meta.json`; malformed scale
or bounding-box fields are reported and that patch is skipped.

### Source imports and link reconciliation

Legacy file import seeds object names, tags, and metadata on first registration.
Subsequent fiber/track imports preserve those database-owned annotations and
refresh only source descriptors and measurements. A changed legacy fiber file
invalidates connected-link approvals. Once an object has stored geometry,
legacy import refuses to refresh it from external files; use geometry ingestion
to publish a new revision instead. Stored geometry remains authoritative.

Newly imported fiber branches carry `meta.source_import="fiber-json"`. Re-import
refreshes their anchors while preserving UUIDs and graph annotations, resets
approval on changes, and removes branches absent from both successfully parsed
endpoint files. Missing, unreadable, and omitted files do not establish removal.
Unmarked pre-existing branches have ambiguous ownership and are left alone;
they are not retroactively claimed as importer-owned.

Detector reruns refresh their own pending edges when replacement is enabled,
including coordinates, directions, scores, and supplied metadata fields. They
retain edge UUIDs, tags, and metadata keys outside the detector's supplied fields.
Unchanged edges remain no-ops. Reviewed edges, manual edges, and edges belonging
to other detectors remain protected. Reports distinguish `edges_updated` from
newly created and skipped edges. Patch-overlap stale deletion is limited to
pairs whose patches both loaded successfully, including when a run finds no
candidates. Damaged TIFFs and interrupted saves are excluded from that scope.

Stored-geometry publication already resets connected-link approvals and leaves
revision bindings stale. Refreshing edge content does not assert that an external
file detector ran against the stored revision: explicit geometry revalidation
is still required before approving stale links.

Object and edge insertion share a UUID namespace: ordinary writes reject a UUID
already used by the other record type, including casing variants. This does not
repair collisions that may already exist in older databases.

`objectCount` / Python `object_count` uses an indexed aggregate. Component and
bounded-depth queries probe endpoint indexes as they visit nodes instead of
materializing all graph edges. Bridge graph queries execute on worker threads
and retain the opened database through controller close/reopen. Responses retain
the existing JSON shape; deferred queries have a 30-second deadline, and component
traversal observes cancellation when the request expires or the server closes.
