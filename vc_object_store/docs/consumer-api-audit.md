# Object-store consumer API audit

Date: 2026-10-20

Scope: fibers, tifxyz surfaces, and point collections used by VC3D and the sibling `spiral-fitting` project.

This is a source-level audit. Performance concerns below are inferred from code paths, not measured latency. No implementation changes or builds were performed for the audit.

The store already provides tiled reads, sparse edits, spatial indexing, packed arrays, transactions, and graph snapshots. The largest remaining gaps concern complete object round-tripping, revision/lifecycle semantics, and efficient access for interactive workloads.

Each finding has space for answers and decisions. Proposed APIs are suggestions, not agreed requirements. References point to the code inspected during the audit; line numbers may change.

## 1. One revision-history model for geometry, metadata, and relationships

Spiral's editing workspace tracks accepted, applied, and persisted revisions of complete inputs. The current store preserves geometry history, while ordinary annotations and graph relationships remain mutable records. An old geometry revision therefore does not recover the matching winding annotations, tracing settings, or relationships.

**Decision:** use one object-history model. Each immutable object revision describes its geometry, ordinary metadata, and graph relationships. Relationship references identify the target object, target revision, and point/anchor identity. Changes to any of these participate in the same revision mechanism.

For example:

```text
P revision 7: point position and a link to A revision 4, point ID 12
P revision 8: edited position; the link is explicitly validated and retained
A revision 5: point 12 moves and no longer connects to P's point
```

P revision 8 still records its connection to A revision 4. That historical target remains available. Creating A revision 5 does not silently retarget the relationship or require rewriting every object that references an older revision of A. Removing or updating P's relationship creates another revision of P; earlier revisions retain the earlier relationship.

Readers must distinguish the recorded relationship from a relationship valid against selected endpoint revisions. Resolving a recorded reference follows its pinned target revision. A query for current connections must check applicability to the latest revisions rather than silently substituting them. Restoring historical content into the current state likewise requires checking affected relationships.

**Required contract clarifications:**

1. **Shared relationship ownership.** The example describes a reference owned by P, while the existing store also represents edges shared by two endpoints. Define which object revisions change when a shared edge is added, modified, or removed, and ensure queries from either endpoint agree for the same revision selection. Immutable relationship records may be shared between object revisions without duplicating their contents. Ownership and revision membership must be explicit; changing an endpoint does not by itself rewrite historical references held by other objects.
2. **Explicit relationship preservation.** Pinning A revision 4 preserves the historical target; it does not establish that the relationship remains applicable after P moves. At commit/save, changes affecting either endpoint require an explicit keep, modify, or remove decision, with validation against the resulting state, as required by item 10. Keeping a historical reference and validating it for use with newer endpoint revisions are distinct operations. A metadata-only edit must not invalidate a geometric relationship solely because the object revision changed, unless that metadata affects the relationship.
3. **Reads against selected revisions.** Support an explicit selection of object revisions for relationship queries and validation; latest revisions are one convenient selection. Spiral must be able to construct a graph against the inputs actually applied to the fit, which may differ from the latest accepted or persisted inputs. Spiral remains responsible for tracking accepted, applied, and persisted workflow states. Geometry, metadata, and relationships returned for a selection must describe those selected revisions consistently.
4. **Atomic publication and conflict checks.** Publish all geometry, metadata, and relationship changes belonging to one action atomically, including actions spanning multiple objects. Require expected-revision checks for all affected state, including metadata and relationships, and check endpoint revisions used for relationship validation. A conflict must publish none of the action. This is a foundational requirement of item 1 and also applies to the structural operations in item 5.

**Cost requirements:** store only changed geometry/channel data, metadata fields, and relationship records; reuse unchanged content. Metadata-only or relationship-only edits must not copy or rehash geometry. An unchanged relationship should not be copied into every revision. This is one logical history model; the physical encodings may differ for geometry deltas, metadata changes, and small relationship records. Keep indexed current-state views so ordinary reads do not replay history.

**References:**

- [Spiral revision model](../../spiral-fitting/input_workspace.py#L68)
- [Current geometry and edge schemas](../src/internal/Schema.hpp)
- [Geometry delta implementation](../src/geometry/Geometry.cpp#L318)
- [Current relationship updates](../src/relationships/Edges.cpp#L15)
- [Metadata update implementation](../src/Metadata.cpp#L51)

**Answers / decision:**

- Response: We should have one model for object history. Geometry, ordinary metadata, and graph-like metadata belong to the same revision model. A relationship should point to the revision of its target where it correctly applies. Later edits to that target must preserve the historical relationship by keeping that revision reference. Do this cheaply through deltas and reuse of unchanged data.
- Priority: 1
- Agreed action: Implement unified object revisions with revision-qualified relationship references and incremental storage. Define shared relationship ownership and revision membership. Provide explicit recorded-reference reads and validity queries against selected revisions, including latest revisions. Require explicit handling of affected relationships and atomic publication with expected-revision checks across geometry, metadata, relationships, and participating objects. Implement this before the other audit items.

### Implementation status / notes

- Status: Implemented and validated (2026-10-20).
- Notes: Schema v9 publishes immutable object revisions spanning metadata, geometry references, and relationship membership. Metadata is stored as JSON field deltas, geometry retains its existing chunks/sparse deltas, and unchanged relationship records/memberships are inherited. Indexed current metadata and relationship views avoid replay for ordinary head reads. Implementation: `src/History.cpp`; C++ and Python contracts and examples: [Object store API](README.md#object-revisions-and-checked-actions).
  Shared edges belong to both endpoint objects: addition, modification, removal, and explicit rebinding revise both endpoints in one transaction. A historical-preservation decision (`keep_recorded`, or `keep_recorded_relationship`) retains the original immutable reference and does not revise the other endpoint. Explicit revalidation (`retain`/`update`, or `retain_relationship`) is a separate shared edit. `recorded_relationships` follows recorded object/geometry pins; `relationships_at` checks shared membership and declared dependencies against an explicit selection, with latest heads supported. `revision_read_view` returns selected metadata, geometry references, and relationships from one snapshot. Anchors may declare ordinary `metadata_keys` dependencies; unrelated annotations do not invalidate geometry relationships.
  Checked actions use `write(expected_revisions=...)` / `write(RevisionSelection)`. They require expected object heads for every modified object and validation endpoint, including shared metadata/edge changes. A failed coverage check or revision conflict rolls back the whole action; caught geometry conflicts also prevent commit. The original serialized `write()` remains compatible with ingestion and read/modify/write under the writer lock; applications preparing edits from earlier reads must use the checked entry point.
  Each revision carries `published_at_us` (UTC Unix microseconds). All revisions in one action share the same timestamp, strictly increasing across actions even on clock ties/rollback. `revision_selection_at` and `revision_read_view_at` recover the live objects, geometry, metadata, and complete recorded graph at a chosen time without a separate commit-marker table. Physical deletion leaves a timestamped history event so past queries do not resurrect deleted objects. Existing deletion eligibility is unchanged. Restoring historical content through the mutators creates a new action and requires normal relationship decisions.
  Migration preserves current state and historical geometry pins, but cannot reconstruct annotations or links overwritten before v9. Such baselines are explicitly marked `metadata_complete=false`; detached geometry baselines are untimed and excluded from temporal live-state selection. Publication times belong to this store: existing synchronization merges current records and geometry history, not the peer's historical metadata/graph timeline. Imported older bindings without local object history likewise receive explicitly incomplete, untimed baselines; transporting a complete timeline requires a database copy. Spiral still owns accepted/applied/persisted workflow states. Historical reconstruction and incident-edge discovery remain the separate performance concerns in items 11 and 10.
  Validation: 194 Python object-store tests and all 3 C++ object-store suites passed. Coverage includes historical geometry/metadata/links, shared membership, explicit selected revisions, whole-action rollback and caught conflicts, restoration, dependency metadata, immutable/incremental storage, migration/reopen, temporal graph recovery, deletion, and clock rollback. The real PHerc 0172 patch fixture (`core/test/data/segments/20241113070770`) exercises metadata revisions alongside exact sparse geometry history. These are correctness/storage checks, not new latency benchmark claims. Commands:

  ```sh
  AGENTS_AGENT_MODE=1 ninja -C build -j 4 test_object_store test_object_records test_object_links vc_object_bindings
  ctest --test-dir build -R '^test_object_(store|records|links)$' --output-on-failure
  AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python .venv/bin/python -m pytest -q tests/python/test_object*.py
  ```

  Python plugin auto-loading is disabled to avoid unrelated installed Napari test fixtures; no object-store tests are disabled.

## 2. Tifxyz ingestion/export omits important ancillary data

VC3D uses approval masks and generation channels. Spiral-fitting loads `winding.tif` and `overlapping.json`.

Patch ingestion stores XYZ and `meta.json`, while export writes only XYZ and metadata. Adding auxiliary arrays manually does not solve round-tripping: native export explicitly rejects them. The store also lacks native `uint16` support, which VC3D uses for the generations channel.

**Consequence:** using the store as the authoritative surface representation loses editing/review information and potentially fitting constraints.

**Suggested capability:** named channels, channel-specific reads/edits, native dtype preservation, and import/export of overlap relationships.

**References:**

- [VC3D channels](../../volume-cartographer/apps/VC3D/segmentation/growth/SegmentationGrowth.cpp#L85)
- [Spiral surface inputs](../../spiral-fitting/tifxyz.py#L409)
- [Patch ingestion](../src/geometry/GeometryIngest.cpp#L116)
- [Export restriction](../src/geometry/GeometryExport.cpp#L226)
- [Supported dtypes](../src/geometry/GeometryInternal.hpp#L41)

**Answers / decision:**

<!-- Which channels and sidecars must remain authoritative? Should they be stored with the surface revision or managed separately? -->

- Response: we should allow extra named channels and channel specific reads/edits, and uint16 support. they should be stored with the surface revision but we must make sure that this is done in a way which is cheap (for ex, making sure that an edit to an approval mask or something doesnt incur whole object snapshots)
- Priority: 2
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## 3. Each point collection needs its own volume and coordinate scale

VC3D writes coordinate identity at the document's top level. Ingestion retains each collection's fields but drops the surrounding document metadata. Export produces one collection per document.

Each collection should be an independent object that stores its own volume and coordinate scale. The shared file-level metadata is a consequence of the historical disk format, not a reason to introduce a shared collection-set object or metadata pointer. Import should preserve the legacy document's volume and coordinate-scale information on each imported collection.

The original audit also identified incomplete export of linked collections: exporting one collection does not include its linked targets. Export is out of scope for now; it is not a planned use case. The intended architecture is for VC3D and spiral-fitting to use the object store directly.

**Agreed capability:** per-collection volume and coordinate-scale metadata, including preservation during import. No document/group identity or linked-collection export work is required for this finding.

**References:**

- [Coordinate metadata producer](../../volume-cartographer/apps/VC3D/CState.cpp#L234)
- [Collection ingestion](../src/geometry/GeometryIngest.cpp#L151)
- [Collection export](../src/geometry/GeometryExport.cpp#L82)
- [Native link reconstruction](../src/geometry/SourceMetadata.cpp#L68)

**Answers / decision:**

- Response: Each collection should be its own thing. Shared information exists only because of the historical decision about how collections were stored on disk. Each collection must store its volume and coordinate scale itself, not through a shared pointer for a set of collections. Do not worry about exporting for now; it is not something we plan to use. The future purpose is for VC3D and spiral-fitting to use the object store directly.
- Priority: 3
- Agreed action: Store volume and coordinate scale independently on each collection and preserve those values when importing legacy files. Do not introduce a collection-set abstraction to hold this information. Leave export work out of scope for now.

### Implementation status / notes

- Status:
- Notes:

## 4. Deleting or retiring an object with managed geometry is unsupported

VC3D deletes fibers and retires inputs during structural edits. Spiral supports deletion revisions.

`deleteObject()` performs a physical deletion, but retained geometry references the object with `ON DELETE RESTRICT`, and revision deletion is prohibited. Removing incident edges therefore still does not make an ingested object deletable.

**Suggested capability:** tombstone/retire/restore operations, default exclusion of retired objects from queries, and an explicit retention policy.

A metadata flag can approximate retirement, but every consumer must remember to filter it.

**References:**

- [VC3D deletion](../../volume-cartographer/apps/VC3D/LineAnnotationController.cpp#L3275)
- [Spiral deletion/restoration](../../spiral-fitting/service_editing.py#L260)
- [Store deletion](../src/Records.cpp#L52)
- [Retention constraints](../src/internal/Schema.hpp#L90)

**Answers / decision:**

<!-- Should deletion mean retirement, reversible deletion, or eventual physical removal? How should incident relationships behave? -->

- Response: i think deletion can physically remove ONLY IF it has no metadata references via edges/graph-like objects or other references to it from different objects. in those cases we need to just "retire" it to keep the history intact
- Priority:
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## 5. Structural editing falls outside the efficient edit API

Existing workflows include:

- Appending/deleting PCL points and reversing collections.
- Merging/splitting fibers and retracing variable-length polylines.
- Expanding surface grids.

`editGeometry()` only replaces scalar ranges without changing shape, dtype, or attributes. Structural operations require `putGeometry()` with complete arrays, which hashes/processes the full replacement despite chunk deduplication.

There is an additional consistency burden: PCL annotations and fiber control-point attributes live in positional JSON arrays. Callers must resize/reorder those alongside geometry and remap positional graph anchors.

Export detects annotation length mismatches, but an incorrect same-length reorder can still associate annotations with the wrong points.

**Decision:** extend the unified object-history model with structural edits that reuse unchanged geometry. Structural operations create ordinary object revisions through the same history API.

Agreed capabilities:

1. **Explicit structural operations.** Support append, insert, delete, span replacement, reversal, and reordering for sequences. Support grid growth, cropping, rectangle replacement, and coordinated channel resizing for surfaces, with explicit fill values.
2. **Stable point identity independent of order.** Preserve PCL point IDs and introduce stable fiber control-point IDs. Associate annotations with those IDs and store sequence order separately. Generated fiber samples do not necessarily need stable IDs; retracing may replace them. Revision-qualified positional references remain valid for historical geometry.
3. **Reuse unchanged storage.** Represent sequence edits using references to existing immutable prefix/suffix blocks plus new data, so an insertion does not shift and rewrite the entire suffix. Give surface tiles stable grid coordinates so right/bottom growth writes new tiles and affected boundary tiles while retaining the interior. Resize geometry and associated channels together so channel values remain attached to the correct locations.
4. **Atomic publication.** Commit structural edits, annotation changes, and explicit relationship updates together, checking the expected starting revision. Historical links retain their pinned targets. Following a point into a new revision requires an explicit, validated relationship preservation or remapping. For merges/splits, applications retain responsibility for domain rules; the store publishes resulting objects, relationship changes, and source retirements in one transaction.
5. **Efficient reads.** Continue exposing contiguous arrays and rectangles without making consumers understand shared storage blocks. Bound fragmentation so long editing histories do not progressively degrade reads.

**Implementation order within this item:** stable point identity, append/span replacement, and surface growth first; build split/merge and reversal support on those primitives.

**Performance goal:** write new data and affected boundaries, and reuse unchanged content. Adding a point or growing an edge should not process the whole object. A complete retrace still costs proportionally to the newly generated geometry. Achieving this requires storage-layout changes; wrapping full replacement in an `insert()` API is insufficient.

**References:**

- [PCL editing](../../volume-cartographer/apps/VC3D/SpiralPointCollectionEdit.cpp#L115)
- [Fiber merging](../../volume-cartographer/apps/VC3D/LineAnnotationStructuralEdits.cpp#L123)
- [Geometry edit contract](../include/vc_object_store/objects/Geometry.hpp)
- [Replacement work](../src/geometry/Geometry.cpp#L132)
- [Annotation validation](../src/geometry/GeometryExport.cpp#L52)

**Answers / decision:**

- Response: The proposed structural-editing approach is sound and is accepted: explicit structural operations, stable point identities, reuse of unchanged storage, atomic geometry/annotation/relationship publication, and efficient reads with bounded fragmentation.
- Priority:
- Agreed action: Implement the capabilities and staged order above within the unified object-history model. Keep merge/split domain decisions in the applications and support atomic publication in the store.

### Implementation status / notes

- Status: Decision recorded; implementation pending.
- Notes: Start with stable point identity, append/span replacement, and surface growth. Preserve revision-qualified historical references from item 1. Coordinate channel resizing with the named-channel support in item 2, and source retirement with item 4.

## 6. Repeated ingestion is not a refresh mechanism for existing file-based editors

Source annotations are seeded once; subsequent ingestion deliberately preserves database annotations. Source relationships are likewise imported only for newly seeded source metadata.

Consequently, editing winding annotations, branch links, or tracing settings in VC3D and re-ingesting does not update those fields. Geometry changes affecting existing links can instead fail because ingestion supplies no edge-edit decisions.

There is also an incremental-import performance concern: importing a new PCL triggers enumeration of all stored PCLs and loading their point IDs/coordinates to resolve document-scoped links.

**Suggested capability:** a distinct refresh/reconcile operation with explicit conflict handling and source identity indexes.

The existing ingest behavior is reasonable for initial migration, but cannot transparently bridge ongoing file-based editing.

**References:**

- [Publication policy](../src/geometry/GeometryIngest.cpp#L234)
- [Relationship import scan](../src/imports/SourceRelationships.cpp#L91)

**Answers / decision:**

<!-- Will files remain editable authorities after migration, or will applications write directly to the store? Is bidirectional reconciliation required? -->

- Response: suggestion is fine
- Priority:
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## 7. Fiber graph semantics exceed the generic traversal filters

Spiral excludes pending links and adjacent-winding links when constructing same-winding components. VC3D uses adjacent links differently, including winding offsets.

In the store, both ordinary and adjacent links use `fiber_branch`; adjacency is JSON metadata. `componentOf()` and `reachableWithin()` filter by edge kind, so they cannot directly express the fitting graph. Graph snapshots support approval filtering, but lack adjacency and stale-binding predicates.

There is also a concrete round-trip change:

1. Branch ingestion ignores the source `pending` value.
2. New store edges start pending.
3. Export derives `pending` from store approval.
4. Previously non-pending links therefore export as pending unless explicitly approved, causing spiral's default resolver to exclude them.

**Suggested capability:** explicit relationship semantics/predicates and a deliberate mapping between source link status and store review status.

Fetching edges and rebuilding a filtered graph client-side remains possible.

**References:**

- [Spiral filtering](../../spiral-fitting/spiral_helpers.py#L406)
- [VC3D adjacent winding offsets](../../volume-cartographer/apps/VC3D/FiberNetworkLayout.cpp#L304)
- [Branch ingestion](../src/imports/SourceRelationships.cpp#L70)
- [Status export](../src/geometry/SourceMetadata.cpp#L58)
- [Selection API](../include/vc_object_store/objects/ObjectStore.hpp#L145)

**Answers / decision:**

<!-- Should adjacent and ordinary links be different edge kinds? Is source pending status the same concept as store approval? Which graph filters belong in the store? -->

- Response: suggested fix is good
- Priority:
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## 8. Interactive fiber browsing lacks an equivalent bounded query API

VC3D's existing `FiberCollection::viewRegion()` provides:

- Oriented-slice clipping and contiguous visible runs.
- Selected-fiber priority and stable limiting.
- Cancellation.
- An 8 MiB decoding budget at the VC3D call site.

Store `spatialRegions()` returns AABB candidates with count-based pagination.

**Consequence:** an adapter must load candidates, perform exact clipping, implement selection priorities, and enforce its own decoding budget. An oblique slice's enclosing AABB can generate substantial extra work. A result-count limit cannot substitute for a geometry-byte budget.

**Suggested capability:** a cancellable, budgeted region-reader adapter over the spatial/chunk APIs, preserving the current viewer contract.

**References:**

- [Existing viewer contract](../../volume-cartographer/core/include/vc/core/types/FiberCollection.hpp#L72)
- [VC3D invocation](../../volume-cartographer/apps/VC3D/FiberCollectionController.cpp#L798)
- [Store spatial query](../src/Queries.cpp#L122)

**Answers / decision:**

<!-- Should the store replace the AFV viewer backend? If so, which viewRegion guarantees must be preserved? -->

- Response: suggestion is good
- Priority:
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## 9. Packed reads do not match spiral's heterogeneous surface atlas

Spiral flattens differently sized `(H,W,3)` grids into vertices plus offsets, widths, and heights, with bounded host staging.

Store packed reads concatenate along the first dimension and require matching trailing dimensions. Surfaces with different widths therefore cannot share a packed read.

Furthermore, one delta revision causes the entire packed request to fall back to per-revision reads and vector appends, losing the snapshot-only batch path and introducing intermediate buffers.

**Suggested capability:** ragged flattened geometry reads carrying per-object shapes, bounded iteration or `read_into`, and mixed snapshot/delta batching.

Existing NumPy views already avoid an additional copy of the final packed buffer; that part is covered.

**References:**

- [Spiral atlas](../../spiral-fitting/fit_spiral.py#L369)
- [Bounded host staging](../../spiral-fitting/fit_spiral.py#L423)
- [Mixed-revision fallback](../src/geometry/GeometryReader.cpp#L445)
- [NumPy buffer ownership](../python/ObjectReadBindings.hpp)

**Answers / decision:**

<!-- Is direct atlas loading from the store a target? What memory budget and batch representation should the API support? -->

- Response: suggestion is fine. and yes we plan on being able to load the spiral fit from the store. we should be able to stream tiles or something pretty cheaply
- Priority:
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## 10. Sparse edits can do work proportional to the entire relationship set

Every affected-edge calculation loads all incident edges and examines their dependencies. A tiny surface brush edit on a heavily linked patch therefore has a cost tied to total degree.

PCL dependency resolution additionally loads and sorts the complete point-ID array. Previewing affected edges and subsequently committing repeats relevant preparation.

**Suggested capability:** a reverse index from geometry regions/point IDs to dependent edges, and potentially a validated edit plan reusable at commit.

The explicit retain/update/remove policy is valuable; the expensive part is discovering which relationships need it.

**References:**

- [Incident-edge scan](../src/geometry/Geometry.cpp#L61)
- [Point-ID indexing](../src/geometry/Dependencies.hpp#L74)
- [Sparse edit preparation](../src/geometry/Geometry.cpp#L221)

**Answers / decision:**

<!-- How densely linked will editable surfaces/collections be? Should relationship repair happen on every edit, at stroke completion, or at save/publication? -->

- Response: we only need to handle relationships on commit/save (which is after the "edit" action finishes). and edits that change relationships must be _explicitly_ handled by either keeping, modifying, or removing them
- Priority:
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## 11. Frequent edits affect cache reuse, historical reads, and catalog summaries

Geometry caches clear whenever the connection observes a database commit, including metadata-only changes. Reviewing/tagging objects can therefore evict unchanged geometry used by the viewer.

Historical delta revisions reconstruct their manifest through the delta chain. The current-head materialization and reverse-payload optimization do not eliminate that historical manifest work.

Separately, length/count summaries intentionally remain ingestion annotations. After direct geometry edits, length filtering and ordering can become outdated unless the application updates them itself.

**Suggested capability:**

- Revision/content-keyed cache retention.
- Bounded historical reconstruction.
- Revision-consistent derived summaries for catalogs.

**References:**

- [Cache invalidation](../src/geometry/GeometryReader.cpp#L12)
- [Historical reconstruction](../src/geometry/GeometryReader.cpp#L111)
- [Summary contract](../include/vc_object_store/objects/ObjectStore.hpp#L17)

**Answers / decision:**

<!-- How much historical browsing is expected? Should catalog summaries describe source ingestion or current geometry? -->

- Response: we don't expect much historical browing outside of small undo edits for thinks like erasing/push and pull brush/ and drag brush
- Priority:
- Agreed action:

### Implementation status / notes

- Status:
- Notes:

## Proposed priorities and validation

Agreed first priorities, followed by the remaining suggested order:

1. Unified object history for geometry, ordinary metadata, and revision-qualified relationships.
2. Extra surface channels, channel-specific reads/edits, and uint16 support with incremental storage.
3. Per-collection volume and coordinate-scale metadata.
4. Retirement semantics, structural editing, and graph semantics.
5. Bounded viewer queries and ragged atlas access.
6. Performance improvements guided by real-data measurements.

VC3D and spiral-fitting are intended to use the object store directly. Export work is out of scope for now.

Representative performance workloads:

- Oblique fiber browsing, including cancellation and selected-fiber visibility.
- Small edits to heavily linked surfaces.
- Mixed snapshot/delta atlas loading with heterogeneous surface sizes.
- Historical reads after long editing sessions.
- Incremental PCL ingestion into a large existing store.

**Overall answers / revised priorities:**

<!-- Note intended architecture, accepted limitations, and which findings should become implementation tasks. -->

- Intended architecture:
- Accepted limitations:
- Revised priorities:
- Representative datasets/workloads:
