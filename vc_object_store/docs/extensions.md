# Typed extension tables

Extensions add ordinary, typed SQL tables without adding domain-specific columns to core objects or relationships. The core schema owns table registration, common input-state references and transactional change stamps (introduced in schema 21 and retained in schema 22). Each extension owns its column layout, indexes, format version and small provisioning JSON object. Existing object/edge UUIDs remain the references: results do not require duplicate geometry, proxy objects or an attachment edge per row.

Use this for measurements and derived results that generic viewers need not understand, such as fiber-constraint distances, classifications or winding runs. Keep ordinary geometry, endpoint anchors and graph relationships in the core tables. Arbitrary `meta` JSON remains available, but large numerical result sets belong in typed columns rather than JSON blobs.

## Metadata and identity

The optional `extension_tables` registry has one descriptor per physical extension table, referring to its existing `table_directory` UUID. Extension tables use names beginning with `ext_`. SQLite's table schema already describes columns and indexes; the registry does not duplicate every row's layout or create a per-object directory.

| Information | Meaning |
| --- | --- |
| Extension identifier | The consumer-defined extension family |
| Extension format version | The interpretation of this extension's data, independent of the core schema version |
| Physical table UUID/name | The table instance registered in the existing directory |
| `basedOn` | Common input-state references for the rows in this table |
| `lastModifiedCommit` | Destination-store commit which last wrote this table through the extension API |
| `provisioning` | Small extension-owned JSON object: units, field conventions, source-format identity or processing configuration |
| `keySpace` | Optional colocation with the core `objects` or `edges` UUID routing rule |
| `versioned` | Whether manual edits participate in their owning object/edge's versions and history |

Numerical data is stored as `INTEGER`, `REAL`, `TEXT` or `BLOB` columns. UUID fields use binary 16-byte values, matching the rest of the store. Primary keys and optional indexes are ordinary SQL indexes; no per-row JSON parsing is needed for indexed measurement lookups. NULL is distinct from zero.

### Provisioning contract: metadata, not payload

Every extension must construct its provisioning JSON explicitly from the relevant values. Do not copy a source metadata/configuration/report tree and then filter it with an allowlist or denylist. Provisioning describes how to identify and interpret the data; it is not an archive of the import operation or another storage location for the dataset.

- Source identity: an actual source UUID or checksum, if available, solely to identify/check that source. Omit unknown identity rather than inventing it. Do not store filesystem paths, source locations or command lines containing them in provisioning; store resolution belongs to the separate collection/location APIs.
- Interpretation: compact format identifiers, units, field meanings, coordinate conventions and categorical encodings. Use the table definition for columns/indexes and the common input-state references for vcobj versions instead of duplicating them in JSON.
- Settings and summary statistics: explicitly chosen small configuration values and aggregate counts/statistics are permitted. Per-run settings or summaries belong to the run when the table contains multiple runs, not to a repeatedly rewritten table descriptor.
- No data dumps: selection-ID lists, per-object mappings, samples, coordinates, normals, probabilities, constraint records, per-iteration histories and full reports belong in typed rows or appropriate binary payloads. Do not put them in provisioning, including under nested `provenance`, `settings` or `stats` keys. Compact schema/category descriptions are metadata; arrays whose size follows the imported objects, constraints, samples or iterations are payload.

Provisioning size should depend on the extension's description and compact configuration, not on dataset size. This matters especially for sharded stores, where a descriptor exists in each physical table. Keep shared input references at table or run scope; do not add per-object identity lists to reproduce them.

This is an extension-author contract, not a generic JSON sanitizer or a universal field allowlist. The generic API validates the provisioning JSON object but cannot infer whether arbitrary extension-owned fields are metadata or payload. Each producer must construct the intended small descriptor directly and test that source paths and dataset-sized arrays are not copied. The fiber import builders enforce this by accepting explicit identity/count inputs rather than an arbitrary source JSON object.

## Input versions are not write versions

`ExtensionSource` identifies `(storeUuid, generation, commitSequence)`. Here `generation` is the source's lineage UUID, not a numeric counter. The common source-reference format is shared across extensions.

If a computation reads source commit 40 and writes its result after the source reaches 43, its input reference remains 40. The destination's write commit is separate. An unrelated update never changes what an existing result was based on. A source reference records identity, not retention: historical reconstruction still needs retained source history or the sealed source file.

Use table-wide `basedOn` only when every row shares those inputs. For a table containing several runs with different inputs, the extension keeps per-run input references in its own run schema, using the same reference format. Do not overwrite one table-wide source reference when a later run is appended. Extension-specific dependencies, including previous result runs, belong in that extension's provisioning/run schema.

An external AFV/AFC file has no vcobj commit sequence. Record its actual source-format identity/provenance in provisioning instead of inventing a vcobj snapshot reference.

## Native API

The definitions are in [`objects/Extensions.hpp`](../include/vc_object_store/objects/Extensions.hpp); `ObjectStore` exposes creation, discovery and typed row operations. The following immutable result table uses one common input state:

```cpp
#include <vc_object_store/objects/ObjectStore.hpp>
using namespace vc::objects;

ExtensionTable table;
table.name = "ext_fiber_classification";
table.extension = "fiber-classification";
table.formatVersion = 1;
table.provisioning = R"({"family_codes":{"H":0,"V":1}})";
table.basedOn = {{sourceStoreUuid, sourceLineageUuid, sourceCommit}};
table.columns = {
    {"fiber_uuid", ExtensionType::Blob},
    {"family", ExtensionType::Integer},
    {"probability", ExtensionType::Real},
};
table.primaryKey = {"fiber_uuid"};
table.indexes = {{{"family", "probability"}, false}};
table.keySpace = "objects";

auto write = store.write();
store.createExtensionTable(write, table);
std::vector<ExtensionRow> rows = {
    {extensionUuid(fiberUuid), int64_t(0), 0.9},
};
store.appendExtensionRows(write, table.name, rows);
write.commit();

auto read = store.read();
std::vector<ExtensionValue> key = {extensionUuid(fiberUuid)};
store.scanExtensionRows(read, table.name, [&](const ExtensionRow& row) {
    // Values are in declared column order; return false to stop streaming.
    return true;
}, key);
```

`extensionTables(transaction)` returns registered descriptors. `scanExtensionRows` streams in primary-key order within a physical table; an optional equality prefix uses that key's index. `ExtensionValue` is a variant of null, signed 64-bit integer, double, string and byte vector. `extensionUuid()` converts between UUID text and the common binary representation. The shared `extensionSourcesJson()`/`extensionSourcesFromJson()` codec can also be used by extension-owned run metadata.

Registration does not replace an existing table definition or its input provenance. Repeating the same definition is a no-op; incompatible definitions fail. A table containing immutable results uses `versioned=false` and accepts appended rows with distinct keys. Manual data uses the versioned operations described below, not an append that bypasses owner versioning.

Extension writes share the caller's SQLite transaction. Failed append operations and rolled-back transactions do not leave partial rows or change stamps. Committing a logical write advances the destination store's commit sequence once, not once per inserted row. Sealed stores reject writes. Because the object/relationship change feed does not encode arbitrary extension rows, extension writes invalidate older record-only feed cursors rather than reporting a misleading empty change.

## Immutable results and editable data

This covers all three ownership cases: extra fields use an existing object's or relationship's UUID; an extension-created entity uses an ordinary core object with an extension-defined `kind`; custom binary data is stored in typed BLOB fields and included in that owner's versioned snapshot chunks. Known geometric data still uses the normal geometry API. These cases share the same checked writes, owner versions, history policy and restore behavior, rather than introducing a separate extension version counter.

Immutable computation results and editable extension data have different write policies. An immutable result preserves the input state it was computed from; a new computation creates a new result/run rather than overwriting that provenance. An editable extension participates in the ordinary object or relationship versioning and history policy of its owner. It must not silently change independently of the record version users rely on.

These are not separate numerical change clocks. The store's commit sequence describes committed writes; ordinary owner versions describe editable record states; input references describe the state a computation used. Enabling history retains the corresponding editable states, while history-off retains current state and freshness checks without pretending old versions can be reconstructed.

For editable fields, set `versioned=true` and `keySpace="objects"` or `"edges"`. The first primary-key column must be the owner's binary UUID; additional key columns allow multiple rows per owner. `ObjectStore::replaceExtensionRows(write, tableName, ownerUuid, rows)` replaces that owner's complete row set, not the whole table. An empty row set removes those fields. The operation creates a writable owner override when the selected owner is inherited; both store and collection APIs use the same implementation.

```cpp
// Create a separate manual table with its own immutable schema definition.
ExtensionTable manual = table;
manual.name = "ext_manual_classification";
manual.extension = "manual-classification";
manual.basedOn.clear();
manual.versioned = true;
auto edit = store.write();
store.createExtensionTable(edit, manual);
store.replaceExtensionRows(edit, manual.name, fiberUuid, rows);
edit.commit();
```

A changed row set advances the ordinary owner version once in the transaction and clears its approval, just like other content edits. Supplying the same rows in a different order is a no-op. Use the ordinary checked-write/version APIs when modifying an existing owner; this extension API does not introduce a separate optimistic-concurrency counter.

`scanExtensionRowsAtVersion` accepts either an `ObjectVersion` or a `RelationshipVersion`, according to the table's owner kind. A version preceding the field's introduction returns no rows; an unavailable version throws instead of returning current data. Restoring an object restores its extension rows from the selected version. Deleting an owner removes its current editable rows; history-enabled stores retain the earlier states. Immutable computed rows are not silently deleted or rewritten by these owner edits.

Current editable rows remain typed SQL data for querying. To participate in the existing owner-history format, their snapshot is encoded as typed CBOR in the existing content-addressed `geometry_chunks` storage, split into chunks of at most 1 MiB. A reserved `meta.vcobj_extensions` descriptor on the owner references those chunks and a schema fingerprint. The fingerprint describes the extension/table definition rather than a physical table UUID, so relocating a snapshot into an equivalent table does not change its interpretation. Ordinary metadata setters and initial imports cannot forge that descriptor. This is snapshot reuse, not the postponed delta-history redesign, and it adds no extension-history tables or clock. History-off retains the current descriptor/payload but does not archive superseded owner states. Immutable result tables, including the fiber-import tables below, do not create these owner descriptors or snapshots.

The public versioned-row scan follows each selected owner's snapshot through the shared reference reader, for current and historical reads alike. It does not require a matching physical row copy beside the owner. An owner-UUID prefix restricts owner selection; remaining key filters are applied to decoded snapshot rows. This path is not an indexed cross-owner query engine and has not been throughput-benchmarked on large editable extensions. Immutable result scans still use their indexed typed tables directly.

## Python API

Local `ObjectStore` and write-transaction bindings expose `create_extension_table`, `append_extension_rows`, `replace_extension_rows`, `extension_tables`, `extension_rows`, `scan_extension_rows` and `extension_rows_at_version`. Definitions use dictionaries with `name`, `extension`, `format_version`, `columns`, `primary_key`, optional `indexes`, `key_space`, `versioned`, `based_on` and `provisioning`. Column types are `integer`, `real`, `text` and `blob`; cells are `None`, signed 64-bit integers, floats, strings or `bytes`. `extension_uuid()` converts UUID text to 16 bytes and back. An empty `bytes` value remains distinct from `None`.

```python
from vc_object_store import extension_uuid

with store.write() as tx:
    tx.append_extension_rows("ext_fiber_classification", [
        [extension_uuid(fiber_uuid), 0, 0.9],
    ])
matches = store.extension_rows("ext_fiber_classification",
                               key_prefix=[extension_uuid(fiber_uuid)])
```

`extension_rows` materializes its result; use `scan_extension_rows(table, visitor, key_prefix=...)` for streaming. A callback returning `False` stops the scan; `None` continues. `extension_rows_at_version` takes the ordinary version-reference dictionary with `store`, `generation`, `version` and exactly one of `object` or `relationship`.

Both current-row methods also accept `key_filter={column: value, ...}` for equality on any subset of primary-key columns, including non-prefix columns such as a run UUID. Immutable scans push these predicates into SQLite, allowing a declared run index to serve the query without materializing other runs. Only primary-key columns are accepted: filtering mutable payload columns before overlay resolution could incorrectly reveal an older row. Versioned fields apply the same key predicates to the selected owner's snapshot rows. A prefix and a key filter are combined with AND.

## Parallel construction and store_reference reads

Extension payloads use the existing bounded object/edge import stream and are written beside their owning object or edge in the same transaction. They do not require another source scan, a per-object routing directory or cross-shard endpoint reads.

For colocated tables, the entity's binary UUID follows the existing object or edge shard rule. Table descriptors are per-table metadata; physical table UUIDs differ across shards even though they implement the same logical extension. Extension-aware collection reads use the declared routing and dependency order.

`replaceExtensionRows` on an ordinary `ObjectStore` transaction creates a writable override of an inherited owner, then replaces only the requested extension's fields. `ObjectCollection` delegates to that same implementation. Ordinary metadata edits preserve extension snapshot references without copying extension rows or snapshot chunks. The local store registers the referenced table definitions so those snapshots remain interpretable. Versioned table scans follow the snapshot selected by the winning owner's metadata, not a union of individual row keys: unchanged snapshots can be read from the pinned base, a replacement hides all older rows for that owner, an explicit clear stays empty, and a deletion marker prevents fallback to inherited fields. Immutable result tables instead merge by their own primary keys and keep their original input provenance independently of current owner edits. Exact historical reads use `objectAtVersion`, `relationshipAtVersion` and `scanExtensionRowsAtVersion` with the ordinary store/lineage/entity/version references through either API. They share the same resolver as current snapshot reads. They never substitute the current owner for an unavailable version. This is exact-reference retrieval, not collection-wide history enumeration.

`InitialImportOptions::extensions` supplies table definitions before the shard producers run. `InitialObjectRecord::extensions` and `InitialCollectionEdge::extensions` carry optional `InitialExtensionRow` payloads through the existing queues. A colocated table declares `keySpace="objects"` or `"edges"`; its first primary-key column is the owning entity's binary UUID. Initial immutable rows use `versioned=false`.

Result provenance does not become current merely because a core object is overridden or deleted in an overlay. Consumers compare the result's input reference with their intended source state. The generic layer does not propagate millions of stale flags or silently retarget immutable results after edits.

## Compatibility and preservation

The core SQLite schema is 22. An optional extension has its own format version; it is not a fractional SQLite schema version. Readers that do not understand an extension can still use core geometry and edges. Such readers must not infer that extension results describe newly edited inputs.

Operations that preserve complete databases retain optional tables. `exportJson` rejects a store with registered extensions, including projected metadata exports, because that format has no table/row representation. `readObjectCopy` likewise rejects a source reporting extensions rather than silently making a core-only copy. Use a database-preserving copy when the whole store is wanted; extension-aware selected-object copy/export is not implemented. An extension that changes the meaning of core data cannot rely on optional discovery: it needs an explicit required capability or a core format change.

Generic extension read/write methods are currently local C++/Python APIs. `HostedStore`, server transaction operations and hosted extension-row queries have not been added. Hosting a store does not make its optional rows remotely queryable through core object listings. Existing core reads remain usable, but remote consumers must not treat those listings or object-copy operations as a complete extension export.

## Fiber graph solve extension

The shared extension family is `ext_fiber_graph_solve`. Import definitions and row builders are in [`objects/FiberImportExtension.hpp`](../include/vc_object_store/objects/FiberImportExtension.hpp); workflow-result definitions are in [`objects/FiberGraphSolveExtension.hpp`](../include/vc_object_store/objects/FiberGraphSolveExtension.hpp). These are optional typed tables, not additional core-schema columns. Importers create only the applicable source/constraint tables; merely requesting a solver-table definition creates nothing. This supplies storage for workflows without changing or rerunning any solver algorithm.

Formats are table-specific. Source and workflow-result tables use format 1. Native extraction and AFC import publish compact constraint format 4 on core schema 22. The numerical graph reader rejects earlier measurement formats explicitly; a core-schema conversion preserves unknown extension tables but does not reinterpret their payloads.

### Imported observations

| Table | Key/routing | Contents |
| --- | --- | --- |
| `ext_fiber_graph_solve_sources` | `fiber_uuid`, object shard | Original numeric `source_id` and free-form source `annotation` |
| `ext_fiber_graph_solve_constraints` | `edge_uuid`, edge shard | Compact canonical-endpoint observations |

Chunk extraction stages piece-local measurements on the existing shard-writer connections. A TEMP-only staging payload identifies source fibers and exact sample arcs for the final sign transport. Once identical shared-boundary samples determine the along-fiber signs, transfer writes ordinary format-4 rows with aligned normal bytes. Unresolved pieces produce undefined endpoint normals. No piece IDs, correction tables or source-ID directories persist. See [initial import staging](initial-import.md).

The obsolete `normal_frames` workflow role is not exposed by current factories. Legacy extension data remains accessible through generic extension APIs without being silently accepted as a compact numerical graph.

Fiber line geometry and source identity remain in the core objects and source extension. Each accepted observation creates one core edge and one measurement row, without integer-step expansion. The core edge carries UUID endpoints and integer kind; it does not carry XYZ anchors or dependency JSON in this workflow. Adjacency indexes are disabled.

| Column | Encoding | Value bytes |
| --- | --- | --- |
| `edge_uuid` | Binary UUID, primary key | 16 |
| `arc_a`, `arc_b` | Separate little-endian float32 BLOBs, base voxels | 4 each |
| `normal_a`, `normal_b` | Lasagna compact XY plus hemisphere sign | 3 each |
| `connector` | Compact signed unit direction from canonical A to B | 3 |
| `winding_distance` | Unsigned byte; code 0 invalid, otherwise (code−1)/16 turns | 1 |
| `extra` | Explicitly supplied optional JSON | NULL normally |

The maximum winding distance is 15.875 turns. Extraction/import skips larger measurements before quantization; it never clamps them into range. Endpoint arcs use the canonical core A/B order. A normal blob is two unsigned Lasagna hemisphere coordinates plus an int8 sign: 0 undefined, +1 positive Z, −1 negative Z. The shared `vc_reader/compact_normal.hpp` implementation is also used by the Lasagna sampler. The connector supplies ordering direction only, not physical connector length. Coincident endpoints have an undefined connector.

No source constraint ID, source fiber IDs, reversal flag, scores, evidence weight, endpoint XYZ, temporary piece IDs, support diagnostics or winding-distance index is persisted. Persistent identity comes from UUIDs; source catalog mapping remains per fiber in the source extension. Runtime graph indices are dense positions in deterministic edge-UUID order, not persistent identities.

Provisioning is explicitly constructed from compact metadata: known source UUID/checksum, optional counts, units, coordinate space and codec conventions. It is never copied wholesale from configuration or source metadata. Paths, selection lists and lengthy numerical arrays are forbidden. Source annotations are separate opaque source data, not provisioning.

Stored XYZ access fails explicitly; consumers needing source geometry evaluate it separately at the requested arc. Numerical winding and UV consumers do not request or reconstruct missing measurement XYZ. The compact graph reader supports plain, sharded, packed and referenced collections through `ObjectCollection`, resolves overlay ownership, and validates byte lengths, normal signs, canonical UUID order and finite arc values.

Native extraction records the selected cohort in `_fibers` rows under a `constraint_extraction` run, including fibers without observations. These refer to existing core objects rather than copying geometry. `_inputs` and table `basedOn` identify the input snapshot; compact `normal_source_sha256` identifies the normal source.

Measurement provisioning may include `fiber_source` as the standard `{store, generation, commit_sequence}` snapshot pin, preserved unchanged by normal overlays; generic format-4 reads accept its absence, while consumers requiring frozen feature or geometry evidence reject a missing pin explicitly.

Normal orientation writes changed measurement rows into a sealed derived store referencing the input graph. Edge UUIDs, canonical arcs and every non-normal field are retained. Subsequent normal-curve/prior export requires matching normal-source identity, sampling spacing and endpoint validity. Since float32 arc and compact-axis quantization are intentional, it checks transported orientation and reports mean/max angular discrepancies rather than asserting impossible float64 byte equality. An arc rounded upward at the final endpoint is accepted only up to the exact float32 representation of that endpoint and is evaluated at the source endpoint.

Python `CollectionReader.fiber_constraints()` returns the same composed compact graph. Shared winding workflows use `fiber_graph_store.FiberGraph` and UUID-indexed geometry rather than opening SQLite directly. See [fiber storage and extraction](../../vesuvius/docs/fiber_layout.md#storage-and-identities).

### Run results

Every following table name begins with `ext_fiber_graph_solve`. Multiple runs share the original core fibers/edges; `run_uuid` distinguishes results without copying those objects. Create only the tables needed by the stage.

| Suffix | Primary key | Data |
| --- | --- | --- |
| `_runs` | `run_uuid` | Stage, method identifier and explicitly supplied scalar settings |
| `_inputs` | `run_uuid, role` | Input store UUID, lineage UUID, commit sequence and optional earlier `input_run_uuid` |
| `_fibers` | `fiber_uuid, run_uuid` | H/V/broken classification/probabilities, enabled state, aligned normal sign/confidence, integer-winding distribution, selection and U/V anchor |
| `_samples` | `fiber_uuid, run_uuid, sample_index` | Arc position, optional fiber direction and sheet normal, explicit normal validity |
| `_hypotheses` | `fiber_uuid, run_uuid, winding` | Probability at an integer winding hypothesis |
| `_widths` | `run_uuid, winding` | Width in base voxels for each winding |
| `_positions` | `fiber_uuid, run_uuid, layer, hypothesis, node_index` | Arc position, optional constraint UUID/source endpoint, continuous U in turns and/or U/V in base voxels |
| `_relations` | `edge_uuid, run_uuid, layer, evaluation_index` | Inclusion, endpoint hypotheses, fulfillment/sign result, probabilities, meeting weight, required/realized step, winding/U/V residuals and loss |
| `_metrics` | `run_uuid, iteration, scope, name` | One floating value or integer count, with units |

Fiber-owned tables route with object UUIDs and relation results with edge UUIDs. Run-oriented tables are not object aliases; each run/input/width/metric record is stored once in a chosen database and found by ordinary collection scans. Run-to-fiber and run-to-edge indexes support lookup from the result as well as from its source entity.

For numerical stage setup, `CollectionReader.fiber_result_columns(run_uuid)` returns owned, read-only NumPy columns in fiber-UUID order: `fiber_uuid` (`uint8[N,16]`), contiguous `uuid_ascii` bytes (36 bytes per UUID), `values` (`int64[N,5]`) and `valid` (`uint8[N,5]`). `columns` names the five integer fields: family, enabled, normal sign, selected winding and component. A false validity byte means SQL NULL; zero in the value buffer does not imply a resolved label. The native reader selects only the requested run and applies the same referenced/sharded overlay semantics as ordinary extension scans. These are result projections, not replacements for the run's recorded source/input references.

`fiberGraphRunRow` accepts a stage/method identifier and a map of boolean, integer or finite real settings. It does not accept an arbitrary configuration/report object. `fiberGraphInputRow` stores each input pin as a typed row. A later run can reference an earlier result by its store/lineage/commit plus `input_run_uuid`; this does not overwrite the earlier run or update its source pins. Role names distinguish multiple inputs to one run. Per-run input pins are not repeated in every table's provisioning.

`fiberGraphSolveRow` constructs rows by named fields, fills omitted nullable values with NULL, and rejects missing required fields, unknown fields, wrong types and invalid scalar/vector conventions. UUID values are binary. Family codes are H=0, V=1, broken=2; NULL means unresolved. Normal sign is −1 or +1, flags are 0/1 and probabilities lie in [0,1]. Invalid normal samples can omit their XYZ values; a valid normal requires all three coordinates. These builders validate representation; they do not normalize probabilities, invent normals or perform optimization. Hot bulk paths can supply the same declared column order directly.

`p_h`, `p_v` and `p_broken` are joint classification probabilities (not H/V probabilities conditional on being enabled). `selected_probability` and `_hypotheses.probability` are winding probabilities conditional on being enabled; `joint_probability` includes disablement. An explicit `enabled` flag records the output selection rather than imposing a probability threshold in storage. Sample directions and normal axes are in the fiber's core XYZ coordinate frame. `normal_sign` is the whole-fiber multiplier applied to the locally aligned sample axes; `sign_probability` is the probability of multiplier +1, not an inward/outward label. A downstream stage can reference the sampling/classification run through `_inputs` instead of repeating its rows.

Metrics use exactly one of `value` or `count`. For example, a mean residual and its sample count are two separately named metrics, not a floating value with an overloaded count field. Missing metrics stay absent/NULL rather than pretending to be zero.

### Winding and coordinate conventions

For probabilistic solvers, `winding_mean` is **N**, the continuous center of a distribution over integer hypotheses **k**. It contributes no continuous phase. `anchor_u_turns` and position `u_turns` are **U**, excluding k; U may lie outside [0,1] and `U mod 1` is the phase. A realized hypothesis has winding coordinate **k + U**, not N + U. Thus N=5.5 and U=2.3 can give equally probable positions at 7.3 and 8.3; they are not a single physical position at 7.8. There is no additional half-turn H/V position offset. Continuous graph solvers can preserve their unrounded winding estimates in `winding_mean` and omit distribution/selected-hypothesis fields they do not produce; this storage API does not manufacture integer assignments or probabilities.

Layer codes are Shared=0, Intermediate=1, Final=2 and Refined=3. Shared positions use `hypothesis=0` as a sentinel meaning “shared over hypotheses”; it does not force k=0. Select the actual k from the winding result when realizing the shared position. Intermediate/final/refined rows may have different U for different k, but always store U excluding k. When `u_vx` is present it is the actual unwrapped voxel coordinate, including the width-mapped k; `v_vx` is absolute. A voxel-only final layout may omit `u_turns` rather than fabricate a conversion. A constraint node supplies both `edge_uuid` and `endpoint` (source A=0/B=1), or neither.

Schemas default to immutable result rows. For manually editable fields, an owner-bound schema can be cloned under its own table name with `versioned=true`, then written with `replaceExtensionRows` to use ordinary object/edge versions and history. Run/width/metric tables have no such owner; do not mark them versioned without defining an appropriate owner-bound schema.

Python uses the same native definitions and row constructors, for example:

```python
from vc_object_store import (
    extension_uuid, fiber_graph_solve_table, fiber_graph_solve_row,
    fiber_graph_run_row, fiber_graph_input_row,
)

# source_state is the source snapshot actually read by the computation.
source = dict(store_uuid=source_state["store"], generation=source_state["generation"],
              commit_sequence=source_state["commit_sequence"])
with results.write() as tx:
    for role in ("runs", "inputs", "fibers"):
        tx.create_extension_table(fiber_graph_solve_table(role))
    rows = [
        fiber_graph_run_row(run_uuid, "winding", "irls", {"iterations": 100}),
        fiber_graph_input_row(run_uuid, "graph", source),
        fiber_graph_solve_row("fibers", dict(
            fiber_uuid=extension_uuid(fiber_uuid), run_uuid=extension_uuid(run_uuid),
            enabled=1, winding_mean=5.3,
        )),
    ]
    for row in rows:
        tx.append_extension_rows(row["table"], [row["values"]])
```

Use a new run UUID for another independently saved result/checkpoint. Metric progression can be appended under the same run UUID at distinct iterations. The definitions provision the full data path; existing solver/checkpoint producers are not automatically migrated to these new tables.

## Validation

The regression suite covers typed values (including empty BLOB versus NULL), registration and transactional stamps, owner-versioned checked writes, history on/off, restore, multi-chunk snapshots, database-copy preservation, store_reference clears/deletions, and plain/packed sharded imports. Fiber graph extension tests cover independent runs, input references and shared native/Python definitions. Compact provisioning builders reject source/configuration dumps. See the [build and test instructions](../README.md#build-and-test).
