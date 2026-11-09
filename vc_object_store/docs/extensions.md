# Typed extension tables

Extensions add ordinary, typed SQL tables without adding domain-specific columns to core objects or relationships. Core schema 21 owns table registration, common input-state references and transactional change stamps. Each extension owns its column layout, indexes, format version and small provisioning JSON object. Existing object/edge UUIDs remain the references: results do not require duplicate geometry, proxy objects or an attachment edge per row.

Use this for measurements and derived results that generic viewers need not understand, such as fiber-constraint distances, classifications or winding runs. Keep ordinary geometry, endpoint anchors and graph relationships in the core tables. Arbitrary `meta` JSON remains available, but large numerical result sets belong in typed columns rather than JSON blobs.

## Metadata and identity

The optional `extension_tables` registry has one descriptor per physical extension table, referring to its existing `table_directory` UUID. Extension tables use names beginning with `ext_`. SQLite's table schema already describes columns and indexes; the registry does not duplicate every row's layout or create a per-object directory.

| Information | Meaning |
| --- | --- |
| Extension identifier | The consumer-defined extension family |
| Extension format version | The interpretation of this extension's data, independent of core schema 21 |
| Physical table UUID/name | The table instance registered in the existing directory |
| `basedOn` | Common input-state references for the rows in this table |
| `lastModifiedCommit` | Destination-store commit which last wrote this table through the extension API |
| `provisioning` | Small extension-owned JSON object: units, field conventions, source-format identity or processing configuration |
| `keySpace` | Optional colocation with the core `objects` or `edges` UUID routing rule |
| `versioned` | Whether manual edits participate in their owning object/edge's versions and history |

Numerical data is stored as `INTEGER`, `REAL`, `TEXT` or `BLOB` columns. UUID fields use binary 16-byte values, matching the rest of the store. Primary keys and optional indexes are ordinary SQL indexes; no per-row JSON parsing is needed for indexed measurement lookups. NULL is distinct from zero.

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

For editable fields, set `versioned=true` and `keySpace="objects"` or `"edges"`. The first primary-key column must be the owner's binary UUID; additional key columns allow multiple rows per owner. `ObjectStore::replaceExtensionRows(write, tableName, ownerUuid, rows)` replaces that owner's complete row set, not the whole table. An empty row set removes those fields. The direct store operation requires a local owner; the collection operation described below handles localization from linked stores.

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

## Parallel construction and linked reads

Extension payloads use the existing bounded object/edge import stream and are written beside their owning object or edge in the same transaction. They do not require another source scan, a per-object routing directory or cross-shard endpoint reads.

For colocated tables, the entity's binary UUID follows the existing object or edge shard rule. Table descriptors are per-table metadata; physical table UUIDs differ across shards even though they implement the same logical extension. Extension-aware collection reads use the declared routing and dependency order.

`ObjectCollection::replaceExtensionRows(write, tableName, ownerKey, rows)` localizes an inherited owner and its manual fields before editing. Ordinary metadata localization also preserves those fields. Versioned table scans use the winning owner, not a union of individual row keys: a replacement hides all older rows for that owner, an explicit clear stays empty, and a deletion marker prevents fallback to inherited fields. Immutable result tables instead merge by their own primary keys and keep their original input provenance independently of current owner edits. Historical version reads remain the direct `ObjectStore` API, not a cross-collection history query.

`InitialImportOptions::extensions` supplies table definitions before the shard producers run. `InitialObjectRecord::extensions` and `InitialCollectionEdge::extensions` carry optional `InitialExtensionRow` payloads through the existing queues. A colocated table declares `keySpace="objects"` or `"edges"`; its first primary-key column is the owning entity's binary UUID. Initial immutable rows use `versioned=false`.

Result provenance does not become current merely because a core object is overridden or deleted in an overlay. Consumers compare the result's input reference with their intended source state. The generic layer does not propagate millions of stale flags or silently retarget immutable results after edits.

## Compatibility and preservation

The core SQLite schema remains 21. An optional extension has its own format version; it is not a fractional SQLite schema version. Readers that do not understand an extension can still use core geometry and edges. Such readers must not infer that extension results describe newly edited inputs.

Operations that preserve complete databases retain optional tables. `exportJson` rejects a store with registered extensions, including projected metadata exports, because that format has no table/row representation. `readObjectCopy` likewise rejects a source reporting extensions rather than silently making a core-only copy. Use a database-preserving copy when the whole store is wanted; extension-aware selected-object copy/export is not implemented. An extension that changes the meaning of core data cannot rely on optional discovery: it needs an explicit required capability or a core format change.

Generic extension read/write methods are currently local C++/Python APIs. `HostedStore`, server transaction operations and hosted extension-row queries have not been added. Hosting a store does not make its optional rows remotely queryable through core object listings. Existing core reads remain usable, but remote consumers must not treat those listings or object-copy operations as a complete extension export.

## Fiber graph solve extension, format 1

The shared extension family is `ext_fiber_graph_solve`. Import definitions and row builders are in [`objects/FiberImportExtension.hpp`](../include/vc_object_store/objects/FiberImportExtension.hpp); workflow-result definitions are in [`objects/FiberGraphSolveExtension.hpp`](../include/vc_object_store/objects/FiberGraphSolveExtension.hpp). These are optional typed tables, not additional core-schema columns. Importers create only the applicable source/constraint tables; merely requesting a solver-table definition creates nothing. This supplies storage for workflows without changing or rerunning any solver algorithm.

### Imported observations

| Table | Key/routing | Contents |
| --- | --- | --- |
| `ext_fiber_graph_solve_sources` | `fiber_uuid`, object shard | Original numeric `source_id` and free-form source `annotation` |
| `ext_fiber_graph_solve_constraints` | `edge_uuid`, edge shard | Typed extraction measurements and diagnostics |

Fiber line geometry, names, family, length/bounds and coordinate bindings stay in their existing core representation. Each source constraint produces exactly one ordinary core edge carrying its endpoint UUIDs, kind and XYZ anchors, plus one `ext_fiber_graph_solve_constraints` row. A measured distance such as 1.3 remains one REAL value; import does not expand it into separate integer-step edges or duplicate the endpoints.

`ext_fiber_graph_solve_constraints` contains these fields:

| Group | Columns |
| --- | --- |
| Source identity/order | `source_id`, `source_fiber_a`, `source_fiber_b`, `source_reversed` |
| Base-point arc positions | `arc_a`, `arc_b` |
| Classification/evidence | `parallel_score`, `perpendicular_score`, `evidence_weight` |
| Lasagna measurements | `las_winding`, `parallel_las_winding` |
| Endpoint sheet normals | For each of `normal_a` and `normal_b`: `_valid`, `_x`, `_y`, `_z`, `_reason` |
| Local relation diagnostics | `piece_a`, `piece_b`, `closest_distance`, `hard_continuity`, `artificial_tile_seam`, `allow_measurement_disable`, `local_event`, `local_orientation` |
| Parallel support/fit | `support_a_begin`, `support_a_end`, `support_b_begin`, `support_b_end`, `parallel_correspondence_samples`, `parallel_advance_residual`, `parallel_connector_tangent_dot`, `parallel_connector_length_change`, `parallel_connector_direction_change`, `parallel_limit_hit_fraction` |
| Signed/orientation evidence | `signed_winding_delta`, `perpendicular_normal_alignment`, `winding_normal_component`, `signed_parallel_winding_delta`, `parallel_normal_alignment` |
| Unrecognized or inconsistent source extras | Nullable `extra` JSON text, not the ordinary measurement payload |

Arc/length values use base voxels; winding distances use turns. Supported numerical fields are nullable typed scalars, not zero-filled for absent measurements. Normals remain sheet normals; they are not inserted into the core anchor's fiber-tangent direction field. Source normal validity and reason are retained separately from coordinates.

Core edges canonicalize their endpoint UUID order. Measurement fields deliberately retain the source A/B convention: `source_reversed=1` means source A/B is opposite to core edge A/B. Consumers map endpoint fields accordingly. The importer does not guess sign inversions for measured scalars merely because UUID ordering changed.

The source's `measurement` JSON repeats some scalar and XYZ data. Equal duplicate copies are omitted from the extension payload; differing copies are retained in `extra`, together with genuinely unknown fields. Supported diagnostics and normals are typed columns. The fiber's original free-form annotation remains opaque JSON text because this importer does not impose a schema on arbitrary source annotations.

Provisioning is explicitly constructed from compact typed inputs, not copied or filtered out of a source metadata/configuration tree. `FiberSourceInfo` supplies the source UUID when known, optional SHA-256 and optional fiber/point counts. `FiberConstraintSourceInfo` supplies the input fiber-source UUID when known, optional constraint-file SHA-256 and optional constraint count, plus fixed units/order conventions. AFC has no independent file UUID: its AFV UUID identifies the input fibers, not the constraint file. Unknown identity is omitted, never guessed. Filesystem paths, reports, selection-ID lists and nested source configuration are not stored in provisioning. Per-fiber source annotations are a separate opaque source field, not provisioning.

The source table indexes `source_id`; the constraint table indexes `source_id` and `las_winding` in addition to its edge UUID primary key. Existing core adjacency indexes still find relationships from a fiber; the extension row is then looked up by edge UUID.

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

## Validation and historical matched import check

Regression coverage includes typed values (including empty BLOB versus NULL), registration and transactional stamps, owner versions/checked writes, history on/off, restore, binary snapshots larger than one chunk, database-copy preservation, linked clears/deletions, and plain/packed multi-shard imports. Python uses the same native row/schema builders. Extension-specific hosted operations are not exposed yet; metadata-only JSON export and object-only copy explicitly reject extension-bearing stores rather than dropping their data.

The compact-provisioning cleanup and workflow-table additions passed 51 native tests and 136 Python tests; the optional SSH integration test was skipped. The no-adapter build also passed all 45 executed tests. The new `object_store_fiber_graph_extension` regression covers independent runs, typed values, input references, and linked plain/packed sharded storage. Python exercises the same definitions and rejects wholesale source/configuration dictionaries in the compact builders. The local argument-free verification command is `bash /home/hendrik/villa2/local/scripts/run_vcobj_tests`; portable build/test instructions remain in the project README.

The following numbers were measured **before** the compact-provisioning cleanup, extension-family rename and workflow-table additions. They are not current performance results; those changes have not been rebenchmarked here. The historical matched input contained 50,000 fibers, 15,486,987 XYZ points and 300,000 constraints. Build: GCC 13.3, RelWithDebInfo, SQLite 3.45.1, Linux amd64. Packed output used RAM-first storage, 512 MiB SQLite caches, a 4 GiB resident budget and one exporter to tmpfs. Prepared ingestion excluded source decoding; streaming ingestion included parallel decoding. Final persistence was separate. Each 32-worker result was three trials; the one-worker control was one trial. Every benchmark invocation had a 30-second hard limit. Use the shareable `vc_bench_prepared_import CONFIG.json` and `vc_object_import_afv CONFIG.json` workflows with matching input/settings for a new comparison.

| Path | Workers | Earlier ingestion median (s) | Pre-cleanup typed ingestion mean / median / min–max (s) | Earlier → pre-cleanup persistence median (s) |
|---|---:|---:|---:|---:|
| Prepared records | 1 | 4.252 | 3.666 / 3.666 / single trial | 0.394 → 0.178 |
| Prepared records | 32 | 0.229 | 0.352 / 0.352 / 0.345–0.359 | 0.315 → 0.288 |
| Source streaming | 32 | 0.918 | 0.898 / 0.895 / 0.885–0.915 | 0.330 → 0.280 |

That pre-cleanup implementation incorrectly copied an approximately 8 MB source selection list into each physical table's provisioning, adding fixed setup/storage overhead. This has been removed; current provisioning is explicitly constructed from compact typed fields. Historical output sizes were approximately 0.785 GB for one partition and 1.145–1.150 GB for 32 partitions, versus 1.199 GB and 1.288 GB before typed extensions; these are not post-cleanup size claims. Prepared readback checks, including typed rows and exact geometry bytes, ran outside the timed interval. These were bounded tmpfs checks, not full-volume runtime or physical-disk throughput measurements; the unequal fiber/point/constraint fractions do not support multiplying total time by one full-volume scale factor. macOS/arm64 were not tested in that run.
