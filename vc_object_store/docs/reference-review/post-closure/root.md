# Post-closure root review

Frozen production snapshot: `de3afb065`. Implementation commits: `b1f242597` (extension reference decoding), `59910fb54` (shared shell refresh), `75f9b7d88` (provenance). This review writes evidence only; no production fixes.

## Binding and facade inventory

| File | Entry points / disposition |
| --- | --- |
| `python/objects.cpp` | Reviewed reference, selection, local/hosted/session read/write adapters, selected-input and surface boundaries; native codecs/resolvers remain authoritative. Saved selection position conversion is part of the selection-codec candidate in the core report. Detector keyword adaptation forwards native parser input; it is not a second detector policy. |
| `python/StorageBindings.hpp` | Reviewed object/edge/geometry conversions and store/write-transaction lifetime adapters. Native writes own guards, savepoints and mutation semantics; Python typed-value conversion is appropriate. Shared metadata JSON serialization is a language boundary, not a separate persistent schema. |
| `python/ObjectReadBindings.hpp` | Reviewed graph projections, graph lifetime/array owners and selected versus current read paths. Native field validation is reused; current graph reads retain a transaction, selected graph reads retain selected records. Immutable geometry-array reads do not authorize edits. |
| `python/TypedGeometryBindings.hpp` | Reviewed current/exact factory dispatch, native selected reference decoding, view references and array/geometry adaptation. Current UUID versus exact-reference dispatch is an explicit boundary; no consumer ownership inference. Pure array conversion/numerical geometry methods are not independently re-proved. |
| `python/ExtensionBindings.hpp` | Reviewed object/relationship exact reads and owner-versioned mutation delegates. Both exact branches now use native codecs. Type/name conversion into native extension structs is language adaptation; no pre-existing native JSON-table decoder is bypassed or duplicated here. Shared native table/row validation remains authoritative. |
| `python/SessionBindings.hpp` | Reviewed attachment lifecycle/copy/import/events. Delegates to shared session; process attachment identity is not a replacement object version. |
| `python/GeometryRegionBindings.hpp` | Reviewed native region request/cancellation and selected result conversion. Full object reference and retained mapping accompany each result. Native query owns filtering and placement. |
| `python/AfvBindings.hpp` | Reviewed native AFV source iteration adapter. Tuple is source fiber data, not a versioned vcobj reference. Native source reader owns source-format behavior. |
| `python/CMakeLists.txt` | Build/provisioning only; not runtime store resolution. No setup/install executed. |
| `python/vc_object_store/__init__.py` | Native exports only. |
| `python/vc_object_store/selected.py` | Value facades retain selected native record/revision; no late current lookup. |
| `python/vc_object_store/locations.py` | Local versus hosted attachment/API adapters; limited query support is explicit rather than a second filesystem resolver. Exact reads/selected inputs delegate native operations. |
| `python/vc_object_store/surfaces.py` | Grid/array adaptation and surface I/O through native session. Raw numbered selector reaches native validation; complete reference stays unchanged. |
| `python/vc_object_store/documents.py` | Native attached-document/point-collection APIs, no local object placement or history interpretation. |
| `python/vc_object_store/derivation.py` | Explicit application metadata policy, not a copy/restore resolver; managed descriptors are refused or deliberately omitted, with source reference retained. |

## Cross-lane traces

- Checked `StoreReader.cpp` current/exact selection, locator decoding and selected-input snapshot validation against shared reference/position codecs. No new exact-read current-head dependency found in traced paths.
- Checked `Selections.cpp` encode/decode and Python `selectionRecordFromPy`: saved selection commit positions bypass the shared strict position codec. This is merged into the core candidate, not counted again as a separate binding bug. Astra independently reproduced fractional input becoming an integer; selected object references themselves remain complete.
- Checked `Snapshots.cpp` metadata import against identity/lifecycle rules: idempotent/conflicting live-record checks use live-only lookups before low-level insertion. Astra confirmed same-kind object/edge resurrection locally with both history policies; cross-kind tombstones still reject and referenced behavior is source-traced. The import atomicity savepoint does not by itself enforce reserved deleted identities. This is the single shared identity-policy finding PC-A01, not a request for more per-import directories.
- Revisited the newest shell notification/full/incremental/merge/display chain and provenance emitter. Their intended role boundary is retained: shell is displayed/refreshed but excluded from fitted patch endpoint selection. No new regression established in the committed three fixes.

## Limits

This is a per-file store-sensitive source/caller trace, not a claim to have proved every numerical method, integer cast, network interleaving or allocation failure in these translation units. Other native/storage/consumer files are covered by the independent lane ledgers. The approved review runner is run on the frozen source; runtime outcomes are recorded in the overview after completion. No new dependency, schema, API implementation, full-volume run or interactive GUI/macOS test is added by this review.

## Closing initially unassigned native entries

The core inventory initially exposed 25 unassigned entries rather than silently claiming them clean. The formats and spatial/linking lanes revisit 18 of those at the store-policy boundary. Root dispositions for the remaining seven follow; this does not claim fresh numerical/codec proof.

| File | Store-sensitive disposition |
| --- | --- |
| `src/CMakeLists.txt` | Build/source ownership reviewed: public geometry JSON dependency remains exported; formats/objects/linking boundaries and sources are wired into their corresponding native libraries. No runtime reference/schema fallback here. |
| `src/Uuid.cpp` | Shared UUID-v5 generation/namespace hashing helper. Canonical textual UUID parsing delegates shared binary parser; no alternate store/object routing or record identity directory. Cryptographic implementation and entropy properties are not re-audited. |
| `src/MemMap.cpp` | Portable read-only file mapping helper, not a vcobj store resolver; caller passes a physical source already selected. OS handle/map lifetime handling inspected at boundary. No allocation/fault-injection proof. |
| `src/Tiff.cpp` | Standalone TIFF read/write/validity helpers for external format adapters; no ObjectVersion/history/collection resolution. No new runtime schema path. Image layout/decompression algorithms excluded from this vcobj-policy review. |
| `src/TrackCrossings.cpp` | In-memory coordinate/offset input to crossing/CSR geometry computation; no database access or selected-version reconstruction. Internal track ordinals are not replacement persisted UUIDs. Parallel numerical algorithm correctness/performance not re-proved. |
| `src/CtSheetNormal.cpp` | External CT/normal sampling using vc_reader and an independent sparse normal-pack format, not a vcobj metadata/history resolver. Queries receive physical volume sources and coordinates explicitly. Sampling/normals model behavior excluded from the store review. |
| `src/UmbilicusFile.cpp` | External text/JSON source parsing and frame interpretation, then import through `setVolumeUmbilicusRequest` plus ordinary checked session submission. Optional external file stamps are not support for an obsolete vcobj schema or an incomplete internal object handle. Source-format/numerical scaling policy not redesigned. |

These classifications are explicit evidence for store-boundary scope, not a claim that every independent I/O algorithm in the repository has been exhaustively verified. No additional concrete store defect was established by these traces.
