# Targeted native codec class sweep

Frozen source: `d13d33629c3ccee94fa61a9ef873ec82ea0b9db9`. Read-only production
review before fixing the three next-iteration findings. No production changes,
dependency installation or build by this lane. Contracts: repository AGENTS,
review contract and single versioned-reference contract, with reading/coherent
API constraints retained from the full review. The two classes here are (1)
obsolete vcobj reference repair/alternate old reference forms, and (2) acceptance
of non-array JSON for fields whose representation is an array. Numeric scalar
validation and arbitrary metadata schema design are not new tasks in this sweep.

## Boundary rule and discovery

An array needs runtime type enforcement at its external/persisted decode boundary,
not another guard on every subsequent loop. A typed native vector, a checked JSON
array, or data already admitted by its owning schema can be iterated normally.
Debug assertions may express internal assumptions but are not substitutes for
external validation and are not themselves findings. Optional fields deliberately
allowing null keep that policy. Metadata maps are not reclassified as arrays.

Discovery covered `vc_object_store/src`, public headers and native Python bindings:
JSON parsers/CBOR readers; range iteration, `.items()`, typed vector conversions;
`is_array` gates; `legacy`/`older`/`obsolete`/`compatib` branches; reference,
version and source-revision parsing. Candidate hits were traced to producers and
public or persisted-data ingress. Qt/application consumers and Python reference
flows have complementary lane reports, adjudicated separately below. This is a
finite class sweep, not a fresh proof of numerical algorithms or platform locks.

## CS-A01 — expanded missing declared-array ingress validation

**Low, pre-existing class; expands the NX-C01 seed rather than counting it again.**
Native nlohmann range iteration traverses object values and treats null as an
empty range. A loop therefore does not enforce a declared JSON list. Multiple
unrelated decode boundaries rely on that iteration alone. Complete reference
validation inside each element does not validate the outer list. Normal typed
producers emit arrays; the defect is malformed-input admission, not proof that
ordinary producers lose data or that complete selected identities are replaced.

The table enumerates the exact declared fields and the boundary responsible for
them. Sibling loops are listed to prevent an incomplete fix; they are **not**
separate defects requiring duplicate per-loop checks. Runtime evidence is limited
to the fixed runner probe described below; all other rows are source-established.

### Public command, metadata and wire boundaries

| Owning ingress / field | Producer and affected siblings | Concrete effect / limits |
| --- | --- | --- |
| `src/Selections.cpp::decodeSelectionRecord`: header `transforms`, `origins` | `encodeSelectionRecord` emits vectors; Python `decode_selection_record` delegates native. | Known NX-C01, not new. Object values become list entries; null becomes empty. Per-transform complete refs and per-origin pair arrays are already checked. Count scalar fixed at this baseline. |
| `src/Commands.cpp::transactionRequestFromJson`, raw `ObjectStore::submit`: root `operations` | `requestJson` emits array. `requestHeader`, `payloadHashesIn`, `transactionJsonPayloadHashes`, `transactionJsonHasPreparedOperations`, and raw submit/replay paths need one common admission policy, including missing-payload paths. Python local/hosted/session submit adapters parse through the shared native codec; server payload/preparation discovery also uses the raw JSON helpers. | A named object containing a valid operation is interpreted as a list of its values; null becomes an empty request. Ordinary guards and mutation validation still execute. No authorization bypass asserted. |
| `Commands.cpp::relationshipsFrom`: operation `relationships` | `Encoder::relationships`; called by put geometry, edit geometry, rectangle edit, set surface channels. | Object-valued decisions/null admitted as vector/empty. Actual relationship revalidation still owns semantic safety. Validate at shared relationship-list decode, not each operation consumer. |
| `Commands.cpp::dataFrom`: `data.arrays` | Encoder geometry data; put geometry, update geometry arrays, nested ingest items; `collectPayloads` scans same serialized arrays before bytes are fetched. | Object descriptors admitted by value. Typed shape vectors/payload hashes remain checked. An empty/null geometry array may fail later geometry semantics; that later error is not array-shape enforcement. |
| `Commands.cpp::operationFrom`: `edit_geometry.edits`, `set_surface_channels.channels`, `ingest_geometry.items`, `record_catalog_transforms.registrations` | Matching Encoder overloads emit arrays; payload discovery visits edits/channels/items and nested data arrays. | Same value-iteration admission. Schema validation must be available before payload lookup and shared with materialized decode; do not fix only the mutation path or duplicate parsing policy in Python. |
| `src/Snapshots.cpp::ObjectStore::importJson`: `objects`, `edges` | `exportJson` emits both arrays. Object preflight and insertion loops share the same parsed document; Python `import_json` delegates native. | Object-valued records/null are accepted as records/empty import. Existing identity/tombstone/lifecycle/content checks and savepoint remain effective. One root-array check before all import loops is sufficient. |
| `src/internal/Anchors.cpp::dependenciesFromJson`: `points` | `dependenciesToJson` emits array. `anchorFromJson`, record/command edge parsing, Python `anchorFromPy`, and retained relationship dependency reads share this helper. | `{}` behaves as no points; object values containing valid point descriptors can become points. `index` already explicitly requires array; `channels`/`metadata_keys` use typed vector conversion. Python `omitNullFields=true` deliberately removes null optional fields: that null policy is not a defect. |
| `src/Wire.cpp::transactionConflictFromJson`: `details` | `transactionConflictToJson`; hosted error decode. | Null/object values interpreted as zero/multiple conflict details; scalar detail fields still parsed. |
| `Wire.cpp::changePageFromJson`: `commits`, each commit's `changes` | Native change-feed serializer, hosted feed client and session freshness processing. | Malformed null collections can appear empty; object values appear as commits/changes. Cursor/commit semantics not independently bypassed by this finding. Check nested list schema once at page decode. |
| `Wire.cpp::geometryRevisionFromJson`: `arrays` | `geometryRevisionToJson`; hosted geometry descriptors and cache rehydration. | Missing array shape, though individual descriptor shape uses typed vector conversion. |
| `Wire.cpp::{objectQueryResultFromJson,edgeQueryResultFromJson}`: envelope `objects`, `edges` | Matching query serializers; HostedStore and cache/query adapters. | Null/object values interpreted as result vectors. `unwrap` itself handles scalars and objects too, so do not universally require arrays in `unwrap`; use a typed array extraction for array-result callers. |
| `Wire.cpp::{objectHistoryFromJson,objectsAtVersionsResultFromJson}`: `revisions` | History and exact-read serializers; hosted/cache readers. | Array shape missing. Null *entries* in exact lookup remain legitimate unavailable-version results; null *container* is distinct. Some callers enforce response length, not container type. |
| `Wire.cpp::relationshipQueryResultFromJson`: `relationships`; `selectedInputsFromJson`: `inputs.objects`, `inputs.relationships` | Matching result serializers, HostedStore relationship/coherent-input reads and HostedSession relationship cache. | Element refs remain strict; a missing/malformed collection shape can be mistaken for empty membership. Shared boundary checks serve both clients and cached answers. |
| `Wire.cpp::regionQueryResultFromJson`: `result.runs`; `proximityQueryResultFromJson`: `hits` | Matching result serializers and hosted spatial clients. | Null/object result collections admitted. Point/grid vectors within runs/hits already use typed conversions. |
| `src/client/HostedStore.cpp::submit`: 422 response `missing` | Server constructs missing-payload response from a native vector. | Object values or a scalar string can be interpreted as hash entries; repeated 422 retry limit still applies and unknown hashes reject. This HTTP response decode is separate from transaction command parsing. |

### Persisted schema-owned and declared document boundaries

These rows are source-only. Normal writers generate the declared arrays. Checksums,
schema hashes, size checks and native type validation often reject unrelated
damage; they do not establish JSON container shape. No raw database/archive
tampering, synthetic packed file or damaged checkpoint was executed here.

| Owning ingress / field | Siblings that should reuse the admitted value | Concrete scope / exclusions |
| --- | --- | --- |
| `src/PackedStore.cpp::readDirectory`: `databases` | Packed VFS open/access, root/member inspection, finalization recheck and compaction use this reader. `finalize` writes a JSON array. | An object keyed arbitrarily with valid member values passes the iteration; null later fails missing-root check, so no accepted empty pack claim. Preserve extent/count/overlap/checksum checks. `offsets` typed vector already enforces array. |
| `src/ExtensionHistory.cpp::readSnapshot`: descriptor `chunks` | Exact object/relationship extension scans, replacing/restoring rows; `currentExtensionChunks` calls readSnapshot before enumerating the descriptor again. | No outer array check; one hash string/object values can supply chunks. Encoded CBOR rows, each row and each cell already explicitly require arrays. Validate descriptor once; no additional guard is needed in the downstream `currentExtensionChunks` loop after that validation. |
| `src/geometry/GeometryDelta.cpp::loadGeometryManifest` / delta descriptor decode: `delta.chunks` | `encodeGeometryDelta` writes array; `decodeGeometryDelta` and `GeometryReader` reconstruction, hash updates and changed-chunk selection all consume it. | Delta ingress currently validates hash/version/result, not this list type. Valid-value object form is structurally admitted; payload byte accounting still applies. Central manifest/delta schema admission should establish the invariant for later loops. Geometry manifest `arrays` is a name-to-descriptor **map**, not part of this array issue. |
| `src/geometry/SequenceMetadata.cpp::unpackAnnotation`: `annotation_chunks`; `metadataPayloads` descriptor reads | `packAnnotation` emits array; selected annotation materialization and metadata-payload closure/copy consume descriptors. | Hash/byte-length checks remain. Shared annotation descriptor admission should cover both read and closure traversal; metadata root/bucket maps are not arrays and stay excluded. |
| `src/DraftCheckpoints.cpp::DraftDirectory::load`: `pending` | `write` emits array; session recovery consumes loaded pending requests. | Null/object values admitted; no unknown-outcome recovery loss reproduced. Drafts itself is deliberately an object map. Existing base refs decode through shared full-reference codec; draft revision counters are not ObjectVersion handles. |
| `src/AttachedDocument.cpp::attachDocument`: supplied `document[layout.listKey]` | `DocumentLayout` public header explicitly calls it a list of point records; Python session attach_document delegates here; export rebuilds the document list. | Object values can silently become point records. Absent list deliberately defaults empty; arbitrary remainder of document is metadata and must not receive list checks. This is a declared-list adapter, not obsolete source-version support. |
| `src/catalog/CatalogStore.cpp::catalogReference`: catalog metadata `files` | Placeholder publisher emits `files` array; fetch metadata decodes into `CatalogReference.files`. | Object-valued file records iterate as list; absent files defaults empty. Do not constrain unrelated catalog/source metadata or alter external-format compatibility. |
| `src/formats/FiberJson.cpp::decodeFiberJson`: source `branches`, `adjacent_branches`; shared source-topology ingestion | `FiberInput.cpp` normalizes source peers, `fiberBatchEndpoints` discovers external endpoints, `imports/SourceRelationships.cpp::importSourceRelationships` creates branches. Native fiber export emits both as arrays. | Source boundary leaves topology collections unvalidated, so object values/null reach the same branch interpretation. Check topology schema once at ingress; not every later loop. Supporting external fiber versions remains legitimate and is not the defect. Direct prepared source ingestion must share topology admission rather than depend solely on file loading. |
| `src/formats/PointCollectionInput.cpp::collections` / shared source-topology ingestion: per-point `links`, collection `windings_linked` | PointMap decoding retains arbitrary annotations; SourceRelationships consumes these specific topology keys. Native export emits lists. | Point-map and collection root **maps** are legitimate and already checked. These known topology lists are not checked before object/null value iteration; target identity validation remains. Do not constrain unrelated point annotations. Source-only; no malformed source fixture executed. |

Minimal class-wide direction: put representation validation in shared native
decode/schema helpers, reuse it in raw/prepared/replay and cached/result siblings,
and retain parsed native values downstream. Add malformed object/null controls
alongside valid array fixtures at each owning boundary, preserving explicitly
nullable fields. No new reference format, schema version, SQL proxy or repeated
consumer-side `is_array` checks are justified.

## Legacy-reference class disposition

The confirmed ordinary-runtime repair remains **NX-A01**, the existing
`Wire.cpp::relationshipFrom` clearing of any array-prefixed endpoint archive
string after complete endpoint references decode. Both relationship-result and
selected-input-result decoders/cache use it. Current History producers emit UUID
or empty archive fields, never encoded arrays. It is not counted as a new finding.
No additional obsolete vcobj reference-repair branch was confirmed in this
native-codec inventory.

False positives and explicit exclusions:

- `History.cpp::referenceColumn` accepts 16-byte archive UUID or 56-byte complete
  reference BLOB: both are **current schema** storage projections. It rejects
  text; it does not reconstruct obsolete public arrays.
- `geometry/GeometryToken.hpp` TEXT vs BLOB handling reads current `storage_ref`
  versus archive columns. `source_revision` manifest fields are current shared
  payload ownership/borrowing metadata; erase-on-comparison omits physical hints,
  not a legacy object-reference fallback.
- `Lifecycle.cpp::bindLifecycle` compares supplied epochs to current selected
  endpoint lifecycle. The compatibility comment does not imply old reference
  repair; mismatches throw and no handle is synthesized.
- `Commands.cpp::requireCurrentCommand/requestHeader/selectionFrom` require format
  3, reject old reference sidechannels, and require complete refs/null (or explicit
  tombstone wrapper). `Versions.cpp::receiptFromJson` rejects missing complete
  references, rather than rebuilding them from its numeric projection.
- `StoreReader::decodeObjectReference` supports explicit current discovery and
  owner-bearing textual locators at the boundary. Ownerless numbered/flattened
  selectors and archive-revision inputs reject. Complete exact refs are not
  repaired by a current lookup.
- `migration/ConvertStore.cpp` is the explicit old-schema conversion boundary;
  its old-record interpretation is permitted. Fiber JSON, point-collection files,
  Umbilicus source files, VCTRK packed tracks, CT-normal/respool formats and
  disposable line-surface numeric caches are independent source/cache formats,
  not obsolete vcobj reference schemes. This sweep does not remove their support.

## Checked list/map and binding exclusions

- `ObjectVersionJson::objectVersionsFrom`, `Wire::manifestReferences` and
  `Extensions::extensionSourcesFromJson` explicitly gate list containers.
  Manifest optional absent/null list fields are intentional current policy;
  object-valued lists reject. Per-entry shared identity checks remain required.
- `Artifacts::parseManifest` checks the manifest array before entries. Geometry
  `geometryLayout` checks ordinary array-chunk lists before downstream use;
  structural/sequence/source annotation row validators check arrays and counts.
  These are not missing checks merely because internal consumers iterate.
- Typed `.get<std::vector<...>>()` / `.get_to(vector)` paths enforce arrays:
  record tags, selection UUIDs/kinds/groups, wire fields/UUID/tag results,
  per-array shape, packed offsets, receipt created UUIDs and region point/grid
  arrays. No redundant explicit per-loop check requested there.
- `LinkObjects` detector plan lists, UUID lists and fixed weight arrays have
  explicit admission. `LineSurface` cache verifies its grid list before decoding;
  stale cache rebuild is not silent reconstruction of a selected reference.
- Record meta/source/annotation values, command `annotations`, geometry
  name-to-array descriptor maps, sequence metadata bucket maps, receipt/expectation
  maps, checkpoint `drafts`, source point-ID maps and arbitrary JSON recursion
  (`reachableGeometryMetadata`, `annotationDirectional`) are not declared arrays.
- Python native bindings adapt `nb::list` / `nb::sequence` / typed STL vectors
  into native types. Tuples/sequences can be legitimate language adaptation, not
  obsolete JSON formats. `ExtensionBindings` based_on and exact object/relationship
  refs delegate shared codecs. `StorageBindings::anchorFromPy` delegates native
  anchor decode and deliberately omits null optional fields. Public Python
  local/hosted/session submission delegates the affected Commands parser, so
  schema policy belongs there, not in a new Python list-check table.

## Diagnostic recipe, results and limits

At root's request this lane added only an ignored local diagnostic file,
`local/scripts/vcobj_class_sweep_probes.py`, using `apply_patch`. It accepts no
arguments and uses `TemporaryDirectory` for isolated test stores. Root invokes
it through the already approved `bash local/scripts/run_vcobj_review_checks`;
this lane did not independently run a new Python command or build.

The fixed matrix covers selection transforms/origins and local history off/on
command operations, metadata objects/edges and anchor points, with array control,
object and null inputs. It reports acceptance/rejection without claiming malformed
acceptance is a passing regression. All fixtures are automatically removed.
Root returned the actual runner `91144` matrix: all valid array controls accepted.
Selection `transforms`/`origins` object and null accepted; `commands.operations`
object form executed the contained metadata write and null returned a no-op;
metadata import `objects`/`edges` object and null forms returned successfully;
anchor `points: {}` created an edge. Both local history policies had identical
results. Python anchor `points: null` also accepted, but is **excluded** from the
defect because its documented/native binding adaptation explicitly omits null
optional fields. This reviewer authored the fixed probe, root ran it through the
approved runner; no independent execution is claimed. Do not treat these results
as runtime verification of the remaining wire/archive/document boundaries.

No hosted fault server, packed/archive corruption fixture, checkpoint corruption,
GUI runtime, macOS/arm64 execution or concurrent mutation test was run in this
lane. Cross-lane adjudication follows below.

## Independent Python boundary adjudication

Read `python-references.md` and independently inspected the candidate callers and
producers. **Accept CS-P02 and CS-P03 as additional sites of CS-A01**, not new
wrong-owner/identity-loss root causes:

- `lasagna/fit2tifxyz.py::_stored_sources` reads saved job `input_references` at
  the checkpoint `_export` path. Its producer writes an array; an empty dict or
  string skips the scalar reference codec and returns empty sources. This is
  raw saved-data admission, not a public API intentionally accepting iterables.
  Keep `job_spec is None`'s explicit policy separate. No actual preexisting source
  references lost or incorrect geometry are demonstrated by the malformed empty
  examples. Native list-reference validation should precede projection to UUIDs.
- `spiral-fitting/object_store_inputs.py::InputPins.from_provenance` decodes
  position/ref scalars but not saved pin structure. Empty list/string `references`
  normalize to `{}`; a dict in a per-kind `selection` is iterated as UUID keys,
  ignoring values. The `provenance` writer emits reference maps and selection
  arrays. Validate that saved structure once, preserving the complete decoded
  references; no repeated checks are needed on the resulting application values.

`InputPins.from_manifest` validates through native manifestReferences first and
is not a duplicate missing-guard finding. Hosted facade `list(refs)`/`list(ids)`
accepting tuples/generators is not a defect without an external JSON-array
contract at that call; no such raw saved-data path was established. Likewise,
change-feed identity adaptation starts from a feed envelope, not a discarded
complete ref. The known numeric marker reconstruction remains NX-P01/CS-P01,
not another new occurrence. Stale ownerless-locator CLI help is documentation
only: the runtime decoder rejects that old spelling.

Native reference-construction ledger and completed Qt supplement read: its
authoritative row/codec/feed/CLI constructors, selected-value copies, display
projections and new-draft sentinels agree with the shared-identity contract.
The Qt supplement classifies its 29 `toArray` sites and indirect dispatch/point
handlers. Shared AgentBridge parameter validation already gates array inputs;
required fixed-size shapes fail their length checks. Optional display diagnostics
and unrelated remote task lists are not vcobj selected-state decoders. CS-Q02's
malformed-point skipping is an explicitly excluded hardening possibility: no
reachable unvalidated authoritative vcobj edit path was established. No additional
Qt/reference finding accepted solely because a loop or `toArray` occurs.

## Final class-wide adjudication

- **Reference downgrade/reconstruction:** retain known NX-P01 (CS-P01 alias).
  The full Python/native call-site sweep found no additional confirmed instance.
- **Obsolete reference repair:** retain known NX-A01. No additional confirmed
  ordinary-runtime legacy vcobj repair was found; current binary storage forms,
  explicit conversion tools and independent source-format support remain valid.
- **Declared-container ingress:** expand NX-C01 under **CS-A01**, with the exact
  native/public/persisted boundary table above and accepted saved Python sites
  **CS-P02/CS-P03**. These are occurrences of the same missing structural admission
  policy, not a new architectural feature or one finding per downstream loop.

The source-topology rows are also traced through their actual native producers
and ingestion consumers: fiber export emits branch arrays; point collection
export emits link arrays; file decoding retains these keys into native topology
ingestion without a prior collection-shape gate. They are source-only array
admission sites, not objections to supported external file versions. No topology
corruption runtime claim is made. Persisted descriptor rows likewise retain their
explicit source-only exposure and checksum/semantic-check qualifications.

Root reports the approved review runner completed with exit 0, including syntax
checks, 66 combined tests and the diagnostic matrix. This is compatibility/test
evidence alongside reproduced malformed acceptance, **not** proof that the open
class is fixed. `git diff --check` passes. This lane changed only this review
document and the explicitly requested ignored diagnostic script; no production
fixes or permanent regression tests were made. Final closure means the requested
finite class/site inventory and adjudication are complete, not that unrelated
code or unexecuted platform/concurrency behavior is finding-free.
