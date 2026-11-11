# Failure-class fixes

Baseline: `d13d33629`. Scope is the three adjudicated mechanisms in this directory, not a schema redesign. The 26 container entries are affected groups, not 26 independent formats. Reject malformed serialized containers once at their owning decoder; retain typed values downstream. Share the extraction primitive and schema admission across execution, payload discovery and replay. Preserve deliberate maps, optional missing/null policies, independent source formats, binary identities and history-off storage.

- [x] Shared JSON container admission; transaction root/nested collections and payload/preparation discovery.
- [x] Wire/feed/selection/snapshot/anchor collections; reject obsolete relationship archive encodings rather than clearing them.
- [x] Packed directory, extension/geometry/annotation descriptors, checkpoint/document/catalog collections.
- [x] Fiber/point source topology, including prepared/batch ingress.
- [x] Python saved provenance/checkpoint arrays through existing native reference handling; preserve marker receipt reference without numeric reconstruction.
- [x] Permanent regressions for malformed object/null/scalar containers, valid controls and explicit nullable exceptions; approved test runners; sibling review and line deltas.

No persistent tables, per-object directories, schema changes or compatibility repair paths are needed. Tests and documentation are counted separately from production code. Verification results will be recorded here after execution.

## Owning admission and sibling closure

| Reviewed family | Shared implementation and downstream behavior |
| --- | --- |
| JSON container shape | `internal/JsonFields`: `jsonArray` and `jsonArrayField` reject non-arrays. An optional missing field returns a stable empty array; explicit null is not silently normalized. They return const references, not copies or a new representation. |
| Commands: operations, data arrays, relationship decisions, edits, channels, items, registrations | `requestHeader` invokes one collection-schema admission before payload lookup. Materialization, raw submit, missing-payload replay, payload discovery and preparation discovery all call that header. Internal loops over the admitted document do not repeat guards. Scalar/semantic validation remains in existing typed decoders. |
| Selection transforms/origins | Native list-reference decoder admits transforms; the shared array helper admits origins before fixed-pair validation. Python selection decoding delegates native. |
| Metadata snapshot objects/edges | Both arrays admitted before the first preflight/write; insertion and preflight use the same admitted references. Existing operation savepoint remains. |
| Anchor points | `dependenciesFromJson` admits points; all edge, command, history and Python adapters share it. Missing points and explicitly omitted optional Python null remain supported. |
| Wire conflicts, feed commits/changes, geometry arrays, queries/history/exact results, selected inputs, region runs, proximity hits | Shared field extraction at each owning wire decoder; `unwrapArray` admits list-result envelopes without incorrectly constraining scalar/object envelopes. Null entries in exact results remain valid. Local/hosted/cache callers use these decoders. |
| Hosted missing-payload response | The response decoder uses the same array helper before upload requests. Unknown payload/retry safeguards remain. No synthetic faulty-HTTP-server response was run in this pass. |
| Packed databases directory | `readDirectory` admits the member array before interpreting entries; open, inspection, finalization and compaction share it. New test starts with a builder-produced file and recomputes framing/checksum after malformed-directory injection, so checksum rejection cannot mask admission. |
| Extension snapshot chunks | `readSnapshot` admits once before payload reads; restore/exact scans and current chunk enumeration reuse that decoded path. CBOR row/cell checks remain. |
| Geometry delta chunks | `loadGeometryManifest` admits once; `decodeGeometryRecord` supplies the admitted descriptor to forward/reverse geometry reconstruction and delta decoding. The array-name descriptor map is unchanged. |
| Annotation chunks | Materialization and closure/export independently admit descriptors through the shared array-field helper, without duplicating their schema policy. Inline values and metadata maps remain valid. |
| Draft pending requests | `DraftDirectory::load` admits the pending array, leaving the drafts map unchanged. |
| Attached document records / catalog files | Their native owning decoders use optional array-field admission. Unrelated document/catalog metadata remains unconstrained. |
| Fiber branches and adjacent branches | `validateSourceTopology` is shared by file decode, public endpoint discovery and prepared/batch metadata admission. Normalization and relationship insertion consume the admitted topology. External fiber versions remain supported. |
| Point links and winding links | The same source-topology admission checks decoded point annotations and collection winding links in file and prepared/batch inputs. Point-ID and collection maps remain maps. |
| Lasagna saved input-reference list | A minimal Python binding exposes the existing native `objectVersionsFrom` codec. No Python scalar-reference or iterable-repair implementation is added. |
| Spiral saved pins | The application decoder checks its own reference/selection maps and per-kind UUID arrays before normalization; scalar references still delegate native. Discovery-only pins without references remain supported. |
| Marker receipt / legacy relationship revision repair | Marker reads pass `receipt_object_reference` unchanged into exact lookup. Wire archive selectors must be UUIDs or empty, never repaired from obsolete encoded arrays. |

## Validation status

Commands: `bash local/scripts/run_vcobj_review_checks` and `bash local/scripts/run_vcobj_pr_tests`. Both are the existing argument-free approved runners; the permanent native regression is a normal CMake/CTest target (`object_store_json_boundaries`). Temporary stores are owned by automatic-cleanup fixtures. No dependencies were installed.

The full PR runner reports 54 native tests passed (optional SSH skipped), 279 store Python tests passed, 369 Spiral tests plus 54 subtests passed (one optional skip), and 37 Lasagna tests plus 31 subtests passed. It exits nonzero for exactly the same three recorded baseline Spiral failures: `test_only_a_preview_of_this_session_is_committed`, `test_clients_cannot_choose_the_binding_or_host_input_paths`, and `test_a_service_packs_remote_arrays_as_it_packs_local_ones`. The first two are message-expectation mismatches, and the third is the existing fake remote-fetch expectation. No new failure remains in that run.

Two new-fixture issues were corrected during validation: the packed test now uses the existing shared SQLite/VFS opener, and the existing marker-race test hook now accepts the complete reference rather than its removed numeric signature. No production compatibility workaround was added for either. The final focused rerun also rebuilt and checked the shared helper's explicit parent-object admission, and exited 0: ten focused native targets, two repeat/umbilicus checks, 25 source-import tests, ten consumer runtime executables, 48 selected Python tests, all consumer syntax checks and 85 combined fiber/surface/manifest/grower/position tests passed. `git diff --check` passed. All logged implementation tasks are closed; unrelated baseline failures remain explicitly recorded rather than suppressed.

Production delta: shared/native infrastructure +111/−39 (net +72); Spiral/Lasagna consumers +15/−14 (net +1); combined +126/−53 (net +73). This adds missing validation rather than claiming a line reduction: the marker path shrinks, while saved-schema admission adds checks. Regression code/build declarations add 373 net lines, separately from these production totals and review documentation.

Platform/coverage limits: Linux native/Python and compiled consumer checks only; no interactive GUI, macOS/arm64, full-volume performance run or injected malformed 422 HTTP server. Wire admission is exercised directly through the same codecs used by hosted/cache consumers; normal hosted paths are covered by the existing native/Python suites.
