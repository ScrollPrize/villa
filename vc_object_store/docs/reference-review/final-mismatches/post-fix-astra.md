# Astra post-fix source review

Reviewed dirty working tree against `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4`, distinguishing the preceding PF changes from the newly implemented FM fixes. No production/test edits, builds or test execution by this reviewer. Root's running approved verification is separate evidence.

## Finding requiring a small correction

**FM-C01 public build interface: propagate the JSON header dependency.** The new installed/public `geometry/PointMap.hpp` includes `<nlohmann/json.hpp>` and embeds `nlohmann::json` in `AnnotatedPointMap`. `src/CMakeLists.txt` still links `vc_geometry` to `nlohmann_json::nlohmann_json` privately. A downstream consumer using the public PointMap API and linking only vc_geometry will not inherit JSON include paths when nlohmann_json is installed outside default system include directories. Current umbrella vc_objects/GUI builds may mask this, so a green monorepo compile is not sufficient evidence. Make nlohmann_json a PUBLIC dependency of vc_geometry; vc_reader::catalog can remain PRIVATE. The package config already finds nlohmann_json, so this adds no new dependency or installation requirement. This is introduced by the new public API, not an older runtime defect.

## Reviewed fix dispositions

| Fix | Source disposition |
| --- | --- |
| FM-A01 artifact output | Local writeArtifact now validates each final destination against the resolved store closure before creating parents or publishing. Both materializeArtifacts and artifactFile delegate here. Remote downloads and intentional internal sidecar-copy writer remain separate legitimate contexts. Per-file, not whole-materialization atomicity is preserved; no new all-or-nothing guarantee claimed. |
| FM-P01 retirement | Consumer selects native records and builds complete object_reference guards in one listing read; empty input is handled. It no longer submits removed numeric guards. Unknown objects remain failure cases rather than inferred identities; checked-write validation remains the final authority. |
| FM-P02 position ingress | InputPosition.from_provenance and service ingress use the native position codec; trusted of_status adaptation no longer receives unvalidated external scalar values from these paths. No duplicate Python numeric policy added. |
| FM-P03 surface selector | Wrapper stops int coercion; binding accepts the raw Python value and invokes shared positive-version JSON validation before GIL release. Bool, fraction, string and overflow cannot be silently converted. Complete-reference exclusivity and generation rules remain. |
| FM-C01 point map | Shared native decoder centralizes numeric IDs/aliases, XYZ shape/finite checks and annotations. Both importer and StoredCollection delegate. Importer document-wide renumber/link policy remains local to the format adapter. Build-interface correction above remains. Standalone PointCollections uint64 document loader was not migrated; its separate range/domain policy is a recorded exclusion, not a claim that every stoull in that loader was removed. |
| FM-C02 manifest ingress | Native/Python consume manifestReferences; public manifestPins validates independently, internal construction uses the decoded result once. Saved ObjectVersion is retained in StoreInputPin; relationships retain decoded refs. Validation precedes partial filtering; comparison/display still intentionally reads current content, without requiring historical payload retention. |

Manifest validator source checks cover absent reference, codec validation, entry/reference UUID agreement, strict optional numeric version, object/shell duplicates and relationship duplicates. Normal complete identities are not reconstructed from attachment ownership. Canonical saved entry UUID spelling remains required as in the preceding Python validation; the decoder canonicalizes the complete reference and rejects disagreeing entry IDs.

## FM-A02 relationship-key delta

The new `Membership = map<edge UUID, RelationshipVersion>`, shared bindings map, union set and archive-locator cache follow the approved design. Physical archive UUIDs are retained only to resolve archived membership once; they are not synthetic operational publication identities. Current discovery still uses root-first scopes and per-endpoint seenBindings, overrides that endpoint's membership for an admitted visible edge, and checks selected endpoints before admission. Final applicability preserves both membership equality and the prior missing-membership/endpoint-match fallback, followed by endpointApplies.

`RelationshipVersion` receives ordinary structural ordering. One `orderRelationships` helper sorts native results and both wire decode families, so reopened hosted graph metadata caches are normalized too. This closes the previously identified cache-order gap without invalidating raw geometry/artifacts or changing schema. UUID then full-reference output order is explicitly documented in the native API. Fiber JSON branch-array order may change as already accepted; it is not represented as byte-for-byte output preservation.

One implementation characteristic is worth recording without inventing a finding: archived membership is resolved before a matching current binding may override it, whereas the prior code could defer decoding a subsequently unused archive entry. Under valid retained-history invariants those archives are available; no supported failing counterexample was established. The new archive cache avoids repeatedly decoding the same physical entry. No benchmark or speedup claim is made.

## Verification expectations and limits

New regression source covers malformed surface/position selectors, native point-map aliases/shape/types/annotation ordering, manifest duplicates and reference mismatch, artifact collisions, and wire order normalization. Existing local/history-policy/hosted graph cases must remain green for A02; singleton cases alone do not prove multi-relationship dedup/applicability parity. Review the final test run and any remaining fixture modernization separately. No macOS, GUI runtime, numerical-performance or all-consumer coverage claim is made here.

Provisional verdict: the runtime/shared-layer changes are coherent in the inspected delta; correct the public CMake dependency visibility before final source approval. No other blocking source finding established in this pass.

## Final source disposition

Rechecked the corrective delta: `vc_geometry` now links `PUBLIC nlohmann_json::nlohmann_json` and `PRIVATE vc_reader::catalog`. The new public PointMap header's dependency is propagated correctly; the build-interface finding is resolved in source.

Also checked removal of the unused optional version argument from private `InputPins._decode_reference`: the only remaining call is the two-argument provenance-map adapter. Manifest optional-version agreement is enforced by the shared native manifest validator, so no validation path was removed. A repository Python call search found no stale three-argument caller.

**Final source verdict: approved, with no remaining blocking issue established in the inspected FM delta.** The main agent reports passing native/store-Python and focused checks, with broader Python/Spiral verification still running; this reviewer did not execute or independently verify those suites and does not claim their results as its own. Final runtime totals and any newly test-triggered changes remain the main agent's responsibility. Source/diff checks only were performed for this final disposition.

## A02 lazy archive follow-up

Reviewed the subsequent ordering of archive resolution in `relationshipsAtSelected`. Physical `members(...)` locators are collected first, but decoded only after admitted current bindings have populated that endpoint's full-reference membership map. Archived entries with an already-selected edge UUID are skipped. Root-first scope order, selected-endpoint matching, per-endpoint seenBindings and current-binding precedence are unchanged; surviving archives still resolve once through the archive cache. Final membership union, full-reference comparisons and endpointApplies are unchanged.

Approved. This removes the earlier eager-resolution caveat and preserves the old avoidance of decoding archive rows superseded by admitted current bindings. No further blocking source issue identified in this follow-up; no tests executed by this reviewer.
