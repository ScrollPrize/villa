# Independent adjudication of follow-up findings

Baseline inspected: `5920625ccaa7b72ee8f60a9d659f6b49190df299`. This adjudication is source review only; no production edits or tests support its conclusions. The versioned-reference, reading and coherent-store contracts govern the recommendations.

## Disposition

| Finding | Decision | Qualification |
| --- | --- | --- |
| F-01 | Accept | `umbilicusFingerprint()` omits the owner. Replace the partial string with a complete selected reference, retaining status, application generation and mapping/volume invalidation separately. Existing event invalidation is not disproved. |
| F-02 | Accept as consolidation | The named scalar wrappers preserve identity today. Use the existing scalar codec and a shared binding conversion; do not change genuine list or two-endpoint operations merely because they use the same batch helper. |
| F-03 | Accept with semantic constraints below | Numeric maps are a real alternate internal representation. Some callers are already protected by complete guards, one transaction or a commit guard; this is not proof that every numeric assignment is an exploitable race. |
| F-04 | Accept | Python `StoredRelationship` and native `consumedIdentity` discard relationship ownership/lineage. Preserve the relationship's full reference beside the deliberately projected consumed content. |

No finding is rejected, but neither F-02 nor the already guarded F-03 call sites should be reported as reproduced wrong-version reads.

## Native transaction design

`Transactions.hpp` explicitly defines the old `VersionSelection` null value as **no live record**, including deletion. `Versions.cpp::headVersion` and `relationshipHeadVersion` implement that rule. It is not an assertion that a UUID was never allocated, nor permission to re-import a reserved identity. Retired objects are live records for these checks. Preserve those distinctions.

One shared expectation map with a discriminated value is coherent. Its ordinary alternatives should retain `ObjectVersion` or `RelationshipVersion` unchanged, plus an explicit `NoLiveRecord` expectation. A tagged legacy numeric alternative may remain only for external compatibility, with no parallel numeric/full maps. Object and relationship expectation containers may share that implementation, provided validation rejects a relationship reference in an object/read-dependency slot, the converse, a map-key/reference-UUID mismatch and invalid references. Do not introduce another flattened identity struct.

Complete live-record expectations and a selected deleted record are not interchangeable. The current complete-object guard calls `resolveObject`, which excludes deletion markers. `StoreReader::currentObject` also excludes them; `currentVersionOf` can instead return the last retained history entry, including its deletion number. That number cannot mechanically become a live expectation: the existing numeric head check returns null for deletion. Restoration or deferred edits that actually select a deletion marker need an explicit selected-deleted expectation carrying its full reference and shared tombstone-aware resolution. A caller requiring merely no live head should use that weaker, explicitly named alternative. Never fabricate version zero, infer a deleted identity from the attachment, or require retained history just to discover the winning tombstone in a history-off store.

Centralize helpers for object writes, object read dependencies and relationship expectations. Preparation, snapshot caches, session fetch callbacks and document APIs must transport the same expectation values. Numeric projections belong only at SQL/checking, diagnostics or wire boundaries; they must not become a second synchronized cache of consumer state. Preserve commit-sequence/membership and loaded mapping guards, which complete references cannot replace.

Merging requires explicit rules: equal complete references merge; different complete identities conflict even at the same counter. A compatible legacy number plus a complete reference must retain the full reference, not downgrade it. Removing a redundant read dependency is safe only when the write expectation entails the same complete dependency; numeric equality alone is insufficient. Preserve loaded guards when expanding endpoints rather than reselecting a newer head and calling that the loaded reference.

`StoredGeometry` really stores numeric `versions_` and exports numeric preparation expectations. `transformDependencies` similarly drops selected record identity. The numeric endpoint-expansion overload has no demonstrated production caller and can be removed rather than supplemented. `publishArtifacts(attachment, uuid, number, ...)` discovers a full selected object but retains only the number across upload; its pinned overload demonstrates the coherent path. Fetch eligibility and locking remain independent of the selected reference. Attached-document loaded overrides and the production growth-annotation editor guard already supply important protection and must survive the migration.

## Wire compatibility and replay

Old numeric commands are externally meaningful preconditions, not recoverable old complete handles. Parsing JSON without a store cannot resolve them, and replacing their counters with freshly selected heads silently changes their meaning. Keep them explicitly legacy until shared transaction-boundary validation, or use an equally explicit store-aware normalization that preserves the old counter check under the same writer snapshot. Do not let that compatibility representation spread into newly selected production inputs.

Existing command version 2 carries a numeric projection and `expected_object_references`; these may be generated from one internal map. Decoding must reject disagreement and duplicate canonical keys. New relationship/read-reference fields must be version-gated so an older reader cannot silently ignore a stronger guard. Preserve old canonical serialization/digests for old requests: durable retries and saved receipts depend on them. Receipt replay must precede current-head checks. Old receipts lacking complete identity remain reload-required rather than being repinned from the committing store.

## Relationship identity and provenance

`RelationshipVersion` already exists but `RelationshipRevision` lacks the equivalent shared `reference()` constructor at this baseline. Add it at the native record boundary using the relationship's owning record store and its selected publication generation/version. Supply shared scalar codecs and binding adaptation. An endpoint's object owner is not the relationship's owner; geometry UUIDs and archive UUIDs are not substitutes either. Cover retained historical and history-off selected relationships as well as current ones.

Python provenance should retain the shared reference as one field beside content projection/hash. Native comparison should compare that identity for new manifests while preserving the intentionally limited materialized-content projection. Existing manifests without full relationship references are legacy external inputs: retain a documented weaker comparison or explicitly require reload; do not invent their historical owner from today's head. Do not accidentally classify a missing legacy field as proof of complete equality.

## Implementation acceptance checks

Use the approved runners in `validation.md`, not ad-hoc tests. Add cases for equal counters across different owners/lineages; full-reference preparation/read-dependency merging; artifact upload followed by owner change; loaded deferred edits; absent versus retired versus deleted selections; history-on/off restoration; relationship publication identity independent of endpoint ownership; old command digest/replay and malformed mixed wire fields; legacy manifests and new complete relationship provenance. Keep no-op ownership, receipt repinning and membership/mapping guards intact. No schema changes, persistent identity tables or copied referenced payloads are required.
