# Independent post-receipt review

Freeze: `371961148a66a09b79a527e6bd3fadb777d219b4`. Initial working tree was
clean. Read the review and single-reference contracts, current reading rules,
the class-sweep fix tracker and PCS-01 receipt finding/fix record. This lane
performs read-only source review and documentation only; no production/test
edits, installation, build or new runtime test execution.

## Independent receipt-fix review

PCS-01 is closed in source at its owning decoder, not worked around in a
consumer. `Versions.cpp::selectionFromJson` now requires map shape,
canonicalizes every key, rejects canonical collisions and uses the shared
positive-version decoder for non-null values. `receiptFromJson` delegates
nullable sequence decoding to the shared nonnegative position codec.
Complete-reference keys reject canonical collisions, and every reference
must have a live object result. Conversely, each live object result requires
a matching complete reference. Null absent/deleted results remain legal.

The two-way check occurs after normalized result maps have been built;
checking each newly inserted reference does not depend on insertion order
of other references. Strict scalar codecs reject booleans, fractions,
strings, negative values and unsigned overflow as applicable. Zero commit
sequence remains valid, zero live version does not. References retain their
own owners, including unchanged inherited inputs; the receipt's committing
store/generation is not substituted.

| Entry point / sibling | Fresh source disposition |
| --- | --- |
| `Commands.cpp::transactionReceiptFromJson` | Wraps decoder failures as `ReceiptDecodeError`; no alternate receipt parser. |
| `Versions.cpp::savedReceipt`, `ObjectStore::receipt` | Persisted replay uses the same public decoder, including receipts whose geometry payloads are no longer staged. |
| `client/HostedStore.cpp::{submit,receipt}` | Successful response bodies use the same decoder; no numeric identity reconstruction. |
| `python/objects.cpp::receipt_object_reference` | Thin JSON adaptation into native decoder/accessor, not Python schema policy. |
| `ObjectStoreSession.cpp::{sendPending,resolvePending,conclude}` | Decoder errors retain uncertain pending request identity. Recovery decodes before concluding; success repins drafts from complete receipt references. |
| `Versions.cpp::receiptToJson` and receipt construction | Producer retains resolved owner references; the serializer has no new format or fallback. This review does not demand redundant checking of every typed internal value. |
| Native `completeReceipts`, Python `test_reference_inputs.py` additions | Source-inspected valid producer/replay controls, malformed maps/scalars, uppercase keys, collisions, missing/extra/mismatched references, nullable values, history on/off and inherited-owner coverage. Tests were not run by this lane. |

No regression found in the receipt fix. No wrong-owner/geometry effect is
inferred from malformed scalar projections, and raw informational header
strings are not promoted into a new finding without a concrete operational
misuse. The demonstrated prior omission is fixed rather than shifted to a
different consumer.

## Container patch and shared-invariant cross-check

The complete changed-family/symbol ledger from
`../post-class-sweep/astra.md` applies to the committed container portion;
that portion was independently read in the preceding pass, including the
new native regression. This pass rechecks owning admission and receipt
interactions, rather than pretending unchanged source was all newly read.

| Family | Disposition / limits |
| --- | --- |
| `JsonFields`, `Commands::requestHeader`, materialization/payload/preparation/raw-submit/replay | Array admission is once at the shared boundary, before payload-dependent branching. Helper references have valid lifetime. Absent optional arrays remain distinct from malformed explicit null. Typed downstream loops do not need another policy implementation. |
| `Wire`, `Selections`, shared object-reference list codec and binding | List envelopes and nested collections delegate shared admission. Exact null entries and history-off empty archive UUIDs remain intentional. Obsolete archive-array strings reject rather than repair. Full references remain strict and unchanged. |
| `Snapshots::importJson` | Both arrays admitted before mutation; scoped object/edge resolvers reserve deleted UUIDs. Same-kind live idempotence and cross-kind conflicts remain. Savepoint encloses all writes, preserving atomicity if the caller catches a later failure. Metadata export is explicitly not a geometry transfer format. |
| `GeometryDelta`, `GeometryReference`, `GeometryReader`, `SequenceMetadata`, `ExtensionHistory` | Owning descriptor decode precedes chunk traversal; geometry/annotation/extension materialization and closure paths use shared storage resolution. Array-name maps and arbitrary inline metadata remain maps. No per-placement parsing alternative is introduced. |
| `SourceRelationships`, `FiberJson`, `PointCollectionInput`, `GeometryIngestBatch` | Shared topology admission covers file and prepared/batch inputs before normalization and relationship insertion. Savepoint covers later-source failures. Independent source versions/ordinal topology are not obsolete vcobj reference formats. |
| `PackedStore`, `PackedCollection::packObjectCollection` | Directory admission belongs to shared pack reader. Packing imports member closure, calls shared relocation and validates rewritten references before finalization. No new root-only copy fallback or borrowed-payload proxy. |
| `ObjectCopy::{readObjectCopy,writeObjectCopy}`, `Artifacts` | Copy reads one selected current revision for metadata/geometry and writes under a savepoint. Informational `copied_from` is not a located operational reference reconstruction path. Artifact payload lookup resolves the pinned store scope. No copy/pack format change in this patch. |
| `History::objectHistory`, `StoreReader::{currentObject,objectHistory}`, `ReferenceResolver`, extension exact bindings | Owner retention policy remains distinct from attachment policy. Exact reads route through immutable selected identity, not a current-head requirement. Relationship extension binding uses the shared relationship codec. |
| Checkpoints, attached documents, catalog and hosted missing-payload response | Declared lists use owning array admission; deliberate maps and public Python iterable adaptation are not false-positive list violations. Malformed HTTP server injection was not executed here. |
| Lasagna saved reference list and Spiral provenance/marker paths | Native reference-list decoding and application-owned map/list checks remain separate. Marker repins via complete receipt reference; receipt failure must be fixed at native owner, never by consumer fallback. |

No concrete new defect was established in this independent delta/sibling
review.

## Final cross-lane adjudication

Read `native.md` and `consumers.md` completely after completion. Agree with
their fresh traces and explicit reuse limits. In particular, stricter native
receipt failures remain `ReceiptDecodeError`, so deferred GUI/session save
paths do not mistake them for a definitively rejected request. The native
decoder remains the sole schema owner for local, hosted, replay and Python
receipt paths. No additional consumer validator is needed.

No new candidate requires acceptance or a probe. Prior malformed-container
and marker cases are closed by shared admission/full-reference handling;
ordinary typed iteration, explicit discovery, public iterable adaptation and
informational copy provenance are not re-counted as defects. Metadata
snapshot same-kind live idempotence remains intentional; tombstone and
conflicting identities remain rejected.

Final verdict at the frozen commit: **no additional confirmed defect or
regression in the reviewed invariants**. PCS-01 is source-confirmed fixed.
All lane adjudications for this pass are complete; this is not a guarantee
of defect absence or a claim of fresh line-by-line whole-repository review.

Incremental adjudication: also read the native lane's subsequent **Fresh
broad reopen beyond the fix diff** section covering collection/shard routing,
history/publication selection, geometry views, visible queries, extensions
and export/copy/pack/artifacts. Its dispositions do not change the verdict.
Independently verified both rejected concerns at their public boundary:
`Versions.cpp::relationshipsAt` rejects repeated resolved object UUIDs before
calling its private helper, and `StoreReader::currentObject` defaults to
`includeDeleted=false`, as used by object copy. Neither is an open defect.

## Verification boundaries

The existing recorded approved runs for this commit's fixes are evidence
reported by the coordinator, not executions by this lane: 54 native,
333 store Python, 369 Spiral plus the three documented baseline failures,
37 Lasagna, and the focused runner's 102 selected Python and 85 combined
consumer checks. No new all-platform or full-volume certification is
claimed. Source traces establish shared entry points; they do not replace
malformed-response transport, crash-injection or interactive GUI tests.

Unchanged detailed whole-file inventories remain explicitly inherited from
the preceding native/consumer/format ledgers. Numerical solvers, SQL engine
internals and platform-specific scheduling are not independently re-proved
by this review. Any post-freeze source change reopens its affected boundary.
