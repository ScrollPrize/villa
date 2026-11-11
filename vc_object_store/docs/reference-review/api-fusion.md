# Selected-value API fusion

Baseline: `0c7e4e521`. Consolidate the BV-01–07 implementation, remove redundant interfaces and keep one shared implementation per operation. Preserve schema 21, binary stored UUIDs, exact references, historical selection, loaded edit/mapping guards and atomic graph capture. No legacy fallback or new persistent structures. Astra reviews the candidates before implementation and the resulting patch afterwards.

Subproject: vc_object_store (C++/nanobind/Python), with Spiral and surface/editor consumers. Existing CMake and fixed local PR/review runners provide native, Python and consumer regression checks on Linux; portability remains Ubuntu/macOS. No installations or solver/numerical changes.

| ID | Candidate | Review / implementation |
| --- | --- | --- |
| AF-01 | Remove Mapping/attribute duality and record/revision sidecars. | Accepted and implemented: attribute-only values; SelectedObject derives record from its sole revision. Native metadata-only listings retain SelectedRecord. Dataclass value equality includes the fitter's frame. Spiral selection, shell, marker and pin callers migrated. |
| AF-02 | Fuse discovery, payload construction and transform mapping. | Accepted and implemented: UUID discovery resolves once, selected construction and explicit payload reads share one payload constructor. Remove GeometryView's late ObjectRevision transform override; reuse its retained state for mapping, including optional transform context. Migrate SurfaceGrid, VolumeUmbilicus, SpiralStoreInputs and SurfaceCatalog; generic Snapshot conveniences delegate. |
| AF-03 | Remove native JSON text round trips inside request/reference/status adapters. | Accepted and implemented: one JSON-value codec for references, lists and store status/position. Transport/text wrappers delegate; wire metadata must retain exact text, not accept old object-shaped metadata. Omission remains valid for projected queries. |
| AF-04 | Remove FeedCursor as an alternative graph position. | Accepted and implemented: StoreStatus capture position and commit_sequence throughout. FeedCursor remains only for real change-feed pagination. Reject obsolete sequence aliases and non-integral/overflow positions. |
| AF-05 | Fuse local/hosted/session Python capture adaptation. | Accepted and implemented: one small callable adapter for conversion, GIL release and result conversion. Local store lease is acquired before releasing the GIL; native transaction, session and hosted RPC ownership remain distinct. |
| AF-06 | Remove redundant generic before/after capture checks. | Accepted and implemented: LocalStoreReader owns the pinned transaction check once; HostedStoreReader delegates one atomic RPC. Stale inherited-position equality remains mandatory. |
| AF-07 | Remove obsolete saved-identity alternatives. | Ownerless locators/flattened selectors, v1 selection artifact decoding, incomplete receipts and missing wire-owner/status-field fallbacks are rejected. Ordinary current discovery and explicit numeric selection that immediately yields a complete reference remain supported. |
| AF-08 | Remove project-resource UUID aliases. | Remove legacyResourceUuid, resourceForEdit fallback scanning and resolveProjectResourceIdentity. Edit actual stored UUIDs through the common session reader. Keep valid-v21 creation-order normalization, accurately named refineCreationStamp. |
| AF-09 | Remove old-schema SQL decoding from runtime history. | Remove the entirely unused historicalDependencyMetadata helper and legacyTextUuid switch, plus the unused text alternative in SqlReference. Repository-wide search found no converter callers, so no decoder copy or replacement is needed. Offline conversion remains isolated. |
| AF-10 | Preserve pending request identity when strict receipt decoding fails. | Astra found a safety interaction with invalid_argument cleanup. Shared ReceiptDecodeError now covers unreadable wire and saved replay receipts; sendPending retains the same request for both local and hosted cases. Remove the obsolete conclude fallback for identity-less receipts. |

## Completion

Initial Astra adjudication accepts AF-01–06: use attribute-only Python selected values; one payload constructor; JSON-value reference codecs; existing StoreStatus for capture positions; a small binding callable adapter; local-only transactional validation and one hosted RPC. Preserve explicit source imports/offline conversion. The added legacy-removal scope includes ownerless numbered locators, selection-record v1 decoding, missing-reference receipt acceptance, project UUID aliases and the unused text-UUID history switch. Creation-time precision normalization is a valid-v21 ordering invariant, not obsolete schema routing, and must remain unless migrated without changing order.

- [x] Astra adjudication recorded for every candidate, including rejection/constraints: [review](api-fusion-astra.md).
- [x] Implement accepted fusions; enumerate migrated sibling callers and removed interfaces.
- [x] Approved fixed PR/review runners; distinguish known baseline service failures from regressions.
- [x] Astra patch review; address substantive findings. Store-lease race, permissive sequence coercion and pending-identity loss were all corrected and re-reviewed. The final transform fusion was additionally adjudicated before implementation.
- [x] Record production/shared/consumer and test deltas separately; commit the atomic API removal with all required caller migrations, then its review/validation documentation.

## Scope of legacy removal

Runtime store opening still accepts only the current schema and rejects unsupported stores untouched. The separate offline upstream converter remains as previously required. Do not confuse valid v21 data/format features with legacy APIs: current geometry/source-import encodings, extension formats, readonly arbitrary shard routing, change-feed pagination and second-precision creation timestamps are supported semantics, not old schema readers. Frozen historical review reports remain evidence, not current usage documentation. Current reading/API documentation is updated to reject removed forms.

Regression verification includes canonical current/exact selectors, deliberate rejection of ownerless inputs/artifacts/wire metadata, preserved projected metadata omission, strict integer positions, retained payload-only non-authority, and durable local replay plus hosted HTTP-success receipt corruption followed by recovery under the same request UUID. Test fixtures are cleaned automatically by the fixed approved runners; no new install or argument-dependent runner is required.

## Validation

The fixed `bash local/scripts/run_vcobj_pr_tests` built the native code and passed 53 native tests (optional SSH skipped), all 230 Python store tests, 353 Spiral tests plus 54 subtests (one optional data skip), and 37 Lasagna tests plus 26 subtests. Its overall exit remains nonzero for the same three documented baseline Spiral service failures: preview error wording, service binding error wording, and the fake-S3 GET assertion. No new failures remain. The separate fixed `bash local/scripts/run_vcobj_review_checks` exited successfully, including the native core regressions, all ten native consumer executables, ten selected-value/transformed-view Python cases and all configured consumer syntax checks. `git diff --check` passed. Linux/GCC 13/Python 3.14 were tested; full GUI, macOS and performance measurements are not claimed.

Astra reviewed the candidates, the patch, all three corrective changes, and the additional transform and strict-metadata fusions. The receipt tests actually commit once, deliberately corrupt a saved replay or successful hosted response, assert the pending request is retained, then recover that same UUID without a second object version. The local store lease is obtained through the existing withStore API before releasing the GIL; there is no new parallel lifetime protocol.

## Line delta from 0c7e4e521

Excluding documentation/local runners: shared production +197/−248 = **−51**; production consumers +17/−19 = **−2**; total production +214/−267 = **−53**. Regressions +155/−23 = **+132**. Therefore code plus tests +369/−290 = **+79**. This pass reduces actual production code, rather than hiding it in tests or replacing explicit code with opaque macros. Consumer line savings are small because many changes replace field interpretation with an equally short canonical attribute/API call; obsolete alias/fallback machinery accounts for most shared-code savings.
