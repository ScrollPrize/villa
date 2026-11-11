# Reference-bearing consumer values: review

Frozen review baseline: e7bdba35f. Focus: consumer handling that should become automatic through an existing or extended shared selected-value API, not another set of parsing helpers or wrapper formats. No finding quota was used. The source-review findings below predate implementation; the implementation checklist records subsequent changes and validation, not a claim that every consumer is defect-free.

| ID | Finding | Classification |
| --- | --- | --- |
| [BV-01](BV-01.md) | Selected geometry is not uniformly available through native/Python value APIs | Shared API/binding gap |
| [BV-02](BV-02.md) | SurfaceGrid callers rebuild identity already contained in its record | Shared value consolidation |
| [BV-03](BV-03.md) | Spiral flattens selected revisions, then reconstructs objects through sidecar maps | Consumer/type consolidation |
| [BV-04](BV-04.md) | Spiral implements inherited-input graph consistency policy itself | Shared snapshot capability gap |
| [BV-05](BV-05.md) | Point-collection editing splits identity/mapping from editable data | Consumer/type consolidation |
| [BV-06](BV-06.md) | Lasagna export source collection misses canonical references | Source-traced correctness defect |
| [BV-07](BV-07.md) | Lasagna materialization drops selected correction handles and persists unresolved selectors | Source-traced provenance gap |

All seven exist at this baseline. BV-06 was also checked in 6944d335c: it predates the latest classifier cleanup. No other finding is labeled a new regression without tracing its introduction.

## Coverage and limits

Screened 105 production Python/C++ files directly mentioning vc_object_store/vc::objects in Lasagna, Spiral, VC3D/core and Vesuvius source; expanded to 143 candidates through surface/fiber adapters and loader references. [The inventory](inventory.md) records candidate paths, not a claim of line-by-line semantic verification. Detailed tracing followed the flows in the findings through producers, shared APIs, adapters and consumers. Shared binding/value headers were additionally checked to distinguish missing capabilities from callers ignoring available APIs. That initial source review did not run runtime probes; subsequent implementation checks are recorded below.

## Deliberate exclusions

- External current/numbered selection, change-feed classification, historical selection, freshness checks, draft recovery and conflict/repin logic legitimately care about versions. Do not replace them with fresh-head guesses.
- Immutable geometry-revision keys are appropriate for channel/array reads; they are not necessarily incomplete object references.
- StoreSurfaces/ToolObjects already form a shared tool adapter. Its selection/mapping/write-guard work must exist somewhere; moving that whole adapter is not deduplication.
- Neural XYZ/ZYX adaptation and version-qualified content digests have real application semantics.
- JSON/service/file boundaries still serialize references. The problem is repeated interpretation inside a flow, not serialization itself.
- New-object UUIDs, runtime UI IDs and missing/deleted states cannot all be replaced by an existing selected reference.

## Implementation order

Expose/extend existing selected geometry and surface values first (BV-01/02), then migrate callers and remove reconstruction paths (BV-03/05), without retaining both models. Fix Lasagna source/provenance losses through the same reference-bearing inputs (BV-06/07). BV-04 needs a shared coherent graph/input-capture capability: do not hide the limitation behind a consumer fallback or delete its checks. Preserve membership, coordinate and loaded-version guarantees. Use approved runners for local/referenced and history-on/off regressions before claiming fixes.

## Implementation checklist

- [x] BV-01: geometry factories accept an already-selected native revision or Python canonical reference; views retain the selected metadata and complete reference. Metadata-only versions sharing a payload no longer depend on a current-head lookup. Payload-only reads remain explicit and do not authorize inverse edits.
- [x] BV-02: native/Python SurfaceGrid derive identity from their record. Lasagna retains that grid rather than separate reference/geometry/channel state; neural surface export uses its reference directly.
- [x] BV-03: shared Python SelectedRecord/SelectedObject facades carry native answers. Spiral keeps the revision, with its application frame alongside; copied record fields and the sidecar reference map are removed. Materialized point/fiber documents serialize the standard reference, not another numeric identity.
- [x] BV-04: shared selectedInputs captures selected objects and relationship membership, enforcing root position/retention policy. Hosted calls and session readers use one server transaction. Stale inherited membership fails closed; there is no old-server or current-head fallback.
- [x] BV-05: point drafts decode references at document ingress and serialize them at document output only; saves use the typed reference and receipts. PointCollectionSync retains AttachedCollection values instead of separate revision/mapping maps. Standalone volume collections are not owner/role membership replacements: their existing per-object edit guards and event-driven freshness remain; AttachedCollectionsSnapshot's distinct replace-all contract is not copied into the GUI.
- [x] BV-06: export sources project UUIDs from the job's captured canonical references, not a recursive old-envelope recognizer. A supplied legacy checkpoint without captured references must regenerate provenance instead of querying today's heads.
- [x] BV-07: resolved correction collections retain references; selected surface/model/atlas-base references accompany checkpoint inputs. Original user selectors remain only as request/display/rerun intent, separately from actual selected input provenance. Artifact content addresses remain distinct from store references.
- [x] Finish approved PR/consumer runners and record their results and limitations.

No schema changes, persisted routing/index tables, history-off branches, payload copies on edit, or solver changes are part of these fixes. The shared capture capability adds transport/bindings; deleting consumer checks without that capability would weaken correctness. New regression fixtures are automatically cleaned by the existing fixed, argument-free local runners: `bash local/scripts/run_vcobj_pr_tests` and `bash local/scripts/run_vcobj_review_checks` (approval uses their fixed absolute paths on the test host; local provisioning is not part of the PR).

## Validation and size

Linux, GCC 13, RelWithDebInfo native build and Python 3.14, using the fixed runners above. The full run passed 53 native tests (optional SSH skipped), all 227 Python store tests, 353 Spiral tests plus 54 subtests (one optional data test skipped), and 37 Lasagna tests plus 26 subtests. The overall PR runner still exits nonzero for three previously reproduced, unrelated Spiral service failures: `test_only_a_preview_of_this_session_is_committed` and `test_clients_cannot_choose_the_binding_or_host_input_paths` expect old error wording; `test_a_service_packs_remote_arrays_as_it_packs_local_ones` expects a fake S3 GET list that is empty. They are not hidden or counted as passes.

Regressions cover all typed geometry factories receiving complete references; distinct metadata bindings sharing a geometry payload; exact views after deletion; refusal of historical inverse edits; local and hosted input capture; inherited membership invalidation with history on/off; point-editor ingress/repeated saves; and saved correction references surviving subsequent source edits. The focused consumer runner also compiles the standalone point-editor regression against the shared store library; its checked-in test target now declares that dependency. No full GUI, macOS or full-volume performance claim is made.

The final approved review runner exited successfully: shared native regressions, all ten focused consumer executables (including 13 standalone point-editor checks), ten selected-value/transformed-view Python cases, and every configured C++ consumer syntax check passed. `git diff --check` passed. Temporary fixtures were removed by the runner traps. The PR runner's three unrelated failures above remain explicitly separate from this successful focused validation.

Code delta from e7bdba35f, excluding documentation/local runners: shared production +311/−65 = +246; production consumers +98/−128 = −30; total production +409/−193 = +216. Tests and test-build definitions +142/−24 = +118. Thus code plus tests totals +551/−217 = +334. This is a consumer reduction, not an overall production reduction: adding the missing coherent-capture capability, its native/hosted transport and Python selected-value facades costs more than the duplicated consumer glue removed. No old reconstruction/fallback path is kept alongside the new capability.
