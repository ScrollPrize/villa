# Coherent store follow-up fixes

Baseline: `3df5b7783`. All 21 findings in [the full review](coherent-store-review-3df5b7783.md) are implemented and validated as described below. The review's descriptions are frozen evidence; this file records current progress. The earlier 88 was a file count, not a defect count.

## Constraints and shared fixes

Preserve schema 21, binary identities, reference routing, slim history-off storage and immutable dependencies. No proxy rows, per-object directories, new history-off identity paths or edit-time copies of referenced payloads. Resolve a complete state once and carry it through geometry, annotations, transforms, caches and edits. Share logical mutation, relocation, lifecycle and source-validation helpers; remove divergent caller handling. Keep production growth separate from regression tests/docs. No installs or long experiments. Use existing argument-free local test runners; temporary fixtures must clean themselves up.

## Checklist

| Finding | Required correction | Status / evidence |
| --- | --- | --- |
| R2-1 | One captured transform for all components of a loaded fiber | Shared selected-view transform with the captured context; native consumer validation |
| R2-2 | Selected graph/geometry for chain display | Exact listing references and shared recorded-endpoint graph closure; native consumer validation |
| R2-3 | Preserve unchanged relative track source on full-record override | Implemented in shared full-record writer; regression added |
| R2-4 | Shared external locator rebasing for copy, pack and conversion | Shared locator rebasing and archived-record transformation; copy/pack/repack/conversion regressions |
| R2-5 | Keep-recorded semantics independent of attachment history mode | Implemented; explicit relationship decisions override the default; mixed-policy regression added |
| R2-6 | Shared no-op full-record edit detection before override | Implemented in common writer; removed collection-only comparator; local/inherited regressions added |
| R2-7 | Neural iteration provenance follows the actual preceding input | Loaded source identity retained through each iteration; chained-output regression |
| R2-8 | Restore borrowed delta descriptors without destination history tables | Implemented; removed delta payload copying; mixed-policy regression added |
| R2-9 | Whole-operation rollback for public batch ingestion | Implemented; caught failure and unrelated caller commit regression added |
| R2-10 | Typed current views select metadata/geometry head once | Implemented; deterministic hosted interleaving regression added |
| R2-11 | Correction edits preserve loaded references, mapping and membership baseline | Shared attached-collection snapshot guards and receipt repinning; stale edit/membership/mapping regressions |
| R2-12 | 3D renderer retains listed references | Shared exact selected geometry; renderer regression |
| R2-13 | Consumers use selected geometry instead of ingestion summaries | Shared batched geometry reads for catalogs; native counts/length use displayed geometry; geometry-edit regressions |
| R2-14 | Refuse persisted winding annotations from incomplete intersections | Shared completeness guard before adding annotations; no empty collection precreation; completeness regression |
| R2-15 | Bind persisted ordinal index to its source catalog | Existing catalog digest required before interpreting cached ordinals; changed-source regression |
| R2-16 | Bind solved evidence and visualization to the same source revision | Compact source snapshot recorded in extraction transaction; shared guarded reads used by evidence consumers; stale-source regressions |
| R2-17 | Version-aware completion/resume | Completion keyed by full input reference, not UUID alone; modified-input regression |
| R2-18 | Support valid dense fibers without controls through shared span handling | Shared dense-line span handling; actual store-to-Spiral regression |
| R2-19 | Shared Python store lease through callbacks/GIL release | Implemented through withStore; reentrant-close regression added |
| R2-20 | Canonical UUID keys at hosted cache boundary | Implemented; upper/lower spelling invalidation regression added |
| R2-21 | Identical deleted/retired semantics in cached and uncached queries | Implemented; cache-on/off lifecycle regression added |

Also correct obsolete geometry-token history-switch documentation and MCP export return-field documentation identified in the review.

Shared relocation operates only on private unpublished snapshots. `rebaseExternalLocator` is shared with ordinary override preparation; `relocateExternalLocators` serves copy/pack/conversion. `transformArchivedObjectRecords` reconstructs full archived states, transforms them, then updates child deltas and cached heads together; conversion numbering also uses this helper. These are physical locator changes, not new object versions or copies of external datasets.

The saved normal-evidence format now requires `fiber_source` (store UUID, generation, commit sequence). Older artifacts without this identity must be regenerated. This conservative check also rejects unrelated source commits; no stale-data fallback was added.

The referenced/history-on correction regression exposed a missed no-op sibling: `retainRelationship` performed a second validation after ordinary edge repair had correctly done nothing, queuing an inherited edge for local publication. It now delegates directly to the shared edge writer, which owns both validation and no-op behavior. The existing four source/overlay history-policy regression also covers explicit retention.

Attached collection snapshots retain the querying store UUID, generation and commit sequence as well as exact owner/collection references and transforms. Inherited owners are not republished just because local relationships change; membership is guarded by the existing query commit precondition. Unrelated commits also require reloading the editor baseline. This conservative conflict is explicit, not silently bypassed by refreshing heads during save.

## Verification

Add each reported scenario to permanent regression coverage; distinguish runtime cases from GUI code tracing. Exercise history on/off and referenced/local storage where relevant, retained exact reads, no-op ownership preservation and deterministic hosted interleavings. Run `bash local/scripts/run_vcobj_pr_tests`, the existing consumer checks, and focused neural tests. Record actual commands/results and skipped capabilities here. Review the final diff for repeated resolvers and unnecessary branches; report production/test/doc line changes separately. Do not mark a finding complete on code inspection alone when a regression is feasible.

Focused Python regressions: **78 passed**, followed by **8 passed and one optional dataset skip**. Commands from the repository root (existing environment, no installs):

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=local/vcobj-pr-tests/build/python:local/vcobj-pr-tests/reader-python:vesuvius/src:spiral-fitting:local/vcobj-pr-tests/spiral-build:. /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_fiber_store_versions.py vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_fiber_store_sources.py vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_sheet_strip.py vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_sheet_strip_io.py vesuvius/tests/neural_tracing/test_surface_store_io.py vesuvius/tests/test_fiber_curve_bank.py vesuvius/tests/test_fiber_normal_sign_visualization.py spiral-fitting/tests/test_object_store_inputs.py
AGENTS_AGENT_MODE=1 PYTHONPATH=local/vcobj-pr-tests/build/python:local/vcobj-pr-tests/reader-python:vesuvius/src:spiral-fitting:local/vcobj-pr-tests/spiral-build:. /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/test_fiber_uv_surface.py vesuvius/tests/test_constraint_extraction.py
```

`bash local/scripts/check_vcobj_consumers` passed all listed native consumer translation units, including both extraction/provenance sources. These are compile checks, not full GUI or full-volume extraction execution.

Final integrated validation on Linux, RelWithDebInfo, Python 3.14:

| Command | Result |
| --- | --- |
| `bash local/scripts/run_vcobj_pr_tests` | 53 native tests passed, optional SSH skipped; 202 store Python tests passed; 122 Spiral tests passed, optional real-data test skipped; 13 Lasagna tests passed |
| `bash local/scripts/run_vcobj_consumer_checks` | Eight executables passed, including the correction local/referenced × history-on/off matrix and 15 Fiber3D checks |
| `bash local/scripts/check_vcobj_consumers` | All listed native consumer/extraction translation units compiled |
| Focused Python commands above | 78 passed; then eight passed and one optional dataset skip |
| `git diff --check` | Passed |

The deterministic hosted typed-view race and canonical cache invalidation regressions passed. Native GUI rendering paths were compiled and reviewed for shared selected-reference/context use; no injected GUI rendering race or interactive GUI session is claimed. No macOS or full-volume performance/extraction run was performed. The pre-existing Spiral `PatchSamplingAtlas` nanobind shutdown-leak warning remains; the test process exits successfully.

Production diff versus `3df5b7783`: 420 added / 231 removed, **net +189**. Regression tests and test wiring: 538 added / 60 removed, **net +478**. Documentation: 902 added / two removed, including the previously uncommitted full review reports and this tracker. This pass does not achieve a net production-line reduction: shared relocation/history transformation, loaded editor snapshot guards and frozen-source validation add missing functionality. Duplicate full-record comparators, delta replay, geometry-copy restoration and redundant relationship revalidation were removed; no schema change, persistent table, per-object directory, proxy record or edit-time referenced payload copy was introduced. Review reports and this progress tracker are documentation, not production-code growth.
