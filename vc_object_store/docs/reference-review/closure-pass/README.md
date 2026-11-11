# Post-commit closure review

Frozen source: `69ba26311`, containing implementation commit `878525dd1`. The implementation and preceding review evidence were committed before this review. This pass is read-only for production code; the findings below are open, not silently fixed.

## Adjudicated findings

| ID | Priority | Finding | Disposition |
| --- | --- | --- | --- |
| CP-C01 | Medium correctness | Spiral full reads load the outer shell, but incremental notification admission, rereading, state application and overlay installation omit it. Editing/deleting the shell leaves installed geometry/state stale until full reload; a generic freshness warning can still appear. | Astra confirmed; pre-existing, not introduced by the committed fixes. |
| CP-CORE-01 | Low boundary consistency | Python relationship-extension exact reads construct a reference manually, bypassing the strict shared decoder used by the object sibling. Unknown reference keys are discarded rather than rejected. | Astra confirmed; pre-existing. Use the existing relationship reference codec, not a new validator. |
| CP-C02 | Low contract cleanup | Neural displacement output provenance flattens a complete input reference alongside an application `surface` field instead of retaining it as the `reference` component. | Astra confirmed narrowly as a representation cleanup. Full identity is retained today; no geometry corruption or identity loss claimed. |

No new regression was confirmed in the inspected implementation delta. This is not a claim of exhaustive proof or that the remaining issues can be ignored. The source findings need dedicated regressions when fixed; passing existing suites does not close them.

## Evidence and scope

- [Core/transport lane](core.md): invariant traces, sibling entry points, storage versus consumer responsibilities and explicit omissions.
- [Consumer lane](consumers.md): native GUI/CLI and Python workflow coverage and concrete source traces.
- [Independent Astra adjudication](astra.md): independent delta/placement review, all candidates and focused native runtime evidence.
- [Root integration traces](root.md): cross-lane checks and latest shared helpers.

No schema change, dependency installation, dataset mutation or production edit occurred in this review. Geometry algorithms, full-volume performance, interactive GUI execution and macOS/arm64 are not newly validated here.

## Verification

Astra independently ran 12 existing focused native tests successfully. The root ran `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks` on the frozen source; it exited 0. Seven focused native tests, two repeat/umbilicus checks, five source-repair tests, ten consumer executables, twenty selected-value/reference tests, consumer syntax checks and 59 combined fiber/manifest/grower tests passed. The existing Torch Python 3.14 deprecation warnings remain. `git diff --check` passed.

The preceding full PR run, recorded in the committed final-mismatch fixes, passed 53 native, 238 store Python, 358 Spiral and 37 Lasagna tests with the same three known baseline Spiral failures; it is prior evidence, not a newly rerun full suite. This review adds documentation only, so there is no production line delta or performance claim.
