# Single-reference implementation validation

Baseline `38c72d972`; implementation commits `d9b8a58f2`, `272db30b9`, `dc396eca6` and `5d1aa2eb8`. Schema 21 and the binary database formats are unchanged.

## Commands

Final verification uses only the existing fixed, argument-free runners:

```sh
bash local/scripts/run_vcobj_pr_tests
bash local/scripts/run_vcobj_consumer_checks
bash local/scripts/check_vcobj_consumers
```

The PR runner was extended locally to include the additional changed Spiral service/headless suites and Lasagna approval-inpaint tests. Native consumer checks run the actual shared reader/editor/geometry code; syntax checks also cover SpiralPanel, the project-input fixture and the final Lasagna producer. These local build/environment scripts are not part of the submitted source.

## Completed checks

The original PR suite completed successfully after the shared conflict-classification fix: 53 native tests passed (optional SSH skipped), 222 Python store tests passed, 124 Spiral tests passed (one optional real-data skip), and 15 Lasagna tests plus six subtests passed. Native consumer runtime checks and the affected GUI/CLI syntax checks passed. This includes standard/legacy input selectors, exact input after deletion, receipt and pin changes, typed selected graphs, source-owner guards and config argument transport.

The extended run completed with 352 Spiral tests passed, three failed, one skipped and 54 subtests passed. The combined Lasagna suite passed all 34 tests and six subtests. Native and Python store results remained as above. The extended runner preserves its nonzero exit status; failures are not silently deselected.

The three extended failures also reproduce using baseline `38c72d972` Python source under the same built native/dependency environment:

- `test_service_commits.py::test_only_a_preview_of_this_session_is_committed`: expected error-message substring differs from the actual message.
- `test_service_object_store.py::test_clients_cannot_choose_the_binding_or_host_input_paths`: expected error-message substring differs from the actual message.
- `test_service_object_store.py::test_a_service_packs_remote_arrays_as_it_packs_local_ones`: expected a recorded remote GET, but the list is empty.

The baseline control ran through the same approved runner in an automatically cleaned fixture. This is a baseline Python-source comparison, not a separate pristine native rebuild. These unrelated failures were left unchanged rather than expanding this reference-cleanup task. The existing Spiral `PatchSamplingAtlas` shutdown diagnostic remains. No full-volume solve, interactive GUI session or macOS/arm64 build is claimed.

## Size and scope

Across the implementation commits, production changes are **470 added / 439 removed: net +31 lines**. Consumer production code (Lasagna and Spiral) is 60 lines smaller; shared/native infrastructure is 91 lines larger. Regression changes are 267 added / 97 removed (net +170); specifications/review documentation are separate. Counts exclude Markdown, Python/C++ test paths and native `*Test.inc` fixtures; the new native codec header is included.

This is **not** the large net line-count reduction initially expected. The simplification removes competing internal reference representations, repeated resolution/serialization, dual pin maps and unused APIs. Shared native external-input decoding and explicit validation/compatibility offset the removed consumer code. No code was compressed to make a misleading line-count claim.
