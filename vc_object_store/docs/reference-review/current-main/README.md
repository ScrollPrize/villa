# Review after current-main integration

Frozen review commit: `7b08dfebd030168a6127f43e060301c55c7c20c3`. Merged `origin/main` at `f0e441016`; the merge was conflict-free. The earlier push integration at `9d15061bc` also merged upstream main at `08df18e46`. Review the combined upstream changes since the last reviewed production commit `371961148`, not only the final nine-file merge. No vcobj or volume-cartographer production file changes in that combined upstream delta.

Existing uncommitted [post-receipt review records](../post-receipt/README.md) are retained, not overwritten. Production and tests remain unchanged during review. Reports distinguish source inspection, runtime results, prior unchanged-code evidence and exclusions.

## Scope

- [Native lane](native.md): store and consumer contracts, especially integration with reader caching and shared native reader APIs.
- [Consumer lane](consumers.md): Python/native consumer identity handling and cumulative upstream Spiral/reader/data-loader changes.
- [Independent Astra adjudication](astra.md).
- Coordinator: cumulative commit/file inventory; reader Python/native API changes and test-environment alignment; upstream Spiral pair sampling, satisfaction memory fallback and theta-map lifetime changes; unchanged shared receipt/selection/reference contracts.

Unrelated website/prize automation changes are preserved from upstream and excluded from the vcobj review. Independent numerical/model redesign is outside scope. No dependency installation, full-volume solve or user dataset mutation is authorized by this review.

## Validation

Use the existing argument-free approved commands `bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests` and `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks`, sequentially because they share build products. The latter also includes the three upstream Spiral integration test files for relative-winding pair sampling, packed satisfaction and theta crossings. Both runners use automatically cleaned temporary fixtures.

The PR runner completed: 54 native tests passed (one optional SSH skip), 333 store Python tests passed, 369 Spiral tests plus 54 subtests passed (one optional skip), and 37 Lasagna tests plus 31 subtests passed. It exits nonzero for the same three previously recorded Spiral failures: two message-expectation mismatches (`test_only_a_preview_of_this_session_is_committed`, `test_clients_cannot_choose_the_binding_or_host_input_paths`) and the fake remote-fetch expectation (`test_a_service_packs_remote_arrays_as_it_packs_local_ones`). No new failure appeared in this run.

Validation provenance: this full runner rebuilds native reader dependencies of vcobj from current sources, but its historical Python-reader staging still copies binaries from another checkout. The review runner therefore configured/built matching reader Python/native modules from this checkout, ran reader cache/channel controls and reran the consumer/Lasagna checks with those modules. The old-binary run is not certification of new Python-reader APIs. No dependencies were installed and the approved command paths remain unchanged.

The matching-build review runner completed with exit 0. Six native reader cache/channel tests and nine reader Python tests (plus eleven subtests) pass, including local HTTP ranges/revision changes and fork/pickle registration. Eleven focused native store targets, two repeat/umbilicus checks, 25 source-import tests, ten consumer runtime executables, all configured consumer syntax checks, 102 selected store Python tests and 85 combined consumer/position tests pass. The three upstream Spiral integration files pass with 27 tests and one platform skip. Lasagna rerun with matching reader modules passes with 36 tests and 31 subtests; its hosted-server class skips one test because this runner does not set `VC_OBJECT_SERVER` (the full runner above did run it). The reader build is CPU-only. Existing Qt environment, PyTorch and Python fork deprecation warnings remain. `git diff --check` passes. These are merged-tree executions, not substituted previous-review counts.

## Source-review verdict

Native and consumer reviewers found no new confirmed store/consumer defect; Astra independently agrees. No production fix was made during this pass. The latest main merge and cumulative reader/Spiral integration preserve the existing schema, binary identifiers, complete-reference transport, shared native routing and checked-write boundaries. This verdict does not claim exhaustive cache concurrency correctness, identical stochastic model trajectories, GPU/interactive GUI execution, macOS/arm64 validation or full-volume performance.
