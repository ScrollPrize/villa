# Final-mismatch fixes

Starting state: `153c9a10f` plus the completed PF working patch, SHA-256 `286b96ae90b5960043810003370f6ec3a66110597251dbcdd39477b313501235`. Preserve those changes. Astra pre-fix adjudication is recorded in `pre-fix-astra.md`; the original review evidence remains in this directory.

- [x] FM-A01: protect artifact output through the shared backing-store closure validator.
- [x] FM-P01: retire track-grower outputs using complete selected write guards.
- [x] FM-P02: share strict native position decoding at saved/service ingress.
- [x] FM-P03: preserve and validate numbered surface selector values at the native boundary.
- [x] FM-C01: extract shared native point-map decoding; remove the weaker GUI parser.
- [x] FM-C02: share saved manifest object/relationship identity validation before native/Python filtering.
- [x] FM-A02: resolve physical memberships once, then use complete relationship publication references in memory, preserving precedence and applicability, with explicit canonical result order.
- [x] Add permanent regressions, review the delta and run the fixed approved review/PR runners.

Schema 21, binary UUIDs, history retention and placement semantics stay unchanged. No proxy rows, replacement identity formats, copied referenced payloads or compatibility fallback. Public file materialization remains atomic per file, not a newly promised all-files transaction. Report production, test and documentation line changes separately.

## Implementation and review

Astra approved all previously unreviewed findings before implementation in [pre-fix-astra](pre-fix-astra.md). [Post-fix review](post-fix-astra.md) approved the final inspected source delta, including the lazy-archive follow-up. Its one blocking build-interface observation was fixed: the public PointMap header's existing nlohmann JSON dependency is now PUBLIC on `vc_geometry`, while the reader catalog dependency stays PRIVATE. No dependency installation was needed.

Artifact writes call the same local backing-closure destination validator as geometry export before creating output directories or publishing bytes. Normal safe writes and hosted client-local output keep their existing semantics. The track grower selects complete object references in one query instead of submitting numeric write expectations; empty retirements avoid a query.

Position and surface selector parsing delegates strict integer/UUID handling to native codecs. The service retains its own binding, receipt and minimum-position rules but no longer maintains a second numeric position schema. The obsolete optional version argument on the private provenance-map helper was removed; its only caller carries complete references.

`geometry/PointMap.hpp` centralizes numeric IDs, numeric-alias rejection, finite XYZ, ordering and annotation extraction. The existing file importer retains its document-wide collision/renumber/link rules; the GUI-to-store adapter no longer has a permissive partial parser. Unrelated standalone uint64 JSON formats were not silently narrowed to native int64 IDs.

`manifestReferences` owns saved object/outer-shell/relationship reference validation and duplicate membership checks. Native public manifest selection, full and incremental comparison, and the Python validator call it before filtering. Native pins retain the returned complete reference rather than discarding it. Valid stale references do not require old geometry to remain readable merely to compare current content.

Selected relationship memberships and deduplication use the complete `RelationshipVersion`. Physical archive UUIDs remain only at the storage-resolution boundary; superseded archive entries are not decoded, and each surviving distinct archive is decoded once. Root-first scope precedence, per-endpoint admission and applicability are preserved. One native ordering helper serves local results and wire/cache decoding. The intentional output-order change is documented: relationship UUID then full reference replaces incidental archive/synthetic-string lexical order; fiber branch arrays may reorder, but their contents are unchanged. No old synthetic identity is retained just for sorting.

## Regression scope

Permanent tests cover protected root/dependency/sidecar/artifact paths and cache symlink aliases across all four source/overlay history-policy pairs; safe materialization; native and wrapper malformed scalar selectors; saved position/pin parsing; point IDs/XYZ/annotations; public native manifest selection and full/partial comparison; duplicate shell membership; native manifest decode-once/reference reuse; grower retirement; and wire ordering for retained graph-cache values. Existing packed, sharded, hosted and current/exact graph tests remain part of the full native run; not every new counterexample is independently repeated for every placement.

Fixture modernization was explicit: a native artifact test used the shared session reader instead of a nonexistent ObjectCollection method; native fit manifests now include complete references and represent shells separately; a grower test uses current SurfaceGrid fields instead of removed tuple unpacking; fake service IDs are valid UUIDs so the unavailable-service assertion tests its intended failure; the decode-once spy now observes the shared native validator rather than removed Python parsing. No production compatibility shim or validation relaxation was added.

## Verification

Commands: `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks` and `bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests`. Linux/GCC 13, native RelWithDebInfo, focused consumers Release, Python 3.14. Final totals follow. No macOS, interactive GUI, full-volume or performance result is claimed.

Final approved review runner exits 0: seven focused native tests (including artifacts), two repeat/umbilicus checks, five source-repair tests, ten native consumer executables, twenty selector/selected-value/transformed-view tests, consumer syntax checks and 59 combined fiber/manifest/grower tests passed. The subsequent final full run also exercised the lazy archive-resolution version: 53 native tests passed, optional SSH skipped; 238 Python store tests passed; 358 Spiral tests plus 54 subtests passed, with one optional skip; 37 Lasagna tests plus 26 subtests passed.

The full PR runner exits 1 solely because the same three pre-existing Spiral failures remain: `test_service_commits.py::test_only_a_preview_of_this_session_is_committed` (wording), `test_service_object_store.py::test_clients_cannot_choose_the_binding_or_host_input_paths` (wording), and `test_service_object_store.py::test_a_service_packs_remote_arrays_as_it_packs_local_ones` (fake remote-fetch expectation). No new failing test remains in this task's runs. Existing Torch/dependency deprecation and shutdown diagnostics are not claimed fixed. `git diff --check` passes.

## Line accounting

Relative to the preceding PF working state, this FM pass has net **+134 shared-infrastructure lines** and **−28 consumer lines**, net **+106 production lines**. Required shared validation and four small shared-code files account for the growth; this is not claimed to reduce total production lines. Consumer-specific parsing/validation and synthetic relationship identities were removed rather than retained beside the new APIs.

For a reproducible cumulative comparison against HEAD `153c9a10f` (which also includes the earlier uncommitted PF pass): shared production +238/−81 = +157, consumers +48/−72 = −24, total production +286/−153 = +133. Subtract the recorded PF production net +27 to obtain this pass's +106. The four new shared source/header files contribute 82 added lines and are included in those totals. Regression-test changes are separately +276/−18 = +258 cumulatively; documentation/review logs are excluded from production/test counts. No commit or push was requested or performed for this task.
