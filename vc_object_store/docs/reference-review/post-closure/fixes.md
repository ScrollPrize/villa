# Post-closure fixes and next review

Baseline: `b73a31f33`. Implement the three independently adjudicated findings using shared store infrastructure, then rerun the approved checks and review the resulting snapshot. No schema changes, identity directories, copied dependency payloads or legacy compatibility branches.

- [x] PC-A01: shared tombstone-aware admission for metadata imports; preserve idempotence, conflict handling and whole-operation rollback.
- [x] PC-CORE-01: reuse native position/version validation for selection artifacts, Python ingress and record metadata; preserve local publication semantics.
- [x] PC-S01: canonicalize stale-link scope UUIDs once through shared UUID handling.
- [x] Add sibling regressions, run approved runners, review delta with Astra, record production/test line accounting.
- [x] Run another review with explicit file/operation dispositions, candidate adjudication and unchecked-scope limits.

## Implementation

Metadata imports now use the existing deleted-aware `objectSource`/`edgeSource` resolution for same-kind and cross-kind admission, without a new helper, SQL directory or restore path. Live idempotence/conflict behavior and the operation savepoint remain unchanged. [Import regressions](import-fix.md) exercise local history on/off and all four referenced policy pairs, preserve inherited ownership, and catch an error before committing unrelated outer work to prove earlier imported rows roll back.

Selection encode/decode and Python selection ingress reuse `storePositionFrom`; record metadata/projected records and numbered wire-envelope siblings reuse `commitSequenceFrom`/`versionNumberFrom`. Nullable commit positions and absent/null optional projected record fields retain their documented distinctions. Source metadata counters still republish as local counters on import. A newly reproduced fractional selection `count` was folded into the same scalar fix: count must be an integer equal to the actual byte-derived entry count, with the existing remainder check retained. Binary schema and entry layout are unchanged; the unrelated list-container shape gap is separately logged for the next pass.

Stale object/touching scopes canonicalize UUIDs once with the existing utility at the shared native entry point. The [scope regression](scope-fix.md) tests deterministic letter UUIDs, uppercase-only touching, duplicate object spellings, both public overloads, local/referenced endpoints, history on/off, empty scope semantics, reviewed-edge protection and caught-error outer commit. No detector algorithm or eligibility policy changed.

[Astra's delta review](fix-astra.md) found no production blocker. Its requested uppercase-only test correction is included; no lowercase alternative now masks that counterexample. Native scalar tests use public decoding APIs rather than requiring private header visibility. Initial new-test compilation errors (private include, then assertion message type) were corrected without changing production behavior or test build dependencies.

## Line accounting

Production +40/−30 = net **+10 shared-infrastructure lines**; no production consumer growth. Required shared admission/validation accounts for that small increase; repeated scalar/UUID policy was not copied into consumers. Regression tests add 225 lines separately. Review documentation and ignored local runner configuration are excluded. No persistent schema, binary UUID representation, retention policy, routing directory or referenced payload copy changed.

## Next review

[The next review](../iteration-next/README.md) is complete with explicit source freezes, re-opened selection obligations, per-file/caller evidence and independent Astra adjudication. It reports three low-priority, pre-existing issues, not regressions: obsolete relationship-wire array handling, selection list-container shape validation, and Spiral receipt-marker reference reconstruction. They are open for the next fix pass, not silently included here. Specified runtime checks do not prove all other behavior free of defects.

## Commits and verification

The fixes are separate commits: import `56dc9c792`, scalar validation `395e86ace`, stale scopes `9223b5b22`. The final source/test patch relative to `b73a31f33` hashes to `89d698050e0e41d28aa8ec2acb55129048467147b2e40a5f27feb1094fe4daab`, identical to the reopened review snapshot. Source stayed unchanged during final validation.

Commands remain the approved argument-free `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks` and `bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests`. Existing Linux/GCC 13 environment, core RelWithDebInfo, focused consumers Release, Python 3.14. No dependencies installed; temporary fixture directories are automatically cleaned. No new real-volume, interactive GUI, macOS/arm64, performance or concurrency-stress claim.

Final full PR runner: 53 native tests pass, optional SSH skipped; 244 Python store tests pass; 358 Spiral tests plus 54 subtests pass (one optional skip); 37 Lasagna tests plus 26 subtests pass. Exit 1 is solely the same three pre-existing Spiral failures: `test_only_a_preview_of_this_session_is_committed`, `test_clients_cannot_choose_the_binding_or_host_input_paths` (wording expectations), and `test_a_service_packs_remote_arrays_as_it_packs_local_ones` (fake remote-fetch expectation). Existing dependency/Torch warnings are unchanged. Earlier runs before the count correction are not substituted for this final run.

Final approved review runner exits 0: nine focused native tests (including generations and selections), two repeat/umbilicus checks, five source-repair tests, ten consumer executables, 33 selected-value/reference/extension tests, consumer syntax checks and 66 combined fiber/surface/manifest/grower tests pass. `git diff --check` passes. Results apply to the exact committed source snapshot above; only documentation changes followed.
