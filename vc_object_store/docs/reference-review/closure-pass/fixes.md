# Closure-pass implementation

Baseline: `70da58935`. All three findings were independently accepted by Astra before implementation; see [review](README.md). Preserve schema 21, binary UUIDs, retention and shared placement-independent resolution. No legacy compatibility branch or new persistent structures.

- [x] CP-CORE-01: use the existing native relationship reference decoder at the Python extension boundary; test object/relationship and store/transaction parity.
- [x] CP-C01: enumerate manifest inputs once with their shell role, share full/incremental reading and GUI surface installation, and refresh/count/merge shell states without treating shells as fitting patches.
- [x] CP-C02: retain the selected source reference as a component of derived-surface provenance; update producer regressions without a legacy shape fallback.
- [x] Run approved runners, inspect the delta, record line accounting and verification limits.

## Shared implementation and review

`extension_rows_at_version` uses the same native full-reference decoder for relationship and object owners. The regressions use one malformed-input matrix at both store and transaction boundaries; missing fields retain the existing native JSON exception, while invalid keys/numbers use the existing validation exceptions.

Manifest object enumeration now carries one shell-role flag beside the unchanged complete reference. Discovery/notification admission, full reads and incremental reads share that enumeration. Both read paths use `readObject`; shell state, geometry and problems remain in the distinct shell result rather than being copied out of a temporary patch result. Count/unchanged and partial merging include the shell. Fitting endpoint kinds and incident-endpoint discovery explicitly exclude the shell role, retaining the original solver semantics. Full and incremental GUI installation use the same `fitSurfaceChanges` builder for category, key, replacement and removal; the old separate missing-patch path is gone.

The shell regression compares partial updates with full reads for metadata/geometry edits and deletion, including counts, state and surface grid. It runs local history on/off and all four source/overlay retention-policy pairs. A real stored shell-to-PCL edge tests that the broader notification enumeration does not broaden fitting membership. An unrelated object refresh leaves the shell untouched. The new fixture explicitly removes that edge and supplies the expected geometry revision before replacing the surface, as required by ordinary checked writes; no validation was relaxed.

Derived output provenance now stores `input.reference` directly and removes the redundant `surface` field; its existing source UUID list is unchanged. The regressions check both the original selected input and a chained output, including use of the actual selected preceding version. No old-format fallback is added.

[Astra's independent delta review](fix-astra.md) found no blocking defect or new regression. Notification admission and GUI overlay replacement/removal are source-traced, not an interactive event-loop test. Existing test fixtures cover ordinary store routing; no macOS/arm64, full-volume performance or additional concurrency stress result is claimed.

## Line accounting

Production: shared binding +1/−1 (net 0), consumers +41/−48 (net −7), total +42/−49 (net **−7**). Regression tests are separately +84/−5 (net +79). Review/task documentation and the ignored local runner configuration are excluded. No schema, UUID format, persistent table or retention change.

## Verification

Ran the same approved, argument-free commands: `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks` and `bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests`. Existing Linux/GCC 13 build and Python 3.14 environment; native RelWithDebInfo, focused consumers Release. No installation or bootstrap was required. Temporary fixture directories are automatically cleaned by the runners. Inputs are the checked-in temporary-store fixtures, not a new full-volume experiment.

The review runner exits 0: seven focused native tests, two repeat/umbilicus checks, five source-repair tests, ten native consumer executables (including the extended shell matrix), 27 selected-value/reference/extension tests, consumer syntax checks and 66 combined fiber/surface-provenance/manifest/grower tests pass. The existing runner now permanently includes the extension and derived-surface provenance files without changing its invocation.

The full PR runner passes 53 native tests (optional SSH skipped), 238 Python store tests, 358 Spiral tests plus 54 subtests (one optional skip), and 37 Lasagna tests plus 26 subtests. Its exit 1 is exclusively the same three baseline Spiral failures: `test_only_a_preview_of_this_session_is_committed`, `test_clients_cannot_choose_the_binding_or_host_input_paths` (both wording expectations), and `test_a_service_packs_remote_arrays_as_it_packs_local_ones` (fake remote-fetch expectation). No new failure remains. Existing dependency/Torch warnings are unchanged. `git diff --check` passes.

Two initial fixture corrections were needed: supply the expected geometry revision and explicitly remove the attached edge for replacement; and assert the shared decoder's existing RuntimeError for a missing JSON field, rather than expecting a different Python exception class. Production validation was not relaxed to make tests pass. No commit or push was performed in this implementation turn.
