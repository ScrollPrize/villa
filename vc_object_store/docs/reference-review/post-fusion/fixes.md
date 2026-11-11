# Post-fusion fixes

Baseline: `153c9a10f`. Implement the nine accepted findings through shared APIs; PF-C03 remains intentionally unchanged. Preserve schema 21, binary UUIDs, history policy and placement-independent resolution. No compatibility fallback or new persistent structures.

- [x] PF-A01: require a transformed surface view's complete selected identity.
- [x] PF-A02: share strict commit-sequence integer validation.
- [x] PF-A03: decode hosted current-object envelopes once.
- [x] PF-A04: validate graph projections through shared native policy without SQL.
- [x] PF-A05: reject generation selectors without a numbered version.
- [x] PF-C01: keep bound-only fiber volume discovery/catalog selection consistent.
- [x] PF-C02: submit no-op surface guards rather than skipping validation.
- [x] PF-C04: reject duplicate saved object membership.
- [x] PF-C05: compare effective selected native coordinate frames for all sources.
- [x] Add permanent regressions; run approved review and PR runners; review the delta.

Use `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks` and `bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests`. Add the Vesuvius regressions to the fixed review runner so validation stays reproducible without a new invocation format. Record results and limitations here when complete.

## Implementation

Surface-grid reads compare the complete selected source identity as well as the immutable geometry identity. Anonymous geometry-only reads remain supported, but cannot certify selected metadata. Generation-only selectors now fail before current lookup. Commit-sequence decoding is shared by snapshot positions and request headers, rejecting fractional, boolean, negative, string and overflowing values.

The hosted server and client share one current-object envelope codec, avoiding nested serialization/parsing while preserving exact metadata text. Graph projection validation calls the existing native field policy without executing SQL; bindings contain no replacement field schema. Tool source frames are normalized through `geometryCoordinates`/`volumeCoordinates`, resolving native metadata at the loaded reference for every source rather than inspecting only the first source. Explicit output-volume selection remains supported.

Unchanged surface updates submit the ordinary expected-object and mapping guards. They neither create a new version nor implicitly repin the tool's read baseline. Duplicate saved object membership, including objects repeated as an outer shell, is rejected. The fiber reader certifies its documented direct volume/scale contract using the native scalar-frame helper; it does not treat registration-mappable fibers as directly bound. Volume discovery now reads all fiber metadata (not geometry); no performance claim is made for this correctness fix.

No schema, persistent tables, UUID representation, history retention, routing or payload ownership changes. No new resolver, fallback format or copy of referenced payloads. PF-C03 remains unchanged as agreed.

## Verification

Linux, GCC 13, core RelWithDebInfo and focused consumers Release, Python 3.14. Both fixed approved runners completed. The review runner exits 0: six focused native tests, two repeat/umbilicus checks, five source-repair tests, ten consumer executables, eleven selected/transformed Python tests, consumer syntax checks and 49 combined fiber-version/Spiral-manifest tests passed. The new native checks exercise surface identity and guard contracts with history on/off; referenced surface and stale no-op cases cover all four source/overlay history-policy combinations, including retained exact reads after deletion.

The full PR runner reports 53 native tests passed (optional SSH skipped), 232 Python store tests passed, 353 Spiral tests plus 54 subtests passed (one optional test skipped), and 37 Lasagna tests plus 26 subtests passed. It exits 1 because the same three previously recorded Spiral failures remain:

- `test_service_commits.py::test_only_a_preview_of_this_session_is_committed`: error wording assertion.
- `test_service_object_store.py::test_clients_cannot_choose_the_binding_or_host_input_paths`: error wording assertion.
- `test_service_object_store.py::test_a_service_packs_remote_arrays_as_it_packs_local_ones`: fake remote-fetch expectation.

The first attempt to expand the fiber checks lacked the existing reader package on `PYTHONPATH` and exposed two older numeric-only test preconditions. The runner path was corrected and those two test fixtures now use `object_reference`; the complete 49-test group passes without exclusions, shims, dependency installation or production ML changes. Existing Torch deprecation and native shutdown diagnostics are not claimed fixed.

`git diff --check` passes. No macOS, GUI-interactive, full-volume or performance result is claimed. The new cases do not individually repeat every hosted/packed/sharded combination; the existing full store suite still runs those shared paths. Changed production code adds 60/removes 37 shared-infrastructure lines (net +23), and adds 12/removes 8 consumer lines (net +4): total net +27, mostly required validation. Regression and documentation growth are separate.
