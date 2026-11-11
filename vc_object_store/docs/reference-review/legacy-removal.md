# Remove legacy assumptions from the full-reference model

Baseline: `510cc7fe1`. Goal: one current full-reference model, not a union of old numeric and new selected identities. Numeric input convenience, when explicitly requested, resolves and checks through the shared reader at the API boundary and never enters transaction/preparation state as a number. Already selected consumers pass the original reference, never late-resolve discarded identity.

- [x] Remove numeric `RecordExpectation` alternatives and implicit numeric/map conversions.
- [x] Emit and accept only current command format 3; remove command-1/2 modes, side-channel maps, digest reconstruction and merge-strengthening branches.
- [x] Keep only explicit numeric selection/write entrypoints that immediately check requested live counters and produce complete handles. Preserve null as no-live, not an exact tombstone.
- [x] Remove weak relationship-manifest comparison and require complete references at the manifest boundary, including missing/dropped/out-of-scope entries.
- [x] Remove application legacy pin/provenance reconstruction and unsupported weak comparisons rather than guess missing owners.
- [x] Use shared JSON-value reference codecs without internal text round trips or repeated interpretation.
- [x] Migrate callers and regressions; use only approved fixed runners.
- [x] Record supported behavior, intentional limitations, source/test deltas and validation.

Limits agreed with Astra's source review: an unqualified old numeric request cannot recover its historical owner/lineage. Old command formats are rejected, not automatically upgraded. Unresolved durable requests may already have committed: regenerate only after explicit reconciliation, never automatically under the old request UUID. Old incomplete manifests require regeneration. Retained store/history records remain readable through the current reference model; no database schema or history layout redesign is authorized.

Astra's final source check found two issues, now fixed: missing-payload replay must validate the same header/guards before looking up a saved receipt, and explicit numeric selectors must reject duplicate canonical UUID keys. `requestHeader` now serves decoding, payload/preparation discovery and retry validation. Native regression coverage includes old formats, numeric guard rejection, duplicate UUID aliases, exact tombstones, inherited owners and current-format replay. No additional source blocker was found; source review is separate from the validation results below.

## Validation and size

Used only the existing fixed, argument-free approved commands:

```sh
bash local/scripts/run_vcobj_pr_tests
bash local/scripts/run_vcobj_consumer_checks
bash local/scripts/check_vcobj_consumers
```

Final Linux RelWithDebInfo native build: 53 tests passed, optional SSH skipped. All 223 Python store tests passed. Spiral: 352 tests and 54 subtests passed, one optional real-data test skipped; the same three service failures previously reproduced against baseline remain (`test_only_a_preview_of_this_session_is_committed`, `test_clients_cannot_choose_the_binding_or_host_input_paths`, `test_a_service_packs_remote_arrays_as_it_packs_local_ones`). The PR runner preserves exit status 1, not a false green result. The earlier baseline control is recorded in [validation](validation.md); it was not repeated against the new incompatible command protocol. Lasagna: 34 tests and six subtests passed.

The separate consumer runtime runner passed, including the final rebuilt receipt decoder, source-repair tests, selected geometry/catalog/manifest/editor checks and all 20 fiber-map staleness tests. The affected native GUI/CLI syntax checks and `git diff --check` passed. This is not a full GUI build, interactive session, full-volume solve or macOS/arm64 validation.

Production delta versus `510cc7fe1`: 242 added, 231 removed, net +11. Shared store/bindings account for +15; external consumer production code is −4. Tests add 340 and remove 259 (net +81); documentation is counted separately. This is fewer supported paths and representations, not a claim of a large net line reduction. No schema, persistent identity tables, routing directories, history layout or solver/model semantics changed.
