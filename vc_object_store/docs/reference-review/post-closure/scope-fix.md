# PC-S01 stale-scope fix

Baseline: `b73a31f33`. The accepted source finding and Astra adjudication are in [spatial-linking.md](spatial-linking.md) and [astra.md](astra.md).

- [x] Canonicalize public stale-scope object and touching UUIDs once at `commitCandidatesImpl`, using existing native `util::canonicalUuid`.
- [x] Keep kind strings and empty-scope semantics unchanged; no new normalizer, handle, schema, routing table or fallback.
- [x] Add regressions in the existing `test_object_store_links` executable for both public candidate-commit overloads, source history on/off, local versus referenced endpoints, uppercase/duplicate object and touching IDs, empty object/kind/touching scopes, and reviewed-edge protection.
- [x] Test malformed object and touching IDs in caller-owned transactions: catch failure, commit the outer transaction, verify the edge remains and no partial commit sequence was published.
- [x] Parent ran the existing fixed approved review/PR runners; native link regressions pass. See [final verification](fixes.md#commits-and-verification).

The shared implementation is only three added/two removed production lines (including the explicit UUID header). Ordinary detector scopes already contain canonical IDs from shared selection, so their semantics are unchanged. UUID normalization is performed at the shared native boundary, not duplicated across standalone/caller-owned/internal detector routes. It remains within the operation savepoint and precedes mutation; malformed input fails rather than silently becoming an unmatched scope.

The fixture uses the existing native test helpers and temporary-directory lifecycle. It exercises eight combinations: two history policies × two public commit overloads × local/referenced endpoint storage. This is not a new numerical detector or full-volume benchmark, and no sharded/packed/hosted runtime claim is made for the new case. The boundary change is independent of placement, while existing regression suites continue to cover their shared edge APIs.

Astra's delta review caught and corrected a regression-test weakness: the touching-only case now supplies only the uppercase endpoint, with no lowercase alias that could let the pre-fix code pass. Fixture object UUIDs deliberately contain fixed hex letters so uppercase normalization is exercised deterministically; object-scope duplicate aliases are covered separately.

No build/test invocation or commit was performed by this agent; root owns the approved runner and grouped integration. `git diff --check` passed before handoff. Regression growth is separate from the production delta.

Integration follow-up: committed as `9223b5b22`. The parent completed the focused link executable and full native suite on the final source patch; no new failure remains. Original handoff wording is retained to distinguish this lane's source-only checks from the parent's runtime evidence.
