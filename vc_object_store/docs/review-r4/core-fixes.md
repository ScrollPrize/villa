# R4 shared-core fixes

Baseline `2b5d1e01f`. C-002/C-003/C-004 implemented through shared structural allocation, artifact publication and region-query infrastructure; schema 21 and binary identities remain unchanged. No referenced payload copies or persistent tables/directories added. Validation is recorded below after completion.

- C-002: foreign structural slices cannot reuse IDs allocated previously by their target. References to the target's own retained geometry remain valid restoration. Tests cover local/referenced targets with history on/off, rejected checked-action commits leaving state unchanged, and legal restoration when that geometry is retained.
- C-003: a loaded-tool publisher preserves its complete selected reference, write lineage and optional loaded coordinate-mapping guard. Numeric convenience publication keeps its documented current-head semantics. Both overloads share manifest merge/upload/submission; mapping validation uses the ordinary checked operation executor. Tests distinguish generation-reset rejection for loaded tools from valid explicitly current publication.
- C-004: session region queries refill bounded backend pages lost to draft/attachment precedence, skipping already returned UUIDs before backend object admission. Budgets and cancellation counters stay cumulative, including output discarded due to precedence; visible intersection counts are recomputed. No unbounded prelisting or budget reset. The runtime exclusion option travels through the ordinary region-query wire codec only when nonempty; ordinary empty-option requests retain their previous wire shape. Existing frame traversal and session refill share statistic accumulation. Limits remain per attachment/draft group, not global.

## C-006 correction: no defect to fix

The original discovery and independent adjudication incorrectly assumed that `StatementPool::idle` is a unique-key map. It is actually `std::unordered_multimap<std::string, sqlite3_stmt*>` (`internal/Database.hpp:83`), unchanged in the frozen baseline. Duplicate SQL keys are accepted and both handles remain owned. A proposed insertion-success check failed to compile because multimap `emplace` returns an iterator, revealing that incorrect premise; that production edit was fully reverted. There is no statement-wrapper production change for C-006.

A regression now constructs overlapping identical statements, verifies that both handles are tracked, tears down the pool and checks successful `sqlite3_close`. This is evidence for withdrawal, not a fabricated fix. Parent review summaries must reflect the correction.

## Validation

Focused build uses the existing `local/vcobj-pr-tests/build` RelWithDebInfo configuration and synthetic self-cleaning native fixtures. No installs, GUI/full-volume performance run or macOS claim.

```sh
cmake --build local/vcobj-pr-tests/build --parallel 4 --target test_coherent_store test_object_store_artifacts test_object_store_surface_drafts test_object_store_session
ctest --test-dir local/vcobj-pr-tests/build --output-on-failure -R 'object_store_(coherent_api|artifacts|surface_drafts|session)$' --parallel 2
```

Final result: **4/4 passed, 3.96 seconds**. Region regressions exercise draft displacement, an off-screen earlier owner, the wire exclusion codec, visible intersection counts, charged discarded output and a strict processing budget. Structural regressions cover both local and inherited targets with history on/off, retained-ID rejection and legal own-source restoration. Artifact regressions exercise the selected session overload and actual ToolObjects path after an active-history reset; the existing numeric convenience test still succeeds. Native consumer tests additionally exercise loaded mapping changes for both geometry and artifact writes; their final result belongs to the native report.

Initial fixture failures were corrected rather than weakening contracts: surface fixture z must be positive, and failed structural checked actions reject their outer commit. One test attempted a nonexistent StoreReader region method and was changed to the actual public ObjectStore API. `git diff --check` passes. No active builds or commits remain in this lane.
