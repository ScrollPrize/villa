# R5 full coherent-store inspection

Implementation follow-up: [R5 fixes and verification](fixes.md).

Frozen baseline: `bea32d5d807731d8c3b2db17c815cde748523226`. Worktree was clean; the prior fixes were already committed in `2296c7227`, `8ddb85fab`, `876dacdde`, and `bea32d5d8`, so no additional implementation commit was needed before review.

Discovery model: GPT-6.1 Sol. Independent adjudication model: GPT-6 Astra. Inspection only: no production fixes, installs, new test commands or environment setup. Any needed execution must use the existing fixed pre-approved runner. Prior successful tests are not evidence that a new scenario passes.

## Protocol and coverage

Refresh the previous file inventories from tracked files and caller closure, review all assigned store-facing paths, and disposition every file. Do not cap findings, ignore unchanged code or declare an uninspected file clean. Boundaries are the store-facing identity, version, geometry/mapping, lifecycle, mutation, cache, transaction, transport and workflow paths; unrelated numerical/rendering/training bodies and test harness internals are not exhaustively mathematically audited.

Every candidate gets a separate `findings/<ID>.md`: priority, exact source location, supported reachable trigger, actual versus required behavior, evidence/reproduction status and a shared fix direction. Architectural duplication without demonstrated failure must be labeled as such, not silently promoted to a correctness defect. Check concrete types, contracts and callers: R4's independently accepted C-006 was later disproved because the actual container was a multimap. Historical fixed or rejected findings require new evidence before reopening.

Every candidate must then receive a separate Astra decision in `adjudication/<ID>.md`: confirmed, rejected, duplicate or unresolved, with corrected scope/severity and reasons. Preserve original evidence and explicit superseding corrections. Review remains open until coverage complements and all candidate dispositions are complete. An empty complement is coverage evidence, not proof of bug freedom.

| Lane | Scope | Ledger | Status |
| --- | --- | --- | --- |
| Core | Logical core, geometry/history/query/mutation/extensions/session | [122 files](core-coverage.md) | Complete |
| Physical/storage | Databases, collections/shards/VFS, import/export, transport, linking, CLI/build/scripts | [118 files](storage-coverage.md) + [15 CLI/build entries](cli-coverage.md) | Complete |
| Native consumers | Native GUI/CLI/async/cache/workflow callers outside the store | [145 files](native-coverage.md) | Complete |
| Python/bindings | Binding adapters and Python callers across subprojects | [76 files](python-coverage.md) | Complete |

The ledgers contain **468 unique paths** after deduplication. A tracked-file complement check found no omitted production/build/script paths inside `vc_object_store/` (docs and test-only files excluded from that complement). External native/Python consumers use tracked discovery plus caller closure; each ledger states its inventory and intentional exclusions. The Python lane additionally followed indirect imports not included in the previous inventory. Test sources were consulted where relevant, not audited as production or executed.

## Result

**24 candidates adjudicated: 23 source-confirmed defects (20 P2, three P3), plus one P3 architecture item.** No rejected, duplicate or unresolved candidates. The [independent decision table](validation-summary.md) links each separate decision; the original candidate files remain in `findings/`. The agreement is source-level evidence, not a runtime guarantee. The Python anchor-codec duplication is not counted as a runtime defect.

The findings group into shared boundaries rather than 24 independent workarounds:

- **Geometry validity:** canonical validity is applied in transformed coordinates in native reads and inverse edits; graph point dependencies also need to honor the selected validity channel. See [N-001](adjudication/N-001.md), [C-001](adjudication/C-001.md), [C-007](adjudication/C-007.md).
- **Frozen evidence:** UV preparation/replay and Spiral relationship membership need selected source and graph state, not a fresh digest or immutable endpoint references alone. See [P-001](adjudication/P-001.md), [P-004](adjudication/P-004.md), [P-005](adjudication/P-005.md).
- **Publication and failure boundaries:** correction close must honor failed saves; durable checkpoints must not publish unstaged requests; conversion mapping, cache and tracing files should use shared staging/publication; caller-owned link batches need the common operation savepoint. See [N-002](adjudication/N-002.md), [C-004](adjudication/C-004.md), [S-005](adjudication/S-005.md), [S-006](adjudication/S-006.md), [S-003](adjudication/S-003.md), [P-002](adjudication/P-002.md).
- **Identity, ownership and policy:** canonicalize UUIDs before cursor/filter comparison, use selected closure information for copy/extension decisions, preserve managed metadata during catalog refresh, and keep resource identity/ordering independent of spelling and history policy. The individual decisions separate these contracts from minor CLI argument validation and the architecture-only anchor codec item.

No production or test changes, builds, dependency installs, server launches, or runtime probes were made. No new runner or fixture path was introduced. Use the existing approved runner(s) for the subsequent regression pass.

## Attribution to the last fixes

Source comparison of `2b5d1e01f..bea32d5d8` found the proposed trigger logic already present before R4 implementation. In the two candidate-bearing files touched by R4, the R4 changes are to `writeObjectCopy` (savepoint, not `readObjectCopy`'s extension guard) and `ObjectStoreSession` artifact/region code (not its durable checkpoint sequencing). The other implicated code paths are unchanged in that range. This is source-level attribution, not a before/after runtime test, and does not identify the original introducing commit for each older issue. Several are missed callers or layers of an invariant previously fixed in another path rather than new behavior introduced by the last fix commits.
