# R5 independent Astra validation

Frozen source baseline: `bea32d5d807731d8c3b2db17c815cde748523226`. Discovery: GPT-6.1 Sol. Adjudication: GPT-6 Astra. Every candidate currently present in `findings/` has a separate decision under `adjudication/`; original candidate files were preserved.

This is source-based adjudication, not runtime reproduction. The review checked the actual function overloads, field/container types, validation order, caller paths and supported contracts behind each candidate. No production or test code was changed, no environment was installed, and no builds, test fixtures, servers, conversion commands or destructive probes were run. Existing regression source is cited as contract evidence, not as a newly executed test result. This document does not certify whole-repository coverage; the discovery ledgers own that separate claim.

## Dispositions

| ID | Decision | Priority | Independently checked failure boundary |
| --- | --- | --- | --- |
| [C-001](adjudication/C-001.md) | Confirmed defect | P2 | Changed transformed surface positions are tested against canonical validity before inverse mapping. |
| [C-002](adjudication/C-002.md) | Confirmed defect | P2 | Measured/volume selection compares a raw cursor spelling to canonical UUIDs. |
| [C-003](adjudication/C-003.md) | Confirmed defect | P2 | Selected-object copy's root-only extension flag misses inherited immutable result tables. |
| [C-004](adjudication/C-004.md) | Confirmed defect | P2 | Checkpoint can publish an in-memory pending request before its replay files are staged. |
| [C-005](adjudication/C-005.md) | Confirmed defect | P2 | Automatic keep-recorded decisions reject unrelated metadata edits with local history-off edges. |
| [C-006](adjudication/C-006.md) | Confirmed defect | P2 | Catalog refresh replaces independent current metadata/artifact manifests. |
| [C-007](adjudication/C-007.md) | Confirmed defect | P2 | Explicit surface point dependencies do not consult per-vertex validity. |
| [N-001](adjudication/N-001.md) | Confirmed defect | P2 | Native grid adapter reapplies raw tifxyz nonnegative validity after transformation. |
| [N-002](adjudication/N-002.md) | Confirmed defect | P2 | Anchored correction save failure is discarded by close/switch notification and teardown. |
| [P-001](adjudication/P-001.md) | Confirmed defect | P2 | UV preparation reads current geometry after its frozen-source verification has ended. |
| [P-002](adjudication/P-002.md) | Confirmed defect | P2 | Concurrent tracing exports share and unlink the same intermediate publication name. |
| [P-003](adjudication/P-003.md) | Confirmed architecture debt | P3 | Python and native JSON paths maintain separate anchor schema codecs; no demonstrated common-input runtime failure. |
| [P-004](adjudication/P-004.md) | Confirmed defect | P2 | Saved-solution comparison stamps current source geometry without validating the saved source identity. |
| [P-005](adjudication/P-005.md) | Confirmed defect | P2 | Spiral object pins do not freeze current overlay relationship membership. |
| [S-001](adjudication/S-001.md) | Confirmed defect | P2 | Hard-linked database aliases obtain distinct pathname-based service owner locks. |
| [S-002](adjudication/S-002.md) | Confirmed defect | P2 | Existing remote-object ingestion validates an unnecessary second source UUID before substituting the registered identity. |
| [S-003](adjudication/S-003.md) | Confirmed defect | P3 | Catalog cache publication does not check close-time write errors or clean failed partial files. |
| [S-004](adjudication/S-004.md) | Confirmed defect | P2 | Copy unseals a single immutable initial shard and then rejects its own staged result. |
| [S-005](adjudication/S-005.md) | Confirmed defect | P2 | In-place conversion discards the legacy input before staging its required mapping output. |
| [S-006](adjudication/S-006.md) | Confirmed defect | P2 | Caller-owned link candidate batches lack an operation-wide rollback boundary. |
| [S-007](adjudication/S-007.md) | Confirmed defect | P3 | CLI port values are narrowed to uint16_t before range validation. |
| [S-008](adjudication/S-008.md) | Confirmed defect | P3 | Project ordering compares publication and creation times in incompatible pair representations. |
| [S-009](adjudication/S-009.md) | Confirmed defect | P2 | Resource key concatenation does not encode location/owner tuple boundaries. |
| [S-010](adjudication/S-010.md) | Confirmed defect | P2 | Hosted catch-up removes qualifying changed rows when raw allow-list UUID spelling differs. |

Totals: **24 candidates; 23 confirmed defects, 1 confirmed architecture debt, 0 rejected, 0 duplicate, 0 unresolved.** Defect priorities: **20 P2 and 3 P3**. Architecture debt: **1 P3**, excluded from the correctness-defect total. Per lane: core 7 defects; native 2 defects; Python/bindings 4 defects plus 1 architecture item; storage/CLI/hosted 10 defects.

## Qualifications and deduplication

- C-001 and N-001 share the canonical-versus-transformed validity theme but fail at different stages: changed inverse edits versus initial native reads. C-007 is a separate graph-dependency validity omission.
- C-002 and S-010 share UUID normalization debt but are not duplicates: C-002 affects measured/volume server selection without concurrent changes; S-010 independently loses ordinary hosted object/edge allow-list matches during feed catch-up. One shared normalization design can address both.
- P-001 requires an edit after verification during preparation. P-004 accepts a completed source change before evaluating an already completed solve. They require separate guarded producer/replay paths even if shared source-snapshot helpers provide the fix.
- C-004 establishes an unreplayable pending submission; its checkpoint can still retain draft geometry. N-001 establishes lost loaded coordinates; a corrupt persisted round trip was not executed. S-001 establishes ownership-guard bypass, not SQLite corruption or an authentication exploit. S-003 requires an I/O failure. S-005's mapping-loss consequence is particularly relevant to legacy numbered-history reconstruction and is recoverable if a separate original backup exists.
- P-003 remains architectural only. Python None adaptation versus JSON null rejection is not, by itself, evidence of a supported cross-transport correctness failure.
- S-007's final identity is CLI port narrowing. The briefly reused discovery ID for project resource tuple hashing is superseded by S-009; there are not two resource-key findings.

All candidates survived the checks with the above scope restrictions; no rejection was invented to manufacture a validation quota. Actual declarations were checked rather than inferred from names, including optional/vector UUID filters, the project ordering map value type, port width, void/bool save signatures, and the public caller-owned transaction overload. The rejected R4 multimap allegation was not revived.

Future fixes should be validated through the existing approved runners and permanent regression targets, with explicit failure/interleaving cases described in each decision. This review provides no new Linux/macOS runtime result, no real-data validation, and no performance measurement.
