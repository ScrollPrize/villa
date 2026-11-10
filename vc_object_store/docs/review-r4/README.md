# R4 full code inspection

Frozen production baseline: `2b5d1e01fd26f27b3410d1b024c3f10a73533287`. Inspection complete; no production fixes or dependency changes. Discovery used GPT-6.1 Sol; independent adjudication used GPT-6 Astra. Each original candidate remains separately logged under `findings/`, with its independent decision under `adjudication/`. Original candidate statuses describe discovery time; the adjudications and [validation summary](validation-summary.md) are authoritative.

## Protocol

- Inventory production store APIs and consumers, follow wrappers/callers, and record each file's disposition. Keyword hits seed the inventory; they are not the coverage claim. No finding quotas.
- Give every concrete finding its own Markdown file under `findings/`, with ID, priority, exact location, reachable trigger, actual/required behavior, evidence status and shared fix direction. Do not treat green existing tests as proof of an uncovered case.
- Deduplicate by root cause without deleting the original evidence. Distinguish intentional capability limits, maintenance opportunities, unsupported inputs and confirmed failures. Historical fixed findings need a new reachable failure before being reopened.
- Independent adjudication records confirmed, rejected, duplicate or unresolved for every finding, with reasons. No production fixes in this pass. Inspection is not a guarantee of zero undiscovered defects.

## Coverage lanes

| Lane | Scope | Ledger | State |
| --- | --- | --- | --- |
| Core | Logical native APIs, history/geometry/queries/extensions/mutations/session | [122 files](core-coverage.md) | Complete |
| Native consumers | Native consumers outside the store, including GUI and CLI wrappers | [145 files](native-coverage.md) | Complete |
| Python | Bindings and Python consumers across subprojects | [73 files](python-coverage.md) | Complete |
| Physical/storage | Provisioning, references/shards/VFS, copy/conversion, import/formats, transport, linking, internal CLI/build/scripts | [133 files](storage-coverage.md) | Complete |

Do not mark review complete while any lane has undispositioned production entries or findings lack independent adjudication. Existing R3 test results describe the preceding fix pass, not runtime reproduction of R4 findings.

## Result

**17 candidates reviewed: after implementation verification, 12 confirmed (one P1, eight P2, three P3), five rejected, none unresolved.** The original independent review incorrectly confirmed C-006; checking the actual multimap type during implementation disproved it. See [implementation tracker](fixes.md) for subsequent changes and tests; the review itself was inspection-only.

| ID | Priority | Confirmed issue | Independent decision |
| --- | --- | --- | --- |
| S-006 | P1 | AFV report destination can truncate an input file or completed database | [Evidence and precise output-mode qualification](adjudication/S-006.md) |
| C-002 | P2 | Foreign structural slices bypass retained target point-ID checks | [Decision](adjudication/C-002.md) |
| C-003 | P2 | Loaded-tool artifact publication drops the complete pin/write-generation guard | [Decision](adjudication/C-003.md) |
| C-004 | P2 | Hidden/drafted objects consume backend region limits before visible results are assembled | [Decision](adjudication/C-004.md) |
| N-001 | P2 | Restart after a failed correction save can discard pending correction UI data | [Corrected trigger and severity](adjudication/N-001.md) |
| N-002 | P2 | CLI surface edits omit the loaded coordinate-mapping guard | [Decision](adjudication/N-002.md) |
| P-001 | P2 | Lasagna discards the shared validity mask and recomputes it in transformed coordinates | [Decision](adjudication/P-001.md) |
| S-002 | P2 | Direct public object-copy helper lacks operation-wide rollback | [Public API only; no production caller found](adjudication/S-002.md) |
| S-007 | P2 | Installed server CMake target lacks Boost dependency provisioning | [Narrowed finding; zstd rationale rejected](adjudication/S-007.md) |
| P-002 | P3 | Region minimum-length documentation describes the old ingestion-summary behavior | [Decision](adjudication/P-002.md) |
| S-004 | P3 | Oversized stamped dimensions narrow without a representability check | [Decision](adjudication/S-004.md) |
| S-005 | P3 | Maximum accepted source point ID overflows its successor counter | [Decision](adjudication/S-005.md) |

Rejected candidates remain available individually: [C-001](adjudication/C-001.md) (no supported disappearing-endpoint trigger), [C-005](adjudication/C-005.md) (API-hardening concern without a demonstrated supported caller), [C-006](adjudication/C-006.md) (multimap type disproves the claimed leak), [S-001](adjudication/S-001.md) (documented remote-import limit), [S-003](adjudication/S-003.md) (existing null-publication guard prevents trigger). The second model also narrowed N-001 and S-007 rather than accepting their initial arguments wholesale. Its C-006 confirmation was wrong and explicitly superseded during implementation.

## Coverage check and limits

The four ledgers contain **465 unique file paths** (overlapping lane entries deduplicated). A tracked-file complement check found no omitted files under `vc_object_store/src`, `include`, or `python`, or elsewhere inside that subproject after excluding documentation and test-only sources. A final residual pass explicitly added its CLI/support files, benchmark and build/package/script entries. Consumers outside the subproject were discovered through tracked source searches plus caller/wrapper closure; see the native/Python ledgers for exact discovery expressions and exclusions.

Coverage means store-facing identity, routing, state, mutation, geometry/coordinate, cache, transaction and transport control flow. It does not claim mathematical review of every unrelated numerical kernel, every rendering/training body, or exhaustive review of test harness internals. Tests were consulted as evidence, not all independently audited or executed. No GUI/macOS/full-volume runtime or performance result is claimed. An empty tracked-file complement and finite ledger are coverage evidence, not a proof of defect freedom.
