# Final-mismatch review

Subsequent implementation, Astra adjudication and validation are tracked in [fixes](fixes.md). The review evidence below describes the pre-fix state.

Reviewed baseline: HEAD `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4` plus the preceding PF fixes, tracked working diff SHA-256 `286b96ae90b5960043810003370f6ec3a66110597251dbcdd39477b313501235`. The checksum is unchanged after this review. No production/test changes, commits, installs or PR updates in this pass; findings and coverage logs only. These are older omissions/missed migrations, not new regressions identified in the immediately preceding PF working patch.

## Findings

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| [FM-A01](astra-artifact-output.md) | High | Artifact materialization can overwrite an attached database or other protected backing target; it bypasses shared export destination validation. | Native Astra source review; root reproduced replacement of an isolated test database pathname. |
| [FM-P01](root-retirement.md) | Medium | Track-grower overwrite retirement still submits obsolete numeric write guards and fails before retirement. | Independent Astra agreement; root runtime reproduction. |
| [FM-P02](root-position.md) | Low | Saved/configured Spiral positions coerce fractional/bool/string sequences before strict native validation. | Independent Astra agreement; root reproduced `1.5`, `True` and `'1'` becoming sequence 1. |
| [FM-P03](root-surface-selector.md) | Low | Python surface wrapper silently rounds down a fractional numbered-version selector. | Root source/runtime: version 2.5 reads version 2. |
| [FM-C01](root-point-json.md) | Low | GUI collection adapter duplicates a weaker point-ID decoder: numeric aliases collapse silently and partial numeric keys are accepted. | Root source/runtime; native file decoder already rejects these inputs. |
| [FM-C02](root-manifest.md) | Low | Native fit-input comparison validates saved relationship references but ignores missing/malformed saved object references and membership. | C++ lane source candidate; root reproduced manifests without valid object references being reported unchanged. |
| [FM-A02](astra-membership-key.md) | Optional cleanup | Selected relationship membership uses archive UUIDs versus synthetic strings instead of one complete in-memory publication reference. | Source only; no supported collision or incorrect answer demonstrated. |

Six correctness/API gaps and one optional representation cleanup. Numeric coercion and JSON/manifest validation findings concern malformed input, not demonstrated corruption under valid canonical inputs. Current-content comparison and external file-format adaptation remain legitimate; fix the shared validation/decoder boundaries rather than deleting their semantics or adding consumer-specific resolvers.

## Coverage and independent review

The [consumer inventory](consumer-inventory.md) screens 128 directly matching production files and follows indirect wrappers such as Lasagna tifxyz/approval adapters and Python store facades. It records the detailed native GUI/CLI, editor, project, Spiral, Lasagna and Vesuvius traces and exclusions. The [native inventory](astra-inventory.md) records core/bindings/hosted invariant traces and latest-PF-delta review. This is not proof that every line or concurrency interleaving in all 251 native/binding/header files was checked. No percentage of all bugs found or claim of no remaining issues is made.

Astra independently completed the native lane and adjudicated FM-P01/P02. Additional independent C++/Python lanes reached their model usage limit; one had already identified FM-C02. The root completed their listed source traces and runtime checks, but FM-P03 and FM-C01/C02 do not have completed independent final-model adjudication. [Adjudication](astra-root-adjudication.md) and per-finding files preserve that distinction.

PF-C03 (automatic ToolObjects repinning) is not reopened; the tool intentionally retains the state it read. Explicit UUID discovery, immutable geometry payload identities, domain content hashes, offline conversion and source-file adapters are not defects by themselves.

## Runtime verification

Command: `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks`, exit 0. Linux/GCC 13; existing RelWithDebInfo core and Release consumers; Python 3.14. Temporary C++ and Python diagnostic hooks reproduced all six correctness candidates without modifying production/tests. Diagnostics created automatically cleaned temporary stores; the deliberate artifact collision affected only its throwaway database. No user data was touched. The hook/source files were removed after completion so future fixes do not leave bug-expecting diagnostics in the approved runner; fixes should add checked-in assertions for corrected behavior.

The same run passed the six focused native checks, the two repeat/umbilicus checks, five source-repair tests, ten native consumer executables, eleven selected/transformed-view Python tests, consumer syntax checks and 49 combined fiber/manifest tests. These ordinary counts are not evidence for every proposed edge-case matrix. New counterexamples cover local, history-enabled temporary stores only, not a full hosted/packed/sharded/history-policy matrix. No full PR-suite rerun, macOS, GUI-interactive, full-volume, destructive real-data test or performance claim is included in this review.
