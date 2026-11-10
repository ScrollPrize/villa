# R5 CLI, packaging and benchmark coverage supplement

Baseline `bea32d5d8`. GPT-6.1 Sol, source inspection only. These 15 files complete the physical lane alongside `storage-coverage.md` (118 core physical files). Existing test/build fixtures were inspected as evidence when relevant, not executed. This does not audit unrelated benchmark arithmetic/performance claims or every OS implementation of the shared filesystem primitives.

| File | Disposition |
| --- | --- |
| `vc_object_store/apps/support/ProcessMetrics.hpp` | Host RSS units and unavailable result checked; diagnostic only, no object identity or persistence. |
| `vc_object_store/apps/support/SqliteMutexProfile.hpp` | Opt-in startup-before-connections forwarding profiler; real SQLite lock calls retained, per-thread counters and bounded stack samples, no second locking policy. |
| `vc_object_store/apps/vc_object_convert.cpp` | S-005: mapping/report publication sits after irreversible in-place replacement; checked existing-path rules and distinct output requirements. |
| `vc_object_store/apps/vc_object_export.cpp` | Common selected-object resolver/export path; explicit complete owner/generation/version and read-only local destination protection. |
| `vc_object_store/apps/vc_object_import_afv.cpp` | Shared native bounded streaming/shard import; aggregate checks before publication; optional readback intentionally separate; R4 protected report paths and atomic writer traced, no recurrence established. |
| `vc_object_store/apps/vc_object_ingest.cpp` | Common format/channel ingestion and coordinate normalization; separate per-source/channel transaction results are explicit, not a promised all-input transaction. |
| `vc_object_store/apps/vc_object_links.cpp` | Argument adaptation to existing detector APIs; sequential detector invocations are separate CLI operations, not another detector engine. S-006 applies to the distinct public caller-owned helper, not this CLI's transaction ownership. |
| `vc_object_store/apps/vc_object_server.cpp` | Thin configured service wrapper; S-007 is the missing range check before typed port conversion. Default trusted-network/no-auth policy is documented, not a newly introduced auth defect. |
| `vc_object_store/benchmarks/prepared_import.cpp` | Input preparation, shared native queue/shard path, separate prototype SQL, verification/inventory switches, sample timing and disposable output cleanup reviewed. Prototype is deliberately not a supported vcobj implementation. |
| `vc_object_store/benchmarks/README.md` | Diagnostic setup, boundaries and statistics semantics checked against implementation. Private disposable fixture root required; no real-volume scaling validation in this pass. |
| `vc_object_store/scripts/import_afv.py` | Uses shared native AFV reader and typed extension provisioning, private staging, aggregate source totals and no-clobber publication; intentional Python convenience importer, no duplicated source codec. |
| `vc_object_store/scripts/run_cpp_tests.sh` | Fixed no-argument test entrypoint; build/log root and self-cleaning fixture separation checked. Read only; not executed. |
| `vc_object_store/CMakeLists.txt` | Adapter/linking/Python/tool prerequisites, source target wiring and installed export set checked; optional server remains static. |
| `vc_object_store/cmake/vc_object_storeConfig.cmake.in` | R4 Boost dependency now provisioned for server packages; no unsupported zstd alias finding reopened. |
| `vc_object_store/pyproject.toml` | Python-only wheel install boundary and embedded native source rebuild keys; setup/install excluded from execution. |

Additional candidates are [S-005](findings/S-005.md) (conversion publication) and [S-007](findings/S-007.md) (typed CLI port boundary). Other files above have no new concrete defect established in this scope. No production changes or additional test runner were made.
