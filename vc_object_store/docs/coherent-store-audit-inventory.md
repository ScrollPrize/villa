# Coherent vcobj API audit inventory

Baseline: `557a6f925`, reviewed 2026-11-10. The original review inventoried the implementation and consumers without changing production code. Findings were code-traced; subsequent implementation and regression evidence are recorded below.

## Review method

Three agents reviewed core, storage/transport, and C++ consumer lanes; the coordinating agent reviewed Python consumers, followed wrapper calls and reconciled overlapping findings. There was **no finding quota**. Each assigned file has a disposition and the relevant symbols or exclusion reason. Findings are grouped by cause, not multiplied by the number of affected files.

The initial tracked-source inventory had 408 files. Wrapper/caller closure added 36 unique files, giving **444 unique files with final dispositions and zero pending entries**. Three additional transport evidence rows refer to files already covered by the core lane and are not counted twice. Discovery seeds were all C++/Python source and headers under `vc_object_store/` excluding tests, plus repository-wide matches for the following expression outside that project:

```text
vc_object_store|vc::objects|\.vcobj|ObjectStoreSession|StoreReader|PinnedObject
```

The external search covered `.cpp`, `.hpp`, `.h`, `.py` and `.inc`, excluding conventional tests. Wrapper closure followed `object_store_inputs`, `stored_surfaces`, `service_commits`, neural `fiber_store`/`surface_store`, GUI controllers and their callers. A second selector search covered `geometry_snapshot`, `read_surface_grid`, `current_object_revision`, `current_version`, `graph_read` and `vcobj:`. Additional-language searches covered `.c`, `.cc`, `.cxx`, `.cu`, `.rs`, `.js`, `.ts`, `.tsx`, `.sh` and `.cmake`; their additional hits were build configuration, explicitly classified in the C++ ledger. Test code accidentally matched through included `.inc` files is explicitly excluded, not counted as production reviewed.

Search matches are seeds, not sufficient evidence of review. The ledgers preserve the expanded concrete inventory and each inspected call path:

- [Core ledger](audit-core.md): records, metadata, versions/history, selection, geometry, transforms and spatial APIs.
- [Storage/transport ledger](audit-transport-storage.md): collections, packed VFS, imports/exports, extensions, hosted/session APIs, bindings and codecs.
- [C++ consumer ledger](audit-cpp-consumers.md): GUI, geometry adapters, tools, native pipeline readers and service clients.
- [Python consumer ledger](audit-python-consumers.md): Lasagna, Spiral and neural-tracing adapters/workflows, including indirect wrappers.

| Lane | Files | Reviewed paths | Finding-bearing files | Declarations | Explicit exclusions | Pending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Core | 111 | 54 | 14 | 37 | 6 | 0 |
| Storage/transport | 142 | 73 | 12 | 43 | 14 | 0 |
| C++ consumers | 127 | 47 | 20 | 42 | 18 | 0 |
| Python consumers | 64 | 50 | 7 | 2 | 5 | 0 |
| **Total unique** | **444** | **224** | **53** | **124** | **43** | **0** |

“Reviewed” means the named relevant path was inspected without an additional finding, not that unrelated code in the file is proven correct. A declaration disposition identifies its implementation or owning review lane. A finding may affect multiple files; a finding-bearing file may contain multiple findings. These file counts therefore are not bug counts.

## Implementation pass

Started from `557a6f925`. Close the existing 37 entries by shared invariant, not by adding consumer-specific resolvers. Implementation order: source/export safety and checked edits; complete selection/identity through consumers; owner-based retention; source/copy capabilities; no-op/normalization/validity. Keep schema 21 and binary identities unchanged, introduce no persistent routing/history structures, and retain the explicit unsupported-case errors until their shared capability exists. Record regressions and production versus test/documentation line counts before closing entries. Use the existing argument-free PR and consumer runners; no full-volume runs or dependency installation is part of this correctness pass.

Implemented shared boundaries:

| Shared boundary | Findings served | Implementation / regression evidence |
| --- | --- | --- |
| Selected identities and checked edits | CORE-3, TRANS-1/5, CPP-1/3, AL/AM/AN/AO/AP, PY-1/4 | `expectObject` validates complete references under the write lock; editable collections also retain their load-time mapping. Wire, receipts, queues, umbilicus, cache fingerprints and provenance preserve owners. Wrong-owner equal-counter, stale-registration and queued deleted-source tests. |
| Owner retention and selected graph | CORE-2/4, TRANS-2, CPP-2/4/6, AJ/AQ | Reader-backed exact export, graph selection and recorded-endpoint closure; owner-based restore/relationship retention. Mixed source/overlay history, cross-shard relationships, revalidated peer, deleted/overridden exact export and off-screen graph regressions. |
| Geometry/transform interpretation | AI, TRANS-4, CPP-5/7/8, PY-5 | One hosted transform snapshot; selected metadata transforms; shared resident-grid cutter and validity; typed Spiral geometry; guarded async results. Hosted concurrent-read, pinned overlap, resident/draft parity and consumer tests. |
| Source and output safety | CORE-1/6, CPP-6/9, PY-3 | Shared physical-closure overwrite protection and source-locator resolution, including packed containers and artifact symlinks. Fatal importer errors stay fatal. Explicit derived-metadata policy refuses managed-data copying or records deliberate geometry-only derivation. |
| Identity allocation and schema/no-op invariants | AH/AK/AR, CORE-5/7, TRANS-3, PY-2 | Shared extension compatibility; retained point IDs; compare before override; common logical geometry equality; UUID canonicalization. Identity resize/empty splice, no-op write hooks, incompatible inherited schema and uppercase UUID regressions. |

No new tables, columns, stored per-object routing data or geometry payload duplication are introduced. Runtime API additions include complete relationship endpoint references, reader-backed selected graph/export/source APIs and immutable resident-surface queries. Transaction command encoding 2 protects complete write references; number-only encoding 1 remains readable and keeps its digest. Database schema remains 21.

Two new production files extend the frozen baseline inventory: `include/vc_object_store/objects/ResidentSurface.hpp` declares the thin shared wrapper over the existing resident/draft intersection kernel; `python/vc_object_store/derivation.py` centralizes derived-metadata policy. Both were reviewed with their callers and regression tests. The 444-file count above describes the original audit, not these additions.

Intentional boundaries: native tools that require a filesystem dataset still require one; native export itself now accepts hosted readers. Derived geometry writers do not pretend to copy managed extension/artifact payloads: the shared portable policy rejects that request, while geometry-changing snapping explicitly chooses and records geometry-only derivation. C++/Python Spiral's small domain-specific metadata projection remains a parity-tested language boundary, not duplicated storage/version routing. Exact graph closure follows recorded references, not a fictitious globally synchronized history across stores.

AH–AR are the eleven previously recorded, now implemented tasks in [the cleanup tracker](coherent-store-api-cleanup.md#open-findings-from-review-of-557a6f925). A lane referring to one of those does not add another finding. New `CORE-*`, `TRANS-*`, `CPP-*` and `PY-*` IDs are stable task identifiers, not temporary labels to renumber after every review. Their detailed sections include evidence, affected callers, suggested shared boundary and regression scenario.

### Previously recorded tasks (implemented)

| Task | Missing shared behavior |
| --- | --- |
| AH | Collection-wide extension schema compatibility before registration |
| AI | Coherent hosted transform selection instead of independent current reads |
| AJ | Restore source selection independent of destination history policy |
| AK | Retained structural point-ID allocation in Spiral editing |
| AL | Complete source reference through queued neural-copy jobs and provenance |
| AM | Owning-store identity in neural fiber version fingerprints |
| AN | Complete identity comparison in the editor freshness guard |
| AO | Umbilicus reference constructed from selected owner |
| AP | Copy provenance constructed from selected owner |
| AQ | Complete endpoint references in current-only relationship exact reads |
| AR | Metadata/approval no-op comparison before preparing an override |

These remain tracked in the cleanup file; the detailed lane reports identify the additional callers and shared boundaries, rather than declaring them newly discovered again.

### Additional inventory findings

These are implementation tasks, not a count of independent architectural causes: several share the identity/snapshot, retention, source-routing, canonicalization or no-op invariants described below. All items below have been addressed through the shared boundaries above. Full evidence and regression scenarios are in the linked lane, under the same stable ID.

#### Core — [evidence](audit-core.md#findings)

- [x] CORE-1: Export overwrite protection omits referenced backing databases/containers; it can destroy a dependency.
- [x] CORE-2: Exact export wrongly requires a retained current-only source version to remain the visible current head.
- [x] CORE-3: Tool-derived coordinates reread current metadata instead of using the loaded source pin.
- [x] CORE-4: Exact historical graph membership is overwritten according to the attachment's history policy.
- [x] CORE-5: Identity resize/empty splice publishes unnecessary overrides and new versions.
- [x] CORE-6: Track source dataset paths still resolve relative to the attachment rather than the source owner.
- [x] CORE-7: Public comments describe removed history-off identity rules and an incorrect volume-ID restriction. **Documentation inconsistency.**

#### Storage/transport — [evidence](audit-transport-storage.md#findings)

- [x] TRANS-1: Project rename can submit a stale full record with a newer expected version, overwriting concurrent metadata.
- [x] TRANS-2: Exact export/restore wrappers cannot accept the complete reference needed to select a previous owner.
- [x] TRANS-3: AFC source matching and fiber-network export roots omit shared UUID canonicalization.
- [x] TRANS-4: Pinned overlap materialization caches transformed data by geometry alone, conflating metadata-dependent coordinate states.
- [x] TRANS-5: Historical relationship production, wire transport and Python bindings lose the relationship's own owner identity.

#### C++ consumers — [evidence](audit-cpp-consumers.md#findings)

- [x] CPP-1: Point-collection saves discard loaded versions, permitting lost edits.
- [x] CPP-2: Fiber navigation/decoration and line editing mix selected geometry with fresh current metadata/relationships.
- [x] CPP-3: GUI checks, edit expectations, caches and auto-link repinning retain incomplete version identity.
- [x] CPP-4: Fiber editor imposes an attachment-wide history gate instead of shared relationship-retention policy.
- [x] CPP-5: Picking, overlap decoration and ink selection query current state for a potentially historical displayed surface.
- [x] CPP-6: Native export lacks a reader/session-backed hosted path. **Explicit capability gap, not silent corruption.**
- [x] CPP-7: Spiral GUI duplicates typed geometry decoding and adapter interpretation. **Centralization/maintenance gap; the deliberate scalar-frame restriction is not itself a bug.**
- [x] CPP-8: Surface overlay caches omit transform invalidation and async cut-result identity checks.
- [x] CPP-9: Import-controller exception translation can turn a fatal infrastructure failure into apparent success.

#### Python consumers — [evidence](audit-python-consumers.md#findings)

- [x] PY-1: Training catalogs/caches combine old metadata or cached points with later current geometry/fingerprints.
- [x] PY-2: Fiber helpers compare uncanonicalized caller UUIDs against canonical returned IDs.
- [x] PY-3: Derived fiber writers copy managed metadata without extension/artifact copy support. **Fail-closed integration gap; core guards correctly reject detached extension descriptors.**
- [x] PY-4: Umbilicus owner identity is lost throughout the GUI/service/manifest protocol, even if AO is fixed.
- [x] PY-5: Spiral's raw surface materialization duplicates a weaker vertex-validity rule than shared readers.

Total: **26 additional entries plus the existing 11 = 37 addressed task entries**. Related tasks must be addressed together: AI/CORE-3/TRANS-4/CPP-2/CPP-5/CPP-8/PY-1 concern coherent selected-state use; AN/AO/AP/AL/AM/TRANS-5/CPP-3/PY-4 concern complete identity; AJ/AQ/CORE-2/CORE-4/TRANS-2/CPP-4 concern exact retention; CORE-1/CORE-6/CPP-6/PY-3 concern shared source/copy/export boundaries; TRANS-3/PY-2 share UUID normalization; AR/CORE-5 share no-op publication rules. These associations do not erase the distinct failing scenarios or their regression requirements.

Recommended fix order: first CORE-1's backing-store overwrite risk and the lost-update paths (CPP-1/TRANS-1); then shared selection/identity and retention families with their consumer migrations; then source/copy/export capabilities, no-op/canonicalization/validity and remaining documentation/maintenance items. CPP-9's fatal-error reporting should accompany import-path changes, not be hidden among recoverable per-object problems.

## How this ends the repeated sampled-review cycle

Earlier reviews had broad assignments and capped finding lists; they did not record what remained uninspected. That was selective coverage, not evidence of a clean architecture. This review freezes one code revision, records the full discovered file/call inventory, and leaves a durable negative-review record as well as defects. No implementation changed between these lanes, so discoveries here are not new regressions introduced by another lane's fixes.

For implementation, close invariant families across their inventoried consumers, not just the first failing line:

1. Complete selected identity from read through transform/cache, IPC, edit expectation and receipt repinning.
2. Retention determined from the selected owner; destination policy controls retention of new edits, not visibility of source history.
3. One physical-closure/source-locator API for linked/sharded/packed storage, artifacts, copy/export and overwrite protection.
4. Shared UUID normalization, selected metadata interpretation and geometry validity.
5. No-op detection before overrides/publication, without skipping required relationship validation.

Each fix needs a failing regression for its reported case and coverage of relevant sibling paths. The reusable matrix is: local/referenced/sharded/packed; local/hosted; both history settings for source and destination; active/retired/deleted/overridden; exact/current/no-op; equal counters from different owners; concurrent edits where relevant. Record inapplicable combinations explicitly. A passing ordinary CRUD suite does not close omitted matrix cases.

After fixes, review the implementation delta and affected callers against this inventory. Reopen a closed item only with a concrete counterexample or changed contract. Add newly discovered consumers to the ledger instead of starting another unconstrained hunt with a different sample.

## Limits and evidence

An exhaustive disposition of the discovered vcobj call paths is not a proof that no defects exist. Unrelated numerical algorithms, complete concurrency/security analysis, dynamically generated consumers, external repositories and future edits are outside this audit. The ledgers distinguish confirmed defects, explicit unsupported capabilities, documentation inconsistencies and investigated non-findings. Their original discovery sections describe code inspection; subsequent implementation sections separately record executed regressions. No dependencies were installed and no full-volume production experiments were run during this correctness pass.

## Final validation and change accounting

The implementation delta and affected callers were reviewed again after integration. Four concrete counterexamples from that review were fixed with regressions: selected exports losing inherited graph peers, contradictory numeric/full-reference command guards, current edge bindings overriding a selected historical peer, and artifact-directory symlinks bypassing export overwrite protection. These close the existing invariant families rather than creating another sampled review list.

Linux, existing Python 3.14 environment, native RelWithDebInfo build, synthetic local/referenced/sharded/packed/hosted fixtures. The fixed argument-free runner installs no dependencies and cleans its temporary fixtures. Reuse the approved command:

```sh
bash /home/hendrik/villa3/local/scripts/run_vcobj_pr_tests
```

Final run completed successfully on 2026-11-10; local log `local/vcobj-pr-tests/logs/20261110T180431Z-1430790.log`:

| Suite | Result | Test time |
| --- | --- | ---: |
| Native object store | 52 passed; optional SSH test skipped | 27.87 s |
| Python object store | 201 passed | 35.52 s |
| Spiral object-store/runtime | 121 passed; 1 skipped | 40.30 s |
| Lasagna stored surfaces | 13 passed | 1.28 s |

Additional completed checks: `bash local/scripts/check_vcobj_consumers` checked 29 C++ translation units; `bash local/scripts/run_vcobj_consumer_checks` passed seven linked consumer suites. The three neural consumer test modules listed in [the Python ledger](audit-python-consumers.md#implementation-and-regression-evidence--2026-11-10) passed 19 tests. `git diff --check` passed. These are correctness checks, not throughput benchmarks.

Production versus baseline `557a6f925`: **1,129 lines added, 731 removed, net +398**. Regression tests and their build registration: **727 added, 48 removed, net +679**. Documentation is separate from both counts. Native-consumer production code alone is net −35. Overall production line count did not decrease: full-reference guards, shared exact graph/export capabilities and hosted forwarding fill previously missing behavior. Consumer-specific resolvers were removed or migrated; no new persisted routing/history structures were added. Do not describe this pass as an overall LOC reduction.

Implementation commits: `4ea661176` (shared store), `23f777ef6` (native consumers), `ccb15e21c` (Spiral), `91f9a24b8` (neural consumers). All 37 tracked entries are addressed with the explicit capability boundaries above; this is not a proof of absence of unrelated defects. Full GUI execution, macOS/arm64, SSH integration and full-volume performance were not exercised. Spiral still prints a native `PatchSamplingAtlas` shutdown leak warning. An exploratory numerical neural-snapping test crashed in `vc_reader.read_regions`; its numerical path remains unvalidated in this pass, while the changed source/copy policy has isolated passing regressions. The earlier restricted Spiral run has been superseded by the successful full object-store/runtime suite above.
