# vcobj coverage: python-consumers

Review baseline: `557a6f925`. Scope: shared handling, identity/retention/placement correctness and duplication, not an exhaustive numerical/security audit. This inventory is generated from tracked core sources and repository-wide consumer search; add newly discovered wrapper/caller files with their discovery reason. No implementation edits during this review.

A disposition applies to the relevant vcobj symbols/paths, not every unrelated line in a large file. Allowed dispositions: PENDING, REVIEWED (name symbols/shared contract checked), FINDING (local ID and scenario), DECLARATION (corresponding implementation), EXCLUDED (specific reason). REVIEWED means no additional issue found in the stated path, not proof of correctness. No finding quota. Report all substantiated defects, combine duplicates, and keep unknowns visible. Code-traced and runtime-tested evidence must be distinguished.

## File ledger

| File | Disposition | Symbols / evidence / finding |
| --- | --- | --- |
| `lasagna/fit2tifxyz.py` | REVIEWED | _store_surface, result documents and _publish_side_files use shared writers; temporary side files cleaned in finally. |
| `lasagna/fit_service.py` | REVIEWED | Stored model/grid selectors preserve refs via stored_surfaces; request-volume guard is domain policy. |
| `lasagna/scripts/bootstrap_venv.py` | EXCLUDED | Dependency provisioning only. |
| `lasagna/setup.py` | EXCLUDED | Package dependency declaration. |
| `lasagna/stored_surfaces.py` | REVIEWED | Reference parser/selector, retained channels and artifact reads carry selected owner/generation/version. |
| `spiral-fitting/config.py` | DECLARATION | Options consumed by object_store_inputs. |
| `spiral-fitting/cpp/track_crossings.cpp` | EXCLUDED | Numerical TrackCrossings binding, no database/identity handling. |
| `spiral-fitting/find_inconsistent_windings.py` | REVIEWED | build_fit_inputs explicitly selects current read-only inputs through FitContext. |
| `spiral-fitting/fit_spiral.py` | REVIEWED | Loading/resume/replacement uses InputPins/load_fit_inputs; domain views delegate object_store_inputs. |
| `spiral-fitting/grow_track_graph.py` | REVIEWED | store_surface, stored_seed_surfaces, retire_surfaces use shared creation/query/current-version guards. |
| `spiral-fitting/object_store_inputs.py` | FINDING | PY-5 validity duplication; selected refs, snapshots, geometry materialization, link refresh receipts and content digest traced. |
| `spiral-fitting/runners/run_single.py` | REVIEWED | _resume_invocation uses shared locations; store UUID is intentional resume binding. |
| `spiral-fitting/service_commits.py` | REVIEWED | Queries/writes use shared APIs; incomplete CommitResult is a protocol limitation, current UI opens UUID intentionally. |
| `spiral-fitting/spiral_service.py` | FINDING | PY-4 umbilicus owner omitted in _commit_context; binding/startup, updates, resume and commit flows otherwise use shared adapters. |
| `vesuvius/src/vesuvius/fiber_spline_generator/__init__.py` | EXCLUDED | Package description only. |
| `vesuvius/src/vesuvius/fiber_spline_generator/cli.py` | REVIEWED | Preview/final write closures delegate write_store; fresh stores only. |
| `vesuvius/src/vesuvius/fiber_spline_generator/store.py` | REVIEWED | Validated fresh single-file creation and checked geometry writes; staging is not collection copying. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_anchored_dataset.py` | REVIEWED | Selection/copy via digest/read_polylines/write_fibers; AM/PY-3 inherited. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_constraint_score.py` | REVIEWED | Volume compatibility, signatures, combined-store writes via helpers; AM/PY-3 inherited. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_gt_inpainting.py` | REVIEWED | Before/after fingerprints via fibers_digest (AM). |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_joint_gt_eval.py` | REVIEWED | Combined curves/writes and signatures via helpers; AM/PY-3 inherited. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_json_solve.py` | REVIEWED | JSON conversion writes/roundtrips through write_fibers/read_polylines. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_expansion.py` | REVIEWED | Catalog selection, geometry, round exports and final selection use helpers; AM/PY-3 inherited. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_neighbor_pair.py` | REVIEWED | Paired signature via fibers_digest (AM). |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_selected_dataset_eval.py` | REVIEWED | Coordinate equality and signature use common helpers; AM inherited. |
| `vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py` | FINDING | Existing AL: _StoredSurfaces incomplete queued/source provenance reference; common surface API otherwise. |
| `vesuvius/src/vesuvius/neural_tracing/inference/surface_store.py` | FINDING | Existing AL: read_surface discards owner in store_version and exposes no full reference selector. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/scripts/snap_fibers.py` | REVIEWED | CLI delegates snap_store; see PY-3 for downstream writer. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/build_index.py` | REVIEWED | Index geometry/manifests via FiberStore/fibers_digest (PY-1/AM). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer.py` | REVIEWED | Inference constructs FiberStore (PY-1). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/infer_all.py` | REVIEWED | Batch inference constructs FiberStore (PY-1). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/train.py` | DECLARATION | Dataset path passed to sheet_strip.data; no independent vcobj access. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/volume.py` | REVIEWED | Export delegates volume_store; no independent vcobj selection. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/volume_store.py` | REVIEWED | run_fibers/export_store create fresh geometry via write_fibers; template supplies volume only. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/shared/annotations/fiber_store.py` | FINDING | AM plus PY-1/PY-2/PY-3: digest owner, mutable catalog/cache, raw UUID comparison, reserved metadata copying. All read/write/spatial helper paths checked. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/build_fiber_pool.py` | REVIEWED | Pool via FiberStore (PY-1). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/sheets/scripts/match_fibers_to_patches.py` | REVIEWED | Matching geometry via FiberStore (PY-1); numerical matching excluded. |
| `lasagna/approval_inpaint.py` | REVIEWED | load_approval_surface reads approval/d channels from selected StoredSurface. |
| `lasagna/tifxyz_io.py` | REVIEWED | load_tifxyz delegates exact selection; remaining tensor conversion is domain adaptation. |
| `spiral-fitting/fit_session.py` | FINDING | PY-4 ProjectUmbilicus/_checked_umbilicus strip owner from selected identity. |
| `spiral-fitting/pcl_views.py` | EXCLUDED | In-memory numerical views of already materialized inputs. |
| `spiral-fitting/spiral_runtime.py` | REVIEWED | Input commands/adoption forward pins and manifests; runtime generation counters are not store versions. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_constraint_database_validation.py` | REVIEWED | Source catalog/raw curves use common helpers; direct SQL is separate constraint database, not vcobj. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_curve_bank.py` | REVIEWED | CurveBank uses read_polylines and verifies sorted IDs; no second resolver. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction.py` | REVIEWED | Raw geometry through read_polylines; SQL addresses constraint graph. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_direction_review.py` | REVIEWED | OBJ geometry/signature via helpers (AM). |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_full_slab.py` | REVIEWED | Dispatches source identity to fibers_digest (AM), other file identities to SHA. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_hv_stage.py` | REVIEWED | Fingerprint and catalog via helpers; SQL is constraint-result schema. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_min_flow_benchmark.py` | REVIEWED | Seam geometry read via read_polylines; solver math excluded. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_normal_sign_visualization.py` | REVIEWED | Catalog, spatial regions and curves use common readers; SQL is separate result database, store UUID check is dataset provenance. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_piece_split.py` | REVIEWED | Raw curves/named volume via helpers; split math excluded. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_reference_comparison.py` | REVIEWED | Projection signatures and arc samples via helper (AM). |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_soft_winding.py` | REVIEWED | Prior geometry via fiber_volume/read_polylines. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_layout.py` | REVIEWED | Source/catalog/curve reads via helpers; SQL is separate constraint database. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_reference_subset.py` | REVIEWED | Subset raw curves/volume via helpers; projection math excluded. |
| `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/fiber_uv_surface.py` | REVIEWED | Source identity via fibers_digest; output tifxyz is not vcobj persistence. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/datasets.py` | REVIEWED | open_fiber_source creates training/validation FiberStore (PY-1/AM). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_bank.py` | REVIEWED | Neighbor queries and fingerprints through FiberStore/fibers_digest (PY-1/AM). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/fiber_snapping.py` | FINDING | PY-3 snap_store forwards retained metadata to unrelated fresh-store writer. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/data/location_weights.py` | REVIEWED | Cache key through fibers_digest; rows via FiberStore catalog (AM/PY-1). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/scripts/build_location_weights.py` | REVIEWED | Cache generation constructs FiberStore (PY-1/AM). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/sheet_strip/data.py` | REVIEWED | Sheet-strip units use FiberStore geometry (PY-1). |
| `vesuvius/src/vesuvius/neural_tracing/tracers/fibers/train/dagger/collect.py` | REVIEWED | FiberStore type/catalog used for mining, no independent persistence. |

| `lasagna/fit.py` | REVIEWED | Added by selector search: surface loads forward refs/store/volume through tifxyz_io. |

## Findings

All findings below are code-traced, not runtime-reproduced. AL and AM are existing tasks, not new findings. The following IDs have no finding-count limit and group sibling callers by the shared defect.

### PY-1 — Training catalogs/caches combine metadata and geometry from different versions

`tracers/shared/annotations/fiber_store.py:48,180,228,253`: `fiber_catalog` retains scale, family and lengths but no selected reference. `FiberStore.geometry` later reads current geometry and scales it with the cached catalog's scale; its LRU is keyed only by index. `_blocks` retains spatial-region geometry IDs but still applies old catalog scale. `manifest_entries` fingerprints the current store independently of cached geometry. A scale/geometry edit while the read-only training handle exists therefore gives new points in the old frame, or a new digest describing old cached points. Opening read-only is not a guarantee that another writer cannot edit the database. Catalog pagination also spans independent query snapshots.

Shared correction: capture a versioned catalog and use its complete references for both geometry and coordinate interpretation; either pin the training dataset or explicitly reject a changed source. Migrate the training, sheet-strip and fiber-pool callers listed above. Regression: populate a cache, edit coordinates/geometry from another handle, then exercise cached and uncached geometry, spatial blocks and manifest generation. Do not silently mix states.

### PY-2 — Fiber helpers compare raw caller UUID spelling with canonical stored UUIDs

`fiber_store.py:76,82,106,116`: `read_polylines` and `fibers_digest` normalize caller IDs only with `str`, then compare them directly to canonical UUIDs returned by the store. A valid uppercase UUID resolves inside the store but is subsequently rejected as an unknown requested fiber; a canonical and uppercase spelling can also appear as two requested IDs. This repeats the identity-normalization problem fixed in native traversal, at a different public adapter boundary.

Shared correction: expose/use the common UUID canonicalization entry point before set/order/membership operations; it is not currently a public Python helper, so extend the binding rather than introducing another parser. Regression: canonical/uppercase/equivalent spellings, duplicates and invalid UUIDs in both helpers.

### PY-3 — Derived fiber copying treats managed metadata as portable application metadata

`fiber_store.py:152,165` copies `row.meta` wholesale into a new independent store. `fiber_snapping.snap_store` explicitly promises to keep source metadata and forwards it unchanged apart from snapping provenance. Subset and combined-dataset workflows do likewise. With a manual extension snapshot, the new write fails at `internal/RecordSql.cpp` / `internal/ExtensionHistory.hpp::rejectInitialExtensionMetadata`, because the reserved `vcobj_extensions` descriptor is not a portable JSON field. Artifact descriptors also require payload closure and cannot be copied by this root-file-only writer. The core guard is correct; this is an incomplete consumer copy contract, not evidence of silently corrupted extension data.

Shared correction: define the supported derived-fiber copy operation using the shared selected-record/copy infrastructure, including an explicit policy for extension/artifact preservation versus intentionally new geometry-only data. Merely calling `readObjectCopy` is insufficient: `ObjectCopy.cpp:11–12` currently refuses all registered extensions, and it does not transfer artifact payload files. That shared capability must be extended if preserving those data is required; do not strip reserved metadata ad hoc in each caller. Regression: source fiber with manual extension and artifact, referenced source, unchanged subset and changed-geometry derivative; preserve supported data or fail clearly before publication.

### PY-4 — Umbilicus service protocol loses the selected owner even after AO is fixed

`spiral-fitting/fit_session.py:985,1237` stores and serializes only object UUID/generation/version in `ProjectUmbilicus`; `_checked_umbilicus` ignores any supplied store field. `spiral_service.py:3535` drops it again when recording commit provenance. The native sender `volume-cartographer/apps/VC3D/SpiralProjectInputs.cpp:284` also omits the owner. Two independent branches with equal object/generation/version counters therefore produce indistinguishable version provenance. Supplied points still reach the solver; the defect is identity/reproducibility, not a demonstrated numerical geometry error.

Shared correction: carry the common complete reference through sender, parser, manifests and result provenance. AO fixes construction in `VolumeUmbilicus`; this task covers the downstream protocol, which otherwise still discards that fix. Regression: equal counters from distinct owning stores survive the complete request/manifest/commit roundtrip distinctly.

### PY-5 — Spiral raw surface materialization duplicates and weakens the store's validity rule

`spiral-fitting/object_store_inputs.py:1094`: `_surface_zyxs` marks only all-minus-one points and an optional zero validity mask invalid. Shared surface readers also handle the coordinate validity convention, including finite values and invalid coordinates. Raw `putGeometry` accepts schema-valid arrays without typed-surface normalization, so e.g. a finite vertex with zero Z and no validity array can reach this path and be interpreted differently from `SurfaceView`. The core review confirmed that raw geometry schema validation does not normalize these values; typed `Surface::validate` is not a universal input boundary.

Shared correction: materialize the selected surface through shared surface-grid/validity handling and retain the selected frame, rather than duplicating a partial validity rule in Spiral. Regression: raw arrays with sentinel/partial-sentinel, non-finite and zero-Z points, with and without validity masks; compare the supported mask semantics with the shared reader rather than inventing another rule.

## Coverage limitations

64 files dispositioned, including 27 added through wrapper/selector closure after the initial 37-file seed. Inspection covers their vcobj entry points, ownership/version flow and wrapper boundaries, not every numerical algorithm in these large modules. The scalar/direct-volume restriction in the training/fitting adapters is explicit domain policy; no demand to support arbitrary transforms is inferred.

`service_commits.CommitResult` exposes only object UUID/numeric version/geometry ID. The C++ receiver was traced by the consumer lane: it intentionally opens the current UUID and does not fabricate an exact reference from this result. This is a limitation if exact result pins are later required, not a confirmed wrong-version read in the current flow. `object_store_inputs` content hashing deliberately includes selected geometry and relevant metadata, so stripping ownership there is not the version-only collision reported by AM.

No new tests or runtime probes were run for this lane. Code-traced findings need regression reproduction before fixes. Core implementation findings inherited by these wrappers are counted once in their owning lane. The source/wrapper searches cannot prove the absence of dynamically generated or out-of-repository consumers; subsequent changes must be reviewed as deltas against this inventory.

## Implementation and regression evidence — 2026-11-10

This section records the subsequent implementation pass; the review-baseline ledger above remains the discovery record. The consolidated inventory owns overall closure, including the corresponding native/binding changes.

| Finding | Shared correction | Regression evidence |
| --- | --- | --- |
| PY-1 | Catalog selection is one query snapshot and retains complete references. Geometry uses the shared exact `graph_read(versions=...)` path. Dataset manifests, cached/uncached geometry and spatial queries reject any changed source commit instead of mixing old metadata with new geometry. | `test_fiber_store_versions.py`: metadata and sparse-geometry edits, cache populated/empty, history on/off; reopening yields the new digest/geometry. Existing fiber source/bank tests also pass. |
| PY-2 | `read_polylines`, `fibers_digest` and the derived writer use native `canonical_uuid`; no separate Python parser. Explicit empty selections stay empty. | Canonical/uppercase spellings, duplicates, invalid IDs and empty selections in `test_fiber_store_versions.py`. |
| PY-3 | Shared `vc_object_store.derivation.derived_metadata` supplies an explicit policy. Portable writes refuse managed extension/artifact descriptors before publication. Snapping explicitly chooses geometry-only derivation, preserving application metadata and recording the complete source reference without pretending to copy payload closure. | A source with real versioned manual extension rows and an artifact is rejected by the default writer; explicit derived output and the snapping caller omit managed descriptors and preserve source identity. The selected source remains unchanged. |
| PY-4 | Umbilicus request parsing, project serialization, manifests and commit provenance retain the owning `store` in addition to UUID/generation/version. Legacy requests retain an empty owner rather than guessing the attachment. | `test_umbilicus_owner_survives_request_manifest_and_commit`: distinct owners at equal counters remain distinct through request/manifest/provenance roundtrip. Native sender covered by the C++ consumer lane. |
| PY-5 | Raw surface materialization delegates validity to the shared typed `Surface` rule and uses the already selected frame. | `test_raw_surface_validity_uses_shared_reader`, with and without a mask: full/partial sentinel, nonfinite and zero-Z vertices match the native typed snapshot reader. |
| AM | Digest format v3 hashes complete owner/generation/object/version identity; dataset manifests reuse the catalog references rather than querying unrelated current heads. | UUID-normalization/digest regression distinguishes identical counters from different owners; the format bump invalidates old caches. |
| AL | Neural-copy queue CLI accepts a complete reference and returns it in source provenance. Selected names also resolve through the exact reference, not a current-head lookup. | `test_surface_store_io.py`: queue serialization, source metadata changes and current deletion preserve exact geometry, frame, name and provenance; mixing exact and legacy selectors is rejected. |

The overlap materialization caller also migrated to the transport lane's complete `a_reference`/`b_reference` interface, removing loose endpoint metadata/geometry arguments. No consumer-specific storage-location resolver, persistent schema or payload copy was added.

Complexity review replaced a separate exact-record lookup, custom missing-record check and geometry snapshot lookup with the existing exact graph-read operation. Source-change guards remain necessary: the training dataset uses current spatial queries as well as pinned geometry, so silently continuing after a source edit would violate its immutable-dataset contract. This deliberately rejects unrelated commits too; reopen the dataset to adopt a changed store. The Spiral `adapter_metadata`/C++ `adapterMetadata` projection remains a documented language-boundary protocol with a parity fixture: coordinates, selected source tuning, sampling group and fiber/track family fields are fitting semantics, not generic vcobj placement policy.

Validation on Linux/Python 3.14 used the existing native build; no dependencies were installed:

```sh
PYTHONPATH=local/vcobj-pr-tests/build/python:local/vcobj-pr-tests/reader-python:vesuvius/src /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_fiber_store_versions.py vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_fiber_store_sources.py vesuvius/tests/neural_tracing/test_surface_store_io.py
PYTHONPATH=local/vcobj-pr-tests/bootstrap:local/vcobj-pr-tests/spiral-build:local/vcobj-pr-tests/build/python:spiral-fitting:spiral-fitting/tests FIT_SPIRAL_DEVICE=cpu timeout 60s /home/hendrik/.venv_las/bin/python -m pytest -q spiral-fitting/tests/test_object_store_materialize.py spiral-fitting/tests/test_object_store_startup.py -k 'not packed_normals and not rank and not real_data and not fit_context and not resumed_fit'
```

Results: 19 neural consumer tests passed; 29 Spiral tests passed with six runtime/native-normal/optional-data cases deselected. `compileall` and `git diff --check` passed. The first broader neural run segfaulted in the existing threaded native `vc_reader.read_regions` numerical snapping test, before the changed writer; the broader Spiral run stalled after 21 passes around native-normal coverage and was stopped. Those are validation limitations, not proven unrelated defects or waived requirements. The isolated snapping-policy regression replaces only the numerical sampler, so it verifies metadata/copy behavior without claiming numerical snapping validation.

These fixtures use local synthetic stores. Native referenced/sharded/packed routing and independent-branch storage cases belong to the core/transport suites; the Python tests demonstrate complete-reference propagation and do not independently instantiate every placement. Portable derived writes intentionally do not implement artifact/manual-extension preservation; callers needing a real copy must use shared collection copy/export instead of this new-geometry writer.
