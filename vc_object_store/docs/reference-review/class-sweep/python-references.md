# Python class sweep: numeric projections versus complete references

Frozen source: `d13d33629c3ccee94fa61a9ef873ec82ea0b9db9`. **Read-only targeted review before fixes.** No production/test edits, installs, builds or runtime probes. Scope is the complete Python vcobj facade and repository-wide Python consumer search for constructing/stripping/rebuilding versioned references, not an arbitrary fixed number of findings. Native C++ binding internals belong to the parallel native lane.

## Result and candidate

**One known numeric-reference seed; no additional confirmed numeric-to-reference reconstruction defect.** CS-P01 below is an alias/disposition of **NX-P01**, not a new unique finding. The requested additional saved-protocol container screen identifies CS-P02/P03 for Astra adjudication; they are boundary shape-validation candidates, not reference-owner or geometry counterexamples. Other candidate-shaped call sites are explicitly classified below. No fixes performed.

### CS-P01 / known NX-P01 — receipt marker is downgraded then reconstructed

`spiral-fitting/object_store_inputs.py:2259` (`submit_link_plan`) selects the marker from the numeric `receipt['objects']` projection. `_link_marker_at` (`2311–2324`) then assembles `{store: request.store, generation: request.generation, object: link_marker_uuid(), version: int(number)}` for `objects_at_versions`. The complete marker reference is already available in the receipt. The sibling `_refresh` (`2598–2601`) uses the shared `receipt_object_reference` correctly for its selected inputs.

Disposition: **confirmed low-priority duplicate handle reconstruction**, previously adjudicated as NX-P01. Use the existing shared receipt reference instead of interpreting the number/request. Retain the full reference through the marker read; no second resolver or legacy fallback is needed. The native LinkObjects operation (`src/linking/LinkObjects.cpp:375–389`) publishes the marker in its committing root with a new commit stamp, so the ordinary current implementation's owner assumption works; do not claim a wrong-owner counterexample that this review has not demonstrated. No fixes performed in this sweep.

### CS-P02 — saved Lasagna input-reference collection is not validated as an array

**Low boundary-validation candidate.** `lasagna/fit2tifxyz.py:386–394::_stored_sources` requires a saved job's `input_references` key, but iterates its raw value and validates only each element with scalar `object_version_from_json`. At the actual saved checkpoint ingress, `_export` loads `st['_job_spec_']` (`1312–1319`) and calls this function to derive sources for stored output. There is no earlier native array decode on that field. `_stored_sources({'input_references': {}})` or the same field set to `''` reaches an empty comprehension and returns `()`, so malformed saved provenance is silently treated as an empty list. Nonempty dicts generally fail later as malformed individual references; that does not validate the container.

This is an **external/saved store-provenance boundary**, not an internal typed list loop or an arbitrary optimizer config. Expected representation is the array emitted by `fit_service._body_with_resolved_job_spec`. Validate that shape once at saved ingress, using shared native reference-list decoding where available; preserve ordinary iteration over the resulting typed references. No wrong geometry or authorization bypass is claimed; demonstrated source trace establishes invalid saved evidence acceptance/loss of source provenance. No runtime diagnostic run in this read-only sweep.

### CS-P03 — saved Spiral pin collection shapes can be silently reinterpreted

**Low boundary-validation candidate.** `spiral-fitting/object_store_inputs.py:188–208::InputPins.from_provenance` is used at saved startup/update input ingress (`2650`, `2998`). It decodes the shared position and every reference scalar, but not the enclosing application pin schema. `references = value.get('references') or {}` silently maps empty lists/strings to the expected dictionary; selection values are iterated/sorted without requiring their saved JSON array type. A position-valid pin with `references: []` and `selection: {'fiber': {}}` becomes a valid empty reference map and an empty fiber list. With a real complete reference, `selection: {'fiber': {fiber_uuid: 'ignored'}}` is treated as `[fiber_uuid]`, discarding the mapping values. Native scalar validation preserves exact references, but cannot detect the wrong collection shape once Python has normalized it.

The source's own `provenance()` emits a mapping of references and per-kind arrays; no upstream structural pin-schema validator runs before this reader. This is distinct from `InputPins.from_manifest`, which calls native `manifestReferences` **before** interpreting raw object/relationship arrays, and is safe delegation even if a separate native decoder bug is being reviewed. Suggested direction is one pin-ingress structural check/decoder, not repeated checks in internal iteration or duplicate scalar/reference parsing. No changed-owner or wrong-geometry result is asserted.

## Exact search method

Executed from the repository root. `rg` was the initial search tool; a tracked-file `git grep` crosscheck followed. Searches below are discovery seeds; every relevant construction/consumer was traced to its caller and the result sink. Test fixtures and independent meanings of “reference”, “version”, “generation” and “store” were classified rather than proposed for deletion.

```sh
rg -n -g '*.py' -g '!**/tests/**' -g '!**/test_*.py' -e 'version_store' -e 'version_generation' -e 'object_references' -e 'relationship_references' -e 'expected_versions' -e 'ObjectVersion' -e 'RelationshipVersion' -e 'receipt' .
rg -n -U -g '*.py' -g '!**/tests/**' -g '!**/test_*.py' '("store"|\x27store\x27).{0,100}("generation"|\x27generation\x27)(.|\n){0,200}("object"|\x27object\x27|"relationship"|\x27relationship\x27)(.|\n){0,180}("version"|\x27version\x27)' .
rg -n -g '*.py' -g '!**/tests/**' -g '!**/test_*.py' -e 'dict\(.*store=' -e 'dict\(.*generation=' -e '\[.(store|generation|object|relationship|version).\]\s*=' -e '"store"\s*:' -e "'store'\s*:" -e '"generation"\s*:' -e "'generation'\s*:" -e '"object"\s*:' -e "'object'\s*:" -e '"relationship"\s*:' -e "'relationship'\s*:" .
rg -n -g '*.py' -g '!**/tests/**' -g '!**/test_*.py' -e 'object_reference\(' -e 'relationship_reference\(' -e 'object_version' -e 'relationship_version' -e 'receipt_object_reference' -e 'objects_at_versions' -e 'object_at_version' -e 'relationships_at_versions' -e 'relationship_at_version' -e 'selected_inputs' -e 'store_version' -e 'current_object_revision' .
rg -n -g '*.py' -g '!**/tests/**' -g '!**/test_*.py' -e 'current_version\(' -e 'current_relationship_version\(' -e '\bversion=.*generation=' -e '\breference\s*=\s*\(' -e '\breferences\[.*\]\s*=\s*\(' -e '\*\*.*(reference|loaded_version|store_version)' .
rg -n -g '*.py' -g '!**/tests/**' -g '!**/test_*.py' -e 'vcobj:' -e 'versions_at\(' -e 'expected_relationship' -e 'expected_object' -e 'read_dependencies' -e '\b(reference|references|pins)\b.*\b(zip|tuple|sorted|dict|list)\(' -e '(store|generation|object|relationship|version)\s*=.*\b(reference|revision|receipt|record)\b' .
rg -n -g '*.py' -g '!**/tests/**' -g '!**/test_*.py' -e 'fiber_reference' -e 'surface_reference' -e 'object_reference' -e 'relationship_reference' -e 'source_reference' -e 'version_generation' -e 'version_store' .
git grep -n -E 'version_store|version_generation|object_references|relationship_references|objects_at_versions|relationships_at_versions|receipt_object_reference' -- '*.py' ':!**/tests/**' ':!**/test_*.py'
rg -n 'isinstance\([^\n]*(list|tuple)|list\(|tuple\(' vc_object_store/python/vc_object_store/*.py spiral-fitting/{object_store_inputs.py,spiral_service.py,fit_session.py,service_commits.py} lasagna/{stored_surfaces.py,fit_service.py,fit2tifxyz.py}
```

The multiline literal search finds the two actual dictionary-building sites `object_store_inputs.py:2319–2320` and `2435–2436`, plus the full-reference digest serialization in `fiber_store.py:166`. The independent field/kwargs/assignment/tuple/call searches cover alternate key order, names and representations instead of relying on that one pattern. No production Python manual relationship-reference constructor was found.

## Complete relevant call-site dispositions

### Shared Python facade

| File / line / entry point | Disposition |
| --- | --- |
| `vc_object_store/python/vc_object_store/__init__.py` | Native exports; no Python-built reference or alternate schema parser. |
| `selected.py:11–12`, `SelectedRecord.reference`; `SelectedObject` | Delegates `object_reference(self.record)` to the canonical native helper; selected record remains attached to its revision/geometry. `.version` is a numeric display/property projection, not an internal write guard or rebuilt handle. |
| `surfaces.py:28–37`, `SurfaceGrid.reference/version/generation` | Canonical native reference from the retained selected record. SurfaceGrid's NamedTuple groups geometry/application values, not a replacement reference tuple. |
| `surfaces.py:40–51`, `read_surface` | Explicit external numbered selector or complete reference passed unchanged to native read_surface_grid. Does not construct an owner from numbers. Native selection returns the retained full record. |
| `locations.py:103–125`, `receipt`, `current_object_revision`, `current_version`, exact/selected reads | Delegates to session; numeric current_version is an explicit projection API. No production caller found that takes it then reconstructs a full reference. Exact reads forward caller-provided full references. |
| `locations.py:132–157`, query/edge helpers | UUID-based explicit current selection and query adaptation; not conversion of a selected exact record to a new handle. |
| `derivation.py:29`, `derived_metadata` | Deep-copies the complete selected source reference beside a derivation policy; never reconstructs it from fields. |
| `documents.py:13–44`, document/collection helpers | Explicit UUID/current document APIs delegate native checked operations. No numeric version argument or reconstruction. |

### Spiral

| File / line / entry point | Disposition |
| --- | --- |
| `object_store_inputs.py:177–226`, `InputPins` ingress/refs | Shared native scalar/manifest codec validates saved complete refs and keeps them. `refs()` merely returns those values. Selection UUID keys/outer-shell role are application data beside each reference. |
| `object_store_inputs.py:335–390`, StoredObject/StoredRelationship provenance | Complete reference retained; numeric `version`/UUID fields are output projections checked by the shared manifest validator, not a second internal handle. `stored_order` is endpoint order, not version identity. |
| `object_store_inputs.py:415–450`, InputSnapshot pins/manifest/digest | Carries `obj.reference` directly; digest intentionally excludes versioned pins for separate consumed-content comparison. It does not reconstruct references from that digest. |
| `object_store_inputs.py:480–534`, selectors/current_record | Native selected rows wrapped with SelectedRecord/SelectedObject; no late owner inference. Current discovery is explicit. |
| `object_store_inputs.py:603–619`, filter_relationships | Uses native `relationship_reference` on the selected result and carries it beside family/anchor data. |
| `object_store_inputs.py:658–695`, historical-position/pinned reads | `store.versions_at` already returns full references, forwarded into `objects_at_versions`; pinned reads keep the saved references. |
| `object_store_inputs.py:785–816`, resolve_selection/read capture | Complete selected references merged and passed to native `selected_inputs`; no scalar downgrade. |
| `object_store_inputs.py:1234,1380,1450–1451,1509`, materialized PCL/fiber/overlap metadata | Complete object/endpoint/relationship reference stored as its own field; extra domain metadata remains alongside it. Immutable geometry revision strings are payload identifiers, not new ObjectVersion formats. |
| `object_store_inputs.py:2190–2215`, link_request | Native-selected references become format3 `read_dependencies` unchanged. Request store/generation fields identify transaction scope, not endpoint ownership. |
| `object_store_inputs.py:2259,2311–2324`, marker receipt read | **CS-P01 / existing NX-P01**. Full receipt reference is available; numeric projection/request reconstruction is unnecessary. |
| `object_store_inputs.py:2390–2448`, StoreChange/changes_after | **Legitimate feed-boundary adaptation.** Native `Transactions.hpp::ChangeRecord` contains recordKind/UUID/number/deleted, not a complete reference. Its feed cursor supplies the publishing store/lineage. This code constructs one exact object reference from that external feed envelope; it has not thrown away a previously supplied full handle. Relationship classification intentionally reads a current edge kind, not a reconstructed historical relationship reference. No new bug of the requested class established. |
| `object_store_inputs.py:2597–2605`, _refresh | Numeric receipt iteration only discovers which selected inputs changed; actual repinning uses canonical `receipt_object_reference`. Numeric commit_sequence updates a store position, not an object handle. |
| `fit_session.py:1236–1248`, _checked_umbilicus | External project-input reference validated by native codec, retained beside points. No tuple/flattened substitute. |
| `grow_track_graph.py:2416–2423`, retire_surfaces | Native selected records → shared object_reference → checked write guards; no numeric-map guard. |
| `service_commits.py:98–104,304–321`, CommitResult/commit_in_place | **Display/current-output projection, not reference rebuilding.** A current revision is projected to output surface UUID, geometry UUID and numeric version for status. Traced `spiral_service.py:3469–3471` → native `SpiralServiceManager::CommitResult` → `SpiralPanel::showCommitResult`: the panel uses UUID current-fit discovery; numeric version is not reconstructed into a handle or used as edit authority. Exact metadata selection is not promised by that status result. No same-class defect established; do not silently redesign this service protocol here. |
| `spiral_service.py:1519–1566`, input position binding/receipt checks | Native position decoding plus application attachment/minimum-position checks. Store positions are not object references; no endpoint-owner reconstruction. |
| `fit_spiral.py`, `spiral_runtime.py`, `runners/run_single.py`, `config.py`, `pcl_views.py`, `find_inconsistent_windings.py` | Indirect materialized/pinned input consumers; found no reference builder outside object_store_inputs. Application summaries/selection/content processing do not restore a handle from numeric identity. |

### Lasagna

| File / line / entry point | Disposition |
| --- | --- |
| `stored_surfaces.py:78–82,139–151`, decode_reference/read_surface | Native shared ingress decoder for locators/current discovery; already-complete refs carried unchanged into grid read. |
| `stored_surfaces.py:85–101`, artifact_file/publish_artifacts | Artifact reads use selected complete ref; publication explicitly selects a new/current output and calls shared object_reference. No number → reference construction. |
| `stored_surfaces.py:124–136`, StoredSurface/channel | Reference comes from grid; channel uses immutable selected geometry payload identity. |
| `fit_service.py:672–675,692,1038–1058,1077,1091` | Selected source/model/collection references retained for artifact reads/job input_references; extra job data is separate. No local alternate owner/reference constructor. |
| `fit2tifxyz.py:386–394`, _stored_sources | Validates complete saved references with native codec then projects source UUIDs **only for provenance sources**, not exact read or write guards. Full input_references remain in job/checkpoint state; no reconstruction. |
| `tifxyz_io.py:63–76`, `approval_inpaint.py:88–99`, fit adapters | Delegate selected stored-surface read; external directory alternatives are source-format dispatch, not alternate vcobj references. |

### Vesuvius and other repository callers

| File / line / entry point | Disposition |
| --- | --- |
| `vesuvius/.../inference/surface_store.py:11–21` | Explicit scalar CLI selector or full reference delegates native surface API; returned `extra.store_version` is the complete selected reference. |
| `vesuvius/.../inference/infer_rowcol_triplet_wraps.py:1995–2046`, _StoredSurfaces | Keeps native loaded_version/source_version handles; exact name read forwards the pin. Derived output uses `parameters.input.reference` unchanged. Numeric version/generation options are explicit external selectors, not a rebuilt owner guess. |
| `vesuvius/.../neural_tracing/trace_service.py:283–286` | Echoes external selector options in diagnostic response; no full reference constructed from them. Output surfaces are UUIDs for new/current output discovery. |
| `vesuvius/.../tracers/shared/annotations/fiber_store.py:71–86,132–166` | Native selected record reference used for graph reads, catalog evidence and complete-identity digest. Digest string serializes all fields for hashing only, never as a handle reconstructed later. `_position` at106 is a store-position cache-comparison tuple, not an ObjectVersion. |
| `vesuvius/.../tracers/fibers/sheet_strip/infer_all.py:129,155,162` | Retains source_reference directly in resume evidence/outputs. |
| `vesuvius/.../tracers/fibers/sheet_strip/build_index.py:59` | Compares complete producer fiber_reference against catalog reference; no ordinal-only restore of identity. |
| `vesuvius/.../tracers/sheets/scripts/match_fibers_to_patches.py:58–64,91–92` | Retains catalog reference beside geometric match; tuple job alternative is external file key+points, not a versioned store identity. |
| `vesuvius/.../fiber_spline_generator/store.py:103`, fiber_store.py:191 | New-object absent guards (`UUID: None`) are not numeric references or stripped selected state. |
| `volume-cartographer/python/vc/fiber_evaluation.py:258–266` | Copies complete optional native reference into case results. Geometric `references` arrays are independent evaluation data, not ObjectVersion tuples. |
| `volume-cartographer/tools/vc3d-mcp/vc3d_mcp/tools/stores.py:86,135`, catalog_volume.py:57 | RPC attachment/store-name arguments and result forwarding; exported result `version` is documented as a complete reference, not rebuilt in Python. |

## Non-candidates and explicit exclusions

### Collection/list handling distinctions

- CS-P02/P03 above reopen the two ledger entries whose **scalar identity handling remains correct**, but whose raw saved container shape is not validated. No extra identity-loss finding is counted for the same boundary.
- `HostedStoreHandle` in `locations.py` uses `list(refs)` at119/122/125/163 and `list(ids/fields)` at37/42/138/140/143. This accepts Python iterables such as tuples/generators, and mapping/string coercions could differ from the direct native binding. However these are Python public argument adapters, **not identified raw JSON/saved-protocol decoders**; actual Spiral callers supply `InputPins.refs()` or native full-reference lists. No path from an unvalidated JSON field through these facade calls was established. Do not promote this as a confirmed defect merely because Python `list({})` is empty. Native session/codec receives the resulting typed array and owns its element/reference validation. Astra should record this explicit nonfinding/contract distinction rather than demand repeated internal loop checks.
- `InputPins.from_manifest`/`_compared` call `validated_manifest_references` (native `manifestReferences`) before iterating raw manifest membership. That is the correct validate-once ownership boundary. An array-schema limitation in native manifestReferences, if found by the native lane, is not an independent Python-loop bug.
- Lasagna request `external_surfaces`, `corr_point_collections`, `linked_surfaces` at `fit_service.py:1030,1047,1096` require their documented external lists once before loops; individual store locators then delegate the native scalar decoder. `/objects/query` at2378 also checks its list, but its type/name/hash artifact references are a **separate Lasagna artifact schema**, not an old vcobj-reference format to remove.
- `request_from_config`'s tuple conversion at `object_store_inputs.py:1943–1944` receives Config's documented `strings` fields; command-line `main` at3076 receives argparse `action='append'` output. Those are typed configuration/CLI adaptations, not saved vcobj manifest list decoders. No arbitrary ML-config array audit was added.
- `surface_grid` consumes native `read_surface_grid` output (`scale/channels/record`) and converts arrays/tuples. It must not redundantly validate every list already produced by native code. `StoredSurface.meta` similarly adapts native grid scale. `derived_metadata` copies a complete reference without inventing a new list schema.
- Existing Python native request/response boundaries use canonical list/manifest decoders; ordinary `for` loops over their validated vectors are not bugs. The list representation itself is correct. These findings concern only acceptance of the wrong raw external/saved container **before** a typed value exists.

- `lasagna/fit.py:1038` still advertises the **unsupported ownerless exact locator** in CLI help (`vcobj:<uuid>[@<generation>/<version>]`). Native decoder/stored_surfaces documentation requires store identity and rejects it; no runtime fallback or handle construction implements this obsolete help text. Record as an ancillary documentation inconsistency, not another unique numeric-reconstruction defect or permission to change production during this sweep.
- `spiral-fitting/service_artifacts.py`, preview_index/lasagna_publish/spiral_service status generations, bracket_fiber_links, gpu_manager schemas/engine, DDP FileStore and inference_provenance “store” fields are independent artifact/task/source identities, not vcobj references. They are not eligible for blanket replacement.
- Neural tracer `trace.py` reference dictionaries (`seed_pos`, `seed_tangent`, `seed_age`) and geometry loader/reference axes are algorithm state, not persisted ObjectVersion values. GT/reference-surface solver dictionaries use “reference” in the mathematical sense. No unrelated solver refactor is authorized by this sweep.
- `vc_reader` geometry/training reference parameters concern coordinate sources, not vcobj version references.
- Native Python bindings/hosted codecs are being reviewed by the separate native lane; this report covers the Python facade and consumer call sites, not a second interpretation of C++ codecs.
- Test/smoke fixture dictionaries intentionally construct current/future/malformed references to test validation and inspect numeric projections. They are not production owner-inference logic. This sweep ran no tests and does not infer broad correctness from existing suites.

All relevant production matches of the stated construction/selection/receipt searches have dispositions above, including the second literal builder and selected-state projections. This is a finite **class-wide source sweep**, not a claim to prove every possible dynamically generated program, numeric algorithm, event interleaving or unrelated bug in the monorepo. No additional confirmed candidate of this class remains unreported by this lane.
