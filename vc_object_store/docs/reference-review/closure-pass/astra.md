# Independent closure-pass adjudication

Frozen production commit: `69ba26311`; implementation delta: `878525dd1` against
`153c9a10f`. Review-only work; no production changes or schema proposals.

## Outcome

No new regression confirmed in the inspected implementation delta. Two earlier
omissions reported by the other lanes are independently supported by source
traces; neither is dismissed because existing suites pass. A third, lower-priority
representation cleanup is accepted separately below:

1. **Relationship extension reference decoding** (`python/ExtensionBindings.hpp`,
   `extension_rows_at_version`, relationship branch). The object sibling calls
   `objectVersionFromJson`; the relationship sibling constructs a reference from
   four independent Python field reads. Unknown keys therefore bypass the strict
   shared codec, and numeric adaptation differs from the object branch. The same
   code exists in `878525dd1^`. Accepted as a boundary-validation/shared-policy
   omission, not a regression. Fix through `relationshipVersionFromJson`, with
   object/relationship malformed-input parity tests. Native typed overloads in
   `ExtensionHistory.cpp` correctly delegate exact selection to `objectAtVersion`
   and `relationshipAtVersion`; they should not acquire a second JSON decoder.

2. **Outer-shell incremental refresh** (`SpiralStoreInputs.cpp` and
   `SpiralWorkspace.cpp`). Full `readFitInputs` reads the manifest shell separately,
   but `manifestPins` returns only `manifest.objects`. Workspace `_fitRecords`
   therefore excludes the shell; `noteStoreChanges` treats its surface update as
   an unrelated rebuild input, not `_pendingObjectChanges`. There is no automatic
   full reload in this branch. Even directly calling `readFitInputChanges` with
   the shell UUID omits it, and `applyFitInputChanges`/`installFitInputChanges`
   contain no shell replacement/removal path. Accepted as stale installed shell
   geometry/state until full reload. A generic changed-other-inputs freshness
   warning can still appear: this is not a claim that every warning is absent.
   The incomplete incremental route predates this delta. Required coverage is
   full load followed by shell edit/deletion and incremental apply, plus GUI
   notification admission and shell-overlay replacement/removal.

3. **Flattened neural-copy input provenance**
   (`vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py`,
   `_StoredSurfaces.save`). `parameters.input` expands `loaded_version` beside
   the extra application field `surface`. The selected full identity is preserved;
   there is no demonstrated geometry error or wrong-owner selection. Nevertheless,
   this deliberately persisted input pin is an alternate application reference
   shape, contrary to the explicit extra-data-wraps-reference contract. Accept as
   low-priority representation cleanup, not a high-severity correctness defect:
   retain the value under `input.reference`, keep application metadata beside it,
   and update the two provenance assertions in `test_surface_store_io.py`. No
   generic provenance framework, schema change or inferred migration requirement
   follows. Existing code/tests show this predates the reviewed delta.

## Coverage and dispositions

Read the completed `core.md`, `consumers.md` and `root.md` reports. Adjudication:
accept CP-CORE-01 (low boundary consistency), CP-C01 (medium stale shell
display/state), and CP-C02 (low representation cleanup) with the qualifications
above. In CP-C02 the four identity fields survive in generic provenance metadata;
this is not evidence that the resulting decorated object itself passes the strict
reference codec. None of these source-established omissions is runtime-reproduced
by this independent reviewer. No additional finding is inferred from inventory
counts or untested combinations.

| Invariant / concrete entry points | Disposition and evidence |
| --- | --- |
| Strict reference and position ingress: `Wire.cpp::manifestReferences`, `internal/ObjectVersionJson.hpp`, `objects.cpp` surface/manifest bindings; `StoreReader::selectedObject` | Delta source-traced. Native codecs own parsing; exact selection executes before current lookup. Complete exact reads do not gain a current-head dependency. Extension exception above remains open. |
| Selected relationship identity: `History.cpp::relationshipsAtSelected`, `internal/RelationshipOrder.hpp`, wire relationship/selected-input decoding | Delta source-traced. Membership keys now retain complete `RelationshipVersion`; physical archive lookup stays in storage layer. Stable UUID/full-reference ordering is documented. No counter-only replacement found in these paths. |
| Selected transformed surface: `SurfaceGrid.cpp::readSurfaceGrid`, typed view reference checks | Delta source-traced. Requires selected reference as well as geometry identity; anonymous payload views cannot certify selected metadata. |
| Tool frame inference: `ToolObjects::coordinatesFor` | Delta source-traced. Loaded pins retained for each source; compares actual per-source declared/loaded frame instead of first-source inference. |
| Point import: `formats/PointCollectionInput.cpp`, `geometry/PointMap.cpp` | Delta source-traced. Shared ID/point decoder owns validation; source importer retains application annotations and explicit collision renumbering. |
| Artifact read/write/export: `Artifacts.cpp`, `ObjectStoreSession::writeArtifact`, `SourcePaths.cpp::validateExportDestination` | Source-traced shared resolver and atomic publication. New local session export guard runs before directory creation; guard traverses dependency databases/container and sidecars. Hosted destinations are client-local, legitimately not server paths. Artifact native regression passed. |
| Copy/packing/conversion: `ObjectCollection.cpp::copyObjectCollection`, `PackedCollection.cpp::packObjectCollection`, `PackedStore.cpp` import/finalize call sites, `internal/DatabaseCopy.hpp`, `internal/RecordOverrides.cpp`, conversion relocation call | Source-traced shared snapshot copying, dependency closure, artifact verification and locator relocation, including archived metadata. Placement logic is storage-layer work, not consumer duplication. Existing packing/conversion/relocation tests passed. No new defect confirmed. |
| Extension current/exact/read/write siblings: `Extensions.cpp`, `ExtensionHistory.cpp`, `ObjectCollection.cpp` extension delegates, Python `ExtensionBindings.hpp` | Native paths share visible-state selection, exact revisions, extension descriptors and savepoints. Native tests passed; Python relationship ingress omission above remains open. |
| Import/reference/history-policy siblings | Existing `initial_import`, `collection_reference_endpoints`, `collection_audit`, `history_policy`, `history_off_views`, `coherent_api` tests passed. This is regression coverage, not a claim that every Cartesian combination was independently recreated. |
| Geometry export: `geometry/GeometryExport.cpp::exportGeometry` | Source-traced selected metadata/geometry and owner publication graph selection; local destination guard shared. Exact path does not first require current head. No new defect confirmed. |

## Runtime evidence

Ran against the existing Linux native build; no competing rebuild or install:

```sh
AGENTS_AGENT_MODE=1 ctest --test-dir local/vcobj-pr-tests/build --output-on-failure --parallel 2 -R 'object_store_(external_locator_relocation|coherent_api|extensions|collection_reference_endpoints|history_off_views|collection_audit|history_policy|collection$|packed|initial_import|artifacts|convert)'
```

Result: **12/12 passed**, 8.74 seconds. Inputs are the checked-in synthetic
temporary-store fixtures. This does not close the newly identified malformed
Python extension boundary or missing shell incremental cases.

## Limits

Independent source review and focused native regression run, not a full GUI
runtime test. No macOS/arm64 execution, performance benchmark, new concurrency
stress run or exhaustive malformed-input fuzzing. Other-lane runtime evidence
must be read alongside their reports; this report does not label their probes
as independently rerun. Full external consumer inventory belongs to the consumer
lane. No claim of repository-wide absence of undiscovered bugs.
