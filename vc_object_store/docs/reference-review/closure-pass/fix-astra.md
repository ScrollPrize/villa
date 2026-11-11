# Independent review of closure fixes

Baseline: `70da58935`. Reviewed the working production/test delta and
`fixes.md`; no production edits by this reviewer. No competing build or test
runner started while the main approved runner was active.

## Result

No blocking defect or new regression confirmed in the three-finding fix delta.
The implementation follows the requested sharing/deletion direction rather than
adding parallel identity decoding, shell resolvers or provenance compatibility
branches. Runtime completion belongs to the main runner and must be recorded in
`fixes.md`; this report is an independent source-level delta review.

### CP-CORE-01

`ExtensionBindings.hpp::extension_rows_at_version` now uses the existing native
relationship reference codec, matching its object sibling. Native exact-version
selection remains unchanged. The added Python helper tests unknown fields,
invalid numeric types/ranges and a missing lineage through both store and
transaction entry points, and is invoked by both object and relationship tests.
Object tests already span history on/off. This is sufficient scoped boundary
coverage: input rejection occurs before placement/history-dependent resolution.

### CP-C01

Traced the complete sibling chain:

- `objectPins` enumerates ordinary objects and the shell once; a role bit is
  application data beside the unchanged shared reference.
- Both full and incremental reads use that enumeration and `readObject`.
  `manifestPins` also uses it, so `_fitRecords` admits shell notifications.
- `readObject` routes the shell to its separate result slot, including missing
  geometry and read errors; an invalid non-surface shell fails explicitly.
- Count/unchanged and incremental merge now include the shell. A missing shell
  remains an engaged result with `Missing` state, so deletion is not confused
  with an incremental read that did not inspect the shell.
- `fitSurfaceChanges` replaces the former duplicate full/incremental display
  paths, deriving the shell category and exact display key consistently. Missing
  shell results emit that same key in `removedIds`. Existing
  `installInputSurfaces` removes category membership, source mappings, colors and
  CState geometry for those keys. Full reads still reset both relevant categories;
  incremental reads only replace/remove supplied entries.
- Error display now includes shell problems. `_fitRecords` also excludes fitted
  shells from `syncSurfaceDrafts`, consistent with the existing rule for fitted
  patches rather than introducing a draft for a committed displayed input.

**Endpoint regression check:** the broadened enumeration does not broaden the
fitter selection. `compareRelationships` excludes shell pins from endpoint kinds;
incremental incident enumeration excludes them too. For an independently supplied
relationship notification, shared `fittingRelationshipInput` rejects an edge
whose shell endpoint is outside that kind map. No duplicated family policy was
added. Existing manifest duplicate-object validation also prevents one UUID from
silently serving both roles.

Inspected the new `shellRefreshMatchesFullRead` fixture: local history on/off and
all four referenced source/overlay policy pairs; metadata edit, geometry edit,
deletion, count/unchanged, unrelated refresh and full-versus-incremental state/grid
comparison. The existing shell-to-PCL edge remains in the store even though it is
removed from the consumed manifest, so the no-added-relationship assertion really
checks endpoint exclusion. `manifestPins` role/admission is separately asserted.

Remaining coverage limit: no actual GUI event-loop/notification/overlay runtime
test was added. Admission and shared display replacement/removal were inspected
against their real callers; the focused native adapter test alone does not prove
those GUI operations ran. This is an explicit validation limit, not a newly
identified implementation failure.

### CP-C02

`_StoredSurfaces.save` retains the selected source reference under
`parameters.input.reference` and removes the redundant application `surface`
field. Existing actual-source and chained-output provenance assertions now check
the shared shape. Geometry, source UUID list and selected-version loading are
unchanged; no legacy-reader branch or new schema is introduced.

## Scope and limits

Reviewed all nine changed production/test files against the baseline and the
named sibling consumers above. No independent runtime, GUI, macOS/arm64 or
concurrency-stress execution is claimed. No additional broad architectural audit
or schema redesign is implied by this targeted fix review.
