# Selective catalog project creation

## Final verification and commits

- Rebuilt VC3D, test_open_data_manifest and test_catalog_project_dialog.
- Final focused tests passed: test_open_data_manifest (12.08s),
  catalog_project_dialog (0.09s), unified_browser_dialog (0.57s).
- The previously unresolved extra-import report was identified and fixed as
  automatic rebased scan attachment; see correction below. Existing saved
  projects are not rewritten. No new live GUI verification claimed.
- User requested committing the catalog work and browser fixes separately.

## Unselected rebased scan correction

- User's PHerc1299 JSON identifies the unwanted volume exactly: the selected
  2.399um scan plus an automatically generated #vc-base-scale=2 view (9.596um).
  This was not an additional scan or a failure to propagate checkbox selection.
- Removed automatic rebased source attachments whenever source selection is
  explicit. Full Open Sample retains its previous exhaustive behavior.
- Removed the misleading dialog tooltip advertising implicit source views.
- Regression checks Lasagna-only and scan-plus-Lasagna source attachment and
  verifies the selected L2 Lasagna resolves directly against the L0 scan with
  workingToBaseScale=1. Existing saved projects are not silently rewritten.

## Directory navigation and creation follow-up

- Reproduced Enter on a local directory accepting the dialog: QLineEdit's
  return handler navigated, then the same key activated the default Open button.
  A path-field event filter now consumes Return/Enter after handling navigation.
- Added a local Show hidden files toggle and regression coverage.
- The create-dialog name now follows destination filename changes unless the
  user explicitly edits the name. Previously it remained the sample ID.
- Extended the offscreen dialog test to submit one checked source with all
  representations and segments unchecked and inspect attachment output.
- Extra-import report is not yet reproduced: selection propagation and fresh
  project isolation are intact in the inspected paths. Do not claim it fixed;
  the reported saved project/selection is still needed to identify divergence.

## Mixed catalog/local overlays

- Found a second, stricter check in ViewerManager::setOverlayVolume, despite
  the dropdown already allowing tagged/untagged mixtures. This caused runtime
  rejection before volume sampling or size matching.
- Extracted coordinate tag lookup and compatibility into
  OpenDataCoordinateIdentity.hpp. Dropdown population, overlay restoration and
  viewer application now call the same helper; removed duplicated predicates.
- Regression covers tagged scan/untagged fiber in both directions, both
  untagged, matching tags and explicit scan/level conflicts. Sampling unchanged.
- Validation: rebuilt VC3D and test_open_data_manifest; focused CTest passed
  (11.69s). No live GUI overlay validation performed. No commit requested.

## Volume dropdown aliases

- Shared display-only aliases and stable priority ordering between main and
  overlay selectors: fiber for manifest presence, scan for catalog CT source
  views, surf for tagged surface predictions. Original names and IDs retained.
- Unknown/untagged volumes are not classified by filename; they retain their
  original labels. No new metadata requests or persisted renames.
- Added regression checks for recognized tags and nonmatching normal channels.
- Built VC3D and test_open_data_manifest in the existing build directory.

## Filtered catalog dialog crash

- Reproduced SIGSEGV with an offscreen Qt test: select Paris4, filter the sample
  list to Paris4, then invoke Create Project. GDB identifies null
  QTableWidgetItem::text in createSelectedProject's row-copy lambda.
- Filtering repopulated sample rows without reliably refreshing detail tables.
  The new dialog incorrectly treated those widgets as authoritative data.
- Now share model-to-display row formatters between browsing and creation; the
  creation dialog uses its manifest snapshot, not detail-table cell pointers.
  Sample repopulation blocks intermediate selection signals and explicitly
  refreshes the selected sample's details at the end.
- Added catalog_project_dialog offscreen regression plus an injected-manifest
  constructor for deterministic no-network GUI testing. Optional live-metadata
  input: VC_CATALOG_TEST_MANIFEST=/absolute/path/to/metadata.json.
- Reproduction before fix: catalog_project_dialog failed SIGSEGV (3.25s).
- After fix, offscreen open/cancel after filtering passes, including running with
  downloaded live catalog metadata and its Paris4 sample (0.82s). No volume data
  download needed. The regression verifies the correct sample title, resource
  identities/counts and unchecked initial state, not only absence of a crash.
- Commands: `cmake --build volume-cartographer/build --target VC3D test_catalog_project_dialog -j16`;
  `ctest --test-dir volume-cartographer/build -R '^(catalog_project_dialog|test_open_data_manifest)$' --output-on-failure`.

## Default destination and metadata follow-up

- Share the ordinary new-project destination policy through
  ProjectCreationDefaults.hpp; prefill the catalog JSON path and use the same
  directory for its browser. Added a default-directory regression test.
- Replace the misleading Source / type column (which contained a path) with
  separate Volumes/Representations/Segments checkbox tabs. Headers, display order
  and cell values are taken directly from the catalog tables, retaining all their
  metadata without duplicating formatting/classification. Full values use tooltips.
- VC3D and test_open_data_manifest rebuilt successfully; catalog CTest passed
  (11.61s), including the shared default-directory regression. Whitespace check
  passed. Expanded GUI tabs still need manual visual validation.

## Empty fiber directory follow-up

- Project creation now eagerly creates the annotation fiber folder, including
  empty selections and ordinary File -> New Project. Extracted the existing
  annotation path naming into ProjectFiberPaths.hpp; creation and discovery share
  that helper, preserving current naming and source-redirection behavior.
- Directory failures propagate through existing project-creation errors. Existing
  contents are preserved; regression coverage checks empty creation, filename
  sanitization, relative paths, fallback roots and preservation on repeated creation.
- Rebuilt VC3D and test_open_data_manifest with the existing build command above;
  catalog CTest passed (11.23s). Diff whitespace check passed.

Existing volumeIds filter gates derived resources too; add independent raw-only
selection without breaking old callers. Segment selection will reuse reconciliation
with a filtered segment list while retaining coordinate metadata. Fresh projects
must not load cached full-sample entries.

The playbook's scripts/build_dependencies.sh is absent; use existing CMake build
without installing dependencies.

## Implementation and review

- Independent plan review clarified coordinate dependencies and save-failure
  semantics. Implementation review found active autosave writes and source-view
  pairing dependency; both fixed before completion.
- Added Create Project with grouped checkboxes, all/none, name/path browser,
  overwrite confirmation, empty default and immediate open.
- Shared resource attachment receives independent raw-source and segment filters.
  Selected representations retain required rebased source views/channel volumes.
- Segment selection uses individual entries; existing aggregate folders would leak
  cached unselected segments. Reuse placeholder generation and shared entry-tag
  helpers, skip orphan marking and stale-root removal for selective attachment.
- Fresh preparation uses a detached package, then saves and reloads chosen JSON.
  Current project stays open on save failure; annotation save gate runs at switch.
  Reject overwriting the active project's own file.
- Regression tests cover independent source/representation selection, Lasagna's
  required rebased view, empty project, cached project isolation, individual
  segments, unchanged unselected metadata/autosave, save failure and reopening.
- Final VC3D and catalog test build succeeded. Catalog CTest passed (11.25s),
  including the new regressions and existing full-sample behavior. Diff whitespace
  check passed.
- Final independent review confirms autosave and source-view fixes. Coverage of
  successful remote prediction descriptor pairing remains a live-data validation
  gap; unit tests cover independent filtering and Lasagna rebased-view creation.
- Build command: `cmake --build volume-cartographer/build --target VC3D test_open_data_manifest -j16`.
- Test command: `ctest --test-dir volume-cartographer/build -R '^test_open_data_manifest$' --output-on-failure`.
- No live GUI or remote catalog import has been exercised; these remain manual
  validation limitations, not claims of verification. No commits requested.
