# Selective catalog projects

Select a sample in Open Data Catalog and choose **Create Project...** beside
**Open Sample**. Everything starts unchecked. Select source volumes, individual
derived representations (grouped by type), and tifxyz segments. Group checkboxes
and Select all/Select none also work. Set the project name and destination
`.volpkg.json`, then choose **Create and Open**. Existing files require confirmation;
the currently open project cannot be overwritten through this action.

Creation also makes `fibers/<sanitized project JSON filename>/` beside the JSON,
even for an empty project. Copy fiber JSON files into that directory. For example,
`My Project.volpkg.json` uses `fibers/My_Project.volpkg.json/`. File -> New Project
creates the same directory. Existing files there are preserved.

An empty selection creates an empty project. Open Sample still uses the existing
full-sample workflow. The selective path never loads the cached full-sample project.
Saved projects reopen through ordinary project loading and use the global cache.

The initial JSON destination uses the same directory as File -> New Project
(normally the per-user `.VC3D` directory). Volumes, Representations and Segments
have separate checkbox tabs that share the catalog's model-to-display formatters,
including source filename, resolution, energy, export date, prediction/artifact
type, model, level, coordinates, parameters and URL where applicable. Long values
have full-text tooltips. Select all/none operates across all three tabs.

Selecting a derived resource includes its channel volumes, using existing
attachment/coordinate validation. It does not add raw scans, rebased scan views,
or unrelated representations. Lasagna can resolve directly against a selected
L0 scan without a rebased view. Segments include their catalog
coordinate representations and available transformed views, as in ordinary catalog
attachment, but project entries point to individual segment directories rather
than cache-wide aggregate folders. Unselected cached segments remain untouched.

Preparation runs on the existing asynchronous open worker. A detached package
prevents preparation from replacing the active project's autosave. After successful
save, the new JSON is loaded and the usual annotation-session save/switch gate runs.
Canceling the selection dialog or failing to save leaves the active project open.

Implementation: `OpenDataCatalogWindow` owns the dialog; `MenuActionController`
reuses the asynchronous open flow; `OpenDataSampleProject` handles fresh saving and
resource filters; `OpenDataSegmentCache` provides individual-entry reconciliation.

Main and overlay volume selectors put recognized volumes first and prefix their
existing labels with `fiber -` (manifest presence group), `scan -` (catalog source
scan, including rebased source views), or `surf -` (surface prediction). These are
display aliases only: stored names, IDs and paths are unchanged. Unknown/untagged
volumes retain their original labels. Ordering within each priority group remains
unchanged. `VolumeDisplayNames.hpp` shares this behavior between selectors.

Overlay coordinate compatibility is shared by selection, restoration and viewer
application through `OpenDataCoordinateIdentity.hpp`. Only two nonempty,
conflicting coordinate-space tags reject an overlay. A tagged catalog scan and
an untagged local prediction use the ordinary untagged overlay sampling/scaling
path; absent tags are not evidence of an incompatible coordinate system.
