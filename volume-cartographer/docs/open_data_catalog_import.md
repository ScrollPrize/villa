# Selective catalog projects

Select a sample in Open Data Catalog and choose **Create Project...** beside
**Open Sample**. Everything starts unchecked. Select source volumes, individual
derived representations (grouped by type), and tifxyz segments. Group checkboxes
and Select all/Select none also work. Set the project name and destination
`.volpkg.json`, then choose **Create and Open**. Existing files require confirmation;
the currently open project cannot be overwritten through this action.

Fibers and the user's own surfaces live in an explicitly attached object store
(`.vcobj`). Project creation does not create one. Use File → Object Store →
Attach Object Store..., then the explicit fiber import command for current-format
transfer JSON.

An empty selection creates an empty project. Open Sample still uses the existing
full-sample workflow. The selective path never loads the cached full-sample project.
Saved projects reopen through ordinary project loading and use the global cache.

Native catalog source scans use the catalog volume ID (for example,
`20260102150214`) when opened, including from existing saved projects. This lets
object-store fibers bound to that ID match the streamed scan automatically.
Predictions and rebased source views retain distinct IDs; sharing a catalog tag
does not make their coordinate frames interchangeable.

The annotation ID is separate from the remote chunk-cache identity. Catalog
scans reuse the original source cache directory (the URL-derived ID, or the
upstream metadata UUID when present), with separate directories for rebased
views. Existing directories are neither renamed nor merged. Any cache created
under the plain catalog ID by the earlier implementation is left untouched.

The initial JSON destination uses the same directory as File -> New Project
(normally the per-user `.VC3D` directory). Volumes, Representations and Segments
have separate checkbox tabs that share the catalog's model-to-display formatters,
including source filename, resolution, energy, export date, prediction/artifact
type, model, level, coordinates, parameters and URL where applicable. Long values
have full-text tooltips. Select all/none operates across all three tabs.

Selecting a derived resource includes its channel volumes, using existing
attachment/coordinate validation. It does not add raw scans, rebased scan views,
or unrelated representations. Lasagna can resolve directly against a selected
L0 scan without a rebased view. Selected segments are imported as described
under *Catalog surfaces*; unselected ones are not imported.

Preparation runs on the existing asynchronous open worker. A detached package
prevents preparation from replacing the active project's autosave. After successful
save, the new JSON is loaded and the usual annotation-session save/switch gate runs.
Canceling the selection dialog or failing to save leaves the active project open.

Implementation: `OpenDataCatalogWindow` owns the dialog; `MenuActionController`
reuses the asynchronous open flow; `OpenDataSampleProject` handles fresh saving and
resource filters; `OpenDataSurfaceStore` imports segments.

## Catalog surfaces

Open Data segments are surfaces of one internal object store per sample,
`<remote cache>/open_data/stores/<sample>.vcobj`, written only by the importer.
Open Sample imports every segment of the sample; Create Project imports the
selected ones. Each segment representation is one surface record with a
deterministic UUID (a UUIDv5 of the representation id), so every machine names it
the same way:

- the authored tifxyz of the segment's source volume (native, in that volume's
  downscaled voxels);
- a published transformed tifxyz for another sample volume;
- for the other volumes with a sample matrix from the source volume, a generated
  representation: the source moved into that volume by the matrix.

Records are bound to their catalog volume id at L0 (`scale_factor` is
`1/downscale` for native representations; coordinates are not rewritten) and keep
the catalog details in `meta.open_data` (sample and segment ids, lineage,
representation, source URL, the generation recipe, ink detections). Re-importing
updates the records; geometry already fetched or generated stays.

A record starts without geometry. Activating it fetches the registered tifxyz
into the store (or computes a generated one, fetching its source first); the
catalog window's **Fetch Selected** and **Fetch All Surfaces** do the same ahead
of time and show each segment as missing, remote or stored.

While an Open Data volume is open (recognized by its volume tags), VC3D lists the
sample's store read-only beside the user's store, scaled from L0 to the open
volume's pyramid level. Catalog surfaces can be viewed, focused and drawn as
overlays; they are never copied. Editing one (or rotating it) edits it in the
user's store as the same surface: its catalog version, as stored, becomes its
first revision there, it is listed from the user's store from then on, and
**Reset to Open Data Version** (surface panel context menu) restores that
revision. **History...** views or restores any revision of a surface. Ink detections stay the published PNG/JPEG
images; the ink overlay downloads them into the surface's artifact directory.

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
