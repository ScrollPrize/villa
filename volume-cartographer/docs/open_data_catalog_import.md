# Selective catalog projects

Select a sample in Open Data Catalog and choose **Create Project...** beside
**Open Sample**. Everything starts unchecked. Select source volumes, individual
derived representations (grouped by type). Group checkboxes and Select all/Select
none also work. Segments are not chosen here: they are surfaces of the sample's
store (*Catalog surfaces*). Set the project name and destination
`.volpkg.json`, then choose **Create and Open**. Existing files require confirmation;
the currently open project cannot be overwritten through this action.

Fibers and the user's own surfaces live in an explicitly attached object store
(`.vcobj`). Project creation does not create one. Use File → Object Store →
Attach Object Store..., then the explicit fiber import command for current-format
transfer JSON.

An empty selection creates an empty project. Open Sample still uses the existing
full-sample workflow. The selective path never loads the cached full-sample project.
Saved projects reopen through ordinary project loading and use the global cache.

A remote volume the catalog publishes is named by the catalog when opened,
including from existing saved projects (`vcr::catalog::OpenDataManifest::volumeAt`).
A source scan uses the catalog volume ID (for example, `20260102150214`), so
object-store surfaces and fibers bound to it match the streamed scan. A source
opened at pyramid level n is the volume `<id>@Ln` (`levelVolumeId`), related to
the scan by the catalog's level registration (scale 2^-n), so objects bound at
any level are seen at every other. Predictions keep IDs of their own.

The annotation ID is separate from the remote chunk-cache identity. Catalog
scans reuse the original source cache directory (the URL-derived ID, or the
upstream metadata UUID when present), with separate directories for rebased
views. Existing directories are neither renamed nor merged. Any cache created
under the plain catalog ID by the earlier implementation is left untouched.

The initial JSON destination uses the same directory as File -> New Project
(normally the per-user `.VC3D` directory). Volumes and Representations
have separate checkbox tabs that share the catalog's model-to-display formatters,
including source filename, resolution, energy, export date, prediction/artifact
type, model, level, coordinates, parameters and URL where applicable. Long values
have full-text tooltips. Select all/none operates across both tabs.

Selecting a derived resource includes its channel volumes, using existing
attachment/coordinate validation. It does not add raw scans, rebased scan views,
or unrelated representations. Lasagna can resolve directly against a selected
L0 scan without a rebased view.

Preparation runs on the existing asynchronous open worker. A detached package
prevents preparation from replacing the active project's autosave. After successful
save, the new JSON is loaded and the usual annotation-session save/switch gate runs.
Canceling the selection dialog or failing to save leaves the active project open.

Implementation: `OpenDataCatalogWindow` owns the dialog; `MenuActionController`
reuses the asynchronous open flow; `OpenDataSampleProject` handles fresh saving and
resource filters.

## Catalog surfaces

Open Data segments are surfaces of one store per sample,
`<remote cache>/open_data/stores/<sample>.vcobj`, built and owned by the object
store from vc_reader's catalog (`vc_object_store/objects/CatalogStore.hpp`).
While a catalog volume is open, VC3D attaches its sample's store
(`attachCatalog`) and lists it read-only beside the user's store. Each segment
representation (`vcr::catalog::OpenDataSample::surfaces`) is a placeholder
record with a deterministic UUID (a UUIDv5 of the representation id), bound to
its catalog volume, with its bbox from the catalog's volume coverage and the
catalog reference (sample, segment, tifxyz URL, published ink images) in
`meta.catalog`.

A placeholder has no geometry. The store fetches it on the first read that needs
it: a viewer region or nearest-surface query that reaches it, or opening it.
Failed fetches are not kept and are tried again later. Surfaces are read through
the stores' registrations, so a catalog level shows them at its scale.

Catalog surfaces can be viewed, focused and drawn as overlays. Editing one (or
rotating it) imports it into the user's store as the same surface
(`ObjectStoreSession::importObject`): its catalog version, as stored, becomes its
first revision there, the user's copy hides the catalog one, and **Reset to Open
Data Version** (surface panel context menu) restores that revision
(`restoreImported`). **History...** views or restores any revision of a surface.
Ink detections stay the published PNG/JPEG images; the ink overlay has the store
download them into the surface's artifact directory (`publishedFile`).

The catalog window's **Refresh from Open Data** re-reads the published catalog
into the sample's store (`refreshCatalog`): placeholder metadata follows it, new
surfaces are added, and fetched surfaces whose published tifxyz changed are
fetched again as a new revision. The user's store is untouched. Nothing is
refreshed automatically.

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
