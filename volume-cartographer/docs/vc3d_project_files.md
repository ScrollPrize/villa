# VC3D project files

A project is a versioned JSON `.vcproj` file. It references one or more `.vcobj`
object stores, volumes, normal grids, and Lasagna datasets. Stores contain the
objects, their history, registrations, umbilici, and object-owned artifacts.
Multiple projects can reference the same store while keeping independent
resource lists and settings.

New Project asks for a project file and an existing or new store. Open Project
accepts `.vcproj`; old store-based projects are not converted or imported.
Create a project and attach an existing `.vcobj` to use its objects.

The Project workspace lists every attached store, including unavailable ones.
Surfaces, fibers, point collections, umbilici, and registrations each select one
store (or None) for display and editing. The initial store is selected for all
five categories. Attaching another store does not change these selections.
Read-only stores can be displayed, but objects cannot be edited in them.

Changes autosave atomically to the project file. Save As copies the description
and rebases its paths; it references the same stores and does not copy objects.
A read-only project can still display resources and edit writable stores, but
changes to its resource list or store selections require a writable project.
Viewer navigation in a read-only project remains temporary.

Detach removes a reference, never store data. Affected selections become None;
the last store reference cannot be detached. Unavailable references and their
selections survive opening. Locate / Replace reconnects such a reference.

Open Data creates a project and store automatically. Sample provenance records
which store received the catalog objects, independently of the display
selection. Reset to Open Data prepares a fresh store and restores the sample's
resource list and selections, retaining previous stores as attachments.

The version 1 document contains:

- `format: "vcproj"`, `version: 1`, and `name`.
- `stores`: entries with a stable project-local `id`, local `location`, and
  `read_only` preference. An
  entry with `"hosted": true` names a `vc_object_server` instead: its
  `location` is the service endpoint (`http://` or `https://`, e.g.
  `http://127.0.0.1:8750` through an SSH tunnel). Stores > Attach Hosted
  Store... adds one. VC3D never opens its database; the status bar shows each
  hosted store as current as of its last check (every 2 s) or disconnected,
  with saves still pending. An unreachable endpoint shows as Unreachable in
  the project's store list until the project is reopened. Command-line tools
  and the Lasagna service still need a store file.
- `selections`: `surface`, `fiber`, `point_collection`, `umbilicus`, and
  `volume_transform`, each a store reference ID or null. Registrations and their
  route choices use the `volume_transform` selection.
- `volumes`, `normal_grids`, and `lasagna_datasets`: location/tag entries.
- `view`: focus bounds, current volume, and selected Lasagna datasets.
- Optional `open_data`: catalog `sample` and populated `store` reference ID.

Store references and resources within the project directory use relative paths.
Volume resources can additionally declare `channels`, a list of named scalar
views in CZYX storage order. Manifest attachment fills this list for compound
volumes. Each channel appears separately in the volume dropdown, but the source
location, decoded cache, and annotation coordinate frame remain shared. No
channel files are generated. Older entries without this list remain unchanged.

External resources may use absolute paths. Relative local paths resolve against
the project-file directory. Remote URLs
remain URLs. Resource entries and view settings belong to the project file;
there is no author-specific resource filtering. Credentials, user identity,
cache limits, and global UI/tool preferences remain application settings.

## Lasagna data

## Project Entries

VC projects store both regular and fiber inference manifests in
`lasagna_datasets`. Each entry has the ordinary project-entry shape:

```json
{
  "location": "s3://bucket/path/predictions.lasagna.json",
  "tags": ["vc-lasagna-fiber"]
}
```

`vc-lasagna-fiber` is the reserved fiber-inference role tag. No role tag means
regular Lasagna data. `selected_lasagna_dataset` and
`selected_fiber_inference_dataset` independently select each role.

## Attachment

The VC3D File menu provides `Attach Lasagna Manifest...` and
`Attach Remote Lasagna Manifest...`. The role chooser defaults to regular
Lasagna and can mark the entry as fiber inference data. Local locations are
stored as the selected filesystem path; remote locations remain portable URLs.
A remote manifest URL can be pasted directly into the browser's path field and
opened with either Enter or the Open button.

Attachment materializes and validates the manifest, opens the referenced Zarr
descriptors, prepares the manifest's volumes, then commits the manifest entry,
derived volumes, and selected role in one project write. Failure restores the
prior in-memory and on-disk project state. Remote descriptors and chunks use
VC3D's process-active global remote cache root and per-source authentication;
chunks stay demand-loaded. Changing the configured cache root takes effect on
the next VC3D start and does not move existing cached data.
Recognized S3 locations are opened anonymously first and use credentials only
when anonymous access is denied. This allows public manifests and their Zarr
groups to open even when ambient AWS temporary credentials are stale, while
private data continues using signed requests. The chosen access mode is
runtime-only and does not change persisted manifest or volume locations.

Both remote-origin forms remain valid:

- a local manifest with an adjacent `lasagna-remote.json`, whose
  `artifact_url` is authoritative;
- a direct remote manifest URL, whose parent URL resolves relative group paths.

Absolute remote group URLs are also supported. Plain local relative groups
continue to resolve from the manifest directory.

## Channel display volumes

Separate ZYX sources appear as ordinary scalar volumes, as before. A packed
CZYX source exposes one scalar view per channel listed in its manifest. Those
views appear in the same volume dropdown, labeled `source / channel`, and are
also available for overlays. They retain one source entry rather than copying
or extracting channel files. For example:

```json
{
  "location": "s3://bucket/path/compound.zarr/sd2",
  "channels": ["presence", "nx", "ny", "width", "sheet_nx", "sheet_ny", "sheet_grad_mag"],
  "tags": [
    "vc-lasagna-manifest:s3://bucket/path/predictions.lasagna.json",
    "vc-lasagna-group:sd2"
  ]
}
```

Project reopening restores the named views directly from the volume entry.
Channel zero keeps the original volume ID; other display IDs add a channel
suffix. Annotation coordinates continue to use the original source ID, so
switching channels does not switch the coordinate frame or hide annotations.
The saved current-volume ID restores the selected channel. Channel names must
match the array's C dimension; spacing, dtype and shapes remain Zarr metadata.

All views share the opened arrays and decoded packed RAM cache. With CUDA
enabled, uint8 views upload selected scalar bricks into the common GPU cache;
other dtypes and CPU fallback use the same selected channel. No preprocessing
or additional disk cache is needed. Reattach a manifest to expose its channels
in projects saved before channel views were supported. A bare packed source
without a channel declaration continues to display channel zero.
