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
  `read_only` preference. Hosted replicas remain local store references.
- `selections`: `surface`, `fiber`, `point_collection`, `umbilicus`, and
  `volume_transform`, each a store reference ID or null. Registrations and their
  route choices use the `volume_transform` selection.
- `volumes`, `normal_grids`, and `lasagna_datasets`: location/tag entries.
- `view`: focus bounds, current volume, and selected Lasagna datasets.
- Optional `open_data`: catalog `sample` and populated `store` reference ID.

Store references and resources within the project directory use relative paths.
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

## Ordinary 3D Volumes

Every manifest group is exposed in the project as one ordinary 3D volume. The
group must name exactly one channel and reference a ZYX array. VC3D does not
support channel-first CZYX Lasagna arrays or acquire generic 4D volume handling.

The original Lasagna preprocessing and fitting pipeline used flat CZYX arrays
as an intermediate representation before conversion to separate per-channel
3D OME-Zarr volumes. Those older Lasagna intermediates are not VC3D project
volumes and must be converted before their manifest is attached.

Derived entries use the resolved Zarr source directly as their location. For
example, a relative group in the manifest above is stored as:

```json
{
  "location": "s3://bucket/path/presence.zarr",
  "tags": [
    "vc-lasagna-derived:s3://bucket/path/predictions.lasagna.json"
  ]
}
```

The path-bearing `vc-lasagna-derived:<manifest location>` tag records ownership
only. Group, channel, spacing, dtype, and shape are read from the authoritative
manifest and Zarr descriptor and are not duplicated in project tags. Reload
uses the ownership tag to rebuild the exact-byte Lasagna runtime volume rather
than opening a remote group through the generic volume cache. Repeated
references to the same source share one project volume and add one ownership
tag per manifest. Detach removes one ownership tag and only removes the volume
after the last owner is gone. If the source was already an independent primary
project volume, attachment reuses it without adding an ownership tag, so
manifest detach cannot remove it.

Project load reconciliation is in-memory until the caller explicitly saves;
opening a project does not rewrite it solely because runtime derived-volume
objects were reconstructed.
