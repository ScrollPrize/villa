# VC3D Project Lasagna Data

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

Older project files may contain `fiber_inference_datasets`. The reader adds the
fiber role tag and merges those entries into `lasagna_datasets`; subsequent
writes use only the canonical collection. The old field is a project-schema
compatibility input, not a statement that the manifest data or tracer is
obsolete.

## Attachment

The VC3D File menu provides `Attach Lasagna Manifest...` and
`Attach Remote Lasagna Manifest...`. The role chooser defaults to regular
Lasagna and can mark the entry as fiber inference data. Local locations are
stored as the selected filesystem path; remote locations remain portable URLs.

Attachment materializes and validates the manifest, opens the referenced Zarr
descriptors, prepares the manifest's volumes, then commits the manifest entry,
derived volumes, and selected role in one project write. Failure restores the
prior in-memory and on-disk project state. Remote descriptors and chunks use
the project's remote cache root and authentication; chunks stay demand-loaded.

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

Derived entries use `lasagna-derived://<identity>` locations and the following
provenance tags:

- `vc-lasagna-derived`;
- `vc-lasagna-manifest:<hex identity>`;
- `vc-lasagna-group:<group name>`;
- `vc-lasagna-channel:<channel name>`;
- `vc-lasagna-spacing:<scale>`.

These tags identify ownership and exact channel identity. They let reload
rebuild runtime volume objects without treating the synthetic location as a
generic source, avoid duplicate entries, replace stale channels when a
manifest changes, and remove manifest-owned volumes during detach. A derived
volume is removed only when no other attached manifest still references its
identity.

Project load reconciliation is in-memory until the caller explicitly saves;
opening a project does not rewrite it solely because runtime derived-volume
objects were reconstructed.
