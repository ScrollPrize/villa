---
name: vc3d-open-data
description: Create or open local VC3D projects and work with the Open Data catalog: attach/select volumes, choose remote representations by metadata, open bounded resource subsets, preserve coordinate identity, and verify lazy reads. Load for project, volume, catalog, or remote-data workflows.
---

# VC3D local and Open Data workflows

Assume `vc3d-bridge-session` and use absolute paths visible to the VC3D
process.

## Local project workflow

1. Create a project with `vc3d_create_project(path, volume, ...)` when needed.
   This writes the `.volpkg.json` but does not open it.
2. Open it with `vc3d_open_volume`; optionally pass `volume_id`. When a local
   fixture is a directory named `*.volpkg`, resolve and pass its inner
   `*.volpkg.json` manifest rather than assuming the directory itself opens.
3. Use `vc3d_attach_volume(..., wait=true)` to add another local Zarr or remote
   Zarr URL without changing the primary volume.
4. Call `vc3d_list_attached_volumes`, then `vc3d_select_volume` explicitly.
5. Re-read `vc3d_get_state` before viewer- or segment-scoped work.

Do not overwrite a project unless the user requested it. Attaching an already
attached location is an idempotent success and preserves its existing tags.

## Open Data workflow

1. Call `vc3d_list_catalog_samples`. Refresh only when current catalog state is
   needed or no cache exists.
2. Filter samples by their returned capabilities and counts; never choose by
   array index or remembered sample name.
3. Call `vc3d_describe_catalog_sample` for candidates.
4. Select volume ids and representation `ref` values from `kind`,
   `sourceCoordinateLevel`, `artifactType`, `targetVolumeId`, `modelId`, and
   `url`. A ref is manifest-revision-local; never guess it.
5. Open only the required subset with
   `vc3d_open_catalog_sample(resources=..., wait=true)`.
6. Record the terminal job's `vpkgPath`, attached resources, volume ids, and
   messages. Then list/select the attached volume deliberately.

For cached samples, `vpkgPath` may name the projects directory rather than the
concrete manifest. If a later disposable-copy workflow needs the file, retain
the `Loaded cached sample project: ...volpkg.json` terminal message and confirm
that exact path before copying; do not invent it from the directory.

Catalog descriptor volume ids are manifest identities, not guaranteed live
VC3D selectors. After opening, use `vc3d_list_attached_volumes().volumeIds` and
the terminal result/state; attached ids may include a canonical basename/hash
suffix. Passing the raw descriptor id to `vc3d_select_volume` can therefore be
`-32007` even though the volume opened successfully.

Omitting `resources` attaches everything. Do that only when the task actually
requires the complete sample. A raw volume must pass `volumeIds`; a derived
representation must pass every supplied filter axis.

The filter limits what the current open operation attaches; it does not prune
resources already recorded in a cached sample project. On macOS the remote
project/cache remains under `~/.VC3D/remote_cache` even when a test changes
`XDG_CONFIG_HOME`. Record the terminal `attached` block and pre-existing volume
ids separately, and do not claim a clean subset from an already cached project.

## Coordinate and representation rules

- Tool coordinates are full-resolution L0 voxels unless the tool says
  otherwise.
- `sourceCoordinateLevel=L` means a published representation is sampled at
  that pyramid level. Convert a point from level L to L0 by multiplying each
  coordinate by `2**L`; verify against the selected volume's metadata.
- Select the base volume that matches the representation's provenance. Virtual
  `@L<n>` views are useful for display but are not interchangeable with the
  preferred source expected by fiber/Lasagna workflows.
- Fiber tracing and Atlas need a compatible `lasagna` representation for the
  current volume. A normal-grid store is not a substitute. The catalog does not
  expose a normalized regular/fiber role flag; when role is ambiguous, attach a
  known manifest explicitly with `vc3d_attach_lasagna_manifest(..., role=...)`
  instead of inferring it from a ref.

## Bounded remote verification

- Open Data bucket reads are anonymous and cached lazily. Do not expose or
  inject credentials.
- Prefer the smallest representation and bounded region that proves the task.
- Do not run unrestricted full prefetch during validation.
- Record sample id, representation identity, level, region, cache/download
  bytes, and any warning or unavailable resource.
- A local success does not prove the remote workflow. Preserve remote failure
  evidence and report an external blocker when retries cannot proceed.

For seed derivation, catalog nuances, and measured examples, read
[`references/catalog-and-seed-notes.md`](references/catalog-and-seed-notes.md).
