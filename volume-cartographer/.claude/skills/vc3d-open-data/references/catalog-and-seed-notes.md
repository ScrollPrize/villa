# Catalog, coordinate, and remote seed field notes

Preserved from the original skill for detailed catalog interpretation, remote
seed derivation, and measured examples. The workflow and contract corrections
in the parent `SKILL.md` take precedence.

## Contents

- Catalog discovery and representation selection
- Base-volume and overlay identity
- Remote seed derivation
- Coordinate levels, bounded reads, and segment availability

Assumes the ground rules in `vc3d-bridge-session`.

## 1. Discover, then select on fields — never on index

```
vc3d_list_catalog_samples()            -> sample ids
vc3d_describe_catalog_sample(sample)   -> volumes + representations
vc3d_open_catalog_sample(sample, resources={...})   -> a `catalog` job
```

Each representation reports `ref` (`"<volumeIdx>:<artifactIdx>"`), `kind`,
`artifactType`, `sourceCoordinateLevel`, `targetVolumeId`, `modelId`, and
`url`.

**The ref is not a name.** It indexes the volume's full artifact array
including raw source volumes, so the numbering is sparse (raw entries are
rejected as refs), and it is stable only for a given manifest revision.
Filter on the returned metadata and read the ref out of the match. A run that
hard-codes `"0:4"` is guessing.

`kind` is one of exactly three values:

- `lasagna` — a manifest-backed dataset. **Required for fiber tracing and atlas
  creation.** A sample commonly publishes two of these that differ only in
  level and role.
- `normal_grids` — a separate resource. It does **not** enable tracing; passing
  it where a lasagna dataset is needed produces `-32005` no matter what else
  you do.
- `prediction` — a prediction zarr (e.g. surface predictions).

There is no `mask` kind and nothing marks a volume as a mask.

A sample may publish several representations at different coordinate levels.
Which ones exist is a property of the current manifest, so read it from
`describe_catalog_sample` rather than carrying a remembered inventory.
`normal_grids` remains a separate store and is not a tracing dataset.

**Do not assume which samples have what.** The catalogue changes; enumerate it
at the time you need it rather than trusting any list, including one in a skill:

```python
reps = describe_catalog_sample(sample)["representations"]   # top-level, not per-volume
lasagna = [r for r in reps if r["kind"] == "lasagna"]
```

`representations` sits at the **top level** of the response, not nested under
`volumes` — reading the nested path yields an empty list and looks exactly like
"this sample publishes nothing".

The descriptor does not expose a normalized regular/fiber role flag. When a
sample has multiple Lasagna entries and the role is not unambiguous from its
authoritative manifest, do not guess from ref ordering. Use
`vc3d_attach_lasagna_manifest(location, role, ...)` when the exact manifest and
role are known, or report that the catalog metadata is insufficient.

## 2. After attaching: pick the base volume deliberately

`vc3d_list_attached_volumes` / `vc3d_list_overlay_volumes` show more volumes
than you attached, because each published level also appears as a **virtual
source view** of the same zarr with a `#vc-base-scale=N` selector. Read the
tags:

- `vc-open-data-preferred-source` — the sample's real base volume. Select this
  with `vc3d_select_volume` unless you have a reason not to.
- `vc-open-data-coordinate-space:<sample>/<volume>@L<n>` — the coordinate space.
  An overlay must share this with the base volume or the pick is rejected.
- `vc-open-data-source-coordinate-level` / `-scale-factor`, `-voxel-size-um`,
  `vc-open-data-name` (e.g. `masked.zarr [source L1, 4.798000 um]`).

A name ending `masked.zarr` is **not** a binary mask: it is windowed intensity
CT with the background masked out. It is the normal base volume.

The `#vc-base-scale` selector is project provenance only and is stripped before
any HTTP/S3 request. The virtual views are not clutter: an overlay must share
the base volume's coordinate space exactly, so to display an L1 prediction over
CT you select the `@L1` view as the base, overlay the prediction, then select
the preferred source again.

## Deriving a seed that is inside the material

A seed for tracing must be **inside the papyrus**, and the published volume is
mostly air. But do the cheap thing first.

**Try the volume centre.** On a tightly-rolled scroll the centre of the volume
is often inside material; in one measured case it traced fine on the first
attempt (301 line points, a second control point accepted,
`traceState: "predictions"`). Centring the `xy plane` viewer there, zooming off
the coarsest level, and launching with the **default `space: "volume"`** can be
the whole procedure. It is a first guess to test, not a property of any sample —
the centre is the hollow core on some rolls, and masked-out on others, so
validate it (`vc3d-fiber-tracing` §2) before building on it.

**Before blaming the seed, check the coordinate space.** The symptom of a bad
seed — `-32010 "Fiber line points have no valid sampled normals"` when adding
the second control point — is produced just as readily by passing volume
coordinates while declaring `space: "scene"`. That is a measured comparison on
one sample, one viewer, one coordinate triple:

| launch call | linePointCount | second control point |
|---|---:|---|
| `position {10289,10289,24616}` (default `space: "volume"`) | **301** | added, `traceState: "predictions"` |
| same position, `space: "scene"` | 77 | `-32010 no valid sampled normals` |

A stunted `linePointCount` and an empty `generatedSurfaces` right after launch
are the tell. See `vc3d-fiber-tracing` §2.

**Only if the centre genuinely is air** — a sample whose scroll sits off-centre,
or a launch that still fails with the space parameter correct — derive the seed
from the rendered image. The recipe, in order:

1. **Centre on the volume.** Take `shapeZYX` from
   `vc3d_describe_catalog_sample` and `vc3d_center_viewer` the `xy plane`
   viewer on `(x/2, y/2, z/2)`.
2. **Calibrate scene units.** Read `vc3d_get_cursor_point` at scene `(0,0)`,
   `(100,0)` and `(0,100)`. The volume delta per 100 scene units gives
   `vol_per_scene` for each axis. **Scene coordinates have their origin at the
   pane's top-left, not its centre** — so the point you just centred on sits at
   scene `(cx, cy)` where `cx = (centre.x − v00.x) / vol_per_scene_x`.
3. **Get the device-pixel ratio.** Capture the pane and compare:
   `dpr = image_width / (2 * cx)`. On a Retina display this is 2, and assuming
   1 puts your seed at a quarter of the offset you intended.
4. **Find the material.** Decode the PNG and take the centroid of pixels above
   a low threshold (~30), skipping the top rows that hold the green stats
   label. That centroid is inside the scroll body.
5. **Convert back.** `vc3d_get_cursor_point` at scene
   `(centroid_x / dpr, centroid_y / dpr)` returns the volume point — your seed.

Set the display window first (see `vc3d-visual-evidence` §3), or a clipped
render makes the threshold meaningless.
Keep the derived point and screenshot checksum in the workflow evidence so the
seed choice can be reproduced.

## 3. Levels compose; nothing is silently rescaled

A published manifest may describe its arrays relative to its own source view
(`source_to_base = 1`) while the catalogue places that view at L1 or L2 of the
canonical volume. Both are exact dyadic mappings to the same L0 base, and VC3D
composes `source_to_base × 2^level` exactly once to reach it
(`core/include/vc/core/util/OpenDataCoordinateIdentity.hpp`). This is why an L1
prediction can pair correctly with L2 normals.

Mismatched or ambiguous provenance must be rejected rather than silently
rescaled. Preserve the manifest identities when reporting such a failure.

You do **not** need to fetch a real foreign manifest to demonstrate it. A
locally written manifest carrying another sample's coordinate identity is
rejected on the same path, which makes the check a few lines of local JSON
rather than an S3 prefix listing. Say which you used — a synthetic manifest
proves the guard fires, not that any particular published artifact is
mismatched.

## 4. Remote reads are anonymous and bounded

Reads from `vesuvius-challenge-open-data` are forced anonymous even when the
process holds stale AWS credentials; other buckets keep the credential chain.
No RPC accepts or returns credentials. Fetched objects land in the project's
bounded exact-byte cache under `~/.VC3D/remote_cache/`; a second open of the
same region reads from cache without growing it.

Prefer the smallest suitable representation and bounded lazy reads. Do not
turn a catalog-open validation into an unrestricted full-volume download.

## 5. Opening a sample gives you no segments

`vc3d_open_catalog_sample` attaches volumes and derived representations. Some
samples publish **zero segments** — `describe_catalog_sample` reports
`segmentCount`, so check it rather than assuming. With none there is no surface
to activate and the main workspace's `L0 Surface` pane stays empty.
That is correct behavior, not a failure — see `vc3d-visual-evidence` before
capturing that pane as evidence of anything.

Samples that do publish segments attach them as placeholders;
`vc3d_fetch_segment` materializes one and `vc3d_activate_segment` makes it the
`segmentation` surface.
