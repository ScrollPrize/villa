---
name: vc3d-segment-lifecycle
description: Managing segments in a VC3D project through the bridge — attaching sources, listing, fetching Open Data placeholders, activating, renaming, deleting, generating masks, cropping bounds, recalculating area, and the review tags and filters. Load before any vc3d_attach_segments, vc3d_list_segments, vc3d_fetch_segment, vc3d_activate_segment, vc3d_*_segment_mask, vc3d_crop_segment_bounds, vc3d_recalc_segment_area, vc3d_set_segment_tag or vc3d_review_segments call.
---

Assumes `vc3d-bridge-session`. Editing a segment's geometry is
`vc3d-segmentation-editing`; what has no RPC at all is
`vc3d-capability-boundary` (13 of the 28 segment context-menu operations,
including merge and patch).

## 1. The order

```
vc3d_attach_segments(location=…)   # register a source directory
vc3d_list_segments()               # discover ids
vc3d_fetch_segment(id)             # ONLY for Open Data placeholders (async)
vc3d_activate_segment(id)          # make it the editing target
```

`vc3d_activate_segment(auto_fetch=True)` does the fetch-then-activate in one
call, which is what you usually want on a catalog project.

`vc3d_list_segments` reports `{id, path, loaded, active}`. `loaded` means the
surface is in memory, not that it exists — a catalog placeholder is listed but
not materialized until fetched.

## 2. Attach

`vc3d_attach_segments` takes one **absolute** path, identifying either a segment
or a directory of segments, and is idempotent by normalized location.

- A source whose **directory name** is already used case-insensitively by
  another attachment is rejected `-32010` — VC3D's source picker identifies
  entries by that name.
- It is refused `-32004` while editing is enabled, but only when it would
  actually reload surfaces (a genuinely new entry, or `select` changing the
  current source).
- If the attach commits but the UI refresh throws, the attach is **not** rolled
  back and `vc3d_list_segments` may not show the new ids yet.

## 3. Fetch and activate

`vc3d_fetch_segment` is a `source:"catalog"` job and shares that slot with
`vc3d_open_catalog_sample` — a concurrent catalog operation is rejected up
front. **Catalog jobs are not cancellable.** An already-materialized segment
returns synchronously with `alreadyMaterialized: true`.

`vc3d_activate_segment` on the already-active id is an inert no-op returning
`alreadyActive: true`. While a growth job runs the panel is selection-locked and
activation fails `-32004`. If the surface then fails to load, activation is
reported `-32005` even though the underlying call succeeded — so read
`activeSurface.id` back from `vc3d_get_state`.

Remember activation **re-centres the viewers**, invalidating any canvas point
you captured earlier.

## 4. Masks — and the order that makes area work

```
vc3d_generate_segment_mask(id)     # writes <segment>/mask.tif
vc3d_recalc_segment_area([id])     # needs that mask
```

**`vc3d_recalc_segment_area` silently fails without a mask.** With no `mask.tif`
it returns overall success while that segment's entry carries
`success: false`, `areaVx2: 0`, `errorReason: "no mask.tif"`. Read the
per-segment entries, not the call's return. On success it writes `area_vx2`,
`area_cm2` and `date_last_modified` into the segment's `meta.json`.

**`vc3d_generate_segment_mask` is a no-op when `mask.tif` already exists**,
returning `{"generated": false, "alreadyExists": true}`. There is no force flag;
regenerating means deleting the file outside the bridge.
`vc3d_append_segment_mask` adds the current volume to it as extra pages.

The mask render **stamps the current volume's coordinate identity onto the
segment**, so select the intended volume first or you write a wrong identity
into `meta.json`.

Mask generation and `vc3d_crop_segment_bounds` are mutually exclusive
(`-32004`): both mutate the same in-memory surface.

Both mask calls are **deferred replies, not jobs** — there is nothing to poll,
the call simply takes up to ~2 minutes to return.

## 5. Crop, reoptimize, refine

**`vc3d_crop_segment_bounds` writes the surface to disk in place** and returns
`cropped: true` even when the bounds were already tightest. It can also **delete
`generations.tif`** when the cropped dimensions no longer divide the grid —
which matters because some growth methods require that file. Treat it as
destructive.

`vc3d_reoptimize_segment` and `vc3d_refine_segment_alpha_comp` take the shared
`tool` job slot, so they contend with rendering and `vc3d_run_trace`
(`-32004`). Both accept `param_overrides`.

## 6. Rename and delete

Both are refused `-32004` while editing is enabled. `vc3d_delete_segment`
requires `confirm=True` (`-32602` otherwise) and deletes on disk.
`vc3d_rename_segment` renames the directory and rewrites the meta UUID, with
rollback on failure — but a partial failure leaves the segment **unloaded**.
There is no surface-reload RPC, so the way back is to reopen the project with
`vc3d_open_volume` on the same path, which rebuilds the segment list from disk.
Do that before reporting the segment lost; the directory itself is intact.

## 7. Review tags

`vc3d_set_segment_tag(segment_id, tag, enabled)` accepts exactly
`approved`, `defective`, `reviewed`, `inspect`. Side effects worth knowing: it
changes the surface-panel selection, and it writes all four tag states plus a
username into `meta.json` immediately. A disabled checkbox yields
`-32010 Tag could not be set`.

`partial_review` is **read-only** — `vc3d_review_segments` reports and filters on
it, but no RPC sets it.

`vc3d_review_segments` is the programmatic equivalent of the panel's review
filters, ANDed together, returning `{segments:[{id, path, loaded, active, tags,
reviewState}], total, returned}`. Its precedence is not obvious: `defective`
outranks `approved` (a defect needs attention regardless of a stale approval),
and `partial_review` ranks below `reviewed`. It deliberately re-reads each
`meta.json` rather than trusting the copy parsed at project-open, so it can
legitimately disagree with anything you cached — trust `review` over `list` for
tag state.

## 8. Flattening produces a new segment — sometimes invisibly

Use `vc3d-flattening` for operation choice, job lifecycle, geometry checks,
attachment, and render verification. The edge cases below remain important.

`vc3d_flatten_slim`, `vc3d_flatten_abf` and `vc3d_flatten_straighten` write a
**new segment directory** and share the `flatten` job slot; none is cancellable.

- **SLIM's output does not appear in `vc3d_list_segments`.** The other two
  refresh the surface panel on success; SLIM does not. The directory is on disk
  and the job says `succeeded`, but the package does not know about it. Check
  the returned `outputDir` on the filesystem instead of concluding it failed.
- SLIM writes intermediates **into the source segment directory**, and if the
  output directory equals the source — which happens when the segment already
  ends in `_flatboi`, or when you pass one — it **rebuilds in place**.
- SLIM needs external executables (`flatboi`, `vc_tifxyz2obj`,
  `vc_obj2tifxyz`, plus `vc_obj_uv_lift` when `keep_percent < 100`);
  a missing one is `-32006`.
- ABF has no `output_dir` at all — always `<segment>_abf`, in place if the
  segment already ends that way. It is the only one that does not need a
  current volume.
- Straighten refuses an existing output directory; the second straighten of the
  same segment silently becomes `_v2`, and the third fails.

Relative `output_dir` values are rooted at the volpkg.

## 9. Verify

`vc3d_list_segments` for existence and active state, `vc3d_review_segments` for
tags, the returned `maskPath`/`alreadyExists` for masks, the per-segment
`success` entries for area, `outputDir` on disk for flatten. Leave the fixture
as you found it — revert tags you set, and say what you created.
