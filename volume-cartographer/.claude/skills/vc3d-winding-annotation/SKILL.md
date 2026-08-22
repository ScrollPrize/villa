---
name: vc3d-winding-annotation
description: Creating winding constraints in VC3D — point collections with same/relative/absolute winding annotations, the same-wrap annotation tool, and the save/load paths that feed the spiral fit. Covers what a winding annotation means, the destructive global mutators, and why several of these calls return success while doing nothing. Load before any vc3d_*_point*, vc3d_*_winding* or vc3d_*_wrap_annotation call.
---

Assumes `vc3d-bridge-session`. Limits are in `vc3d-capability-boundary`.

## 1. What this is for

The project's stated priority: *"we believe that the fastest way to unroll
scrolls at scale is to develop methods for creating winding constraints that are
precise and fast enough to use widely"*
(`scrollprize.org/docs/39_winding_annotations.md`). Winding constraints are
sparse geometric evidence that the spiral fit turns into a whole-scroll model,
and they are much cheaper than full segmentation.

Three kinds, classified by the information they carry, not by the tool that made
them:

- **Same-winding** — these points all lie on the same wrap. A traced fiber, a
  line along the surface, or a verified patch each provides one.
- **Relative-winding** — these observations are *N* complete wraps apart. Points
  placed radially *across* sheets. Singled out by the project as the highest
  impact and the least automated.
- **Absolute-winding** — this observation is on numbered winding *k*. Anchors
  the global numbering.

A grown patch is already a same-winding annotation. For a same-winding point
collection, start *inside* an existing patch and extend outward — that is what
links the two.

Point collections are **session state, not project state**. Nothing is on disk
until you save, and opening any project or catalog sample silently clears the
entire store.

## 2. Collections and points

`vc3d_commit_points(collection=…, points=[…], winding=…)` creates the
collection if it does not exist — `vc3d_add_point_collection` first is optional.
Points are volume-space; the winding annotation is a float, and unset reads back
as `null`.

Selectors take **either** `collection` (name) or `collection_id`; the numeric id
wins if both are given. Names are **not unique** — a rename can create a
duplicate, after which name lookup resolves to an arbitrary one of them. Prefer
ids once you have them.

`vc3d_add_point_collection` with an existing name is a silent no-op that returns
the *existing* id, which may already hold points.

`vc3d_list_points` is the read-back: `{id, name, color, points:[{id, position,
winding}]}`. Note what it does **not** report — `absoluteWindingNumber`, tags,
auto-fill mode, `anchor2d`, linked collections. Those setters echo their input
and have no getter, so they are effectively write-only over the bridge.

For ordinary point CRUD, retain the numeric ids returned by the list/commit
calls:

1. `vc3d_update_point` changes one point's position and/or winding; re-list it
   immediately because coordinates are the evidence, not the success flag.
2. `vc3d_remove_point` removes exactly one point id. Use it instead of clearing
   a collection when correcting an isolated mistake.
3. `vc3d_rename_point_collection`, `vc3d_set_point_collection_color`, and
   `vc3d_set_point_collection_metadata` mutate collection presentation or
   metadata. Color is exactly three numeric components (RGB), not RGBA. Avoid
   duplicate names even though the bridge permits them.
4. Collection tags use separate add/remove tools. Winding links are also
   write-only; save an evidence record of every requested link or tag.
5. Save the whole store before bulk clear, reset, or load operations.

## 3. The destructive ones

These have no selector and no confirmation, and all return a cheerful `true`:

| call | actually does |
|---|---|
| `vc3d_clear_all_points` | deletes **every** collection |
| `vc3d_reset_windings` | clears windings across **all** collections, *and* resets each one's absolute-winding flag and auto-fill mode |
| `vc3d_apply_anchor_offset` | shifts the 2D anchor of **every** collection |
| `vc3d_clear_point_collection` | deletes the collection itself, not just its points — the id dies, and committing to the same *name* later mints a new one |

`vc3d_load_points_json` **clears the whole store before it validates the file**.
A version mismatch therefore leaves you with zero collections and
`{"loaded": false}`. Save first.

## 4. Auto-fill has two different algorithms

`vc3d_auto_fill_windings(mode=…)` **overwrites every point** in the collection,
sorted by point id: `incremental` counts up from 1, `decremental` counts down
from the point count.

`vc3d_set_auto_fill_mode` sets the mode for *future* points, and that path
computes max+1 / min−1 over the existing annotations instead.

So setting a mode and committing points does **not** produce the same numbers as
calling auto-fill afterwards. Pick one and stay with it.

## 5. The two save paths are not interchangeable

- **`vc3d_save_points_json` / `vc3d_load_points_json`** — the whole store, and
  what you want in almost every case.
- **`vc3d_save_points_segment_path` / `vc3d_load_points_segment_path`** — only
  collections carrying a 2D `anchor2d`, written as a segment's
  `corrections.json`.

**No bridge call can set `anchor2d`.** It comes from a GUI "convert point to
anchor" action, from an interactive correction *drag* (see
`vc3d-segmentation-editing` and its surface-growth reference), or from growth
restore. So in a bridge-only
session `vc3d_save_points_segment_path` has nothing to write — and when nothing
qualifies it **deletes any existing `corrections.json`** and returns
`{"saved": true}`. Likewise `vc3d_load_points_segment_path` clears every
anchored collection first and returns `{"loaded": true}` even when no file
exists. Do not call either unless you know anchored collections exist.

`vc3d_apply_anchor_offset` is inert for the same reason.

## 6. Coordinate identity travels with the volume, not the points

Selecting a different volume re-stamps the point store's file metadata with the
new volume's coordinate identity — **the points themselves are not
transformed**. Commit against volume A, select volume B, save, and you have
written A-space coordinates labelled as B. Commit, save, *then* change volume.

## 7. Committed points do not automatically feed corrections growth

The corrections solver snapshots its collection list when the corrections state
is built or a collection is explicitly made active. A collection created
mid-session by `vc3d_commit_points` is not in that list, so growth ignores it —
while the UI may still report that corrections are available. For correction
points that actually influence growth, use the corrections drag path in
`vc3d-segmentation-editing`.

## 8. The same-wrap annotation tool

```
vc3d_set_wrap_annotation_mode(active=True)
vc3d_shift_click(viewer=<a plane viewer>, position=…)   # one or more
vc3d_commit_wrap_annotation()
```

Each commit creates a **new** collection named `same_wrap<N>` with the absolute
flag false; that is the read-back. `vc3d_undo_wrap_annotation` pops an entire
committed collection, not a point, and is never an error.

Traps:

- **The preview is computed from the rendered image, not the volume.** Zoom,
  pyramid level and window/level all change the result. On a pane still at the
  coarsest level it produces nothing useful — and `vc3d_shift_click` returns
  `clicked: true` regardless, so the failure is silent.

  **Zoom in first. `vc3d_zoom_viewer` does this** — it is not missing, and a run
  that concludes the preview cannot be seeded because nothing can change the
  zoom has simply not looked. It takes `factor` and `viewer`, and returns the
  `level` it landed on, so you can drive it until the level is fine enough:

  ```
  center_viewer(viewer=plane_id, point=<a point with data>)
  zoom_viewer(viewer=plane_id, factor=2.0)   # repeat, watching `level`
  ```

  Measured on the local Scroll 1 fixture, `xy plane` centred on the curated
  seed with the **raw** volume selected — six doublings walk it
  `level 5 → 0` (`scale 0.0469 → 3`), and at that point the *first*
  `vc3d_shift_click` sets `hasPreview: true`. Three shift-clicks then committed
  a `same_wrap1` collection of **55 points**. At level 5 the same sequence
  yields `hasPreview: false` and `{"committed": false, "hadPreview": false}`.

  Select a volume where wraps are actually visible: this is a CT-intensity
  measurement, so the raw volume works and a surface-prediction volume is the
  wrong input.
- **Pass an explicit plane `viewer`.** Without one the click lands on the
  `segmentation` pane, where wraps are not visible as concentric bands. A commit
  without a viewer scans all panes and takes the first that succeeds, which is
  nondeterministic when several hold previews.
- **`hasPreview: false` has two causes, and only one is fatal.** Check
  `vc3d_get_state().sameWrapAnnotation.hasPreview` after your first shift-click,
  and read it like this:
  - **Pane still coarse** — by far the common case, and fixable: zoom in as
    above and click again. Do this *first*, before suspecting anything else.
  - **A human left the panel's path type on `Manual`** — then shift-click does
    nothing, permanently, and there is no RPC to change the path type. The
    symptom is identical: `vc3d_commit_wrap_annotation` returns
    `{"committed": false, "hadPreview": false}` with no error.

  Only conclude the second after the pane is at a fine level (`level` 0-1 from
  `vc3d_zoom_viewer`) and a click there still leaves `hasPreview` false. That
  distinction is the difference between a two-call fix and a genuine
  human-only blocker; reporting the blocker without ruling out the zoom is the
  mistake this note exists to prevent.
- Consecutive shift-clicks **extend** the same preview; the bridge cannot start
  a fresh one except via `vc3d_undo_wrap_annotation`.
- Turning the mode off destroys an uncommitted preview.

The *relative*-winding annotation family in that panel has no bridge
equivalent — see `vc3d-capability-boundary`.

## 9. Verify, then say what you built

`vc3d_get_state().sameWrapAnnotation` → `{enabled, hasPreview}` is the only
confirmation that a shift-click seeded anything. `vc3d_list_points` is ground
truth for everything else — never a screenshot.

When reporting, say which kind of constraint you produced (same / relative /
absolute), how many points, on which volume and coordinate space, and where you
saved it. A collection whose kind is ambiguous is not usable evidence for the
fit.
