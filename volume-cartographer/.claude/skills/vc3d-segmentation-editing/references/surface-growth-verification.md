# Surface growth and editing field notes

Preserved from the original skill for detailed surface-point derivation,
growth verification, and editing edge cases. The concise workflow in the
parent `SKILL.md` takes precedence if guidance conflicts.

## Contents

- Editing preconditions and coordinate traps
- Surface-point derivation and growth verification
- Manual-add and correction workflows
- Push/pull, saving, and cleanup

Assumes `vc3d-bridge-session`. What is *not* reachable is in
`vc3d-capability-boundary` — read that too, because this domain has more
GUI-only operations than reachable ones (no undo, no Apply/Reset, no approval
mask, no brush configuration, no smoothing, no direction fields).

This is what VC3D is for. A *segment* (also patch, surface, trace) is a tifxyz
grid of 3D points sampling one sheet of papyrus. Growing extends it; correcting
fixes where it drifted onto the wrong sheet.

## 1. The chain, and it is mandatory

```
vc3d_activate_segment(id)      →  sets the "segmentation" surface slot
vc3d_enable_editing(True)      →  refuses -32007 with no active surface
vc3d_get_state()               →  confirm segmentationEditingEnabled
… growth / manual add / corrections / push-pull …
```

Enabling editing **makes a copy of the base surface** — edits are not applied to
the original until saved. Run activation *before* any editing call: without an
active surface every editing RPC only ever exercises its guard path, which looks
like the calls "working" if you only read return codes.

**An Open Data catalog segment cannot be edited in place.** It is an immutable
cache, and `vc3d_enable_editing` refuses `-32009` with `data.detail` naming the
editable-copy path. Make the copy in the GUI, activate *that*, and enable
editing on it.

## 2. Two coordinate traps that cost whole sessions

- **Plane viewers never move in Z.** `vc3d_center_viewer` pans x/y only; the
  plane's depth stays put. Reusing a curated seed's `z` therefore fails
  `-32003 point is not on this viewer's view`. On a *plane* viewer, call
  `vc3d_get_cursor_point` first and overwrite `z` with what it returns.
- **On the `segmentation` viewer, do the opposite.** That pane shows the
  surface, not a plane, and `vc3d_get_cursor_point` there returns
  `{-1, -1, -1}` — the tifxyz "missing cell" sentinel — wherever the grid has
  no valid cell, *without* raising an error. Treat that value as "no point",
  never as a coordinate. Click the surface with `space:"volume"` at a 3D point
  you already know is on it (from a plane viewer, or a segment's bbox); that
  path resolves through the surface itself and works where the cursor query
  does not.
- **Pick a segment with valid geometry where you intend to work.** A large
  patch can have a hole at its own bbox centre, so the obvious "centre of the
  biggest segment" is a poor first guess. If clicks keep landing on invalid
  cells, choose a different, smaller segment rather than fighting the point.
- **Activating a segment re-centres the viewers.** Any canvas point you
  captured before activation is likely no longer on the view. Re-fetch it after
  activating, or your first click throws `-32003`.

### Deriving a point that is actually on the surface

This is the step that decides whether the whole session works, so do it first
and do it properly. The curated seeds in `manual_bridge_support.py` are volume
points for *plane* viewers; none of them is a segmentation-pane coordinate, and
feeding one to `vc3d_click` on a plane viewer fails `-32003` on Z.

The segmentation pane draws the active surface at whatever pyramid level it is
on — commonly the coarsest, `level 5` of 6 — so the grid covers a small part of
the pane and most scene coordinates map to no cell. **Probing one point and
concluding the surface is unreachable is the most common way this workflow
dies.** Sweep instead. `vc3d_get_cursor_point` is cheap, never raises on this
viewer, and returns the `{-1,-1,-1}` sentinel exactly where there is no cell:

```python
hits = []
for sx in range(-400, 401, 100):
    for sy in range(-400, 401, 100):
        p = get_cursor_point(viewer=seg_id, scene={"x": sx, "y": sy})["volumePoint"]
        if (p["x"], p["y"], p["z"]) != (-1.0, -1.0, -1.0):
            hits.append(((sx, sy), p))
```

Measured on the local Scroll 1 fixture with
`auto_grown_20260416135719054_inp_hr` active, that 9×9 sweep returns **3 valid
cells out of 81** — all at scene `y=400`, `x` in 200…400. That hit rate is
normal, not a sign anything is wrong.

Then click a hit with `space:"scene"`. **The click echoes back a `volumePoint`,
and that echo is your confirmation it resolved to a real cell** — a click that
lands on no cell returns `{-1,-1,-1}` there while still reporting
`clicked: true`. With corrections point mode on, scene `(300, 400)` gives:

```
click      -> volumePoint {x: 4648.375, y: 4817.909, z: 14549.016}
points.list -> collection "correction1", point 1 at the same coordinates
```

If a sweep comes back empty, widen the range or halve the step before
concluding the segment has no usable geometry — and consider a different, smaller
segment (see the bullet above).

## 3. Growth

```
vc3d_grow_segment(steps=…, direction=…, method=…, inpaint_only=…)
```

`direction` is exactly `all` | `up` | `down` | `left` | `right` | `fill`
(default `all`); `steps` is required and ≥ 1. **`method` has no enum** — it is a
free string defaulting to `tracer`, so a typo is not rejected at the schema and
you will not find a list to choose from.

**Directions are relative to the flattened 2D segmentation pane**, not to
volume axes: left / up / down / right / all mean the edges of that window;
`fill` works inward on holes rather than outward on edges. A step is roughly 20
voxels. The GUI's guidance is 10-30 steps at a time, then inspect and fix, then
repeat — not one large run.

Growth wants a **surface-prediction volume** selected, not the raw CT.

**`vc3d_grow_segment` always reports success.** The job is closed on the
running→idle edge with success hard-coded, so `state: "succeeded"` means
"growth stopped", not "growth grew anything".

### Proving growth actually grew something

`vc3d_recalc_segment_area` is the obvious route and it is usually **blocked**:
it needs a `mask.tif` most segments do not have, and returns
`success: false, errorReason: "no mask.tif"` for that segment while the call as
a whole reports success (see `vc3d-segment-lifecycle` §4). Do not conclude from
that refusal that growth is unverifiable — it is not, and three cheaper routes
need no mask at all.

Once the surface has been written — an explicit `vc3d_save_segment`, or the
autosave that `vc3d_enable_editing(False)` performs — read the segment directory
directly:

| where | what changes |
|---|---|
| `meta.json` → `area_vx2` | recomputed on write, **independently of `recalc_area`** |
| `x.tif` / `y.tif` / `z.tif` dimensions | the grid itself gets bigger |
| `meta.json` → `max_gen` | the generation counter advances |

Measured on the local Scroll 1 fixture, `auto_grown_20260416135719054_inp_hr`,
`grow(steps=3, direction="all", method="tracer")`:

```
grid (x.tif)  1225 x 645  ->  1265 x 685      (+40 per dimension, +9.7% cells)
area_vx2      230676359.9 ->  244270266.8     (+5.9%)
max_gen       6912        ->  6920
```

TIFF dimensions come out of `tiffinfo x.tif` or a 20-line header parse; file
size works too, since the grid is dense 32-bit float (`bytes ≈ 4·w·h + header`).

Snapshot `meta.json` and the tif sizes **before** growing — after the fact there
is nothing to compare against. `autosave.pending` in `vc3d_get_state` tells you a
write is queued, but only the on-disk numbers tell you the surface changed.

**Growth is not cancellable** — `vc3d_cancel_job` refuses `source:"growth"`
with `-32010`. Size the run so you are willing to wait for it.

A quality segment follows the sheet in the cross-section views *and* shows
horizontal and vertical fibers in the flattened view. A segment that looks
plausible flattened but crosses sheets in cross-section is wrong.

## 4. Manual add — the documented order is incomplete

`vc3d_manual_add_begin` fills a hole. The tool docstring says to place plane
constraints with `vc3d_shift_click`; on its own that does nothing, because
there are no candidate vertices to snap a constraint to yet.

The real order:

1. `vc3d_manual_add_begin()`
2. **A plain, unmodified `vc3d_click` on the `segmentation` pane** — this
   defines the fill region. Skipping it makes every later shift-click a no-op
   that still returns `clicked: true`.
3. `vc3d_shift_click(viewer=<a plane viewer>, …)` to add plane constraints;
   shift+right removes the nearest. These must be on a **plane** viewer —
   shift-clicking without an explicit `viewer` goes to the segmentation pane,
   where the constraint path is skipped entirely.
4. `vc3d_manual_add_finish(apply=True)`

**`finish(apply=True)` can return `{"applied": false}` and leave the mode on.**
When the interpolation path finds nothing to commit it returns before clearing
the mode. Recovery is `vc3d_manual_add_finish(apply=False)`, which always
closes it. **Always re-read `vc3d_get_state().manualAddMode` after finishing**
— a stuck mode makes every later growth or push/pull call behave strangely,
because the panel's growth method is still ManualAdd.

Two more side effects: `begin` tears down push/pull, drags, annotate mode and
approval-mask editing, and swaps the panel's growth method (restored on
finish). And `vc3d_enable_editing(False)` discards uncommitted manual-add work
by restoring the entry snapshot.

`vc3d_manual_add_set_line_mode` and `set_interpolation` **persist to the real
`~/.VC3D/VC3D.ini`**, changing the panel default for every future VC3D launch
by anyone on that machine. Read the current values from `vc3d_get_state`
(`manualAddLineMode`, `manualAddInterpolation`), and restore what you changed.

## 5. Correction points — click versus drag is the whole story

`vc3d_corrections_set_point_mode(active=True)` then click on the
**`segmentation` viewer**. A plane viewer's slice may not sit on the surface
grid at all, and the click is silently dropped.

- **Click, or a drag ≤ 1 voxel** → a plain, un-anchored correction point.
- **Drag > 1 voxel** → an *anchored* point, and it **immediately starts a
  `source:"growth"` job**. Poll `vc3d_job_status` to a terminal state before
  issuing any further editing RPC, or you race it.

The drag delta is genuinely sensitive to local surface curvature: too small and
it counts as a click, too large and it leaves `vc3d_drag`'s 2-voxel round-trip
tolerance. Expect to retry.

**Correction points do nothing on their own.** They assert "the surface should
have gone through here"; the next `vc3d_grow_segment` is what acts on them.
Placing them is half the workflow.

The mode is **sticky** — it is not cleared on mouse release. Turn it off with
`active=False` when done. Points land in an auto-created collection named
`correction<N>`; find them there with `vc3d_list_points`.

**Verify placement with `vc3d_list_points`, never with a screenshot.** A
genuinely committed correction point can render byte-identical window captures
before and after.

## 6. Push/pull is a running timer, not a one-shot

`vc3d_push_pull_start` acts at the **last hovered position**, not one you pass,
and it keeps deforming until stopped.

```
vc3d_drag(viewer="segmentation", button="none", …)   # hover to position
vc3d_push_pull_start(direction="push"|"pull")        # those two, nothing else
… brief wait …
vc3d_push_pull_stop()
```

`direction` is required and is exactly `push` or `pull` — not `up`/`down`, which
is the intuitive guess and is rejected. The optional `alpha` boolean switches to
alpha-compositing mode.

Forgetting the stop leaves VC3D permanently deforming the surface, and
`vc3d_get_state` exposes **no push/pull field**, so there is no way to detect a
leaked one except by calling stop. Call it even when unsure.

The hover target must be a segmentation-family viewer (`segmentation`,
`seg xz`, `seg yz`, `xy plane`); a fiber or atlas pane does not qualify.
`start` returns `{"active": false}` — **not an error** — for a zero direction,
no hover target, a wrong viewer, or no session.

`vc3d_push_pull_set_config` returns the *sanitized effective* config, which may
differ from what you sent. It is the only read-back; there is no getter.

## 7. Saving

`vc3d_save_segment` returns `{jobId: null, state: "idle"}` in two different
situations: nothing was dirty, **and** the save could not start at all. Check
`dirtyAfterSave` and `pending` rather than treating "idle" as "saved".

`vc3d_enable_editing(False)` performs a pending autosave — turning editing off
is itself a write.

## 8. Producing a new segment rather than editing one

`vc3d_grow_patch_from_seed(seed, volume_id, …)` grows a fresh patch from a
point and **does** refresh the panel, so the new id appears in
`vc3d_list_segments`. `vc3d_run_trace` continues an existing trace and needs
`trace_params.json` in the volpkg root (`-32007 kind:"file"` otherwise); it
refuses remote volumes with `-32009`. Both take the shared `tool` job slot, so
they contend with rendering and reoptimization (`-32004`).

## 9. Leave it clean

Turn off corrections point mode, stop any push/pull, close any manual-add
session, and either save or explicitly abandon your edits — and say which in
your write-up. See `vc3d-bridge-session` §9.
