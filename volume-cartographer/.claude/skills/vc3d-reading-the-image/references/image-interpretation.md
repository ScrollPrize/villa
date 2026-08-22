# Image interpretation field guide

Preserved from the original skill because the measured observations and visual
heuristics remain useful. The concise workflow in the parent `SKILL.md` takes
precedence if guidance conflicts.

## Contents

- CT slice anatomy and fiber/sheet distinctions
- Pyramid-level limitations
- Prediction channel semantics
- Overlay alignment and coordinate spaces
- Specialized panes and evidence language

`vc3d-visual-evidence` covers how to *capture* a screenshot that proves
something. This one covers how to *read* it. They are different failures: a
technically perfect capture of a feature you have misidentified is worse than no
capture, because it looks like evidence.

Everything below with a number attached was measured on PHerc0332
(`20251211183505`, 2.399 µm) unless stated. Treat the numbers as
order-of-magnitude anchors, not thresholds to hard-code.

## 1. What is actually in a CT slice

A slice through a rolled scroll shows, in order of brightness:

| What | Grey (8-bit, window 0-255) | Looks like |
|---|---:|---|
| Air / void between sheets | ~20-40 | flat near-black |
| **Papyrus sheet** | ~120-200 | mid-grey ribbons, tens of px thick, following the roll |
| **Mineral / dense inclusion** | ~240-255 | small, saturated, irregular blobs |

**A pane with no mid-greys at all is a clipped window, not data.** If everything
is either black or white — solid white shapes with black speckles, structures
reduced to silhouettes — the intensity window has collapsed, and you are reading
a threshold rather than the scan. Observed live at
`volumeWindow {low: 10, high: 11}`: a one-unit window renders every voxel ≤10
black and ≥11 white, so papyrus, air and inclusions become indistinguishable
while the picture still looks confidently like something.

Check it before believing any grey: `vc3d_get_render_settings` reports
`volumeWindow`, and `vc3d_set_render_settings(volume_window={"low":0,"high":255})`
restores the full range. The window is session and volume state that survives
project changes, so a clipped one can arrive from anything that touched the app
earlier — including a previous automated run. Record the window beside every
capture (`vc3d-visual-evidence` §3); two greys are only comparable under the
same window.

**The single most common misreading is treating the brightest thing as the most
interesting thing.** A max-brightness search returns inclusions, essentially
every time. Measured: a 5×5-mean brightness search over a mid-Z slice returned a
pixel of luminance 247 that, zoomed in, was plainly a dense lump sitting
*between* sheets. Papyrus is mid-grey; if your candidate is saturated, it is
probably not papyrus.

## 2. Fiber, sheet, bundle — these are three different things

Get this wrong and every downstream decision is wrong.

- A **fiber** is a single plant cell: *"each fiber is surrounded by a cell wall
  enclosing a hollow lumen cavity"*
  (`../scrollprize.org/docs/37_2026_open_problems.md`). In a cross-section it is
  an **ellipse** — a brighter wall around a darker lumen. Cut obliquely, that
  ellipse stretches, and a long one reads as a short ribbon.
- A **bundle** is many fibers packed side by side. The crossing horizontal and
  vertical bundles define the sheet's U and V axes.
- A **sheet** (what a segmentation follows) is the papyrus layer — in
  cross-section the long mid-grey ribbon spiralling around the roll.

So a wide grey ribbon in a slice is a **sheet**, not a fiber; and what looks
like one ribbon can be two fibers lying close together. "Trace a fiber" means
following one cell along its length, not following the sheet.

**A fiber only looks like an ellipse where the plane cuts across it.** In an XY
slice that is the vertically-running fibers; the ones running in-plane appear as
elongated strips in the same image. Which you are seeing depends on the fiber's
orientation relative to the slice, not on the fiber. Say which case you are
looking at — the tracer's own panes make the same distinction (§7).

At 2.4 µm a single fiber is only a few pixels across, so at any pane level
coarser than about L1 you are not looking at individual fibers at all — you are
looking at bundles and sheets. Say which you mean in a write-up.

## 3. Pyramid level decides what you can honestly claim

**Read the green pane header before the picture.** It carries the level, the
scale, the pane size, the point the pane is tracking as `[X=…, Y=…, Z=…]`, and
on generated panes a fixed `POI (…)` reference that does *not* move as you pan.

`[X=-1, Y=-1, Z=-1]` in that header is the **same "no valid point" sentinel**
`vc3d_get_cursor_point` returns on a surface with no cell there
(`vc3d-segmentation-editing` §2). A pane reporting it is tracking nothing — do
not read coordinates off that image, and do not treat what it draws as located.

Opening a volume parks the axis-aligned panes at the **coarsest** level. The
green label in the top-left of every pane (`L3 XY scale 0.23`) is ground truth;
`vc3d_get_state` reports the same as `level` / `scale`, and `vc3d_zoom_viewer`
returns the `level` it landed on.

Measured on PHerc0332, `xy plane`, six doublings from the default:

| level | scale | what is legible |
|---:|---:|---|
| L5 | 0.014 | scroll outline only — a smudge |
| L3 | 0.23 | the spiral of sheets, air gaps, roll centre |
| L1 | 0.92 | individual sheet texture, inclusions distinct |
| L0 | 3.67 | CT voxels; predictions become blocks (§5) |

Any claim about fine structure made from an L5 or L4 capture is unsupported.
State level and scale next to every image.

## 4. The prediction channels, and what each should look like

A `lasagna` representation is not one array. **Read its manifest and let the
`groups` tell you what it contains** — never assume from the name:

```
~/.VC3D/remote_cache/open_data/lasagna/<sample>/…/*.lasagna.json   ->  "groups"
```

In the fiber-inference datasets seen so far the groups are:

| group | what it means | expected appearance |
|---|---|---|
| `presence` | fiber-presence field | filament paths with bright nodes along sheets |
| `nx` | fiber direction, x component | smooth signed field, sign flips across a sheet |
| `ny` | fiber direction, y component | same, orthogonal component |

Each group names its own zarr and a `scaledown`, which is what tells you how
coarse it is relative to CT. A non-fiber (normals) lasagna publishes its own
`nx`/`ny` for a different purpose; the manifest, not the filename, is what
distinguishes them.

A **surface prediction** (`kind: "prediction"`) is a different thing again: it
marks sheet surfaces, not fibers.

### Neither channel alone tells you where to put a point

`presence` answers *"does the model think a fiber is here?"* and CT answers
*"is there material here?"* — and **they disagree**, so a usable point needs
both. Measured, both failure modes in one session:

- **CT alone** → a max-brightness search returned a mineral inclusion (§1).
- **`presence` alone** → the presence maximum landed where the base volume
  reads **luminance 0 at every level**. A fiber seeded there still produced a
  474-point trace with `traceState: "predictions"` and no error.

**A valid point satisfies both: CT in the papyrus range (~120-200) *and* high
`presence`.** Capture the two at one fixed camera and intersect them in pixel
space; do not pick from one image and hope.

### 0 is not air — it is "no data"

The published base volume is usually `…-masked.zarr`: windowed CT with the
background **zeroed**. So the grey scale has three regimes, not two:

| value | meaning |
|---:|---|
| **exactly 0** | masked out, or not loaded — **no information** |
| ~20-40 | real air inside the scroll (void between sheets) |
| ~120-200 | papyrus |

Predictions are computed from an *unmasked* source, so `presence` is routinely
non-zero over regions the masked CT has zeroed. That mismatch is the trap: the
overlay looks informative, the CT under it is blank, and nothing errors.

Before using any point taken from an image, sample the base volume there and
confirm it is not 0. A black pane at a fine level while coarser levels show
structure is the same warning.

## 5. Reading an overlay without fooling yourself

Four things routinely produce a wrong conclusion. All four were hit in one
session.

**Coverage is bounded.** An inference run covers a *region*, not the volume. A
`presence` overlay over an uncovered area renders nothing, and looks exactly
like a broken overlay. Measured: the first seed chosen sat entirely outside the
inference region, and the before/after captures were byte-identical while
`vc3d_get_overlay` echoed the volume id, opacity and window back correctly. Zoom
out until you find the covered patch before concluding anything.

**Opacity scales the pixels.** With the `fire` colormap at opacity α, full
presence renders at about 255·α, not 255. At α=0.45 the brightest fiber node
measured `rgb(115,115,41)` — a threshold of `r > 200` finds nothing and reads as
"no signal". Scale any colour test by the opacity you set, or read it back from
`vc3d_get_overlay`.

**Predictions are coarser than CT.** These arrays are `scaledown: 3`, i.e. ~8×
coarser per axis. Pushing the pane to L0 renders each prediction voxel as a
large flat block — a picture of the sampling grid, not of anatomy. The legible
band is roughly pane level 2-3 for a `scaledown: 3` channel.

**`fire` maps low values to near-black**, which is also what "no data" looks
like. Distinguish them by moving to a known-covered region, not by staring
harder.

Prove an overlay rendered the way `vc3d-visual-evidence` §4 says: capture
off → on → off at a fixed camera and checksum. Identical checksums across an
"off" and an "on" means it did not render, whatever the call returned.

## 5a. Check alignment before you trust an overlay

An overlay can be *rendered* correctly and still be *registered* wrongly. Three
checks, cheapest first:

1. **Do the shapes agree?** A correct prediction follows the CT's own
   structures — filaments running along sheets, not across them; edges of
   coverage that follow anatomy rather than a straight box unrelated to it. If
   the overlay's structure ignores the sheets under it, stop.
2. **Is the prediction even for the volume you selected?** A sample can publish
   several volumes — different scans, different resolutions (µm) — and a
   prediction is computed against **one** of them. Selecting a different volume
   of the same sample pairs the prediction with data it never saw, and nothing
   refuses: the overlay still draws, in the wrong place.

   Two identity checks, both cheap:

   - the lasagna entry's `vc-open-data-volume-id:` tag must equal the **active
     volume's** `vc-open-data-volume-id:` tag;
   - the manifest's `base_shape_zyx` must equal the volume's `shapeZYX` (from
     `vc3d_describe_catalog_sample`), or exceed it by exactly 1 per axis —
     VC3D's own `validatePrepared` accepts `parent` or `parent + 1`, the padding
     a published grid gets when the parent extent is odd.

   Anything else — a factor of two, a different scan id, a mismatch in µm —
   means you have the wrong base volume selected, not a rendering problem.

   **Passing both checks does not prove registration.** A published prediction
   can satisfy every identity check and still be displaced from the CT; that has
   been observed. Check 1 is the one that catches it, so do it too.
3. **Are there cross-scan transforms in play?** A sample can publish several
   scans, and the catalog can record affine `transforms` between volumes. Those
   map *between scans*; they are not applied to make a prediction fit a volume
   it was not computed on. If you find yourself wanting one, either you have the
   wrong base volume selected, or the prediction genuinely is not registered to
   it — a real, observed condition, not only user error.

Levels are the fourth: a prediction published at `@L<n>` belongs with the
`@L<n>` view of the base volume. Mixing levels is not a rendering nuisance, it
changes what downstream tools compute — see `vc3d-fiber-tracing` §1.

## 6. Coordinate spaces are visual too

Selecting a virtual `@L<n>` source view changes the coordinate space you must
speak. On an `@L1` view the extent is half of L0, so an L0 point fails
`-32003 Point is outside volume bounds`; divide by 2ⁿ going in, multiply going
out. A point read from a capture of the `@L1` view is an **L1** point.

Plane viewers never move in Z. After `vc3d_center_viewer` the plane keeps its
own depth, so read the z back from `vc3d_get_cursor_point` rather than assuming
the z you asked for — it is the z of every point you then derive from that
image.

## 7. Panes other than the plane views

- **`segmentation` / `L0 Surface`** — the flattened active surface. Empty and
  flat grey when no segment is active, which is *not* a failure
  (`vc3d-visual-evidence` §1). A good surface shows papyrus fiber texture
  running horizontally and vertically; a surface that looks plausible here but
  crosses sheets in the cross-section views is wrong.
- **Line Annotation panes** — four of them, and they fall into two kinds that
  show the *same* fiber completely differently. Capturing the wrong kind is the
  easiest way to file misleading evidence about a trace.

  | pane | kind | what a correct trace looks like |
  |---|---|---|
  | `…_line_current_cut` | cut **across** the fiber | the fiber **intersected** — a small ellipse, with the cut centred on it |
  | `…_line_side_cut` | cut across, offset | same, from the side |
  | `…_line_surface` | **strip along** the fiber | a long horizontal band: the fiber unrolled, the centerline running its length, control points spaced along it |
  | `…_line_side_slice` | **strip along**, offset | same, and it carries the `optimized` badge |

  In the GUI the same four appear as a detail slice, a **coarse overview slice
  carrying the trace's whole winding route through the sheets** (the fastest way
  to see where a fiber goes), a **working strip** whose zoom changes constantly,
  and a **fixed-scale reference strip** that stays put so the whole fiber stays
  comparable. Control points are drawn as numbered yellow discs.

  Two header fields matter here: `[X=…]` tracks the pane's current position and
  moves as you step along the fiber, while `POI (…)` is a **fixed anchor** that
  does not move — if you are checking whether a view moved, compare the bracket,
  not the POI.

  **A strip can be rendered from different channels.** A selector in that
  workspace chooses what the strips sample — the scan, or a prediction channel
  such as `presence`. Switching it re-renders the same fiber completely
  differently, and a strip that goes black after a switch usually means the
  chosen channel has no data at that location, not that the trace is wrong. Say
  which channel a strip is showing whenever you present one.

  A fiber is an **ellipse only where the plane cuts across it** — the way a
  vertical fiber appears in an XY slice. Along its length it is a strip. So:

  - **the strip panes are the evidence that a trace follows a fiber.** They show
    whether the centerline stays on one continuous filament or wanders off it,
    and they carry the per-span trace error (`T 0.1 vx`) and the green
    `optimized` badge;
  - the cut panes only tell you the cut is centred on *something*. An ellipse
    there is expected and proves nothing about the trace as a whole.

  Capture a strip pane when the claim is "I traced a fiber". Capture a cut pane
  only when the claim is about the cross-section at one position.

  **The small labels on a strip are per-span quality numbers, and they are the
  fastest read on whether a trace is any good**
  (`spanAlignmentMetricText`, `LineAnnotationDialog.cpp:200`). Each sits between
  the two control points bounding its span, and reads
  `<mode marker> <value>` with an optional status line under it:

  | label | what it is |
  |---|---|
  | `T 0.1 vx` | **meeting error** in base voxels — the tracer fires from both bounding control points and this is how far apart the two runs were where they met. Sub-voxel is good; several voxels means that span barely joined up |
  | `12°` | max angular error over the span |
  | marker only, no value | a spline span — nothing measured |

  Colour is the other signal: white on dark is normal, **dark red on a light
  background flags the span** — always for a failed native span, and for an
  angular error over threshold. A red span is where the tracer went wrong, which
  is the thing worth reporting; a strip of sub-voxel numbers is a clean trace.

  So read a strip as: does the centerline stay on one filament, and what do the
  span numbers say between each pair of control points.
- **Fiber Slice** — switching to the workspace does not select a saved fiber;
  the current MCP contract has no direct show-saved-fiber call. Do not treat an
  empty pane as evidence that the saved fiber has no geometry.
- **Spiral** — the panel reports connection state in words; a *failed* panel
  next to a failed `vc3d_spiral_status` is legitimate agreement evidence
  (`vc3d-spiral` §8).

## 8. Say what you saw, not what you expected

When a task asks you to log your reading of a scene, write down, per capture:
the pane, its level and scale, the intensity window, any overlay and its
opacity; then what structures you identify and *how you told them apart* —
"mid-grey ribbons ~20 px thick spiralling clockwise; two saturated blobs I read
as inclusions, not papyrus, because they are ~250 and sit in the air gap".

A reader can check that. "I see papyrus" cannot be checked, and is the sentence
most often written just before a wrong seed.
