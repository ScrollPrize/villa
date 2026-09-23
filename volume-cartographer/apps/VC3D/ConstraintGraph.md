# Constraint Graph workspace

Open **Constraint Graph** in VC3D with a CT volume loaded. Select a fiber JSON
folder, a patch tifxyz folder, and the corresponding Lasagna normals manifest.
Set the coordinate scales before **Load folders / New graph**. The largest
linked fiber component is selected first; the component dropdown can start a
new graph from another component.

Fibers are trimmed to their first and last controls. Their sheet strips are
seven-row ribbons at 20-voxel columns built by marching outward from each
fiber sample along the in-sheet cross direction while re-reading the Lasagna
normal field, re-centring on the `pred_dt` sheet score when the manifest has
one, with invalid, folded or disagreeing normals interpolated from their
neighbours (see *Automatic linking* below for the measurements). This
replaced the [v7 robust normal fit](RobustLineStrips.md) used by Line
Annotation, whose regularization pulled the outer rows off the sheet and cost
seconds to minutes per fiber. Saved strips of any other generation regenerate
on opening a full graph or first selecting a review fiber, preserving patch
placements and review decisions.
The output is a dynamically sized, strict-quad tifxyz grid.
Missing support remains invalid rather than connecting unrelated boundaries.

## Reviewing additions

### Fiber review mode

Choose **Fiber review** in the toolbar to review patch associations one fiber at
a time. Select the fiber and patch folders and the usual CT volume, Lasagna
normals, and coordinate scales, then **Load folders / New graph**. The input
panel collapses after loading; **Graph inputs** in the toolbar reopens it.
The left list shows each fiber and its number of accepted patches.
The sidebar's **Fiber line opacity** and **Fiber line thickness** control the
center line and contact highlights (defaults: 35%, 1 screen pixel). Set opacity
to 0% to hide them. These settings are saved in the review workspace.

Committed context patches have a tint in their boundary color, even when
boundaries are hidden. **Committed tint** controls its strength from 0–100%
(default 4%; 0% disables it) and is saved with the workspace. The tint follows **Committed opacity**
and leaves holes untouched. The active proposal is untinted; reopening an
accepted patch removes its context tint while it is being reviewed.

Selecting a fiber frames its first proposal. Advancing, committing, rejecting,
or reopening a proposal preserves the current zoom and pan. Use **Fit view**
to explicitly reset the view.

The selected fiber has a fixed, straight 2D ribbon. Proposals are ordered from
left to right for horizontal fibers and top to bottom for vertical fibers;
**Reverse order** reverses traversal. Discovery finishes before review begins,
so worker completion order cannot reorder proposals. Proximity defaults to
**1.5 graph voxels**, after coordinate scaling, using the existing exact
fiber-point-to-patch-surface query. A patch qualifies when at least **two dense
fiber points** fall within Proximity. Discovery always checks every dense point
in fiber review; the stride control applies only to full graph mode.

Only the current proposal moves; the fiber and earlier accepted patches stay
fixed. Accepted patches on the selected fiber supply overlap constraints to
the proposal, but do not change discovery eligibility. Each patch has one proposal:
alignment uses all dense contacts within **twice Proximity** (3 vx with the
default setting), including separated regions and contacts far apart in the
initial flattened view. The solver displays its best valid fit even when some
residual alignment error exceeds Proximity; the reported RMS helps review it.
Overlap matching searches immutable 3D surface positions within twice Proximity
and rejects strongly disagreeing surface normals. The resulting targets use
the accepted patches' fitted chart positions. Their total weight is capped at
the fiber-contact weight so dense patch grids do not overwhelm the fiber.
Rejected/skipped patches and the proposal itself never supply targets.

Review starts with a deterministic, iteratively reweighted rigid alignment,
then refines the proposal parameterization with symmetric Dirichlet energy.
An additional squared per-triangle excess-energy term discourages concentrating
the deformation into a few severely stretched or compressed triangles. Review
allows up to 240 solver iterations, stopping earlier on convergence.
Automatic fiber contacts use Huber losses; committed-patch targets use Cauchy
losses so incompatible regions of an already distorted placement lose influence.
Manual keypoints retain their strict constraints. The complete
proposal is free to move: review does not impose the full graph editor's local
region boundary or require a fused fiber/patch surface. The displayed patch
layer must still pass geometry validation, including the checks for collapsed
or inverted charts. Failed fits report their actual solver, keypoint, or patch
layer error. Overlap constraints are rebuilt from current accepted decisions
on reopen, resume, and manual refit; saved associations remain independent.
If deformation creates a folded raster cell, or any triangle exceeds 4x stretch
or compression in a principal direction, review retries with progressively
stronger shape regularization (4x, 16x, then 64x). This checks every triangle,
including shear, rather than relying on a percentile over edge lengths. Remaining
excessive distortion reports its measured maximum and prevents committing the
proposal. All contacts remain in the fit with their robust weights,
and invalid geometry is still rejected. The activity log records retries; the
preview reports the overlap sample count and any regularization used.

Previously committed placements keep their saved geometry until reopened and
committed again. Reopening uses the refined objective and current accepted
overlaps; it does not move the fixed fiber or other committed patches.

The refinement regression replay on saved fiber
`anon_20260817T021012847_000129.json` covers all 29 accepted patches, with
fiber-only, earlier accepted context, all current context, and saved-placement
reopening (116 fits). All 116 fits pass. For `band-seed4179111`, the same
historical five overlap targets previously produced 27.45x maximum edge stretch;
refinement produces 1.39x. `band-seed3343090` improves from 5.40x to 1.53x.
The worst edge stretch across all 29 reopened patches is 3.57x. These are edge
measurements for comparison; acceptance uses principal stretch and compression.
Regression tests also check the refined metric's derivatives and inversion
barrier, recovery from a saved spike with conflicting overlap evidence, and
explicit failure when manual constraints require excessive distortion.

**Add keypoint pair** opens the source
pane for manual alignment; valid fits can be committed. A failed fit cannot be
committed and never falls back to deforming the fiber.

- **Commit / Enter** accepts the current alignment; **Reject / Delete** rejects
  the pair. Both automatically advance and suppress all remaining placements
  of that patch on this fiber. Other fibers retain independent decisions.
- **Skip** advances and suppresses that patch for the current fiber, recording
  **Skipped** separately from **Rejected**. It persists on save/reopen, leaves
  other fibers unaffected, and can be undone or reopened from the decisions
  list. Skipped pairs are excluded from rejection audits and remain unlabeled
  in learning exports.
- **Quarantine patch** moves the current patch's whole source folder to the
  same quarantine destination used by full graph mode: a sibling of the saved
  workspace directory named `<directory>_quarantined`. It removes that source's
  proposals and accepted associations across all fibers, saves the updated
  review, and advances without changing the camera. To quarantine an accepted
  patch, reopen its pair first. A new review prompts for a save location.
  The shared transaction creates a `before-quarantine` backup and move journal
  and rolls back file moves if publication fails. Quarantine clears undo/redo;
  the saved exclusion remains active when reopening or rediscovering proposals.
- Accepted patches remain visible at **Committed opacity**, initially 20%.
  **Patch opacity**, initially 50%, controls only the active proposal. Accepted
  patches overlap in acceptance order; the active proposal is drawn last.
  Holes reveal underlying layers. Opacity changes reuse sampled images.
- **Show all boundaries** is off initially and uses the ordinary workspace's
  boundary display, including the current proposal outline.
- Click an accepted patch, or double-click an accepted/rejected/skipped entry in the
  decision list, to reopen that pair. Undo/redo covers the last 20 decisions
  during this session; each entry restores the decided fiber and returns to
  it, leaving other fibers' discovery and decisions untouched. **Pause /
  Resume** retains pending alignment edits.
- Switching fibers preserves decisions and each fiber's current review position.
  The camera keeps its zoom and pans to each next proposal's contact. The
  previous fiber stays on screen until the new layout is ready. Accepted
  layers are rasterized in parallel and kept in a per-session cache (768 MB,
  least recently shown evicted) so returning to a fiber reuses them.
- The viewer renders each accepted layer only over the pixels its patch can
  cover, so a fiber with a hundred accepted patches pans and zooms at roughly
  the cost of the fiber itself. `--profile` logs `surfaceLayers elapsed_ms`.

### Automatic linking

**Auto link fiber** decides every undecided proposal of the selected fiber
from geometry alone; **Auto link all fibers** discovers and decides every
fiber in the catalog (cancel keeps the fibers finished so far; not undoable).
Decisions made by hand are kept. Linked patches are fitted with the ordinary
review solver, using earlier automatic accepts as overlap targets, and stored
as **Auto-accepted**; a linked patch whose fit fails becomes **Auto-skipped**
with the solver message; everything else becomes **Auto-rejected**. Each
automatic decision carries a note (shown as the decision's tooltip) with the
measurements behind it. **Next auto-rejected** reopens the next automatically
rejected pair in proposal order so a reviewer can walk through what the
linker declined; deciding a reopened pair by hand clears its automatic mark.
The workspace stores `automatic` and `note` on decisions; older files load
with both absent.

The linker (`vc::constraints::autoLinkFiber`, `core/src/constraints/AutoLink.cpp`)
has two mechanisms and no learned model:

- **Anchors** are patches the fiber lies on for at least **Anchor contact**
  (default 75 graph voxels of contact runs, merging gaps up to twice the strip
  spacing) whose vertices inside the fiber strip's footprint agree with the
  strip more than they sit one sheet away from it. Agreement is the fraction
  of vertices within 3 vx of the strip with parallel normals, adjacency the
  fraction 4.5–16 vx away with parallel normals, both weighted 1, 0.7, 0.4,
  0.15 by strip rows from the centerline, since the normal-field strip is
  less reliable away from the fiber.
- **Growth** links any remaining patch whose vertices overlap the linked set
  on the same sheet (≥10% within 3 vx with parallel normals) clearly more
  than on an adjacent sheet (more than twice the adjacent fraction), repeated
  until nothing changes. A wrong-sheet cluster is internally consistent but
  never touches the linked set, so it stays unlinked for human review.

Validation on the `fiber_linking.json` review (2998 accepted, 335 rejected
pairs after the reviewer re-checked every disputed rejection): 10-fold
cross-validation grouped by fiber gave 0 false links and 84% recall of the
accepted pairs. Without the strip gate anchors needed 150 vx of contact to
avoid a wrong-sheet cluster on fiber `…000123`; with it, false links only
appear below 60 vx. The C++ port reproduces the Python prototype's decisions
on 3630 of 3644 pairs and links 0 rejected pairs on the same review:

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/constraint_graph_probe --auto-link-review /home/sean/Desktop/fiber_linking.json /tmp/auto-link.jsonl [anchor_contact]
```

`vc_fiber_auto_link` runs the same linker without the UI over a fiber folder,
patch folder and normals manifest, writing a `vc_fiber_patch_links` JSON that
`spiral-fitting/auto_link_fibers.py` converts into `explicit_link:<patch>`
tags on the fiber files for `fit_spiral.py`. The CLI builds each fiber's
strip with the same builder as VC3D (`makeFiberNode`): the fiber resampled
every 20 vx, each row marched outward along the in-sheet cross direction
while re-reading the normal field, with invalid, folded or disagreeing
normals interpolated from their neighbours and a light smoothing along the
fiber. When the normals manifest has a `pred_dt` channel (a sheet score that
peaks on the papyrus), every marched point also climbs that score along the
local normal within 2 vx, so the ribbon re-centres on the sheet instead of
drifting: measured against the same-sheet accepted patch, the ±60 vx rows go
from 1.6 to 0.95 vx median (79% to 94% within 3 vx), as close as the fiber
line itself. It has none of the robust fit's validity guarantees, which the
distance-based linker does not need, and takes milliseconds per fiber where
the v7 fit's repair cascade takes seconds to minutes. Because a re-centred
ribbon agrees more with a neighbouring sheet where two sheets touch, the
anchor gate requires the strip same-sheet fraction to exceed **1.5x** the
adjacent fraction (`--anchor-strip-ratio`); with a plain 1x gate the snapped
strip linked 8 rejected patches on two fibers. On the reviewed fibers
the two ribbons share the centerline and differ by a median 0.7 / 1.4 / 2.1 vx
at the ±20 / 40 / 60 vx rows; where they disagree by more than 12 vx the fast
ribbon lies on the human-accepted patches (median 3 vx) and the robust one
does not (median 16 vx), so the fast strip is the better sheet estimate for
linking. On the review of 2026-09-22 (3492 accepted, 409 rejected) the snapped
fast strip links 83% of accepted pairs, the unsnapped 80%, both with the same
single rejected pair linked (fiber `…000201`, `band-seed2763590…`: 14 vx of
contact but 93% same-sheet overlap with the linked patches). `-j` spawns worker *processes* over longest-first shards and
merges their outputs; the result is independent of the job count.
`--strip-cache DIR` stores every strip keyed by the fiber file's bytes and the
strip settings, so re-runs skip the strip stage entirely. The
whole 765-fiber catalog runs in about a minute with 12 workers.

```sh
volume-cartographer/build/bin/vc_fiber_auto_link --fibers <fibers> --patches <patches> --normals <las.lasagna.json> --fiber-scale 0.25 --normal-scale 4 -j 12 --strip-cache <fibers>_strips -o fiber_patch_links.json
```

**Save / Save as** writes the compact `vc_fiber_patch_links` document (version
2): the settings and, per fiber, each decided patch with its status, decision
order, automatic flag, note, linker measurements and any manual keypoint pairs,
plus keypoints added to a still-undecided proposal. A whole review is a few
hundred kilobytes. Strips, proposals and fitted placements are not stored:
selecting a fiber rebuilds its strip, discovers its proposals and fits its
accepted patches in decision order (about half a second per patch, cached for
the session), each fit using the placements fitted before it as overlap
targets. An accepted link that cannot be registered becomes Auto-skipped with
the solver message. Because placements are refitted, a review reopened after a
solver change shows the new fit. **Open** accepts this document directly,
including the version-1 file `vc_fiber_auto_link` writes, so the CLI's result
can be reviewed in the workspace without any conversion, and it still opens
the legacy multi-gigabyte `vc3d_fiber_review` workspace described below, which
saves back down to the compact form.

The legacy `vc3d_fiber_review` version-2 JSON workspace used It stores shared immutable source geometry once, per-pair
contacts and placements, rejected decisions, fixed fiber charts, pending edits,
source paths and scales, review settings, and positions. **Open** restores the
mode automatically and still reads version-1 workspaces. Version 2 stores
accepted, rejected, and skipped decisions explicitly. Saved accepted views can be reconstructed without their
original files; new proposals require the source catalog and normals. CT still
requires the volume. Keep input folders unchanged during a review. After reopening
with a successfully loaded source catalog, saved proposal lists drop unavailable
patches and advance past them. Decisions and embedded pending alignments are
retained; save the workspace to persist the cleaned lists. If the source catalog
is unavailable, proposal lists are left intact.
Older proposal caches refresh to use whole-fiber contacts when sources are
available. Saved decisions and manual keypoint edits are retained.

An accepted patch can have a different placement for each fiber. These charts
are deliberately stored as separate associations, not as one globally fitted
surface. Source files are never rewritten. There is no spiral-fitting integration
or tifxyz export for this mode beyond the compact links document, which
`spiral-fitting/auto_link_fibers.py` reads directly.

The read-only real-data validation probe takes input locations from an existing
ordinary graph session, creates single-fiber reviews, checks that fits preserve
the fibers, and verifies a saved review round trip:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build VC3D constraint_graph_probe test_constraint_graph test_constraint_graph_overlay test_constraint_graph_activity -j 4
QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 ctest --test-dir volume-cartographer/build --output-on-failure -R '^constraint_graph(_overlay|_activity)?$'
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/constraint_graph_probe --fiber-review /path/to/graph.json /tmp/fiber-review.json
volume-cartographer/build/bin/VC3D --volpkg /path/to/project.json --constraint-graph-session /tmp/fiber-review.json
```

Validation on Ubuntu arm64 used the real inputs referenced by
`/home/sean/Desktop/new_graph_2.json`: 765 catalog fibers, two reviewed fibers,
four accepted patches and one rejection. Fiber coordinates stayed unchanged,
and the saved review round-tripped exactly. An offscreen UI replay with
`s1_ds2.zarr` checked opacity, commit/reject, decision reopening, undo/redo,
rapid fiber switching, and save/open. All three constraint-graph CTest suites
passed, including the new review and layered-opacity cases.

#### Registration validation

The read-only replay below samples up to 12 rejected decisions and four
accepted controls from a saved review. A rejected decision is a human label,
not evidence that its automatic fit failed. `--review-fit-audit` isolates the
fiber-only solver; `--review-overlap-audit` also measures overlap RMS before
and after fitting against the other accepted patches on the same fiber.
Outputs must be separate from the input workspace.

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/test_constraint_graph
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/constraint_graph_probe --review-fit-audit /home/sean/Desktop/fiber_linking.json /tmp/review-fiber-audit.jsonl 12 4
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/constraint_graph_probe --review-overlap-audit /home/sean/Desktop/fiber_linking.json /tmp/review-overlap-audit.jsonl 12 4
```

On Ubuntu arm64, RelWithDebInfo, workspace revision `41f9d05d3bc17956`, the
same 16 pairs produced 12 valid previews before the review-specific solver
changes and 16 afterward, both fiber-only and with overlap constraints.
The four original failures reported only a generic error; their underlying
cause was the local-region boundary. Four overlap fits required stronger
regularization to pass patch-raster validation (two at 4x, two at 16x).
Fourteen pairs had accepted overlaps; all fourteen reduced their overlap RMS.
The median per-pair RMS fell from 31.181 to 9.295 working voxels compared with
the new fiber-only fit. These are registration residuals on matched targets,
not independent ground-truth accuracy: the worst remaining overlap RMS was
36.009 voxels. Human review is still needed, especially for nearby sheets and
inconsistent accepted placements. No macOS or amd64 validation was performed.
All 53 constraint tests passed. An offscreen replay using `s1_ds2.zarr` and a
temporary two-fiber copy of that workspace also passed opacity changes,
commit/reject, undo/redo, accepted-pair reopening, rapid fiber switching, and
save/open. The original workspace was not modified.

#### Rendering performance validation

Accepted layers reuse rendered pixels while their surfaces, camera, volume
chunks, bounds and display settings remain unchanged. The viewer also retains
their composited image, so a new proposal only samples and blends its own
layer. Chunk arrivals and view changes invalidate reuse. Scaled accepted
surfaces keep their identities across commits. Blending retains the original
layer order, per-layer rounding and full precision.

After a normal build, prepare and build the isolated render benchmark. It
links existing application objects and writes its own object/executable:

```sh
AGENTS_AGENT_MODE=1 ninja -C build -j 12
python3 apps/VC3D/test/prepare_fiber_review_benchmark.py --build-dir build
AGENTS_AGENT_MODE=1 ninja -C build -f fiber-review-benchmark/build.ninja
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 build/bin/benchmark_fiber_review_render /tmp/fiber-review-real.json /home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr /tmp/fiber-review-render-after.json
```

Run these commands from `volume-cartographer`. Substitute a review with an
accepted patch and volume scale 1, and its CT volume. The benchmark repeats a
real accepted patch as independently rendered layers to stress accumulation;
it is not a timing of fitting or disk loading. Each count uses three warm-ups
and 12 timed proposal replacements at a fixed camera, 0.75 pixels per voxel,
with a 1320 × 910 viewport. It also compares cached images with forced fresh
renders after layer replacement/reordering/removal, opacity changes, tint
changes/removal, zoom, pan and windowing.

Ubuntu arm64, GCC 13, RelWithDebInfo, one OpenMP thread; input was the saved
review and `s1_ds2.zarr` above. Times below include rendering and UI composition:

| Accepted layers | Before mean / p50 / p95 (ms) | After mean / p50 / p95 (ms) |
| --- | --- | --- |
| 1 | 61.7 / 61 / 80 | 25.7 / 27 / 36 |
| 8 | 164.5 / 163 / 195 | 22.9 / 22 / 29 |
| 32 | 493.5 / 500 / 529 | 16.8 / 14 / 25 |

All three final framebuffer hashes matched before/after. At 32 layers,
profiling showed worker time dropping from 403 to 13 ms: the fiber and 32
accepted layers were reused and only the proposal was sampled. Previously,
repeated coordinate generation, chunk lookup/sampling and composition scaled
with the layer count. Camera changes still require resampling the layers;
this benchmark measures stationary-camera proposal review with warmed chunks.

### Full graph mode

- Orange marks identify attachment locations within **Proximity (vx)**, initially
  4 voxels in graph coordinates. Marks appear progressively as exact proximity
  checks finish and remain visible while selecting or fitting proposals.
  Click a marked location to preview a patch; the selection uses cached discovery.
- **Discovery stride** checks every **4th** dense fiber line point by default,
  plus both endpoints. Choose **1** for every point or **8** for fewer checks.
  This applies to both patch proposals along placed fibers and fiber proposals
  near placed patches. Larger strides can miss short contacts and yield fewer
  proposals; all retained contacts still pass the exact
  proximity check. Fiber geometry and rendering resolution are unchanged.
  Changing the stride restarts discovery and discards the current preview. The
  preference persists across app launches and does not alter saved sessions.
- Drag the **Drag to move** handle on the selection bar to place it anywhere in
  the workspace. It stays there through selection and camera changes.
- Patches have distinct outline colors, shared between the graph and source
  panes. Only proposals show colored fiber lines and patch boundaries by default;
  their boundaries disappear on **Commit**. Enable **Show all boundaries** in
  the toolbar to show committed objects' boundaries too. Selecting a patch
  thickens its visible outline without changing its color.
- After a proposal finishes fitting, enable **Compare fiber / patch CT** in
  the selection bar. **Patch opacity** blends the independently sampled patch
  CT over the fiber CT at their fitted chart positions: 0% shows fibers, 50%
  exposes texture misalignment, and 100% shows patches where they have coverage.
  Patch holes leave the fiber image visible. Disable comparison to return to
  the joint surface. This changes display only; it does not refit or affect
  Commit, saved geometry, or export. Comparison layers are generated on demand
  in a separate cancellable worker; Commit remains available while they load.
  The control is unavailable during the initial fit, with **Show committed**,
  or without both source types.
- Use **Tab / Shift+Tab** or the arrows in the selection bar to cycle proposals.
- The main pane keeps the accepted surface visible while the attachment fits.
  The second pane shows the proposed source as soon as it is loaded. A validated
  local fit replaces the main pane's surface and enables **Commit**.
  The last three completed previews are cached until
  the committed graph or its settings change.
  **Show committed**
  temporarily restores the accepted graph in the main pane.
- To adjust alignment, choose **Add keypoint pair**, click a recognizable feature
  in the proposal pane, then its match in the main pane. Add another pair to
  constrain rotation. User pairs replace automatic guesses between that node
  pair. The list supports removing and replacing pairs.
- **Commit** accepts a valid fit. **Cancel / Escape** discards the proposal.
  Undo and redo operate on committed edits.
- Selecting a committed patch finds nearby unplaced fibers. **Include linked
  component** adds all fibers linked to the proposed fiber.
- If an attachment cannot pass validation within the local work budget, its
  proposal stays available and the status asks for **Refit graph**. That action
  fits the accepted graph together with the pending addition; review the result
  and **Commit** or **Cancel**. Ordinary selection never starts a full refit.

Ordinary additions refit the patch, overlapping neighbors, and nearby sections
of fibers. Distant accepted chart vertices and surface samples remain fixed.
When adding fibers, the local region includes the entire accepted patches they
contact, so their existing UV distortion can relax across patch boundaries.
If this joint fit exceeds its work budget, the proposal requests **Refit graph**;
it does not accept a fallback that freezes those patches and bends only the fiber.
Selecting a patch verifies every dense point on every placed fiber against
that patch, independently of the discovery stride. Adding a fiber or linked
component performs the same verification against every accepted patch; other
fibers in the new component also receive their own patch contacts. These contacts constrain
the joint fit in addition to the clicked cluster. Contacts must pass the same
3D proximity check and lie within four grid intervals (80 vx by default) of
the patch's initial placement in the accepted chart. This prevents a fiber
that revisits the same 3D region on a distant traversal from pulling separate
parts of the layout together. Manual keypoint pairs still replace automatic
matches between their node pair.
Real 3D patch overlaps add correspondence constraints; symmetric Dirichlet terms
penalize stretching and compression in both the chart and output grid. In
incremental fits both accepted and added fibers are the reference frame: their charts
resist deformation ten times harder than before and an added patch's chart
half as hard, so a patch deforms onto the fibers it touches instead of bending
them (see *Fibers as the reference frame* below). Source fiber and patch XYZ
are never rewritten by any of this. Invalid fits cannot be committed. Spacing ratio
and attachment RMS in the status area describe fit quality; they are not a
guarantee that the selected patches follow the correct sheet. Inspect fiber
continuity in the rendered volume before committing.

Horizontal fibers share a fitted row and vertical fibers share a fitted column.
Each fiber can bend within two grid spacings (40 voxels) on either side of its
row/column without an alignment penalty; larger departures are penalized softly.
An additional endpoint constraint keeps each horizontal level end to end and
each vertical upright, while retaining local bends inside that band. Both
constraints apply during rigid alignment and joint deformation, including when
a patch attaches to several fibers. Use **Refit graph**
to preview this alignment on an existing session, then **Commit** or **Cancel**.

At explicit horizontal–vertical links, the horizontal supplies the 3D position
of the sheet. The linked vertical can trace its back: its ribbon receives the
full XYZ offset needed to reach the horizontal control. The correction fades
smoothly to zero over four grid intervals (80 voxels at the default spacing)
along the vertical. Nearby links interpolate between their offsets instead of
adding them together. The fit and rendered surface use this same derived
geometry. The shared output grid also constrains its interpolated junction
samples to the horizontal positions so smoothing cannot pull them back toward
the vertical's original depth. Measured patches retain precedence over ribbon
support. Unlinked crossings do not introduce corrections; source fibers and
patches are not rewritten. Incompatible horizontal positions at the same
vertical control produce an error. Saved surfaces reopen as saved; use
**Refit graph**, then **Commit**, to apply this rule to an existing session.

## Coordinates and saved sessions

The default fiber-to-graph scale is 0.25, patch-to-graph scale is 1, and
graph-to-normal-base scale is 4. These defaults match fibers in full-resolution
coordinates and patches at quarter resolution. Fibers declaring a base shape
use that metadata for conversion. **Graph → volume scale** changes volume
sampling without changing attachment tolerance or saved geometry.

**Save** writes a versioned JSON session containing immutable source geometry,
links, keypoints and placements, plus a content-addressed tifxyz sibling folder.
**Open** verifies a checksum binding the saved grid, origin, and graph and restores
the exact accepted surface. Legacy sessions, or missing/damaged derived grids,
rebuild from their source mappings. Unavailable original inputs still allow viewing
and exporting the saved graph. Growth requires the original source catalog and
normals. **Export tifxyz** writes the accepted graph to a chosen directory.
Source fibers and patches are never edited.

Large components and catalogs run asynchronously. Proposal discovery has its
own cancellable worker and continues when selecting, cycling or canceling a
preview. Graph/radius changes restart discovery. A newer preview cancels the old
fit; completed stale work cannot replace a newer graph.
The output grid is limited to 16 million samples. Input folders should remain
unchanged during a session.

## Navigation and interaction

The **Activity** panel stays visible below the viewers. It shows the current
operation, its latest stage, elapsed time, and an animated busy indicator.
**Refit graph** reports alignment, flattened-layout fitting, and output-surface
refinement iterations. Iteration limits are per solve, not an overall percentage
or an estimated completion time; local fits can expand and repeat a solve.
File loading and attachment discovery report counts where available.

Attachment search and source comparison have independent activity rows, so
background work cannot replace a refit's progress. **Show activity log** retains
the last 200 messages, including failures and completed stages. Errors remain
selectable in the panel instead of being duplicated in the sidebar and status
bar. **Stop operation** requests cancellation and shows **Stopping…** until the
worker actually returns. It is disabled during save/export, which do not support
interrupting file writes. A long individual solver step or file read can delay
cancellation; the elapsed clock continues while waiting.

Opening a graph or selecting a new seed component centers a seed-fiber junction
at ten screen pixels per grid interval (0.5 pixels per graph voxel with the
default 20-voxel spacing). This makes fiber continuity inspectable immediately.
**Fit view** remains available to show the entire graph.

Graph outlines and deduplicated proposal locations are retained until the graph
or proposals change. Pan/zoom transforms the cached outlines and updates a
batch of constant-size, visible proposal markers. Camera updates do not rebuild
patch boundaries or repeat source-location lookup and proposal deduplication.

For an isolated test instance, set `VC3D_PROFILE_CONSTRAINT_GRAPH=1` to log
overlay rebuild and camera-update timings in milliseconds. After loading a
session and waiting for proposal discovery to finish, choose **Fit view** and
run the repeatable zoom benchmark from the repository root:

```sh
python3 volume-cartographer/apps/VC3D/agent_bridge/test/benchmark_constraint_graph_interaction.py \
  --socket /tmp/constraint-graph-benchmark --viewer v6 --scale 0.02 \
  --iterations 40 --output /tmp/constraint-graph-interaction.json
```

Start the test app with `--agent-bridge-name constraint-graph-benchmark`; check
`state.get` for the Constraint Graph viewer ID if it differs from `v6`. The
benchmark alternates 1.05× zoom and its reciprocal, waits 100 ms between
requests, and reports mean, median, p95 and individual request times. It measures
UI request latency, including synchronous interaction work, rather than time
until all CT tiles finish rendering. Compare the same session, viewport,
camera, build type and completed proposal discovery.

Validation on Linux arm64, `RelWithDebInfo`, using the 95-node saved session
`/tmp/constraint-graph-real/axis-aligned-grown.json` (92 fibers, three patches),
the local `s1_ds2.zarr` volume, and all 175,755 proposal locations:

| Run (40 zoom requests each) | Zoom | Mean | Median | p95 |
| --- | ---: | ---: | ---: | ---: |
| Before, onscreen overview | 0.02 | 704.1 ms | 684.8 ms | 743.3 ms |
| After, offscreen overview | 0.02 | 45.5 ms | 45.5 ms | 49.1 ms |
| After, offscreen opening view | 0.5 | 1.85 ms | 1.80 ms | 2.45 ms |

The final runs used `QT_QPA_PLATFORM=offscreen` because additional input in the
live window changed its camera and selected proposals. Thus the before/after
request figures are an indicative comparison, not a controlled frame-rendering
speedup: backend, viewport and camera center can differ. The instrumented
baseline isolated the repeated overlay rebuild at 177.6 ms median, including
172.1 ms for proposal deduplication/marker construction. Those operations now
run only when their inputs change; camera updates reuse the cached data.
The opening zoom and alignment were also inspected in the live volume view.
No fitting, discovery thresholds, source geometry or solver numerics changed.

Rendering regression tests cover transformed retained outlines, constant-size
markers, switching batch styles, and safe cache handles after scene deletion.
The geometry tests cover initial focus and zoom independence from graph extent.

## Source fiber cycle check

Loading folders or reopening a session checks the **current source fiber folder**,
across all components. The persistent **Fiber cycle check** label and **Cycle
check / input details…** button show results even while working with proposals.
Warnings are advisory; they do not alter annotations or prevent graph editing.
When evaluated cycles agree, the label says **No inconsistencies found**.
Only detected closures or input problems receive a **Warning** label; coverage
notes remain in the details. Unavailable checks are labeled **unavailable**.

The check unwraps each original, control-point-bounded dense fiber around the
umbilicus and follows annotated links around an independent cycle basis. Each
link constrains the whole-turn difference between the two fibers' winding
coordinates. A cycle must sum to zero for a globally consistent winding
assignment. Nonzero closures list all participating links, filenames, and
zero-based control indices; they do not assign blame to a particular link.
Reciprocal links are counted once. Missing reciprocal links and stale endpoint
position snapshots are reported separately as input warnings.

Linked control points may have different 3D positions. Their measured angular
offset is preserved; proximity alone is not an annotation error. This is a
**winding consistency check**, not a proof that every metric or rotational
closure is zero. Curved sheets can have geometric holonomy. A nonzero winding
closure needs review or a chart cut and is not, by itself, proof of a bad
annotation. Patch registration and manually added keypoints are outside this
source-fiber check.

The local Lasagna manifest supplies `umbilicus_json`, resolved relative to the
manifest and interpreted in its base coordinates, then divided by the graph's
normal-base scale. The angular unwrapping/interpolation helper is shared with
Fiber Map. Missing/invalid centerlines and unsupported remote centerline inputs
produce an **unavailable** result. Pending links, endpoints with angular gaps of
at least a quarter turn, fibers passing within 417 base voxels of the center,
and angular sample steps exceeding a quarter turn are excluded. Cycles involving
untraced/interpolated fiber spans or centerline extrapolation are **uncertain**.
The result reports how many cycles were evaluated. Details distinguish these
from cycles supported entirely by traced geometry. Coverage limitations alone
do not trigger a warning. Any uncertain span makes its entire fiber uncertain.

## Build and validation

From the repository root, with the existing CMake configuration and dependencies:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build VC3D test_constraint_graph test_constraint_graph_overlay test_constraint_graph_activity constraint_graph_probe -j 4
QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build --output-on-failure -R '^constraint_graph(_overlay|_activity)?$'
volume-cartographer/build/bin/VC3D
```

Open a session directly:

```sh
volume-cartographer/build/bin/VC3D --volpkg /path/to/project.json --constraint-graph-session /path/to/graph.json
```

The read-only input validation harness loads a catalog, fits a selected component,
saves a session, and tries patch proposals. Its output files go only to the
specified session destination:

```sh
volume-cartographer/build/bin/constraint_graph_probe \
  /mnt/bigpc/spiral_dataset_working/fibers \
  /home/sean/Desktop/spiral_dataset/graph_patches \
  /home/sean/Documents/volpkgs/s1_2um.volpkg/las_008_s1_full/las_008.lasagna.json \
  /tmp/constraint-graph-validation/graph.json
```

An optional final integer selects a different component (zero-based, largest
first). To validate fiber-component growth and a second overlapping patch from
a saved proposal, run:

```sh
volume-cartographer/build/bin/constraint_graph_probe --grow /tmp/constraint-graph-validation/graph.json.proposal.json
```

This writes a `.grown.json` session next to its input and fails if none of the
tested candidates passes either growth stage.

Audit annotations without fitting, opening volume data, or loading patches:

```sh
volume-cartographer/build/bin/constraint_graph_probe --audit \
  /mnt/bigpc/spiral_dataset_working/fibers \
  /home/sean/Documents/volpkgs/s1_2um.volpkg/las_008_s1_full/las_008.lasagna.json
```

This command only reads inputs and prints its report. Exit code 1 indicates
input problems, a nonzero closure, or an unavailable check; 0 indicates no
findings. Coverage notes are printed separately and do not change the exit code.

To refit an existing graph using its embedded source geometry:

```sh
volume-cartographer/build/bin/constraint_graph_probe --refit /path/to/graph.json /tmp/refitted-graph.json
```

Inspect horizontal–vertical junctions in a saved surface:

```sh
volume-cartographer/build/bin/constraint_graph_probe --junctions /path/to/graph.json
```

The report separates the original source offset, the derived vertical anchor
error, and the saved surface's distance from the horizontal at each supported
junction. `corrected_xyz_rms` describes the correction that a refit will use;
it does not imply an old saved surface has already been updated.

Validation on Linux arm64, RelWithDebInfo, using
`/home/sean/Desktop/patch_test.json` (92 nodes, 131 H/V links):

```sh
volume-cartographer/build/bin/constraint_graph_probe --refit \
  /home/sean/Desktop/patch_test.json /tmp/cg-horizontal-junctions/final.json
volume-cartographer/build/bin/constraint_graph_probe --junctions /home/sean/Desktop/patch_test.json
volume-cartographer/build/bin/constraint_graph_probe --junctions /tmp/cg-horizontal-junctions/final.json
```

Across the same 128 junctions with complete output quads, rendered position
error relative to the horizontal changed from 5.0713 to 0.0024 voxels RMS
(maximum 16.1094 to 0.0151 voxels). Three junctions lacked complete quads in
both versions. The final fit passed validation with attachment RMS 0.0047 and
spacing ratio p95 1.0169. Embedded source geometry and annotated links remained
identical. Regression tests cover fractional controls, local fade-out,
overlapping blends, link reversal, unlinked fibers, horizontal surface anchors,
conflicting controls, exact save/reopen, and preservation of distant surface
samples during incremental edits.
The linked seed junction was also inspected before/after in an isolated
offscreen VC3D using the real `s1_ds2.zarr` CT volume at 2 pixels per graph
voxel. Captures are `/tmp/cg-horizontal-junctions/before-ct.png` and
`/tmp/cg-horizontal-junctions/after-ct.png`; the original session was not edited.

Regression tests use temporary synthetic fiber/patch fixtures and cover
trimming, arclength mapping, chart handedness, isotropy, real surface proximity,
patch overlaps, component growth, paired registration, cancellation, malformed
sessions, and save/reopen.

Activity tests cover independent foreground/background rows, retained history,
late updates after completion, and stop-button availability. Solver tests verify
iteration reporting, cancellation from a progress callback, and identical chart
and surface results with and without reporting. An isolated offscreen VC3D was
also exercised with `/home/sean/Desktop/new_graph_2.json`: opening/preparing,
concurrent discovery, full-refit iteration reporting, and acknowledged Stop were
observed without modifying the session. Captures are
`/tmp/cg-progress-validation/refit.png` and `cancelled.png`.
Repeating the 440-contact fiber addition below produced iteration updates
throughout its 26-second fit, followed by surface-refinement updates. Its graph
JSON and saved-surface checksum were identical to the fit before progress was
added; the resulting session also reopened exactly.

## Prepared graph and bounded attachments

The workspace prepares each accepted graph revision once. This caches chart
bounds and triangle blocks, contact interpolation weights and incident-contact
lists, horizontal/vertical junction geometry, and distortion statistics. A
proposal contains added sources and added/removed contacts relative to that
revision. It does not copy and compare the full contact list to rediscover the
edit. Chart UV and contact storage use copy-on-write snapshots for previews and
history; immutable source XYZ remains shared.

The chart solve gathers only affected triangles and contacts, retaining fixed
boundary constraints and their original order. The surface update rasterizes
the affected region and reuses distant accepted samples. Cached statistics for
unchanged charts and 64×64 surface tiles avoid remeasuring their geometry.
After Commit, preparation reuses unchanged chart, junction, contact-weight,
and tile data. Cancellation and revision checks prevent old fit or comparison
results from replacing a newer preview.

Ordinary edits have a deterministic work budget: four chart attempts, at most
32,768 movable chart vertices plus the sizes of accepted patches contacted by
new fibers (capped at 131,072 total), and four surface expansions covering at most
262,144 grid sites. If neighborhood fitting fails, additions other than fibers
contacting accepted patches get one attempt against fixed accepted charts.
The same residuals, precision, attachment
tolerance, and geometry checks apply. A remaining failure requests an explicit
**Refit graph**; it cannot be committed. These are work limits, not relaxed
acceptance thresholds or a hard wall-clock deadline. Full seed construction
and explicit refitting can still be expensive.

The existing dense QuadSurface still requires a final grid copy, and exact
global quality reporting aggregates cached values. Completed edits materialize
the contact list once. Thus the entire preview path is not constant-time in
graph size; the expensive geometry and solver setup now follow the local edit.
The older `fitIncremental` API remains available for comparison tests; the
workspace uses `proposeEdit` / `fitEdit`.

### Reproducing attachment measurements

Use the build commands above. The following commands read the supplied input
and write only explicitly requested output sessions. Catalog loading and
preparing the accepted revision happen before each repeated-selection timer;
`prepare_ms` reports preparation separately. Timed work includes proposal
construction, source preview, and validated fitting, but not CT tile painting,
comparison layers, saving, or discovery.

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 VC3D_PROFILE_CONSTRAINT_FIT=1 \
  volume-cartographer/build/bin/constraint_graph_probe \
  --edit-proposal-benchmark /home/sean/Desktop/new_graph_2.json 3
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 \
  volume-cartographer/build/bin/constraint_graph_probe \
  --edit-proposal-scan /home/sean/Desktop/new_graph_2.json 15
mkdir -p /tmp/constraint-graph-fresh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 \
  volume-cartographer/build/bin/constraint_graph_probe \
  --fresh-edit-growth /home/sean/Desktop/new_graph_2.json /tmp/constraint-graph-fresh/growth 4
```

The fresh-growth command takes only locations and coordinate settings from the
session, constructs a fresh largest fiber component, and commits successive
valid local additions with incremental cache preparation. Each saved result
must reopen with exactly the same surface. It reports rejected candidates too,
without silently refitting them globally. The proposal benchmark accepts an
optional output prefix, candidate index or source ID, and selected node ID (`patch` selects
the first accepted patch).
An additional final proximity argument overrides the radius in memory for a
diagnostic run; the input session is not modified. `discovered_contacts` counts
all discovery clusters for the selected source, whereas `contacts` counts the
clicked cluster. Attachment comparison includes both link orientations.

### Adding fibers to warped committed patches

The old local region could cut through a committed patch before its deformation
had settled. Exceeding the 32,768-vertex budget then retried with every accepted
chart frozen, producing a valid preview in which only the new fiber deformed.
Fiber additions now include the full bounds of every accepted patch contacted
by the addition, allow room for those patches in the bounded solve, and use the
same fiber stiffness for new and accepted fibers. If that joint fit still fails,
it requests **Refit graph** instead of freezing the patches. Patch additions and
the full-refit objective retain their previous behavior.

Real-data validation used `new_graph_2.json`, patch `band-seed3889268…117`, and
horizontal fiber `anon_20260822T114841824_000171.json`. The saved radius is 2 vx
and finds no proposal for that pair. This diagnostic uses 20 vx to exercise its
440-contact crossing; it does not change the application's proximity default.
From the repository root, after the build above:

```sh
mkdir -p /tmp/cg-warp-validation/empty-patches
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 \
  VC3D_BENCHMARK_PATCH_FOLDER=/tmp/cg-warp-validation/empty-patches \
  volume-cartographer/build/bin/constraint_graph_probe \
  --edit-proposal-benchmark /home/sean/Desktop/new_graph_2.json 1 \
  /tmp/cg-warp-validation/result \
  fiber:anon_20260822T114841824_000171.json \
  patch:band-seed3889268-20260729-161032-117.tifxyz 20
```

The empty patch catalog avoids indexing unrelated patches during fiber discovery;
the committed patches are embedded in the session. On Linux arm64,
RelWithDebInfo, one run per version, the old path froze all 119 accepted charts.
The corrected path deformed the six contacted patches and the nearby portion of
one accepted fiber; the other 112 charts remained exactly unchanged. On the
last patch, attachment RMS fell from 2.507 to 1.153 vx and maximum error from
15.192 to 7.148 vx; its largest UV correction was 39.107 vx. Source geometry was
identical, 7,307,352 distant output samples were unchanged, and the saved result
reopened exactly. Fitting took 23.2 s before and 26.1 s after on this shared host;
large patch corrections can still take tens of seconds.

`addedFiberDeformsCommittedPatch` fails on the previous frozen-patch fallback
and passes with the correction. It checks actual patch movement, preservation
of the new fiber's length, immutable sources, and distant reference vertices.
All 34 constraint-graph cases and the overlay, spiral-brush, and agent-bridge
contract suites pass.

### Prepared attachment measurements

On Linux arm64, the existing RelWithDebInfo build (`-O2 -g -DNDEBUG`), three
warm-cache repetitions of candidate 0 (`band-seed3879901…821`, 9 clicked
contacts) on `new_graph_2.json` gave:

| Path | Mean | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Previous workspace proposal/source/initial-placement/local-fit path | 1,576.06 ms | 1,559.33 ms | 1,579.69 ms | 1,589.16 ms |
| Prepared edit proposal/source/local-fit path | 97.35 ms | 95.26 ms | 96.18 ms | 100.60 ms |

Preparation took 111.89 ms once. Both fits report attachment RMS 0.135639 vx,
95% spacing ratio 1.03235, and 334 movable chart vertices / 352 surface
samples. The baseline command was `--local-proposal-benchmark` with the same
input and three repetitions, recorded before the restructuring. These are
RelWithDebInfo measurements, not Release or cross-platform claims.
They are diagnostic runs on a shared host, not isolated performance-lab results.

Ceres reports accounted for only about 8.6 ms of the old path. A 120-sample
GDB profile found 62 samples in full contact-multiset comparison/tree work;
the old path also rendered the entire initial placement before local fitting.
The prepared path eliminates those passes. Its phase trace spends roughly
15 ms proposing, 1 ms gathering the local problem, 4 ms fitting charts,
48–53 ms updating the surface, and 27 ms materializing final contacts.

Regression coverage compares prepared and legacy local fits, preserves distant
samples and immutable sources, checks incremental preparation against a full
cache rebuild, exercises manual contact overrides/removal and invalid edits,
checks snapshot isolation, and verifies exact surface save/reopen. The
constraint graph, overlay, surface-index, spiral-brush, and agent-bridge
contract test suites pass.

Fresh source construction produced a 91-fiber seed. Four successive valid
patch commits took 52.36, 272.97, 67.05, and 183.44 ms (mean 143.96, median
125.25 ms); incremental preparation after each commit took 0.94–1.53 ms.
Two intervening attempts at `band-seed11286360…611` requested an explicit refit
in 2.38 and 2.42 seconds. Those are rejections, not completed-fit speedups.
Every accepted surface round-tripped exactly.

A separate branch of that fresh graph adds `band-seed1121911…930`, then
`fiber:anon_20260816T200950449_000106.json` with 88 clicked contacts:

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 volume-cartographer/build/bin/constraint_graph_probe \
  --edit-proposal-benchmark /tmp/constraint-graph-fresh/growth.0.json 3 \
  /tmp/constraint-graph-fresh/bridge patch:band-seed1121911-20260729-203225-930.tifxyz
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 VC3D_PROFILE_CONSTRAINT_FIT=1 \
  volume-cartographer/build/bin/constraint_graph_probe \
  --edit-proposal-benchmark /tmp/constraint-graph-fresh/bridge.session.json 3 \
  /tmp/constraint-graph-fresh/fiber 0 patch:band-seed1121911-20260729-203225-930.tifxyz
```

The bridge patch took mean 187.10 ms (min/median/max 182.64/189.13/189.52).
The fiber took mean 487.12 ms (min/median/max 466.24/484.15/510.98), with all
three fits valid: RMS 0.092951 vx, spacing ratio 1.01898, 4,970 chart vertices,
and 4,979 surface samples. It added one fiber, and the saved surface reopened
exactly. This fresh-layout measurement is not directly comparable to the older
saved-layout fiber timings below.

An offscreen Qt app smoke test with `s1_ds2.zarr` and the fresh seed clicked a
visible proposal (`band-seed11523519…`): its source appeared at 53 ms and Commit
became enabled at 212 ms. Comparison loading left Commit enabled; 0/50/100%
comparison, restoration, commit, undo, and redo completed. CT tile loading is
asynchronous; these are control-availability timings, not complete CT-paint
timings. macOS and amd64 have not been run in this environment.

## Earlier implementation measurements

The following benchmark history records earlier implementations and objectives.
In particular, references to automatic background global fitting and eager
comparison-layer rendering do not describe the current workspace.

## Proposal preview benchmark

Build and test from the repository root (no dependency installation needed):

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build VC3D constraint_graph_probe test_constraint_graph test_constraint_graph_overlay -j 4
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build --output-on-failure -R '^(constraint_graph|constraint_graph_overlay|spiral_brush_patch|agent_bridge_contract)$'
volume-cartographer/build/bin/VC3D
```

The phase-timed probe loads a session, discovers the first seed-fiber proposal,
and repeats its initial attachment and full fit. Catalog/discovery startup is
outside the measured selection interval. An optional output prefix writes each
final graph for exact before/after comparison:

```sh
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe \
  --proposal-benchmark /tmp/constraint-graph-real/axis-fresh.json 3 /tmp/cg-seed-after
```

Linux arm64, RelWithDebInfo, three iterations per version, 91-fiber session,
`/mnt/bigpc/spiral_dataset_working/fibers`, local `graph_patches` folder. Selected
patch: `band-seed3879901-20260729-183612-821.tifxyz` (58 contacts).

| Selection-to-surface availability | Mean | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Before: waits for full joint fit | 19,221 ms | 18,770 ms | 19,237 ms | 19,657 ms |
| After: initial placement | 336 ms | 281 ms | 348 ms | 379 ms |

These are core surface preparation times, excluding CT tile loading and Qt
painting. The phase profile puts the original chart alignment/deformation near
10.4 seconds and shared-grid fitting near 8.8 seconds. Full solves remain in the
background (20.3–24.9 seconds in the after run, during concurrent build work);
this change improves time to inspect a proposal, not solver throughput. Cached
completed previews avoid a repeat solve. Exact final graph JSON matched in all
three before/after runs; RMS was 0.103449 vx and spacing p95 was 1.01829.
The temporary placement skips shared XYZ optimization and cannot be committed.
The final solver, tolerances, iteration limits, precision and acceptance checks
are unchanged. Regression tests cover unchanged final fit after an initial
placement, streamed exact contacts, toolbar dragging/clamping, and patch colors.

An isolated offscreen UI smoke test used the same 91-fiber graph with a one-patch
catalog and the real CT volume. Orange markers were visible before selection;
the rendered initial proposal appeared in 470 ms with Commit disabled. Dragging
moved the toolbar, all controls remained accessible across the two panes, and
Commit enabled after the joint fit. Reopening the completed proposal took 2 ms
and reused its validated fit. The full 95-node/catalog view was also inspected
for progressive markers and distinct patch outlines.

## Incremental fitting

`fitIncremental()` shares residuals, solver options, and validation with the
full solver. It starts with the changed patch/link footprint plus four grid
intervals (80 vx at the default spacing). Existing vertices outside that region
are constant; triangles, direction terms, and correspondences crossing the
boundary remain in the solve. Existing fiber row/column centers stay fixed.
Only added nodes receive the initial rigid alignment.

The neighborhood doubles when the last free ring moves more than one quarter
of the proximity tolerance (1 vx at the default), or a boundary edge changes
length by more than a factor of 1.02. These checks apply to both chart and
surface boundaries. New attachment errors must also meet the proximity limit.
The chart region keeps doubling while its boundary remains strained, and each
settled chart can expand its XYZ surface region independently, retaining the
same boundary and geometry checks (see *Local expansion instead of the global
fallback* below). Broad edits (over two thirds of source vertices), changed
global settings, and explicit **Refit graph** use the global solver.

The shared surface is rasterized and optimized only around changed triangles,
including their old footprint and a margin. A fixed outer ring supplies the
boundary. The result is copied into the accepted grid; unaffected XYZ samples
are retained exactly, including holes. Bounds can grow. The implementation still
scans chart triangles for validity/bounds and copies the output grid, but avoids
building or solving an optimization problem for distant geometry.

Local results can differ slightly from a global relaxation because their
surroundings are fixed. Solver precision, iteration limits, symmetric Dirichlet
terms, proximity settings, no-flip checks, and commit gating are unchanged.
An invalid proposal is rejected by the local path; the workspace then asks the
full solver for its verdict in the background (see below).

### Measurements

Linux arm64, RelWithDebInfo, three iterations each, same 91-fiber session and
58-contact patch as the proposal benchmark above. Commands from the repo root:

```sh
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe \
  --proposal-benchmark /tmp/constraint-graph-real/axis-fresh.json 3 /tmp/cg-local-baseline
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe \
  --local-proposal-benchmark /tmp/constraint-graph-real/axis-fresh.json 3 /tmp/cg-local-final
```

| Measured interval | Mean | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Global fitting | 19,413 ms | 19,100 ms | 19,120 ms | 20,020 ms |
| Local fitting | 140.4 ms | 139.3 ms | 140.1 ms | 141.9 ms |
| Global proposal including initial preview | 19,745 ms | 19,377 ms | 19,458 ms | 20,401 ms |
| Local proposal including initial preview | 549 ms | 531 ms | 559 ms | 559 ms |

Catalog discovery, CT loading/painting, and session I/O are outside these timed
intervals. The original phase profile was dominated by chart fitting (~10 s)
and shared-surface fitting (~9 s). Local fitting used 336 of 339,011 chart
vertices and 352 surface samples. Of 338,555 already accepted chart vertices,
338,412 remained bit-for-bit unchanged. RMS was 0.107859 vx (global 0.103449 vx),
and spacing p95 was 1.02052 (global 1.01829). The first local result's surface
also round-tripped exactly through save/reopen.

The probe accepts an optional candidate index after the output prefix. It logs
local vertex/sample counts, expansion attempts, and fallback phases. Tests cover
fixed distant charts and XYZ samples, overlapping patches with a manual match,
a strained attachment that expands successfully, global-settings fallback,
cancellation, exact surface restoration, and checksum failure recovery.

The final isolated offscreen UI run (91-fiber fixture, one-patch catalog, real
CT volume) displayed the initial proposal at 461 ms and enabled Commit after
the completed local fit at 659 ms. Markers remained visible, dragging worked,
and reopening the cached fit took 2 ms. Four regression suites passed. On the
95-node fixture, the previously invalid first proposal exhausted its local
regions and retained the global solver's inverted-quad rejection.

## Discovery sampling benchmark

Build and validate from the repository root:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build VC3D constraint_graph_probe test_constraint_graph test_constraint_graph_overlay -j 4
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build --output-on-failure -R '^(constraint_graph|constraint_graph_overlay|spiral_brush_patch|agent_bridge_contract)$'
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --discovery-benchmark /tmp/constraint-graph-real/axis-aligned-grown.json 3
volume-cartographer/build/bin/VC3D --constraint-graph-session /tmp/constraint-graph-real/axis-aligned-grown.json
```

The benchmark loads the session/catalog once, then alternates strides 1, 4 and 8
for three iterations in each direction. It searches the seed fiber and the
first committed patch separately. Timings exclude session/catalog loading and
rendering. The baseline is the original every-point scan (stride 1). Linux arm64,
GCC 13.3, RelWithDebInfo; real working fibers from
`/mnt/bigpc/spiral_dataset_working/fibers`, 765 fiber files, 84,312 patches from
`/home/sean/Desktop/spiral_dataset/graph_patches`, 95 committed nodes.

| Direction | Stride | Points checked | Mean ms | Min / median / max ms |
|---|---:|---:|---:|---:|
| Fiber → patches | 1 | 4,420 | 827.58 | 786.89 / 833.94 / 861.91 |
| Fiber → patches | 4 | 1,106 | 592.90 | 550.19 / 579.75 / 648.77 |
| Fiber → patches | 8 | 554 | 584.30 | 516.91 / 614.11 / 621.88 |
| Patch → fibers | 1 | 2,631,750 | 96.44 | 96.11 / 96.32 / 96.88 |
| Patch → fibers | 4 | 658,692 | 25.13 | 24.86 / 24.89 / 25.63 |
| Patch → fibers | 8 | 329,858 | 12.97 | 12.58 / 12.78 / 13.55 |

Stride 4 missed two patch IDs found by stride 1; stride 8 missed sixteen.
The selected patch had no new fiber contacts at any stride, so the reverse
timing measures rejection work. These are individual searches, not whole-workspace
startup timings. Sampling intentionally changes the contact set; precision,
proximity thresholds, contact clustering and solver tolerances are unchanged.

The probe also reports elapsed time at discovery phase boundaries. Hardware
profiling was unavailable (`perf_event_paranoid=4`); this uses steady-clock
instrumentation. In a separate three-iteration profile, patch loading/indexing
and exact contact verification dominated the forward search. Striding reduces
point queries but still requires loading nearby candidate patches, explaining
the smaller forward speedup: mean patch verification was 821 ms at stride 1
and 673 ms at stride 4, versus 62 ms and 17 ms for bounding-box queries.
The regression fixture checks both directions,
off-stride endpoints, retained contact coordinates, exact radius rejection,
default stride and invalid strides. All four suites above passed.
An isolated offscreen UI smoke test switched 4 → 8 → 4, verified persisted
settings and refreshed marks, then displayed a proposal at 465 ms and enabled
Commit after fitting at 651 ms. The user's running app was left untouched.

## Multi-fiber attachment validation

Use the existing build/test commands above. The read-only probe can list the
verified contacts for candidate patches, reproduce the old clicked-cluster-only
constraint set, and compare both fits over the same final contact locations:

```sh
volume-cartographer/build/bin/constraint_graph_probe --proposal-contacts /tmp/constraint-graph-real/axis-fresh.json 40
volume-cartographer/build/bin/constraint_graph_probe --clicked-proposal-benchmark /tmp/constraint-graph-real/axis-fresh.json 1 /tmp/cg-fit-multi-before 39
volume-cartographer/build/bin/constraint_graph_probe --local-proposal-benchmark /tmp/constraint-graph-real/axis-fresh.json 1 /tmp/cg-fit-multi-after 39
volume-cartographer/build/bin/constraint_graph_probe --compare-attachments /tmp/cg-fit-multi-before.0.json /tmp/cg-fit-multi-after.0.json
```

Linux arm64, GCC 13.3, RelWithDebInfo; the 91-fiber real-scroll session and
source catalog used above. Candidate 39 was
`band-seed11889625-20260729-195627-792.tifxyz`. Its clicked cluster supplied
8 constraints on one fiber. The new proposal supplied 510 contacts across
three fibers. Both fits passed surface validation. Errors below are chart
separation in graph voxels, evaluated at those same 510 contacts:

| Fiber filename suffix | Contacts | Before RMS | After RMS | Before max | After max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `204438006_000112.json` | 76 | 14.7164 | 0.2511 | 21.4991 | 0.9991 |
| `164827563_000384.json` | 217 | 45.8720 | 0.1648 | 80.9266 | 1.0718 |
| `173952029_000114.json` | 217 | 39.3877 | 0.1780 | 73.9118 | 1.1968 |

The spacing-ratio p95 was 1.02051 before and 1.01785 after. The new surface
round-tripped exactly through save/reopen. This is a correctness comparison,
not a speedup: the single observed proposal runs took 0.51 s before and 27.23 s
after, excluding catalog loading, CT rendering and source-layer preparation.
The added constraints required the existing global fallback. No solver
precision, iteration limits, proximity tolerance, or commit validation was
relaxed. Previously accepted patches retain their stored constraints; the
expanded contact gathering applies when proposing a new patch.

A separate real one-patch UI fixture displayed its initial placement at 499 ms
and enabled Commit after 1266 ms. An isolated offscreen VC3D instance sampled
`s1_ds2.zarr` and exercised 0%, 50%, and 100% opacity, followed by restoring the
joint surface. Comparison images were inspected in `/tmp/cg-fit-ui/`.
The four regression suites listed above passed. Added cases cover multiple
placed fibers, duplicate prevention, exact-radius rejection, distant chart
traversals, manual-pair overrides, unchanged source-layer placements, opacity
endpoints, true black CT pixels, and missing patch coverage. macOS and amd64
were not exercised in this session.


## Interaction performance: discovery and local fitting

Discovery now queries bounds in batches of 32 sampled points, then applies the
same individual box and exact surface tests. Up to four workers verify independent
patches; results, ties, errors, and streamed batches remain in catalog order.
Contiguous query indices are stored as runs. An approximately 64-MiB bounds cache
and 128-MiB contact cache reuse immutable source geometry across chart edits and
commits. Keys include source identity, radius, stride, and the applicable scale;
source replacement or reloading the catalog invalidates the relevant entries.
Caches hold weak source references and assume input folders remain unchanged.
Unsuccessful proximity queries no longer allocate a surface-info hash table.

Surface rasterization stores XYZ scratch values only for covered samples. It
retains first-hit ownership, patch precedence, arithmetic, validation, and Ceres
parameter/residual order. Local surface boundary expansion reuses a settled chart
instead of consuming another chart attempt. This can accept a local solution
where the previous implementation fell back to a global relaxation. Such results
can differ from the global fit; precision, solver iterations, tolerances,
no-flip checks, attachment checks, and Commit gating remain unchanged.

The chart solver uses Ceres's per-solve `SILENT` logging mode. Its no-flip
barrier deliberately rejects invalid trial steps, which otherwise produce
repeated "Step failed to evaluate" warnings while Ceres shrinks the trust region.
Unusable solutions and final geometry/keypoint validation failures still reach
the workspace as fit errors. This logging change preserves solver settings and
results; it does not reduce the number of retries or guarantee faster convergence.
The `incompatibleKeypointsStayRejected` and `rejectedChartStepsStayQuiet` tests
exercise a reflected keypoint pairing and capture the solver's stderr in a child
process to check both rejection and quiet logging.
Validation on Linux arm64, RelWithDebInfo: all seven suites below passed. A
single `--fit-benchmark /tmp/cg-horizontal-junctions/final.json 1 /tmp/result.json`
run per version took 20269.1 ms before and 20342.6 ms after (no speedup claimed).
The saved 92-node graphs were identical, apart from the output surface filename;
both surfaces had checksum `1ecfd2803732b6d2`.

### Reproducing the measurements

From the repository root, using the existing configured Ninja build:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build VC3D constraint_graph_probe test_constraint_graph -j 4
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build --output-on-failure -R '^(constraint_graph|constraint_graph_overlay|test_surface_patch_index|test_surface_patch_index_more|test_surface_patch_index_cache|spiral_brush_patch|agent_bridge_contract)$'
volume-cartographer/build/bin/VC3D --constraint-graph-session /path/to/session.json

AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --render-benchmark /tmp/cg-horizontal-junctions/final.json 10 /tmp/render-result.json
AGENTS_AGENT_MODE=1 VC3D_BENCHMARK_PATCH_FOLDER=/home/sean/Desktop/spiral_dataset/graph_patches volume-cartographer/build/bin/constraint_graph_probe --local-proposal-benchmark /tmp/cg-horizontal-junctions/final.json 3 /tmp/proposal-result 0
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --discovery-benchmark /tmp/cg-horizontal-junctions/final.json 4 all /tmp/discovery-result 4
```

The optional discovery arguments are selected node (`all` for the workspace),
contact-output prefix, and a single stride. Without the last argument it cycles
strides 1, 4, and 8. The contact files allow exact before/after comparisons.
`VC3D_BENCHMARK_PATCH_FOLDER` overrides catalog access only; it does not edit the
saved session or invalidate its surface checksum. The proposal benchmark accepts
a candidate index and then a selected node ID (`patch` chooses the first patch).
`--local-proposal-scan` uses the iteration count as a candidate count and cancels
at global fallback, allowing local behavior to be surveyed without full solves.
`--fit-benchmark` runs the full solver with the render benchmark's arguments.

Linux arm64, GCC 13.3, RelWithDebInfo. The fitted real session has 92 nodes
(91 fibers and one patch). Proposal 0 is
`band-seed3879901-20260729-183612-821.tifxyz`, with 58 clicked contacts. Times
exclude session/catalog loading, CT tile loading/painting, independent comparison
layers, and output-file serialization. These are core operation timings, not Qt
input-to-paint measurements.

| Operation | Runs | Mean ms | Min / median / max ms |
|---|---:|---:|---:|
| Initial surface, before | 10 | 464.9 | 453.1 / 463.3 / 491.2 |
| Initial surface, after | 10 | 214.5 | 210.7 / 213.4 / 221.6 |
| Proposal 0 through fitted surface, before | 3 | 864.4 | 856.1 / 859.9 / 877.2 |
| Proposal 0 through fitted surface, after | 3 | 516.1 | 512.8 / 516.8 / 518.8 |

Initial surfaces, proposal 0's final graph and surface, and a separate full refit
had identical before/after surface checksums. Proposal RMS was 0.159886 vx and
spacing-ratio p95 was 1.01695 in both versions. A 100-sample GDB profile of initial
surface construction attributed 56 samples to allocation/fill/copy/free, 16 to
map/set operations, and 28 to other rasterization/validation. Hardware perf events
were unavailable on this host.

A second real proposal, candidate 2
(`band-seed11488028-20260730-004516-912.tifxyz`, 35 clicked contacts), exhausted
its chart attempts and needed one more XYZ boundary expansion. Using the same
proposal command with final argument `2`:

| Candidate 2 outcome | Runs | Mean ms | Min / median / max ms |
|---|---:|---:|---:|
| Before: global fallback, rejected inverted patch quad | 3 | 34628.7 | 33181.5 / 34700.6 / 36004.0 |
| After: validated local fit | 3 | 1280.5 | 1227.4 / 1294.5 / 1319.5 |

These are different solutions, not an identical-output speedup. The accepted
local result used 2,877 chart vertices and 6,822 surface samples, with attachment
RMS 0.146257 vx and spacing-ratio p95 1.01699; its surface round-tripped exactly.
A 12-candidate local survey required global fallback for four candidates before
and three after. The remaining fallbacks are still necessary for these inputs.

Whole-workspace discovery used 23,946 real patches on
`/mnt/bigpc/spiral_dataset_working/verified_patches`, 765 source fiber files,
stride 4, and 120,734 placed-fiber sample queries. All **6,599 proposals and
112,019 contacts**, including exact coordinates, distances, and ordering, matched
the baseline output.

| Discovery state | Runs | Mean ms | Min / median / max ms |
|---|---:|---:|---:|
| Before, scan without application cache | 1 | 153703 | 153703 / 153703 / 153703 |
| After, first scan | 1 | 41753 | 41753 / 41753 / 41753 |
| After, cached scans | 3 | 194.8 | 189.0 / 197.1 / 198.1 |

The first exact proposal arrived at 628 ms on the first scan and 26–28 ms on
cached scans. Cold and warm rows are deliberately separate: network and OS-cache
variation affect cold reads. The initial folder/catalog load is outside these
intervals. A cold remote scan still takes tens of seconds to complete, while
verified marks are streamed for interaction. Large or invalid proposals and new
global layouts can still take seconds or require a global fallback.

Seven regression suites passed, including cache invalidation for changed sources,
radius, stride and scale, sampling across batch boundaries, streamed cancellation,
and independent surface expansion with fixed distant samples. An isolated
offscreen VC3D run opened the real 91-fiber session with `s1_ds2.zarr` and displayed
CT and proposal markers (`/tmp/cg-speed-ui.png`). macOS and amd64 were not exercised.


## Fit alignment and solver benchmark

A large attachment error first receives rigid registration. If that brings the
RMS within the existing proximity tolerance, fitting proceeds directly to the
axis-aligned deformation solve. Graphs with unresolved loop errors still receive
the initialization deformation pass. Triangle and correspondence Jacobians are
computed analytically; residuals, double precision, solver tolerances, single-thread
execution, distortion penalties and inverted-triangle rejection are unchanged.
The added endpoint constraints and completed fiber-to-patch contacts intentionally
change the fitted layout. Source coordinates and annotations remain immutable.

Build, test, and reproduce from the repository root:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build VC3D constraint_graph_probe test_constraint_graph test_constraint_graph_overlay -j 4
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build --output-on-failure -R '^constraint_graph(_overlay)?$'
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --fit-benchmark /home/sean/Desktop/patch_test.json 3 /tmp/patch-refitted.json
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --fit-benchmark /tmp/constraint-graph-real/axis-aligned-grown.json 3 /tmp/multi-refitted.json
volume-cartographer/build/bin/VC3D
```

Set `VC3D_PROFILE_CONSTRAINT_FIT=1` on the probe or VC3D to print Ceres timing,
iteration and convergence reports separately for rigid, chart and shared-surface
solves. This is opt-in and does not change solver settings.

Linux arm64, GCC 13.3, RelWithDebInfo, three full refits of each unchanged input
per version. Session loading and output serialization are outside the timer.
These measurements cover the core fit, not CT loading or UI painting.

| Input | Version | Mean ms | Min / median / max ms |
| --- | --- | ---: | ---: |
| `patch_test.json`: 91 fibers, one patch, 339,419 vertices | Before | 34,513.9 | 34,218.5 / 34,335.7 / 34,987.6 |
| Same | After | 13,536.1 | 13,284.3 / 13,439.1 / 13,884.8 |
| `axis-aligned-grown.json`: 92 fibers, eight patches, 370,742 vertices | Before | 26,509.8 | 26,062.6 / 26,635.3 / 26,831.4 |
| Same | After | 24,797.2 | 24,017.8 / 25,100.4 / 25,273.3 |

On `patch_test.json`, attachment RMS changed from 0.004657 to 0.005530 voxels
and spacing-ratio p95 from 1.01689 to 1.01960. Maximum horizontal endpoint tilt
fell from 1.10244 degrees to 0.000021 degrees; the real fibers retain their local
bends. On the eight-patch input, attachment RMS was 0.094687 before and 0.094692
after, with spacing-ratio p95 1.01503 and 1.01486. All six final fits passed the
existing geometry checks. The larger graph still takes about 25 seconds.

A 150-sample GDB profile of the baseline initialization found 39 samples in
sparse factorization stacks, nine in chart-metric autodiff, and 102 in other
Ceres/evaluation/allocation work. Hardware perf events were unavailable. Ceres
reports on the one-patch fit show the unnecessary initialization chart pass
costing about 12 seconds; the final chart solve now takes 4–5 seconds and the
shared XYZ solve about 8 seconds. The latter remains a substantial cost.
Analytic Jacobians were checked against automatic differentiation, including
reflected charts, partial Jacobian requests and rejected degenerate triangles.
An isolated analytic-only refit differed by at most 7.4e-8 voxels in UV from the
baseline due to floating-point derivative arithmetic; this is not a claim of
bit-identical solver output.

Regression coverage includes level fibers under dense two-fiber patch contacts,
local bend preservation, both directions of multi-source contact discovery,
manual-pair precedence, proximity/traversal rejection, cancellation, omitted
redundant deformation passes, exact saved-surface restoration and fixed distant
geometry on local edits. The new attachment and redundant-pass regressions fail
against the original library. Ubuntu arm64 was exercised; macOS and amd64 were
not tested in this run. Use **Refit graph**, inspect the preview, then **Commit**
to apply the new alignment to an existing saved graph.

A growth replay used `working.json.proposal.json.fiber-grown.json` from
`/tmp/constraint-graph-real`, removing only
`fiber:anon_20260816T200950449_000106.json` and its incident links to re-add it
against the two accepted patches. No source samples were synthesized. The
selected patch was `band-seed1121911-20260729-203225-930.tifxyz`, with 106 contacts.
One run per version took 36,567 ms before and 25,786 ms after through the fitted
surface, including local attempts and global fallback. Attachment RMS was
0.172782 / 0.172795 vx; spacing-ratio p95 was 1.01746 / 1.01628. This particular
fiber had contacts on only one of the two patches, so it validates growth but
is not evidence of improved multi-patch attachment accuracy; that missing-contact
case is covered by the regression fixture.

The replay also exposed saved-grid pruning against original ribbon normals
rather than the corrected normals used during fitting. Loading now preserves
the authenticated grid after its finite-value and source-support checks. The
round-trip probe checks exact grid equality, and the unit fixture covers strong
junction corrections whose normals differ from the source. Fit benchmarks with
an output path now also verify exact save/reopen outside the timed interval.

An isolated offscreen VC3D opened the before/after one-patch fits on the real
`s1_ds2.zarr` volume. Captures are `/tmp/cg-fit-speed/ui-before.png` and
`/tmp/cg-fit-speed/ui-after.png`. Display copies disable catalog discovery to
make the comparison repeatable; original sessions and input files are unchanged.


## Local expansion instead of the global fallback

Adding a large patch, or one attached to several fibers, used to take 25–70
seconds whenever the four local attempts failed and the incremental fit fell
back to the full solver. Profiling the full solver on the eight-patch session
(`axis-aligned-grown.json`, 370,742 vertices, RelWithDebInfo, one thread)
gave 24.6 s: the chart solve took 14.7 s over 27 iterations and the shared
surface solve 8.5 s over its 12 iterations. Of the 20.5 s spent in the sparse
normal Cholesky solver across those solves, 12.8 s was CHOLMOD factorization
(measured with an `LD_PRELOAD` timing shim). Ceres threads only parallelize
evaluation: eight threads brought the same refit from 24.6 s to 19.6 s, so no
thread count reaches the interactive budget. Threads remain opt-in through
`VC3D_CONSTRAINT_FIT_THREADS` for experiments; the default is unchanged.

The fallback was also not a refinement. Compared with the committed layout, a
full solve moved every existing chart vertex by a median of 22 vx (p99 71 vx)
on a session fitted by earlier solver code, and by a median of 0.7 vx
(p99 3.5 vx) when refitting a graph the current code had just fitted. The
chart objective has flat directions inside the fiber dead band, so the global
solver re-lays out the whole graph on every call. Benchmark sessions on disk
must therefore be refit once before incremental fits are measured against
them; on a stale committed layout the local boundary strain grows with the
free region instead of shrinking.

`fitIncremental()` now behaves as follows. Residuals, solver options, iteration
limits, tolerances, the proximity rule, the 1 vx / 2% boundary-settling checks
and the two-thirds broad-edit rule are unchanged, and a proposal that was
already placed near its contacts runs exactly the same first four attempts as
before.

- The chart margin keeps doubling from 80 vx (up to twelve doublings) until the
  boundary settles or the free region exceeds two thirds of the vertices, which
  still goes to the full solver. Surface regions expand the same way.
- An added chart whose contacts are more than 80 vx from their partners is
  rigidly placed before the free region is chosen. Re-adding a node from its
  unplaced source coordinates previously spent six attempts (about 12 s) growing
  a region toward the real location.
- A geometry rejection (for example *Patch geometry contains an inverted quad*)
  that repeats after the region has doubled is returned at once with
  `localChartVertices > 0`. A worst attachment error above the proximity radius
  that a doubled region does not improve by at least 25% is likewise returned
  as *Attachment error exceeds proximity tolerance*.
- Progress messages report the boundary movement and edge stretch of a
  strained attempt, and the worst attachment error of a rejected one.
- The workspace reports a local rejection immediately, then asks the full
  solver for its verdict on the same draft while the proposal stays selected.
  Selecting another proposal or cancelling stops that solve as before. The
  full solver's result replaces the local rejection when it arrives.

With `VC3D_PROFILE_CONSTRAINT_FIT=1`, an inverted-quad rejection also logs the
offending grid cell, the cosine between the fitted and source normals, the
fitted and rasterized edge lengths, and how far the surface solve moved the
sample. On the rejected real proposals that distance was 18–52 vx at a fidelity
weight of 0.25/vx: the rejections came from the shared-surface relaxation
inside a frozen ring, not from the chart. See *Inverted quads* below.

### Measurements

Linux arm64, GCC 13.3, RelWithDebInfo, `OMP_NUM_THREADS=1`, one run per case.
"Before" is the session-start solver built into a copy of the probe. Both
sessions were refit once with the current solver before measuring:

```sh
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --fit-benchmark /tmp/cg-horizontal-junctions/final.json 1 /tmp/cg/refit1.json
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --fit-benchmark /home/sean/Desktop/axis-aligned-grown.json 1 /tmp/cg/grown-refit.json
```

Proposal survey: the 92-node session, twelve real proposals from the local
`graph_patches` copy, timed from the draft to the incremental fit's verdict
(catalog discovery, initial placement, source layers and I/O excluded):

```sh
AGENTS_AGENT_MODE=1 VC3D_BENCHMARK_PATCH_FOLDER=/home/sean/Desktop/spiral_dataset/graph_patches \
  volume-cartographer/build/bin/constraint_graph_probe --local-proposal-benchmark /tmp/cg/refit1.json 1 /tmp/cg/candN N
```

| Candidate | Contacts | Before | Before outcome | After | After outcome |
| --- | ---: | ---: | --- | ---: | --- |
| 0 `band-seed3879901…821` | 58 | 528 ms | valid, local | 602 ms | valid, local |
| 1 same patch | 16 | 533 ms | valid, local | 808 ms | valid, local |
| 2 `band-seed11488028…912` | 35 | 27,456 ms | rejected after global: inverted quad | 2,679 ms | rejected locally: inverted quad |
| 3 same patch | 30 | 27,722 ms | rejected after global: inverted quad | 2,619 ms | rejected locally: inverted quad |
| 4 same patch | 34 | 27,440 ms | rejected after global: inverted quad | 2,462 ms | rejected locally: inverted quad |
| 5 same patch | 49 | 27,310 ms | rejected after global: inverted quad | 1,810 ms | valid, local (RMS 0.146 vx) |
| 6 same patch | 62 | 946 ms | valid, local | 1,088 ms | valid, local |
| 7 `band-seed11299011…179` | 30 | 468 ms | valid, local | 608 ms | valid, local |
| 8 same patch | 24 | 498 ms | valid, local | 582 ms | valid, local |
| 9 `band-seed1121911…930` | 112 | 518 ms | valid, local | 531 ms | valid, local |
| 10 same patch | 2 | 547 ms | valid, local | 538 ms | valid, local |
| 11 `band-seed11473808…488` | 207 | 644 ms | valid, local | 773 ms | valid, local |

Re-adding committed nodes: the 100-node eight-patch session. Each node is
removed with its links, the accepted surface is rebuilt without it, and the
node is re-added incrementally from its unplaced source chart. This covers the
large patches with 1,700–2,600 contacts on three partners:

```sh
AGENTS_AGENT_MODE=1 volume-cartographer/build/bin/constraint_graph_probe --readd-benchmark /tmp/cg/grown-refit.json 'patch:band-seed10980845-20260729-182209-499.tifxyz' 1 /tmp/cg/readded.json
```

| Node | Contacts | Before | Before outcome | After | After outcome |
| --- | ---: | ---: | --- | ---: | --- |
| `fiber:anon_…000106` | 106 | 26,064 ms | valid after global | 3,365 ms | rejected locally: inverted quad; the workspace's background full solve reproduces the earlier result |
| `band-seed10980845…499` | 1731 | 69,219 ms | rejected after global: inverted quad | 2,222 ms | rejected locally: attachment error |
| `band-seed11074660…159` | 2568 | 71,661 ms | rejected after global: inverted quad | 2,240 ms | rejected locally: attachment error |
| `band-seed1121911…930` | 243 | 823 ms | valid, local | 408 ms | valid, local |
| `band-seed11304273…287` | 2628 | 72,833 ms | rejected after global: inverted quad | 2,245 ms | rejected locally: attachment error |
| `band-seed1310793…956` | 42 | 1,582 ms | valid, local | 277 ms | valid, local |
| `band-seed3219408…892` | 38 | 22,519 ms | valid after global | 496 ms | valid, local |
| `band-seed3355050…635` | 513 | 17,890 ms | valid after global | 1,394 ms | valid, local |
| `band-seed3879901…821` | 58 | 299 ms | valid, local | 202 ms | valid, local |

Every accepted fit had attachment RMS 0.0947 vx and spacing-ratio p95 near
1.016 in both versions, and all nine accepted surfaces round-tripped exactly
through save/reopen. Valid results are local relaxations with fixed
surroundings; the full solver's re-layout of the same additions differs. The
one case that the full solver accepted and the local path rejects (the 106
contact fiber) still receives that acceptance from the workspace's background
solve, about 26 s later, instead of blocking the verdict on it. Rejected
proposals now report within about 2.7 s instead of 27–73 s; the full solver
rejected the same proposals.

Regression tests: `farReachingLocalEditKeepsExpandingInsteadOfRefittingEverything`
(a strip with 6% shorter contact spacing needs five doublings and must stay
local), `repeatedGeometryRejectionIsReportedWithoutTheGlobalSolver` (a folded
patch is rejected after two identical surface rejections without a full solve)
and `addedChartIsPlacedBeforeChoosingTheLocalRegion` (an unplaced chart is
enclosed by the first region). All 31 `constraint_graph` cases and the seven
related ctest suites pass. The workspace's background full solve after a local
rejection was compiled and follows the existing progress-callback path; it was
not exercised interactively in this session. macOS and amd64 were not tested.

### Fibers as the reference frame

Layouts drifted into a state where even small additions kept expanding their
local region. The mechanism: a fiber may bend anywhere within 40 vx of its row
or column at no cost, so inside that band only the chart metric resists a
patch's pull, and a patch has far more chart area near a contact than a
seven-row ribbon. The patch won, the fiber curved, and every later edit near
that fiber inherited the curvature; local commits also lock in up to 1 vx and
2% of residual strain at each frozen ring. In the twelve-proposal survey the
uniform weights moved committed fibers by 5–14.5 vx per accepted proposal.

In a local fit the chart-metric weights are now type dependent: committed
fiber triangles carry ten times the weight (a hundredfold in cost) and an
added patch's triangles half of it. Added fibers, committed patches, all link
weights, the dead band, the endpoint ties, rigid alignment, the surface solve,
proximity checks, no-flip barrier and commit gating are unchanged, and no
source XYZ changes: only where a patch lands in the flattened chart. The
defaults can be overridden for experiments with
`VC3D_CONSTRAINT_FIT_FIBER_WEIGHT` and `VC3D_CONSTRAINT_FIT_PATCH_WEIGHT`.

The full solver keeps its uniform weights. A variant that applied the fiber
weight everywhere (with fiber-to-fiber links scaled to match) was measured and
rejected: after a refit under it, five of the twelve surveyed proposals were
rejected with inverted patch quads that the local weighting accepts, and
re-adding the 106-contact fiber stopped at a constant 41.7 vx contact error
because a stiff fiber can only satisfy disagreeing contacts by translating,
which a local fit's frozen surroundings forbid. Consequently **Refit graph**
still relaxes fibers toward patches; the layouts it produces are the ones the
local weighting was measured against.

`VC3D_CONSTRAINT_FIT_LOG=1` echoes every workspace progress line with its
elapsed time to stderr, so the boundary-strain and attachment numbers can be
read after the fact; the status bar only shows the latest line.

#### Measurements with the fiber-reference weights

Same hardware, sessions and commands as above; "before" is the session-start
solver. Time is from the draft to the incremental fit's verdict. The last
column is how far any committed fiber vertex moved in the accepted fit.

| Proposal | Contacts | Before | After | Committed fiber moved (before → after) |
| --- | ---: | ---: | ---: | --- |
| 0 `band-seed3879901…821` | 58 | 528 ms valid | 406 ms valid | 8.9 → 0.6 vx |
| 1 same patch | 16 | 533 ms valid | 420 ms valid | 8.9 → 0.6 vx |
| 2 `band-seed11488028…912` | 35 | 27,456 ms rejected | 521 ms valid | 14.5 → 1.8 vx |
| 3 same patch | 30 | 27,722 ms rejected | 474 ms valid | 14.5 → 1.8 vx |
| 4 same patch | 34 | 27,440 ms rejected | 477 ms valid | 14.5 → 1.8 vx |
| 5 same patch | 49 | 27,310 ms rejected | 534 ms valid | 14.4 → 1.8 vx |
| 6 same patch | 62 | 946 ms valid | 495 ms valid | 11.4 → 1.8 vx |
| 7 `band-seed11299011…179` | 30 | 468 ms valid | 394 ms valid | 8.5 → 0.1 vx |
| 8 same patch | 24 | 498 ms valid | 402 ms valid | 8.5 → 0.1 vx |
| 9 `band-seed1121911…930` | 112 | 518 ms valid | 554 ms valid | 5.2 → 0.8 vx |
| 10 same patch | 2 | 547 ms valid | 531 ms valid | 5.2 → 0.8 vx |
| 11 `band-seed11473808…488` | 207 | 644 ms valid | 450 ms valid | 10.4 → 0.4 vx |

The "before" movement column is the earlier local-only version without the
weights (same sessions); the uniform-weight solver moved committed fibers by
5–14.5 vx per accepted proposal, the weighted one by at most 1.8 vx. All twelve
surfaces round-tripped exactly.

| Re-added node | Contacts | Before | After |
| --- | ---: | ---: | --- |
| `fiber:anon_…000106` | 106 | 26,064 ms valid after global | 3,235 ms rejected locally (inverted quad); the background full solve then reproduces the earlier result |
| `band-seed10980845…499` | 1731 | 69,219 ms rejected | 2,173 ms rejected (attachment error) |
| `band-seed11074660…159` | 2568 | 71,661 ms rejected | 2,299 ms rejected (attachment error) |
| `band-seed1121911…930` | 243 | 823 ms valid | 500 ms valid |
| `band-seed11304273…287` | 2628 | 72,833 ms rejected | 2,201 ms rejected (attachment error) |
| `band-seed1310793…956` | 42 | 1,582 ms valid | 1,420 ms valid |
| `band-seed3219408…892` | 38 | 22,519 ms valid after global | 362 ms valid |
| `band-seed3355050…635` | 513 | 17,890 ms valid after global | 886 ms valid |
| `band-seed3879901…821` | 58 | 299 ms valid | 194 ms valid |

On this session, whose fibers were bent by patches under the uniform solver,
re-adding a patch lets the stiff fibers inside the free region straighten by up
to 12 vx relative to the committed layout, which costs a few more doublings
(1.4 s instead of 0.3 s for the 42-contact patch). Layouts committed with the
weighted local fits do not carry that curvature.

Regression coverage: `localAdditionPreservesDistantGeometry` now checks that
contacts disagreeing with the fiber spacing stretch the added patch while the
fiber moves under 0.5 vx, and
`farReachingLocalEditKeepsExpandingInsteadOfRefittingEverything` drives its
far-reaching strain through a committed patch strip pinned to the fiber at
both ends. All 32 `constraint_graph` cases pass three times in a row; the
seven related ctest suites pass.

### Inverted quads

"Patch geometry contains an inverted quad" was the slowest and most frequent
rejection. It fires after the shared-surface relaxation when a grid triangle's
normal opposes the source normals at its three samples and all three belong to
one patch with no fiber underneath. In every rejected real case the rasterized
triangle was fine (20 by 20 vx edges) and the relaxation had moved a sample by
18–52 vx, once by 237 vx. Two causes: a local fit re-rasterizes its interior
from raw source positions while the frozen ring keeps committed samples that an
earlier relaxation had already moved by up to 20 vx, so the grid starts torn
along that seam; and a patch chart sheared by its contacts maps a grid step onto
a stretched piece of source, leaving the rasterized triangle nearly edge-on.

Two changes:

- In a local fit, free samples now start where the accepted surface already is
  when it has a sample there, and only newly covered samples start at raw
  raster positions. The fidelity term still pulls every free sample toward its
  new rasterized position, so the target is unchanged; only the starting point
  of the non-converging 12-iteration relaxation is. Full solves and saved
  surfaces are unaffected.
- A fold that appears only after relaxation is pruned like a fiber-supported or
  mixed-owner cell, counted in "unsupported boundary samples omitted", instead
  of rejecting the fit. A patch whose *rasterized* triangle already folds is
  still rejected with the same message, since that fold comes from the source
  or its chart. This applies to local and full solves alike.

Same sessions and commands as above, one run per case:

| Case | Contacts | Before | After |
| --- | ---: | ---: | --- |
| 12 surveyed proposals | 2–207 | 8 valid in 0.5–0.9 s, 4 rejected after 27 s | 12 valid in 0.39–0.55 s |
| `fiber:anon_…000106` re-add | 106 | 26.1 s valid after global | 1.57 s valid, local |
| `band-seed3219408…892` re-add | 38 | 22.5 s valid after global | 0.37 s valid, local |
| `band-seed3355050…635` re-add | 513 | 17.9 s valid after global | 0.91 s valid, local |
| Three 1,700–2,600-contact patches | | 69–73 s rejected | 2.2–2.3 s rejected (attachment error) |
| Other four re-adds | 42–243 | 0.3–1.6 s valid | 0.18–1.44 s valid |

No inverted-quad rejection occurred in any of the 21 cases. Accepted fits have
attachment RMS 0.09–0.23 vx and spacing-ratio p95 1.015–1.016; all twelve saved
surfaces round-tripped exactly. The three remaining rejections are patches whose
contacts disagree with the accepted fibers by 41–509 vx on their worst link, and
the full solver rejected them too. `foldedPatchCellsArePrunedInsteadOfRejectingTheFit`
covers the pruning on a synthetic folded patch; all 32 `constraint_graph` cases
and the seven related ctest suites pass.

## Catalog startup with large patch folders

Indexing a patch folder read one `meta.json` after another on the calling
thread: with 86,000 patch folders on a network share that is 86,000 serial
round trips, about ten minutes before the workspace appeared. The scan now runs
on a thread pool (up to 32 workers, in batches of 4,096 folders) and assembles
entries and warnings in the same sorted folder order, so the catalog, the
bounds index, candidate ordering and warning messages are unchanged. Progress
reports batches instead of every thousand folders. On the local 84,312-folder
copy with a warm cache the probe's session-plus-catalog load went from 10.1 s to
8.0 s; the remaining time is session and fiber loading. The 23,946-folder
network share (`/mnt/bigpc/…/verified_patches`) loaded in 20.4 s including the
session, measured once with the new code only.
