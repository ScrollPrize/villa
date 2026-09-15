# Constraint Graph workspace

Open **Constraint Graph** in VC3D with a CT volume loaded. Select a fiber JSON
folder, a patch tifxyz folder, and the corresponding Lasagna normals manifest.
Set the coordinate scales before **Load folders / New graph**. The largest
linked fiber component is selected first; the component dropdown can start a
new graph from another component.

Fibers are trimmed to their first and last controls and sampled at uniform
20-voxel arclength intervals. Sheet-tangent ribbons use the same surface builder
as Line Annotation. The output is a dynamically sized, strict-quad tifxyz grid.
Missing support remains invalid rather than connecting unrelated boundaries.

## Reviewing additions

- Orange marks identify attachment locations within **Proximity (vx)**, initially
  4 voxels in graph coordinates. Marks appear progressively as exact proximity
  checks finish and remain visible while selecting or fitting proposals.
  Click a marked location to preview a patch; the selection uses cached discovery.
- Drag the **Drag to move** handle on the selection bar to place it anywhere in
  the workspace. It stays there through selection and camera changes.
- Patches have distinct outline colors, shared between the graph and source
  panes. Selecting a patch thickens its outline without changing its color.
- Use **Tab / Shift+Tab** or the arrows in the selection bar to cycle proposals.
- The main pane renders the proposed joint placement from the selected volume.
  An initial attachment appears while the full joint fit runs; **Commit** stays
  disabled until fitting and geometry validation succeed. The initial attachment
  can move during fitting. The last three completed previews are cached until
  the committed graph or its settings change.
  The second pane renders the proposed source separately. **Show committed**
  temporarily restores the accepted graph in the main pane.
- To adjust alignment, choose **Add keypoint pair**, click a recognizable feature
  in the proposal pane, then its match in the main pane. Add another pair to
  constrain rotation. User pairs replace automatic guesses between that node
  pair. The list supports removing and replacing pairs.
- **Commit** accepts a valid fit. **Cancel / Escape** discards the proposal.
  Undo and redo operate on committed edits.
- Selecting a committed patch finds nearby unplaced fibers. **Include linked
  component** adds all fibers linked to the proposed fiber.

Ordinary additions refit the patch, overlapping neighbors, and nearby sections
of fibers. Distant accepted chart vertices and surface samples remain fixed.
Real 3D patch overlaps add correspondence constraints; symmetric Dirichlet terms
penalize stretching and compression in both the chart and output grid. Invalid fits cannot be committed. Spacing ratio
and attachment RMS in the status area describe fit quality; they are not a
guarantee that the selected patches follow the correct sheet. Inspect fiber
continuity in the rendered volume before committing.

Horizontal fibers share a fitted row and vertical fibers share a fitted column.
Each fiber can bend within two grid spacings (40 voxels) on either side of its
row/column without an alignment penalty; larger departures are penalized softly.
This applies during rigid alignment and joint deformation, so a small local
tilt cannot accumulate into large drift across the graph. Use **Refit graph**
to preview this alignment on an existing session, then **Commit** or **Cancel**.

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
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build VC3D test_constraint_graph test_constraint_graph_overlay constraint_graph_probe -j 4
QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build --output-on-failure -R '^constraint_graph(_overlay)?$'
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

Regression tests use temporary synthetic fiber/patch fixtures and cover
trimming, arclength mapping, chart handedness, isotropy, real surface proximity,
patch overlaps, component growth, paired registration, cancellation, malformed
sessions, and save/reopen.

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
There are at most four local attempts. Broad edits (over two thirds of source
vertices), unresolved boundary strain/validation failures, changed global
settings, and explicit **Refit graph** use the global solver.

The shared surface is rasterized and optimized only around changed triangles,
including their old footprint and a margin. A fixed outer ring supplies the
boundary. The result is copied into the accepted grid; unaffected XYZ samples
are retained exactly, including holes. Bounds can grow. The implementation still
scans chart triangles for validity/bounds and copies the output grid, but avoids
building or solving an optimization problem for distant geometry.

Local results can differ slightly from a global relaxation because their
surroundings are fixed. Solver precision, iteration limits, symmetric Dirichlet
terms, proximity settings, no-flip checks, and commit gating are unchanged.
An invalid proposal can still require a slow global fallback before rejection.

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
