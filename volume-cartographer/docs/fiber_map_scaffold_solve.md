# Scaffold winding solve (`vc_fiber_graph_solve`)

Integer winding assignment for a manually annotated scaffold, with the
residual analysis the fiber map's approval workflow ranks by. The library is
`vc_fiber_graph_solve` (`vc/fiber_graph/`); `vc_scaffold_solve` is its command
line, and VC3D calls the same functions on a worker thread. Nothing in it is
Qt, and nothing in it is Python: one solve, two callers.

## What a scaffold is here

The annotated fibers of one vcobj store, their `fiber_branch` links, their
piece approvals, and the volume's umbilicus:

- fibers come from the store's `fiber` objects in UUID order, which is the
  solve's node order. A fiber's family is its record's `family`
  (`manual_family` selects which labels count, `family_source`); untagged
  fibers carry no winding relation and are left out;
- links carry their two anchor positions, `meta.adjacent`, `meta.auto` and
  their record approval state;
- piece approvals come from `ext_fiber_piece_approval`, as the live rows and
  the spans they currently cover (per geometry/normals aspect);
- the umbilicus is the one mapping to the frame the fibers are annotated in,
  and the scroll's winding handedness is read from its `winding_handedness`
  meta. A store that states the handedness nowhere is an error: a mirrored
  solve must not happen silently.

Coordinates are voxels of the annotation frame throughout. The loader maps
every fiber to the requested target volume (by default the one the fibers
declare); fibers declaring different volumes are an error.

## Constraints

Every constraint asserts an integer winding step between two fibers at two
anchors. A link's anchor-level assertion is 0 for an ordinary link and one
winding for an adjacent link (the V fiber inside the H fiber, so
`W_V - W_H = -1`, from the stored H/V tags only). The whole-fiber step adds
the seam and connector transport, exactly as the automatic measured
constraints do:

    target = step + connector(x_a, x_b) - turns_b(arc_b) + turns_a(arc_a)

Anchors are projected onto their fiber's polyline; an anchor farther than
`links.max_anchor_distance` (default 5 voxels) fails the run, as a
coordinate-frame guard rather than a tolerance to widen. Identical
`(a, b, target)` relations are aggregated, their weights summed.

### Measured constraints

A crossing of two annotated fibers asserts a winding step too, and the
library states those exactly as the automatic pipeline does
(`windingEventRelation`, shared with `prepareFiberWindingGraph`):

- **normals** (`ScaffoldNormals.hpp`). Each fiber gets a strip normal from its
  own geometry (`solveFiberStripNormals`), which is an axis without a sign.
  The annotation fixes the sign where it states one, in this precedence: a
  control point's `display_normal` with source `manual`; the display normals
  of the spans approved for the normals aspect; an adjacent link, whose two
  anchors say which way is inward (the V fiber lies one winding inside the H
  fiber, so both inward normals point from the H anchor towards the V one).
  The fiber format states display normals outward, so the inward target is
  their negation; the adjacent-link anchor assumes no convention at all.
- **crossings** (`ScaffoldConstraints.hpp`). `extractFiberTraceConstraints`
  runs on the annotation's own polylines, which stay whole (no piece
  splitting). The geometry decides whether a crossing is parallel or
  perpendicular; a crossing whose verdict disagrees with the two annotated
  families is dropped and counted, never forced. The winding separation along
  each connector comes from a Lasagna normal volume's `grad_mag` density
  through a sampler the caller supplies; without one every relation keeps a
  NaN separation, which asserts no arc but still witnesses an enclosure.
- **signs**. The fibers no anchor fixed are decided by the perpendicular
  crossings, with the native sign solve (`vc/fiber_tracer/FiberNormalSign.hpp`,
  `irls` or `ceres`), from the radial prior of each fiber's own normals. An
  anchored fiber's sign is held. The run records how many signs each source
  decided, and how many anchored fibers the crossings disagreed with.
- **relations**. A perpendicular crossing asserts the rounded separation plus
  the half winding between the two families, a parallel one the separation
  itself; a fractional separation asserts the two integers around it with the
  weight split between them. The signed order of a perpendicular crossing also
  bounds the pair's H->V step from one side, and a pair bounded from both
  sides that agree fixes its step exactly: the *enclosure* relation. Weights
  are `weights.perpendicular`, `weights.parallel` and `weights.enclosure`.

Every kind lands in the same constraint list, so the solve, the contraction of
approved constraints and the residual analysis treat a crossing exactly like a
link.

## Solve

`solveWindingMinCostFlow` (LEMON, `network_simplex` by default, capacity scale
1000) minimizes `sum_e weight_e * |k_b - k_a - target_e|` exactly, one
connected component at a time, each gauged to minimum winding 0. Fibers
without a constraint keep their own component and winding 0.

With `fix_approved_subgraphs` (default on) the approved constraints are solved
alone first. If they contradict each other the run stops and names the pairs,
so one of them can be revoked; otherwise each approved subgraph is contracted
into one node with per-member offsets, and the remaining arcs are rewritten
`k_S(b) - k_S(a) = target + offset_a - offset_b`. Approved relations are
therefore held exactly, whatever the rest of the evidence says.

## Residual analysis

A violated constraint on its own does not say which annotation is wrong: the
winding is overdetermined only along cycles. For each violated constraint the
analysis finds

- the shortest cycle through it over the *satisfied* constraints (fewest hops,
  ties by the total length of the fiber stretches the cycle runs along);
- the pieces of that cycle: for every fiber on it, the stretch between the
  constraint that enters the fiber and the one that leaves it, with its arc
  range and the control-point range covering it;
- the redundancy: how many edge-disjoint alternative paths exist between the
  two fibers, up to `max_alternative_paths` (default 4).

`cycle_hops` is the smallest cycle the violation lies on, `cost` is
`weight * residual`, and `joint` is `cost / cycle_hops`. Each violated
constraint's cost is distributed over its cycle's pieces in proportion to
`1 / max(length, 1 voxel)` — a short stretch is a more specific suspect than a
long one — and the per-piece sums are the piece ranking.

## Command line

```sh
vc_scaffold_solve config.json
```

| Key | Default | Meaning |
|---|---|---|
| `store` | required | the scaffold store |
| `volume`, `scale` | the fibers' declared frame, 1 | target frame |
| `handedness` | the umbilicus meta | `+1` or `-1`, overrides the store |
| `family_source` | `any` | `any`, `manual` or `automatic` record families |
| `links.approval_states` | `["approved"]` | link record states to use |
| `links.include_auto` | `false` | also use `meta.auto` links |
| `links.max_anchor_distance` | `5` | voxels, a hard guard |
| `weights.manual` | `1` | weight per manual link arc |
| `constraints.enabled` | `false` | also measure the scaffold's own crossings |
| `constraints.resample_spacing`, `.maximum_distance`, `.tangent_window` | `32`, `128`, `32` | voxels, the extraction's geometry |
| `constraints.relax_endpoint_certification` | `false` | accept crossings near a hand-drawn fiber's ends |
| `constraints.winding_distances` | `true` | measure each connector's separation |
| `constraints.winding_step` | `8` | voxels per integration step |
| `constraints.maximum_winding_distance`, `.enforce_maximum_winding` | `1.5`, `false` | turns, and whether to reject beyond it |
| `constraints.perpendicular_offset`, `.winding_scale`, `.enclosures` | `0.5`, `1`, `true` | relation conventions |
| `weights.perpendicular`, `.parallel`, `.enclosure` | `1`, `1`, `1` | weight per measured arc |
| `winding.manifest`, `winding.remote_cache` | — | the Lasagna normal dataset the separations are measured on |
| `normals.twist_weight`, `.lateral_weight`, `.smoothing_sigma`, `.resample_spacing` | `100`, `1`, `32`, `32` | the strip solve |
| `normals.manual_display_normals`, `.approved_pieces`, `.adjacent_links` | `true` | which anchors fix a sign |
| `sign.solve`, `sign.backend` | `true`, `irls` | decide unanchored signs; `irls` or `ceres` |
| `include_untagged` | `false` | load the fibers without an H/V tag too |
| `lemon.backend`, `lemon.capacity_scale` | `network_simplex`, `1000` | flow backend |
| `fix_approved_subgraphs` | `true` | contract approved subgraphs |
| `max_alternative_paths` | `4` | redundancy cap |
| `workers` | `1` | geometry packing threads |
| `output`, `windings_output` | — | JSON summary / per-fiber windings |

The summary holds the input and link counts, the solve metrics, the
quantization report, the ranked residuals with their cycles and pieces, and
the piece ranking.

## Runs in the store

`persist_run` (CLI) appends the solve to the store it solved, as extension rows
of the existing `ext_fiber_graph_solve` family: the run and its scalar settings
(`_runs`), the solved store state (`_inputs`), every fiber's winding, component
and family (`_fibers`), the run metrics (`_metrics`), the ranked violated
constraints (`_residuals`) and the cycle stretches they accuse
(`_residual_pieces`). There is no result store: a scaffold solve is recorded
where the scaffold lives, and the fiber map's solves list is the set of run
rows. `scaffoldSolveRunRows` builds the rows and the caller submits them, so
VC3D writes them through its own checked request.

`readScaffoldRuns`, `readScaffoldResiduals` and `readScaffoldWindings` read
them back through any StoreReader; Python reads the same rows through
`CollectionReader.extension_rows` and the `residuals` / `residual_pieces`
table roles.

## Approving a measured constraint

An approved constraint is held rigid: the next run solves the approved
constraints alone first and contracts each approved subgraph into one node, so
an approved relation is never traded away (and approved relations that
contradict each other stop the run instead of being averaged). A link's
approval is its record's review state. A measured crossing has no record, so
its approval is a row of `ext_fiber_constraint_approval`, keyed by the two
fibers, the kind and the two anchors quantized to 64 base voxels, and carrying
the piece-approval fingerprints of the two covering spans. `applyScaffoldConstraintApprovals`
marks the constraints a live row covers, and skips a row whose spans have been
edited since: the annotation moved under it, so the decision is no longer
about what is there now. Re-running the extraction does not lose an approval -
the crossing lands in the same quantum - but editing either fiber's covering
span does.

## In the fiber map

Three docks (`FiberMapScaffoldSolve.hpp`, with the Qt-free ranking in
`FiberMapScaffoldRanking.hpp`):

- **Graph solver.** Which links a run uses (by review state, and whether
  automatic links count), the link weight, the LEMON backend and *fix approved
  subgraphs* (on by default). *Measure the fibers' crossings* adds the measured
  stages: the per-type weights, the sign backend, and whether the winding
  separations are measured at all. The winding density they are measured on is
  the package's own selected Lasagna dataset, the one the annotation optimizes
  against; a run that would measure separations without a bound dataset, or on
  a dataset that states no `grad_mag` density, is refused before it starts
  rather than quietly measuring none. The solve runs in the frame the map was
  built in and on its own worker; the store's umbilicus states the winding
  handedness, and only when it does not does the dock offer the catalog's sense
  as a fallback. The run is saved in the store, and its one-line report lands in
  the map's status line.
- **Solves.** Every run saved in the store, newest first, with its method,
  fibers, components, weighted L1, violated constraints and largest residual.
  Selecting one loads its ranking.
- **Fiber and piece approval.** The selected run's violated constraints and the
  stretches they accuse, in one table, ordered by *smallest cycle* (then the
  shorter stretch), *highest cost* (then the shorter cycle and stretch) or
  *cost per hop*. Selecting a row selects its fibers on the map and draws the
  stretch; double-clicking a stretch opens it as an approval piece between the
  control points that bracket it, which is where its geometry and normals are
  approved. Two switches outline what is already approved: every span with a
  current geometry approval, the spans whose normals are approved too, and the
  fibers approved from end to end. Selecting a *measured* constraint enables
  **Approve constraint**, which records that this crossing is right
  (`ext_fiber_constraint_approval`); selecting one already approved offers to
  revoke it instead. A link is never approved there: a link's approval is its
  review state in the annotation.

The approval overlays are halos behind the fibers rather than a change of the
fibers' own colour, so what the map says about a fiber's geometry stays what it
always said.

### Laying the map out from a solve

*Lay the map out from this solve* (in the solves dock) makes the map walk
exactly the relations the selected run walked: a link that run did not use
(its review state, or an automatic link it excluded) carries no constraint in
the layout and is not marked suspect for it, and a link it used and could not
satisfy carries none either and is drawn suspect, which is what the run said
about it. The map's own crossing evidence still places the fibers the run says
nothing about, so the map stays a map. Which solve is followed is part of the
layout's input digest, so selecting another run rebuilds rather than reusing a
memoized layout.

## Measured parity

On the PHerc0332 scaffold store (`objects-v22.vcobj`, 293 fiber records, 292
tagged H/V fibers, 281 links, 851,285 line points, AMD EPYC 7J13 CPU, one
thread for the solve) the native run reproduces the reference Python stage's
LEMON run (`vc_wgraph winding lemon-scaffold`) exactly:

| Links used | Setting | Arcs | Components | Weighted L1 (turns) | Max residual (turns) |
|---|---|---|---|---|---|
| 232 | approved and pending, no automatic | 232 | 68 | 0 | 0 |
| 13 | approved only | 13 | 279 | 0 | 0 |

Both runs' link reports match the reference counts (209 ordinary, 23 adjacent,
49 automatic skipped for the first; 12 ordinary, 1 adjacent, 268 skipped by
approval state for the second), as do the component partition and the
per-fiber windings up to the per-component gauge each solve is free to choose.
Wall time, single run: 1.02 s reading the store, 0.024 s packing the geometry,
0.011 s for the solve and residual analysis.

The measured stages reproduce the reference extraction on the same store. With
every fiber record loaded (293 fibers, 852,536 line points) and geometry strip
normals, the native run finds 10,208 crossings — 8,315 parallel and 1,893
perpendicular — which is what the reference Python extraction
(`vc_wgraph extract`, `normal_source: geometry`) reports for the same store.
Loading only the 292 H/V-tagged fibers leaves 10,195 crossings (8,302 parallel,
1,893 perpendicular).

A full variant A run on that store (292 tagged fibers, approved and pending
links, winding separations from the PHerc0332 Lasagna normal volume, 32
workers on an AMD EPYC 7J13 CPU):

| Stage | Result |
|---|---|
| normals | 292 of 292 fibers solved, 46 anchored by the 23 adjacent links |
| crossings | 10,195 crossings, all with a measured separation |
| signs | 46 held, 228 decided by crossings, 18 left undecided; 26.9 % of the 353 strict crossings (axes within 30 degrees, less than one turn apart) disagree with the decided signs |
| relations | 6,993 crossings assert an arc (11,796 after the fractional split), 786 dropped where the geometry's verdict disagrees with the annotated families; 747 enclosure pairs give 51 exact relations, 652 one-sided, 26 non-unique, 18 contradictory |
| solve | 12,079 constraints over 1,969 aggregated arcs, 15 components, weighted L1 2,942.8 turns, 5,724 violated constraints, largest residual 9 turns |

Wall time of that run: 1.05 s reading the store, 0.02 s packing the geometry,
0.98 s for the strip normals, 0.37 s for the crossings, 0.03 s for the sign
solve, 0.01 s assembling the relations and 1.74 s for the LEMON solve with its
residual analysis.

Those residuals are a statement about the *evidence*, not a verdict: a
scaffold's geometry-only normals are weak evidence, as the strict sign
disagreement of 26.9 % says, and the measured relations contradict the manual
links accordingly. The 51 exact enclosure relations are the strongest of the
measured kinds — they are integer bounds from both sides rather than a rounded
measurement — and adding them alone to the 232 link constraints leaves 3
violated constraints with a weighted L1 of 5 turns, which is what the approval
dock is there to adjudicate.
