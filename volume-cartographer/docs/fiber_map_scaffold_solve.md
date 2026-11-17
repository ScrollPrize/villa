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

Measured perpendicular/parallel/enclosure constraints are not part of this
step; the constraint kind is already in the result rows so they can join
without changing the readers.

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

## In the fiber map

Three docks (`FiberMapScaffoldSolve.hpp`, with the Qt-free ranking in
`FiberMapScaffoldRanking.hpp`):

- **Graph solver.** Which links a run uses (by review state, and whether
  automatic links count), the link weight, the LEMON backend and *fix approved
  subgraphs* (on by default). The solve runs in the frame the map was built in
  and on its own worker; the store's umbilicus states the winding handedness,
  and only when it does not does the dock offer the catalog's sense as a
  fallback. The run is saved in the store, and its one-line report lands in the
  map's status line.
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
  fibers approved from end to end.

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
