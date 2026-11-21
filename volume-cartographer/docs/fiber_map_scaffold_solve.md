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

Only a fiber's annotated line is read. A manual fiber (two or more control
points) is fiber between its first and last control point; the line points
the optimizer extrapolates past them (the open tails) are not, and the solve
never sees them: no normal, prediction sample, crossing, link anchor,
enclosure or residual lies on a tail, and arc 0 of a fiber is its first
control. Automatic traces (fewer than two controls) have no tails. The one
rule is `vc::geometry::annotatedLineRange` (vc_object_store
`typed-geometry.md`); the dock's control arcs and the cache keys
(`line annotated` in `inputs`) follow it, so rows and runs solved on whole
lines no longer match and are solved again.

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

### Links are refined onto their encounter

A link is one statement about one encounter of two fibers, and so is a
measured crossing. Before a link constrains anything it is brought onto that
encounter by the crossing extraction's own refinement, seeded at the link's two
anchors (`refineFiberCrossing`, `vc/fiber_tracer/FiberTraceConstraints.hpp`),
and it has to pass the extraction's acceptance rules. This is not optional:

- the two fibers must meet at the angle their families state - perpendicular
  for an H and a V fiber, parallel for two of the same family, within the
  extraction's 30 degree tolerance, at the encounter and at its probes;
- both fibers must continue far enough on both sides of the encounter to
  certify it (the extraction's endpoint support; a link at a fiber's last
  point fails it);
- the two refined points may lie at most `links.refine_maximum_distance`
  (default 512 voxels) apart.

A link with an anchor on a tail - past its fiber's first or last control
point - asserts nothing either (`links_rejected_tail`): a tail is not fiber,
whoever placed the link there.

**Gap spans** are treated like tails. A span between two control points tagged
`break` (the span's `gap` tag) bridges damaged papyrus; it is not fiber. No
crossing on a gap span, or closer to one than a crossing needs fiber on both
sides (32 voxels for a perpendicular, 80 for a parallel one - the extraction's
own end rule), is kept (`crossings_rejected_gap`), and a link whose anchor or
refined point lies there asserts nothing (`links_rejected_gap`). A gap at a
fiber's end is simply part of what is ignored.

A link that fails asserts nothing. The run counts them by rule
(`links_rejected_angle`, `links_rejected_end_support`,
`links_rejected_distance`, `links_rejected_consistency`) and records the
farthest a refinement moved an anchor (`link_refinement_max_shift`). An
accepted link is a constraint of the geometry's own kind, `perpendicular` or
`parallel`, whose source is the map link (`ScaffoldConstraint::link`, the
`edge_uuid` of a stored residual): there is no separate link kind.

A measured crossing of the same pair within `links.deduplicate_arc` (default
64 voxels of arc, on both fibers) of a refined link is the same encounter. The
link is kept, with its own step, and the crossing's relation is dropped
(`links_with_crossing`, `crossings_replaced_by_links`), so an encounter is
never stated twice and a link cannot close a cycle with its own crossing. An
enclosure is a statement about the pair, not about one encounter, and stays.
On the PHerc0332 scaffold 244 of the 281 links are accepted (13 fail the
angle, 24 the end support), 239 of them have a crossing, and 256 crossings are
replaced.

### Normal flips

A map link also states an order: an ordinary H-V link puts the V fiber outside
the H fiber, an adjacent one inside, and a same-family link puts the two side
by side with their normals alike. A measured run's solved normals state the
same order at the link's refined encounter (`fiberSignedOrder`, the order the
crossings use). Two normals that face each other across the connector state no
order at all, which two fibers of one sheet cannot do: that is a disagreement
whatever the link says (`normal_order_links_opposed`); only a connector lying
nearly in the sheet leaves a link undetermined. Where the two disagree, one of the two fibers' normals points
the wrong way there - a flip somewhere along it - and every crossing beyond the
flip measures the wrong side. `scaffoldLinkNormalFlips` lists one suspect per
disagreeing link and fiber, with the stretch to look at: from that fiber's
nearest link whose order agrees to the disagreeing one, or the anchor's own
span when none of its links agrees. A run stores them beside its residuals
(`kScaffoldNormalFlipRowKind`, `readScaffoldNormalFlips`) and counts the links
compared (`normal_order_links_checked`, `_agreed`, `_disagreed`,
`_undetermined`); the CLI summary lists them under `normal_flips`.

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
  With a prediction bound (*variant B*) every chain node is also pulled
  towards the sheet axis the fiber-inference volume states there: the axes
  are sampled at the fibers' own vertices, which lie in the sheet, weighted
  by the prediction's presence between `presence_floor` (weight 0) and
  `presence_full` (weight 1), and gathered per node as the principal axis of
  their weighted outer products. The anchors still win at the nodes they fix.
- **crossings** (`ScaffoldConstraints.hpp`). `extractFiberTraceConstraints`
  runs on the annotation's own polylines, which stay whole (no piece
  splitting). The geometry decides whether a crossing is parallel or
  perpendicular; a crossing whose verdict disagrees with the two annotated
  families is dropped, never forced, and the run says so rather than leaving
  it in the gap between the crossings read and the relations used:
  `measured_skipped_family_mismatch` (with `_invalid`, `_untagged` and
  `_self`) is a run metric, and the solver dock's report names it when it is
  not zero. On the PHerc0332 scaffold 786 of the 10,195 crossings are dropped
  that way, which is a question about the H/V tags of those pairs, not about
  the measurement. The winding separation along
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

### Scope

A run reads every fiber of the store unless it is given a scope. `fibers` (CLI)
or *the selected network* (dock) names the fibers to load; the others are not
read at all, so the links, crossings, normals and components of the run are
those of that part of the annotation, and the run records how many fiber
records its scope left out (`scope_excluded_fibers`). Measured on the
PHerc0332 scaffold: restricted to the 108 fibers of its largest links-only
component, the run reads 510,667 of the 851,285 line points, assembles 111 of
the 232 link constraints, finds one component with objective 0, and places
every one of the 108 fibers at the same winding as the whole-scaffold run.

### Progress and cancellation

A measured run reads the volume along every fiber and takes a minute, so it
reports where it is and can be taken back (`ScaffoldRunMonitor`). The stages
name themselves and count their work off - the prediction sampling per chunk
of vertices, the normals per fiber, the solve per component, the residual
analysis per violated constraint - and check for cancellation at those same
points. A cancelled run throws `ScaffoldCancelled` out of the stage it was in
and writes nothing: no run row, no partial result. The dock polls the monitor
rather than being signalled per fiber, and the **Cancel** button next to
*Solve* is what sets it.

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
| `links.refine_maximum_distance` | `512` | voxels between a link's two refined points |
| `links.deduplicate_arc` | `64` | voxels of arc within which a crossing is the link's own encounter |
| `weights.manual` | `1` | weight per map link arc |
| `constraints.enabled` | `false` | also measure the scaffold's own crossings |
| `constraints.resample_spacing`, `.maximum_distance`, `.tangent_window` | `32`, `128`, `32` | voxels, the extraction's geometry |
| `constraints.relax_endpoint_certification` | `false` | accept crossings near a hand-drawn fiber's ends |
| `constraints.winding_distances` | `true` | measure each connector's separation |
| `constraints.winding_step` | `8` | voxels per integration step |
| `constraints.maximum_winding_distance`, `.enforce_maximum_winding` | `1.5`, `false` | turns, and whether to reject beyond it |
| `constraints.perpendicular_offset`, `.winding_scale`, `.enclosures` | `0.5`, `1`, `true` | relation conventions |
| `weights.perpendicular`, `.parallel`, `.enclosure` | `1`, `1`, `1` | weight per measured arc |
| `winding.manifest`, `winding.remote_cache` | — | the Lasagna normal dataset the separations are measured on |
| `normals.twist_weight`, `.lateral_weight`, `.smoothing_sigma`, `.resample_spacing` | `1000`, `1`, `64`, `64` | the strip solve; the measurements below say why these and not the tracer's |
| `normals.manual_display_normals`, `.approved_pieces`, `.adjacent_links` | `true` | which anchors fix a sign |
| `normals.source` | `geometry` | `geometry` (variant A) or `fused` (variant B) |
| `normals.axis_weight`, `.presence_floor`, `.presence_full` | `100`, `0`, `128` | the fused axis term and its presence mapping |
| `prediction.manifest`, `prediction.remote_cache` | — | the fiber-inference dataset a fused run reads |
| `sign.solve`, `sign.backend` | `true`, `irls` | decide unanchored signs; `irls` or `ceres` |
| `normals.prediction_chunk` | `65536` | vertices per call into the prediction sampler; changes no result |
| `include_untagged` | `false` | load the fibers without an H/V tag too |
| `fibers` | all of them | restrict the run to these fiber UUIDs (the dock's scope) |
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

## Fractional separations

A measured winding separation is rarely a whole turn. The automatic stage
splits a fractional one into the two integers around it, weighted `1 -
fraction` and `fraction` - faithful to the measurement, but the two halves
assert different windings on the same pair, so one of them is always violated,
its shortest cycle is its own sibling (two constraints) and the stretch
between them has zero length. Nothing can be edited or approved to fix that.

`ScaffoldSeparationMode` keeps both rules, as the Python reference stage does:
`Split` is the automatic stage's, `Rounded` states one constraint at the
nearer integer with the event's whole weight (exactly one half rounds down).
The CLI chooses with `constraints.separations` (`"split"`, the default, or
`"rounded"`); the scaffold dock always runs `Rounded`, because an approval is
a statement about an integer relation. A run records which it used
(`separations_rounded`) and how many crossings were fractional
(`measured_fractional_separations`).

## Laying the map out from a solve

With "Lay the map out from this solve" on, the map is built **from the run**:
`FollowedSolve::windings` carries the run's winding and component per fiber,
and `buildGlobalLayout` places each covered fiber at `u = winding + the
fraction of a turn its geometry makes around the umbilicus`, `v = height`. No
crossing detection runs, no constraint graph is built and no winding is
solved - the run already decided all of that, and its residuals already say
which constraints it could not satisfy (drawn suspect).

What the map still does: the per-fiber geometry pass (unwrap the angle,
radius, resample - cached per fiber), the relative placement of components,
which a solve gauges separately and does not relate, and the networks, which
come from the annotation's links rather than from the solve. A fiber the run
did not cover - untagged, isolated, outside a restricted scope, or created
since - is **not drawn**: it is reported as unplaced, because the map is of
the solve.

## One run, one frame

A run is solved in one volume at one scale: the fibers' geometry is read
through `viewTransformedAt(volume, scale)`, and the links' anchors - recorded
in the stored frame on the edge record - are mapped with the *same* affine
(`loadScaffoldGraph`, per fiber). Without that the anchors stay in the stored
frame, and solving in another one puts every anchor a volume away from its
polyline: the run is then refused by the anchor-distance guard ("link anchors
lie N voxels from their fiber polyline"), which is the guard doing its job
rather than an annotation error. The transforms do not move anything to a
canonical base frame; they map into the frame the run asked for.

## What a run reuses

The normals and the crossings are cached in the store
(`ext_fiber_strip_normals`, `ext_fiber_pair_crossings`;
vc_object_store/docs/extensions.md). A run states what it would compute from -
the parameters, each fiber's object version, and the version of every link
and peer fiber that anchors it - looks that up, and computes only what is
missing: an edited fiber is solved again and so are the pairs it is in,
everything else is read. The keys are the store's versions and the parameters
in the clear; nothing is hashed. A fused run is cached under the identity of
the prediction it read (its manifest location). The extraction is told to skip
the cached pairs through its trace-pair filter, with
`parallelPairFilter` set so the search keeps its workers.

Measured on PHerc0332 (`objects-v22.vcobj`, 292 fibers, 851,285 line points,
32 workers, enclosure run with winding separations, warm on-disk volume
cache): the first run solved 292 fibers' normals in 0.337 s and extracted
10,195 crossings over 21,189 pairs in 0.380 s; the second reused all of them
(0.090 s and 0.106 s) and sampled the volume not at all, reporting every
number of the first run identically. On a cold remote cache the saving is the
whole sampling time, which is where a measured run's minutes go.

A run records what it reused: `normal_fibers_reused`, `normal_fibers_cached`,
`events_reused_pairs`, `events_reused`, `events_cached_pairs`.

## Stages, progress and cancellation

A run announces how many stages it will enter (4 for a links-only run, 12
measured, 13 fused) and each stage counts its own work: reading the fibers
(per fiber), the fiber normals (per fiber), the crossing extraction's own
phases (prepare, spatial_index, search, candidate_order, score, deduplicate),
the sign solve (per IRLS step), the winding solve (per component) and the
residual cycles (per violated constraint). The dock shows `i/n stage: done of
total` with the rate's estimate of the time left in that stage, and the bar
shows the stage's fraction.

Cancellation is asked inside those stages, not only between them: the
extraction takes a predicate and stops within a block (OpenMP regions are
left by skipping their remaining blocks, never by an exception), the sign
solve stops at its next step, and the strip solve between fibers. A cancelled
run writes nothing.

The map's Update is cancellable the same way: the rebuild carries the same
monitor, reports the conversion and the pair sweep, and a cancelled build
publishes nothing and leaves the previous layout and its cache in place.

## Approving a measured constraint

An approved constraint is held rigid: the next run solves the approved
constraints alone first and contracts each approved subgraph into one node, so
an approved relation is never traded away (and approved relations that
contradict each other stop the run instead of being averaged).

Nothing is held rigid over geometry nobody reviewed. Whatever states the
approval, a constraint only counts as approved when the two pieces its anchors
lie on are *currently* approved for geometry
(`scaffoldConstraintPiecesApproved`). That covers both kinds: a link whose
record says approved but whose pieces are not counts as an ordinary link and
is reported as `link_approved_without_pieces`, and a measured crossing with a
live approval row whose pieces have since been revoked is reported as
`constraint_approvals_without_pieces`. A link's approval is otherwise its
record's review state. A measured crossing has no record, so
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
  stages: where the normals come from (the fibers' geometry, or that fused
  with the package's fiber-inference prediction), the per-type weights, the
  sign backend, and whether the winding separations are measured at all. The
  *scope* is the whole scaffold or the map's selected network (the selection
  grown to the networks it touches); a scope that resolves to nothing refuses
  the run rather than solving everything, and **Cancel** takes a running solve
  back. The winding density they are measured on is
  the package's own selected Lasagna dataset, the one the annotation optimizes
  against; a run that would measure separations without a bound dataset, or on
  a dataset that states no `grad_mag` density, is refused before it starts
  rather than quietly measuring none. The solve runs in the frame the map was
  built in and on its own worker; the store's umbilicus states the winding
  handedness, and only when it does not does the dock offer the catalog's sense
  as a fallback. The map itself follows the same order on every Update - the
  umbilicus' `winding_handedness` first, the catalog's orientation second - and
  refuses to build when neither states the sense: it is never decided from the
  geometry. The run is saved in the store, and its one-line report lands in
  the map's status line.
- **Solves.** Every run saved in the store, newest first, with its method,
  fibers, components, arcs, weighted L1, violated constraints, largest
  residual, *sign!* (`sign_fixed_disagreements`: anchored fibers whose sign the
  crossings would have flipped, empty for a run that solved no signs) and
  whether it is still *current*. A run is **stale** when a fiber,
  a link or the umbilicus changed after it read the store - taken from the
  records' own commit sequences (`scaffoldAnnotationCommit`), not from the
  change feed, because a store may retain no feed and because a run's own rows
  are extension rows, which change no record: saving a run does not make it
  stale. Selecting one loads its ranking. Selecting a *second* run compares
  the two: `diff` counts the fibers one run places differently from the other,
  and the line below spells the comparison out. Windings are gauged per
  component and per run, so the comparison is relative - the fibers counted
  are those that moved against the neighbours they share a component with in
  both runs; a pure gauge shift or a renumbered component counts as no
  difference.
- **Fiber and piece approval.** The selected run's violated constraints and the
  stretches they accuse, in one table. Columns: what, fiber (the pair for a
  constraint), where (the control spans the anchors lie on), *off* (turns the
  solve could not satisfy), *state*, *cost* (weight times residual, weighted
  turns), *joint* (cost per constraint of the shortest cycle), *cycle*
  (constraints on that cycle), *loop* (which cycle, numbered from 1 in rank
  order - rows sharing a number are about the same loop, and a stretch lists
  every loop running along it), *length* (the stretch, or the cycle's total
  stretch length, voxels) and *paths* (edge-disjoint alternatives for a
  constraint, accusing constraints for a stretch). The Order combo applies
  the three documented orders with their tie-breaks - *smallest cycle* (then
  the shorter stretch), *highest cost* (then the shorter cycle and stretch),
  *cost per hop* - and a header click sorts by that column's own number.
  The component filter restricts the table to one component of the run; a
  component was solved on its own, so its violations can only be explained by
  its own annotation.

  The stretch rows are *segments*, not the stored stretches. Cycles that
  leave a fiber at different crossings accuse nearly the same piece of it, so
  listing every stored stretch would ask for that piece many times over. Per
  fiber, the accused stretches are taken in order of cost per voxel and what
  each adds that no earlier segment and no approved piece covers becomes a
  new segment, or - within 500 voxels of a segment's original extent, so
  absorptions never chain - part of that segment. A segment shorter than 100
  voxels joins the one it touches. Approved geometry is subtracted: only the
  unapproved part is offered, reaching 50 voxels back into the piece it
  continues (*extends approved*). Ends move outward to a control point within
  750 voxels and the boundary two touching segments share moves to one inside
  both; they never move inward. An end that is on no control point is stated
  as a control position - `cp 3.42` is 42 % of the way from control point 3
  to 4 - and the control points an edit needs there are inserted when the
  piece is edited or saved. A segment's *cost* is the sum of the shares of
  every constraint accusing it and *cost/kvx* the densest of them, per 1000
  voxels; *cost per voxel* is the default order. The segments are computed
  from the run's rows and the current approvals, so approving one shrinks or
  removes the ones that overlapped it without a new solve. The dock places
  the run's arcs against control arcs measured in the run's own frame.

  Each constraint row carries the two pieces its anchors lie on as child
  rows, with their approval state: those are what has to be approved before
  the constraint can be, and a double-click on one opens it as an approval
  piece between the control points that bracket it. The same goes for a
  stretch row. Selecting any row selects its fibers on the map and draws the
  spans it names - for a constraint, both covering pieces, which is how a
  crossing is found on the map at all - and brings them into view, zooming out
  only when the current scale cannot hold them. The selection is a halo, the
  same kind the fiber list's selection draws.

  **Fade unapproved** states what the map dims: *nothing*, *geometry* (every
  span without a current geometry approval) or *normals* (every span whose
  normals are not approved, which includes the spans whose geometry is not,
  since a normals approval requires one). An approved span is drawn exactly as
  the map draws everything else; a faded one has the background painted over
  it at half opacity. There is no second colour for approval: what is bright is
  approved.

  The list starts with the run's **normal flip** suspects (see *Normal
  flips*): the accused fiber against the link's other fiber, the stretch to
  look at, and how many of that fiber's links disagree with its normal.
  Activating one opens the stretch, as for any other stretch.

  The *state* column reads `approved`, `approvable` (both pieces approved),
  `blocked` (they are not) or `link`, and **Approve constraint** is enabled
  only for an approvable or an approved crossing - a constraint is a statement
  about two pieces of geometry, and pinning a solve to geometry nobody
  reviewed is what the workflow is meant to prevent. The write refuses the
  same case regardless of the button, and the refusal appears on its own line
  below the counts. A link is never approved here: a link's approval is its
  review state in the annotation.

A run records what its approvals actually fixed:
`constraint_approval_rows` (live rows), `constraint_approvals_applied`,
`constraint_approvals_stale` (the covering spans were edited),
`constraint_approvals_without_pieces` (the pieces are no longer approved) and
`link_approved_without_pieces`.

Halos are reserved for the selection - the fiber list's and the approval
dock's - so what the map says about a fiber's own geometry stays what it always
said. One path per layer covers the whole map, which is what keeps panning with
the fade on as cheap as panning without it.

A piece opened from the dock also shades the strip view outside the opened
range, so the part that can be edited and approved is the part that stayed
bright.

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
| signs | 46 held, 233 decided by crossings, 13 left undecided; 0.78 % of the 514 strict crossings (axes within 30 degrees, less than one turn apart) disagree with the decided signs |
| relations | 8,878 crossings assert an arc (16,011 after the fractional split), 786 dropped where the geometry's verdict disagrees with the annotated families; 931 enclosure pairs give 181 exact relations, 735 one-sided, 7 non-unique, 8 contradictory |
| solve | 16,424 constraints over 2,413 aggregated arcs, 13 components, weighted L1 3,297.9 turns, 7,890 violated constraints, largest residual 8 turns |

Wall time of that run: 1.05 s reading the store, 0.02 s packing the geometry,
0.26 s for the strip normals, 0.32 s for the crossings, 0.009 s for the sign
solve, 0.006 s assembling the relations and 1.74 s for the LEMON solve with
its residual analysis.

Those residuals are a statement about the *evidence*, not a verdict. A
rounded separation is the weakest of the measured kinds: it asserts a whole
number of windings from a measurement that lands between two of them, and
7,890 of those 16,424 constraints disagree with the solve's compromise. The
181 exact enclosure relations are the strongest, because they are integer
bounds from both sides rather than a rounded measurement, and reading the
ranking from the strongest evidence down is what the approval dock is for.

What decides the quality of a scaffold's normals is the stiffness of the
strip solve, not the normal source. The same run at three settings, all with
measured separations:

| | strip 100/32/32 | strip 1000/64/64 | fused, strip 1000/64/64 |
|---|---|---|---|
| strict crossings (axes within 30 degrees, under one turn apart) | 353 | 514 | 736 |
| of those, disagreeing with the solved signs | 26.91 % | **0.78 %** | 2.31 % |
| weight-weighted disagreement over all crossings | 31.65 % | 1.12 % | 2.46 % |
| mean agreement of a solved normal with the outward radial | 0.486 | 0.817 | 0.803 |
| crossings that assert a relation (of 10,195) | 6,993 | 8,878 | 9,060 |
| exact enclosure relations (contradictory pairs) | 51 (18) | 181 (8) | 182 (9) |
| components | 15 | 13 | 14 |

The first column is the tracer's own strip defaults (twist weight 100,
smoothing and resampling 32 voxels), which a hand-drawn polyline is too
sparse and too noisy for; the scaffold's defaults are therefore the second
column's (twist weight 1000, smoothing and resampling 64 voxels), and
`ScaffoldNormalOptions` states them. With those, the fiber geometry alone
decides the normals about as well as the prediction does: the prediction adds
one exact enclosure relation and costs 79 s of an 83 s run to read along all
851,285 line points, against 0.26 s for the geometry-only strip solve.
Everything after the normals is unchanged (0.32 s for the crossings, 0.009 s
for the sign solve, 1.74 s for the LEMON solve and its residual analysis).
Variant B is worth its minute where a fiber's own geometry says little - a
straight V fiber has no preferred normal axis at all - not as a default.

The links-plus-enclosures configuration (perpendicular and parallel weight 0)
shows what the exact relations alone assert: 232 links and 181 enclosure
relations leave 4 violated constraints and a weighted L1 of 6 turns, and the
fused normals leave exactly the same four. One of them, a link violated by 3
turns on a 7-hop cycle, is violated under every sign configuration measured,
including the 32 % disagreement one; the 1-turn violations move when the
signs improve. A residual that survives the signs is the kind the approval
dock is for.
