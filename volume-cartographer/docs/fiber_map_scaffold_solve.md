# Scaffold winding solve (`vc_fiber_graph_solve`)

Integer winding assignment for a manually annotated scaffold, with the
residual analysis the fiber map's approval workflow ranks by. The library is
`vc_fiber_graph_solve` (`vc/fiber_graph/`); `vc_scaffold_solve` is its command
line, and VC3D calls the same functions on a worker thread. Nothing in it is
Qt, and nothing in it is Python: one solve, two callers.

## What a scaffold is here

The annotated fibers of one vcobj store, their `fiber_branch` links, their
approval layers, and the volume's umbilicus:

- fibers come from the store's `fiber` objects in UUID order, which is the
  solve's node order. A fiber's family is its record's `family`
  (`manual_family` selects which labels count, `family_source`); untagged
  fibers carry no winding relation and are left out;
- links carry their two anchor positions, `meta.adjacent`, `meta.auto` and
  their record approval state;
- approvals come from each fiber's approval layer (`ext_fiber_approval*`,
  `vc_object_store/docs/fiber-approval.md`): its current approved and broken
  stretches per geometry/normals aspect, its normal points and the normals
  shown when it was saved. A stretch marked broken in either aspect is
  removed from every relation, like a gap. Broken geometry also splits the
  fiber there: each part takes its own winding, and the per-fiber result
  columns hold the longest part;
- the umbilicus is the one mapping to the frame the fibers are annotated in,
  and the scroll's winding handedness is read from its `winding_handedness`
  meta. A store that states the handedness nowhere is an error: a mirrored
  solve must not happen silently. See *The winding direction* below for how
  it gets there.

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
measured crossing. A link is brought onto that encounter by the crossing
extraction's own refinement, seeded at the link's two anchors
(`refineFiberCrossing`, `vc/fiber_tracer/FiberTraceConstraints.hpp`), under
the extraction's acceptance rules:

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

A **manual** link the refinement finds no such encounter for is the
annotator's statement all the same and is taken where it was placed, at its
two anchors (`links_as_placed`): a link at a fiber's end - where one sheet's H
fiber ends at a kollesis and the next sheet's begins - has no fiber on one
side to certify it, and that end is what it links. An **automatic** link
(`meta.auto`, placed by the map's Link gesture at a pass of two fibers) that
fails asserts nothing; the run counts those by rule (`links_rejected_angle`,
`links_rejected_end_support`, `links_rejected_distance`,
`links_rejected_consistency`). The run records the farthest a refinement
moved an anchor (`link_refinement_max_shift`). A link is a constraint of the
kind its two families state, `perpendicular` or `parallel`, whose source is
the map link (`ScaffoldConstraint::link`, the `edge_uuid` of a stored
residual): there is no separate link kind.

A measured crossing of the same pair within `links.deduplicate_arc` (default
64 voxels of arc, on both fibers) of a refined link is the same encounter. The
link is kept, with its own step, and the crossing's relation is dropped
(`links_with_crossing`, `crossings_replaced_by_links`), so an encounter is
never stated twice and a link cannot close a cycle with its own crossing. An
enclosure is a statement about the pair, not about one encounter, and stays.
On the PHerc0332 scaffold 230 of the 232 manual links are used (2 lie on a
gap), 34 of them as placed; the 49 automatic links are left out by default.

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
  One convention holds for every signed normal, here and in the whole-volume
  pipeline: it points outward, away from the umbilicus. The winding relations
  order two fibers by the direction the winding falls in, which is the
  outward one where the scroll's winding falls outward (handedness -1) and
  the inward one otherwise; `scaffoldFallingWindingSign` is that single
  conversion. A fiber whose geometry carries no normal (fewer than three
  distinct points) fails the run: a solve has a normal for every fiber.
  The annotation fixes the normal where it states one, as an **axis**: the
  annotator approves the cross-section they see and cannot know which of its
  two sides faces outward.
  A stretch approved for the normals aspect carries the normals that were
  shown at its line points when it was approved (stored with the layer,
  `vc_object_store/docs/fiber-approval.md`). Every chain node on the
  stretch is held on that axis (the principal axis of the stored normals of
  the line points nearest to it), and the stretch's line points come out with
  the stored normals themselves. This holds for a stretch nobody corrected
  too: approving it approves what was shown.
  A normal point of the layer outside an approved stretch holds the chain
  node at its position on the normal point's axis.
  A control point's `display_normal` with source `manual` outside an approved
  stretch holds the one chain node at that control point on its axis; without
  an approval nothing says how far it reaches.
  Between held nodes the strip solve runs as everywhere else, and it faces the
  held nodes by continuity along the fiber, so a fiber has one facing. That
  facing is resolved like any other fiber's: by an adjacent link, whose two
  anchors give the direction of falling winding (the V fiber lies one winding
  below the H fiber) and with it the outward normal of both fibers; by the
  neighbour normals; or by the sign solve. A link that lands on a held node
  keeps the annotated axis and only says which way it faces.
  A stretch or normal point that no longer matches its line points (the line
  moved) is not used. The run counts those
  (`normal_approvals_stale`) and the dock's summary names them: approve the
  stretch again.
  With a prediction bound (*variant B*) every chain node is also pulled
  towards the sheet axis the fiber-inference volume states there: the axes
  are sampled at the fibers' own vertices, which lie in the sheet, weighted
  by the prediction's presence between `presence_floor` (weight 0) and
  `presence_full` (weight 1), and gathered per node as the principal axis of
  their weighted outer products. The anchors still win at the nodes they fix.
  With a neighbour normal field (`normals.source: neighbours`, see *Neighbour
  normals* below) every chain node is pulled towards the signed normal the
  automatic fibers around it state, and a fiber no anchor fixed takes its
  facing from them.
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
- **signs**. The fibers neither an adjacent link nor the neighbour normals
  faced are decided by the perpendicular crossings, with the native sign solve
  (`vc/fiber_tracer/FiberNormalSign.hpp`, `irls` or `ceres`), from the radial
  prior of each fiber's own normals. A faced fiber's sign is held. An approved
  or manual normal fixes an axis and never a sign. The run records how many signs each source
  decided, and how many anchored fibers the crossings disagreed with.
- **relations**. A perpendicular crossing asserts the rounded separation plus
  the half winding between the two families, a parallel one the separation
  itself; a fractional separation asserts the two integers around it with the
  weight split between them. The signed order of a perpendicular crossing also
  bounds the pair's H->V step from one side, and a pair bounded from both
  sides that agree fixes its step exactly: the *enclosure* relation. Weights
  are `weights.perpendicular`, `weights.parallel` and `weights.enclosure`.

- **sign** (`weights.sign`, 3 by default; times `weights.sign_approved_factor`,
  10, where one of the two anchors lies on a stretch approved for normals). The signed order of one
  perpendicular crossing is a bound on its own: the V fiber lies outside the H
  fiber there, so the pair's H->V step is at least the crossing's transport,
  or inside, so it is at most that minus one. An enclosure needs a pair
  bounded from both sides that agree; a sign constraint states each bound
  as it is (`ScaffoldConstraintKind::SignAtLeast` / `SignAtMost`), with a
  one-sided cost `weight * max(0, bound - step)` that is zero while the bound
  holds. It needs no winding separation. The flow solve takes it as an arc
  with one of its two capacity ends removed
  (`solveWindingMinCostFlow` with a side per arc), so the optimum stays exact.
  The bounds are those of the Python min-flow experiment's `sign_only` graph
  (`fiber_min_flow_benchmark.graph`) for perpendicular crossings; that
  experiment also states one for signed parallel crossings, which this
  does not. It counts as part of a residual cycle only where it holds with equality.

- **crossing confidence.** A crossing's perpendicular, parallel and sign
  weight is its type's weight times the confidence the sign stage gives the
  same crossing (`fiberSignEventWeight`): dot² of its two normal axes times
  min(1, 1/d²) of its winding distance d. A clean near crossing keeps its
  weight; an oblique or far one still counts but gives way. Enclosures keep
  their weight.

The defaults are the scaffold workflow's run: no parallel and no enclosure
relations, perpendicular weight 1, sign weight 3, crossing confidence on;
the manual links (approved and pending) at weight 100 beside the measured
crossings, split separations, neighbour normals, the cost-scaling flow
backend. That run is stated once, in `vc/fiber_graph/ScaffoldSolveProfile.hpp`
(`scaffoldSolveProfile()`): the solver dock runs it as it is and
`vc_scaffold_solve` starts from it before its configuration overrides
anything, so a default changed there changes both and neither states one of
its own.

Every kind lands in the same constraint list, so the solve and the residual
analysis treat a crossing exactly like a link.

Measured, before continuation links existed, on the PHerc0332 scaffold with
neighbour normals, no links and perpendicular weight 1 (1,423 perpendicular constraints, 1,423 sign
constraints), against the links-only solution (216 fibers comparable):

| sign weight | fibers placed differently from the links | violated perpendicular | violated sign |
|---|---|---|---|
| 0 | 13 | 406 | - |
| 10 | 4 | 424 | 2 |
| 100 | 2 | 437 | 1 |

### Neighbour normals

The whole-volume pipeline resolves the normal and its sign for every automatic
fiber (strip solve, sign stage). A scaffold fiber lies among those fibers, so
its normal can be read off them instead of being solved from its own geometry
and signed by the scaffold's few crossings:

1. The full-volume extraction writes its solved normals at a regular arc
   spacing (`strip_samples_output`, vesuvius `docs/fiber_workflows.md`).
2. `vc_fiber_constraints` with `signed_strip_samples_output` multiplies every
   sample by its fiber's sign of a sign store's normal-sign run (`strip_samples`,
   `sign_store`, optional `sign_run`) and writes the signed samples with a
   statement next to them (`<output>.json`: `coordinate_space`, `volume_uuid` -
   the volume the points are voxels of, as the scaffold store names it - the
   sign store and run, the counts).
3. A run with `normals.source: neighbours` opens that file
   (`openScaffoldNeighbourSource`) and asks the field every 16 voxels along
   each fiber's annotated line (`FiberNeighbourField::infer`, the full-volume
   pipeline's own fiber frame: crossing scales 128 / 32 / 128, parallel scales
   192 / 192 / 64, radius 384; a parallel contact is measured in the frame of
   an isotropic first estimate, since a scaffold fiber carries no prediction
   axis). The answers become signed soft targets of the strip solve (weight
   300, Gaussian sigma 32 along the arc), exactly as the full-volume extraction
   uses its own neighbour normals.

4. The same run adds the whole-volume strip solve's crossing term
   (`setFiberCrossingTargets`, weight 100, sigma 64, Cauchy falloff of the gap
   scaled 64 / 32 / 1024): where another fiber crosses, the normal lies along
   own tangent x partner tangent, both tangents those of the fibers
   themselves. The crossings of a scaffold fiber by the field's fibers are
   found in the field's own grid (`FiberNeighbourField::crossings`: a contact
   where the two tangents meet at more than 30 degrees), those by the
   scaffold's other fibers the same way in a field of the scaffold's own
   samples. Both are on by default (`neighbours.field_crossings`,
   `neighbours.scaffold_crossings`). Against the whole-volume run: the contact
   is the closest pair of samples 16 voxels apart rather than a refined
   crossing, within the field's radius of 384 voxels rather than 512, and a
   scaffold fiber carries no prediction axis, so the gap is measured in the
   crossing's own frame and no crossing is weighted by its agreement with a
   prediction.

Annotated normals still hold the nodes they fix, and those nodes' signed
targets still vote on the fiber's facing. A fiber no adjacent link faces
whose chain nodes got a target on at least a quarter of its length is faced by
the field (`ScaffoldSignSource::Neighbours`) and held in the sign solve like an
anchored one; a fiber the field does not cover is left to the crossings. The
field must be in the run's frame: a run in another volume or at another scale
is refused, and so is a field with no fiber within its radius of any scaffold
fiber. The signed normals are taken as given: the sign stage faces them the
way this gauge faces an anchored fiber (on PHerc0332 the order of 194 of the
196 map links confirms the normals solved from the field, one contradicts).

Measured on the PHerc0332 scaffold (`objects-v22.vcobj`, 292 fibers) with the
field of the full-volume run (239,064 fibers, 15,326,161 samples, 1.2 GB),
enclosure relations only:

| | geometry normals | neighbour normals |
|---|---|---|
| fibers faced before the sign solve | 46 (adjacent links) | 292 (46 links, 246 neighbours) |
| fibers left undecided | 19 | 0 |
| strict crossings, of those disagreeing | 415, 0 | 575, 0 |
| map links whose order the normals confirm (of 196) | 190, 1 against, 5 open | 194, 1 against, 1 open |
| exact enclosure relations (contradictory pairs) | 160 (1) | 160 (1) |

The strip solve runs at the whole-volume pipeline's stiffness (twist 300).
Against the former scaffold value of 1000, with neighbour normals: the map
links' order is confirmed on 194 of 196 links either way (2 against at 300,
1 against and 1 open at 1000), 576 against 575 strict crossings with none
disagreeing, 161 against 160 exact enclosure relations.

With the crossing term (555,674 crossings by the field's fibers, 9,352 among
the scaffold's own, a crossing axis on 92,935 of the 93,261 chain nodes) the
same checks read: 194 of 196 link orders confirmed, 585 strict crossings with
none disagreeing, 160 exact enclosure relations; the default solve's weighted
residual falls from 312.5 to 306.0 turns with the same 441 violated
constraints. It adds 4 s to the neighbour stage with 32 workers.

The neighbour stage takes 9.6 s with 32 workers (AMD EPYC 7J13) and 31 s with
8, and the run peaks at 3.9 GB with 8 workers: the field is held in memory.
Normals solved from a field are cached like the others, keyed by the field's
sign run and sample count and the inference parameters.

## The winding direction

The solve and the map need the scroll's winding direction in the annotation
volume's coordinates: `+1` when the winding index grows with `atan2(dy, dx)`
about the umbilicus, `-1` against it. It is never inferred from the fibers.

- **Where it is kept.** On the umbilicus record of the fiber store:
  `meta.winding_handedness` (`+1` or `-1`) and `meta.winding_handedness_source`,
  who stated it - `catalog <sample>/<volume>` or `manual`
  (`vc_object_store/objects/VolumeUmbilicus.hpp`,
  `setUmbilicusWindingHandednessRequest`).
- **Where it comes from.** The open-data catalog records no direction of its
  own. Its convention is one spiral for every scroll seen from its top, so
  the direction follows from the volume's orientation
  (`z_direction_is_top_to_bottom`, `left_handed_coordinates`): `-1` exactly
  when one of the two holds (`vcr::catalog::windingHandedness`). Both
  properties are needed; a volume the catalog leaves one of them unset for
  has no catalog direction.
- **Kept from the catalog.** When the store states none and the catalog
  orients the volume, VC3D stores the catalog's direction on the umbilicus as
  soon as the volume or its umbilicus appears (`keepCatalogWindingDirection`,
  `ProjectResources.hpp`), and `vc_winding_handedness <store> --set catalog`
  does the same from the command line. After that nothing is looked up: every
  reader finds it in the store.
- **Set by hand.** For a volume the catalog does not orient: *File > Set
  Winding Direction...*, beside the umbilicus import, or
  `vc_winding_handedness <store> --set +1|-1`. A direction set by hand is
  recorded as `manual` and is not replaced by the catalog's.
- `vc_winding_handedness <store> [--volume <id>]` prints what the store states
  and its source. `vc_scaffold_solve` reads the store, or takes `handedness`
  in its config for a store that cannot be written.

## Solve

`solveWindingMinCostFlow` (LEMON, `network_simplex` by default, capacity scale
1000) minimizes `sum_e weight_e * |k_b - k_a - target_e|` exactly, one
connected component at a time, each gauged to minimum winding 0. Fibers
without a constraint keep their own component and winding 0.

Nothing is held rigid. An approval makes a constraint heavier, never exact:
whether two statements contradict each other can only be said around a whole
cycle, and a cycle is rarely approved end to end.

### Approved normals are faced first

Before any winding is solved the stretches approved for normals are faced
(`scaffoldFaceApprovedNormals`). A normals approval states an axis along a
stretch, not which of its two sides is outward. Every perpendicular crossing
whose two anchors lie on approved stretches carries the facing across (the two
normals point the same way through the sheets), so the stretches the crossings
connect are one object with one free sign. That sign is the one that agrees
best with the signed normals of the resolved fiber volume around the object's
stretches, each sample counting with cos^2 of its angle; a run without
neighbour normals takes the radial prior of the sign solve instead.

Crossings that contradict each other inside one object are an annotation
error - every part of that loop was approved - and fail the run, naming the
crossing that closed the contradiction. No winding is solved in this step.

The fiber normals are then traced out from the approved stretches
(`ScaffoldNormals::face`): each stretch takes its facing, the fiber before the
first and after the last follows the nearest one, and between two stretches
the solved normal is kept where the two agree. Where they do not, the solved
normals hold a half turn between them: it is taken out evenly over the gap, in
the direction that leaves the least turning. Whatever error this hides shows up
as violated constraints of the solve. A faced fiber's sign is held by the sign
solve (`ScaffoldSignSource::ApprovedPiece`); its faced normals are a cache row
of their own, keyed by the fiber's inputs and the facing, and that is the row
the run references. The run records `approved_normal_stretches`, `_objects`,
`_crossings`, `_objects_faced_by_neighbours`, `_weakest_agreement` (the share
of the weighted samples that agree, for the object where it is lowest) and
`_gaps_unrolled`.

### Continuation links

A fiber is not one node of the solve. It is cut at both ends of every approved
stretch (geometry or normals; an approved stretch is one node and never cut)
and, in the fiber between them, into pieces of at most `continuation.interval_vx`
(2,000 voxels), each of those cuts moved to the middle between the two
constraints around it (`scaffoldContinuationArcs`). Every cut is a
*continuation link*: a constraint of kind `continuation` that puts the two
parts on the same winding at `continuation.weight` (10). So a fiber can break
where the evidence asks for it, at a cost, and a break shows where to look.
Broken geometry still cuts a fiber without a link.

The per-fiber winding of a run is that of the fiber's longest part. A run
stores every link (residual row kind 19: the place, the weight, and by how
many windings the solve broke it), and a broken one is also a violated
constraint.

On the PHerc0332 scaffold (292 fibers, 10 stretches approved for normals in 7
objects, all faced by the neighbour normals with every sample agreeing): 1,798
continuation links, 1 broken; 412 of 4,570 constraints violated, 223.9
weighted turns; 17.7 s for the whole run with 16 workers.

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
| `links.approval_states` | `["approved", "pending"]` | link record states to use |
| `links.include_auto` | `false` | also use `meta.auto` links |
| `links.max_anchor_distance` | `5` | voxels, a hard guard |
| `links.refine_maximum_distance` | `512` | voxels between a link's two refined points |
| `links.deduplicate_arc` | `64` | voxels of arc within which a crossing is the link's own encounter |
| `weights.manual` | `100` | weight per map link arc: an annotator's link outweighs a measured crossing (equation at most 1, sign 3) |
| `constraints.enabled` | `true` | also measure the scaffold's own crossings |
| `constraints.resample_spacing`, `.maximum_distance`, `.tangent_window` | `32`, `128`, `32` | voxels, the extraction's geometry |
| `constraints.relax_endpoint_certification` | `false` | accept crossings near a hand-drawn fiber's ends |
| `constraints.winding_distances` | `true` | measure each connector's separation |
| `constraints.winding_step` | `8` | voxels per integration step |
| `constraints.maximum_winding_distance`, `.enforce_maximum_winding` | `1.5`, `false` | turns, and whether to reject beyond it |
| `constraints.perpendicular_offset`, `.winding_scale`, `.enclosures` | `0.5`, `1`, `true` | relation conventions |
| `weights.perpendicular`, `.parallel`, `.enclosure` | `1`, `0`, `0` | weight per measured arc |
| `weights.sign` | `3` | weight per sign constraint (one per signed perpendicular crossing) |
| `weights.sign_approved_factor` | `10` | factor on a sign constraint with an anchor on a stretch approved for normals |
| `weights.reviewed` | `10` | weight of a separation a reviewer stated |
| `weights.crossing_confidence` | `true` | scale a crossing's perpendicular, parallel and sign weight by the sign stage's term, dot² × min(1, 1/d²) |
| `constraints.crossing_filter` | `true` | follow every perpendicular crossing through its two fibers' normal surfaces and apply the shared crossing rule (vc_object_store `geometry/FiberCrossingFilter.hpp`, the one the traced volume's sign and winding stages use): the sign solve drops crossings with nothing left or mixed relations, the constraints drop crossings with nothing left and those that disagree under the solved signs but have an agreeing crossing in the same encounter |
| `winding.manifest`, `winding.remote_cache` | — | the Lasagna normal dataset the separations are measured on; opened when the first separation has to be measured, so a run whose crossings are all cached does not open it |
| `normals.twist_weight`, `.lateral_weight`, `.smoothing_sigma`, `.resample_spacing` | `300`, `1`, `64`, `64` | the strip solve, at the whole-volume pipeline's settings |
| `normals.manual_display_normals`, `.approved_pieces` | `true` | which annotated normals are held as axes |
| `normals.adjacent_links` | `true` | whether adjacent links fix a fiber's facing |
| `normals.source` | `neighbours` | `geometry` (variant A), `fused` (variant B) or `neighbours` |
| `neighbours.samples` | `<store>.neighbour_normals.vcstrip` | the signed strip samples a `neighbours` run reads; by default the file beside the store that the dock reads too. A missing file fails the run |
| `neighbours.weight`, `.sigma`, `.minimum_coverage` | `300`, `32`, `0.25` | the signed target's weight, its arc sigma, and the share of a fiber the field must cover to face it |
| `normals.axis_weight`, `.presence_floor`, `.presence_full` | `100`, `0`, `128` | the fused axis term and its presence mapping |
| `prediction.manifest`, `prediction.remote_cache` | — | the fiber-inference dataset a fused run reads |
| `sign.solve`, `sign.backend` | `true`, `irls` | decide unanchored signs; `irls` or `ceres` |
| `normals.prediction_chunk` | `65536` | vertices per call into the prediction sampler; changes no result |
| `include_untagged` | `false` | load the fibers without an H/V tag too |
| `fibers` | all of them | restrict the run to these fiber UUIDs (the dock's scope) |
| `lemon.backend`, `lemon.capacity_scale` | `cost_scaling`, `1000` | flow backend |
| `continuation.enabled`, `.weight`, `.interval_vx` | `true`, `10`, `2000` | continuation links: a fiber may break at the ends of its approved stretches and about every interval between them |
| `max_alternative_paths` | `4` | redundancy cap |
| `workers` | `1` | geometry packing threads |
| `output`, `windings_output` | — | JSON summary / per-fiber windings |
| `approvals` | `true` | `false` solves as if nothing had been approved, marked broken or reviewed |
| `layout_output`, `layout_spacing_vx` | —, `64` | the map of the solve: per fiber part its line as `u` = winding + turns along the fiber + turn fraction around the umbilicus, `v` = height |

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

A run that solved normals references, per fiber, the normals row it used
(`ext_fiber_scaffold_run_normals`: run, fiber, fiber version, normals row).
`readScaffoldRunFiberNormals` follows that reference and nothing else: a
stretch opened from a run shows that run's normals, and when the reference
does not resolve - a run saved before runs referenced their normals, a row
that is gone, a fiber edited since - the stretch is not opened and the reason
is stated. There is no search for a fiber's "current" normals, and no other
normals stand in. The same holds in the annotation views: solve normals whose
line lies elsewhere than the open one (another volume or scale) are an error.

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
The default scaffold solve runs `Split`, in the dock and on the command line;
the CLI can choose with `constraints.separations` (`"split"` or `"rounded"`).
A run records which it used
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

The crossing filter is not cached: its findings depend on the run's normals,
so each run follows the perpendicular crossings through the two fibers'
normal surfaces again. The two surfaces of a pair are intersected once for all
of the pair's crossings (`fiberCrossingFindings` over a list of crossings),
each span asks only the other fiber's segments near it, and after the approved
normals were faced only the crossings of a fiber with a turned stretch are
followed again.

The fiber map also keeps, for the store it is solving, what it read: each
fiber as its version reads (`ScaffoldFiberReads`, `ScaffoldLoadOptions::reads`:
an edited fiber is read again, the approval layers always are) and the cache
rows (read on the first run, then added to by what each run saved). Rows
another process wrote meanwhile are not seen until the store is opened again;
they would only have saved work. Each run logs its seconds per part
(`scaffold solve: ...`), and `vc_scaffold_solve` reports them under `timings`.

Measured on PHerc0332 (`objects-v22.vcobj`, 292 fibers, 16 workers,
`vc_scaffold_solve`, everything cached, QuickBuild): the crossing filter took
4.0 s and again 4.2 s after the facing before these changes, 0.35 s and 0.21 s
after, with an identical result; reading the fibers went from 1.4 s to 1.05 s,
and to 0.14 s with the reads of an earlier run kept.

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

An approval is never exact: a link's review state does not change its
weight, and a reviewed crossing is a heavy constraint.

A measured crossing has no record, so what a reviewer states about it is a row
of `ext_fiber_constraint_approval`, keyed by the two fibers, the kind and the
two anchors quantized to 64 base voxels, and carrying the span fingerprints
(`fiberSpanHash`) of the two covering spans. The row states a distance only -
direct neighbour or distant - never which fiber lies outside.
`scaffoldCrossingApprovals` reads what the live rows state per crossing and
skips a row whose spans have been edited since. `scaffoldMeasuredConstraints`
applies it:

- direct: the separation is the perpendicular offset, whatever was measured
  (with no known side nothing can be stated, `crossings_stated_without_sign`);
- distant, measurement not direct either: the measurement is kept;
- distant, measurement direct or missing: the separation is dropped;
- distant, always: the sign bound moves one winding out.

A reviewed separation takes `weights.reviewed` (10), whatever the crossing's
confidence. Re-running the extraction does not lose a
row - the crossing lands in the same quantum - but editing either fiber's
covering span does. The run stores its perpendicular crossings (residual row
kind 18: the measured separation and whether the side is known), which is
what the approval window lists.

## In the fiber map

Three docks (`FiberMapScaffoldSolve.hpp`, with the Qt-free ranking in
`FiberMapScaffoldRanking.hpp`):

- **Graph solver.** Which links a run uses (by review state, and whether
  automatic links count), the link weight, the LEMON backend and *fix approved
  subgraphs* (on by default). *Measure the fibers' crossings* adds the measured
  stages: where the normals come from (the full volume's neighbour normals,
  the fibers' geometry, or that fused with the package's fiber-inference
  prediction), the per-type weights (perpendicular, parallel, enclosure and
  order), the
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
  *Full-volume neighbours* is the first choice of the normals combo. It
  reads `<fiber store>.neighbour_normals.vcstrip` (and its `.json` statement)
  from the fiber store's directory - a fixed name for now - once, and keeps
  it for later runs until the file changes. Without the file the run is
  refused and names the path it looked for.
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
- **Fiber and piece approval.** The stretches to approve, and nothing else.
  A stretch is not part of a solve: the list is derived here
  (`scaffoldStretches`) from what the selected run left violated (its
  residual rows and its stored crossings) and from the approval layers as
  they are now. An approval therefore changes the list at once; a new solve
  only changes the costs.

  An annotation states a normal without a facing and a crossing's distance
  without a side, so a stretch always ties constraints together: from a
  violated constraint's anchor to the nearest other constraint of that fiber
  (however far), over the next ones until it is 500 voxels long, and over
  further ones within 300 voxels while it stays within 2,000 voxels. It
  reaches 100 voxels past its outer constraints. One row per fiber of the
  violated constraint; overlapping rows of a fiber are one. *cost* is the
  summed cost of every violated constraint anchored on the stretch.

  A stretch that is already approved (geometry or normals) is fixed: it is
  listed exactly as approved, with the violated constraints on it, and is
  never cut up or shifted. New stretches lie between approved ones and reach
  at most 100 voxels into one.

  Columns: *geo* and *nrm* (a check when the stretch is approved for that
  aspect), *approved w.* (crossings with an approved winding distance - direct or distant - of all on the stretch),
  *winding* (violated winding constraints - separations and links - of all
  on it, as X/Y), *sign* (violated sign constraints, of the crossings whose
  side is known, as X/Y), *cont* (continuation links on the stretch the
  solve broke, of all on it, as X/Y), *where*, *cost*,
  *cost/kvx*, *length* and *fiber*. The Order combo offers *highest cost*
  (the default) and *cost per voxel*; a header click sorts by that column.
  The component filter restricts the table to one component of the run.

  What a run accuses outside a fiber's present control points stays a row
  without a control range, and selecting it is an error that says the run is
  older than the annotation.

  A row only ever shows a stretch: selection, hover and opening have no
  whole-fiber form, and a stretch that cannot be resolved reports why in the
  status line (a stretch spanning a whole fiber is still drawn as one).

  Each constraint row carries the two pieces its anchors lie on as child
  rows, with their approval state: those are what has to be approved before
  the constraint can be, and a double-click on one opens its fiber in the
  approval window on that stretch
  ([fiber_approval_workflow.md](fiber_approval_workflow.md)). The same goes for a
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

  The *state* column reads `reviewed` (the crossing's neighbour state was set
  in the approval window), `sign` or `link`. A link is never approved here: a
  link's approval is its review state in the annotation.

A run records what its approvals stated:
`constraint_approval_rows` (live rows), `constraint_approvals_applied`,
`constraint_approvals_stale` (the covering spans were edited) and
the `crossings_stated_*` counts above.

Halos are reserved for the selection - the fiber list's and the approval
dock's - so what the map says about a fiber's own geometry stays what it always
said. One path per layer covers the whole map, which is what keeps panning with
the fade on as cheap as panning without it.

A stretch opened from the dock is the approval window's selected stretch: the
views dim the line outside it.

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
column's (twist weight 1000, smoothing and resampling 64 voxels). The
default has since moved to the whole-volume pipeline's twist 300 (see
*Neighbour normals*). With those, the fiber geometry alone
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

## Comparing a scaffold solve with a traced volume's winding solve

`vc_scaffold_solve` with a `field_comparison` section scores the run's scaffold solve against the winding solve of a resolved fiber volume, without a union store and without extracting the volume again. The two solves share no fiber; what relates them are the crossings between a scaffold fiber and a fiber of the volume.

| Key | Meaning |
|---|---|
| `field_comparison.samples` | the volume's signed strip samples: its fibers as polylines with their signed normals |
| `field_comparison.winding_store` | the volume's winding result store (families and solved windings); `winding_run` selects a run when it holds several |
| `field_comparison.weights` | optional `perpendicular` / `parallel` / `sign` weights of the cross constraints (default 1 / 0 / 3), independent of the scaffold solve's own weights, so a links-only solve can be compared too |
| `field_comparison.output` | the cross constraints as doubles, 11 per row (scaffold fiber, volume fiber, kind, target, weight, winding distance, scored, step, residual, the two arcs), and `<output>.json` with the fiber ids and the summary |

- The crossings are extracted between the two sets only (the extractor's bipartite mode), with the run's event options, and turned into constraints by the shared assembly, with rounded separations so that every crossing is one statement. The crossing findings and the crossing filter apply as in the solve.
- The scaffold side uses the run's normals and solved signs. With `normals.source: neighbours` reading the same volume, those are the normals and facing transferred from the volume, which is what makes the signed order of a cross crossing meaningful. The volume side uses its signed strip normals.
- The two solutions have independent gauges, one per component each. Every pair (scaffold component, volume component) joined by cross equations gets the integer offset that most of its equation weight agrees on; a constraint is scored against that. The summary states, per constraint type, how many are met, off by one winding and off by more, and per component pair its offset and the share of weight agreeing with it.
- `constraints.enabled` is required: the comparison needs the run's normals and signs.

