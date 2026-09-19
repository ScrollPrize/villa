# Task plan: construct geometry from the prediction, not from the skeleton

## Principle

Topology is fixed by construction and never changed by an optimizer. Geometry
is derived from the prediction field. A point shared by several graph elements
is moved once and every incident element follows it.

## Steps

1. **Recentre the graph onto the prediction.** A new stage between graph
   reduction and topology construction. It does not add, remove or relabel any
   node, edge or incidence; it only moves coordinates.
   - Each node is one point shared by its incident edges. Move it once, by a
     presence-weighted fit over a small isotropic neighbourhood, so all
     incident edges follow coherently.
   - Each edge interior vertex is moved within the plane normal to the local
     edge tangent, by the same Gaussian-weighted fit of presence times squared
     direction agreement already used for ports.
   - Recompute edge lengths afterwards.
2. **Measure the ceiling first.** Before tuning anything, report the distance
   from the refined reference to the recentred graph, and the oracle route over
   the recentred candidate network. If the graph does not approach the 2.00 the
   threshold mask allows, recentring is not sufficient and step 3 applies.
3. **If junctions resist recentring**, restrict the fit where the local
   cross-section is not unimodal, since the ridge is genuinely ambiguous inside
   a merged contact, and let the region connections carry those stretches.
4. **Re-measure the whole chain** with the route-based metrics only: best
   connected route through the graph, oracle route through the candidate
   network, and the committed trace, each as GT-to-route and route-to-GT.

## Testing

- `test_fiber_skeleton_trace` gains a case: a synthetic fiber whose skeleton is
  deliberately displaced from the presence ridge must be pulled back onto it by
  recentring, with the node and edge counts unchanged.
- The existing twelve cases must stay green.
- Case31 through the approved runner, comparing the route-based metrics above.

## Spec update

`planning/specs.md`, skeleton prototype section: the skeleton supplies topology
only; edge and node geometry is recentred onto the prediction before ports are
placed; recentring may not change topology.

## Docs updates

`docs/skeleton_lsq_tracing.md`: replace the stage list with the recentring
stage included, and record the measured chain.

## Changelog

One entry for the geometry-construction change once it measures better.

## What was actually built (2026-09-19)

The plan assumed the skeleton could be pulled back onto a presence ridge. The
measurement showed there is no ridge to pull to: presence saturates across a
ribbon that is narrow across the sheet and wide along it, so the fibre is
localised in one direction only, and inside a merged ribbon nothing local
picks the fibre. Recentring was dropped. Instead every port gets sub-ports
from its section, every line variant and region connection between sub-ports
is refined from a path-search init and kept, and each sub-port carries its own
cheapest continuation on either side. An interim layer that committed one
matching per region and per line before tracing was built, measured to be the
source of every remaining restart, and removed. The tests cover the section
candidates, the path search and the crossing routes; all fourteen cases are
green.
