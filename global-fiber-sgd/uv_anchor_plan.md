# Anchor-relative layout integration

User task: resume global layout after frozen normal/HV/H-direction/V-direction
solutions. Shared unwrapped phase and V anchor; per-winding U/V offsets. Very soft
centering, strong squared hinge outside +/-0.25 turns (90 degrees). Compare all
winding pairs via compatibility likelihood plus separate signed-order penalty.
Test both H circulation conventions; V uses UP. Preserve original observations.

## Implementation

1. Hash-bind accepted direction solutions, freeze all prior exclusions/labels.
   Use signed whole-fiber arc maps inherited by pieces, first-point anchors.
   H convention also changes geometric phase, never outward normal ordering.
2. Opt-in state: 101 logits, shared phase/V adjustments, 100 U/V offsets each.
   Cumulative positive widths map unwrapped k+phase continuously across seams.
   Center offsets to remove anchor gauge. Penalize, do not clip offsets.
3. Conditional retained probabilities: stable all K*K logsumexp compatibility.
   Gate losses by retention, explicitly charge rejection; keep independent
   expected sign-violation penalty. Expected-cost control remains available.
   Continuation compares derived endpoints with all-pair compatibility.
4. Synthetic regression tests: seam/phase continuity, offset hinge and centering,
   reinforcement gradients, rejection/sign handling, direction and provenance.
5. Approved managed CUDA pilot then matched full CCW/CW runs. Report full graph
   retention, sign errors, ambiguity, distance/continuation, offsets and timing.

## Spec update

Opt-in global solver only. Earlier extraction/stages/results unchanged. Finite
penalties are not hard feasibility guarantees. No anatomical accuracy claim.

## Docs updates

Update uv_layout.md, configs/results, task/status/log/changelog. Independent
plan and code review required. Preserve historical results and legacy configs.

## Independent review and pilot findings

Plan/code reviewed independently. All frozen masks/provenance and direction
conventions pass, including exact full prepared-event/piece equality between
CCW/CW. 49tests pass; saved pilot metrics independently reproduced.
Finite/sampled priors do not guarantee quarter-turn bounds or separated buckets;
report chosen and probability-mass exceedance, not only all-bucket averages.
Pilot exposes incoherent radial initialization across splits; defer coherent
same-source initialization to a separately measured change, rather than silently
altering it during the matched full direction comparison.
