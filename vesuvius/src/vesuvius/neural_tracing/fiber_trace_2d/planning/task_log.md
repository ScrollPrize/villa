# Task log: lossless post-stage-two Fiberlet graph simplification

## Initial findings

- A physical Fiberlet ID is the canonical ordered pair of exact anchor keys.
  The storage graph therefore cannot contain two distinct physical Fiberlets
  with the same exact endpoints; such input is rejected as a duplicate stable
  ID. Different anchor variants are different endpoints.
- Stage-two retained output currently persists only physical Fiberlets. Its
  anchor view still exposes all source anchors, so unused anchors must be
  explicitly filtered in the simplified graph.
- Ordinary stored route geometry uses a curved lattice defined by one endpoint
  pair. Concatenating multiple routes and serializing them as one ordinary
  Fiberlet would require resampling and would not be lossless. Macro-Fiberlets
  will reference ordered original directed Fiberlets instead.
- Transition validity and cost depend on the incoming and outgoing directions.
  Physical degree alone is insufficient for contraction; bidirectional
  contraction requires the regular transition to exist in both directions.
- Independent review required an explicit live-direction mask, ordered rather
  than pre-summed authoritative macro costs, atomic hidden-anchor validation,
  and per-box macro identity. It also identified that arbitrary dominated-route
  deletion conflicts with preserving the valid route set. The plan was updated
  before implementation.
- Forward/reverse reachability is deliberately conservative under the
  no-revisit rule: it safely removes proven-dead states but may retain a state
  whose separate reachability witnesses cannot form one simple route.

## Deviations and validation

- The independent review rejected deleting distinct higher-cost parallel
  routes. Exact same-endpoint physical Fiberlets are structurally impossible
  because their canonical stable ID is the endpoint-key pair. Different anchor
  variants are different graph routes and cannot be removed losslessly under
  path-dependent visited-anchor history.
- Physical macro contraction is deliberately stricter than one-successor
  detection: it requires a degree-two interior anchor and both mutual directed
  transitions. One-successor cases that converge from multiple predecessors
  remain separate graph states but receive overlapping deterministic rollout
  descriptors. Disjoint directed contraction additionally requires one
  predecessor; the measured crop had no such states after physical contraction.
- Macros are in-memory ordered references to original Fiberlets. They are not
  persisted into the ordinary route lattice and are not yet consumed by regular
  replay because encoding concatenated geometry there would require resampling.
- Build command:
  `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target vc_fiberlets test_fiberlet_storage test_fiberlet_paths -j32`
- Focused validation passed: `test_fiberlet_storage` (27 cases),
  `test_fiberlet_paths` (87 cases), and `git diff --check`.
- The build repeatedly reported a pre-existing truncated Ninja log and rebuilt
  117 targets; compilation completed successfully. Existing unrelated OpenCV
  deprecation and ignored-`nodiscard` warnings remain.
- Hot Paris4 command used the existing 128-base storage caches, a 512-base
  selected region, eight 256-base stage-one boxes, one centered 256-base
  stage-two box, and 32 threads. It reused all eight reduced chunks and stage
  two plus simplification completed in 2.5-2.6 seconds across repeated runs.
- Measured population: 13,750 original to 7,112 stage one to 4,168 stage two;
  internal 5,730 to 3,436 to 618. Post-stage-two simplification retained 1,464
  of 1,515 materialized anchors, represented 4,168 physical Fiberlets as 4,095
  physical macros (73 merged), and found 1,041 forced-continuation states.
  Forced rollout descriptors averaged 2.19 macros, median 2, maximum 4. No
  further disjoint directed chain contraction survived convergence checks.
