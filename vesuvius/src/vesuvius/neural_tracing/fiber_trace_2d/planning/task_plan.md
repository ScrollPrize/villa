# Plan: exact float-cache fiberlet replay

## 1. Canonical scoring and geometry

1. Expose one canonical batched fiberlet scoring-point sampler from the existing
   `FiberPaths` interpolation implementation. It must use the same stored-grid
   corners, prepared tensors, closed-form principal-axis resolution, float32
   arithmetic, and Lasagna normal sampling as DP endpoint materialization.
2. Evaluate that sampler for the accepted anchors in each generated anchor
   chunk and persist prediction direction, presence, both prediction-validity
   bits, normal, and normal validity. Assert that independently materialized DP
   endpoint samples have the same float bits for shared anchors.
3. Persist each accepted fiberlet's authoritative eager float path length and
   exact first/final nonzero **base-space** steps. Eager scales endpoint points
   to float32 base coordinates before subtraction; do not subtract in prediction
   space and scale afterward. Reconstruct committed routes with the same
   epsilon-based adjacent duplicate suppression used when DP output is finalized.

## 2. Exact costs and graph adapter

4. Replace the float prefix's scalar total with all five `FiberletPathCost`
   components. Preserve their float32 bits and summation order. Compact storage
   may continue to quantize a scalar total because it is intentionally lossy.
5. Make cached `arc()` return the persisted steps and full cost. Make cached
   `transition()` use the persisted shared-anchor scoring sample and the same
   `fiberLocalMetricCost` call as eager graph construction; remove the separate
   prediction/normal resampling path.
6. Replace the unpublished strict payload magic/schema and cache algorithm
   identity. Do not decode prior payloads.
7. Make replay diagnostic node/edge/arc/transition indices canonical functions
   of stable IDs in both graph adapters. Adapter-local encounter indices must
   not obscure otherwise identical replay JSON.

## 3. Verification

8. Extend codec tests for bit-exact scoring, endpoint-step, and cost-component
   round trips. Add direct tests that reconstructed routes suppress the same
   duplicate points and cached joins match eager joins.
9. Build with `-j32` and run storage, path, graph/replay, cache, and anchor
   tests.
10. Run identical cold eager and cached 5,000-base-voxel Paris4 corridors.
   Require equal populations and byte-identical `fiberlet` replay JSON, then
   Also compare stable edge IDs, path-length/step/cost bits, transition
   eligibility/cost bits, route IDs and route-point bits. Report wall/CPU/RSS
   and any remaining performance difference.

## Spec and documentation

Document float-cache transparency, canonical anchor scoring ownership, exact
prefix steps/costs, duplicate-free route reconstruction, and the intentionally
lossy compact profile. Record the schema replacement and measured equivalence
in the changelog and task log.
