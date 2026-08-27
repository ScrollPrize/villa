# Plan: Signed winding belief propagation

## Scope and contracts

- Preserve current H/V and Mixed factors, probabilities, ordering, and output.
- Continue storing the existing unsigned normal-modulated winding magnitude for
  legacy diagnostics and MILP/parity consumers. Add a distinct optional signed
  `B-A` winding measurement for BP; do not silently reinterpret the old field.
- For an ordered measured constraint `A -> B`, orient the unsigned magnitude by
  the sign of `dot(B-A, aligned_normal)`. Parallel evidence targets winding
  difference zero. Perpendicular evidence targets that signed measurement.
- Normal signs are consistent only within a valid aligned-normal component.
  Constraints without one usable component remain valid for H/V but contribute
  no perpendicular winding term. A winding graph component may consume signed
  evidence from at most one normal-alignment component; otherwise inference
  fails loudly because independent sign gauges are not comparable.
- Winding has an additive gauge. Independently choose the variable whose piece
  geometry is closest to the crop center in every
  positive-weight winding component, with stable piece-index tie breaking, and
  fix it to zero. The H/V seed is reused only when it is that component's
  central choice. The normal alignment supplies a deterministic
  sign gauge, but no claim of an absolute physical outward direction.
- H/V and winding factors are independent under this model. Run them as two
  solvers over the same retained piece graph rather than constructing a
  Cartesian H/V-by-winding state space.

## Shared aligned-normal crop field

1. Move standalone manifest sampling, lattice compaction, factor construction,
   and BP alignment into a reusable core helper used by both
   `vc_lasagna_normal_align` and crop BP.
2. Retain globally anchored base-coordinate lattice semantics and expose a
   `componentByNode` identity. Crop alignment samples a one-spacing halo around
   the stored crop, clipped to manifest bounds. Constraint signing looks up the
   globally anchored nearest lattice node at connector A, midpoint, and B;
   each must be within `sqrt(3)/2 * spacing` plus tolerance, present, and in the
   same component. The midpoint aligned normal supplies the sign.
3. Build this field once immediately after opening and validating the already
   open normal dataset/sampler, before the first `extractFiberTraceConstraints`
   call. Use the normal channel's effective base spacing, host worker count,
   and the same decoded cache; never reopen the manifest.
4. Orient all nonzero perpendicular measurements after canonical extraction and
   before pruning/selection so copied reports retain identical signed data.

## Shared topology preparation

Extract the existing private piece/continuity validation, source geometry, and
crop-central selection into one reusable prepared-topology helper. Existing H/V
BP and the winding solver must consume that helper, retaining one failure
contract for duplicate or missing continuity, geometry, and soft endpoints.

## Continuous initialization

1. Keep every split piece as a separate winding variable. Insert canonical
   same-trace continuity as its existing parallel-score-1, zero-difference
   factor so conflicting neighborhood evidence can override it at a cost.
2. Build one deterministic weighted least-squares difference graph. A measured
   edge contributes `p*(delta-0)^2` and, when signed evidence exists,
   `q*(delta-d)^2`, where `p` and `q` are its complementary parallel and
   perpendicular scores and `d` is signed winding distance.
3. Fix the independently selected crop-central variable in each connected
   component to zero and solve the sparse symmetric system.
4. Report residual/error statistics and fail loudly on malformed or
   non-finite systems.

## Adaptive integer BP

1. Give every non-gauge variable the integer labels centered on the rounded
   continuous result with initial radius one. Gauge variables
   own only label zero.
2. Pairwise categorical costs use the original unsquared robust objective:
   `p*abs(delta)` plus `q*abs(delta-d)` when signed perpendicular evidence is
   available. If canonical variable order reverses an original edge,
   negate its signed target. Merge repeated variable pairs by summing
   every measurement's complete cost table; never average signed targets.
3. Run synchronous damped sum-product BP with stable factor and label order.
   Produce normalized marginal probabilities, MAP integer labels, posterior
   means, MAP confidence, and entropy.
4. If a non-gauge node's MAP label is on either boundary or its combined
   boundary probability exceeds one percent, expand that side by one integer
   and restart BP deterministically from zero messages. Continue until no node
   requests expansion. A configurable total-state resource guard may terminate
   only by throwing an explicit error; it must never publish a truncated
   posterior. Continuous centering uses `std::round` and therefore rounds exact
   half values away from zero.
5. Report each piece variable directly. No global winding range is required.
6. Reuse existing BP temperature, damping, residual, and message-iteration
   controls and the command's host worker count. A finite iteration-limit
   result remains usable but is reported explicitly as nonconverged.

## Crop BP integration and output

1. Run winding inference for every `direction-ablation --bp-only` H/V BP result
   using the exact selected constraints and `seedPieceIndex` reported by H/V.
2. Add winding columns to the existing BP consistency CSV. Each piece records
   incident signed/skipped counts, continuous value,
   integer MAP, posterior mean, MAP probability, entropy, and candidate bounds.
3. Print aligned-normal population/components, signed/skipped constraint counts,
   continuous and discrete BP convergence, expansion rounds, winding range,
   and confidence summaries.
4. Write a factor CSV containing canonical endpoints, original orientation,
   normal component, signed-target availability, and signed target.
5. Write complete-piece OBJ layers grouped by integer MAP winding label. Shift
   the arbitrary relative gauge by its global minimum for publication and name
   the consecutive layers `_w_0`, `_w_1`, and so on. Preserve both relative
   and published labels in the CSV. Also partition every winding layer into
   `_h`, `_v`, `_err`, and `_tie`. Artifact names describe their content only;
   the selected BP implementation remains in the CSV metadata rather than the
   filename.

## Spec update

Add the aligned signed-winding convention, preservation of unsigned winding,
component/gauge behavior, factorized H/V and winding solves, continuous
relaxation objective, adaptive local integer candidate sets, categorical BP
controls/semantics, output fields, and failure behavior to `planning/specs.md`.

## Docs updates

Extend `volume-cartographer/docs/fiber_chunk_tracing.md` with the crop alignment
stage, signed `B-A` convention, two-stage winding solver, gauges, adaptive
candidates, CLI-visible diagnostics, CSV columns, and winding OBJ names.

## Testing

- Reusable normal crop sampling matches standalone lattice ordering and aligned
  output; disconnected components and invalid samples retain explicit identity.
- Signed orientation is antisymmetric, keeps unsigned magnitudes unchanged,
  rejects missing/cross-component evidence only for winding, and preserves hard
  zero links.
- Continuous initialization recovers exact synthetic difference chains,
  disconnected gauges, and same-trace continuity factors.
- Integer BP recovers negative and positive labels, exposes uncertainty on
  conflicting loops, expands boundary candidates, and is invariant to stable
  constraint reordering.
- A winding component receiving signed evidence from two independently gauged
  normal components fails, while parallel-only evidence remains usable.
- Canonical endpoint reversal negates signed targets and repeated measurements
  sum complete cost tables.
- Existing binary normal-alignment, H/V/Mixed BP, constraint, and crop-trace
  tests remain unchanged and pass.
- Add a small synthetic end-to-end crop-BP regression covering alignment before
  extraction, pruning preservation, H/V output identity, signed continuous and
  integer inference, factor/piece CSV fields, and stable OBJ filenames.
- Build the affected Release targets with `-j32`, run focused test binaries,
  run `git diff --check`, and record exact results.

## Changelog

Record reusable crop normal alignment and two-stage signed integer winding BP.
