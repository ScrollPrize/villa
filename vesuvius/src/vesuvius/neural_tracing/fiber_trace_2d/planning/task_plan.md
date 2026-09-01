# Task Plan

## Warm-start model

1. Add an optional per-piece joint-grid initialization carrying active/Defect,
   fixed-prepass H/V, integer winding, and component phase sign. It is
   initialization only: it does not add a unary, fix a variable, or change the
   ordinary objective. This experiment therefore probes Defect/winding/sign
   basins; fixed-prepass H/V cannot move.
2. Map the piece initialization into the prepared ordinary nodes, validate
   fixed-orientation and component-sign consistency, and normalize each
   ordinary integer component into its existing gauge without changing any
   relative latent differences.
3. Expand initial integer support to include the normalized seed state. Require
   fixed half-step phase for an unambiguous latent-coordinate gauge, validate
   hard continuation and hard sign compatibility, and explicitly use the first
   active node as the integer origin when an ordinary gauge node is Defect.
4. Initialize each ordinary factor-to-variable message from the factor
   potential conditioned on the seeded state of its opposite endpoint, and
   initialize component-sign messages from the seeded endpoints. Then run the
   unchanged production message updates, support expansion, projection, and
   decoding with no seed prior remaining.

## Experiment

1. Run the existing cold ordinary solve and exact fixed-reference conditioned
   solve. The seed is the conditioned solver's published, hard-projected MAP,
   not its private converged message buffers.
2. Slice the conditioned ordinary MAP states into a warm-start vector and
   discard every reference node and cross-factor. Run a neutral-message control
   with the expanded warm support, then a conditioned-message solve with the
   identical support. Both use the exact original ordinary graph, configuration,
   and weights.
3. Report convergence for all three solves and compare the warm result against
   both the conditioned seed and cold baseline: active/Defect, H/V, and latent
   winding changes. Use the same ordinary gauge normalization so representation
   changes are not counted as physical changes.
4. Report the cold-to-neutral support effect separately from the
   neutral-to-warm initialization effect. Report whether the warm solve returns
   to the neutral expanded-support solution, retains the conditioned solution,
   or converges to a third fixed point. Mark the conclusion unresolved unless
   every solve reaches the residual criterion.

## Tests

1. Verify an empty initialization reproduces the cold solve exactly.
2. Verify support-only and conditioned-message runs use identical expanded
   domains, while a valid warm initialization remains free to leave its seed.
3. Verify invalid size, orientation, phase sign, hard-continuity, and hard-sign
   seed states are rejected, including a Defect endpoint that neutralizes a
   hard sign.
4. Build Release `vc_fiber_trace_chunk` and the winding BP tests, run the
   focused test executable, run `git diff --check`, and execute the 1024
   reference experiment once.

## Spec Update

Document that joint-grid warm starts affect only initial support and messages,
preserve the ordinary objective, and normalize integer gauges before seeding.

## Documentation Update

Document the fixed-reference release experiment and its three-way comparison
in `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog

Record the reference-conditioned warm-start experiment and observed basin
behavior.
