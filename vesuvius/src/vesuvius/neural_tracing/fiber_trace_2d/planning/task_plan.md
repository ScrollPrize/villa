# Plan: robust sampled-direction anchor refinement

## Follow-up Performance Checkpoint

1. Treat commit `d89b0aba0` as the behavior and performance baseline. Its
   idle-host canonical replay takes 18.17 seconds wall: 11.59 seconds in anchor
   extraction and 5.95 seconds in fiberlet extraction.
2. Optimize `robustDirectionProposal()` without changing assignment, trimming,
   direction, or spatial-objective semantics:
   - store each assigned observation's residual bin temporarily in the existing
     retained-membership byte;
   - derive the weighted deviation histogram directly from the residual-mass
     histogram instead of rescanning observations;
   - accumulate symmetric direction tensors by component and residual bin in
     the assignment pass, then combine retained bins after choosing the cutoff;
   - avoid a separate normalized-direction or Gaussian-mass memory stream.
3. Let the final post-iteration membership refresh omit principal-axis tensor
   construction because that call consumes only assignments and retained
   membership; the fitted axes are deliberately fixed at that point.
4. Increment the extraction-profile schema because tensor-visit counters now
   describe the reduced implementation work. Add focused equivalence coverage,
   run GCC and Clang focused tests, and compare canonical anchor/fiberlet
   populations and replay failures before accepting the checkpoint.
5. Measure this checkpoint before considering the independent follow-ups:
   pre-normalized per-cell observations, tile-level presence-gradient reuse,
   or peak-search optimization.

## Baseline And Invariants

1. Use the current version-3 canonical 5,000-base-voxel replay profile as the
   performance baseline. Preserve cell-level parallel extraction, input
   sampling, seed generation, component-count limit, deterministic iteration,
   coordinate systems, support/selection/NMS, and file formats. Replace the
   existing pre-refinement same-direction merge because it removes close modes
   before robust competition can evaluate them.
2. Explicitly relax exact anchor numeric/artifact identity for this experiment.
   Preserve deterministic repeatability for identical inputs and compare
   geometric and downstream quality metrics instead.
3. Keep fiber direction axial/signless throughout: use `abs(dot)` or `dot^2`,
   so `d` and `-d` are equivalent.
4. Record the identifiability limit explicitly: without a network uncertainty
   output, a blended prediction that lies inside a component's coherent angular
   core cannot always be distinguished from a genuine intermediate direction.
   Spatial competition, robust tails, downstream metrics, and visual review
   mitigate this but do not remove the fundamental ambiguity.

## Competitive Observation Assignment

1. Begin each local iteration from the existing one- or two-component sampled
   direction estimates and positions.
2. For every valid observation and active component, calculate:

   ```text
   base_mass = presence * spatial_gaussian(observation, component_line)
   direction_alignment = abs(dot(sample_direction, component_axis))^2
   assignment_score = base_mass * direction_alignment
   ```

3. Hard-assign the observation to the component with the highest positive
   score. Preserve stable component-index tie-breaking. Hard competition avoids
   soft sharing that would pull nearby components toward the same blended
   direction.
4. Recompute assignments after each direct direction/spatial state update.
5. Feed every unique initialized seed component into robust refinement. Do not
   merge candidates solely because their initialized axes are within the
   existing 10-degree threshold. Preserve two surviving, sufficiently
   supported robust components even when their refined directions are close;
   let existing spatial/directional NMS remove genuine duplicate anchors later.
   Record the retired pre-refinement merge transition explicitly in specs and
   migrate affected diagnostics/tests.

## Adaptive Robust Direction Aggregation

1. Within each component, calculate projective angular residual
   `1 - direction_alignment` for assigned observations. Weight residual
   statistics by `base_mass`, not observation count.
2. Use a fixed-size deterministic weighted histogram over residual `[0,1]` to
   estimate the weighted median and weighted MAD without per-cell sorting.
   Use the same histogram bins to accumulate direction outer-product tensors,
   avoiding another full tensor pass after selecting a cutoff.
3. Define an adaptive outlier cutoff from weighted median plus a configurable
   MAD multiplier, with a configurable minimum angular-noise floor. Proposed
   defaults for the first measured experiment are:
   - maximum trimmed evidence-mass fraction: `0.20`;
   - MAD multiplier: `3.0`;
   - minimum angular-noise floor: `5 degrees`;
   - internal histogram resolution: `256` bins.
   Convert the angular floor to residual units as `sin(floor_angle)^2` and use
   `max(weighted_median + multiplier * weighted_MAD, floor_residual)` as the
   candidate cutoff.
4. Map residual `r` to
   `min(255, floor(clamp(r, 0, 1) * 256))`. Histogram bins are half-open except
   the final bin, which includes residual 1. Estimate medians from bin centers;
   retain the entire bin containing either the adaptive or mass-cap cutoff.
5. If no assigned mass lies beyond the adaptive cutoff, retain every assigned
   observation. Coherent single-direction components therefore use all data.
6. If candidate outliers exceed the configured maximum trimmed-mass fraction,
   raise the cutoff to the weighted
   `(1 - maximum_trimmed_mass_fraction)` quantile and retain the complete
   boundary bin. At the default maximum `0.20`, this is the weighted 80th
   percentile. This guarantees the configured minimum retained mass; trimming
   is a maximum budget, never a mandatory amount.
7. Calculate the new component axis as the principal axis of
   `sum(base_mass * d * d^T)` over retained observations and install it directly.
   Do not interpolate old and new axes and do not gate direction installation
   on positional objective improvement.
8. Use the existing `principalFiberAxis()` finite/positive/unique eigenvalue
   criteria unchanged. If the retained tensor fails those criteria, remove the
   component
   from the active local mixture before subsequent evaluation. Compact
   surviving components in stable original-component order while preserving
   diagnostic parent IDs. If none survive, return a degenerate/empty refined
   result; never restore a previous direction.
9. Carry the final robust assignment and retained-inlier membership into
   centroid fitting, the local spatial objective, and the
   direction-conditioned peak objective. Their positive numerators use only
   retained component evidence. Their geometric denominators use every sampled
   lattice site independent of presence, direction, assignment, or trim state;
   this avoids creating normalization holes at rejected positive observations.
   Recompute robust membership only as part of a subsequent outer
   direction/spatial state iteration; do not perform a fresh direction-only
   assignment during peak refinement.
10. Keep final published support/confidence normalization distinct from fitting:
   use retained inliers for aligned/presence numerators, while preserving the
   existing all-site geometric Gaussian denominator used by support thresholds.
   Trimmed observations may lower confidence as unsupported sites but cannot
   alter the fitted direction or position. Document and test this distinction.

## Position-Only Backtracking

1. For each directly updated axis, project and clamp the current position into
   its new transverse plane. Evaluate this fixed-direction state as the spatial
   baseline.
2. Calculate the centroid target from the same retained component evidence.
   Interpolate only from the projected baseline position toward that target;
   axes remain fixed for every spatial candidate.
3. Test fractions `1, 1/2, ...`. Accept the first candidate that improves the
   fixed-direction baseline objective. If no candidate improves, retain the
   baseline.
4. If no earlier candidate is accepted, include the first fraction whose
   maximum Euclidean displacement from the projected baseline across active
   components is at most `peakGridStepPredictionVoxels` (0.5 by default), then
   stop. If the full target is already within the threshold, test it once.
   Retain the current eight-halving bound as a defensive limit.
5. Keep a bounded outer direction-assignment/update pass count. A direct
   direction update is installed even when no spatial candidate improves. The
   canonical experiment established two passes as the default: four and 64
   passes did not improve replay failure counts and substantially increased
   runtime. Do not require exact assignment or histogram-membership equality,
   because boundary samples can flicker without a meaningful geometry change.
   An earlier exit may use projective axis stability and accepted position
   movement no larger than the peak-grid step. Record limit behavior; do not
   add angular damping. The existing later peak search remains the only
   sub-grid positional refinement and must consume final retained memberships.

## Configuration And Diagnostics

1. Add validated anchor configuration and `vc_fiberlets` CLI controls for the
   maximum trim mass, MAD multiplier, and minimum angular-noise floor. Persist
   them in extraction/replay metadata and print effective values in summaries.
   Validate maximum trim mass in `[0, 0.20]`, MAD multiplier as finite and
   nonnegative, and angular floor as finite in `[0, 90]` degrees. A zero trim
   budget disables removal while retaining competitive assignment.
2. Extend profiling with deterministic counts/masses for:
   - components with no detected outliers;
   - components with adaptive trimming;
   - candidate and actually trimmed evidence mass;
   - retained evidence mass;
   - components removed for non-unique direction;
   - spatial candidates tested and accepted by depth;
   - iterations reaching the hard limit.
3. Keep existing phase timings and logical visit counters and advance emitted
   diagnostics to `fiberlet_extraction_profile version=6`. Aggregate every new
   field through worker profiles and cover aggregation in tests.

## Implementation Structure

1. Extract reusable internal helpers for deterministic weighted-histogram
   cutoff selection and spatial-fraction scheduling. Production and tests must
   use the same helpers; do not duplicate private logic in tests.
2. Keep observation storage compact. Do not add a per-observation expanded
   direction cache or per-iteration sort; previous measurements showed that
   increased working-set traffic can erase arithmetic savings.
3. Separate component identity from compact active-array index so dropping a
   degenerate component cannot corrupt diagnostics or initialize a survivor
   under the wrong parent ID.
4. Propagate the refined active-component count and stable identity mapping to
   peak search, final evaluation, result construction, support/selection, and
   diagnostics. Removed components receive an explicit degenerate transition
   and cannot be visited by later loops using the original component count.

## Tests

1. Adaptive trimming unit fixtures:
   - a coherent single mode with deterministic +/-2-degree noise retains 100%
     of its evidence mass;
   - a 10%-mass tail at least 25 degrees from the mode is detected under
     default settings;
   - maximum trim settings `0`, `0.10`, and `0.20` retain at least 100%, 90%,
     and 80% weighted mass respectively;
   - a weighted boundary bin is retained completely;
   - sign-flipped directions produce the same axis and cutoff;
   - zero/invalid/non-finite evidence cannot affect the aggregate.
2. Multi-component fixtures:
   - two nearby directions remain distinct under deterministic competition;
   - ambiguous blended samples do not collapse both components;
   - a non-unique retained tensor removes only that component;
   - zero-mass, isotropic, exact-tie, and near-tie tensors exercise the existing
     `principalFiberAxis()` uniqueness criteria;
   - component compaction preserves stable diagnostic ancestry;
   - trimmed observations stay excluded from centroid, local spatial objective,
     and peak response, while final support retains its documented all-site
     denominator;
   - equal-mass, spatially overlapping modes initialized at 5, 9, 10, and 11
     degrees are exercised; supported modes must survive pre-refinement and
     finish within a stated 1-degree projective error of their generating axes.
3. Direction/spatial semantics fixtures:
   - initialized and newly aggregated axes differ, and the update equals the
     retained sampled-direction tensor axis rather than an interpolation;
   - positional fractions never change axes;
   - scheduling includes the first displacement at or below 0.5 voxel and no
     finer candidate;
   - no improving spatial candidate retains the projected baseline.
4. Build `vc_fiberlets`, `test_fiber_anchors`, `test_fiberlet_paths`, and
   `test_fiber_replay` in the existing GCC RelWithDebInfo tree; run focused
   CTest and `git diff --check`. Run the focused Clang build/tests if the final
   implementation introduces nontrivial template or portability-sensitive
   histogram code.

## Performance And Quality Measurement

1. Run the old and new optimizers symmetrically with the same inputs, commit
   identities, compiler/build flags, host, 32 threads, decoded/disk/OS cache
   state, and fresh output directories. For each variant run one unmeasured
   warmup followed by three measured runs with `--threads 32` explicitly.
   Report wall and process CPU plus versioned anchor/fiberlet subphase
   min/median/max.
2. Compare against the current optimizer:
   - selected/work cells and retained anchors;
   - trimmed/retained evidence and removed components;
   - local iterations, spatial evaluations, and hard-limit hits;
   - accepted fiberlets and graph node/edge counts;
   - greedy and fiberlet replay failures/reference coverage;
   - axis-angle and position-displacement distributions for matched anchors;
   - deterministic hashes across the three new runs, without requiring a match
     to the old optimizer.
3. Emit the normal replay artifacts needed for user visual inspection. Report
   performance separately from quality and do not declare the algorithm
   acceptable until the user checks the resulting geometry.

Canonical command, with a fresh output directory per run:

```bash
/usr/bin/time -f 'PERF_TIME wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/fiberlet-perf/bin/vc_fiberlets \
  fiberlet-replay \
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json \
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json \
  /tmp/fiberlet-replay-robust-direction-N \
  --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
  --threads 32 \
  --length 5000
```

## Spec Update

- Replace angular line-search semantics with direct robust aggregation of
  competitively assigned sampled directions.
- Remove the pre-refinement 10-degree merge; supported close robust components
  survive to ordinary downstream NMS.
- Specify adaptive weighted median/MAD outlier detection, the maximum 20%
  trimmed-mass guarantee, all-data behavior for coherent components,
  deterministic ties/boundaries, and removal of directionless components.
- Specify position-only backtracking through the first candidate at or below
  peak-grid spacing and subsequent peak refinement ownership.
- Specify retained-inlier fitting/peak objectives separately from the existing
  all-site final support denominator.
- Amend the prior blanket exact-numerics rule for this anchor-fitting change:
  old/new artifacts may differ, while repeatability, geometric distributions,
  downstream replay metrics, and visual review become the acceptance gates.

## Documentation And Changelog

- Document robust direction aggregation, spatial-only refinement, parameters,
  diagnostics, and failure semantics in `volume-cartographer/docs/fiberlets.md`.
- Replace `task_log.md` with implementation decisions, deviations, commands,
  measurements, and quality comparisons; update `status.md` incrementally.
- Add a concise changelog entry only after implementation and validation.
