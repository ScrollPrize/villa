# Plan: Mixed-fiber direction-label ablation

## Contract

- Add a separate `direction-ablation` command beside `direction-diagnostic`.
  It reads the same stored trace artifact and compatible normal manifest and
  accepts the same direction, constraint, pruning, broken-cost, MIP-gap,
  thread, and cache controls. It forces the same ordinary discrete H/V-only
  MILP and rejects the same incompatible solver options.
- Classify the full stored population once. Preserve the initially confident
  direction-1 and direction-2 traces as the trusted cohort.
- Rank mixed traces deterministically by descending
  `max(direction1_support,direction2_support)/valid_arc_length`, then ascending
  original trace index. Ranking controls admission order only. Every mixed
  trace has the reference state Defect and is expected to optimize to Broken;
  it never receives a tentative H/V reference. Degenerate zero-length traces
  rank last.
- Ranking controls membership only. At every checkpoint, construct retained
  input by scanning the original stored traces in ordinal order and selecting
  confident traces plus the admitted mixed membership. Without an admission
  limit, the final all-admitted checkpoint must therefore be
  ordinal-for-ordinal equivalent to the full source population. Preserve
  filtered-to-original trace IDs separately.
- Run checkpoint zero with no mixed fibers, then cumulatively admit
  `--ablation-step N` ranked mixed fibers per checkpoint; the default is 5 and
  the final remainder is always included. `--ablation-limit N` restricts the
  ranked prefix for diagnostic parameter sweeps without changing its order;
  omission admits the complete mixed cohort. Every checkpoint
  rebuilds pieces and spatial constraints, applies the configured pruning, and
  solves the same MILP from canonical inputs. No previous solution is used as
  a warm-start or changes a later checkpoint.
- Solve both the discrete MILP and its H/V-only LP relaxation from the same
  checkpoint constraints. Threshold LP active and H/V values independently at
  0.5, map inactive pieces to Broken, and feed the resulting discrete labels to
  the identical comparison. Report both models and their solve times
  separately. Final MILP and thresholded-LP label OBJ families are distinct.
- Pass the original direction-1/direction-2/mixed classification and a
  same-length trusted mask for every filtered trace to comparison. Trusted
  references must be direction 1/2 and admitted references must be mixed
  defects. Resolve each active connected component's H/V gauge using
  only trusted confident pieces. Components are built from the exact
  post-pruning graph supplied to the solver, including hard links, after edges
  incident to final Broken pieces are omitted. A component without a trusted
  active piece retains the identity gauge, and an exact trusted-error tie also
  retains identity. Use that one fixed gauge to evaluate trusted, admitted, and
  combined cohorts. A confident reference optimized to Broken is an error; a
  mixed defect optimized to any active H/V label is an error, while Broken is
  correct. Legacy comparison without a mask remains unchanged and rejects
  Mixed references.
- Print one compact, stably ordered row per checkpoint containing admitted mixed count, latest
  admitted confidence, retained fibers, pieces, constraints,
  solver status/gap, objective, raw H/V/broken pieces, active components, and
  separate H/V, mixed-defect, and combined errors for both MILP and thresholded
  LP. Every active confident mismatch after gauge alignment is
  an orientation error, every Broken confident piece is a broken error, and
  every active mixed-defect piece is a defect-active error. Piece
  denominators are cohort pieces; trace denominators are represented source
  traces in that cohort; trace errors are the union of both error kinds across
  that trace's pieces. Combined counts are recomputed unions, not sums of
  rates. Checkpoint zero prints `-` for latest confidence/reference.
- Write the initial full direction family once and write the existing
  constraint and H/V/broken OBJ families only for the final selected
  checkpoint. Intermediate checkpoints are statistics-only to avoid producing
  hundreds of ambiguous artifact families.
- Empty confident or mixed cohorts are valid. No mixed fibers means one
  checkpoint equivalent to `direction-diagnostic`; all-mixed data uses identity
  gauge until admitted references populate otherwise unanchored components.
  Checkpoint zero on all-mixed data is a valid empty solve/comparison. If every
  trusted piece is Broken, active untrusted components remain unanchored.
- Classify and rank once. Checkpoints and memberships are stable, solver status
  and gap are reported, and changing the worker count must not change retained
  order or reference assignment. A non-optimal but usable solver status remains
  visible rather than being silently compared as if optimal.
- The default output base is sibling `<trace-stem>_direction_ablation` and the
  initial family is exactly `<base>_initial.obj` plus existing direction and
  anchor suffixes. The command requires normal-manifest/remote-cache handling,
  allows MIP gap and H/V-valid no-winding-cutoff, and rejects explicit
  `--hv-only`, LP controls, exact-perpendicular mode, and link exclusion.

## Implementation

1. Extend reusable label comparison with an optional same-length trusted-trace
   mask, explicit direction-versus-defect validation, and
   trusted/admitted/combined summaries while preserving the current diagnostic
   behavior when no mask is supplied.
2. Extract the existing classify/filter/extract/prune/solve operation into a
   shared local command helper used by both direction commands.
3. Add deterministic mixed ranking, cumulative checkpoint execution, compact
   tabular reporting, and final-only artifact emission.
4. Add focused tests for ranking ties/degenerates and near ties, stride/final
   remainder membership, LP threshold boundaries, mixed-as-defect acceptance
   and active-defect errors, trusted-only
   gauge choice/ties, untrusted-only and trusted-all-Broken components,
   Broken-induced component splits, cohort denominators and trace unions,
   interleaved filtered/original mapping, final original-order equivalence,
   checkpoint-zero sentinels, empty cohorts, final-only artifact names/no
   intermediate writes, legacy comparison behavior, repeatable rows, and CLI
   option/default-basename validation.

## Spec Update

Document mixed confidence/ranking, mixed-as-defect references, trusted-only
gauge selection, checkpoint semantics, cohort error denominators, and
final-only artifacts in `planning/specs.md`.

## Docs Updates

Add a runnable `direction-ablation` example and explain the checkpoint table in
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Testing

- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`.
- Run the focused crop-trace tests and `git diff --check`.
- Run the centered-384 artifact at dominance 0.90 through five-fiber cumulative
  checkpoints and report MILP versus thresholded-LP H/V and mixed-defect errors
  plus solve time against admitted count. First sweep broken costs on a fixed
  ten-mixed-fiber prefix, then use a selected value for the complete ablation.

## Changelog

Record the deterministic cumulative mixed-fiber ablation command.
