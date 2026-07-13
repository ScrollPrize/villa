# Trace2CP Side-DP Z Smoothness Plan

## Interpretation

- The side/joint Trace2CP DP should not prefer less z travel.
- The side/joint DP should only regularize abrupt changes in z step
  (`dz_current - dz_previous`), which allows steady lateral/z motion.
- The lower generic DP helper can keep its existing default behavior for the
  top-model DP diagnostic; this task targets the side/joint DP wrapper.

## Implementation

- Add side-DP-specific constants:
  - `z_transition_penalty = 0.0`
  - `dz_smooth_penalty = 0.05`
- Use those constants when `_trace_score_trace2cp_joint_dp_bidirectional` calls
  `_trace2cp_top_monotone_direction_path_z`.
- Leave regular stepwise Trace2CP and the side/top-z experiment unchanged.

## Spec Update

- Update the side-strip DP spec to state that z movement has no default
  absolute transition penalty.
- State that side-strip DP uses dz smoothness to discourage z-step jitter.

## Docs Updates

- Update `docs/code_structure.md` side-DP description with the same z
  regularization semantics.

## Tests

- Add a wrapper-level regression test proving side/joint DP passes zero
  `z_transition_penalty` and nonzero `dz_smooth_penalty`.
- Run the focused DP routing/parameter tests.
- Run the full `test_fiber_trace_2d_loader.py` file.

## Changelog

- Add a dated note that side/joint Trace2CP DP now uses dz smoothness instead
  of a per-step z transition penalty.
