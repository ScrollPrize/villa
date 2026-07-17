# Native 3D Trace2CP Sampled CP Start Direction Plan

## Implementation

- Add a start-specific direction sampler for native 3D Trace2CP:
  - sample all direction branches at the CP;
  - align each ambiguous axis to the CP-local tangent toward the target;
  - choose the valid branch with highest directional agreement to that tangent;
  - do not weight this start branch choice by presence.
- Use the sampled/aligned CP direction as the initial `previous_direction`,
  `history_direction`, and cone axis in both greedy and beam search paths.
- Remove production first-step relaxation:
  - no zeroed first-step smoothness;
  - no first-step tangent-plane neutralization;
  - no first-step normal/elevation gate replacing the regular direction gate.
- Keep later-step behavior unchanged:
  - sample current direction against previous accepted step;
  - score candidate direction, candidate sampled direction, presence, branch
    choices, and normal-aware/cumulative smoothness as before.

## Spec Update

- Update native 3D Trace2CP specs to state that the CP-local tangent is only a
  reference used to select/sign-align the sampled start direction.
- Remove the previous first-step relaxation language from specs.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP section with the new start
  direction semantics.
- Add a changelog entry for the start-direction behavior change.

## Tests

- Update focused 3D Trace2CP tests:
  - start direction chooses the sampled branch most aligned to CP tangent,
    independent of branch presence;
  - first candidate step applies regular smoothness and direction scoring;
  - obsolete first-step normal-gate/relaxation expectations are removed.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check` over touched files.

## Non-Goals

- Do not change candidate generation, beam width/lookahead defaults, native
  whole-fiber rendering, or trace fusion.
