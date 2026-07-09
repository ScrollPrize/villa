# Task Plan: Complete Current-State Documentation

## Task

Implement the TODO item from `planning/todo.md`:

- make docs a more complete documentation of the current state

## Scope

In scope:

- Document the implemented `fiber_trace_2d` loader architecture at a level useful for planning and follow-up coding.
- Document module responsibilities, data flow, config keys, sampling/cache behavior, augmentation behavior, debug/export outputs, tests, and known local workflow constraints.
- Keep the docs aligned with current code instead of aspirational future training work.
- Update planning status and task log according to the subproject process.

Out of scope:

- Code behavior changes.
- New loader features.
- Performance optimization.
- Test expectation changes.
- Volume-cartographer or Lasagna implementation changes.

## Implementation Steps

1. Read current docs and implementation files:
   - `docs/code_structure.md`
   - `planning/specs.md`
   - `planning/local_development.md`
   - `loader.py`, `sampling.py`, `strip_geometry.py`, `augmentation.py`, `runner.py`, and `loader_support.py`

2. Expand `docs/code_structure.md` with:
   - current module map;
   - config schema and key meanings;
   - dataset construction and Lasagna manifest validation;
   - deterministic sample selection;
   - side-strip coordinate construction;
   - VC3D blocking coordinate sampler and chunk prefetch;
   - augmentation and contact-sheet behavior;
   - runner commands and exported files;
   - test coverage and local development caveats.

3. Keep `planning/specs.md` as the normative behavior list, but add a short documentation-specific note only if the current docs contract needs to be explicit.

4. Update `planning/status.md` for this docs task.

5. Append `planning/task_log.md` with what changed and how it was validated.

6. Mark the completed docs TODO in `planning/todo.md`.

## Spec Update

Add a lightweight documentation-maintenance spec:

- `docs/code_structure.md` should describe current implemented behavior and module ownership, while `planning/specs.md` remains the normative behavior source.
- Future behavioral changes should update both specs and code docs when they affect public configuration, data flow, sampling, caching, augmentation, runner outputs, or local workflow.

No runtime semantics should change.

## Docs Updates

Update:

- `docs/code_structure.md`: complete current-state code documentation.
- `planning/status.md`: docs task checklist.
- `planning/task_log.md`: docs task notes and validation.
- `planning/todo.md`: mark/remove completed docs TODO.

## Testing

Since this is a docs-only task:

- Run `git diff --check` to catch formatting/trailing-space issues.
- Do not rerun data-loading or GPU commands unless a doc claim requires validation.

## Review Checklist

- Does the documentation describe current code, not future plans?
- Does it avoid implying unsupported modes?
- Are command examples consistent with the local development notes?
- Are config keys and runner outputs discoverable from docs?
- Did planning status/log/todo get updated?
