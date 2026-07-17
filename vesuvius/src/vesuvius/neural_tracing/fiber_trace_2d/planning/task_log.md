# Native 3D Trace2CP Normal-Aware Smoothness Log

## Implementation

- Added a native candidate normal sampler wrapper in
  `fiber_trace_3d.trace2cp_tool` that converts selected-level ZYX candidate
  points to base ZYX and calls the existing batched Lasagna normal sampler in
  the 2D geometry loader. The wrapper converts returned XYZ normals to native
  ZYX for trace scoring.
- Fixed the wrapper to use the matching 2D geometry-loader record for Lasagna
  `grad_mag/nx/ny`, while using the 3D trace record only for selected-level to
  base-coordinate scaling. This avoids passing a 3D `_Record` without Lasagna
  channels into the 2D normal sampler.
- Added split smoothness scoring:
  - tangent-plane turn around the Lasagna normal axis;
  - normal-tilt turn into/out of the Lasagna normal direction;
  - normal sign ambiguity is invariant by construction;
  - invalid candidate normals fall back to the previous isotropic smoothness
    term for that candidate.
- Threaded the normal sampler through greedy, beam, pair, and whole-fiber
  native 3D Trace2CP paths.
- Added CLI/config fields:
  `--smoothness-tangent-weight` and `--smoothness-normal-weight`. In the
  native CLI, omitted split weights default to `--smoothness-weight` when the
  Lasagna normal sampler is active.
- Added summary JSON fields for the effective smoothness weights and whether
  normal-aware smoothness was active.
- Updated `planning/specs.md`, `docs/code_structure.md`, and
  `planning/changelog.md`.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `91 passed in 8.19s`.
- `git diff --check`
  - Result: clean.

## Deviations / Deferred

- Candidate-substep scoring still uses the final candidate endpoint normal for
  the one-per-step smoothness term. Direction/presence scoring still samples
  all configured substeps.
- Candidate normals are not used to generate candidate directions yet; they
  only affect the smoothness penalty.
