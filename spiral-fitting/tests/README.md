# Spiral test suite

The suite focuses on fitting results and the workflows used to produce them:

- Numerical values and gradients, including native/Triton reference comparisons,
  winding seams, masked samples and deterministic sampling.
- Checkpoint compatibility, optimizer state and resumed training.
- Loading and changing fit inputs, influence masks and configuration at run boundaries.
- Service initialization, run/export, uploads, retries and ownership.
- Input snapshots, revision conflicts, atomic commits and safe workspace cleanup.

Use the existing environment, from the repository root:

```sh
spiral-fitting/.venv/bin/python -m pytest -q spiral-fitting/tests
```

CUDA tests run when CUDA and Triton are available. For CPU and localhost HTTP
tests only:

```sh
CUDA_VISIBLE_DEVICES='' spiral-fitting/.venv/bin/python -m pytest -q spiral-fitting/tests
```

The native `vc_spiral` extension is required. HTTP tests need localhost sockets.
Shared fixtures live in `*_fixtures.py`; reusable helpers should live there
instead of importing another module's test cases.

Real-scroll fits remain opt-in: `RUN_GOLDEN=1` enables the golden and structural
rebuild checks (use `GOLDEN_RUN_SPEC` for a local dataset specification),
`SPIRAL_INTERACTIVE_E2E=1` enables the interactive smoke, and
`SPIRAL_REVISION_LIVE_DATASET=/path/to/dataset` enables live patch revisions.
Run these in separate pytest invocations; their modules document the inputs.

## Keeping the suite compact

Add a test for a meaningful fitting result, user workflow or observed regression.
Prefer an independent numerical reference or a representative workflow over
another permutation of existing validation. Extend an existing case when it
already exercises the behavior. Do not add tests for source text, private
attribute layouts, help wording, or the absence of retired APIs. Remove obsolete
tests rather than leaving them permanently skipped.

Keep each behavior at the layer that best demonstrates it. For example, runner
tests need one invalid-config case to check validation is connected; config
tests own the detailed value checks. Check upload contents and API-key reuse in
the upload/restart workflows rather than repeating their setup in unit tests.
Parameterize distinct code paths, not extra random seeds or flags a backend
ignores (RK4 coalescing only changes the Cartesian kernel).
