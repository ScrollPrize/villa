# R4 Python validity and documentation fixes

Baseline `2b5d1e01f`. Scope: independently confirmed P-001 and P-002. No schema, identity, numerical tolerance, dependency or source-routing changes. Lasagna is a Python consumer of the shared vcobj/tifxyz readers; tests use the existing Python environment and native modules, not a new build or installation.

- [x] P-001: `StoredSurface` retains `SurfaceGrid.valid`. Mesh fitting and approval inpaint carry that mask rather than deriving validity from already transformed coordinates. Approval scaling transforms only source-valid points and keeps both the mask and invalid coordinates unchanged. Local tifxyz loading uses the existing reader's validity too (`mask=False` remains unchanged), removing the duplicate coordinate TIFF/shape/validity handling. This aligns local predicates with the canonical reader rule instead of retaining another interpretation.
- [x] P-002: the region-query paragraph now distinguishes metadata-based family filtering from measured selected-geometry minimum length. The existing earlier length contract is unchanged.
- [x] Permanent regression: actual stored surfaces, history on/off, current and complete exact references, an explicitly masked vertex, a source-invalid sentinel and a valid source point transformed to x = -1. Both mesh and approval loaders preserve the shared mask, invalid fit points are zeroed, and subsequent approval scaling preserves validity. History-on also checks the retained reference after a metadata edit.

## Validation

Linux, Python 3.14, existing local native bindings. All test fixtures use automatically cleaned temporary directories.

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=lasagna:local/vcobj-pr-tests/reader-python:local/vcobj-pr-tests/build/python /home/hendrik/.venv_las/bin/python -m pytest -q -rs lasagna/tests/test_stored_surfaces.py lasagna/tests/test_approval_inpaint.py lasagna/tests/test_volume_scale.py
AGENTS_AGENT_MODE=1 PYTHONPATH=lasagna:local/vcobj-pr-tests/reader-python:local/vcobj-pr-tests/build/python /home/hendrik/.venv_las/bin/python -m pytest -q lasagna/tests/test_fit_service_approval_inpaint.py lasagna/tests/test_fit2tifxyz_output_mask.py lasagna/tests/test_approval_inpaint_configs.py
```

First command: 30 passed, one skipped, six subtests passed. Second: 48 passed, three subtests passed. Total: **78 passed, one skipped, nine subtests passed**. The source/reference variants do not claim a new physical referenced-store fixture: the Python API used here has no collection-creation binding; shared native routing coverage remains separate. No macOS, interactive GUI, training or real-volume result is claimed. `git diff --check` passed for the changed files.

Production Python delta: 21 added / 26 removed, **net −5 lines**. Regression delta: 33 added / two removed, net +31. The two extra helper signature edits allow testing history on/off. Documentation is separate.
