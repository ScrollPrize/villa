# R3 Python evidence fixes

Scope: PY-R3-1 and PY-R3-2 from [the review](review-r3-python.md). Reuse selected source references and frozen-source validation; do not introduce a second identity scheme or restore unverified old evidence.

- [x] Match producer records complete selected fiber reference; index validates that reference before interpreting old matches against current geometry. Existing fully materialized sheet-run consumers remain usable.
- [x] H/V preparation checks the constraints' recorded source version through the same guard used by other frozen-evidence readers, including a source change during the read.
- [x] Regression tests cover missing identity, same-length translation and unchanged valid inputs. Record exact test results here.

Mini playbook: target `vesuvius` Python workflows and shared fiber-store adapter; use the existing Python 3.14 test environment and local vcobj/reader builds, with no dependency installation in this lane. Preserve numerical processing and output order. Fixtures are pytest temporary directories.

Implementation: extend `fiber_catalog` with the existing frozen-source option and share `_require_source_version` with `read_polylines`; H/V consumes `provenance.fiber_source`. Store-fiber JSONL matches now carry the already selected per-object reference, not a new identity scheme or a copied source-ID directory. Index validation precedes interpretation of matched positions. Older unversioned matches/constraints must be regenerated; they are not silently recertified. No schema, solver, geometry processing or sampling changes.

Validation: 31 passed with local native libraries restored by the parent task. This includes actual producer → index → training-consumer acceptance, rejected same-length translation/missing reference/wrong owner in both history modes, H/V fresh-features/old-constraints rejection, and source edits before/during both frozen reader APIs. One test-edit insertion error initially produced six `NameError` failures after the new regression assertions; the misplaced existing assertion was restored and the complete command passed.

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=local/vcobj-pr-tests/build/python:local/vcobj-pr-tests/reader-python:vesuvius/src:spiral-fitting:local/vcobj-pr-tests/spiral-build:lasagna:. /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tests/test_fiber_store_versions.py vesuvius/tests/test_fiber_hv_stage.py vesuvius/src/vesuvius/neural_tracing/tracers/sheets/tests/test_geometry.py::test_fiber_patch_matching_keeps_fibers_on_or_inside_the_sheet
```

No full training/volume workload or macOS execution was performed; these tests isolate the provenance boundary without changing numerical processing. Existing shared catalog/read guard tests cover the mid-read source mutation. No core/native files edited in this lane.

Additional sibling regression run: **37 passed** (`test_fiber_store_sources.py`, `test_sheet_strip.py`, `test_sheet_strip_io.py`, `test_fiber_curve_bank.py`, `test_fiber_normal_sign_visualization.py`) with the same command environment. Total focused coverage: **68 passed**. `git diff --check` passed. Production Python delta: 33 added / 16 removed (**net +17**); test delta: 88 added / seven removed (**net +81**), excluding documentation. The small production growth adds missing evidence fields/checks and exposes the existing shared guard to catalog readers; it does not add duplicate store resolution.

Delta review of coordinate provenance: this adapter converts only the selected record's declared volume and `scale_factor`, using shared `geometry_scale_to_volume`; it does not apply volume-registration routes. A registration-only update therefore cannot change its match points. Added regressions verify that behavior and reject metadata-only coordinate-scale edits through the complete reference in both history modes. The focused command now passes **34 tests**, plus the 37 sibling tests above (**71 total**). No extra whole-store guard was added to match runs for an unused mapping path. H/V's frozen extraction snapshot remains conservatively guarded against all source commits because its native extraction inputs use that recorded snapshot.
