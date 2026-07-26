# 3D Branch Choice Grid Routing Update Plan

## Current State

- Positive two-branch routing computes a detached per-positive-point score:
  `abs(dot(decoded_branch_dir, target_axis)) * branch_presence`.
- Training calls `compute_losses(..., enforce_branch_min_fraction=True)` through
  `_forward_loss(..., backward=True)`.
- Test/eval calls the same loss path with `backward=False`, so it currently
  skips the 10% repair and chooses the branch independently per positive
  supervision point/voxel. That per-voxel eval behavior should remain.
- `FiberTrace3DLoader.load_sample(...)` currently stores the bounded
  `data_sample_index` in `FiberTrace3DSample.sample_index`; with
  `max_sample_index`, that stored value repeats even though the raw
  augmentation sample index passed into `_sample_augment_params(...)` keeps
  increasing.
- The current training repair groups scores by `(patch, floor(z/4), floor(y/4),
  floor(x/4))`, averages detached branch scores per group, enforces the 10%
  minority floor over groups, and broadcasts the selected branch back to points.

## Implementation

### Branch Selection API

- Replace the boolean `enforce_branch_min_fraction` plumbing with an explicit
  branch-selection mode, for example:
  - `train_offset_grid_min_fraction`
  - `eval_voxel`
- Keep direct `compute_losses(...)` compatibility by defaulting to the eval
  behavior unless training explicitly requests the training mode.
- `_forward_loss(..., backward=True)` should request
  `train_offset_grid_min_fraction`.
- `_forward_loss(..., backward=False)` and `evaluate_dense_loss(...)` should
  request `eval_voxel`.

### Stream/Data Index Rename

- Rename the ambiguous public/internal fields and local variables in the 3D
  loader/trainer:
  - `stream_index`: unbounded deterministic stream position;
  - `stream_indices`: batch tensor of unbounded stream positions;
  - `data_index`: bounded dataset-selection index after applying
    `sample_index_limit` / `training.max_sample_index`;
  - `data_indices`: batch tensor of bounded data-selection positions.
- Avoid new uses of bare `sample_index` where the distinction matters. Keep it
  only at compatibility CLI/API boundaries if renaming the public argument would
  be too broad for this task; immediately normalize it to `stream_index`.
- Update `FiberTrace3DSample` and `FiberTrace3DBatch` to carry both
  `stream_index/stream_indices` and `data_index/data_indices`.
- Existing debug/log output may print `data_index` for identifying which CP was
  read, but must print/use `stream_index` when describing deterministic random
  state.
- Replace current legacy bounded-index batch-field usage with the correct new
  `stream_indices` or `data_indices` field at each call site.

### Training Grid

- Change the grouped repair grid size from `4` to `2` selected/output voxels in
  each spatial axis. In 3D this means `2x2x2` groups.
- Add a deterministic per-sample integer grid offset:
  - offset range per spatial axis: `0..chunk_size-1`;
  - offset is generated from `stream_indices`, plus fixed per-axis salts;
  - the offset is looked up by local batch id from `indices_bzyx[:, 0]`;
  - grouping uses the shifted coordinates, e.g.
    `floor((coord + offset_for_sample_axis) / chunk_size)`.
- Keep grouping keyed by the local patch id, so groups never cross samples.
- Keep the current 10% minimum branch quota over grouped positive supervision.
- Keep the existing “only repair the underrepresented side” behavior.
- Keep repair training-only; no grouped repair should run when evaluating.

### Test/Eval Routing

- Keep test/eval branch selection per positive supervision point/voxel:
  `argmax(score)` independently for each sparse positive voxel.
- Test/eval must not use grouped branch choice, random offsets, or the 10%
  quota repair.
- Branch-fraction scalars remain computed from those per-voxel choices.

### Determinism

- The offset generator must be pure/tensor-based and deterministic from the raw
  `stream_index`; it must not use mutable Python, NumPy, or torch RNG state
  inside the loss.
- Because `stream_index` is unbounded, reused bounded `data_index` samples will
  still receive different deterministic offsets on later repeats.
- `data_index` must never seed augmentation, branch-grid offsets, jitter/noise,
  or other random sources.

## Spec Update

- Update `planning/specs.md`:
  - replace the current `4x4x4` grouped repair description with `2x2x2`;
  - state that training groups are shifted by a deterministic per-sample random
    offset derived from `stream_index`;
  - state that test/eval uses raw per-voxel branch argmax and no grouped or 10%
    repair branch choice.
  - add canonical naming rules for `stream_index` and `data_index` and require
    `stream_index` for all random sources/augmentations while limiting
    `data_index` to dataset lookup/debug.

## Docs Updates

- Update `docs/code_structure.md` in the 3D training/loss section to describe:
  - `stream_index` vs `data_index` naming and allowed use;
  - branch-score routing modes;
  - training-only offset `2x2x2` grouped min-fraction repair;
  - test/eval per-voxel branch selection without grouped repair.
- Add a changelog entry for the branch routing update.

## Tests

- Update the existing two-branch positive-supervision tests for `2x2x2`
  grouping.
- Add a test that two samples with different `stream_indices` can produce
  different shifted grouping even for identical local positive coordinates.
- Add a loader/batch regression test that a limited `max_sample_index` repeat
  keeps changing `stream_indices` while `data_indices` wraps/repeats.
- Add a test that augmentation and branch-grid offset helpers consume
  `stream_indices`, not `data_indices`.
- Add a test that training repair uses grouped selection with the shifted
  `2x2x2` grid and still enforces the 10% minority floor.
- Add a regression test that eval/test keeps independent per-voxel argmax
  branch choice.
- Add a regression test that eval/test does not run grouped repair, random
  offsets, or the 10% floor.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'branch or compute_losses'`
- Run the full 3D test file:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `python -m py_compile` on `fiber_trace_3d/train.py`.
- Run `git diff --check`.

## Changelog

- Add a 2026-07-26 changelog bullet noting that 3D two-branch training now uses
  deterministic `stream_index`-seeded per-sample-offset `2x2x2` choice groups,
  while eval/test keeps per-voxel branch selection with no grouped repair.
- Include the `stream_index` / `data_index` rename in the changelog bullet.

## Review Notes / Assumptions

- I interpret “2x2” in this 3D loss context as `2x2x2`, because the current
  code applies the previous scalar `4` to all three spatial axes.
- Corrected interpretation: test/eval should keep branch choice per
  voxel/positive point and only avoid grouping/10% repair.
- `stream_index` and `data_index` are the planned names. `stream_index` is for
  all operations/random sources/augmentations; `data_index` is only for data
  read/CP selection and debug.
- No implementation should change negative presence supervision, per-branch
  negative normalization, or train/test TensorBoard branch-presence columns.
