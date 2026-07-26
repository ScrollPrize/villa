# 3D Branch Choice Grid Routing Update

Change the two-branch 3D positive supervision routing:

- Training should switch the anti-collapse grouped branch-choice grid from the
  current 4-voxel grid back to a 2-voxel grid.
- Each sample/patch in a training batch should get a deterministic random grid
  offset before grouping, so the same physical artifacts are not always aligned
  to fixed 2-voxel cell boundaries.
- Test/eval should keep branch choice per voxel/positive point and must not use
  grouped choice repair or the 10% repair.
- Rename the ambiguous 3D index fields/API wording:
  - `stream_index` / `stream_indices`: unbounded deterministic stream position;
  - `data_index` / `data_indices`: bounded dataset-selection index after any
    `max_sample_index` wrapping.
- Use `stream_index` for every random source, augmentation seed, branch-grid
  offset, and deterministic stream operation.
- Use `data_index` only for data reading/CP selection and debug reporting.

Do not change the model output layout, negative presence loss, branch presence
visualization, or the 10% minimum-branch training quota except for the grid size
and random offset behavior.
