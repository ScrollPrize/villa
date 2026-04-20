# Autoregressive Mesh MVP

This package implements a first MVP for tifxyz quad-lattice continuation from:

- a `1 x D x H x W` crop
- a partial wrap prompt near the split frontier
- frozen Dinovol volume features

## Representation

- Tifxyz is treated as a regular 2D lattice of 3D vertices.
- Training examples come from split conditioning on one wrap per sample.
- The prompt is a narrow frontier band from the conditioning side.
- By default the package serializes the full upsampled tifxyz lattice.
- For very large stored-scale surfaces, set `use_stored_resolution_only=true` to split and serialize directly on the stored tifxyz lattice instead of materializing the full upsampled grid first.
- The continuation is serialized in deterministic frontier order:
  - `left` / `right`: emit columns from the frontier outward
  - `up` / `down`: emit rows from the frontier outward
- Each emitted vertex predicts:
  - a coarse Dinovol patch-cell id
  - per-axis local offset bins inside that cell
  - a stop logit
- The coarse head is configurable:
  - `joint_pointer`: one flattened pointer over the full coarse grid
  - `axis_factorized`: separate `z/y/x` coarse predictions pooled from the same volume memory

## Model I/O

- Dataset output includes:
  - `volume`
  - `prompt_tokens`
  - `prompt_meta`
  - `target_coarse_ids`
  - `target_offset_bins`
  - `target_stop`
  - `target_xyz`
  - `direction`
  - `strip_length`
  - `min_corner`
  - `world_bbox`
  - `wrap_metadata`
- Model output includes:
  - `coarse_logits` for `joint_pointer`, or `coarse_axis_logits` for `axis_factorized`
  - `offset_logits`
  - `stop_logits`
  - `pred_xyz`

Checkpoint resume across coarse-head modes is intentionally unsupported. A checkpoint trained with `joint_pointer` must resume with `joint_pointer`, and likewise for `axis_factorized`.

## Smoke Train

Run the local Click entrypoint with a JSON config:

```bash
uv run --extra models --extra tests python -m vesuvius.neural_tracing.autoreg_mesh.train path/to/autoreg_mesh_config.json
```

The config must include a local `dinov2_backbone` checkpoint path plus the usual dataset paths for `EdtSegDataset`.

For datasets whose tifxyz `scale` is much smaller than `1.0` per lattice axis, the practical starting point is:

```json
{
  "use_stored_resolution_only": true,
  "surface_downsample_factor": 1
}
```

## Inference Example

```python
from vesuvius.neural_tracing.autoreg_mesh.infer import infer_autoreg_mesh

result = infer_autoreg_mesh(
    model,
    sample_or_batch,
    max_steps=None,
    stop_probability_threshold=0.5,
    greedy=True,
    save_path=None,
)
```

The result contains sampled continuation vertices plus reconstructed continuation and merged full `zyx` grids in local and world coordinates.

W&B logging can also emit XY-slice overlays that draw prompt, GT continuation, and predicted continuation lines directly over a selected volume slice for quick geometry/context inspection.
