# Autoregressive Mesh MVP

This package implements a first MVP for tifxyz quad-lattice continuation from:

- a `1 x D x H x W` crop
- a partial wrap prompt near the split frontier
- frozen Dinovol volume features

## Representation

- Tifxyz is treated as a regular 2D lattice of 3D vertices.
- Training examples come from split conditioning on one wrap per sample.
- The prompt is a narrow frontier band from the conditioning side.
- The continuation is serialized in deterministic frontier order:
  - `left` / `right`: emit columns from the frontier outward
  - `up` / `down`: emit rows from the frontier outward
- Each emitted vertex predicts:
  - a coarse Dinovol patch-cell id
  - per-axis local offset bins inside that cell
  - a stop logit

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
  - `coarse_logits`
  - `offset_logits`
  - `stop_logits`
  - `pred_xyz`

## Smoke Train

Run the local Click entrypoint with a JSON config:

```bash
uv run --extra models --extra tests python -m vesuvius.neural_tracing.autoreg_mesh.train path/to/autoreg_mesh_config.json
```

The config must include a local `dinov2_backbone` checkpoint path plus the usual dataset paths for `EdtSegDataset`.

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
