# Dinovol

Dinovol is an implementation of DINO-style self-supervised pretraining on 3D
volumes.

The canonical source is
[`ScrollPrize/villa/dinovol`](https://github.com/ScrollPrize/villa/tree/main/dinovol).
Run project commands from the monorepository's `dinovol/` directory.

- `dinov2_eva` is adapted from [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/dinov2_eva.py), with minimal changes.
- The augmentation library is a modified [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2).
- Normalization is mostly borrowed from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
- RoPE is adapted from the [DINOv3 implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py) and extended to support 3D.

This implementation is still incomplete. Pretraining works, but finetuning has
not yet been implemented.

Use the `v3` preset for new runs, as selected by every example recipe. It uses
axial 3D RoPE, separate local/global CLS output normalization, constant LR after
warmup, constant weight decay and teacher momentum, and KoLeo groups of up to 16.
The `v2` preset and configs without `model_type` retain their older defaults for
checkpoint compatibility. Explicit config values override preset defaults.
To select the new defaults, set `model.model_type` to `v3`:

```json
{
  "model": {
    "model_type": "v3",
    "embedding_type": "default",
    "global_crops_size": [96, 96, 96],
    "local_crops_size": [48, 48, 48]
  }
}
```

## License

Original Dinovol code is available under the [MIT License](LICENSE).
Third-party code retains its upstream terms; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Backbone and learning-rate options

The following settings can be added to a training config:

```json
{
  "lr_schedule": "constant",
  "warmup_steps": 1000,
  "koleo_distributed": true,
  "koleo_loss_group_size": 16,
  "model": {
    "model_type": "v3",
    "rope_type": "axial",
    "untie_cls_and_patch_norms": false,
    "untie_global_and_local_cls_norm": true
  }
}
```

- `lr_schedule` defaults to `constant` for `v3`, and `cosine` for older presets. `constant` uses the existing linear
  warmup and then holds the configured `lr`, including beyond `max_iterations`;
  it ignores `min_lr`. Last-layer freezing and parameter-group LR multipliers
  still apply. Weight decay, teacher momentum, and temperature retain their
  own schedules. In `v3`, weight decay defaults to a constant `0.04` and teacher
  momentum to a constant `0.994`; overriding their initial values also changes
  their default final values. Set `weight_decay_end` equal to `weight_decay` and
  `final_momentum_teacher` equal to `momentum_teacher` to hold those constant too.
  `warmup_steps: 0` disables LR warmup; otherwise the default is 10% of the run.
- `rope_type` selects the existing 3D `axial` or learnable per-block `mixed`
  implementation. The default is `axial` for `v3`, `mixed` for `v2`, and `axial`
  for the original unversioned model. Both support variable crop sizes and coordinate rescaling.
  The attention head dimension must be divisible by six for 3D RoPE.
- `untie_cls_and_patch_norms` gives CLS/register tokens their own output norm;
  patch tokens keep the original `norm`.
- `untie_global_and_local_cls_norm` adds a separate CLS/register output norm for
  local crops during student training. Evaluation and the EMA teacher use the
  global norm. This flag defaults to `true` for `v3` and `false` for older presets;
  `untie_cls_and_patch_norms` remains `false` in all presets.
- `koleo_distributed` defaults to `true`: with DDP, KoLeo compares each view's
  embeddings across ranks, with gradients through gathered embeddings. This
  expects equal per-rank batch sizes, as supplied by the training loaders.
  Set it to `false` for local-only comparisons. A comparison batch with fewer
  than two samples produces zero loss and gradient.
- `koleo_loss_group_size` defaults to `16` for `v3`, and `null` (the whole
  comparison batch) for older presets. Groups are contiguous in rank-major order,
  independently for each global view. If fewer than 16 samples are available,
  all available samples are used. Larger comparison batches must be divisible
  by the requested group size; choose a divisor or `null` otherwise. Grouping
  limits neighbour selection; the current implementation still gathers all ranks.

Model head dimensions and `pretrained_weights` are preserved in the resolved
training config. When enabling separate norms on an older checkpoint, missing
norm weights are initialized from its trained shared norm (or its existing CLS
norm for the new local norm). Use `model.pretrained_weights` to start a new run
when changing normalization options; resuming optimizer state requires the same
parameter layout. Keep `rope_type` consistent with the checkpoint: switching
between learned mixed and axial RoPE is an architecture experiment, not an exact
checkpoint conversion.

## Gram Anchoring And HR Adaptation

`pretrain.py` now supports the three-stage DINOv3-style workflow:

- base pretraining with DINO + iBOT + KoLeo
- late dense-feature refinement with Gram anchoring
- short mixed-resolution HR adaptation with Gram anchoring kept on

The new config surface is:

- top-level `gram`
  - `enabled`
  - `loss_weight`
  - `teacher_checkpoint`
  - `teacher_refresh_every`
  - `teacher_refresh_start_step`
  - `normalized`
  - `img_level`
  - `remove_neg`
  - `remove_only_teacher_neg`
- dataset keys
  - `gram_teacher_crop_size`
  - `gram_teacher_no_augmentations`
  - `variants`

When `gram.enabled=true`, the trainer builds a frozen Gram teacher backbone, loads it from `gram.teacher_checkpoint` when provided, refreshes it from the live EMA teacher on the configured cadence, and adds an image-level Gram loss on patch features. Gram-teacher crops are paired with each global crop from the exact same sampled 3D region. With `gram_teacher_no_augmentations=true`, they receive normalization and the same geometric transforms as the student, but no intensity/noise augmentation. Shared rotations, flips, and transpositions preserve patch correspondence even at different crop resolutions.

Mixed-resolution HR adaptation is enabled by adding `dataset.variants`, where each variant defines its own crop sizes and sampling ratio. The trainer builds one dataloader per variant and samples them with the configured weights. For `embedding_type=deeper`, the dataset automatically derives the needed overscanned `*_view_size` values from the patch halo.

`dinovol_2/example_config.json` remains runnable as a base-pretraining config and also includes a `recipes` object with three complete examples:

- `recipes.base_pretrain`
- `recipes.gram_refinement`
- `recipes.hr_adaptation`

The important stage-specific overrides are:

```json
{
  "gram": {
    "enabled": true,
    "loss_weight": 2.0,
    "teacher_checkpoint": "/path/to/previous/checkpoint.pt",
    "teacher_refresh_every": 10000
  },
  "model": {
    "pretrained_weights": "/path/to/previous/checkpoint.pt",
    "pretrained_backbone_only": false
  }
}
```

For HR adaptation, add weighted crop variants:

```json
{
  "dataset": {
    "variants": [
      {
        "ratio": 0.3,
        "global_crop_size": [128, 128, 128],
        "local_crop_size": [48, 48, 48],
        "gram_teacher_crop_size": [160, 160, 160]
      },
      {
        "ratio": 0.7,
        "global_crop_size": [160, 160, 160],
        "local_crop_size": [80, 80, 80],
        "gram_teacher_crop_size": [192, 192, 192]
      }
    ]
  }
}
```

To sanity-check one batch end to end:

```bash
uv run python -m dinovol_2.verify dinovol_2/example_config.json --no-amp
```

To run the synthetic smoke suite:

```bash
uv run python -m unittest discover -s tests -v
```

With an existing environment, run without installing or updating dependencies:

```bash
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The suite includes synthetic volume smoke tests and regressions for checkpoint
initialization, output norms, both RoPE implementations, constant LR, paired Gram
geometry, and two-rank KoLeo gradients. It does not establish downstream quality;
validate training changes on real held-out volumes before a long run.

## Point-supervised pretraining

Pretraining can optionally reinforce patch embeddings with sparse annotated points while DINO, iBOT, KoLeo, and Gram remain active. The top-level defaults are:

```json
{
  "point_supervision": {
    "enabled": true,
    "sampling_probability": 0.1,
    "loss_weight": 0.05,
    "different_type_margin": 0.0,
    "max_points_per_view": 64
  }
}
```

Attach annotations to each applicable volume (paths are intentionally omitted from the runnable example config):

```json
{
  "volume_path": "/path/to/volume.zarr",
  "volume_scale": 1,
  "point_collections": [
    {"path": "~/points/fibers.json", "type": "fiber"},
    {"path": "annotations/surface.json", "type": "surface"}
  ]
}
```

Each annotation file must be version 1 and contain a `collections` array whose entries contain `points` arrays with `p: [x, y, z]`. The configured `type` applies to every point in that file; equal type strings have equal meaning across volumes. `~` is expanded, while other relative paths remain relative to the process working directory.

Coordinates are scale-0 voxel centers in XYZ order. Dinovol converts them to ZYX and maps voxel centers to the selected pyramid level using the co-registered level-shape ratio. Point-centered sampling and the point loss are inactive when the feature is disabled or no volume has `point_collections`. Only training samples and the two global views are supervised; validation, monitor batches, and local views keep their existing sampling behavior.

## Optional Task Eval During Pretraining

`pretrain.py` can optionally run small downstream segmentation trainings during pretraining.

- set `task_eval_every` to a positive step cadence to enable it
- choose `eval_task` as `both`, `surfaces`, or `ink`
- set `eval_task_train_iters` to control the mini-training length, default `500`
- set `eval_task_decoder_type` to `simple` or `patch_encode_decode`

The task data is downloaded with `python -m dinovol_2.eval.download_data --task both`.

- `both` now means `surfaces` plus `ink`
- `surfaces` is resized 2x before crops are drawn
- `surfaces` and `ink` each use the first 10 sorted samples as the deterministic validation set
- `ink` is not resized before crops are drawn
- train and validation crops are taken from precomputed chunks that contain some foreground and at least 50% background in supervised voxels
- the saved validation image contains one row per validation sample, with image / label / prediction panels
- for `ink`, voxels with `supervision_mask == 0` are ignored and supervised unlabeled voxels are treated as background
- for `ink`, loss/metrics and saved previews use a max projection across Z to match the flat ink trainer

## Napari visualizer

There is a small napari helper for checkpoint inspection at `dinovol_2/eval/napari_visualizer.py`.

Run it with:

```bash
python -m dinovol_2.eval.napari_visualizer
```

Workflow:

- open an OME-Zarr from the widget, click `Load Scales`, choose the desired scale, and click `Open Zarr`
- draw a rectangle in the generated `*_bbox` shapes layer; this 2D YX bbox is applied across the full Z span of the selected scale
- add one or more points in a `Points` layer
- choose a `pretrain.py` checkpoint, image layer, and points layer in the dock widget
- click `Cache Embeddings`
- click `Show Feature PCA` to render a 3-channel PCA view of the cached patch embeddings
- optionally enable `Otsu Foreground Mask` and set `Mask Dilation` before creating the PCA layer
- click `Similarity For Selected Points` or `Similarity For All Points`

The widget rebuilds the teacher backbone from the saved checkpoint config, computes a patch embedding grid only inside the active bbox for the selected OME-Zarr scale, and limits the PCA and cosine-similarity outputs to that same crop. The dock widget opens on the bottom of the napari window.
