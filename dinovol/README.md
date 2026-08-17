# Dinovol

Dinovol is an implementation of DINO-style self-supervised pretraining on 3D
volumes.

- `dinov2_eva` is adapted from [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/dinov2_eva.py), with minimal changes.
- The augmentation library is a modified [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2).
- Normalization is mostly borrowed from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
- RoPE is adapted from the [DINOv3 implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py) and extended to support 3D.

This implementation is still incomplete. Pretraining works, but finetuning has
not yet been implemented.

NOTE: a newer `v2` backbone config exists and should generally be preferred for new runs, but the default remains the older config so older checkpoints continue to load without config changes
To select the newer defaults explicitly, set `model.model_type` to `v2` in the config:

```json
{
  "model": {
    "model_type": "v2",
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

When `gram.enabled=true`, the trainer builds a frozen Gram teacher backbone, loads it from `gram.teacher_checkpoint` when provided, refreshes it from the live EMA teacher on the configured cadence, and adds an image-level Gram loss on patch features. Gram-teacher crops are paired with each global crop from the exact same sampled 3D region, and default to normalization-only.

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
uv run python -m unittest tests.test_pretrain_smoke -v
```

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
