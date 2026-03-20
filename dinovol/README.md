an attempt at a faithful implementation of dinov2-style pretraining on 3d volumes. 

- the dinov2_eva is from [dynamic-network-architectures](github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/dinov2_eva.py) , with some minimal changes
- the augmentation library is a loosely modified [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2)
- normalization is mostly borrowed from [nnunetv2](https://github.com/MIC-DKFZ/nnUNet)
- rope is from the [dinov3 impl](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py), extended to support 3d

this implementation is still incomplete. pretraining works but no finetuning yet written. 

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
