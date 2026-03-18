an attempt at a faithful implementation of dinov2-style pretraining on 3d volumes. 

- the dinov2_eva is from [dynamic-network-architectures](github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/dinov2_eva.py) , with some minimal changes
- the augmentation library is a loosely modified [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2)
- normalization is mostly borrowed from [nnunetv2](https://github.com/MIC-DKFZ/nnUNet)
- rope is from the [dinov3 impl](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py), extended to support 3d

this implementation is still incomplete. pretraining works but no finetuning yet written. 

## Napari visualizer

There is a small napari helper for checkpoint inspection at `dinovol_2/eval/napari_visualizer.py`.

Run it with:

```bash
python -m dinovol_2.eval.napari_cosine_similarity
```

Workflow:

- load a volume into napari
- add one or more points in a `Points` layer
- choose a `pretrain.py` checkpoint, image layer, and points layer in the dock widget
- click `Cache Embeddings`
- click `Show Feature PCA` to render a 3-channel PCA view of the cached patch embeddings
- optionally enable `Otsu Foreground Mask` and set `Mask Dilation` before creating the PCA layer
- click `Similarity For Selected Points` or `Similarity For All Points`

The widget rebuilds the teacher backbone from the saved checkpoint config, computes a patch embedding grid for the selected volume, and creates one cosine-similarity image layer per reference point across the full volume.
