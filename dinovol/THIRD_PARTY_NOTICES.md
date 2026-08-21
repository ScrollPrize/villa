# Third-Party Notices

Dinovol includes code adapted from the projects listed below. Original
third-party code remains subject to its upstream license; modifications made
for Dinovol are licensed under the repository's MIT License where permitted.

## DINOv2

The following files are derived from
[DINOv2](https://github.com/facebookresearch/dinov2), copyright Meta Platforms,
Inc. and affiliates, and are provided under the
[Apache License 2.0](LICENSES/Apache-2.0.txt):

- `dinovol_2/loss/dino_clstoken_loss.py`
- `dinovol_2/loss/ibot_patch_loss.py`
- `dinovol_2/loss/koleo_loss.py`
- `dinovol_2/model/model.py`
- `dinovol_2/ops/collate.py`
- `dinovol_2/ops/masking.py`
- `dinovol_2/pretrain.py`

These files have been modified for Dinovol's three-dimensional training
pipeline.

## DINOv3

The following files are derived from
[DINOv3](https://github.com/facebookresearch/dinov3), copyright Meta Platforms,
Inc. and affiliates, and are distributed under the
[DINOv3 License Agreement](LICENSES/DINOv3-LICENSE.md):

- `dinovol_2/loss/gram_loss.py`
- `dinovol_2/model/rope.py`

These files have been modified for Dinovol, including three-dimensional
position encoding and training integration.

## Dynamic Network Architectures

`dinovol_2/model/dinov2_eva.py` and
`dinovol_2/model/patch_encode_decode.py` are adapted from
[dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures),
with modifications for Dinovol. Copyright 2022 Division of Medical Image
Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany. The
upstream project is licensed under the
[Apache License 2.0](LICENSES/Apache-2.0.txt).

## batchgeneratorsv2

`dinovol_2/augmentation/` is a modified adaptation of
[batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2). Copyright
2019 Division of Medical Image Computing, German Cancer Research Center
(DKFZ), Heidelberg, Germany. The upstream project is licensed under the
[Apache License 2.0](LICENSES/Apache-2.0.txt).

## nnU-Net

`dinovol_2/dataset/normalization.py` is adapted from
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Copyright 2019 Division of
Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg,
Germany. The upstream project is licensed under the
[Apache License 2.0](LICENSES/Apache-2.0.txt).
