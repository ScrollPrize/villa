from contextlib import contextmanager

import numpy as np

from train_resnet3d_lib.config import CFG as DATA_CFG
from train_resnet3d_lib.data_ops import read_image_fragment_mask, read_image_mask


@contextmanager
def _temporary_data_cfg(*, dataset_root, in_chans, layer_read_workers=None):
    original_dataset_root = getattr(DATA_CFG, "dataset_root", "train_scrolls")
    original_in_chans = getattr(DATA_CFG, "in_chans", 62)
    original_layer_read_workers = getattr(DATA_CFG, "layer_read_workers", 1)

    DATA_CFG.dataset_root = str(dataset_root)
    DATA_CFG.in_chans = int(in_chans)
    if layer_read_workers is not None:
        DATA_CFG.layer_read_workers = int(layer_read_workers)

    try:
        yield
    finally:
        DATA_CFG.dataset_root = original_dataset_root
        DATA_CFG.in_chans = original_in_chans
        DATA_CFG.layer_read_workers = original_layer_read_workers


def load_training_segment(
    *,
    segment_id,
    dataset_root,
    layer_range,
    reverse_layers,
    in_chans,
    layer_read_workers=None,
):
    with _temporary_data_cfg(
        dataset_root=dataset_root,
        in_chans=in_chans,
        layer_read_workers=layer_read_workers,
    ):
        image, mask, fragment_mask = read_image_mask(
            segment_id,
            layer_range=layer_range,
            reverse_layers=reverse_layers,
        )

    mask = mask.astype(np.float32, copy=False)
    if mask.max(initial=0.0) > 1.0:
        mask /= 255.0

    return image, mask, fragment_mask


def load_inference_segment(
    *,
    segment_id,
    dataset_root,
    layer_range,
    reverse_layers,
    in_chans,
    layer_read_workers=None,
):
    with _temporary_data_cfg(
        dataset_root=dataset_root,
        in_chans=in_chans,
        layer_read_workers=layer_read_workers,
    ):
        image, fragment_mask = read_image_fragment_mask(
            segment_id,
            layer_range=layer_range,
            reverse_layers=reverse_layers,
        )

    return image, fragment_mask
