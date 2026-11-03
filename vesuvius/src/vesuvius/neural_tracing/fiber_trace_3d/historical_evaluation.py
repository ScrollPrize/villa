"""Historical CP test definition, separate from modern training supervision."""

from copy import deepcopy
import math

import torch


def historical_test_config(raw):
    datasets = raw.get("historical_test_datasets")
    if datasets is None:
        return None
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("historical_test_datasets must be a nonempty dataset list")
    result = deepcopy(raw)
    result["datasets"] = deepcopy(datasets)
    for dataset in result["datasets"]:
        if dataset.get("dense_supervision", False):
            raise ValueError("historical tests require CP-only JSON supervision")
        dataset["dense_supervision"] = False
    result["sampling_strategy"] = "control_points"
    result["batch_size"] = 8
    result["augment_enabled"] = False
    # Historical observation uses direction/presence only. Do not let unused
    # embedding eligibility change its reference sample population.
    result["embedding"] = {"enabled": False}
    result.pop("test_datasets", None)
    result.pop("historical_test_datasets", None)
    return result


def historical_presence_mask(batch, config):
    """Old CP-only presence region; use source validity when supplied.

    Modern native loading does not transport a source-availability mask. Thus
    this reproduces the historical loss region, not old backend hole handling.
    The distinction is explicitly recorded in the metric documentation.
    """
    shape = tuple(batch.presence_target.shape[-3:])
    if shape != tuple(batch.volume.shape[-3:]):
        raise ValueError("historical presence loss requires full-resolution targets")
    margin = config.presence_negative_edge_margin_voxels
    if margin is None:
        margin = math.ceil(max(config.augment_shift_zyx))
    axes = [torch.arange(size, device=batch.volume.device) for size in shape]
    interior = ((axes[0][:, None, None] >= margin) & (axes[0][:, None, None] < shape[0] - margin)
                & (axes[1][None, :, None] >= margin) & (axes[1][None, :, None] < shape[1] - margin)
                & (axes[2][None, None, :] >= margin) & (axes[2][None, None, :] < shape[2] - margin))
    mask = interior[None, None].expand(batch.volume.shape[0], 1, *shape)
    return mask if batch.valid_mask is None else mask & batch.valid_mask.bool()
