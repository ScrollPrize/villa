"""
Alternate implementations of HeatmapDatasetV2._get_quads_in_crop for benchmarking.

Profile reference:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     15264    6.867    0.000   11.626    0.001 dataset.py:221(_get_quads_in_crop)

Goals: add cheap early exits and, optionally, memoize repeated queries while
keeping outputs identical to the base implementation.
"""

from typing import Dict, Tuple

import torch

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2


class BBoxFilteredQuads(HeatmapDatasetV2):
    """
    Precompute per-patch quad-center AABBs; return an all-false mask when the crop
    cannot intersect, avoiding full tensor comparisons.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        quad_bboxes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for patch in self._patches:
            valid_centers = patch.quad_centers[patch.valid_quad_mask]
            if len(valid_centers) == 0:
                bbox_min = torch.zeros(3, dtype=patch.quad_centers.dtype)
                bbox_max = bbox_min
            else:
                bbox_min = valid_centers.min(dim=0).values
                bbox_max = valid_centers.max(dim=0).values
            quad_bboxes[id(patch)] = (bbox_min, bbox_max)
        self._quad_bboxes = quad_bboxes

    def _get_quads_in_crop(self, patch, min_corner_zyx, crop_size):
        bbox_min, bbox_max = self._quad_bboxes[id(patch)]
        crop_min = min_corner_zyx.to(dtype=bbox_min.dtype)
        crop_size_tensor = torch.as_tensor(crop_size, dtype=bbox_min.dtype, device=crop_min.device)
        crop_max = crop_min + crop_size_tensor
        if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
            return torch.zeros_like(patch.valid_quad_mask)
        return super()._get_quads_in_crop(patch, min_corner_zyx, crop_size)


class CachedBBoxFilteredQuads(BBoxFilteredQuads):
    """
    Add last-call memoization on top of AABB rejection for repeated crop queries
    in the same context.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        self._quads_cache_key = None
        self._quads_cache_value = None

    @staticmethod
    def _make_cache_key(patch, min_corner_zyx, crop_size):
        return (
            id(patch),
            tuple(min_corner_zyx.tolist()),
            int(crop_size),
        )

    def _get_quads_in_crop(self, patch, min_corner_zyx, crop_size):
        key = self._make_cache_key(patch, min_corner_zyx, crop_size)
        if key == self._quads_cache_key:
            return self._quads_cache_value
        mask = super()._get_quads_in_crop(patch, min_corner_zyx, crop_size)
        self._quads_cache_key = key
        self._quads_cache_value = mask
        return mask
