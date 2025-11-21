"""
Alternate implementations of HeatmapDatasetV2._get_cached_patch_points for benchmarking.

Profile reference:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       399    0.049    0.000   19.402    0.049 dataset.py:255(_get_cached_patch_points)

These subclasses aim to keep outputs identical while cutting per-call overhead.
"""

from typing import Dict, List, Tuple

import torch

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2


class LastCallMemoizedPatchPoints(HeatmapDatasetV2):
    """
    Cache the most recent _get_cached_patch_points call keyed by (patch, center, crop).
    Useful when multiple perturbations share the same context within one sample.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        self._cache_key = None
        self._cache_value: List[torch.Tensor] = []

    @staticmethod
    def _make_cache_key(current_patch, center_ij, min_corner_zyx, crop_size) -> Tuple:
        return (
            id(current_patch),
            tuple(center_ij.tolist()),
            tuple(min_corner_zyx.tolist()),
            int(crop_size),
        )

    def _get_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        key = self._make_cache_key(current_patch, center_ij, min_corner_zyx, crop_size)
        if key != self._cache_key:
            self._cache_key = key
            self._cache_value = super()._get_cached_patch_points(current_patch, center_ij, min_corner_zyx, crop_size)
        return self._cache_value


class VolumeFilteredPatchPoints(HeatmapDatasetV2):
    """
    Skip scanning patches from unrelated volumes by grouping once in __init__.
    The filtered order matches the base implementation's iteration order.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        patches_by_volume: Dict[int, List] = {}
        for patch in self._patches:
            patches_by_volume.setdefault(id(patch.volume), []).append(patch)
        self._patches_by_volume = patches_by_volume

    def _iter_volume_siblings(self, current_patch):
        return self._patches_by_volume.get(id(current_patch.volume), [])

    def _get_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        quad_main_component = self._get_current_patch_center_component_mask(current_patch, center_ij, min_corner_zyx, crop_size)
        all_patch_points: List[torch.Tensor] = []

        for other_patch in self._iter_volume_siblings(current_patch):
            if other_patch is current_patch:
                continue
            patch_points = self._get_patch_points_in_crop(other_patch, min_corner_zyx, crop_size)
            if len(patch_points) > 0:
                all_patch_points.append(patch_points)

        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        quad_excluding_main = quad_in_crop & ~quad_main_component

        other_patch_points = self._sample_points_from_quads(current_patch, quad_excluding_main)
        if len(other_patch_points) > 0:
            all_patch_points.append(other_patch_points)

        return all_patch_points


class VolumeBBoxFilteredPatchPoints(VolumeFilteredPatchPoints):
    """
    Add an AABB rejection test before sampling patch points, avoiding work when
    the crop and patch quads are disjoint.
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

    def _get_cached_patch_points(self, current_patch, center_ij, min_corner_zyx, crop_size):
        quad_main_component = self._get_current_patch_center_component_mask(current_patch, center_ij, min_corner_zyx, crop_size)
        all_patch_points: List[torch.Tensor] = []

        bbox_min_dtype = self._quad_bboxes[id(current_patch)][0].dtype
        crop_min = min_corner_zyx.to(dtype=bbox_min_dtype)
        crop_max = crop_min + crop_size

        for other_patch in self._iter_volume_siblings(current_patch):
            if other_patch is current_patch:
                continue

            bbox_min, bbox_max = self._quad_bboxes[id(other_patch)]
            if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
                continue

            patch_points = self._get_patch_points_in_crop(other_patch, min_corner_zyx, crop_size)
            if len(patch_points) > 0:
                all_patch_points.append(patch_points)

        quad_in_crop = self._get_quads_in_crop(current_patch, min_corner_zyx, crop_size)
        quad_excluding_main = quad_in_crop & ~quad_main_component

        other_patch_points = self._sample_points_from_quads(current_patch, quad_excluding_main)
        if len(other_patch_points) > 0:
            all_patch_points.append(other_patch_points)

        return all_patch_points
