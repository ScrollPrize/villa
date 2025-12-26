import torch
import numpy as np
from typing import Tuple, Set, List

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class Rot90Transform(BasicTransform):
    """
    Applies random 90-degree rotations to image and associated targets.

    Randomly selects axis pairs and applies torch.rot90 with random multiples
    of 90 degrees. Faster than arbitrary rotations and preserves exact values.

    Parameters:
        num_axis_combinations (RandomScalar): Number of axis combinations to rotate.
        num_rot_per_combination (Tuple[int, ...]): Possible multiples of 90 degrees
                                                    (e.g., (1, 2, 3) for 90, 180, 270).
        allowed_axes (Set[int]): Spatial axes to randomly select rotation axes from
                                 (e.g., {0, 1, 2} for 3D).
    """

    def __init__(
        self,
        num_axis_combinations: RandomScalar = 1,
        num_rot_per_combination: Tuple[int, ...] = (1, 2, 3),
        allowed_axes: Set[int] = {0, 1, 2},
    ):
        super().__init__()
        self.num_axis_combinations = num_axis_combinations
        self.num_rot_per_combination = num_rot_per_combination
        self.allowed_axes = allowed_axes

    def get_parameters(self, **data_dict) -> dict:
        n_axes_combinations = round(sample_scalar(self.num_axis_combinations))
        axis_combinations = []
        num_rot_per_combination = []

        for _ in range(n_axes_combinations):
            num_rot_per_combination.append(int(np.random.choice(self.num_rot_per_combination)))
            # Select 2 axes for rotation plane
            axes = sorted(np.random.choice(list(self.allowed_axes), size=2, replace=False))
            # +1 because we skip channel dimension
            axis_combinations.append([a + 1 for a in axes])

        return {
            'num_rot_per_combination': num_rot_per_combination,
            'axis_combinations': axis_combinations
        }

    def _maybe_rot90(
        self,
        tensor: torch.Tensor,
        num_rot_per_combination: List[int],
        axis_combinations: List[List[int]],
    ) -> torch.Tensor:
        """Apply the rot90 operations."""
        for n_rot, axes in zip(num_rot_per_combination, axis_combinations):
            tensor = torch.rot90(tensor, k=n_rot, dims=axes)
        return tensor

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self._maybe_rot90(
            img,
            params['num_rot_per_combination'],
            params['axis_combinations']
        )

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        return self._maybe_rot90(
            seg,
            params['num_rot_per_combination'],
            params['axis_combinations']
        )

    def _apply_to_regr_target(self, regression_target: torch.Tensor, **params) -> torch.Tensor:
        return self._maybe_rot90(
            regression_target,
            params['num_rot_per_combination'],
            params['axis_combinations']
        )

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        return self._maybe_rot90(
            dist_map,
            params['num_rot_per_combination'],
            params['axis_combinations']
        )

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Rot90Transform does not support bounding boxes")

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError("Rot90Transform does not support keypoints")
