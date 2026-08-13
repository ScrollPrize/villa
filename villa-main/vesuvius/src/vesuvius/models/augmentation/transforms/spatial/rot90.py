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

    _is_spatial = True  # Skip per-transform padding restoration

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

        params = {
            'num_rot_per_combination': num_rot_per_combination,
            'axis_combinations': axis_combinations
        }
        crop_shape = data_dict.get('crop_shape')
        if crop_shape is None:
            image = data_dict.get('image')
            if image is not None:
                crop_shape = tuple(int(s) for s in image.shape[1:])
        if crop_shape is not None:
            params['crop_shape'] = tuple(int(s) for s in crop_shape)
        return params

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
        """
        Transform keypoint coordinates for rot90.

        keypoints: (N, 3) tensor in (z, y, x) order
        Requires 'crop_shape' in params (tuple of spatial dimensions).
        """
        if keypoints is None:
            return None
        crop_shape = params.get('crop_shape')
        if crop_shape is None:
            raise ValueError("Rot90Transform._apply_to_keypoints requires 'crop_shape' in params")

        keypoints = keypoints.clone()
        for n_rot, axes in zip(params['num_rot_per_combination'], params['axis_combinations']):
            # axes are 1-indexed (skip channel dim), convert to 0-indexed for keypoints
            a, b = axes[0] - 1, axes[1] - 1
            k = n_rot % 4
            if k == 1:
                # torch.rot90(..., k=1, dims=(a,b)):
                # new_a = shape[b] - 1 - old_b, new_b = old_a
                new_a = crop_shape[b] - 1 - keypoints[:, b]
                new_b = keypoints[:, a].clone()
                keypoints[:, a] = new_a
                keypoints[:, b] = new_b
            elif k == 2:
                # 180° rotation: both axes flip
                keypoints[:, a] = crop_shape[a] - 1 - keypoints[:, a]
                keypoints[:, b] = crop_shape[b] - 1 - keypoints[:, b]
            elif k == 3:
                # torch.rot90(..., k=3, dims=(a,b)):
                # new_a = old_b, new_b = shape[a] - 1 - old_a
                new_a = keypoints[:, b].clone()
                new_b = crop_shape[a] - 1 - keypoints[:, a]
                keypoints[:, a] = new_a
                keypoints[:, b] = new_b
        return keypoints

    def _apply_to_vectors(self, vectors, **params):
        """
        Transform vector components for rot90 (rotation only, no translation).

        vectors: (N, 3) tensor in (z, y, x) order
        """
        if vectors is None:
            return None

        vectors = vectors.clone()
        for n_rot, axes in zip(params['num_rot_per_combination'], params['axis_combinations']):
            # axes are 1-indexed (skip channel dim), convert to 0-indexed
            a, b = axes[0] - 1, axes[1] - 1
            k = n_rot % 4
            if k == 1:
                # Matches torch.rot90 k=1 coordinate mapping
                # (va, vb) -> (-vb, va)
                new_a = -vectors[:, b]
                new_b = vectors[:, a].clone()
                vectors[:, a] = new_a
                vectors[:, b] = new_b
            elif k == 2:
                # 180° rotation: (va, vb) -> (-va, -vb)
                vectors[:, a] = -vectors[:, a]
                vectors[:, b] = -vectors[:, b]
            elif k == 3:
                # Matches torch.rot90 k=3 coordinate mapping
                # (va, vb) -> (vb, -va)
                new_a = vectors[:, b].clone()
                new_b = -vectors[:, a]
                vectors[:, a] = new_a
                vectors[:, b] = new_b
        return vectors

    def apply(self, data_dict: dict, **params) -> dict:
        """Override to handle keypoints and vector_keys."""
        data_dict = super().apply(data_dict, **params)

        # Handle vector_keys
        vector_keys = set(data_dict.get('vector_keys', []) or [])
        for key in vector_keys:
            if data_dict.get(key) is not None:
                data_dict[key] = self._apply_to_vectors(data_dict[key], **params)

        # Transform padding_mask with the same rotation
        if data_dict.get('padding_mask') is not None:
            data_dict['padding_mask'] = self._apply_to_image(data_dict['padding_mask'], **params)

        return data_dict
