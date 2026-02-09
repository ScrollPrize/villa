from typing import Optional, Set
import numpy as np
import torch

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class TransposeAxesTransform(BasicTransform):
    """
    A transformation class to permute specified spatial axes of an image and related data.

    Attributes:
        allowed_axes (Set[int]): Set of spatial axes allowed for permutation (e.g., {1, 2} for y and x axes in an
        image of shape (C, Z, Y, X)). Spatial axes are indexed in (Z=0, Y=1, X=2) order.
    """

    _is_spatial = True  # Skip per-transform padding restoration

    def __init__(self, allowed_axes: Set[int], normal_keys: Optional[Set[str]] = None):
        """
        Initialize the transform with allowed spatial axes for permutation.

        Args:
            allowed_axes (Set[int]): Set of spatial axis indices for permutation.
            normal_keys (Set[str] | None): Optional keys containing normal vectors (shape (3, Z, Y, X))
                that should have their components permuted consistently with the spatial transpose.
        """
        super().__init__()
        self.allowed_axes = allowed_axes
        self.normal_keys = set(normal_keys or [])
        self._skip_when_vector = True

    def get_parameters(self, **data_dict) -> dict:
        """
        Generate a random axis permutation order.

        Args:
            data_dict (dict): Dictionary containing `image` tensor data.

        Returns:
            dict: Permutation order of axes as 'axis_order'.
        """
        shape_of_allowed = [data_dict['image'].shape[1 + i] for i in self.allowed_axes]
        if len(shape_of_allowed) < 2:
            return {'axis_order': list(range(len(data_dict['image'].shape)))}
        if not all(i == shape_of_allowed[0] for i in shape_of_allowed[1:]):
            raise ValueError(f"Axis shapes are not identical: {shape_of_allowed}. Cannot permute.\n"
                             f"Image shape: {data_dict['image'].shape}. Allowed axes: {self.allowed_axes}")

        axes = [i + 1 for i in self.allowed_axes]
        np.random.shuffle(axes)
        axis_order = np.arange(len(data_dict['image'].shape))
        axis_order[np.isin(axis_order, axes)] = axes
        return {'axis_order': [int(i) for i in axis_order]}

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation.permute(params['axis_order']).contiguous()

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return img.permute(params['axis_order']).contiguous()

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return regression_target.permute(params['axis_order']).contiguous()

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        if keypoints is None:
            return None
        axis_order = [int(ax) for ax in params['axis_order']]
        spatial_perm = [ax - 1 for ax in axis_order[1:]]
        return keypoints[:, spatial_perm]

    def _apply_to_vectors(self, vectors, **params):
        if vectors is None:
            return None
        axis_order = [int(ax) for ax in params['axis_order']]
        spatial_perm = [ax - 1 for ax in axis_order[1:]]
        return vectors[:, spatial_perm]

    def apply(self, data_dict: dict, **params) -> dict:
        """
        Extend base apply to also permute normal vectors so their components follow the axis swap.
        """
        if data_dict.get('_skip_spatial_transforms', False) and self._skip_when_vector:
            return data_dict

        # Remove normals so base class doesn't treat them as scalar labels
        normals_cache = {}
        for key in self.normal_keys:
            if key in data_dict:
                normals_cache[key] = data_dict.pop(key)

        data_dict = super().apply(data_dict, **params)

        if normals_cache:
            axis_order = params.get('axis_order')
            if axis_order is None:
                raise ValueError("TransposeAxesTransform requires 'axis_order' to remap normals.")

            axis_order_list = [int(ax) for ax in axis_order]
            channel_perm = [ax - 1 for ax in axis_order_list[1:]]  # map (Z,Y,X) -> permuted order
            expected_channels = len(channel_perm)
            if sorted(channel_perm) != list(range(expected_channels)):
                raise ValueError(f"Invalid axis order for normals: {axis_order_list}.")

            for key, normals in normals_cache.items():
                if normals is None:
                    data_dict[key] = normals
                    continue
                if normals.ndim != len(axis_order_list):
                    raise ValueError(
                        f"Normals '{key}' must have {len(axis_order_list)} dims "
                        f"(got shape {tuple(normals.shape)})."
                    )
                if normals.shape[0] != expected_channels:
                    raise ValueError(
                        f"Normals '{key}' need {expected_channels} channels to match spatial dims "
                        f"(got {normals.shape[0]})."
                    )
                permuted = normals[channel_perm]
                permuted = permuted.permute(axis_order_list).contiguous()
                data_dict[key] = permuted

        # Handle vector_keys
        vector_keys = set(data_dict.get('vector_keys', []) or [])
        for key in vector_keys:
            if data_dict.get(key) is not None:
                data_dict[key] = self._apply_to_vectors(data_dict[key], **params)

        # Transform padding_mask with the same permutation
        if data_dict.get('padding_mask') is not None:
            data_dict['padding_mask'] = self._apply_to_image(data_dict['padding_mask'], **params)

        return data_dict

if __name__ == '__main__':
    t = TransposeAxesTransform((1, 2))
    ret = t(**{'image': torch.rand((2, 31, 32, 32)), 'segmentation': torch.ones((1, 31, 32, 32))})
