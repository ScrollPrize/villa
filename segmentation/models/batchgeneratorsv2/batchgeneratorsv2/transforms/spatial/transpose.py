from typing import Set
import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class TransposeAxesTransform(BasicTransform):
    """
    A transformation class to permute specified spatial axes of an image and related data.

    Attributes:
        allowed_axes (Set[int]): Set of spatial axes allowed for permutation (e.g., {1, 2} for y and z axes in an
        image of shape (c, x, y, z)).
    """

    def __init__(self, allowed_axes: Set[int]):
        """
        Initialize the transform with allowed spatial axes for permutation.

        Args:
            allowed_axes (Set[int]): Set of spatial axis indices for permutation.
        """
        super().__init__()
        self.allowed_axes = allowed_axes

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
        raise NotImplementedError

if __name__ == '__main__':
    t = TransposeAxesTransform((1, 2))
    ret = t(**{'image': torch.rand((2, 31, 32, 32)), 'segmentation': torch.ones((1, 31, 32, 32))})
