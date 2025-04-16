from typing import Tuple, Union, List

import numpy as np
import torch
from torch.nn.functional import grid_sample

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import _create_centered_identity_grid2, _convert_my_grid_to_grid_sample_grid


class ArbitraryRotationTransform(BasicTransform):
    """
    Performs arbitrary angle rotation on each dimension independently.
    
    Args:
        rotation_angle_range: Tuple or range of possible rotation angles in radians
        p_per_axis: Probability of applying rotation for each axis
        interpolation_mode_img: Interpolation mode for image data
        interpolation_mode_seg: Interpolation mode for segmentation data
    """
    def __init__(
        self,
        rotation_angle_range: RandomScalar = (-np.pi/4, np.pi/4),  # Default: +/- 45 degrees
        p_per_axis: Union[float, Tuple[float, float, float]] = 0.5,
        interpolation_mode_img: str = 'bilinear',
        interpolation_mode_seg: str = 'nearest',
    ):
        super().__init__()
        self.rotation_angle_range = rotation_angle_range
        
        # Handle probability per axis
        if isinstance(p_per_axis, float):
            self.p_per_axis = (p_per_axis, p_per_axis, p_per_axis)
        else:
            self.p_per_axis = p_per_axis
            
        self.interpolation_mode_img = interpolation_mode_img
        self.interpolation_mode_seg = interpolation_mode_seg

    def get_parameters(self, **data_dict) -> dict:
        # Get the dimension of the data (2D or 3D)
        dim = data_dict['image'].ndim - 1
        
        # Get device from input data for consistent tensor device usage
        device = data_dict['image'].device
        
        # Sample angles for each axis
        angles = []
        for i in range(dim):
            # Determine if we apply rotation to this axis
            if torch.rand(1, device=device).item() < self.p_per_axis[i]:
                angle = sample_scalar(self.rotation_angle_range, image=data_dict['image'], dim=i)
            else:
                angle = 0
            angles.append(angle)
            
        # Create rotation matrix based on dimension
        if dim == 3:
            affine = self._create_affine_matrix_3d(angles)
        elif dim == 2:
            affine = self._create_affine_matrix_2d(angles[-1])  # Only use the last angle for 2D
        else:
            raise RuntimeError(f'Unsupported dimension: {dim}')
            
        # Move affine matrix to same device as input data
        affine = affine.to(device)
            
        return {
            'affine': affine,
            'dim': dim
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        # Skip if no rotation - using torch.allclose instead of np.allclose
        identity_matrix = torch.eye(params['dim'], dtype=torch.float32, device=img.device)
        if torch.allclose(params['affine'], identity_matrix):
            return img
            
        # Create the identity grid with the same shape as the image
        grid = _create_centered_identity_grid2(img.shape[1:])
        
        # Apply rotation transform to grid - params['affine'] is already a torch tensor now
        grid = torch.matmul(grid, params['affine'])
        
        # Convert grid to the format expected by grid_sample
        grid = _convert_my_grid_to_grid_sample_grid(grid, img.shape[1:])
        
        # Apply grid_sample to perform the rotation
        return grid_sample(
            img[None], 
            grid[None], 
            mode=self.interpolation_mode_img, 
            padding_mode="zeros", 
            align_corners=False
        )[0]

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        # Skip if no rotation - using torch.allclose instead of np.allclose
        identity_matrix = torch.eye(params['dim'], dtype=torch.float32, device=segmentation.device)
        if torch.allclose(params['affine'], identity_matrix):
            return segmentation
            
        # Create the identity grid with the same shape as the segmentation
        grid = _create_centered_identity_grid2(segmentation.shape[1:])
        
        # Apply rotation transform to grid - params['affine'] is already a torch tensor now
        grid = torch.matmul(grid, params['affine'])
        
        # Convert grid to the format expected by grid_sample
        grid = _convert_my_grid_to_grid_sample_grid(grid, segmentation.shape[1:])
        
        # Apply grid_sample with the segmentation interpolation mode
        return grid_sample(
            segmentation[None].float(), 
            grid[None], 
            mode=self.interpolation_mode_seg, 
            padding_mode="zeros", 
            align_corners=False
        )[0].to(segmentation.dtype)

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        # Distance maps should be transformed like images
        return self._apply_to_image(dist_map, **params)

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        # Regression targets should be transformed like images
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        # Not implemented for now
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        # Not implemented for now
        raise NotImplementedError

    def _create_affine_matrix_3d(self, rotation_angles: List[float]) -> torch.Tensor:
        """Create a 3D rotation matrix from the given angles using PyTorch."""
        # Rotation matrices for each axis using torch instead of numpy
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(torch.tensor(rotation_angles[0])), -torch.sin(torch.tensor(rotation_angles[0]))],
                           [0, torch.sin(torch.tensor(rotation_angles[0])), torch.cos(torch.tensor(rotation_angles[0]))]], 
                           dtype=torch.float32)

        Ry = torch.tensor([[torch.cos(torch.tensor(rotation_angles[1])), 0, torch.sin(torch.tensor(rotation_angles[1]))],
                           [0, 1, 0],
                           [-torch.sin(torch.tensor(rotation_angles[1])), 0, torch.cos(torch.tensor(rotation_angles[1]))]], 
                           dtype=torch.float32)

        Rz = torch.tensor([[torch.cos(torch.tensor(rotation_angles[2])), -torch.sin(torch.tensor(rotation_angles[2])), 0],
                           [torch.sin(torch.tensor(rotation_angles[2])), torch.cos(torch.tensor(rotation_angles[2])), 0],
                           [0, 0, 1]], 
                           dtype=torch.float32)

        # Combine rotation matrices (order matters in 3D rotations)
        R = Rz @ Ry @ Rx
        return R

    def _create_affine_matrix_2d(self, rotation_angle: float) -> torch.Tensor:
        """Create a 2D rotation matrix from the given angle using PyTorch."""
        # 2D rotation matrix using torch instead of numpy
        angle_tensor = torch.tensor(rotation_angle, dtype=torch.float32)
        R = torch.tensor([[torch.cos(angle_tensor), -torch.sin(angle_tensor)],
                          [torch.sin(angle_tensor), torch.cos(angle_tensor)]], 
                          dtype=torch.float32)
        return R


if __name__ == '__main__':
    # Simple test code
    rotation_transform = ArbitraryRotationTransform(
        rotation_angle_range=(-np.pi/4, np.pi/4),  # +/- 45 degrees
        p_per_axis=0.8  # 80% chance of rotation per axis
    )
    
    # Create a test image with a simple pattern
    test_image = torch.zeros((1, 64, 64, 64))
    test_image[:, 20:40, 20:40, 20:40] = 1  # Create a cube in the center
    
    # Apply rotation
    data_dict = {'image': test_image}
    result = rotation_transform(**data_dict)
    
    print(f"Rotation applied successfully: {result['image'].shape}")