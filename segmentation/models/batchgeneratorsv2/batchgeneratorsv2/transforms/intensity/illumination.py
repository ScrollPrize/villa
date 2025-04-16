from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

class InhomogeneousSliceIlluminationTransform(BasicTransform):
    """
    Simulates inhomogeneous illumination across image slices for batchgeneratorsv2.
    """
    def __init__(self, 
                 num_defects: Tuple[int, int],
                 defect_width: Tuple[float, float],
                 mult_brightness_reduction_at_defect: Tuple[float, float],
                 base_p: Tuple[float, float],
                 base_red: Tuple[float, float],
                 p_per_sample: float = 1.0,
                 per_channel: bool = True,
                 p_per_channel: float = 0.5):
        super().__init__()
        self.num_defects = num_defects
        self.defect_width = defect_width
        self.mult_brightness_reduction_at_defect = mult_brightness_reduction_at_defect
        self.base_p = base_p
        self.base_red = base_red
        self.p_per_sample = p_per_sample
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    @staticmethod
    def _sample(value, device=None):
        """Sample values using PyTorch random functions"""
        if isinstance(value, (float, int)):
            return value
        elif isinstance(value, (tuple, list)):
            assert len(value) == 2
            # Use torch.rand instead of np.random.uniform
            return value[0] + (value[1] - value[0]) * torch.rand(1, device=device).item()
        elif callable(value):
            result = value()
            # If the function returns a tensor, return it as is
            if isinstance(result, torch.Tensor):
                return result.item() if result.numel() == 1 else result
            return result
        else:
            raise ValueError('Invalid input for sampling.')

    def _build_defects(self, num_slices: int, device=None) -> torch.Tensor:
        """Build illumination defects using PyTorch operations"""
        # Start with ones tensor instead of numpy array
        int_factors = torch.ones(num_slices, device=device)

        # Gaussian shaped illumination changes
        num_gaussians = int(round(self._sample(self.num_defects, device)))
        
        for _ in range(num_gaussians):
            sigma = self._sample(self.defect_width, device)
            
            # Use torch.randint for position selection
            pos = torch.randint(0, num_slices, (1,), device=device).item()
            
            # Create tensor on device
            tmp = torch.zeros(num_slices, device=device)
            tmp[pos] = 1
            
            # Since PyTorch doesn't have a direct equivalent of scipy's gaussian_filter,
            # we'll create a Gaussian kernel and use conv1d
            kernel_size = int(6 * sigma) | 1  # make sure it's odd and covers 3 sigma on each side
            kernel_size = max(3, kernel_size)  # minimum size 3
            
            # Create Gaussian kernel
            x = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, device=device)
            kernel = torch.exp(-0.5 * (x / sigma).pow(2))
            kernel = kernel / kernel.sum()
            
            # Reshape for conv1d: [out_channels, in_channels, kernel_size]
            kernel = kernel.view(1, 1, -1)
            
            # Reshape tmp for conv1d: [batch, channels, length]
            tmp = tmp.view(1, 1, -1)
            
            # Apply convolution
            tmp = torch.nn.functional.conv1d(
                tmp, kernel, padding=kernel_size//2
            )
            
            # Reshape back to original shape
            tmp = tmp.view(-1)
            
            # Normalize
            if tmp.max() > 0:
                tmp = tmp / tmp.max()
                
            strength = self._sample(self.mult_brightness_reduction_at_defect, device)
            int_factors *= (1 - (tmp * (1 - strength)))

        # Clip values
        int_factors = torch.clamp(int_factors, 0.1, 1)
        
        # Calculate probabilities for sampling
        ps = torch.ones(num_slices, device=device) / num_slices
        ps += (1 - int_factors) / num_slices
        ps = ps / ps.sum()
        
        # Sample indices - this is complex in PyTorch, we need a vectorized approach
        # First, determine how many indices to sample
        num_to_sample = int(round(self._sample(self.base_p, device) * num_slices))
        
        if num_to_sample > 0:
            # PyTorch's multinomial sampling for weighted random choice
            idx = torch.multinomial(ps, num_to_sample, replacement=False)
            
            # Generate uniform random noise
            noise = torch.rand(len(idx), device=device) * (self.base_red[1] - self.base_red[0]) + self.base_red[0]
            
            # Apply noise to selected indices
            int_factors.index_add_(0, idx, int_factors[idx] * (noise - 1))
            
        # Final clipping
        int_factors = torch.clamp(int_factors, 0.1, 2)
        return int_factors

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        assert len(img.shape) == 4, "This transform expects 4D input (CDHW)"
        result = img.clone()
        device = img.device
        
        # Use torch.rand instead of np.random.uniform
        if torch.rand(1, device=device).item() < self.p_per_sample:
            if self.per_channel:
                for c in range(img.shape[0]):
                    if torch.rand(1, device=device).item() < self.p_per_channel:
                        # Get defects directly as a tensor on the correct device
                        defects = self._build_defects(img.shape[1], device)
                        # No need for from_numpy conversion, already a tensor
                        # Use unsqueeze instead of None indexing
                        result[c] *= defects.unsqueeze(1).unsqueeze(2)
            else:
                # Single defect pattern for all channels
                defects = self._build_defects(img.shape[1], device)
                for c in range(img.shape[0]):
                    if torch.rand(1, device=device).item() < self.p_per_channel:
                        result[c] *= defects.unsqueeze(1).unsqueeze(2)
        
        return result

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **kwargs) -> torch.Tensor:
        return segmentation  # Don't modify segmentations

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **kwargs) -> torch.Tensor:
        # DO NOT blank anything in the distance map
        # (this is an intensity transform, not geometric)
        return dist_map

    def _apply_to_bbox(self, bbox, **kwargs):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **kwargs):
        raise NotImplementedError

    def _apply_to_regr_target(self, regr_target: torch.Tensor, **kwargs) -> torch.Tensor:
        return regr_target  # Don't modify regression targets