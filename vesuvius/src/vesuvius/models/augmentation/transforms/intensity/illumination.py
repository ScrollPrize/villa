from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform

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
    def _sample(value):
        if isinstance(value, (float, int)):
            return value
        elif isinstance(value, (tuple, list)):
            assert len(value) == 2
            return np.random.uniform(*value)
        elif callable(value):
            return value()
        else:
            raise ValueError('Invalid input for sampling.')

    def _build_defects_torch(self, num_slices: int, device, dtype) -> torch.Tensor:
        """
        Torch implementation of illumination defects (device-native).
        Returns a tensor of shape [num_slices] on the given device/dtype.
        """
        int_factors = torch.ones(num_slices, device=device, dtype=dtype)

        # Gaussian shaped illumination changes
        num_gaussians = int(np.round(self._sample(self.num_defects)))
        if num_gaussians > 0:
            x = torch.arange(num_slices, device=device, dtype=dtype)
            for _ in range(num_gaussians):
                sigma = float(self._sample(self.defect_width))
                pos = int(np.random.choice(num_slices))
                # Build normalized Gaussian bump centered at pos
                tmp = torch.exp(-0.5 * ((x - pos) / max(sigma, 1e-6)) ** 2)
                tmp = tmp / torch.clamp(tmp.max(), min=1e-6)
                strength = float(self._sample(self.mult_brightness_reduction_at_defect))
                int_factors = int_factors * (1 - (tmp * (1 - strength)))

        int_factors = int_factors.clamp(0.1, 1.0)

        # Sampling probabilities for base reductions (CPU rng fine here; used only for indices)
        ps = torch.ones(num_slices, device=device, dtype=dtype) / num_slices
        ps = (ps + (1 - int_factors) / num_slices)
        ps = ps / torch.clamp(ps.sum(), min=1e-6)

        k = int(np.round(self._sample(self.base_p) * num_slices))
        if k > 0:
            # Use torch.multinomial for sampling without replacement
            idx = torch.multinomial(ps, num_samples=k, replacement=False)
            noise_low, noise_high = self.base_red
            noise = torch.rand(idx.shape[0], device=device, dtype=dtype) * (noise_high - noise_low) + noise_low
            int_factors[idx] = int_factors[idx] * noise

        int_factors = int_factors.clamp(0.1, 2.0)
        return int_factors

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        assert len(img.shape) == 4, "This transform expects 4D input (CDHW)"
        result = img.clone()
        
        if np.random.uniform() < self.p_per_sample:
            if self.per_channel:
                for c in range(img.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        defects = self._build_defects_torch(img.shape[1], device=result.device, dtype=result.dtype)
                        result[c] *= defects.view(-1, 1, 1)
            else:
                defects = self._build_defects_torch(img.shape[1], device=result.device, dtype=result.dtype)
                for c in range(img.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        result[c] *= defects.view(-1, 1, 1)
        
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
