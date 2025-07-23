from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from models.augmentation.transforms.base.basic_transform import BasicTransform

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

    def _build_defects(self, num_slices: int) -> np.ndarray:
        int_factors = np.ones(num_slices)

        # Gaussian shaped illumination changes
        num_gaussians = int(np.round(self._sample(self.num_defects)))
        for _ in range(num_gaussians):
            sigma = self._sample(self.defect_width)
            pos = np.random.choice(num_slices)
            tmp = np.zeros(num_slices)
            tmp[pos] = 1
            tmp = gaussian_filter(tmp, sigma, mode='constant', truncate=3)
            tmp = tmp / tmp.max()
            strength = self._sample(self.mult_brightness_reduction_at_defect)
            int_factors *= (1 - (tmp * (1 - strength)))

        int_factors = np.clip(int_factors, 0.1, 1)
        ps = np.ones(num_slices) / num_slices
        ps += (1 - int_factors) / num_slices
        ps /= ps.sum()
        
        idx = np.random.choice(
            num_slices, 
            int(np.round(self._sample(self.base_p) * num_slices)), 
            replace=False, 
            p=ps
        )
        noise = np.random.uniform(*self.base_red, size=len(idx))
        int_factors[idx] *= noise
        int_factors = np.clip(int_factors, 0.1, 2)
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
                        defects = self._build_defects(img.shape[1])
                        result[c] *= torch.from_numpy(defects[:, None, None]).float()
            else:
                defects = self._build_defects(img.shape[1])
                for c in range(img.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        result[c] *= torch.from_numpy(defects[:, None, None]).float()
        
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