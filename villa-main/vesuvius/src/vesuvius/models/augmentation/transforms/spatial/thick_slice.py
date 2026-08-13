import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class SimulateThickSliceTransform(BasicTransform):
    """
    Simulates thicker slice acquisition by downsampling along a single axis.

    Downsamples the image along one randomly chosen spatial axis and then
    restores it to the original resolution using nearest-neighbor interpolation.
    This simulates the effect of acquiring data with thicker slices.

    Different from SimulateLowResolutionTransform which affects all axes.

    Parameters:
        scale_range (Tuple[float, float]): Range of downsampling factors (e.g., 0.25-0.6
                                            means downsample to 25-60% of original size).
        candidate_axes (Tuple[int, ...]): Spatial axes that can be selected for
                                          downsampling (0-indexed, excluding channel dim).
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.25, 0.6),
        candidate_axes: Tuple[int, ...] = (0, 1, 2),
    ):
        super().__init__()
        self.scale_range = scale_range
        self.candidate_axes = candidate_axes

    @staticmethod
    def _resize_tensor(tensor: torch.Tensor, target_size: Tuple[int, ...]) -> torch.Tensor:
        """Resize tensor to target spatial size using nearest interpolation."""
        original_dtype = tensor.dtype
        needs_cast = not tensor.is_floating_point()
        work_tensor = tensor.float() if needs_cast else tensor

        resized = F.interpolate(
            work_tensor.unsqueeze(0),
            size=target_size,
            mode="nearest"
        ).squeeze(0)

        if needs_cast:
            resized = resized.to(original_dtype)
        return resized

    def get_parameters(self, **data_dict) -> dict:
        image = data_dict.get('image')
        if image is None or image.ndim != 4:
            return {'apply': False}

        spatial_shape = image.shape[1:]  # (D, H, W) or similar
        valid_axes = [ax for ax in self.candidate_axes if 0 <= ax < len(spatial_shape)]

        if not valid_axes:
            return {'apply': False}

        chosen_axis = int(np.random.choice(valid_axes))
        scale = float(np.random.uniform(*self.scale_range))

        target_shape = list(spatial_shape)
        target_dim = max(1, int(round(target_shape[chosen_axis] * scale)))

        # Skip if no actual change
        if target_dim == target_shape[chosen_axis]:
            return {'apply': False}

        target_shape[chosen_axis] = target_dim

        return {
            'apply': True,
            'chosen_axis': chosen_axis,
            'scale': scale,
            'target_shape': tuple(target_shape),
            'original_shape': tuple(spatial_shape),
        }

    def _apply_thick_slice(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        """Apply the thick slice simulation to a tensor."""
        if not params.get('apply', False):
            return tensor

        target_shape = params['target_shape']
        original_shape = params['original_shape']

        # Convert to float for interpolation
        original_dtype = tensor.dtype
        needs_cast = not tensor.is_floating_point()
        work_tensor = tensor.float() if needs_cast else tensor

        # Downsample
        downsampled = self._resize_tensor(work_tensor, target_shape)

        # Restore to original size
        restored = F.interpolate(
            downsampled.unsqueeze(0),
            size=original_shape,
            mode="nearest"
        ).squeeze(0)

        # Restore original dtype
        if needs_cast:
            restored = torch.round(restored).to(original_dtype)
        else:
            restored = restored.to(original_dtype)

        return restored

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_thick_slice(img, **params)

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        if not params.get('apply', False):
            return seg

        target_shape = params['target_shape']
        original_shape = params['original_shape']

        original_dtype = seg.dtype
        work_tensor = seg.float()

        # Downsample and restore
        downsampled = self._resize_tensor(work_tensor, target_shape)
        restored = F.interpolate(
            downsampled.unsqueeze(0),
            size=original_shape,
            mode="nearest"
        ).squeeze(0)

        # Handle different dtypes for segmentation
        if original_dtype == torch.bool:
            restored = restored > 0.5
        elif original_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            restored = torch.round(restored).to(original_dtype)
        else:
            restored = restored.to(original_dtype)

        return restored

    def _apply_to_regr_target(self, regression_target: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_thick_slice(regression_target, **params)

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_thick_slice(dist_map, **params)

    def _apply_to_bbox(self, bbox, **params):
        # Bounding boxes are not affected by resolution changes
        return bbox

    def _apply_to_keypoints(self, keypoints, **params):
        # Keypoints are not affected by resolution changes
        return keypoints
