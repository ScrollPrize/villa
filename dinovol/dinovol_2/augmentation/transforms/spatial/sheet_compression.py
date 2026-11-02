import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from dinovol_2.augmentation.transforms.base.basic_transform import BasicTransform
from dinovol_2.augmentation.transforms.noise.gaussian_blur import blur_dimension
from dinovol_2.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class SheetCompressionTransform(BasicTransform):
    """
    Simulates scroll sheets being closer together by preferentially
    compressing low-intensity (gap) regions along a configurable axis.

    The transform works by:
    1. Computing a "gap weight" from intensity (low intensity = gap = 1, high = sheet = 0)
    2. Accumulating gap weights along the compression axis to get displacement
    3. Smoothing the displacement field perpendicular to the compression axis
    4. Applying the displacement via grid_sample

    This preserves sheet thickness while compressing the air gaps between them.

    Parameters:
        compression_strength: Fraction of gap depth to compress (0.1-0.3 typical)
        intensity_sigma: Gaussian sigma for smoothing intensity before gap detection
        spatial_smoothing: Sigma for smoothing displacement perpendicular to compression axis
        use_threshold: If True, use hard threshold; if False, use soft intensity weighting
        threshold: Intensity threshold for gap detection (if use_threshold=True)
        compression_axes: Candidate axes to compress (randomly selects one per sample)
    """

    _is_spatial = True  # Skip per-transform padding restoration

    def __init__(
        self,
        compression_strength: RandomScalar = (0.1, 0.3),
        intensity_sigma: RandomScalar = 1.0,
        spatial_smoothing: RandomScalar = (3.0, 8.0),
        use_threshold: bool = False,
        threshold: RandomScalar = 0.3,
        compression_axes: Tuple[int, ...] = (0, 1, 2),
    ):
        super().__init__()
        self.compression_strength = compression_strength
        self.intensity_sigma = intensity_sigma
        self.spatial_smoothing = spatial_smoothing
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.compression_axes = compression_axes

    def get_parameters(self, **data_dict) -> dict:
        image = data_dict.get('image')
        if image is None or image.ndim != 4:
            return {'apply': False}

        spatial_shape = image.shape[1:]  # (D, H, W)
        valid_axes = [ax for ax in self.compression_axes if 0 <= ax < len(spatial_shape)]

        if not valid_axes:
            return {'apply': False}

        # Select random axis and sample parameters
        chosen_axis = int(np.random.choice(valid_axes))
        compression_strength = sample_scalar(self.compression_strength, image.shape)
        intensity_sigma = sample_scalar(self.intensity_sigma, image.shape)
        spatial_smoothing = sample_scalar(self.spatial_smoothing, image.shape)
        threshold = sample_scalar(self.threshold, image.shape) if self.use_threshold else None

        # Compute displacement field from the image
        # Use first channel if multi-channel
        img_for_gap = image[0] if image.shape[0] > 1 else image[0]

        # Step 1: Smooth intensity for noise reduction
        smoothed = img_for_gap.clone()
        if intensity_sigma > 0.5:
            for dim in range(3):
                smoothed = blur_dimension(smoothed[None], intensity_sigma, dim)[0]

        # Step 2: Normalize to [0, 1]
        smin, smax = smoothed.min(), smoothed.max()
        if smax - smin > 1e-8:
            normalized = (smoothed - smin) / (smax - smin)
        else:
            # Uniform intensity - no compression needed
            return {'apply': False}

        # Step 3: Compute gap weight (1 = gap, 0 = sheet)
        if self.use_threshold:
            gap_weight = (normalized < threshold).float()
        else:
            gap_weight = 1.0 - normalized

        # Step 4: Cumulative sum along compression axis to get displacement
        # Add 1 to axis because gap_weight is (D, H, W), not (C, D, H, W)
        cumulative_gap = torch.cumsum(gap_weight, dim=chosen_axis)

        # Scale by compression strength
        displacement = cumulative_gap * compression_strength

        # Step 5: Smooth displacement perpendicular to compression axis
        if spatial_smoothing > 0.5:
            displacement = displacement[None]  # Add channel dim for blur_dimension
            for dim in range(3):
                if dim != chosen_axis:
                    displacement = blur_dimension(displacement, spatial_smoothing, dim)
            displacement = displacement[0]

        # Normalize displacement to grid_sample coordinates [-1, 1].
        # With align_corners=True, one voxel step spans 2/(dim-1) in grid
        # coordinates, so voxel-unit displacement converts by that factor.
        dim_size = spatial_shape[chosen_axis]
        displacement_normalized = displacement * (2.0 / max(dim_size - 1.0, 1.0))

        return {
            'apply': True,
            'chosen_axis': chosen_axis,
            'displacement': displacement_normalized,
            'spatial_shape': tuple(spatial_shape),
        }

    def _create_sampling_grid(self, shape: Tuple[int, ...], displacement: torch.Tensor,
                               axis: int, device: torch.device) -> torch.Tensor:
        """Create a sampling grid with displacement applied along the specified axis."""
        D, H, W = shape

        # Create normalized coordinate grids [-1, 1]
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)

        # Create meshgrid - grid_sample expects (N, D, H, W, 3) with (x, y, z) order
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')

        # Stack in (x, y, z) order for grid_sample
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D, H, W, 3)

        # Apply displacement to the appropriate axis
        # grid_sample uses (x, y, z) order, so axis 0 (D) -> index 2, axis 1 (H) -> index 1, axis 2 (W) -> index 0
        grid_axis_map = {0: 2, 1: 1, 2: 0}
        grid_idx = grid_axis_map[axis]

        # Add displacement: output[x] = input[x + d(x)] with d monotonically
        # non-decreasing along the axis pulls content toward smaller indices,
        # so accumulated gap width shrinks (local scale ds/dx = 1 + s*gap > 1).
        # (Subtracting instead stretches the volume and *widens* gaps.)
        grid[..., grid_idx] = grid[..., grid_idx] + displacement

        return grid.unsqueeze(0)  # (1, D, H, W, 3)

    def _grid_for(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        """Sampling grid on the spatial shape of ``tensor``.

        For spatial transforms BasicTransform also warps ``paired_spatial_image``.
        The dataset builds that view from the same crop as the main view, resized
        with ``align_corners=False`` (``_finalize_crop``), so the two views share
        their outer edges. The displacement is converted to voxels of the main
        view, resampled to the other shape with the same edge-aligned convention,
        rescaled to the other view's voxels and normalised again for its grid.
        """
        displacement = params['displacement'].to(tensor.device)
        shape = tuple(tensor.shape[1:])
        spatial_shape = tuple(params['spatial_shape'])
        if shape != spatial_shape:
            axis = params['chosen_axis']
            n_main, n_other = float(spatial_shape[axis]), float(shape[axis])
            voxels = displacement * (max(n_main - 1.0, 1.0) / 2.0)
            voxels = F.interpolate(
                voxels[None, None], size=shape, mode='trilinear', align_corners=False
            )[0, 0]
            voxels = voxels * (n_other / n_main)
            displacement = voxels * (2.0 / max(n_other - 1.0, 1.0))
        return self._create_sampling_grid(shape, displacement, params['chosen_axis'], tensor.device)

    def _apply_compression(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        """Apply the compression transform to a tensor."""
        if not params.get('apply', False):
            return tensor

        # Create sampling grid
        grid = self._grid_for(tensor, params)

        # Apply grid_sample
        original_dtype = tensor.dtype
        needs_cast = not tensor.is_floating_point()
        work_tensor = tensor.float() if needs_cast else tensor

        # grid_sample expects (N, C, D, H, W)
        output = F.grid_sample(
            work_tensor.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(0)

        if needs_cast:
            output = output.to(original_dtype)

        return output

    def _apply_compression_nearest(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        """Apply compression with nearest neighbor interpolation (for segmentation)."""
        if not params.get('apply', False):
            return tensor

        grid = self._grid_for(tensor, params)

        original_dtype = tensor.dtype
        work_tensor = tensor.float()

        output = F.grid_sample(
            work_tensor.unsqueeze(0),
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        ).squeeze(0)

        # Restore dtype
        if original_dtype == torch.bool:
            output = output > 0.5
        elif original_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            output = torch.round(output).to(original_dtype)
        else:
            output = output.to(original_dtype)

        return output

    def apply(self, data_dict, **params):
        data_dict = super().apply(data_dict, **params)
        # _is_spatial stops BasicTransform from restoring padding from the
        # un-warped mask, so the mask has to move with the image, like in
        # SpatialTransform.
        if data_dict.get('padding_mask') is not None:
            data_dict['padding_mask'] = self._apply_to_padding_mask(data_dict['padding_mask'], **params)
        return data_dict

    def _apply_to_padding_mask(self, mask: torch.Tensor, **params) -> torch.Tensor:
        """Warp the padding mask with the grid used for image and labels.

        Nearest-neighbour keeps it binary. Zeros padding marks as padding the
        tail that the compression pulls in from outside the volume (the image
        repeats its border there).
        """
        if not params.get('apply', False):
            return mask

        grid = self._grid_for(mask, params)

        output = F.grid_sample(
            mask.float().unsqueeze(0),
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0)

        if mask.dtype == torch.bool:
            return output > 0.5
        return output.to(mask.dtype)

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_compression(img, **params)

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_compression_nearest(seg, **params)

    def _apply_to_regr_target(self, regression_target: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_compression(regression_target, **params)

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_compression(dist_map, **params)

    def _apply_to_bbox(self, bbox, **params):
        # Bounding boxes would need complex transformation - skip for now
        return bbox

    def _apply_to_keypoints(self, keypoints, **params):
        # Keypoints would need coordinate transformation - skip for now
        return keypoints
