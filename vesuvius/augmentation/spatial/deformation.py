import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Union, Optional, Tuple, List, Dict
from ..base import BaseAugmentation, ArrayLike

# For true scipy-based implementation
try:
    from scipy.ndimage import gaussian_filter, map_coordinates
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Helper: 3D Gaussian filtering using conv3d
def gaussian_filter_3d(input_tensor: torch.Tensor, sigma: float, kernel_size: Optional[int] = None) -> torch.Tensor:
    """
    Applies a 3D Gaussian filter to an input tensor using separable convolution.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor of shape (N, C, D, H, W) to be smoothed.
    sigma : float
        Standard deviation of the Gaussian.
    kernel_size : Optional[int]
        Size of the Gaussian kernel. If None, computed as 4*sigma+1 (ensuring odd size).

    Returns
    -------
    torch.Tensor
        Smoothed tensor of the same shape as input_tensor.
    """
    if kernel_size is None:
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=input_tensor.dtype, device=input_tensor.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # Create 3D kernel by outer product (separable filter)
    g3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
    g3d = g3d.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, k, k, k)
    # Expand kernel for each channel (assumes input_tensor shape: (N, C, D, H, W))
    C = input_tensor.shape[1]
    kernel = g3d.repeat(C, 1, 1, 1, 1)
    padding = kernel_size // 2
    return F.conv3d(input_tensor, kernel, padding=padding, groups=C)


class TorchElasticDeformation(BaseAugmentation):
    """
    Applies elastic deformation to 3D volumes using a PyTorch-based implementation.

    The deformation is controlled by:
    - alpha: Controls the intensity of the deformation
    - sigma: Controls the smoothness of the deformation (via Gaussian filtering)

    For volume data, bilinear/trilinear interpolation is used,
    while for label data nearest-neighbor interpolation is applied.
    """

    def __init__(
            self,
            volume: Optional[ArrayLike] = None,
            label: Optional[ArrayLike] = None,
            alpha: float = 500.0,  # Deformation intensity (higher = stronger)
            sigma: float = 20.0,  # Deformation smoothness (higher = smoother)
            order: int = 1,  # (Unused; grid_sample supports 'bilinear' or 'nearest')
            order_label: int = 0,  # (Unused; see mode below)
            random_state: Optional[int] = None,  # Random seed for reproducibility
            device: Optional[Union[str, torch.device]] = 'cpu',
            **kwargs
    ):
        super().__init__(volume, label, device, **kwargs)
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.order_label = order_label
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)

    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply elastic deformation to the input data using PyTorch.

        Parameters
        ----------
        data : torch.Tensor
            Input data to transform. Can be 3D (D, H, W) or 4D (C, D, H, W).

        Returns
        -------
        torch.Tensor
            Deformed data.
        """
        # If input is not a tensor, convert it
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        # Record whether input was 3D (no channel dim)
        orig_was_3d = (data.dim() == 3)
        if orig_was_3d:
            data = data.unsqueeze(0)  # now (1, D, H, W)

        # Ensure data is float type for interpolation
        data = data.float()

        # Assume data shape is now (C, D, H, W)
        if data.dim() != 4:
            raise ValueError("Data must be 3D or 4D (with channel dimension).")
        C, D, H, W = data.shape

        # Generate random displacement field in pixel space (order: dx, dy, dz)
        # Note: grid_sample expects last dim order (x, y, z)
        displacement = torch.randn((1, 3, D, H, W), device=data.device) * self.alpha

        # Smooth the displacement field with a Gaussian filter (apply separately to each channel)
        displacement = gaussian_filter_3d(displacement, self.sigma)

        # Rearrange to (D, H, W, 3) and remove batch dimension
        displacement = displacement.permute(0, 2, 3, 4, 1).squeeze(0)

        # Choose interpolation mode based on whether data is label (nearest) or volume (bilinear)
        mode = 'nearest' if (self.label is not None and data is self.label) else 'bilinear'

        # Create a base grid with normalized coordinates in [-1, 1]
        z = torch.linspace(-1, 1, D, device=data.device)
        y = torch.linspace(-1, 1, H, device=data.device)
        x = torch.linspace(-1, 1, W, device=data.device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # shape (D, H, W, 3)

        # Scale displacement to normalized coordinates (note the order: x, y, z)
        scale = torch.tensor([2.0 / (W - 1), 2.0 / (H - 1), 2.0 / (D - 1)], device=data.device)
        displacement = displacement * scale.view(1, 1, 3)

        # Create the sampling grid and add the displacement
        sampling_grid = base_grid + displacement  # shape (D, H, W, 3)
        sampling_grid = sampling_grid.unsqueeze(0)  # add batch dim -> (1, D, H, W, 3)

        # Prepare data for grid_sample (needs batch dim): (1, C, D, H, W)
        data = data.unsqueeze(0)

        # Apply deformation using grid_sample
        deformed = F.grid_sample(data, sampling_grid, mode=mode, padding_mode='border', align_corners=True)
        deformed = deformed.squeeze(0)  # (C, D, H, W)

        # If the original input was 3D, remove the channel dimension
        if orig_was_3d:
            deformed = deformed.squeeze(0)
        return deformed


class ElasticDeformation(BaseAugmentation):
    """
    Applies elastic deformation to 3D volumes using a coarse grid of random displacements.

    The deformation is controlled by:
    - strength: maximum displacement (alpha)
    - grid_size: spacing between control points (inversely related to sigma)

    Interpolation:
      - Volume data: bilinear/trilinear interpolation
      - Label data: nearest neighbor interpolation
    """

    def __init__(
            self,
            volume: Optional[ArrayLike] = None,
            label: Optional[ArrayLike] = None,
            strength: float = 10.0,
            grid_size: int = 32,
            seed: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = 'cuda',
            **kwargs
    ):
        super().__init__(volume, label, device, **kwargs)
        self.strength = strength
        self.grid_size = grid_size
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _create_deformation_field(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Create a dense displacement field by interpolating random control point displacements.
        """
        # Determine spatial dimensions
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape

        # Compute coarse grid dimensions
        grid_depth = max(2, depth // self.grid_size)
        grid_height = max(2, height // self.grid_size)
        grid_width = max(2, width // self.grid_size)

        # Generate random displacements at control points (for x, y, z)
        control_points = torch.randn(
            3, grid_depth, grid_height, grid_width,
            device=self.device
        ) * self.strength

        # Upsample to full resolution using trilinear interpolation
        displacement_field = F.interpolate(
            control_points.unsqueeze(0),  # add batch dimension: (1, 3, grid_depth, grid_height, grid_width)
            size=(depth, height, width),
            mode='trilinear',
            align_corners=True
        ).squeeze(0)  # now (3, D, H, W)
        # Rearrange to (D, H, W, 3)
        displacement_field = displacement_field.permute(1, 2, 3, 0)
        return displacement_field

    def _apply_deformation(self, data: torch.Tensor, displacement: torch.Tensor,
                           mode: str = 'bilinear') -> torch.Tensor:
        """
        Apply the given displacement field to data using grid_sample.
        """
        # Record if input was 3D and add channel dim if needed
        orig_was_3d = False
        if data.dim() == 3:
            orig_was_3d = True
            data = data.unsqueeze(0)  # now (1, D, H, W)
        C, D, H, W = data.shape

        # Create normalized base grid (order: x, y, z)
        z = torch.linspace(-1, 1, D, device=self.device)
        y = torch.linspace(-1, 1, H, device=self.device)
        x = torch.linspace(-1, 1, W, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D, H, W, 3)

        # Scale the displacement to normalized coordinates
        scale = torch.tensor([2.0 / (W - 1), 2.0 / (H - 1), 2.0 / (D - 1)], device=self.device)
        scaled_disp = displacement * scale.view(1, 1, 3)
        sampling_grid = base_grid + scaled_disp
        sampling_grid = sampling_grid.unsqueeze(0)  # (1, D, H, W, 3)

        # Add batch dimension to data: (1, C, D, H, W)
        data = data.unsqueeze(0)
        deformed = F.grid_sample(data, sampling_grid, mode=mode, padding_mode='border', align_corners=True)
        deformed = deformed.squeeze(0)  # (C, D, H, W)
        if orig_was_3d:
            deformed = deformed.squeeze(0)
        return deformed

    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the elastic deformation transformation.
        """
        # Choose interpolation mode based on whether this is label data
        is_label = (self.label is not None and data is self.label)
        mode = 'nearest' if is_label else 'bilinear'
        # Create displacement field
        displacement = self._create_deformation_field(data.shape)
        # Apply deformation
        return self._apply_deformation(data, displacement, mode=mode)


class SplineDeformation(BaseAugmentation):
    """
    Applies thin-plate spline deformation to 3D volumes along the z-axis.

    The deformation is controlled by control points whose positions are shifted,
    and a radial basis function (thin-plate spline) is used to interpolate a dense
    deformation field.

    Interpolation:
      - Volume data: bilinear/trilinear interpolation
      - Label data: nearest neighbor interpolation
    """

    def __init__(
            self,
            volume: Optional[ArrayLike] = None,
            label: Optional[ArrayLike] = None,
            num_control_points: int = 8,
            max_displacement: float = 10.0,
            smoothness: float = 2.0,
            depth_coherence: float = 0.8,
            seed: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = 'cuda',
            **kwargs
    ):
        super().__init__(volume, label, device, **kwargs)
        self.num_control_points = num_control_points
        self.max_displacement = max_displacement
        self.smoothness = smoothness
        self.depth_coherence = depth_coherence
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _generate_control_points(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate control points and their displaced locations.

        Returns
        -------
        source_points : torch.Tensor of shape (N, 3)
        dest_points : torch.Tensor of shape (N, 3)
        """
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape

        spline_spacing = 32
        num_splines_y = max(3, height // spline_spacing)
        num_splines_x = max(3, width // spline_spacing)

        y_positions = torch.linspace(0, height - 1, num_splines_y, device=self.device)
        x_positions = torch.linspace(0, width - 1, num_splines_x, device=self.device)
        z_positions = torch.linspace(0, depth - 1, self.num_control_points, device=self.device)

        grid_y, grid_x, grid_z = torch.meshgrid(y_positions, x_positions, z_positions, indexing='ij')
        source_points = torch.stack([grid_z.flatten(), grid_y.flatten(), grid_x.flatten()], dim=1)

        displacement_vectors = torch.randn(
            num_splines_y, num_splines_x, self.num_control_points, 2,
            device=self.device
        ) * self.max_displacement

        if self.depth_coherence > 0:
            kernel_size = min(5, self.num_control_points)
            if kernel_size % 2 == 0:
                kernel_size += 1
            depth_kernel = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
            smoothed = F.conv1d(
                displacement_vectors.reshape(-1, 1, self.num_control_points),
                depth_kernel,
                padding=kernel_size // 2
            ).squeeze(1).reshape(num_splines_y, num_splines_x, self.num_control_points, 2)
            displacement_vectors = ((1 - self.depth_coherence) * displacement_vectors +
                                    self.depth_coherence * smoothed)

        displacement_flat = displacement_vectors.view(-1, 2)
        dest_points = source_points.clone()
        dest_points[:, 1:3] += displacement_flat
        dest_points[:, 0] = torch.clamp(dest_points[:, 0], 0, depth - 1)
        dest_points[:, 1] = torch.clamp(dest_points[:, 1], 0, height - 1)
        dest_points[:, 2] = torch.clamp(dest_points[:, 2], 0, width - 1)

        return source_points, dest_points

    def _compute_thin_plate_spline(
            self,
            source_points: torch.Tensor,
            dest_points: torch.Tensor,
            shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Compute a thin-plate spline deformation field.
        """
        if len(shape) == 4:
            depth, height, width = shape[1:]
        else:
            depth, height, width = shape

        source_points_norm = torch.zeros_like(source_points)
        source_points_norm[:, 0] = 2 * (source_points[:, 0] / (depth - 1)) - 1
        source_points_norm[:, 1] = 2 * (source_points[:, 1] / (height - 1)) - 1
        source_points_norm[:, 2] = 2 * (source_points[:, 2] / (width - 1)) - 1

        dest_points_norm = torch.zeros_like(dest_points)
        dest_points_norm[:, 0] = 2 * (dest_points[:, 0] / (depth - 1)) - 1
        dest_points_norm[:, 1] = 2 * (dest_points[:, 1] / (height - 1)) - 1
        dest_points_norm[:, 2] = 2 * (dest_points[:, 2] / (width - 1)) - 1

        z = torch.linspace(-1, 1, depth, device=self.device)
        y = torch.linspace(-1, 1, height, device=self.device)
        x = torch.linspace(-1, 1, width, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        grid_points = torch.stack([grid_z.flatten(), grid_y.flatten(), grid_x.flatten()], dim=1)

        n_control_points = source_points_norm.shape[0]
        n_grid_points = grid_points.shape[0]

        expanded_grid = grid_points.unsqueeze(1).expand(-1, n_control_points, -1)
        expanded_ctrl = source_points_norm.unsqueeze(0).expand(n_grid_points, -1, -1)
        squared_dist = torch.sum((expanded_grid - expanded_ctrl) ** 2, dim=2)
        rbf = squared_dist * torch.log(torch.clamp(squared_dist, min=1e-8))
        rbf = rbf / self.smoothness

        displacements = torch.zeros((n_grid_points, 3), device=self.device)
        for dim in range(3):
            control_disp = dest_points_norm[:, dim] - source_points_norm[:, dim]
            displacements[:, dim] = torch.matmul(rbf, control_disp)
        displacement_field = displacements.reshape(depth, height, width, 3)
        return displacement_field

    def _apply_deformation(self, data: torch.Tensor, displacement_field: torch.Tensor,
                           mode: str = 'bilinear') -> torch.Tensor:
        """
        Apply the spline deformation using grid_sample.
        """
        orig_was_3d = False
        if data.dim() == 3:
            orig_was_3d = True
            data = data.unsqueeze(0)  # add channel dim
        C, D, H, W = data.shape

        z = torch.linspace(-1, 1, D, device=self.device)
        y = torch.linspace(-1, 1, H, device=self.device)
        x = torch.linspace(-1, 1, W, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

        sampling_grid = base_grid + displacement_field
        sampling_grid = sampling_grid.unsqueeze(0)  # (1, D, H, W, 3)
        data = data.unsqueeze(0)  # (1, C, D, H, W)
        deformed = F.grid_sample(data, sampling_grid, mode=mode, padding_mode='border', align_corners=True)
        deformed = deformed.squeeze(0)
        if orig_was_3d:
            deformed = deformed.squeeze(0)
        return deformed

    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the spline deformation to the data.
        """
        is_label = (self.label is not None and data is self.label)
        mode = 'nearest' if is_label else 'bilinear'
        source_points, dest_points = self._generate_control_points(data.shape)
        displacement_field = self._compute_thin_plate_spline(source_points, dest_points, data.shape)
        return self._apply_deformation(data, displacement_field, mode=mode)
