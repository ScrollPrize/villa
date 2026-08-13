import torch
import torch.nn.functional as F
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class SharpeningTransform(ImageOnlyTransform):
    """
    Applies sharpening using Laplacian-based contrast enhancement.

    Uses a Laplacian kernel to detect edges and adds them back to the original
    image with configurable strength. Supports both 2D and 3D data.

    Formula: sharpened = original + strength * laplacian

    Parameters:
        strength (RandomScalar): Sharpening intensity.
        p_same_for_each_channel (float): Probability all channels get same strength.
        p_per_channel (float): Probability to apply to each channel independently.
        p_clamp_intensities (float): Probability to clamp output to original range.
    """

    def __init__(
        self,
        strength: RandomScalar = (0.1, 0.5),
        p_same_for_each_channel: float = 0.5,
        p_per_channel: float = 1.0,
        p_clamp_intensities: float = 0.5,
    ):
        super().__init__()
        self.strength = strength
        self.p_same_for_each_channel = p_same_for_each_channel
        self.p_per_channel = p_per_channel
        self.p_clamp_intensities = p_clamp_intensities

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        ndim = image.ndim - 1  # Spatial dimensions

        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if not any(apply_channel):
            return {
                'strengths': [None] * C,
                'clamp_flags': [False] * C,
                'ndim': ndim
            }

        use_same = np.random.rand() < self.p_same_for_each_channel

        if use_same:
            shared_strength = sample_scalar(self.strength)
            strengths = [shared_strength if apply else None for apply in apply_channel]
        else:
            strengths = [
                sample_scalar(self.strength) if apply else None
                for apply in apply_channel
            ]

        clamp_flags = [
            np.random.rand() < self.p_clamp_intensities if apply else False
            for apply in apply_channel
        ]

        return {
            'strengths': strengths,
            'clamp_flags': clamp_flags,
            'ndim': ndim
        }

    def _get_laplacian_kernel(self, ndim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create a Laplacian kernel for edge detection."""
        if ndim == 2:
            # 3x3 Laplacian kernel for 2D
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], device=device, dtype=dtype)
            # Shape: (1, 1, 3, 3) for conv2d
            return kernel.view(1, 1, 3, 3)
        else:
            # 3x3x3 Laplacian kernel for 3D (6-connected)
            kernel = torch.zeros(3, 3, 3, device=device, dtype=dtype)
            kernel[1, 1, 1] = 6  # Center
            kernel[0, 1, 1] = -1  # Face neighbors
            kernel[2, 1, 1] = -1
            kernel[1, 0, 1] = -1
            kernel[1, 2, 1] = -1
            kernel[1, 1, 0] = -1
            kernel[1, 1, 2] = -1
            # Shape: (1, 1, 3, 3, 3) for conv3d
            return kernel.view(1, 1, 3, 3, 3)

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        strengths = params['strengths']
        clamp_flags = params['clamp_flags']
        ndim = params['ndim']

        if all(s is None for s in strengths):
            return img

        device = img.device
        dtype = img.dtype
        kernel = self._get_laplacian_kernel(ndim, device, dtype)

        for c, (strength, clamp) in enumerate(zip(strengths, clamp_flags)):
            if strength is None:
                continue

            # Get original range for potential clamping
            if clamp:
                orig_min = img[c].min()
                orig_max = img[c].max()

            # Add batch and channel dims for convolution: (1, 1, *spatial)
            channel = img[c].unsqueeze(0).unsqueeze(0)

            # Apply padding and convolution
            if ndim == 2:
                padded = F.pad(channel, (1, 1, 1, 1), mode='replicate')
                laplacian = F.conv2d(padded, kernel)
            else:
                padded = F.pad(channel, (1, 1, 1, 1, 1, 1), mode='replicate')
                laplacian = F.conv3d(padded, kernel)

            # Apply sharpening: original + strength * laplacian
            sharpened = channel + strength * laplacian
            sharpened = sharpened.squeeze(0).squeeze(0)

            # Optionally clamp to original range
            if clamp:
                sharpened = sharpened.clamp(min=orig_min, max=orig_max)

            img[c] = sharpened

        return img
