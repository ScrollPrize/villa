from __future__ import annotations

from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def _create_gaussian_kernel_1d(sigma: float, device_str: str) -> tuple[torch.Tensor, int]:
    """Create and cache 1D Gaussian kernel."""
    device = torch.device(device_str)
    size = int(2 * np.ceil(3 * sigma) + 1)
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d, size


def create_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """Create 2D Gaussian kernel for convolution."""
    kernel_1d, size = _create_gaussian_kernel_1d(sigma, str(device))
    gauss_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return gauss_2d[None, None, :, :]


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur using separable convolution for efficiency."""
    if sigma <= 0:
        return img

    # Get cached kernel
    kernel_1d, size = _create_gaussian_kernel_1d(sigma, str(img.device))

    # For batch processing, we need to handle each image in the batch
    # Reshape input from (B, C, H, W) to (B*C, 1, H, W) for grouped convolution
    B, C, H, W = img.shape
    img_reshaped = img.reshape(B * C, 1, H, W)

    # Create kernels for horizontal and vertical passes
    kernel_1d_h = kernel_1d.view(1, 1, -1, 1)
    kernel_1d_v = kernel_1d.view(1, 1, 1, -1)

    # Apply horizontal convolution
    img_reshaped = F.conv2d(img_reshaped, kernel_1d_h, padding=(size // 2, 0))

    # Apply vertical convolution
    img_reshaped = F.conv2d(img_reshaped, kernel_1d_v, padding=(0, size // 2))

    # Reshape back to original batch shape
    img = img_reshaped.reshape(B, C, H, W)

    return img
