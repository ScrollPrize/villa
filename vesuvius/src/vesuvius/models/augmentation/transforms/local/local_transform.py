import numpy as np
import scipy.stats as st
from abc import ABC
from typing import Tuple

from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class LocalTransform(ABC):
    """
    Base mixin class for spatially-varying transforms.

    Provides methods for generating spatial Gaussian kernels that can be used
    to apply localized modifications to images. The kernels are generated using
    CDF-based Gaussian distributions for smooth spatial weighting.

    Parameters:
        scale (RandomScalar): Controls the width (std dev) of the Gaussian kernel.
        loc (RandomScalar): Controls the center position as a fraction of image size.
                           Default (-1, 2) allows centers outside the image bounds.
    """

    def __init__(self, scale: RandomScalar, loc: RandomScalar = (-1, 2)):
        self.loc = loc
        self.scale = scale

    def _generate_kernel(self, img_shp: Tuple[int, ...]) -> np.ndarray:
        """
        Generate a spatial Gaussian kernel for the given image shape.

        Uses CDF-based integration over each spatial dimension to create smooth,
        properly normalized Gaussian distributions.

        Parameters:
            img_shp: Spatial shape of the image (e.g., (H, W) or (D, H, W))

        Returns:
            Kernel array normalized to [0, 1] range with same spatial shape as input
        """
        ndim = len(img_shp)
        x_grids = [np.arange(-0.5, s + 0.5, dtype=np.float32) for s in img_shp]
        kernels = []

        for d in range(ndim):
            loc_val = sample_scalar(self.loc, img_shp, d)
            scale_val = sample_scalar(self.scale, img_shp, d)
            loc_rescaled = loc_val * img_shp[d]
            cdf = st.norm.cdf(x_grids[d], loc=loc_rescaled, scale=scale_val)
            kernels.append(np.diff(cdf).astype(np.float32))

        # Build N-dimensional kernel via outer products
        kernel = kernels[0][:, None] @ kernels[1][None]
        if ndim == 3:
            kernel = kernel[:, :, None] @ kernels[2][None]

        # Normalize to [0, 1]
        kernel -= kernel.min()
        kernel_max = kernel.max()
        if kernel_max > 0:
            kernel /= kernel_max
        return kernel

    def _generate_multiple_kernel_image(self, img_shp: Tuple[int, ...], num_kernels: int) -> np.ndarray:
        """
        Generate multiple Gaussian kernels and sum them.

        Useful for creating more complex spatial patterns with multiple hotspots.

        Parameters:
            img_shp: Spatial shape of the image
            num_kernels: Number of kernels to generate and sum

        Returns:
            Combined kernel normalized to [0, 1] range
        """
        kernel_image = np.zeros(img_shp, dtype=np.float32)
        for _ in range(num_kernels):
            kernel_image += self._generate_kernel(img_shp)

        kernel_image -= kernel_image.min()
        kernel_max = kernel_image.max()
        if kernel_max > 0:
            kernel_image /= kernel_max
        return kernel_image

    @staticmethod
    def invert_kernel(kernel_image: np.ndarray) -> np.ndarray:
        """
        Invert a normalized kernel: 1 - kernel

        Parameters:
            kernel_image: Input kernel in [0, 1] range

        Returns:
            Inverted kernel in [0, 1] range
        """
        return 1.0 - kernel_image

    @staticmethod
    def run_interpolation(original_image: np.ndarray,
                          modified_image: np.ndarray,
                          kernel_image: np.ndarray) -> np.ndarray:
        """
        Blend original and modified images using kernel as per-pixel weight map.

        Parameters:
            original_image: Unmodified input image
            modified_image: Modified version (e.g., gamma-corrected, blurred)
            kernel_image: Kernel in [0, 1], where 0 = keep original, 1 = keep modified

        Returns:
            Blended result: original * (1 - kernel) + modified * kernel
        """
        return original_image * (1.0 - kernel_image) + modified_image * kernel_image
