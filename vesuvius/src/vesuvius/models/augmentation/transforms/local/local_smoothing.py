import torch
import numpy as np
from scipy.ndimage import gaussian_filter

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform
from vesuvius.models.augmentation.transforms.local.local_transform import LocalTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class LocalSmoothingTransform(ImageOnlyTransform, LocalTransform):
    """
    Applies localized Gaussian smoothing using a spatial weighting mask.

    Creates a blurred version of the image and blends it with the original
    using a spatially-varying kernel. This simulates localized blur, defocus,
    or smoothing artifacts.

    Parameters:
        scale (RandomScalar): Gaussian spread for the spatial weighting mask.
        loc (RandomScalar): Relative center position for the mask (in % of image size).
        smoothing_strength (RandomScalar): Max weight of smoothed image in blend [0, 1].
        kernel_size (RandomScalar): Sigma for the actual Gaussian smoothing.
        same_for_all_channels (bool): Whether to use one kernel for all channels.
        p_per_channel (float): Probability to apply to each channel independently.
    """

    def __init__(
        self,
        scale: RandomScalar,
        loc: RandomScalar = (-1, 2),
        smoothing_strength: RandomScalar = (0.5, 1.0),
        kernel_size: RandomScalar = (1.0, 3.0),
        same_for_all_channels: bool = True,
        p_per_channel: float = 1.0,
    ):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.smoothing_strength = smoothing_strength
        self.kernel_size = kernel_size
        self.same_for_all_channels = same_for_all_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if not any(apply_channel):
            return {'kernels': [None] * C, 'sigmas': [None] * C}

        if self.same_for_all_channels:
            kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)
            strength = sample_scalar(self.smoothing_strength)
            kernel = kernel * strength  # Scale kernel by smoothing strength
            sigma = sample_scalar(self.kernel_size)

            kernels = [kernel if apply else None for apply in apply_channel]
            sigmas = [sigma if apply else None for apply in apply_channel]
        else:
            kernels, sigmas = [], []
            for apply in apply_channel:
                if not apply:
                    kernels.append(None)
                    sigmas.append(None)
                    continue

                kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)
                strength = sample_scalar(self.smoothing_strength)
                kernel = kernel * strength
                sigma = sample_scalar(self.kernel_size)

                kernels.append(kernel)
                sigmas.append(sigma)

        return {'kernels': kernels, 'sigmas': sigmas}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        kernels = params['kernels']
        sigmas = params['sigmas']

        if all(k is None for k in kernels):
            return img

        device = img.device
        dtype = img.dtype
        img_np = img.cpu().numpy()

        for c, (kernel, sigma) in enumerate(zip(kernels, sigmas)):
            if kernel is None:
                continue

            channel = img_np[c]
            # Apply Gaussian smoothing
            smoothed = gaussian_filter(channel, sigma=sigma)
            # Blend using the spatial kernel
            img_np[c] = self.run_interpolation(channel, smoothed, kernel)

        return torch.from_numpy(img_np).to(device=device, dtype=dtype)
