import torch
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform
from vesuvius.models.augmentation.transforms.local.local_transform import LocalTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class BrightnessGradientAdditiveTransform(ImageOnlyTransform, LocalTransform):
    """
    Applies localized brightness modulation using a smooth Gaussian gradient.

    Adds a spatially-varying brightness offset to simulate intensity drift,
    local lighting artifacts, or intensity gradients across the image.

    Parameters:
        scale (RandomScalar): Gaussian std deviation spread (controls kernel width).
        loc (RandomScalar): Relative center position as % of image size.
                           Default (-1, 2) allows centers outside image bounds.
        max_strength (RandomScalar): Peak additive intensity change.
        same_for_all_channels (bool): If True, use one kernel for all channels.
        mean_centered (bool): If True, subtract kernel mean to preserve overall intensity.
        clip_intensities (bool): If True, clamp output to original min/max range.
        p_per_channel (float): Probability to apply to each channel independently.
    """

    def __init__(
        self,
        scale: RandomScalar,
        loc: RandomScalar = (-1, 2),
        max_strength: RandomScalar = (-0.5, 0.5),
        same_for_all_channels: bool = True,
        mean_centered: bool = True,
        clip_intensities: bool = False,
        p_per_channel: float = 1.0,
    ):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.max_strength = max_strength
        self.same_for_all_channels = same_for_all_channels
        self.mean_centered = mean_centered
        self.clip_intensities = clip_intensities
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if not any(apply_channel):
            return {'kernels': [None] * C, 'strengths': [None] * C}

        if self.same_for_all_channels:
            kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)

            # Mean-center if requested
            if self.mean_centered:
                kernel = kernel - kernel.mean()
                max_abs = np.abs(kernel).max()
                if max_abs > 1e-8:
                    kernel = kernel / max_abs

            strength = sample_scalar(self.max_strength)
            kernels = [kernel if apply else None for apply in apply_channel]
            strengths = [strength if apply else None for apply in apply_channel]
        else:
            kernels, strengths = [], []
            for apply in apply_channel:
                if not apply:
                    kernels.append(None)
                    strengths.append(None)
                    continue

                kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)
                if self.mean_centered:
                    kernel = kernel - kernel.mean()
                    max_abs = np.abs(kernel).max()
                    if max_abs > 1e-8:
                        kernel = kernel / max_abs

                strength = sample_scalar(self.max_strength)
                kernels.append(kernel)
                strengths.append(strength)

        return {'kernels': kernels, 'strengths': strengths}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        kernels = params['kernels']
        strengths = params['strengths']

        if all(k is None for k in kernels):
            return img

        device = img.device
        dtype = img.dtype

        for c, (kernel, strength) in enumerate(zip(kernels, strengths)):
            if kernel is None:
                continue

            kernel_tensor = torch.from_numpy(kernel).to(device=device, dtype=dtype)
            offset = kernel_tensor * strength

            if self.clip_intensities:
                orig_min = img[c].min()
                orig_max = img[c].max()
                img[c] = (img[c] + offset).clamp(min=orig_min, max=orig_max)
            else:
                img[c] = img[c] + offset

        return img
