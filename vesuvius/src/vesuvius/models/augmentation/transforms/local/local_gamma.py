import torch
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform
from vesuvius.models.augmentation.transforms.local.local_transform import LocalTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class LocalGammaTransform(ImageOnlyTransform, LocalTransform):
    """
    Applies locally varying gamma correction using a spatial Gaussian weighting mask.

    A Gaussian kernel is randomly placed in the image and used to blend between
    the original image and a gamma-corrected version. This simulates localized
    nonlinear intensity shifts, useful for data augmentation in medical imaging.

    Parameters:
        scale (RandomScalar): Controls the width of the Gaussian (std dev).
                             Recommend large values (e.g., 10-30).
        loc (RandomScalar): Controls Gaussian center as % of image size.
                           E.g., (-1, 2) allows off-canvas kernels.
        gamma (RandomScalar): The gamma exponent applied locally.
        same_for_all_channels (bool): If True, one kernel is reused across all channels.
        p_per_channel (float): Probability to apply gamma correction per channel.
    """

    def __init__(
        self,
        scale: RandomScalar,
        loc: RandomScalar = (-1, 2),
        gamma: RandomScalar = (0.5, 1.5),
        same_for_all_channels: bool = True,
        p_per_channel: float = 1.0,
    ):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.gamma = gamma
        self.same_for_all_channels = same_for_all_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if not any(apply_channel):
            return {'kernels': [None] * C, 'gammas': [None] * C}

        if self.same_for_all_channels:
            kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)
            gamma = sample_scalar(self.gamma)

            kernels = [kernel if apply else None for apply in apply_channel]
            gammas = [gamma if apply else None for apply in apply_channel]
        else:
            kernels, gammas = [], []
            for apply in apply_channel:
                if not apply:
                    kernels.append(None)
                    gammas.append(None)
                    continue

                kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)
                gamma = sample_scalar(self.gamma)
                kernels.append(kernel)
                gammas.append(gamma)

        return {'kernels': kernels, 'gammas': gammas}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        kernels = params['kernels']
        gammas = params['gammas']

        if all(k is None for k in kernels):
            return img

        device = img.device
        dtype = img.dtype
        img_np = img.cpu().numpy()

        for c, (kernel, gamma) in enumerate(zip(kernels, gammas)):
            if kernel is None:
                continue

            channel = img_np[c]
            min_val, max_val = channel.min(), channel.max()
            denom = max(max_val - min_val, 1e-8)

            # Normalize to [0, 1]
            norm = (channel - min_val) / denom
            # Apply gamma correction
            gamma_corrected = np.power(norm, gamma)
            # Blend using the spatial kernel
            blended = self.run_interpolation(norm, gamma_corrected, kernel)
            # Rescale to original range
            img_np[c] = blended * denom + min_val

        return torch.from_numpy(img_np).to(device=device, dtype=dtype)
