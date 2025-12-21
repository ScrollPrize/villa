import torch
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class CutOffOutliersTransform(ImageOnlyTransform):
    """
    Clamps intensities to percentile bounds to remove outliers.

    Optionally rescales the result to retain the original standard deviation.
    Useful for robust preprocessing or as an augmentation to simulate
    clipped/normalized data.

    Parameters:
        percentile_lower (RandomScalar): Lower cutoff percentile (0-100).
        percentile_upper (RandomScalar): Upper cutoff percentile (0-100).
        p_synchronize_channels (bool): If True, same percentiles for all channels.
        p_per_channel (float): Probability to apply cutoff to each channel.
        p_retain_std (float): Probability of retaining original std after clipping.
    """

    def __init__(
        self,
        percentile_lower: RandomScalar = 0.2,
        percentile_upper: RandomScalar = 99.8,
        p_synchronize_channels: bool = False,
        p_per_channel: float = 1.0,
        p_retain_std: float = 1.0,
    ):
        super().__init__()
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.p_synchronize_channels = p_synchronize_channels
        self.p_per_channel = p_per_channel
        self.p_retain_std = p_retain_std

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if self.p_synchronize_channels:
            lower = float(sample_scalar(self.percentile_lower))
            upper = float(sample_scalar(self.percentile_upper))
            percentiles = [(lower, upper) if apply else None for apply in apply_channel]
        else:
            percentiles = []
            for apply in apply_channel:
                if not apply:
                    percentiles.append(None)
                else:
                    lower = float(sample_scalar(self.percentile_lower))
                    upper = float(sample_scalar(self.percentile_upper))
                    percentiles.append((lower, upper))

        retain_std_flags = [
            np.random.rand() < self.p_retain_std if p is not None else False
            for p in percentiles
        ]

        return {'percentiles': percentiles, 'retain_std': retain_std_flags}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        percentiles = params['percentiles']
        retain_std = params['retain_std']

        if all(p is None for p in percentiles):
            return img

        for c, perc in enumerate(percentiles):
            if perc is None:
                continue

            img_c = img[c]
            if retain_std[c]:
                orig_std = img_c.std()

            # Calculate percentiles (convert to numpy for percentile calculation)
            img_c_np = img_c.detach().cpu().numpy()
            lower_val = np.percentile(img_c_np, perc[0])
            upper_val = np.percentile(img_c_np, perc[1])

            # Clamp the values
            img_c_clipped = img_c.clamp(min=float(lower_val), max=float(upper_val))

            # Optionally restore original std
            if retain_std[c]:
                clipped_std = img_c_clipped.std()
                if clipped_std > 1e-8:
                    clipped_mean = img_c_clipped.mean()
                    img_c_clipped = (img_c_clipped - clipped_mean) / clipped_std * orig_std + clipped_mean

            img[c] = img_c_clipped

        return img
