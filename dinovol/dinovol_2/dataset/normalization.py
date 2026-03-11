from abc import ABC, abstractmethod
from typing import Mapping, Tuple, Type

import numpy as np
from numpy import number

DEFAULT_TARGET_DTYPE = np.float32
EPS = 1e-8

__all__ = [
    "normalize_minmax",
    "normalize_zscore",
    "normalize_ct",
    "normalize_robust",
    "DEFAULT_TARGET_DTYPE",
    "NORMALIZATION_FUNCTIONS",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _prepare_image(image: np.ndarray, target_dtype: np.dtype | type = DEFAULT_TARGET_DTYPE) -> np.ndarray:
    arr = np.asarray(image)
    if target_dtype is not None:
        arr = arr.astype(target_dtype, copy=False)
    return arr


def _prepare_mask(mask: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Broadcast the provided mask to match the image shape, ensuring boolean dtype.
    Raises ValueError when broadcasting is impossible.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim > len(image_shape):
        raise ValueError(
            f"Mask with shape {mask_arr.shape} has higher dimensionality than image shape {image_shape}"
        )

    if mask_arr.ndim < len(image_shape):
        expand_dims = (1,) * (len(image_shape) - mask_arr.ndim)
        mask_arr = mask_arr.reshape(expand_dims + mask_arr.shape)

    try:
        broadcast = np.broadcast_to(mask_arr, image_shape, subok=True)
    except ValueError as exc:
        raise ValueError(
            f"Mask with shape {mask_arr.shape} cannot broadcast to image shape {image_shape}"
        ) from exc

    return broadcast.astype(bool, copy=False)


def _select_valid_region(
    image: np.ndarray,
    mask: np.ndarray | None,
    use_mask: bool,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Return the flattened view of valid pixels and the boolean mask used.

    When no valid mask pixels exist, the full image is returned and mask_bool is None.
    """
    if use_mask and mask is not None:
        mask_bool = _prepare_mask(mask, image.shape)
        if np.any(mask_bool):
            return image[mask_bool], mask_bool
    return image.reshape(-1), None


# --------------------------------------------------------------------------- #
# Normalisation functions
# --------------------------------------------------------------------------- #

def normalize_minmax(
    image: np.ndarray,
    *,
    target_dtype: np.dtype | type = DEFAULT_TARGET_DTYPE,
) -> np.ndarray:
    """
    Min-max normalisation that rescales intensities into the [0, 1] range.
    """
    arr = _prepare_image(image, target_dtype)
    min_val = float(arr.min())
    max_val = float(arr.max())

    if max_val > min_val:
        arr -= min_val
        arr /= max(max_val - min_val, EPS)
    else:
        arr.fill(0.0)

    return arr


def normalize_zscore(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    use_mask: bool = False,
    target_dtype: np.dtype | type = DEFAULT_TARGET_DTYPE,
) -> np.ndarray:
    """
    Standard score normalisation (z-score).

    Parameters
    ----------
    image : np.ndarray
        Input image.
    mask : np.ndarray, optional
        Optional mask restricting the statistics to masked pixels.
    use_mask : bool
        When true and mask is provided, restrict mean/std computation to masked region.
    """
    arr = _prepare_image(image, target_dtype)
    valid, mask_bool = _select_valid_region(arr, mask, use_mask)

    mean = float(valid.mean()) if valid.size else 0.0
    std = float(valid.std()) if valid.size else 0.0
    std = max(std, EPS)

    if mask_bool is not None:
        arr_masked = arr[mask_bool]
        arr_masked -= mean
        arr_masked /= std
        arr[mask_bool] = arr_masked
    else:
        arr -= mean
        arr /= std

    return arr


def normalize_ct(
    image: np.ndarray,
    *,
    intensity_properties: Mapping[str, float] | None,
    target_dtype: np.dtype | type = DEFAULT_TARGET_DTYPE,
) -> np.ndarray:
    """
    CT-style normalisation using global percentiles, mean, and standard deviation.
    """
    if intensity_properties is None:
        raise ValueError("normalize_ct requires intensity_properties")

    required_keys = ("mean", "std", "percentile_00_5", "percentile_99_5")
    missing = [key for key in required_keys if key not in intensity_properties]
    if missing:
        raise ValueError(
            f"normalize_ct missing intensity properties: {missing}. "
            f"Provided keys: {list(intensity_properties.keys())}"
        )

    arr = _prepare_image(image, target_dtype)
    mean_intensity = float(intensity_properties["mean"])
    std_intensity = max(float(intensity_properties["std"]), EPS)
    lower_bound = float(intensity_properties["percentile_00_5"])
    upper_bound = float(intensity_properties["percentile_99_5"])

    np.clip(arr, lower_bound, upper_bound, out=arr)
    arr -= mean_intensity
    arr /= std_intensity

    return arr


def normalize_robust(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    use_mask: bool = False,
    percentile_lower: float = 1.0,
    percentile_upper: float = 99.0,
    clip_values: bool = True,
    target_dtype: np.dtype | type = DEFAULT_TARGET_DTYPE,
) -> np.ndarray:
    """
    Robust normalisation based on the median and MAD (Median Absolute Deviation).
    """
    arr = _prepare_image(image, target_dtype)
    valid, mask_bool = _select_valid_region(arr, mask, use_mask)

    if valid.size == 0:
        return arr

    std_preclip = float(np.std(valid))
    min_preclip = float(np.min(valid))
    max_preclip = float(np.max(valid))

    lower_val = upper_val = None
    if clip_values and valid.size > 0:
        lower_val = float(np.percentile(valid, percentile_lower))
        upper_val = float(np.percentile(valid, percentile_upper))
        np.clip(arr, lower_val, upper_val, out=arr)
        valid, mask_bool = _select_valid_region(arr, mask, use_mask)
        if valid.size == 0:
            return arr

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    scaled_mad = 1.4826 * mad

    if not np.isfinite(scaled_mad) or scaled_mad < 1e-6:
        percentile_span = None
        if lower_val is not None and upper_val is not None:
            percentile_span = upper_val - lower_val
        else:
            percentile_span = max_preclip - min_preclip

        fallback_values = []
        if np.isfinite(std_preclip):
            fallback_values.append(abs(std_preclip))
        if percentile_span is not None and np.isfinite(percentile_span):
            fallback_values.append(abs(percentile_span) / 2.0)

        scaled_mad = next(
            (candidate for candidate in fallback_values if candidate >= 1e-6),
            None,
        )

        if scaled_mad is None or not np.isfinite(scaled_mad):
            scaled_mad = 1.0

    if mask_bool is not None:
        arr_masked = arr[mask_bool]
        arr_masked -= median
        arr_masked /= scaled_mad
        arr[mask_bool] = arr_masked
    else:
        arr -= median
        arr /= scaled_mad

    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _normalize_identity(
    image: np.ndarray,
    *,
    target_dtype: np.dtype | type = DEFAULT_TARGET_DTYPE,
    **_: object,
) -> np.ndarray:
    """No-op normalisation helper."""
    arr = _prepare_image(image, target_dtype)
    return arr


NORMALIZATION_FUNCTIONS = {
    "zscore": normalize_zscore,
    "ct": normalize_ct,
    "rescale_to_01": normalize_minmax,
    "minmax": normalize_minmax,
    "robust": normalize_robust,
    "none": _normalize_identity,
}

class ImageNormalization(ABC):
    """
    Abstract base class for image normalization strategies.
    """
    
    def __init__(
        self,
        use_mask_for_norm: bool | None = None,
        intensityproperties: dict | None = None,
        target_dtype: Type[number] = DEFAULT_TARGET_DTYPE,
    ):
        """
        Initialize the normalization.
        
        Parameters
        ----------
        use_mask_for_norm : bool, optional
            Whether to use mask for normalization (not currently used in BaseDataset)
        intensityproperties : dict, optional
            Intensity properties for certain normalization schemes (e.g., CTNormalization)
        target_dtype : Type[number]
            Target data type for the normalized output
        """
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        self.intensityproperties = intensityproperties or {}
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply normalization to the image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image to normalize
        mask : np.ndarray, optional
            Mask for selective normalization (not currently used in BaseDataset)
            
        Returns
        -------
        np.ndarray
            Normalized image
        """
        pass


class ZScoreNormalization(ImageNormalization):
    """
    Z-score normalization: (x - mean) / std
    """
    
    def run(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply z-score normalization.
        """
        use_mask = bool(self.use_mask_for_norm)
        return normalize_zscore(
            image,
            mask=mask,
            use_mask=use_mask,
            target_dtype=self.target_dtype,
        )


class CTNormalization(ImageNormalization):
    """
    CT-style normalization: clip to percentiles and normalize.
    Requires intensity properties with 'mean', 'std', 'percentile_00_5', and 'percentile_99_5'.
    """
    
    def run(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply CT normalization.
        """
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        assert all(k in self.intensityproperties for k in ['mean', 'std', 'percentile_00_5', 'percentile_99_5']), \
            "CTNormalization requires 'mean', 'std', 'percentile_00_5', and 'percentile_99_5' in intensity properties"
        
        return normalize_ct(
            image,
            intensity_properties=self.intensityproperties,
            target_dtype=self.target_dtype,
        )


class RescaleTo01Normalization(ImageNormalization):
    """
    Min-max normalization to [0, 1] range.
    """
    
    def run(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply min-max normalization to [0, 1] range.
        """
        return normalize_minmax(
            image,
            target_dtype=self.target_dtype,
        )


class RobustNormalization(ImageNormalization):
    """
    Robust normalization using median and MAD (Median Absolute Deviation).
    More resistant to outliers than standard z-score normalization.
    """
    
    def __init__(self, percentile_lower: float = 1.0, percentile_upper: float = 99.0, 
                 clip_values: bool = True, **kwargs):
        """
        Initialize robust normalization.
        
        Parameters
        ----------
        percentile_lower : float
            Lower percentile for clipping (default: 1.0)
        percentile_upper : float
            Upper percentile for clipping (default: 99.0)
        clip_values : bool
            Whether to clip values to percentile range before normalization
        """
        super().__init__(**kwargs)
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.clip_values = clip_values
    
    def run(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply robust normalization using median and MAD.
        """
        use_mask = bool(self.use_mask_for_norm)
        return normalize_robust(
            image,
            mask=mask,
            use_mask=use_mask,
            percentile_lower=self.percentile_lower,
            percentile_upper=self.percentile_upper,
            clip_values=self.clip_values,
            target_dtype=self.target_dtype,
        )


# Mapping from string names to normalization classes
NORMALIZATION_SCHEMES = {
    'zscore': ZScoreNormalization,
    'ct': CTNormalization,
    'rescale_to_01': RescaleTo01Normalization,
    'minmax': RescaleTo01Normalization,  # Alias
    'robust': RobustNormalization,
    'none': None  # No normalization
}


def get_normalization(scheme: str, intensityproperties: dict = None) -> ImageNormalization:
    """
    Factory function to get a normalization instance by name.
    
    Parameters
    ----------
    scheme : str
        Name of the normalization scheme ('zscore', 'ct', 'rescale_to_01', 'minmax', 'none')
    intensityproperties : dict, optional
        Intensity properties for schemes that need them (e.g., CT normalization)
        
    Returns
    -------
    ImageNormalization or None
        Normalization instance or None if scheme is 'none'
    """
    if scheme not in NORMALIZATION_SCHEMES:
        raise ValueError(f"Unknown normalization scheme: {scheme}. "
                        f"Available schemes: {list(NORMALIZATION_SCHEMES.keys())}")
    
    norm_class = NORMALIZATION_SCHEMES[scheme]
    if norm_class is None:
        return None
        
    return norm_class(intensityproperties=intensityproperties)
