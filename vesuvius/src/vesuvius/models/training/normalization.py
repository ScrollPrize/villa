from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number

from vesuvius.image_proc.intensity.normalization import (
    DEFAULT_TARGET_DTYPE,
    normalize_ct,
    normalize_minmax,
    normalize_robust,
    normalize_zscore,
)


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
