from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    """
    Abstract base class for image normalization strategies.
    """
    
    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
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
        image = image.astype(self.target_dtype, copy=False)
        
        if self.use_mask_for_norm is not None and self.use_mask_for_norm and mask is not None:
            # Normalize only within the mask region
            mask_bool = mask > 0
            mean = image[mask_bool].mean()
            std = image[mask_bool].std()
            image[mask_bool] = (image[mask_bool] - mean) / max(std, 1e-8)
        else:
            # Normalize the entire image
            mean = image.mean()
            std = image.std()
            image = (image - mean) / max(std, 1e-8)
            
        return image


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
        
        image = image.astype(self.target_dtype, copy=False)
        
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        
        # Clip to percentile bounds
        np.clip(image, lower_bound, upper_bound, out=image)
        
        # Normalize using global mean and std
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        
        return image


class RescaleTo01Normalization(ImageNormalization):
    """
    Min-max normalization to [0, 1] range.
    """
    
    def run(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply min-max normalization to [0, 1] range.
        """
        image = image.astype(self.target_dtype, copy=False)
        
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            # If all values are the same, return zeros
            image = np.zeros_like(image, dtype=self.target_dtype)
            
        return image


# Mapping from string names to normalization classes
NORMALIZATION_SCHEMES = {
    'zscore': ZScoreNormalization,
    'ct': CTNormalization,
    'rescale_to_01': RescaleTo01Normalization,
    'minmax': RescaleTo01Normalization,  # Alias
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
