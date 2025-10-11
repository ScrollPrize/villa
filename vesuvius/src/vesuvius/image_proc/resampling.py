import numpy as np
from scipy.ndimage import zoom

def upsample_array(
        array : np.ndarray,
        upsample_factor: float,
        fill_value=0
) -> np.ndarray:

    return zoom(array, upsample_factor, cval=fill_value)


