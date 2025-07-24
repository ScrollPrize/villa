import numpy as np

def convert_to_uint8_dtype_range(img):
    """
    Convert image data to uint8 based on the input data type range.

    Parameters
    ----------
    img : numpy.ndarray
        Input image data

    Returns
    -------
    numpy.ndarray
        Image converted to uint8 with proper scaling
    """
    # Handle special float values
    if img.dtype in [np.float32, np.float64]:
        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)

    # Convert based on data type
    if img.dtype == np.uint8:
        # Already uint8, no conversion needed
        return img
    elif img.dtype == np.uint16:
        # Scale from 0-65535 to 0-255
        return (img / 256).astype(np.uint8)  # Equivalent to img >> 8
    elif img.dtype == np.int16:
        # Scale from -32768 to 32767 to 0-255
        # First shift to 0-65535 range, then scale to 0-255
        return ((img.astype(np.int32) + 32768) / 256).astype(np.uint8)
    elif img.dtype == np.uint32:
        # Scale from 0-4294967295 to 0-255
        return (img / 16777216).astype(np.uint8)  # Equivalent to img >> 24
    elif img.dtype == np.int32:
        # Scale from -2147483648 to 2147483647 to 0-255
        return ((img.astype(np.int64) + 2147483648) / 16777216).astype(np.uint8)
    else:
        # For float or other types, use min-max scaling
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > min_val:
            return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            # Constant image
            return np.zeros_like(img, dtype=np.uint8)