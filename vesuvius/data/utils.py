import tensorstore as ts
import numpy as np
from typing import Union

# Function to get the maximum value of a dtype
def get_max_value(dtype: np.dtype) -> Union[float, int]:
    """
    Get the maximum value for a given NumPy dtype.

    Parameters:
    ----------
    dtype : np.dtype
        The NumPy data type to evaluate.

    Returns:
    -------
    Union[float, int]
        The maximum value that the dtype can hold.

    Raises:
    ------
    ValueError
        If the dtype is not a floating point or integer.
    """

    if np.issubdtype(dtype, np.floating):
        max_value = np.finfo(dtype).max
    elif np.issubdtype(dtype, np.integer):
        max_value = np.iinfo(dtype).max
    else:
        raise ValueError("Unsupported dtype")
    return max_value