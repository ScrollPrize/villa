
import numpy as np
import zarr
import os
from typing import Union, Dict, Any, Optional, Tuple

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

def open_zarr(path: str, mode: str = 'r', 
              storage_options: Optional[Dict[str, Any]] = None,
              verbose: bool = False,
              # Additional zarr creation parameters
              shape: Optional[Tuple] = None,
              chunks: Optional[Tuple] = None,
              dtype: Any = None,
              compressor: Any = None,
              fill_value: Any = None,
              order: str = None,
              **kwargs) -> zarr.Array:
    """
    Open a zarr array with consistent handling of local and remote URLs.
    
    Parameters:
    ----------
    path : str
        Path to the zarr array. Can be a local path, HTTP URL, or S3 URL.
    mode : str, default 'r'
        Mode to open the zarr array ('r' for read-only, 'r+' for read-write, 'w' for write).
    storage_options : Optional[Dict[str, Any]], default None
        Additional options for storage backend. For S3, {'anon': False} will be added by default.
    verbose : bool, default False
        Whether to print verbose information about opening the zarr array.
    shape, chunks, dtype, compressor, fill_value, order : zarr creation parameters
        Only used when mode is 'w' to create a new zarr array.
    **kwargs : Additional parameters passed to zarr.open
        
    Returns:
    -------
    zarr.Array
        The opened zarr array
    """
    if storage_options is None:
        storage_options = {}
    
    # Ensure parent directory exists for write modes and local paths
    if mode in ('w', 'w-', 'a') and not path.startswith(('http://', 'https://', 's3://')):
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            if verbose:
                print(f"Created local directory: {parent_dir}")
    
    # Handle S3 URLs
    if path.startswith('s3://'):
        # Always use AWS credentials for S3 URLs
        if 'anon' not in storage_options:
            storage_options['anon'] = False
        
        if verbose:
            print(f"Opening S3 zarr store at {path} with storage_options: {storage_options}")
        
        # Create parent directory for S3 URLs in write mode
        if mode in ('w', 'w-', 'a'):
            s3_parts = path.replace('s3://', '').split('/')
            parent_path = 's3://' + '/'.join(s3_parts[:-1])
            if parent_path != 's3://':
                import fsspec
                fs = fsspec.filesystem('s3', **storage_options)
                fs.makedirs(parent_path, exist_ok=True)
                if verbose:
                    print(f"Created S3 directory: {parent_path}")
    
    # Handle HTTP/HTTPS URLs
    elif path.startswith(('http://', 'https://')):
        if mode != 'r':
            raise ValueError(f"HTTP URLs only support read mode ('r'), but got mode '{mode}'")
        
        if verbose:
            print(f"Opening HTTP zarr store at {path} with storage_options: {storage_options}")
    
    # Open zarr store directly with storage_options
    if verbose:
        print(f"Opening zarr store at {path} with mode={mode}")
    
    # If we're creating a new array (mode='w') and shape is provided, pass creation parameters
    if mode == 'w' and shape is not None:
        create_kwargs = {}
        if chunks is not None:
            create_kwargs['chunks'] = chunks
        if dtype is not None:
            create_kwargs['dtype'] = dtype
        if compressor is not None:
            create_kwargs['compressor'] = compressor
        if fill_value is not None:
            create_kwargs['fill_value'] = fill_value
        if order is not None:
            create_kwargs['order'] = order
        
        # Add any other kwargs
        create_kwargs.update(kwargs)
        
        if verbose:
            print(f"Creating new zarr array with shape={shape}, chunks={chunks}, dtype={dtype}")
        
        return zarr.open(path, mode=mode, shape=shape, storage_options=storage_options, **create_kwargs)
    else:
        # Just open the existing array
        return zarr.open(path, mode=mode, storage_options=storage_options, **kwargs)