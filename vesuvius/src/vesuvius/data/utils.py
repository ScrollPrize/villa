
import numpy as np
import zarr
import os
from typing import Union, Dict, Any, Optional, Tuple

_ZARR_V3 = int(zarr.__version__.split('.', 1)[0]) >= 3

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



_CACHE_ENV = "VESUVIUS_CHUNK_CACHE_DIR"


def _chunk_cache_url(path: str, storage_options: Dict[str, Any],
                     cache_dir: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Route a remote store through an on-disk chunk cache.

    Without this every read goes to the network, so overlapping patches
    re-download the same chunks: re-reading an identical patch transfers the
    full amount again. This replaces the caching that was lost when zarr 3
    removed ``LRUStoreCache`` (see the ``use_volume_store_cache`` warning in
    ``neural_tracing/heatmap_single_point/dataset.py``).

    ``simplecache`` rather than ``filecache``: the open-data volumes are
    immutable, so revalidating each chunk against S3 only costs a round trip.
    Measured on PHercParis4 level 2, same patch read three times —
    ``simplecache``: 8.39 MB then 0 MB, 0 requests; ``filecache``: 8.39 MB then
    0 MB but 4 requests each time.

    Returns the (possibly rewritten) path and storage options.
    """
    if not cache_dir:
        return path, storage_options

    protocol = path.split("://", 1)[0]
    nested = dict(storage_options)
    return (
        f"simplecache::{path}",
        {protocol: nested, "simplecache": {"cache_storage": cache_dir}},
    )


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
              zarr_format: Optional[int] = 2,
              cache_dir: Optional[str] = None,
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
    zarr_format : Optional[int], default 2
        Zarr format version for arrays created with mode 'w'. Defaults to 2 because
        the numcodecs compressors used throughout this package (and the logits stores
        blend_logits validates) are v2 constructs: zarr 3 raises for `compressor=` on
        a v3 array. Pass None to accept zarr's own default. Ignored when the
        installed zarr is 2.x, which only writes v2 and does not accept the argument.
    **kwargs : Additional parameters passed to zarr.open
        
    Returns:
    -------
    zarr.Array
        The opened zarr array
    """
    storage_options = dict(storage_options or {})
    is_remote = path.startswith(('http://', 'https://', 's3://'))
    
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

        # Disable boto3 request/response checksums: they show up as the top CPU
        # consumer in inference (httpchecksum.update) and add no value on top of
        # TLS + S3 ETag. Requires botocore >= 1.36.
        config_kwargs = dict(storage_options.get('config_kwargs') or {})
        config_kwargs.setdefault('request_checksum_calculation', 'when_required')
        config_kwargs.setdefault('response_checksum_validation', 'when_required')
        storage_options['config_kwargs'] = config_kwargs

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
    
    # Zarr 3.2 rejects storage_options for local filesystem stores, even when
    # the mapping is empty. Only URI-backed fsspec stores consume this option.
    # Cache chunks on disk for remote reads. Opt-in: explicit cache_dir, or
    # the VESUVIUS_CHUNK_CACHE_DIR environment variable.
    if is_remote and mode == 'r':
        resolved_cache = cache_dir or os.environ.get(_CACHE_ENV)
        if resolved_cache:
            path, storage_options = _chunk_cache_url(
                path, storage_options, resolved_cache)
            if verbose:
                print(f"Caching chunks under {resolved_cache}")

    store_kwargs = {'storage_options': storage_options} if is_remote else {}

    # Open the Zarr store with the protocol-appropriate keyword arguments.
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
        # zarr 2 has no zarr_format argument: it warns "ignoring keyword
        # argument 'zarr_format'" for every array and writes v2 regardless,
        # which is already what this default asks for. Only zarr 3 needs telling.
        if zarr_format is not None and _ZARR_V3:
            create_kwargs['zarr_format'] = zarr_format

        # Add any other kwargs
        create_kwargs.update(kwargs)

        # zarr 3 derives `overwrite` from the mode, so passing it alongside
        # mode='w' raises "got multiple values for keyword argument 'overwrite'".
        # mode='w' already means truncate, so an explicit overwrite=True is
        # redundant; drop it rather than making every caller know this.
        if create_kwargs.pop('overwrite', None) is False:
            raise ValueError(
                "open_zarr(mode='w') always truncates; overwrite=False is contradictory"
            )

        if verbose:
            print(f"Creating new zarr array with shape={shape}, chunks={chunks}, dtype={dtype}")
        
        return zarr.open(path, mode=mode, shape=shape, **store_kwargs, **create_kwargs)
    else:
        # Just open the existing array
        return zarr.open(path, mode=mode, **store_kwargs, **kwargs)
