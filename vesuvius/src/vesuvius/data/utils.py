
import numpy as np
import zarr
import os
from typing import Union, Dict, Any, Optional, Tuple


def _normalize_zarr_creation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Zarr 3 creation keywords for a Zarr 2 runtime."""
    normalized = dict(kwargs)
    if int(zarr.__version__.split('.', 1)[0]) >= 3:
        return normalized

    zarr_format = normalized.pop('zarr_format', None)
    if zarr_format not in (None, 2):
        raise ValueError(
            f"zarr-python 2 cannot create Zarr format {zarr_format}"
        )
    if zarr_format == 2:
        normalized.setdefault('zarr_version', 2)

    config = normalized.pop('config', None)
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("Zarr 2 compatibility requires config to be a dict")
        config = dict(config)
        if 'write_empty_chunks' in config:
            normalized['write_empty_chunks'] = config.pop('write_empty_chunks')
        if config:
            unsupported = ', '.join(sorted(config))
            raise TypeError(
                f"Zarr 2 does not support array config options: {unsupported}"
            )

    return normalized


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
              cache: bool = False,
              cache_size_mb: int = 256,
              # Additional zarr creation parameters
              shape: Optional[Tuple] = None,
              chunks: Optional[Tuple] = None,
              dtype: Any = None,
              compressor: Any = None,
              fill_value: Any = None,
              order: str = None,
              zarr_format: Optional[int] = 2,
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
    cache : bool, default False
        If True (read mode only), wrap the store in an in-memory LRU chunk
        cache (zarr's ``CacheStore``) so repeated reads of the same region
        are served locally instead of re-fetched from the remote store.
        Byte-exact: caches the compressed chunks as stored, so decoded
        values are identical with or without it.
    cache_size_mb : int, default 256
        Maximum size of the LRU chunk cache, in megabytes. Ignored unless
        ``cache=True``.
    shape, chunks, dtype, compressor, fill_value, order : zarr creation parameters
        Only used when mode is 'w' to create a new zarr array.
    zarr_format : Optional[int], default 2
        Zarr format version for arrays created with mode 'w'. Defaults to 2 because
        the numcodecs compressors used throughout this package (and the logits stores
        blend_logits validates) are v2 constructs: zarr 3 raises for `compressor=` on
        a v3 array. Pass None to accept zarr's own default. Ignored when the
        installed zarr is 2.x, where it is translated to ``zarr_version``.
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
        # Zarr 2 calls this keyword zarr_version. Normalize below before the
        # call so either supported runtime receives its native spelling.
        if zarr_format is not None:
            create_kwargs['zarr_format'] = zarr_format

        # Add any other kwargs
        create_kwargs.update(kwargs)
        create_kwargs = _normalize_zarr_creation_kwargs(create_kwargs)

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
        if cache and mode == 'r':
            if not _ZARR_V3:
                raise NotImplementedError(
                    "open_zarr(cache=True) requires zarr>=3 (uses "
                    "zarr.experimental.cache_store.CacheStore, added in zarr 3); "
                    f"installed zarr is {zarr.__version__}"
                )
            # Wrap the store in zarr's built-in read-through LRU chunk cache.
            # Repeated reads of the same region (overlapping training
            # patches, viewer panning, tracer neighborhood revisits) are
            # then served from memory instead of re-fetched over the
            # network. The cache holds the compressed chunk bytes exactly
            # as stored, so decoded values are byte-identical to the
            # uncached path. This restores what use_volume_store_cache
            # provided before zarr 3 removed LRUStoreCache.
            from zarr.experimental.cache_store import CacheStore
            from zarr.storage import FsspecStore, LocalStore, MemoryStore
            if path.startswith(('http://', 'https://', 's3://')):
                inner = FsspecStore.from_url(path, storage_options=storage_options, read_only=True)
            else:
                inner = LocalStore(path, read_only=True)
            cached_store = CacheStore(
                inner,
                cache_store=MemoryStore(),
                max_size=cache_size_mb * 2**20,
            )
            if verbose:
                print(f"Wrapping store in LRU chunk cache ({cache_size_mb} MB)")
            return zarr.open(cached_store, mode=mode, **kwargs)
        return zarr.open(path, mode=mode, **store_kwargs, **kwargs)
