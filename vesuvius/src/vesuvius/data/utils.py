import numpy as np
import zarr
from typing import Union, Dict, Any, Optional

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

def open_zarr(path: str, mode: str = 'r',
              storage_options: Optional[Dict[str, Any]] = None,
              verbose: bool = False,
              cache: bool = False,
              cache_size_mb: int = 256,
              **kwargs) -> zarr.Array:
    """
    Open a zarr array for reading, local or remote. Writers use ``vc_reader.reader.ZarrWriter``.

    Parameters:
    ----------
    path : str
        Path to the zarr array. Can be a local path, HTTP URL, or S3 URL.
    mode : str, default 'r'
        Only 'r'.
    storage_options : Optional[Dict[str, Any]], default None
        Additional options for storage backend. For S3, {'anon': False} will be added by default.
    verbose : bool, default False
        Whether to print verbose information about opening the zarr array.
    cache : bool, default False
        Wrap the store in an in-memory LRU chunk cache (zarr's ``CacheStore``).
        Byte-exact: caches the compressed chunks as stored.
    cache_size_mb : int, default 256
        Maximum size of the LRU chunk cache, in megabytes. Ignored unless ``cache=True``.
    **kwargs : Additional parameters passed to zarr.open

    Returns:
    -------
    zarr.Array
        The opened zarr array
    """
    if mode != 'r':
        raise ValueError(f"open_zarr only reads; write with vc_reader.reader.ZarrWriter (got mode={mode!r})")
    storage_options = dict(storage_options or {})
    is_remote = path.startswith(('http://', 'https://', 's3://'))

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

    # Zarr 3.2 rejects storage_options for local filesystem stores, even when
    # the mapping is empty. Only URI-backed fsspec stores consume this option.
    store_kwargs = {'storage_options': storage_options} if is_remote else {}

    if verbose:
        print(f"Opening zarr store at {path}")

    if cache:
        if not _ZARR_V3:
            raise NotImplementedError(
                "open_zarr(cache=True) requires zarr>=3 (uses "
                "zarr.experimental.cache_store.CacheStore, added in zarr 3); "
                f"installed zarr is {zarr.__version__}"
            )
        from zarr.experimental.cache_store import CacheStore
        from zarr.storage import FsspecStore, LocalStore, MemoryStore
        if is_remote:
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
