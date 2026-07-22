
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
              cache: bool = False,
              cache_size_mb: int = 256,
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
    cache : bool, default False
        If True (read mode only), wrap the store in an in-memory LRU chunk
        cache so repeated reads of the same region are served locally instead
        of re-fetched from the remote store. Byte-exact: caches the compressed
        chunks as stored, so decoded values are identical with or without it.
    cache_size_mb : int, default 256
        Maximum size of the LRU chunk cache, in megabytes. Ignored unless
        ``cache=True``.
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
        
        return zarr.open(path, mode=mode, shape=shape, storage_options=storage_options or None, **create_kwargs)
    else:
        # Just open the existing array
        if cache and mode == 'r':
            # Wrap the store in an in-memory LRU chunk cache (see
            # LRUCacheStore below). Repeated reads of the same region
            # (overlapping training patches, viewer panning, tracer
            # neighborhood revisits) are then served from memory instead of
            # re-fetched over the network. The cache holds the compressed
            # chunk bytes exactly as stored, so decoded values are
            # byte-identical to the uncached path. This restores what
            # use_volume_store_cache provided before zarr 3 removed
            # LRUStoreCache.
            from zarr.storage import FsspecStore, LocalStore
            if path.startswith(('http://', 'https://', 's3://')):
                inner = FsspecStore.from_url(path, storage_options=storage_options, read_only=True)
            else:
                inner = LocalStore(path, read_only=True)
            cached_store = LRUCacheStore(inner, max_size_bytes=cache_size_mb * 2**20)
            if verbose:
                print(f"Wrapping store in LRU chunk cache ({cache_size_mb} MB)")
            return zarr.open(cached_store, mode=mode, **kwargs)
        # zarr 3 rejects storage_options for non-URL (local) paths, even empty
        return zarr.open(path, mode=mode, storage_options=storage_options or None, **kwargs)


class LRUCacheStore(zarr.storage.WrapperStore):
    """A read-through, in-memory, size-bounded LRU cache for zarr 3 stores.

    zarr 2's ``LRUStoreCache`` was removed in zarr 3 (which is why
    ``use_volume_store_cache`` in neural_tracing was deprecated); this is the
    zarr 3 equivalent, composed via ``zarr.storage.WrapperStore``. Cached
    values are the raw stored bytes (compressed chunks), keyed by
    ``(key, byte_range)``, so reads decode identically with or without the
    cache. Writes and deletes pass through and invalidate the affected key.
    """

    def __init__(self, store, max_size_bytes: int = 256 * 2**20):
        super().__init__(store)
        from collections import OrderedDict
        self._lru: "OrderedDict[tuple, bytes]" = OrderedDict()
        self._max_size = max_size_bytes
        self._current_size = 0

    def _evict(self, incoming: int) -> None:
        while self._lru and self._current_size + incoming > self._max_size:
            _, evicted = self._lru.popitem(last=False)
            self._current_size -= len(evicted)

    def _invalidate(self, key: str) -> None:
        for ck in [ck for ck in self._lru if ck[0] == key]:
            self._current_size -= len(self._lru.pop(ck))

    async def get(self, key, prototype, byte_range=None):
        cache_key = (key, repr(byte_range))
        if cache_key in self._lru:
            self._lru.move_to_end(cache_key)
            return prototype.buffer.from_bytes(self._lru[cache_key])
        value = await super().get(key, prototype, byte_range)
        if value is not None:
            raw = value.to_bytes()
            if len(raw) <= self._max_size:
                # Idempotent insert: if a concurrent miss for the same key
                # already populated it (zarr fetches chunks concurrently),
                # drop the prior entry first so _current_size can't
                # double-count. The mutation block has no await, so it runs
                # atomically under asyncio's single thread.
                if cache_key in self._lru:
                    self._current_size -= len(self._lru.pop(cache_key))
                self._evict(len(raw))
                self._lru[cache_key] = raw
                self._current_size += len(raw)
        return value

    async def set(self, key, value):
        self._invalidate(key)
        await super().set(key, value)

    async def delete(self, key):
        self._invalidate(key)
        await super().delete(key)