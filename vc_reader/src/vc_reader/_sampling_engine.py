"""Options, pickling and rebuilding after a fork for the native aligned readers."""
import operator
import os

_INHERITED = []  # Native engines a fork inherited: kept, never destroyed here.


class SamplingEngine:
    """Native reader built from its configuration, again in each process that uses it."""

    def _setup(self, backend, device, limits, reader_options):
        from .reader import DecodedCache, MemoryCache
        from .reader.source_reader import _ZARR_OPTIONS
        if backend not in ('cpu', 'cuda'):
            raise ValueError("backend must be 'cpu' or 'cuda'")
        self.backend, self.device = backend, operator.index(device)
        if min(limits) < 0 or limits[1] == 0 or self.device < 0:
            raise ValueError('cache/device settings must be nonnegative and sampling_work_bytes positive')
        if backend == 'cuda':
            from .cuda import is_available
            if not is_available():
                raise RuntimeError('CUDA sampling requires a configured CUDA build and device')
            import sys
            torch = sys.modules.get('torch')
            if torch is not None and torch.utils.data.get_worker_info() is not None:
                raise RuntimeError('CUDA sampling belongs in the device-owning process, not a DataLoader worker')
        options = dict(reader_options)
        self._own_decoded = 'decoded_cache' not in options
        self._own_memory = not isinstance(options.get('memory_cache'), MemoryCache)
        if self._own_decoded:
            self._decoded = DecodedCache(max_bytes=options.pop('decoded_cache_bytes', 256 << 20),
                                         max_decode_bytes=options.pop('max_decode_bytes', 256 << 20))
        else:
            self._decoded = options.pop('decoded_cache')
        self._memory = options.pop('memory_cache', None)
        if self._own_memory:
            self._memory = MemoryCache(self._memory if self._memory is not None else 64 << 20)
        unknown = options.keys() - _ZARR_OPTIONS
        if unknown:
            raise TypeError(f'unknown reader options: {sorted(unknown)}')
        self._options = options
        self._pid, self._closed, self._native = os.getpid(), False, None

    def _options_for(self, location):
        """The native options of one location, in this process."""
        from .reader.zarr_reader import _make_settings, _open_options
        settings, _ = _make_settings(location, **self._options)
        return _open_options(settings, self._memory)

    @property
    def metadata(self):
        return self._ready().metadata

    def _ready(self):
        if getattr(self, '_lazy', False):
            self._lazy = False
            self._restore(self._configuration)
        if self._closed:
            raise ValueError('sample reader is closed')
        if self._pid != os.getpid():
            if self.backend == 'cuda':
                raise RuntimeError('CUDA sample readers cannot be used after fork')
            _INHERITED.append(self._native)
            self._pid = os.getpid()
            self._native = None
        if self._native is None:
            self._build()
        return self._native

    def stats(self):
        return self._ready().stats()

    def close(self):
        if getattr(self, '_closed', True):
            return
        if getattr(self, '_lazy', False):
            self._closed = True
            return
        if self._pid != os.getpid():
            _INHERITED.append(self._native)
            self._native = None
        if self._native is not None:
            self._native.close()
            self._native = None
        if self._own_decoded:
            self._decoded.clear()
        if self._own_memory:
            self._memory.clear()
        self._closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        if self.backend == 'cuda':
            raise TypeError('CUDA sample readers cannot be pickled')
        if self._closed:
            raise ValueError('sample reader is closed')
        return self._configuration

    def __setstate__(self, state):
        self._configuration = state
        self.backend = 'cpu'
        self._closed, self._lazy, self._pid = False, True, os.getpid()
