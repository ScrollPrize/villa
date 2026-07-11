import hashlib
import os
from collections.abc import MutableMapping
from pathlib import Path


class DiskCachedStore(MutableMapping):
    """Read-through chunk cache over a zarr store mapping. Atomic writes keep
    it safe for concurrent dataloader workers sharing one cache directory.

    `max_bytes` evicts least-recently-used chunks (reads refresh mtime); the
    budget is tracked per process and trued up against the directory on each
    eviction, so concurrent workers overshoot only briefly."""

    def __init__(self, source, cache_dir: Path, max_bytes: int | None = None):
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._size = (
            sum(f.stat().st_size for f in self.cache_dir.iterdir() if f.suffix != ".tmp")
            if max_bytes is not None else 0
        )

    def _path(self, key: str) -> Path:
        return self.cache_dir / key.replace("/", "__")

    def __getitem__(self, key: str) -> bytes:
        path = self._path(key)
        try:
            value = path.read_bytes()
        except FileNotFoundError:
            value = self.source[key]
            temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temp.write_bytes(value)
            os.replace(temp, path)
            if self.max_bytes is not None:
                self._size += len(value)
                if self._size > self.max_bytes:
                    self._evict()
            return value
        os.utime(path)
        return value

    def _evict(self):
        snapshot = sorted(
            (entry.stat().st_mtime, entry.stat().st_size, entry.path)
            for entry in os.scandir(self.cache_dir)
            if not entry.name.endswith(".tmp")
        )
        total = sum(size for _, size, _ in snapshot)
        for _, size, file_path in snapshot:
            if total <= 0.9 * self.max_bytes:
                break
            Path(file_path).unlink(missing_ok=True)
            total -= size
        self._size = total

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists() or key in self.source

    def __setitem__(self, key, value):
        raise NotImplementedError("read-only store")

    def __delitem__(self, key):
        raise NotImplementedError("read-only store")

    def __iter__(self):
        return iter(self.source)

    def __len__(self):
        return len(self.source)


def wrap_store_with_disk_cache(store, source_path, cache_dir, cache_max_gb):
    subdir = Path(cache_dir) / hashlib.sha1(str(source_path).encode()).hexdigest()[:12]
    max_bytes = int(cache_max_gb * 1e9) if cache_max_gb is not None else None
    return DiskCachedStore(store, subdir, max_bytes=max_bytes)
