"""Seekable image files using the reader's HTTP transport and disk budget."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile

from ..errors import CacheMissError
from .auth import is_remote, resolve_access


class EncodedFiles:
    """Configuration plus process-local, lazily resolved file access.

    File descriptors pin open inodes across cache eviction; no private copies
    bypass the disk budget. Each location is revalidated once per owner/process.
    """

    def __init__(self, *, cache_dir=None, cache_max_bytes=None, cache_writes=True,
                 auth='auto', offline=False):
        self.cache_dirs = ((os.fspath(cache_dir),) if isinstance(cache_dir, (str, os.PathLike)) else
                           tuple(os.fspath(p) for p in cache_dir or ()))
        if cache_max_bytes is not None and (type(cache_max_bytes) is not int or cache_max_bytes < 0):
            raise ValueError('cache_max_bytes must be a nonnegative integer')
        self.max_bytes, self.cache_writes, self.auth, self.offline = cache_max_bytes, cache_writes, auth, offline
        self._pid = os.getpid()
        self._temp = None
        self._checked = set()
        self._access = {}
        self._identities = {}
        self.bytes_read = self.cache_hits = 0

    def _process(self):
        if self._pid != os.getpid():
            # A child must not unlink the parent's temporary cache on GC.
            if self._temp is not None:
                self._temp._finalizer.detach()
            self._temp = None
            self._access = {}
            self._identities = {}
            self._checked = set()
            self.bytes_read = self.cache_hits = 0
            self._pid = os.getpid()

    @staticmethod
    def _key(location):
        return hashlib.sha256(location.encode()).hexdigest() + '.blob'

    @contextmanager
    def open(self, location):
        self._process()
        if not is_remote(location):
            with open(Path(location).expanduser(), 'rb') as handle:
                stat = os.fstat(handle.fileno())
                yield handle, (str(Path(location).expanduser().absolute()), stat.st_dev, stat.st_ino,
                               stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            return
        from .disk_cache import disk_budget

        if 'vc3d' in self.cache_dirs:
            raise ValueError('image files need explicit cache directories, not the vc3d chunk-cache alias')
        name = self._key(location)
        roots = [Path(p).expanduser().resolve() for p in self.cache_dirs]
        if self.offline or not self.cache_writes or location in self._checked:
            checked_roots = roots + ([Path(self._temp.name)] if self._temp is not None else [])
            for root in checked_roots:
                try:
                    handle = open(root / 'encoded' / name, 'rb')
                except FileNotFoundError:
                    continue
                with handle:
                    self.cache_hits += 1
                    yield handle, self._identity(location, handle)
                return
            if self.offline:
                raise CacheMissError(f'{location}: encoded file is not cached (offline)')
        root = next((r for r in roots if (r / 'encoded' / name).exists()), roots[0] if roots else None)
        if root is None or not self.cache_writes:
            if self._temp is None:
                self._temp = tempfile.TemporaryDirectory(prefix='vc-reader-images-')
            root = Path(self._temp.name)
        # None reuses the configured directory limit, or the reader's 1 GiB default
        # for a new directory. It never resets another reader's explicit budget.
        budget = disk_budget(str(root), self.max_bytes)
        if location not in self._access:
            self._access[location] = resolve_access(location, self.auth)
        access = self._access[location]
        fd, etag, modified, received = budget.open_file(
            str(root / 'encoded' / name), access.url(location), access.auth_tuple(), False,
            location not in self._checked)
        self.bytes_read += received
        if fd < 0:
            raise FileNotFoundError(f'{location}: remote image does not exist')
        self._checked.add(location)
        if not received:
            self.cache_hits += 1
        with os.fdopen(fd, 'rb') as handle:
            # Hashing has bounded scratch and also gives identical identities
            # online/offline when no server validator can be consulted offline.
            yield handle, self._identity(location, handle)

    def _identity(self, location, handle):
        stat = os.fstat(handle.fileno())
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        previous = self._identities.get(location)
        if previous is not None and previous[0] == signature:
            return previous[1]
        digest = hashlib.file_digest(handle, 'sha256') if hasattr(hashlib, 'file_digest') else None
        if digest is None:
            digest = hashlib.sha256()
            while block := handle.read(1 << 16):
                digest.update(block)
        handle.seek(0)
        token = (location, digest.hexdigest())
        self._identities[location] = signature, token
        return token

    def close(self):
        self._process()
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
        self._checked.clear()

    def __getstate__(self):
        return dict(cache_dir=self.cache_dirs, cache_max_bytes=self.max_bytes,
                    cache_writes=self.cache_writes, auth=self.auth, offline=self.offline)

    def __setstate__(self, state):
        self.__init__(**state)
