"""Exceptions raised by vc_reader lookups."""

from __future__ import annotations

from typing import Any, Sequence


class VcReaderError(Exception):
    """Base class for vc_reader errors."""


class NotFoundError(VcReaderError, LookupError):
    """A query matched nothing."""


class AmbiguousMatchError(VcReaderError, LookupError):
    """A query matched more than one item; ``candidates`` holds them all."""

    def __init__(self, message: str, candidates: Sequence[Any]):
        super().__init__(message)
        self.candidates = list(candidates)


class CacheMissError(VcReaderError, LookupError):
    """An offline read needed data that is in none of the disk caches."""


class LowDiskSpaceError(VcReaderError, OSError):
    """A remote file was not cached: the disk would fall under the cache's free-space floor.

    Nothing cached was removed, and the file was not served."""
