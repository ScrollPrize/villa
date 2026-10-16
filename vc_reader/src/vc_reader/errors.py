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
