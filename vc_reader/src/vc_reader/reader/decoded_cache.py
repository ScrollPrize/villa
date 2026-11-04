"""Explicitly shared, process-local image cache and decode working budget (``vcr::sources::DecodedCache``)."""
from .._reader import DecodedCache

DEFAULT_DECODE_BYTES = 256 << 20

__all__ = ["DEFAULT_DECODE_BYTES", "DecodedCache"]
