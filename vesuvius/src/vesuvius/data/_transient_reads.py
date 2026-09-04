"""Classification and bounded retry for transient remote reads.

zarr and fsspec wrap the underlying transport (aiohttp, botocore, urllib3,
ssl) differently across versions and backends, so transient failures are
classified by message substring rather than by exception type.  Retrying
happens only for failures that look like a network hiccup; deterministic
errors (missing array, bad coordinates, auth) re-raise immediately.

This module is a leaf: it imports nothing from vesuvius, so both
``data.volume`` and ``ink_detection.volume_io`` can import it without any
circular-import risk.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional


_TRANSIENT_READ_MARKERS = (
    'payload is not completed',
    'not enough data to satisfy content length',
    'contentlengtherror',
    'connection reset',
    'connection aborted',
    'connection closed',
    'server disconnected',
    'record layer failure',
    'ssl',
    'timed out',
    'timeout',
    'temporarily unavailable',
    'slowdown',
    'throttl',
    'too many requests',
    'internal error',
    'service unavailable',
    'bad gateway',
    'gateway timeout',
)

# HTTP statuses worth retrying, but only when the number appears as a status:
# "ClientResponseError: 503, message=...", "An error occurred (503) when
# calling GetObject", "HTTP 429", "status: 502". A bare " 504" also matched
# "index out of bounds for dimension with length 504" and "size 5040", which
# turned a coding error into four attempts and 3.5 s of backoff.
_TRANSIENT_HTTP_STATUS = re.compile(
    r'(?:(?:error|status|http|response|code)\W{0,4}|\(\s*)(?:429|500|502|503|504)(?!\d)'
)

# Exceptions that describe a bug or a bad request rather than a network
# hiccup. Their messages are not scanned for markers (zarr's BoundsCheckError
# is an IndexError whose text names the dimension length), but the cause
# chain is still walked in case one wraps a genuine transport error.
_DETERMINISTIC_ERROR_TYPES = (
    IndexError,
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    NotImplementedError,
)


_NON_TRANSIENT_TYPES = _DETERMINISTIC_ERROR_TYPES


def _is_transient_read_error(exc: BaseException) -> bool:
    """True when a read failure looks like a network hiccup rather than a bug.

    Coding errors are never transient, regardless of their message: zarr's
    ``BoundsCheckError`` subclasses ``IndexError`` and its message contains a
    dimension length (e.g. ``... length 504``) that a bare numeric marker
    (``' 503'``, ``' 504'``) would otherwise match.  Skip the marker scan for
    those types, but keep walking the cause chain so a genuine transport
    error wrapped behind a deterministic wrapper is still detected.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if not isinstance(exc, _NON_TRANSIENT_TYPES):
            text = f"{type(exc).__name__}: {exc}".lower()
            if any(marker in text for marker in _TRANSIENT_READ_MARKERS):
                return True
            if _TRANSIENT_HTTP_STATUS.search(text):
                return True
        exc = exc.__cause__ or exc.__context__
    return False


def _read_array_with_retry(
    volume: Any,
    selection: tuple[Any, ...],
    *,
    retries: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    warn: Optional[Callable[[str], None]] = None,
) -> Any:
    """Read ``volume[selection]``, retrying transient remote failures.

    Remote object stores drop connections routinely — truncated payloads,
    SSL record errors, 5xx, throttling.  A caller streaming a whole volume
    issues one read per patch, so without this a single hiccup propagates
    out and aborts the job, discarding everything computed so far.

    Deterministic failures (bad coordinates, missing array, auth) are not
    retried; they re-raise on the first attempt.
    """
    retries = max(1, int(retries))
    delay = base_delay
    for attempt in range(retries):
        try:
            return volume[selection]
        except Exception as e:
            if attempt == retries - 1 or not _is_transient_read_error(e):
                raise
            if warn is not None:
                warn(
                    f"transient remote read error ({type(e).__name__}), retry "
                    f"{attempt + 1}/{retries - 1} in {delay:.1f}s: {e}"
                )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)