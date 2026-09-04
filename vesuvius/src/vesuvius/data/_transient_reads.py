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
    ' 429',
    ' 500',
    ' 502',
    ' 503',
    ' 504',
)


def _is_transient_read_error(exc: BaseException) -> bool:
    """True when a read failure looks like a network hiccup rather than a bug."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(marker in text for marker in _TRANSIENT_READ_MARKERS):
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