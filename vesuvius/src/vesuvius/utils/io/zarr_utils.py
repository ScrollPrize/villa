import math
import os
import time

from vc_reader.reader import ObjectStore


_ZARR_RUN_MARKER = ".vesuvius-run-id"


def mark_zarr_ready(zarr_path, run_id):
    """Publish this run only after array creation completes; object replacement is atomic."""
    ObjectStore(zarr_path).write(_ZARR_RUN_MARKER, run_id)


def zarr_array_exists(zarr_path):
    """
    Check if a zarr array exists at the given path.
    Works for both local paths and S3 paths.

    Args:
        zarr_path: Path to the zarr array directory

    Returns:
        bool: True if the zarr array exists, False if it is absent.

    Raises:
        RuntimeError: if existence could not be determined at all (credentials,
            permissions, network). Reporting that as False would be
            indistinguishable from the array being absent.
    """
    # Accept str or os.PathLike. The previous implementation caught the
    # AttributeError this would otherwise raise and returned False for every
    # Path argument, so coercing here keeps that input working and answers it.
    zarr_path = os.fspath(zarr_path)
    marker = os.path.join(zarr_path, '.zarray')

    if not zarr_path.startswith('s3://'):
        # For local paths, check if the .zarray file exists
        return os.path.exists(marker)

    # The open-data bucket is published public-read, so a machine with no AWS
    # credentials must still be able to answer this. Try the configured
    # credentials first, then fall back to anonymous access.
    last_error = None
    for auth in ("aws", None):
        try:
            return ObjectStore(zarr_path, auth=auth).exists('.zarray')
        except Exception as exc:
            last_error = exc

    # Reporting a credentials or network failure as "the array does not exist"
    # sends callers down the wrong path entirely: wait_for_zarr_creation() below
    # would sit out its full timeout and then report that the array was never
    # created, when the truth is that we were never able to look.
    raise RuntimeError(
        f"Could not determine whether a zarr array exists at {zarr_path}: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def wait_for_zarr_creation(zarr_path, max_wait_time=300, sleep_interval=5, verbose=True, part_id=None,
                           run_id=None):
    """
    Wait for a zarr array to be created by another process.

    Args:
        zarr_path: Path to the zarr array to wait for
        max_wait_time: Maximum time to wait in seconds (default: 300 = 5 minutes)
        sleep_interval: Time to sleep between checks in seconds (default: 5)
        verbose: Whether to print progress messages
        part_id: Optional part ID for logging purposes
        run_id: When set, wait for mark_zarr_ready() to publish this exact run, not an old array.

    Raises:
        RuntimeError: If timeout is reached without array being created
    """
    if not math.isfinite(max_wait_time) or max_wait_time <= 0:
        raise ValueError("max_wait_time must be finite and positive.")
    if not math.isfinite(sleep_interval) or sleep_interval <= 0:
        raise ValueError("sleep_interval must be finite and positive.")
    start = time.monotonic()
    part_str = f"Part {part_id} " if part_id is not None else ""
    store = ObjectStore(zarr_path) if run_id is not None else None
    expected_run = run_id.encode() if run_id is not None else None
    description = f"run {run_id!r} at {zarr_path}" if run_id is not None else f"array at {zarr_path}"

    while True:
        ready = (store.read(_ZARR_RUN_MARKER) == expected_run if store is not None
                 else zarr_array_exists(zarr_path))
        if ready:
            break
        elapsed = time.monotonic() - start
        if elapsed >= max_wait_time:
            raise RuntimeError(f"Timeout waiting for {description}. Waited {max_wait_time} seconds.")
        if verbose:
            print(f"  {part_str}waiting for {description}... ({elapsed:.0f}s elapsed)")
        time.sleep(min(sleep_interval, max_wait_time - elapsed))

    if verbose:
        print(f"Ready: {description}. {part_str}can now proceed.")
