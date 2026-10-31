import os
import time

from vc_reader.reader import ObjectStore


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


def wait_for_zarr_creation(zarr_path, max_wait_time=300, sleep_interval=5, verbose=True, part_id=None):
    """
    Wait for a zarr array to be created by another process.

    Args:
        zarr_path: Path to the zarr array to wait for
        max_wait_time: Maximum time to wait in seconds (default: 300 = 5 minutes)
        sleep_interval: Time to sleep between checks in seconds (default: 5)
        verbose: Whether to print progress messages
        part_id: Optional part ID for logging purposes

    Raises:
        RuntimeError: If timeout is reached without array being created
    """
    wait_time = 0
    part_str = f"Part {part_id} " if part_id is not None else ""

    while not zarr_array_exists(zarr_path):
        if wait_time >= max_wait_time:
            raise RuntimeError(f"Timeout waiting for zarr array to be created. Waited {max_wait_time} seconds.")

        if verbose:
            print(f"  {part_str}waiting for array to be created... ({wait_time}s elapsed)")
        time.sleep(sleep_interval)
        wait_time += sleep_interval

    if verbose:
        print(f"Array found! {part_str}can now proceed.")