from pathlib import Path
import zarr
import json
import os
import fsspec
import time

def _is_ome_zarr(zarr_path):
    """
    Check if a zarr directory has multi-resolution pyramid structure.

    Detects:
    - Standard OME-Zarr with .zattrs multiscales metadata
    - Pyramid zarrs with numbered subdirectories (0, 1, 2, etc.) even without .zattrs
    """
    zarr_path = Path(zarr_path)

    # Check for numbered subdirectories (0, 1, etc.)
    has_level_0 = (zarr_path / '0').exists()
    if not has_level_0:
        return False

    # If level 0 exists, check if it contains array data (not just another group)
    level_0_path = zarr_path / '0'
    has_zarray = (level_0_path / '.zarray').exists()
    if not has_zarray:
        return False

    # At this point we have numbered directories with array data - treat as multi-resolution
    # Optionally verify .zattrs contains multiscales metadata (but not required)
    zattrs_path = zarr_path / '.zattrs'
    if zattrs_path.exists():
        try:
            with open(zattrs_path, 'r') as f:
                attrs = json.load(f)
                if 'multiscales' in attrs:
                    return True
        except Exception:
            pass

    # Even without .zattrs, if we have level 0 with array data, treat as multi-resolution
    return True

def _get_zarr_path(zarr_dir, resolution_level=None):
    """
    Get the appropriate path for opening a zarr array.
    
    For OME-Zarr files, appends the resolution level to the path.
    For regular zarr files, returns the path as-is.
    
    Args:
        zarr_dir: Path to the zarr directory
        resolution_level: Resolution level to use (default: 0 for OME-Zarr, None for regular zarr)
    
    Returns:
        str: Path to use for zarr.open()
    """
    zarr_dir = Path(zarr_dir)
    
    if _is_ome_zarr(zarr_dir):
        # Use resolution level 0 by default for OME-Zarr
        if resolution_level is None:
            resolution_level = 0
        
        zarr_path = zarr_dir / str(resolution_level)
        
        # Verify the resolution level exists
        if not zarr_path.exists():
            available_levels = [d.name for d in zarr_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            raise ValueError(f"Resolution level {resolution_level} not found in {zarr_dir}. Available levels: {sorted(available_levels)}")
        
        return str(zarr_path)
    else:
        # Regular zarr file
        return str(zarr_dir)

def zarr_array_exists(zarr_path):
    """
    Check if a zarr array exists at the given path.
    Works for both local paths and S3 paths.

    Args:
        zarr_path: Path to the zarr array directory

    Returns:
        bool: True if the zarr array exists, False otherwise
    """
    try:
        if zarr_path.startswith('s3://'):
            fs = fsspec.filesystem('s3', anon=False)
            # Check if .zarray file exists within the zarr directory
            return fs.exists(os.path.join(zarr_path, '.zarray'))
        else:
            # For local paths, check if the .zarray file exists
            return os.path.exists(os.path.join(zarr_path, '.zarray'))
    except Exception:
        return False


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