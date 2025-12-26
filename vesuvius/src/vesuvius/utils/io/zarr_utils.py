from pathlib import Path
import zarr
import json

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