"""Reader for tifxyz format files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import tifffile

from .types import Tifxyz

logger = logging.getLogger(__name__)


@dataclass
class TifxyzInfo:
    """Lightweight metadata for a tifxyz segment without loading coordinates.

    Use this for filtering/listing segments before deciding which to fully load.
    """

    path: Path
    scale: Tuple[float, float]
    bbox: Optional[Tuple[float, float, float, float, float, float]]
    uuid: str

    @property
    def z_min(self) -> Optional[float]:
        """Return minimum z coordinate from bbox, or None if no bbox."""
        return self.bbox[2] if self.bbox else None

    @property
    def z_max(self) -> Optional[float]:
        """Return maximum z coordinate from bbox, or None if no bbox."""
        return self.bbox[5] if self.bbox else None

    def load(self, **kwargs) -> Tifxyz:
        """Load the full Tifxyz object for this segment.

        Parameters
        ----------
        **kwargs
            Passed to read_tifxyz (e.g., load_mask, validate).

        Returns
        -------
        Tifxyz
            The fully loaded surface.
        """
        return read_tifxyz(self.path, **kwargs)


def list_tifxyz(
    folder: Union[str, Path],
    *,
    z_range: Optional[Tuple[float, float]] = None,
    recursive: bool = True,
) -> List[TifxyzInfo]:
    """Discover all tifxyz segments in a folder.

    Scans for directories containing the required tifxyz files (x.tif, y.tif,
    z.tif, meta.json) and returns lightweight metadata for each, without
    loading the full coordinate arrays.

    Parameters
    ----------
    folder : Union[str, Path]
        Path to folder containing tifxyz segment directories.
    z_range : Optional[Tuple[float, float]]
        If provided, filter to only segments whose bbox overlaps this z range.
        Format: (z_min, z_max).
    recursive : bool
        If True (default), search recursively. If False, only search immediate
        subdirectories.

    Returns
    -------
    List[TifxyzInfo]
        List of TifxyzInfo objects for discovered segments, sorted by path.

    Examples
    --------
    >>> from vesuvius.tifxyz import list_tifxyz
    >>> segments = list_tifxyz("/path/to/segments", z_range=(1000, 2000))
    >>> for seg in segments:
    ...     print(f"{seg.uuid}: z=[{seg.z_min}, {seg.z_max}]")
    ...     surface = seg.load()  # Load full data when needed
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    results = []

    # Find all meta.json files (indicates a tifxyz directory)
    pattern = "**/meta.json" if recursive else "*/meta.json"
    for meta_path in folder.glob(pattern):
        segment_dir = meta_path.parent

        # Check if required files exist
        required = ["x.tif", "y.tif", "z.tif"]
        if not all((segment_dir / f).exists() for f in required):
            continue

        try:
            reader = TifxyzReader(segment_dir)
            meta = reader.read_metadata()

            info = TifxyzInfo(
                path=segment_dir,
                scale=meta["scale"],
                bbox=meta["bbox"],
                uuid=meta["uuid"],
            )

            # Filter by z_range if specified
            if z_range is not None and info.bbox is not None:
                bbox_z_min, bbox_z_max = info.z_min, info.z_max
                range_z_min, range_z_max = z_range
                # Skip if bbox doesn't overlap z_range
                if bbox_z_min > range_z_max or bbox_z_max < range_z_min:
                    continue

            results.append(info)

        except Exception as e:
            logger.warning(f"Error reading metadata from {segment_dir}: {e}")
            continue

    # Sort by path for consistent ordering
    results.sort(key=lambda x: x.path)
    return results


def load_folder(
    folder: Union[str, Path],
    *,
    z_range: Optional[Tuple[float, float]] = None,
    recursive: bool = True,
    **load_kwargs,
) -> Iterator[Tifxyz]:
    """Load all tifxyz segments in a folder.

    Convenience function that combines list_tifxyz with loading.

    Parameters
    ----------
    folder : Union[str, Path]
        Path to folder containing tifxyz segment directories.
    z_range : Optional[Tuple[float, float]]
        If provided, filter to only segments whose bbox overlaps this z range.
    recursive : bool
        If True (default), search recursively.
    **load_kwargs
        Passed to read_tifxyz (e.g., load_mask, validate).

    Yields
    ------
    Tifxyz
        Loaded surface for each discovered segment.

    Examples
    --------
    >>> from vesuvius.tifxyz import load_folder
    >>> for surface in load_folder("/path/to/segments", z_range=(1000, 2000)):
    ...     print(f"{surface.uuid}: {surface.shape}")
    """
    for info in list_tifxyz(folder, z_range=z_range, recursive=recursive):
        try:
            yield info.load(**load_kwargs)
        except Exception as e:
            logger.warning(f"Error loading {info.path}: {e}")
            continue


def read_tifxyz(
    path: Union[str, Path],
    *,
    load_mask: bool = True,
    validate: bool = True,
    volume_path: Optional[Union[str, Path]] = None,
) -> Tifxyz:
    """Read a tifxyz directory into a Tifxyz object.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the tifxyz directory containing x.tif, y.tif, z.tif, and meta.json.
    load_mask : bool
        If True, load mask.tif if present. Default True.
    validate : bool
        If True, validate the loaded data. Default True.
    volume_path : Union[str, Path], optional
        Path to an OME-zarr volume to associate with the surface.
        If provided, opens the zarr and attaches it to tifxyz.volume.

    Returns
    -------
    Tifxyz
        The loaded surface data. Access coordinates via indexing (e.g., surface[row, col])
        which provides interpolated access at full resolution.

    Raises
    ------
    FileNotFoundError
        If required files (x.tif, y.tif, z.tif, meta.json) are missing.
    ValueError
        If validation fails (shape mismatch, invalid data).
    """
    reader = TifxyzReader(path)
    tifxyz = reader.read(load_mask=load_mask, validate=validate)

    if volume_path is not None:
        import zarr

        tifxyz.volume = zarr.open(volume_path, mode="r")

    return tifxyz


class TifxyzReader:
    """Class-based reader for tifxyz directories.

    Use this for more control over the reading process or for reading
    metadata without loading coordinate arrays.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the tifxyz directory.

    Attributes
    ----------
    path : Path
        The tifxyz directory path.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Initialize reader with path to tifxyz directory."""
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"tifxyz directory not found: {self.path}")

    def _check_required_files(self) -> None:
        """Check that required files exist."""
        required = ["x.tif", "y.tif", "z.tif", "meta.json"]
        missing = [f for f in required if not (self.path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required files in {self.path}: {missing}"
            )

    def read_metadata(self) -> dict:
        """Read and parse the metadata from meta.json.

        Returns
        -------
        dict
            Dictionary with parsed metadata fields:
            - uuid: str
            - scale: Tuple[float, float] (scale_y, scale_x)
            - bbox: Optional tuple
            - format: str
            - surface_type: str
            - area: Optional[float]
            - extra: dict of additional fields
        """
        meta_path = self.path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r") as f:
            meta_dict = json.load(f)

        # Parse scale - C++ stores as [x_scale, y_scale], we use (scale_y, scale_x)
        scale_raw = meta_dict.get("scale", [20.0, 20.0])
        if isinstance(scale_raw, list) and len(scale_raw) >= 2:
            # C++ format: [x_scale, y_scale]
            # We store as (scale_y, scale_x) for consistency with array indexing
            scale = (float(scale_raw[1]), float(scale_raw[0]))
        else:
            scale = (20.0, 20.0)

        # Parse bbox
        bbox_raw = meta_dict.get("bbox")
        bbox = None
        if bbox_raw is not None and len(bbox_raw) == 2:
            # bbox format: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            min_coords = bbox_raw[0]
            max_coords = bbox_raw[1]
            bbox = (
                float(min_coords[0]),  # x_min
                float(min_coords[1]),  # y_min
                float(min_coords[2]),  # z_min
                float(max_coords[0]),  # x_max
                float(max_coords[1]),  # y_max
                float(max_coords[2]),  # z_max
            )

        # Extract known fields and put rest in extra
        known_keys = {"uuid", "scale", "bbox", "area"}
        extra = {k: v for k, v in meta_dict.items() if k not in known_keys}

        return {
            "uuid": meta_dict.get("uuid", self.path.name),
            "scale": scale,
            "bbox": bbox,
            "area": meta_dict.get("area"),
            "extra": extra,
        }

    def read_coordinate(self, component: str) -> np.ndarray:
        """Read a single coordinate component ('x', 'y', or 'z').

        Parameters
        ----------
        component : str
            Which component to read: 'x', 'y', or 'z'.

        Returns
        -------
        np.ndarray
            The coordinate array as float32.
        """
        if component not in ("x", "y", "z"):
            raise ValueError(f"Invalid component: {component}. Must be 'x', 'y', or 'z'")

        tif_path = self.path / f"{component}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"Coordinate file not found: {tif_path}")

        data = tifffile.imread(str(tif_path))
        return data.astype(np.float32)

    def read_mask(self) -> Optional[np.ndarray]:
        """Read the mask if present, otherwise return None.

        Returns
        -------
        Optional[np.ndarray]
            Boolean mask array, or None if mask.tif doesn't exist.
        """
        mask_path = self.path / "mask.tif"
        if not mask_path.exists():
            return None

        mask_data = tifffile.imread(str(mask_path))

        # Convert to boolean: assume non-zero means valid
        if mask_data.dtype == np.bool_:
            return mask_data
        elif mask_data.dtype == np.uint8:
            return mask_data > 0
        else:
            return mask_data != 0

    def read(
        self,
        *,
        load_mask: bool = True,
        validate: bool = True,
    ) -> Tifxyz:
        """Read the complete surface.

        Parameters
        ----------
        load_mask : bool
            If True, load mask.tif if present.
        validate : bool
            If True, validate the loaded data.

        Returns
        -------
        Tifxyz
            The loaded surface.
        """
        self._check_required_files()

        # Load metadata
        meta = self.read_metadata()

        # Load coordinate arrays
        x = self.read_coordinate("x")
        y = self.read_coordinate("y")
        z = self.read_coordinate("z")

        # Validate shapes
        if validate:
            if x.shape != y.shape or x.shape != z.shape:
                raise ValueError(
                    f"Coordinate array shapes must match: "
                    f"x={x.shape}, y={y.shape}, z={z.shape}"
                )

        # Load or derive mask
        mask = None
        if load_mask:
            mask = self.read_mask()
            if mask is not None and mask.shape != x.shape:
                # Mask might be at different resolution - derive from z > 0 instead
                # logger.warning(
                #     f"Mask shape {mask.shape} differs from coordinate shape {x.shape}. "
                #     "Mask will be derived from z > 0."
                # )
                mask = None

        # If no mask, derive from z > 0
        if mask is None:
            mask = (z > 0) & np.isfinite(z)

        # Mark invalid points (z <= 0) with sentinel value
        invalid = ~mask
        x[invalid] = -1.0
        y[invalid] = -1.0
        z[invalid] = -1.0

        return Tifxyz(
            _x=x,
            _y=y,
            _z=z,
            uuid=meta["uuid"],
            _scale=meta["scale"],
            bbox=meta["bbox"],
            area=meta["area"],
            extra=meta["extra"],
            _mask=mask,
            path=self.path,
        )

    def list_extra_channels(self) -> list[str]:
        """List additional TIFF files in the directory (excluding x, y, z, mask).

        Returns
        -------
        list[str]
            Names of extra channel files (without .tif extension).
        """
        excluded = {"x", "y", "z", "mask"}
        channels = []
        for tif_file in self.path.glob("*.tif"):
            name = tif_file.stem
            if name not in excluded:
                channels.append(name)
        return channels

    def read_extra_channel(self, name: str) -> np.ndarray:
        """Read an extra channel by name.

        Parameters
        ----------
        name : str
            The channel name (without .tif extension).

        Returns
        -------
        np.ndarray
            The channel data.
        """
        tif_path = self.path / f"{name}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"Channel file not found: {tif_path}")
        return tifffile.imread(str(tif_path))
