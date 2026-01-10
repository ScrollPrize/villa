"""Writer for tifxyz format files."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile

from .types import Tifxyz
from .upsampling import compute_grid_bounds

logger = logging.getLogger(__name__)


def write_tifxyz(
    path: Union[str, Path],
    surface: Tifxyz,
    *,
    compression: str = "lzw",
    tile_size: int = 1024,
    write_mask: bool = True,
    overwrite: bool = False,
) -> Path:
    """Write a Tifxyz object to a tifxyz directory.

    Parameters
    ----------
    path : Union[str, Path]
        Output directory path. Will be created if it doesn't exist.
    surface : Tifxyz
        Surface data to write.
    compression : str
        TIFF compression method. Default 'lzw'.
    tile_size : int
        TIFF tile size. Default 1024 to match C++ implementation.
    write_mask : bool
        If True, write mask.tif. Default True.
    overwrite : bool
        If True, overwrite existing files. Default False.

    Returns
    -------
    Path
        Path to the written directory.

    Raises
    ------
    FileExistsError
        If directory exists and overwrite is False.
    """
    writer = TifxyzWriter(
        path,
        compression=compression,
        tile_size=tile_size,
        overwrite=overwrite,
    )
    return writer.write(surface, write_mask=write_mask)


class TifxyzWriter:
    """Class-based writer for tifxyz directories.

    Provides more control over the writing process.

    Parameters
    ----------
    path : Union[str, Path]
        Output directory path.
    compression : str
        TIFF compression method. Default 'lzw'.
    tile_size : int
        TIFF tile size. Default 1024.
    overwrite : bool
        If True, overwrite existing files. Default False.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        compression: str = "lzw",
        tile_size: int = 1024,
        overwrite: bool = False,
    ) -> None:
        """Initialize writer."""
        self.path = Path(path)
        self.compression = compression
        self.tile_size = tile_size
        self.overwrite = overwrite

    def _ensure_directory(self) -> None:
        """Ensure output directory exists and handle overwrite logic."""
        if self.path.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Directory already exists: {self.path}. "
                    "Use overwrite=True to overwrite."
                )
            # Don't remove yet - we'll use atomic write
        else:
            self.path.mkdir(parents=True, exist_ok=True)

    def write_tiff(
        self,
        filename: str,
        data: np.ndarray,
    ) -> None:
        """Write a single TIFF file with proper format.

        Parameters
        ----------
        filename : str
            Output filename (e.g., 'x.tif').
        data : np.ndarray
            Data to write.
        """
        output_path = self.path / filename
        h, w = data.shape[:2]

        # Determine tile size (can't be larger than image)
        # tifffile requires tiles to be multiples of 16
        tile_h = min(self.tile_size, h)
        tile_w = min(self.tile_size, w)

        # Round down to multiple of 16, minimum 16
        tile_h = max(16, (tile_h // 16) * 16)
        tile_w = max(16, (tile_w // 16) * 16)

        # For small images, use strip-based writing instead of tiles
        use_tiles = h >= 16 and w >= 16

        if use_tiles:
            tifffile.imwrite(
                str(output_path),
                data.astype(np.float32),
                tile=(tile_h, tile_w),
                compression=self.compression,
                photometric="minisblack",
            )
        else:
            # Fall back to strip-based for very small images
            tifffile.imwrite(
                str(output_path),
                data.astype(np.float32),
                compression=self.compression,
                photometric="minisblack",
            )

    def write_metadata(
        self,
        surface: Tifxyz,
        bbox: Optional[tuple] = None,
    ) -> None:
        """Write meta.json.

        Parameters
        ----------
        surface : Tifxyz
            Surface to get metadata from.
        bbox : Optional[tuple]
            Override bounding box. If None, uses surface.bbox.
        """
        # Build metadata dict in C++ format
        meta_dict = {
            "uuid": surface.uuid,
            # C++ format: scale is [x_scale, y_scale]
            # Our internal format is (scale_y, scale_x)
            "scale": [surface._scale[1], surface._scale[0]],
        }

        # Add bbox
        use_bbox = bbox if bbox is not None else surface.bbox
        if use_bbox is not None:
            # bbox format: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            meta_dict["bbox"] = [
                [use_bbox[0], use_bbox[1], use_bbox[2]],
                [use_bbox[3], use_bbox[4], use_bbox[5]],
            ]

        # Add optional area
        if surface.area is not None:
            meta_dict["area"] = surface.area

        # Add extra fields
        meta_dict.update(surface.extra)

        # Write JSON
        output_path = self.path / "meta.json"
        with open(output_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

    def write_mask(self, mask: np.ndarray) -> None:
        """Write mask.tif.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask array.
        """
        output_path = self.path / "mask.tif"
        h, w = mask.shape[:2]

        # Determine tile size
        tile_h = min(self.tile_size, h)
        tile_w = min(self.tile_size, w)
        tile_h = max(16, (tile_h // 16) * 16)
        tile_w = max(16, (tile_w // 16) * 16)

        # Convert to uint8 (255 = valid, 0 = invalid)
        mask_uint8 = (mask.astype(np.uint8) * 255)

        use_tiles = h >= 16 and w >= 16

        if use_tiles:
            tifffile.imwrite(
                str(output_path),
                mask_uint8,
                tile=(tile_h, tile_w),
                compression=self.compression,
                photometric="minisblack",
            )
        else:
            tifffile.imwrite(
                str(output_path),
                mask_uint8,
                compression=self.compression,
                photometric="minisblack",
            )

    def write(
        self,
        surface: Tifxyz,
        *,
        write_mask: bool = True,
    ) -> Path:
        """Write complete surface.

        Uses atomic write pattern: writes to temp directory first,
        then moves to final location.

        Parameters
        ----------
        surface : Tifxyz
            Surface to write.
        write_mask : bool
            If True, write mask.tif.

        Returns
        -------
        Path
            Path to the written directory.
        """
        # Create temp directory for atomic write
        temp_dir = None
        use_atomic = self.path.exists() and self.overwrite

        if use_atomic:
            temp_dir = tempfile.mkdtemp(
                prefix=".tifxyz_tmp_",
                dir=self.path.parent,
            )
            original_path = self.path
            self.path = Path(temp_dir)
        else:
            self._ensure_directory()

        try:
            # Write coordinate TIFFs
            self.write_tiff("x.tif", surface._x)
            self.write_tiff("y.tif", surface._y)
            self.write_tiff("z.tif", surface._z)

            # Compute bounding box from valid points
            bbox = compute_grid_bounds(
                surface._x, surface._y, surface._z, surface._valid_mask
            )

            # Write metadata
            self.write_metadata(surface, bbox=bbox)

            # Write mask
            if write_mask:
                self.write_mask(surface._valid_mask)

            # Atomic move if using temp directory
            if use_atomic and temp_dir:
                # Remove old directory and rename temp to final
                shutil.rmtree(original_path)
                shutil.move(str(self.path), str(original_path))
                self.path = original_path

            logger.info(f"Wrote tifxyz surface to {self.path}")
            return self.path

        except Exception:
            # Clean up temp directory on failure
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
            raise

    def write_extra_channel(
        self,
        name: str,
        data: np.ndarray,
    ) -> None:
        """Write an extra channel.

        Parameters
        ----------
        name : str
            Channel name (will create {name}.tif).
        data : np.ndarray
            Channel data.
        """
        self._ensure_directory()
        output_path = self.path / f"{name}.tif"
        h, w = data.shape[:2]
        tile_h = min(self.tile_size, h)
        tile_w = min(self.tile_size, w)

        tifffile.imwrite(
            str(output_path),
            data,
            tile=(tile_h, tile_w),
            compression=self.compression,
            photometric="minisblack",
        )
