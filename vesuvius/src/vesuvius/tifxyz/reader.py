"""Reader for tifxyz format files."""

from __future__ import annotations

import json
import logging
import os
import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str((1 << 63) - 1))

import cv2
import numpy as np
import tifffile

from .types import Tifxyz

logger = logging.getLogger(__name__)

SUPPORTED_LABEL_EXTENSIONS = {".tif", ".png", ".jpg"}
RESERVED_IMAGE_FILENAMES = {"x.tif", "y.tif", "z.tif", "mask.tif"}


_BBOX_UNRESOLVED = object()


@dataclass(init=False)
class TifxyzInfo:
    """Lightweight metadata for a tifxyz segment without loading coordinates.

    Use this for filtering/listing segments before deciding which to fully load.

    ``bbox`` (and ``z_min`` / ``z_max``) are resolved lazily. For the common case the
    stored ``meta.json`` bbox is returned untouched and no coordinate grid is read.
    Only when the stored bbox carries the tifxyz ``-1`` missing-point marker
    (villa #1618) is the true extent recomputed from the valid points, and only on
    first access, so plain listing stays metadata-only (raised in review of #1731).
    """

    path: Path
    scale: Tuple[float, float]
    uuid: str
    stored_bbox: Optional[Tuple[float, float, float, float, float, float]] = None
    _bbox: Any = field(default=_BBOX_UNRESOLVED, init=False, repr=False, compare=False)

    def __init__(
        self,
        path: Path,
        scale: Tuple[float, float],
        bbox: Optional[Tuple[float, float, float, float, float, float]] = None,
        uuid: str = "",
        stored_bbox: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> None:
        """Keep the original ``(path, scale, bbox, uuid)`` signature.

        ``TifxyzInfo`` is public API. Making ``bbox`` a lazy property is only safe if the
        constructor still takes ``bbox`` in the same position and by the same name: callers
        writing ``TifxyzInfo(path, scale, bbox, uuid)`` or ``TifxyzInfo(..., bbox=b, uuid=u)``
        must keep working. ``stored_bbox`` is accepted too so the field name is usable
        directly, but ``bbox`` wins if both are given.
        """
        self.path = path
        self.scale = scale
        self.uuid = uuid
        self.stored_bbox = bbox if bbox is not None else stored_bbox
        self._bbox = _BBOX_UNRESOLVED

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """The bounding box, recomputed from valid points only if the stored one
        carries the ``-1`` marker. Resolved once, then cached. ``None`` means the
        segment has no usable bbox (no stored bbox, or no valid points)."""
        if self._bbox is _BBOX_UNRESOLVED:
            if self.stored_bbox is not None and _bbox_carries_missing_marker(self.stored_bbox):
                self._bbox = _bbox_from_valid_points(TifxyzReader(self.path))
                if self._bbox is not None:
                    logger.warning(
                        "%s: meta.json bbox contains the -1 missing-point marker; "
                        "using the extent of the valid coordinates instead",
                        self.path,
                    )
            else:
                self._bbox = self.stored_bbox
        return self._bbox

    @property
    def z_min(self) -> Optional[float]:
        """Minimum z from the (lazily resolved) bbox, or None if there is no bbox."""
        return self.bbox[2] if self.bbox is not None else None

    @property
    def z_max(self) -> Optional[float]:
        """Maximum z from the (lazily resolved) bbox, or None if there is no bbox."""
        return self.bbox[5] if self.bbox is not None else None

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


def _bbox_carries_missing_marker(bbox: Tuple[float, ...]) -> bool:
    """True if any bbox component is exactly the tifxyz missing-point marker."""
    return any(float(v) == -1.0 for v in bbox)


def _bbox_from_arrays(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """(x_min, y_min, z_min, x_max, y_max, z_max) over the valid points, or None if there are none."""
    if not np.any(valid):
        return None
    return (
        float(x[valid].min()), float(y[valid].min()), float(z[valid].min()),
        float(x[valid].max()), float(y[valid].max()), float(z[valid].max()),
    )


def _bbox_from_valid_points(
    reader: "TifxyzReader",
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Extent over the valid points, using the SAME validity as ``TifxyzReader.read``:
    ``mask.tif`` when present and shape-matching, otherwise ``z > 0`` and finite.

    Reading the bbox this way keeps ``list_tifxyz`` and ``read`` consistent for the
    same surface. The earlier version used only ``!= -1`` and so could disagree with
    the full reader when a ``mask.tif`` was present (raised in review of #1731).
    """
    x = reader.read_coordinate("x")
    y = reader.read_coordinate("y")
    z = reader.read_coordinate("z")
    mask = reader.read_mask()
    if mask is not None and mask.shape == x.shape:
        valid = mask
    else:
        valid = (z > 0) & np.isfinite(z)
    return _bbox_from_arrays(x, y, z, valid)


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
                uuid=meta["uuid"],
                stored_bbox=meta["bbox"],
            )

            # Filter by z_range if specified. Reading z_min/z_max resolves the bbox
            # lazily: for an ordinary bbox this just reads the stored value; only a
            # bbox carrying the -1 marker (#1618) triggers a coordinate scan, and only
            # here where it is actually needed. A segment with no usable bbox
            # (z_min/z_max None) is not filtered out.
            if z_range is not None:
                z_min, z_max = info.z_min, info.z_max
                if z_min is not None and z_max is not None:
                    range_z_min, range_z_max = z_range
                    if z_min > range_z_max or z_max < range_z_min:
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


def discover_labels(
    path: Union[str, Path],
    *,
    expected_shape: Optional[Tuple[int, int]] = None,
    validate_shapes: bool = False,
) -> List[Dict[str, Any]]:
    """Discover label images in a tifxyz directory without loading label pixels.

    By default this only inspects directory entries and returns path/name
    metadata. Set ``validate_shapes=True`` to read each candidate label image
    and populate shape/status fields.
    """
    return TifxyzReader(path).discover_labels(
        expected_shape=expected_shape,
        validate_shapes=validate_shapes,
    )


def read_tifxyz(
    path: Union[str, Path],
    *,
    load_mask: bool = True,
    validate: bool = True,
    volume_path: Optional[Union[str, Path]] = None,
    discover_label_shapes: bool = False,
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
    discover_label_shapes : bool
        If True, read candidate label images during discovery to validate their
        shapes. Default False keeps label discovery metadata-only.

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
    tifxyz = reader.read(
        load_mask=load_mask,
        validate=validate,
        discover_label_shapes=discover_label_shapes,
    )

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

    def _mmap_coordinate(self, tif_path: Path) -> np.ndarray:
        """Memory-map a coordinate TIFF without persisting in-place edits."""
        return tifffile.memmap(str(tif_path), mode="c")

    def _mmap_cache_path(self, tif_path: Path) -> Path:
        """Return a stable temp-cache path for a mmapable copy of a TIFF."""
        stat = tif_path.stat()
        cache_key = "|".join(
            (
                str(tif_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
            )
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        cache_dir = Path(tempfile.gettempdir()) / "vesuvius_tifxyz_mmap_cache"
        return cache_dir / f"{digest}_{tif_path.name}"

    def _write_mmapable_coordinate_cache(self, tif_path: Path) -> Path:
        """Write an uncompressed, untiled float32 temp-cache copy of a TIFF."""
        cache_path = self._mmap_cache_path(tif_path)
        if cache_path.exists():
            return cache_path

        data = tifffile.imread(str(tif_path)).astype(np.float32, copy=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            tifffile.imwrite(
                str(tmp_path),
                data,
                compression=None,
                photometric="minisblack",
            )
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return cache_path

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

        try:
            data = self._mmap_coordinate(tif_path)
        except ValueError as exc:
            logger.info(
                "Caching %s as uncompressed, untiled TIFF for mmap: %s",
                tif_path,
                exc,
            )
            data = self._mmap_coordinate(self._write_mmapable_coordinate_cache(tif_path))

        if data.dtype != np.float32:
            logger.info(
                "Caching %s as float32 TIFF for mmap; found dtype %s",
                tif_path,
                data.dtype,
            )
            data = self._mmap_coordinate(self._write_mmapable_coordinate_cache(tif_path))

        return data

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

    def _read_label_shape(self, path: Path) -> Tuple[int, int]:
        """Read label shape using OpenCV."""
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read label image: {path}")
        if image.ndim != 2:
            raise ValueError(f"Label must be 2D grayscale, got shape {image.shape}")
        return int(image.shape[0]), int(image.shape[1])

    @staticmethod
    def _label_name_from_filename(filename: str) -> str:
        stem = Path(filename).stem
        return stem.rsplit("_", 1)[-1]

    def discover_labels(
        self,
        expected_shape: Optional[Tuple[int, int]] = None,
        *,
        validate_shapes: bool = False,
    ) -> List[Dict[str, Any]]:
        """Discover label images in this segment directory.

        By default this does not read label image data. Shape validation is
        opt-in because large label files are expensive to load and most callers
        only need to know which labels are present.
        """
        if validate_shapes and expected_shape is None:
            raise ValueError("expected_shape is required when validate_shapes=True")

        labels: List[Dict[str, Any]] = []
        for path in self.path.iterdir():
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            filename_lower = path.name.lower()
            if suffix not in SUPPORTED_LABEL_EXTENSIONS:
                continue
            if filename_lower in RESERVED_IMAGE_FILENAMES:
                continue

            shape: Optional[Tuple[int, int]] = None
            matches_stored_shape: Optional[bool] = None
            error: Optional[str] = None
            if validate_shapes:
                try:
                    shape = self._read_label_shape(path)
                    matches_stored_shape = shape == expected_shape
                    if not matches_stored_shape:
                        error = (
                            f"Shape mismatch: expected {expected_shape}, got {shape}"
                        )
                except Exception as exc:
                    error = str(exc)

            labels.append(
                {
                    "index": -1,  # populated after sorting
                    "name": self._label_name_from_filename(path.name),
                    "filename": path.name,
                    "path": path,
                    "shape": shape,
                    "matches_stored_shape": matches_stored_shape,
                    "error": error,
                }
            )

        labels.sort(key=lambda item: str(item["filename"]).lower())
        for index, label in enumerate(labels):
            label["index"] = index
        return labels

    def read(
        self,
        *,
        load_mask: bool = True,
        validate: bool = True,
        discover_label_shapes: bool = False,
    ) -> Tifxyz:
        """Read the complete surface.

        Parameters
        ----------
        load_mask : bool
            If True, load mask.tif if present.
        validate : bool
            If True, validate the loaded data.
        discover_label_shapes : bool
            If True, read candidate label images during discovery to validate
            their shapes. Default False keeps label discovery metadata-only.

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

        labels = self.discover_labels(
            expected_shape=x.shape,
            validate_shapes=discover_label_shapes,
        )

        bbox = meta["bbox"]
        if bbox is not None and _bbox_carries_missing_marker(bbox):
            # Same defect list_tifxyz guards against (villa #1618); here the grids are
            # already in hand, so the extent of the valid points costs nothing extra.
            # Downstream, neural_tracing's _segment_z_bounds prefers seg.bbox when it
            # is not None, so a -1 floor here would pass every z_range there too.
            bbox = _bbox_from_arrays(x, y, z, mask)
            logger.warning(
                "%s: meta.json bbox contains the -1 missing-point marker; "
                "using the extent of the valid coordinates instead",
                self.path,
            )
        return Tifxyz(
            _x=x,
            _y=y,
            _z=z,
            uuid=meta["uuid"],
            _scale=meta["scale"],
            bbox=bbox,
            area=meta["area"],
            extra=meta["extra"],
            _mask=mask,
            path=self.path,
            _labels=labels,
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
