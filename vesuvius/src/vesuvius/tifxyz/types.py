"""Core data structure for tifxyz format."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class Tifxyz:
    """A 3D surface represented as a 2D grid of (x, y, z) coordinates.

    Provides lazy access to full-resolution coordinates. Data is stored
    internally at reduced resolution for efficiency, but all access is
    at full resolution - the downsampling is an implementation detail.

    Attributes
    ----------
    shape : Tuple[int, int]
        Full resolution grid dimensions (height, width).
    uuid : str
        Unique identifier for the surface.
    bbox : Optional[Tuple[float, ...]]
        Bounding box (x_min, y_min, z_min, x_max, y_max, z_max).

    Examples
    --------
    >>> surface = read_tifxyz("/path/to/segment")
    >>> surface.shape  # (84300, 87460) - full resolution
    >>> x, y, z, valid = surface[1000, 2000]  # get point
    >>> x, y, z, valid = surface[1000:1100, 2000:2100]  # get tile
    """

    # Internal storage (reduced resolution) - use underscore prefix
    _x: NDArray[np.float32]
    _y: NDArray[np.float32]
    _z: NDArray[np.float32]
    uuid: str = ""
    _scale: Tuple[float, float] = (1.0, 1.0)
    bbox: Optional[Tuple[float, float, float, float, float, float]] = None
    format: str = "tifxyz"
    surface_type: str = "seg"
    area: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    _mask: Optional[NDArray[np.bool_]] = None
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate shapes and ensure arrays are float32."""
        if self._x.shape != self._y.shape or self._x.shape != self._z.shape:
            raise ValueError(
                f"Coordinate array shapes must match: "
                f"x={self._x.shape}, y={self._y.shape}, z={self._z.shape}"
            )
        # Ensure float32
        if self._x.dtype != np.float32:
            object.__setattr__(self, "_x", self._x.astype(np.float32))
        if self._y.dtype != np.float32:
            object.__setattr__(self, "_y", self._y.astype(np.float32))
        if self._z.dtype != np.float32:
            object.__setattr__(self, "_z", self._z.astype(np.float32))
        # Set default uuid from path if not provided
        if not self.uuid and self.path:
            object.__setattr__(self, "uuid", self.path.name)

    def __getitem__(
        self, key
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Get coordinates at the specified location.

        Parameters
        ----------
        key : int, slice, or tuple of int/slice
            Index or slice specification.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            (x, y, z, valid) coordinates at the requested locations.

        Examples
        --------
        >>> surface = read_tifxyz("/path/to/segment")
        >>> surface.shape  # (84300, 87460)
        >>> x, y, z, valid = surface[2000, 4000]  # single point
        >>> x, y, z, valid = surface[1000:1100, 2000:2100]  # tile
        """
        from .upsampling import interpolate_at_points

        # Parse the key into row and column slices
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = slice(None)

        # Convert to slice objects
        h, w = self.shape
        row_slice = self._key_to_slice(row_key, h)
        col_slice = self._key_to_slice(col_key, w)

        # Generate indices for the requested region
        row_indices = np.arange(row_slice.start, row_slice.stop, row_slice.step or 1)
        col_indices = np.arange(col_slice.start, col_slice.stop, col_slice.step or 1)

        # Create meshgrid of target indices
        col_grid, row_grid = np.meshgrid(col_indices, row_indices)

        # Convert full-res indices to internal storage coordinates
        source_grid_y = row_grid.astype(np.float32) * self._scale[0]
        source_grid_x = col_grid.astype(np.float32) * self._scale[1]

        # Interpolate at internal storage coordinates
        return interpolate_at_points(
            self._x, self._y, self._z, self._valid_mask,
            source_grid_y, source_grid_x,
            scale=(1.0, 1.0),  # Already in internal coords
            order=1,
        )

    def _key_to_slice(self, key, size: int) -> slice:
        """Convert an index key to a slice object."""
        if isinstance(key, int):
            if key < 0:
                key = size + key
            return slice(key, key + 1, 1)
        elif isinstance(key, slice):
            start, stop, step = key.indices(size)
            return slice(start, stop, step)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def get_tile(
        self,
        row: int,
        col: int,
        height: int,
        width: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Get a tile at full resolution (lazy interpolation).

        Parameters
        ----------
        row : int
            Starting row in full resolution coordinates.
        col : int
            Starting column in full resolution coordinates.
        height : int
            Tile height.
        width : int
            Tile width.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            (x, y, z, valid) for the tile.
        """
        return self[row:row + height, col:col + width]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the full resolution grid shape (height, width)."""
        h, w = self._x.shape
        scale_y, scale_x = self._scale
        if scale_y == 0 or scale_x == 0:
            return (h, w)
        return (int(h / scale_y), int(w / scale_x))

    @property
    def _stored_shape(self) -> Tuple[int, int]:
        """Return the internal storage shape (implementation detail)."""
        return self._x.shape  # type: ignore[return-value]

    @property
    def _valid_mask(self) -> NDArray[np.bool_]:
        """Return internal validity mask."""
        if self._mask is not None:
            return self._mask
        return (self._z > 0) & np.isfinite(self._z)

    def compute_centroid(self) -> Tuple[float, float, float]:
        """Compute the centroid of all valid points.

        Returns
        -------
        Tuple[float, float, float]
            (x, y, z) centroid coordinates.
        """
        valid = self._valid_mask
        if not valid.any():
            return (0.0, 0.0, 0.0)
        return (
            float(self._x[valid].mean()),
            float(self._y[valid].mean()),
            float(self._z[valid].mean()),
        )

    def compute_normals(
        self,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Compute surface normals for the entire surface at stored resolution.

        This method computes normals over the entire surface at the internal
        stored resolution, which is efficient for whole-surface operations
        like analyzing normal direction or orientation.

        For tile-based access at full resolution, use get_normals() instead.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) - normalized normal components at stored resolution.
            Invalid points and boundary points have NaN normals.
        """
        x, y, z = self._x, self._y, self._z
        valid = self._valid_mask

        h, w = x.shape

        # Initialize output arrays
        nx = np.full((h, w), np.nan, dtype=np.float32)
        ny = np.full((h, w), np.nan, dtype=np.float32)
        nz = np.full((h, w), np.nan, dtype=np.float32)

        if h < 3 or w < 3:
            return nx, ny, nz

        # Create validity mask for interior points where all neighbors are valid
        interior_valid = (
            valid[1:-1, 1:-1] &
            valid[1:-1, :-2] &   # left
            valid[1:-1, 2:] &    # right
            valid[:-2, 1:-1] &   # top
            valid[2:, 1:-1]      # bottom
        )

        # Compute tangent vectors using central differences
        tx_x = x[1:-1, 2:] - x[1:-1, :-2]
        tx_y = y[1:-1, 2:] - y[1:-1, :-2]
        tx_z = z[1:-1, 2:] - z[1:-1, :-2]

        ty_x = x[2:, 1:-1] - x[:-2, 1:-1]
        ty_y = y[2:, 1:-1] - y[:-2, 1:-1]
        ty_z = z[2:, 1:-1] - z[:-2, 1:-1]

        # Normal = ty x tx (cross product)
        n_x = ty_y * tx_z - ty_z * tx_y
        n_y = ty_z * tx_x - ty_x * tx_z
        n_z = ty_x * tx_y - ty_y * tx_x

        # Normalize
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        norm = np.where(norm > 1e-10, norm, np.nan)

        n_x = n_x / norm
        n_y = n_y / norm
        n_z = n_z / norm

        # Apply validity mask
        n_x = np.where(interior_valid, n_x, np.nan)
        n_y = np.where(interior_valid, n_y, np.nan)
        n_z = np.where(interior_valid, n_z, np.nan)

        # Store in output arrays
        nx[1:-1, 1:-1] = n_x.astype(np.float32)
        ny[1:-1, 1:-1] = n_y.astype(np.float32)
        nz[1:-1, 1:-1] = n_z.astype(np.float32)

        return nx, ny, nz

    def get_normals(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Compute surface normals for a tile at full resolution.

        Normals are computed lazily - only the requested region is calculated.

        Parameters
        ----------
        row_start : int
            Starting row in full resolution coordinates.
        row_end : int
            Ending row (exclusive) in full resolution coordinates.
        col_start : int
            Starting column in full resolution coordinates.
        col_end : int
            Ending column (exclusive) in full resolution coordinates.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) - normalized normal components, shape (row_end - row_start, col_end - col_start).
            Invalid points and boundary points have NaN normals.
        """
        height = row_end - row_start
        width = col_end - col_start

        # Need to fetch a larger region to compute central differences
        # Expand by 1 pixel on each side
        h_full, w_full = self.shape
        r0 = max(0, row_start - 1)
        c0 = max(0, col_start - 1)
        r1 = min(h_full, row_end + 1)
        c1 = min(w_full, col_end + 1)

        # Get coordinates for expanded region
        x, y, z, valid = self[r0:r1, c0:c1]

        eh, ew = x.shape  # expanded height/width

        # Initialize output arrays
        nx = np.full((eh, ew), np.nan, dtype=np.float32)
        ny = np.full((eh, ew), np.nan, dtype=np.float32)
        nz = np.full((eh, ew), np.nan, dtype=np.float32)

        if eh < 3 or ew < 3:
            # Trim to requested size
            trim_r = row_start - r0
            trim_c = col_start - c0
            return nx[trim_r:trim_r+height, trim_c:trim_c+width], \
                   ny[trim_r:trim_r+height, trim_c:trim_c+width], \
                   nz[trim_r:trim_r+height, trim_c:trim_c+width]

        # Create validity mask for interior points where all neighbors are valid
        interior_valid = (
            valid[1:-1, 1:-1] &
            valid[1:-1, :-2] &   # left
            valid[1:-1, 2:] &    # right
            valid[:-2, 1:-1] &   # top
            valid[2:, 1:-1]      # bottom
        )

        # Compute tangent vectors using central differences
        tx_x = x[1:-1, 2:] - x[1:-1, :-2]
        tx_y = y[1:-1, 2:] - y[1:-1, :-2]
        tx_z = z[1:-1, 2:] - z[1:-1, :-2]

        ty_x = x[2:, 1:-1] - x[:-2, 1:-1]
        ty_y = y[2:, 1:-1] - y[:-2, 1:-1]
        ty_z = z[2:, 1:-1] - z[:-2, 1:-1]

        # Normal = ty x tx (cross product)
        n_x = ty_y * tx_z - ty_z * tx_y
        n_y = ty_z * tx_x - ty_x * tx_z
        n_z = ty_x * tx_y - ty_y * tx_x

        # Normalize
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        norm = np.where(norm > 1e-10, norm, np.nan)

        n_x = n_x / norm
        n_y = n_y / norm
        n_z = n_z / norm

        # Apply validity mask
        n_x = np.where(interior_valid, n_x, np.nan)
        n_y = np.where(interior_valid, n_y, np.nan)
        n_z = np.where(interior_valid, n_z, np.nan)

        # Store in output arrays (offset by 1 due to central differences)
        nx[1:-1, 1:-1] = n_x.astype(np.float32)
        ny[1:-1, 1:-1] = n_y.astype(np.float32)
        nz[1:-1, 1:-1] = n_z.astype(np.float32)

        # Trim to requested region
        trim_r = row_start - r0
        trim_c = col_start - c0
        return nx[trim_r:trim_r+height, trim_c:trim_c+width], \
               ny[trim_r:trim_r+height, trim_c:trim_c+width], \
               nz[trim_r:trim_r+height, trim_c:trim_c+width]

    def analyze_normal_direction(
        self,
        normals: Optional[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = None,
    ) -> Dict[str, Any]:
        """Analyze whether normals point inward (toward centroid) or outward.

        Parameters
        ----------
        normals : Optional[Tuple[NDArray, NDArray, NDArray]]
            Pre-computed normals (nx, ny, nz). If None, computes them.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'centroid': (x, y, z) centroid of surface
            - 'direction': 'inward', 'outward', or 'mixed'
            - 'inward_fraction': fraction of normals pointing inward
            - 'outward_fraction': fraction of normals pointing outward
            - 'consistent': True if all normals point same direction
            - 'dominant_direction': 'inward' or 'outward' (whichever is more common)
            - 'num_valid_normals': count of valid normals analyzed
        """
        if normals is None:
            nx, ny, nz = self.compute_normals()
        else:
            nx, ny, nz = normals

        # Get centroid
        centroid = self.compute_centroid()
        cx, cy, cz = centroid

        # Find points with valid normals
        valid_normals = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)

        if not valid_normals.any():
            return {
                'centroid': centroid,
                'direction': 'unknown',
                'inward_fraction': 0.0,
                'outward_fraction': 0.0,
                'consistent': False,
                'dominant_direction': 'unknown',
                'num_valid_normals': 0,
            }

        # For each point with a valid normal, compute vector from point to centroid
        # If dot(normal, to_centroid) > 0, normal points toward centroid (inward)
        # If dot(normal, to_centroid) < 0, normal points away from centroid (outward)

        # Vector from each point to centroid
        to_centroid_x = cx - self._x
        to_centroid_y = cy - self._y
        to_centroid_z = cz - self._z

        # Normalize the to_centroid vector
        to_centroid_norm = np.sqrt(
            to_centroid_x**2 + to_centroid_y**2 + to_centroid_z**2
        )
        to_centroid_norm = np.where(to_centroid_norm > 1e-10, to_centroid_norm, 1.0)
        to_centroid_x = to_centroid_x / to_centroid_norm
        to_centroid_y = to_centroid_y / to_centroid_norm
        to_centroid_z = to_centroid_z / to_centroid_norm

        # Dot product: normal . to_centroid
        dot = nx * to_centroid_x + ny * to_centroid_y + nz * to_centroid_z

        # Count inward vs outward
        inward = (dot > 0) & valid_normals
        outward = (dot < 0) & valid_normals

        num_inward = int(inward.sum())
        num_outward = int(outward.sum())
        num_valid = int(valid_normals.sum())

        inward_frac = num_inward / num_valid if num_valid > 0 else 0.0
        outward_frac = num_outward / num_valid if num_valid > 0 else 0.0

        # Determine overall direction
        # Consider "consistent" if >95% point the same way
        consistency_threshold = 0.95

        if inward_frac >= consistency_threshold:
            direction = 'inward'
            consistent = True
        elif outward_frac >= consistency_threshold:
            direction = 'outward'
            consistent = True
        else:
            direction = 'mixed'
            consistent = False

        dominant = 'inward' if num_inward >= num_outward else 'outward'

        return {
            'centroid': centroid,
            'direction': direction,
            'inward_fraction': inward_frac,
            'outward_fraction': outward_frac,
            'consistent': consistent,
            'dominant_direction': dominant,
            'num_valid_normals': num_valid,
        }

    def flip_normals(
        self,
        normals: Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Flip the direction of all normals (negate all components).

        Parameters
        ----------
        normals : Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) normal components.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            Flipped (-nx, -ny, -nz) normal components.
        """
        nx, ny, nz = normals
        return (-nx, -ny, -nz)

    def orient_normals(
        self,
        normals: Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
        direction: str = 'outward',
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Orient all normals to point in a specified direction.

        Flips individual normals that point the wrong way so that all
        normals consistently point either inward or outward relative
        to the surface centroid.

        Parameters
        ----------
        normals : Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) normal components.
        direction : str
            'inward' (toward centroid) or 'outward' (away from centroid).

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) with all normals pointing in the specified direction.
        """
        if direction not in ('inward', 'outward'):
            raise ValueError(f"direction must be 'inward' or 'outward', got {direction!r}")

        nx, ny, nz = normals
        nx = nx.copy()
        ny = ny.copy()
        nz = nz.copy()

        # Get centroid
        cx, cy, cz = self.compute_centroid()

        # Vector from each point to centroid
        to_centroid_x = cx - self._x
        to_centroid_y = cy - self._y
        to_centroid_z = cz - self._z

        # Dot product: normal . to_centroid
        # Positive = pointing toward centroid (inward)
        # Negative = pointing away from centroid (outward)
        dot = nx * to_centroid_x + ny * to_centroid_y + nz * to_centroid_z

        # Determine which normals need flipping
        if direction == 'outward':
            # Flip normals that point inward (dot > 0)
            flip_mask = dot > 0
        else:  # inward
            # Flip normals that point outward (dot < 0)
            flip_mask = dot < 0

        # Only flip valid normals
        valid_normals = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
        flip_mask = flip_mask & valid_normals

        # Flip the normals that need it
        nx[flip_mask] = -nx[flip_mask]
        ny[flip_mask] = -ny[flip_mask]
        nz[flip_mask] = -nz[flip_mask]

        return nx, ny, nz

    def get_normals_pointing_outward(
        self,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Compute normals and orient all to point outward (away from centroid).

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) - normalized normals all pointing away from centroid.
        """
        normals = self.compute_normals()
        return self.orient_normals(normals, direction='outward')

    def get_normals_pointing_inward(
        self,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Compute normals and orient all to point inward (toward centroid).

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) - normalized normals all pointing toward centroid.
        """
        normals = self.compute_normals()
        return self.orient_normals(normals, direction='inward')
