"""Core data structure for tifxyz format."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import zarr
    import torch

import numpy as np
from numpy.typing import NDArray

InterpolationMethod = Literal["linear", "bspline", "catmull_rom"]


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
    area: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    _mask: Optional[NDArray[np.bool_]] = None
    path: Optional[Path] = None
    interp_method: InterpolationMethod = "catmull_rom"
    resolution: Literal["stored", "full"] = "stored"
    volume: Optional["zarr.Array"] = field(default=None, repr=False)
    # Cache fields for quad properties
    _valid_quad_mask_cache: Optional[NDArray[np.bool_]] = field(default=None, repr=False)
    _quad_centers_cache: Optional[NDArray[np.float32]] = field(default=None, repr=False)
    _normals_cache: Optional[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = field(default=None, repr=False)
    _patches_cache: Optional[List[Tuple[Tuple[int, int, int, int], Tuple[float, ...]]]] = field(default=None, repr=False)

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
        # Validate resolution
        if self.resolution not in ("stored", "full"):
            raise ValueError(
                f"resolution must be 'stored' or 'full', got {self.resolution!r}"
            )

    def __getitem__(
        self, key
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Get coordinates at the specified location.

        In 'stored' mode (default), returns direct array access without interpolation.
        In 'full' mode, returns interpolated coordinates at full resolution.

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
        >>> x, y, z, valid = surface[100, 200]  # single point
        >>> x, y, z, valid = surface[100:200, 200:300]  # tile
        """
        # Stored mode: direct array access, no interpolation
        if self.resolution == "stored":
            x = self._x[key]
            y = self._y[key]
            z = self._z[key]
            valid = self._valid_mask[key]
            return x, y, z, valid

        # Full mode: interpolate to full resolution
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
            method=self.interp_method,
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

    @property
    def shape(self) -> Tuple[int, int]:
        """Return grid shape at current resolution.

        In 'stored' mode (default), returns the internal stored dimensions.
        In 'full' mode, returns the full resolution dimensions.
        """
        if self.resolution == "stored":
            return self._x.shape  # type: ignore[return-value]
        # Full resolution
        h, w = self._x.shape
        scale_y, scale_x = self._scale
        if scale_y == 0 or scale_x == 0:
            return (h, w)
        return (int(h / scale_y), int(w / scale_x))

    @property
    def full_resolution_shape(self) -> Tuple[int, int]:
        """Return grid shape at full resolution.

        Unlike `shape` which depends on the current resolution mode,
        this always returns the full resolution dimensions.
        """
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
        like analyzing normal direction or orientation. Results are cached.

        For tile-based access at full resolution, use get_normals() instead.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) - normalized normal components at stored resolution.
            Invalid points and boundary points have NaN normals.
        """
        if self._normals_cache is not None:
            return self._normals_cache

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

        result = (nx, ny, nz)
        object.__setattr__(self, "_normals_cache", result)
        return result

    def get_normals(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Compute surface normals for a tile at current resolution.

        In 'stored' mode (default), slices from cached compute_normals().
        In 'full' mode, computes normals at full resolution via interpolation.

        Parameters
        ----------
        row_start : int
            Starting row in current resolution coordinates.
        row_end : int
            Ending row (exclusive) in current resolution coordinates.
        col_start : int
            Starting column in current resolution coordinates.
        col_end : int
            Ending column (exclusive) in current resolution coordinates.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) - normalized normal components, shape (row_end - row_start, col_end - col_start).
            Invalid points and boundary points have NaN normals.
        """
        # Stored mode: slice from cached normals
        if self.resolution == "stored":
            nx, ny, nz = self.compute_normals()
            return (
                nx[row_start:row_end, col_start:col_end],
                ny[row_start:row_end, col_start:col_end],
                nz[row_start:row_end, col_start:col_end],
            )

        # Full mode: compute at full resolution
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
        normals: Optional[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Flip the direction of all normals (negate all components).

        Parameters
        ----------
        normals : Tuple[NDArray, NDArray, NDArray], optional
            (nx, ny, nz) normal components. If None, computes normals first.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            Flipped (-nx, -ny, -nz) normal components.
        """
        if normals is None:
            normals = self.compute_normals()
        nx, ny, nz = normals
        return (-nx, -ny, -nz)

    def orient_normals(
        self,
        direction: str = 'outward',
        normals: Optional[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Orient all normals to point in a specified direction.

        Flips individual normals that point the wrong way so that all
        normals consistently point either inward or outward relative
        to the surface centroid.

        Parameters
        ----------
        direction : str
            'inward' (toward centroid) or 'outward' (away from centroid).
        normals : Tuple[NDArray, NDArray, NDArray], optional
            (nx, ny, nz) normal components. If None, computes normals first.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            (nx, ny, nz) with all normals pointing in the specified direction.
        """
        if direction not in ('inward', 'outward'):
            raise ValueError(f"direction must be 'inward' or 'outward', got {direction!r}")

        if normals is None:
            normals = self.compute_normals()
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

    def get_zyxs(
        self, *, stored_resolution: Optional[bool] = None, as_tensor: bool = False
    ) -> Union[NDArray[np.float32], "torch.Tensor"]:
        """Get coordinates stacked as a (H, W, 3) array in z, y, x order.

        Returns coordinates stacked into a single array. Invalid points have
        value -1.

        This is useful for neural network training where coordinates are
        needed as a single tensor.

        Parameters
        ----------
        stored_resolution : bool, optional
            If True, return at internal stored resolution (faster, no interpolation).
            If False, return at full resolution (interpolated).
            If None (default), uses self.resolution setting.
        as_tensor : bool
            If True, return as a torch.Tensor instead of numpy array.
            Requires torch to be installed.

        Returns
        -------
        NDArray[np.float32] or torch.Tensor
            Array of shape (H, W, 3) where the last axis is (z, y, x).

        Examples
        --------
        >>> surface = read_tifxyz("/path/to/segment")
        >>> zyxs = surface.get_zyxs()  # Uses current resolution setting
        >>> zyxs_stored = surface.get_zyxs(stored_resolution=True)  # Force stored
        >>> zyxs_tensor = surface.get_zyxs(as_tensor=True)  # torch.Tensor
        """
        # Resolve resolution: explicit param > instance attribute
        if stored_resolution is None:
            use_stored = self.resolution == "stored"
        else:
            use_stored = stored_resolution

        if use_stored:
            result = np.stack([self._z, self._y, self._x], axis=-1)
        else:
            # Get full resolution coordinates via interpolation
            # Temporarily switch to full resolution for this operation
            old_resolution = self.resolution
            object.__setattr__(self, "resolution", "full")
            x, y, z, valid = self[:, :]
            object.__setattr__(self, "resolution", old_resolution)
            # Mark invalid points with -1
            x = np.where(valid, x, -1.0)
            y = np.where(valid, y, -1.0)
            z = np.where(valid, z, -1.0)
            result = np.stack([z, y, x], axis=-1)

        if as_tensor:
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "torch is required for as_tensor=True. "
                    "Install with: pip install vesuvius[models]"
                )
            return torch.from_numpy(result)

        return result

    def get_scale_tuple(self) -> Tuple[float, float]:
        """Get the scale as a (scale_y, scale_x) tuple.

        Returns
        -------
        Tuple[float, float]
            Scale factors (scale_y, scale_x).
        """
        return self._scale

    def use_stored_resolution(self) -> "Tifxyz":
        """Set resolution to 'stored' and return self for chaining.

        In stored mode, shape, indexing, and get_normals all
        operate at the internal stored resolution without interpolation.

        Returns
        -------
        Tifxyz
            Self, for method chaining.

        Examples
        --------
        >>> surface.use_stored_resolution()
        >>> print(surface.shape)  # Stored resolution dimensions
        >>> x, y, z, valid = surface[0:100, 0:100]  # Direct array access
        """
        object.__setattr__(self, "resolution", "stored")
        return self

    def use_full_resolution(self) -> "Tifxyz":
        """Set resolution to 'full' and return self for chaining.

        In full mode, shape, indexing, and get_normals all
        operate at full resolution with interpolation.

        Returns
        -------
        Tifxyz
            Self, for method chaining.

        Examples
        --------
        >>> surface.use_full_resolution()
        >>> print(surface.shape)  # Full resolution dimensions
        >>> x, y, z, valid = surface[0:100, 0:100]  # Interpolated access
        """
        object.__setattr__(self, "resolution", "full")
        return self

    @property
    def valid_quad_mask(self) -> NDArray[np.bool_]:
        """Boolean mask where [i,j] is True if all 4 corners of quad at (i,j) are valid.

        A quad at position (i,j) consists of vertices at:
        (i,j), (i+1,j), (i,j+1), (i+1,j+1)

        Returns
        -------
        NDArray[np.bool_]
            Mask of shape (H-1, W-1) at stored resolution.
        """
        if self._valid_quad_mask_cache is None:
            valid = self._valid_mask
            object.__setattr__(
                self,
                "_valid_quad_mask_cache",
                valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:],
            )
        return self._valid_quad_mask_cache

    @property
    def valid_quad_indices(self) -> NDArray[np.int64]:
        """Indices of valid quads as (N, 2) array.

        Returns
        -------
        NDArray[np.int64]
            Array of shape (N, 2) where each row is (row, col) of a valid quad.
        """
        return np.stack(np.where(self.valid_quad_mask), axis=-1)

    @property
    def valid_vertex_mask(self) -> NDArray[np.bool_]:
        """Boolean mask for valid vertices (alias for _valid_mask).

        Returns
        -------
        NDArray[np.bool_]
            Mask of shape (H, W) at stored resolution.
        """
        return self._valid_mask

    @property
    def quad_area(self) -> float:
        """Surface area computed from valid quad count.

        Returns the number of valid quads divided by the scale factors,
        giving area in full-resolution units.

        Returns
        -------
        float
            Surface area in full-resolution coordinate units.
        """
        scale_y, scale_x = self._scale
        return float(self.valid_quad_mask.sum() / (scale_y * scale_x))

    @property
    def quad_centers(self) -> NDArray[np.float32]:
        """Centers of quads at stored resolution.

        Returns
        -------
        NDArray[np.float32]
            Array of shape (H-1, W-1, 3) with (z, y, x) centers.
            Invalid quads have value -1.
        """
        if self._quad_centers_cache is None:
            zyxs = self.get_zyxs(stored_resolution=True)
            centers = 0.5 * (zyxs[1:, 1:] + zyxs[:-1, :-1])
            mask = self.valid_quad_mask[..., np.newaxis]
            object.__setattr__(
                self,
                "_quad_centers_cache",
                np.where(mask, centers, -1.0).astype(np.float32),
            )
        return self._quad_centers_cache

    def retarget(self, factor: float) -> "Tifxyz":
        """Return new Tifxyz with coordinates scaled by factor.

        Used to adapt coordinates to a downsampled or upsampled volume.

        Parameters
        ----------
        factor : float
            Scale factor. Values >1 mean target volume is downsampled
            (coordinates become smaller). Values <1 mean upsampled.

        Returns
        -------
        Tifxyz
            New instance with scaled coordinates and updated bbox.
            If volume is an OME-zarr group, selects appropriate level.
        """
        valid = self._valid_mask
        new_x = np.where(valid, self._x / factor, -1.0).astype(np.float32)
        new_y = np.where(valid, self._y / factor, -1.0).astype(np.float32)
        new_z = np.where(valid, self._z / factor, -1.0).astype(np.float32)

        # Compute new bbox from scaled coordinates
        new_bbox = None
        if valid.any():
            new_bbox = (
                float(new_x[valid].min()),
                float(new_y[valid].min()),
                float(new_z[valid].min()),
                float(new_x[valid].max()),
                float(new_y[valid].max()),
                float(new_z[valid].max()),
            )

        # Handle OME-zarr: select appropriate resolution level
        new_volume = self._get_volume_level(factor) if self.volume is not None else None

        return Tifxyz(
            _x=new_x,
            _y=new_y,
            _z=new_z,
            uuid=self.uuid,
            _scale=(self._scale[0] * factor, self._scale[1] * factor),
            bbox=new_bbox,
            area=None,  # Computed lazily via quad_area
            extra=dict(self.extra),
            _mask=valid.copy(),
            path=self.path,
            interp_method=self.interp_method,
            resolution=self.resolution,
            volume=new_volume,
        )

    def _get_volume_level(self, factor: float) -> Optional["zarr.Array"]:
        """Get the appropriate OME-zarr resolution level for the given factor.

        For OME-zarr groups with multiple resolution levels (0, 1, 2, ...),
        selects the level closest to the requested downsampling factor.

        Parameters
        ----------
        factor : float
            The downsampling factor.

        Returns
        -------
        zarr.Array or None
            The appropriate resolution level array.
        """
        import zarr

        vol = self.volume
        if vol is None:
            return None

        # If it's a zarr Group (OME-zarr), find the right level
        if isinstance(vol, zarr.Group):
            # OME-zarr levels are typically named "0", "1", "2", etc.
            # Level 0 is full resolution, higher levels are downsampled by 2x each
            level = int(np.log2(factor)) if factor >= 1 else 0
            level = max(0, level)

            # Find available levels
            available = sorted([k for k in vol.keys() if k.isdigit()], key=int)
            if not available:
                return vol  # Not an OME-zarr, return as-is

            # Clamp to available range
            level = min(level, int(available[-1]))
            return vol[str(level)]

        # Already an array, return as-is
        return vol

    def smooth_rows_catmull_rom(
        self,
        *,
        stored_resolution: Optional[bool] = None,
    ) -> NDArray[np.float32]:
        """Apply 1D Catmull-Rom smoothing to each row independently.

        For each row, collects valid points in column order (skipping invalid
        points where z <= 0 or not finite), then applies 1D Catmull-Rom
        smoothing to the (x, y, z) coordinates independently.

        The smoothing uses Catmull-Rom weights at t=0.5 as a 4-tap filter:
        [-1/16, 9/16, 9/16, -1/16], which provides local smoothing while
        preserving the general shape of the curve. Edge points are handled
        by linearly extrapolating phantom control points beyond boundaries.

        Parameters
        ----------
        stored_resolution : bool, optional
            If True, use internal stored resolution (fast, no interpolation).
            If False, use full resolution (interpolated).
            If None (default), uses current self.resolution setting.

        Returns
        -------
        NDArray[np.float32]
            Array of shape (H, W, 3) with smoothed coordinates in [z, y, x] order,
            same format as get_zyxs(). Invalid points have value -1.
            Rows with < 2 valid points are left unchanged (still -1 for invalid).

        Notes
        -----
        - Invalid points (z <= 0 or not finite) are skipped during smoothing
        - Valid points are replaced with their smoothed values in-place
        - Output format matches get_zyxs() for consistency

        Examples
        --------
        >>> surface = read_tifxyz("/path/to/segment")
        >>> smoothed = surface.smooth_rows_catmull_rom()
        >>> smoothed.shape  # (H, W, 3)
        >>>
        >>> # Force full resolution
        >>> smoothed_full = surface.smooth_rows_catmull_rom(stored_resolution=False)
        """
        from .upsampling import catmull_rom_smooth_1d

        # Resolve resolution: explicit param > instance attribute
        if stored_resolution is None:
            use_stored = self.resolution == "stored"
        else:
            use_stored = stored_resolution

        if use_stored:
            x_data = self._x.copy()
            y_data = self._y.copy()
            z_data = self._z.copy()
            valid = self._valid_mask
        else:
            # Get full resolution coordinates via interpolation
            x_data, y_data, z_data, valid = self[:, :]
            x_data = x_data.copy()
            y_data = y_data.copy()
            z_data = z_data.copy()

        h, w = x_data.shape

        for row in range(h):
            # Get column indices of valid points in this row
            valid_cols = np.where(valid[row])[0]

            if len(valid_cols) < 2:
                # Not enough points for smoothing, leave as-is
                continue

            # Extract coordinates for valid points
            x_valid = x_data[row, valid_cols]
            y_valid = y_data[row, valid_cols]
            z_valid = z_data[row, valid_cols]

            # Apply 1D Catmull-Rom smoothing
            x_smooth, y_smooth, z_smooth = catmull_rom_smooth_1d(x_valid, y_valid, z_valid)

            # Write smoothed values back to the valid positions
            if len(x_smooth) > 0:
                x_data[row, valid_cols] = x_smooth
                y_data[row, valid_cols] = y_smooth
                z_data[row, valid_cols] = z_smooth

        # Stack as (H, W, 3) in [z, y, x] order like get_zyxs()
        return np.stack([z_data, y_data, x_data], axis=-1)

    def get_patches_3d(
        self,
        target_size: Tuple[int, int, int],
        *,
        overlap_fraction: float = 0.25,
        coarse_multiplier: float = 2.0,
        num_calibration_samples: int = 200,
        min_new_coverage: float = 0.5,
        verbose: bool = False,
        force_recompute: bool = False,
    ) -> List[Tuple[Tuple[int, int, int, int], Tuple[float, ...]]]:
        """Find hierarchical patches that cover the target 3D volume size.

        Uses multipass hierarchical tiling to find 2D patches that, when
        sampled with 3D context, will cover the specified volume size.
        Results are cached in memory and persisted to JSON in the tifxyz folder.

        Parameters
        ----------
        target_size : Tuple[int, int, int]
            Target 3D volume size (depth, height, width) in voxels.
        overlap_fraction : float
            Fraction of overlap between adjacent tiles (0.0-1.0, default 0.25).
        coarse_multiplier : float
            Multiplier for coarse tile size (default 2.0).
        num_calibration_samples : int
            Number of samples for global calibration (default 200).
        min_new_coverage : float
            For multipass: min fraction of new pixels required (default 0.5).
        verbose : bool
            Print progress information (default False).
        force_recompute : bool
            If True, recompute even if cached (default False).

        Returns
        -------
        List[Tuple[Tuple[int, int, int, int], Tuple[float, ...]]]
            List of (bbox_2d, bbox_3d) tuples where:
            - bbox_2d: (r_min, r_max, c_min, c_max) in 2D grid coordinates
            - bbox_3d: (z_min, z_max, y_min, y_max, x_min, x_max) in 3D volume coordinates
        """
        import json

        cache_key = f"{target_size[0]},{target_size[1]},{target_size[2]},{self._scale[0]},{self._scale[1]}"

        # Check in-memory cache first
        if self._patches_cache is not None and not force_recompute:
            return self._patches_cache

        # Try to load from disk cache
        cache_file = self.path / "patches_cache.json" if self.path else None
        if cache_file and cache_file.exists() and not force_recompute:
            try:
                with open(cache_file, "r") as f:
                    disk_cache = json.load(f)
                if cache_key in disk_cache:
                    # Convert lists back to tuples
                    patches = [
                        (tuple(bbox_2d), tuple(bbox_3d))
                        for bbox_2d, bbox_3d in disk_cache[cache_key]
                    ]
                    object.__setattr__(self, "_patches_cache", patches)
                    if verbose:
                        print(f"Loaded {len(patches)} patches from {cache_file}")
                    return patches
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Cache file corrupted or wrong format, recompute

        # Compute patches
        from .hierarchical_tiling import multipass_hierarchical_tiling

        patches = multipass_hierarchical_tiling(
            self,
            target_size=target_size,
            num_calibration_samples=num_calibration_samples,
            coarse_multiplier=coarse_multiplier,
            overlap_fraction=overlap_fraction,
            min_new_coverage=min_new_coverage,
            verbose=verbose,
        )

        # Cache in memory
        object.__setattr__(self, "_patches_cache", patches)

        # Persist to disk
        if cache_file:
            # Load existing cache or create new
            disk_cache = {}
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        disk_cache = json.load(f)
                except (json.JSONDecodeError, TypeError):
                    disk_cache = {}

            # Convert tuples to lists for JSON serialization
            disk_cache[cache_key] = [
                [list(bbox_2d), list(bbox_3d)]
                for bbox_2d, bbox_3d in patches
            ]

            with open(cache_file, "w") as f:
                json.dump(disk_cache, f, indent=2)

            if verbose:
                print(f"Saved {len(patches)} patches to {cache_file}")

        return patches
