"""A plane slicer which extracts 2d planes from 3d volumes along any axes using arbitrary tilts/rotations."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vesuvius.utils.utils import pad_or_crop_2d, pad_or_crop_3d

from ..find_valid_patches import bounding_box_volume, compute_bounding_box_3d
from ..mesh.handles import MeshHandle

logger = logging.getLogger(__name__)


EPSILON = 1e-6


@dataclass(frozen=True)
class PlaneSliceConfig:
    """Configuration controlling plane slicing behaviour."""

    sample_planes: Sequence[str]
    plane_weights: Dict[str, float]
    plane_patch_sizes: Dict[str, Sequence[int]]
    primary_plane: Optional[str]
    min_labeled_ratio: float
    min_bbox_percent: float
    allow_unlabeled: bool
    random_rotation_planes: Dict[str, object]
    random_tilt_planes: Dict[str, object]
    label_interpolation: Dict[str, object]
    save_plane_masks: bool
    plane_mask_mode: str


@dataclass
class PlaneSliceVolume:
    """Container describing an image volume and associated labels."""

    index: int
    name: str
    image: object
    labels: Dict[str, Optional[object]]
    meshes: Mapping[str, MeshHandle] = field(default_factory=dict)
    _image_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _label_cache: Dict[str, Optional[np.ndarray]] = field(default_factory=dict, init=False, repr=False)


@dataclass
class PlaneSlicePatch:
    """Metadata describing a slice patch."""

    volume_index: int
    volume_name: str
    plane: str
    slice_index: int
    position: Tuple[int, int]
    patch_size: Tuple[int, int]


@dataclass
class PlaneSliceResult:
    """Image/label payload extracted for a given plane patch."""

    image: np.ndarray
    labels: Dict[str, np.ndarray]
    is_unlabeled: bool
    plane_mask: Optional[np.ndarray]
    patch_info: Dict[str, object]
    meshes: Dict[str, Dict[str, object]] = field(default_factory=dict)


class PlaneSlicer:
    """Utility that enumerates and extracts plane-aligned patches."""

    def __init__(
        self,
        config: PlaneSliceConfig,
        target_names: Sequence[str],
        normalizer=None,
    ) -> None:
        if not config.sample_planes:
            raise ValueError("PlaneSlicer requires at least one plane in sample_planes")
        self.config = config
        self.target_names = list(target_names)
        if not self.target_names:
            raise ValueError("PlaneSlicer requires at least one target name")
        self.normalizer = normalizer
        self._volumes: List[PlaneSliceVolume] = []
        self._patches: List[PlaneSlicePatch] = []
        self._weights: List[float] = []
        self._mask_volume_shape: Optional[Tuple[int, int, int]] = None

    # Registration ---------------------------------------------------------------------------------

    def register_volume(self, volume: PlaneSliceVolume) -> None:
        """Register a volume for subsequent slicing."""

        self._validate_volume(volume)
        self._volumes.append(volume)
        if self.config.save_plane_masks and self.config.plane_mask_mode == "volume":
            self._update_mask_volume_shape(volume)

    def register_volumes(self, volumes: Iterable[PlaneSliceVolume]) -> None:
        for volume in volumes:
            self.register_volume(volume)

    def set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer

    def _get_volume_image(self, volume: PlaneSliceVolume) -> np.ndarray:
        if volume._image_cache is None:
            volume._image_cache = self._to_array(volume.image)
        return volume._image_cache

    def _get_volume_label(self, volume: PlaneSliceVolume, target: str) -> Optional[np.ndarray]:
        if target not in volume._label_cache:
            source = volume.labels.get(target)
            if source is None:
                volume._label_cache[target] = None
            else:
                volume._label_cache[target] = self._to_array(source)
        return volume._label_cache[target]

    @staticmethod
    def _to_array(source) -> np.ndarray:
        if hasattr(source, 'read'):
            return np.asarray(source.read())
        return np.asarray(source)

    # Index construction ----------------------------------------------------------------------------

    def build_index(self, validate: bool) -> Tuple[List[PlaneSlicePatch], List[float]]:
        if not self._volumes:
            raise RuntimeError("PlaneSlicer.build_index called before any volumes were registered")

        plane_patches: Dict[str, List[PlaneSlicePatch]] = {axis: [] for axis in self.config.sample_planes}

        for volume in self._volumes:
            collected = self._collect_patches_for_volume(volume, validate=validate)
            for axis, patches in collected.items():
                if patches:
                    plane_patches.setdefault(axis, []).extend(patches)

        patches, weights = self._finalize_patch_index(plane_patches)
        self._patches = patches
        self._weights = weights
        return patches, weights

    @property
    def patches(self) -> List[PlaneSlicePatch]:
        return list(self._patches)

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    # Extraction ------------------------------------------------------------------------------------

    def extract(self, patch: PlaneSlicePatch) -> PlaneSliceResult:
        if self.normalizer is None:
            logger.debug("PlaneSlicer.extract invoked without a normalizer; returning unnormalized data")

        volume = self._get_volume(patch.volume_index)
        return self._extract_slice_patch(volume, patch)

    # Internal helpers ------------------------------------------------------------------------------

    def _validate_volume(self, volume: PlaneSliceVolume) -> None:
        if volume.index != len(self._volumes):
            raise ValueError(
                f"PlaneSliceVolume index {volume.index} does not match expected {len(self._volumes)}"
            )
        missing_targets = [t for t in self.target_names if t not in volume.labels]
        if missing_targets:
            raise ValueError(f"PlaneSliceVolume missing labels for targets: {missing_targets}")

    def _update_mask_volume_shape(self, volume: PlaneSliceVolume) -> None:
        spatial_shape = self._extract_spatial_shape(volume.image)
        if spatial_shape is None:
            raise ValueError(
                f"Unable to determine spatial shape for volume '{volume.name}' while preparing masks"
            )
        shape = tuple(int(v) for v in spatial_shape)
        if self._mask_volume_shape is None:
            self._mask_volume_shape = shape
            return
        self._mask_volume_shape = tuple(max(a, b) for a, b in zip(self._mask_volume_shape, shape))

    def _collect_patches_for_volume(
        self,
        volume: PlaneSliceVolume,
        *,
        validate: bool,
    ) -> Dict[str, List[PlaneSlicePatch]]:
        spatial_shape = self._extract_spatial_shape(volume.image)
        if spatial_shape is None or len(spatial_shape) != 3:
            raise ValueError(f"Volume '{volume.name}' must provide a 3D spatial shape")

        perform_validation = bool(
            validate and any(label is not None for label in volume.labels.values())
        )

        if validate and not perform_validation and not self.config.allow_unlabeled:
            logger.debug(
                "Skipping unlabeled volume '%s' because validation is required and allow_unlabeled=False",
                volume.name,
            )
            return {axis: [] for axis in self.config.sample_planes}

        collected: Dict[str, List[PlaneSlicePatch]] = {axis: [] for axis in self.config.sample_planes}
        for axis in self.config.sample_planes:
            patch_size = self._resolve_patch_size(axis)
            plane_shape = self._plane_shape_from_spatial(spatial_shape, axis)
            if plane_shape is None:
                raise ValueError(f"Unable to compute plane shape for axis '{axis}' in volume '{volume.name}'")

            positions = self._iter_plane_positions(plane_shape, patch_size)
            if not positions:
                continue

            axis_index = {"z": 0, "y": 1, "x": 2}[axis]
            num_slices = spatial_shape[axis_index]

            label_reference = self._resolve_label_reference(volume)

            for slice_idx in range(num_slices):
                mask_slice = None
                if perform_validation and label_reference is not None:
                    mask_slice = self._extract_label_slice_mask(label_reference, axis, slice_idx)
                    if mask_slice is None or not mask_slice.any():
                        continue

                for pos0, pos1 in positions:
                    if perform_validation and mask_slice is not None:
                        mask_patch = mask_slice[pos0 : pos0 + patch_size[0], pos1 : pos1 + patch_size[1]]
                        if mask_patch.shape != tuple(patch_size):
                            mask_patch = pad_or_crop_2d(mask_patch, tuple(patch_size))
                        if not self._mask_satisfies_thresholds(mask_patch):
                            continue

                    collected[axis].append(
                        PlaneSlicePatch(
                            volume_index=volume.index,
                            volume_name=volume.name,
                            plane=axis,
                            slice_index=slice_idx,
                            position=(int(pos0), int(pos1)),
                            patch_size=(int(patch_size[0]), int(patch_size[1])),
                        )
                    )

        return collected

    def _finalize_patch_index(
        self, plane_patches: Dict[str, List[PlaneSlicePatch]]
    ) -> Tuple[List[PlaneSlicePatch], List[float]]:
        available_axes = {axis: patches for axis, patches in plane_patches.items() if patches}
        if not available_axes:
            raise RuntimeError("Plane slicing produced zero patches across all axes")

        patches: List[PlaneSlicePatch] = []
        axis_counts: Dict[str, int] = {}
        for axis, patch_list in available_axes.items():
            for patch in patch_list:
                patches.append(patch)
                axis_counts[axis] = axis_counts.get(axis, 0) + 1

        weights: List[float] = []
        for patch in patches:
            axis_weight = float(self.config.plane_weights.get(patch.plane, 1.0))
            count = axis_counts.get(patch.plane, 0)
            weight_per_patch = axis_weight / count if count > 0 and axis_weight > 0 else 0.0
            weights.append(weight_per_patch)

        return patches, weights

    def _resolve_patch_size(self, axis: str) -> Tuple[int, int]:
        size = self.config.plane_patch_sizes.get(axis)
        if size is None:
            raise KeyError(f"Missing patch size for plane '{axis}'")
        if len(size) != 2:
            raise ValueError(f"Plane '{axis}' patch size must have exactly two elements; got {size}")
        return int(size[0]), int(size[1])

    def _resolve_label_reference(self, volume: PlaneSliceVolume) -> Optional[np.ndarray]:
        for target in self.target_names:
            arr = self._get_volume_label(volume, target)
            if arr is not None:
                return arr
        return None

    def _get_volume(self, index: int) -> PlaneSliceVolume:
        try:
            return self._volumes[index]
        except IndexError as exc:
            raise IndexError(f"Plane slice volume index {index} out of range") from exc

    # Geometry helpers -----------------------------------------------------------------------------

    @staticmethod
    def _extract_spatial_shape(array) -> Optional[Tuple[int, int, int]]:
        if array is None:
            return None
        if hasattr(array, 'spatial_shape'):
            shape = tuple(int(v) for v in array.spatial_shape)  # type: ignore[attr-defined]
            if len(shape) == 2:
                return (1, shape[0], shape[1])
            return shape
        shape = getattr(array, 'shape', None)
        if shape is None:
            arr = np.asarray(array)
            shape = arr.shape
        if callable(shape):
            shape = shape()
        if len(shape) == 4:
            return shape[1:]
        if len(shape) == 3:
            return shape
        if len(shape) == 2:
            return (1, shape[0], shape[1])
        return None

    @staticmethod
    def _plane_shape_from_spatial(spatial_shape: Tuple[int, int, int], axis: str) -> Optional[Tuple[int, int]]:
        if spatial_shape is None or len(spatial_shape) != 3:
            return None
        z, y, x = spatial_shape
        if axis == "z":
            return (y, x)
        if axis == "y":
            return (z, x)
        if axis == "x":
            return (z, y)
        raise KeyError(f"Unsupported plane axis '{axis}'")

    @staticmethod
    def _iter_plane_positions(plane_shape: Tuple[int, int], patch_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        h, w = plane_shape
        ph, pw = patch_size
        if ph <= 0 or pw <= 0 or h <= 0 or w <= 0:
            return []

        if ph >= h:
            y_positions = [0]
        else:
            y_positions = list(range(0, h - ph + 1, ph))
            if y_positions[-1] + ph < h:
                y_positions.append(h - ph)

        if pw >= w:
            x_positions = [0]
        else:
            x_positions = list(range(0, w - pw + 1, pw))
            if x_positions[-1] + pw < w:
                x_positions.append(w - pw)

        positions: List[Tuple[int, int]] = []
        for y in y_positions:
            for x in x_positions:
                positions.append((y, x))
        return positions

    def _extract_label_slice_mask(
        self, label_arr: Optional[np.ndarray], axis: str, slice_idx: int
    ) -> Optional[np.ndarray]:
        if label_arr is None:
            return None
        try:
            if label_arr.ndim == 4:
                if axis == "z":
                    slice_data = label_arr[:, slice_idx, :, :]
                elif axis == "y":
                    slice_data = label_arr[:, :, slice_idx, :]
                else:
                    slice_data = label_arr[:, :, :, slice_idx]
                mask = np.any(slice_data > 0, axis=0)
            elif label_arr.ndim == 3:
                if axis == "z":
                    slice_data = label_arr[slice_idx, :, :]
                elif axis == "y":
                    slice_data = label_arr[:, slice_idx, :]
                else:
                    slice_data = label_arr[:, :, slice_idx]
                mask = slice_data > 0
            elif label_arr.ndim == 2:
                mask = label_arr > 0
            else:
                raise ValueError(f"Unsupported label array ndim {label_arr.ndim} for plane masks")
            return np.asarray(mask, dtype=bool)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract label slice at plane '{axis}' index {slice_idx}: {exc}"
            ) from exc

    def _fallback_patch_size(self) -> Tuple[int, int]:
        primary = self.config.primary_plane or self.config.sample_planes[0]
        return self._resolve_patch_size(primary)

    def _mask_satisfies_thresholds(self, mask_patch: np.ndarray) -> bool:
        if mask_patch is None:
            return True
        if not np.any(mask_patch):
            return False

        bbox = compute_bounding_box_3d(mask_patch)
        if bbox is None:
            return False

        bb_vol = bounding_box_volume(bbox)
        patch_vol = mask_patch.size
        if patch_vol == 0:
            raise ValueError("Mask patch volume is zero")

        if (bb_vol / patch_vol) < self.config.min_bbox_percent:
            return False

        labeled_ratio = np.count_nonzero(mask_patch) / patch_vol
        return labeled_ratio >= self.config.min_labeled_ratio

    def _sample_yaw_angle(self, plane: str) -> float:
        cfg = self.config.random_rotation_planes.get(plane)
        if not cfg:
            return 0.0
        probability = 1.0
        if isinstance(cfg, dict):
            max_deg = float(cfg.get("max_degrees", 0.0))
            probability = float(cfg.get("probability", 1.0))
        else:
            max_deg = float(cfg)
        if max_deg <= 0:
            return 0.0
        if probability < 1.0 and np.random.random() > probability:
            return 0.0
        angle_deg = np.random.uniform(-max_deg / 2.0, max_deg / 2.0)
        return math.radians(angle_deg)

    def _sample_tilt_angles(self, plane: str) -> Dict[str, float]:
        cfg = self.config.random_tilt_planes.get(plane, {})
        probability = 1.0
        axis_cfg = cfg
        if isinstance(cfg, dict) and ("axes" in cfg or "probability" in cfg):
            probability = float(cfg.get("probability", 1.0))
            axis_cfg = cfg.get("axes", {})

        if probability < 1.0 and np.random.random() > probability:
            return {}

        angles: Dict[str, float] = {}
        if isinstance(axis_cfg, dict):
            for axis_key, max_deg in axis_cfg.items():
                max_deg = float(max_deg)
                if max_deg <= 0:
                    continue
                angle_deg = np.random.uniform(-max_deg / 2.0, max_deg / 2.0)
                angles[axis_key] = math.radians(angle_deg)
        return angles

    @staticmethod
    def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        return Rz @ Ry @ Rx

    @staticmethod
    def _ensure_perpendicular(vector: np.ndarray) -> np.ndarray:
        candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(vector, candidate)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        perp = np.cross(vector, candidate)
        norm = np.linalg.norm(perp)
        if norm < EPSILON:
            candidate = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            perp = np.cross(vector, candidate)
            norm = np.linalg.norm(perp)
        if norm < EPSILON:
            raise ValueError("Failed to compute perpendicular vector for plane orientation")
        return perp / norm

    def _compute_plane_center(
        self,
        plane: str,
        slice_idx: int,
        pos0: int,
        pos1: int,
        patch_size: Tuple[int, int],
    ) -> np.ndarray:
        ph, pw = patch_size
        half_u = (ph - 1) / 2.0
        half_v = (pw - 1) / 2.0

        if plane == "z":
            return np.array([pos1 + half_v, pos0 + half_u, float(slice_idx)], dtype=np.float32)
        if plane == "y":
            return np.array([pos1 + half_v, float(slice_idx), pos0 + half_u], dtype=np.float32)
        if plane == "x":
            return np.array([float(slice_idx), pos1 + half_v, pos0 + half_u], dtype=np.float32)
        raise KeyError(f"Unsupported plane axis '{plane}'")

    def _compute_plane_orientation(
        self,
        plane: str,
        slice_idx: int,
        pos0: int,
        pos1: int,
        patch_size: Tuple[int, int],
        yaw_angle: float,
        tilt_angles: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        base_normals = {
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
        base_u = {
            "z": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "y": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "x": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        base_v = {
            "z": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "y": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "x": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        center = self._compute_plane_center(plane, slice_idx, pos0, pos1, patch_size)

        rx = tilt_angles.get("x", 0.0)
        ry = tilt_angles.get("y", 0.0)
        rz = yaw_angle + tilt_angles.get("z", 0.0)

        rot = self._rotation_matrix(rx, ry, rz)

        u_dir = rot @ base_u[plane]
        v_dir = rot @ base_v[plane]

        u_norm = np.linalg.norm(u_dir)
        if u_norm < EPSILON:
            u_dir = base_u[plane]
        else:
            u_dir = u_dir / u_norm

        v_dir = v_dir - np.dot(v_dir, u_dir) * u_dir
        v_norm = np.linalg.norm(v_dir)
        if v_norm < EPSILON:
            v_dir = self._ensure_perpendicular(u_dir)
        else:
            v_dir = v_dir / v_norm

        normal = np.cross(u_dir, v_dir)
        n_norm = np.linalg.norm(normal)
        if n_norm < EPSILON:
            normal = base_normals[plane]
        else:
            normal = normal / n_norm

        return {"center": center, "u_dir": u_dir, "v_dir": v_dir, "normal": normal}

    def _slice_array_patch(
        self,
        array: np.ndarray,
        axis: str,
        slice_idx: int,
        pos0: int,
        pos1: int,
        patch_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if array is None:
            return None

        ph, pw = patch_size

        try:
            arr = np.asarray(array)

            if arr.ndim == 4:
                if axis == "z":
                    patch = arr[:, slice_idx, pos0 : pos0 + ph, pos1 : pos1 + pw]
                elif axis == "y":
                    patch = arr[:, pos0 : pos0 + ph, slice_idx, pos1 : pos1 + pw]
                elif axis == "x":
                    patch = arr[:, pos0 : pos0 + ph, pos1 : pos1 + pw, slice_idx]
                else:
                    raise KeyError(f"Unsupported plane axis '{axis}'")

                patch = np.asarray(patch)
                channels = patch.shape[0]
                padded = [pad_or_crop_2d(patch[c], patch_size) for c in range(channels)]
                stacked = np.stack(padded, axis=0)
                return stacked.astype(arr.dtype, copy=False)

            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)

            if arr.ndim != 3:
                raise ValueError(f"Unsupported image array ndim {arr.ndim} for plane slicing")

            if axis == "z":
                patch = arr[slice_idx, pos0 : pos0 + ph, pos1 : pos1 + pw]
            elif axis == "y":
                patch = arr[pos0 : pos0 + ph, slice_idx, pos1 : pos1 + pw]
            elif axis == "x":
                patch = arr[pos0 : pos0 + ph, pos1 : pos1 + pw, slice_idx]
            else:
                raise KeyError(f"Unsupported plane axis '{axis}'")

            patch_2d = pad_or_crop_2d(np.asarray(patch), patch_size)
            expanded = np.expand_dims(patch_2d, axis=0)
            return expanded.astype(arr.dtype, copy=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to slice array on plane '{axis}' index {slice_idx}: {exc}"
            ) from exc

    def _sample_rotated_plane(
        self,
        array: np.ndarray,
        plane: str,
        slice_idx: int,
        pos0: int,
        pos1: int,
        patch_size: Tuple[int, int],
        orientation: Dict[str, np.ndarray],
        *,
        interpolation: str = "linear",
        return_mask: bool = False,
    ) -> Tuple[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        if array is None or orientation is None:
            return (None, (None, None))

        has_channels = array.ndim == 4

        if has_channels:
            _, depth_z, depth_y, depth_x = array.shape
        elif array.ndim == 3:
            depth_z, depth_y, depth_x = array.shape
        elif array.ndim == 2:
            depth_z, depth_y, depth_x = 1, array.shape[0], array.shape[1]
            array = array[np.newaxis, ...]
        else:
            raise ValueError("Rotated sampling requires a 2D, 3D, or 4D array")

        ph, pw = patch_size
        half_u = (ph - 1) / 2.0
        half_v = (pw - 1) / 2.0

        center = orientation["center"]
        u_dir = orientation["u_dir"]
        v_dir = orientation["v_dir"]

        corners = []
        for su in (-half_u, half_u):
            for sv in (-half_v, half_v):
                corners.append(center + su * u_dir + sv * v_dir)
        corners = np.stack(corners, axis=0)

        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        z_coords = corners[:, 2]

        x_min = max(0, int(math.floor(np.min(x_coords))) - 2)
        x_max = min(depth_x - 1, int(math.ceil(np.max(x_coords))) + 2)
        y_min = max(0, int(math.floor(np.min(y_coords))) - 2)
        y_max = min(depth_y - 1, int(math.ceil(np.max(y_coords))) + 2)
        z_min = max(0, int(math.floor(np.min(z_coords))) - 2)
        z_max = min(depth_z - 1, int(math.ceil(np.max(z_coords))) + 2)

        if z_min > z_max or y_min > y_max or x_min > x_max:
            return (None, (None, None))

        if has_channels:
            sub_volume = np.asarray(array[:, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1], dtype=np.float32)
        else:
            sub_volume = np.asarray(array[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1], dtype=np.float32)

        center_local = np.array([
            center[0] - x_min,
            center[1] - y_min,
            center[2] - z_min,
        ], dtype=np.float32)

        grid_u = np.arange(ph, dtype=np.float32) - half_u
        grid_v = np.arange(pw, dtype=np.float32) - half_v
        uu, vv = np.meshgrid(grid_u, grid_v, indexing="ij")

        coord_local = (
            center_local[np.newaxis, np.newaxis, :]
            + uu[..., np.newaxis] * u_dir[np.newaxis, np.newaxis, :]
            + vv[..., np.newaxis] * v_dir[np.newaxis, np.newaxis, :]
        )

        coords_x = coord_local[..., 0]
        coords_y = coord_local[..., 1]
        coords_z = coord_local[..., 2]

        def normalize(coords: np.ndarray, size: int) -> np.ndarray:
            if size <= 1:
                return np.zeros_like(coords, dtype=np.float32)
            return (2.0 * coords / (size - 1)) - 1.0

        x_norm = normalize(coords_x, sub_volume.shape[-1])
        y_norm = normalize(coords_y, sub_volume.shape[-2])
        z_norm = normalize(coords_z, sub_volume.shape[-3])

        grid = np.stack([z_norm, y_norm, x_norm], axis=-1)

        import torch
        import torch.nn.functional as F

        if has_channels:
            tensor = torch.from_numpy(sub_volume[np.newaxis, ...])
        else:
            tensor = torch.from_numpy(sub_volume[np.newaxis, np.newaxis, ...])

        grid_tensor = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0)

        sampled = F.grid_sample(
            tensor,
            grid_tensor,
            mode="bilinear" if interpolation == "linear" else "nearest",
            padding_mode="zeros",
            align_corners=True,
        )

        sampled_np = sampled.squeeze(0).numpy()
        if not has_channels:
            sampled_np = sampled_np[0]

        mask_2d = None
        mask_3d = None
        if return_mask:
            mask_2d = np.ones((ph, pw), dtype=np.uint8)
            if self.config.save_plane_masks and self.config.plane_mask_mode == "volume":
                mask_shape = self._mask_volume_shape
                if mask_shape is None:
                    raise RuntimeError("Mask volume shape is not initialised for plane masks")
                mask_3d = np.zeros(mask_shape, dtype=np.uint8)
                coords_global = (
                    orientation["center"][np.newaxis, np.newaxis, :]
                    + uu[..., np.newaxis] * u_dir[np.newaxis, np.newaxis, :]
                    + vv[..., np.newaxis] * v_dir[np.newaxis, np.newaxis, :]
                )
                flat = coords_global.reshape(-1, 3)
                xs = np.clip(np.round(flat[:, 0]).astype(int), 0, mask_shape[2] - 1)
                ys = np.clip(np.round(flat[:, 1]).astype(int), 0, mask_shape[1] - 1)
                zs = np.clip(np.round(flat[:, 2]).astype(int), 0, mask_shape[0] - 1)
                mask_3d[zs, ys, xs] = 1

        return sampled_np, (mask_2d, mask_3d)

    def _use_linear_label_interp(self, target_name: str, plane: str) -> bool:
        conf = self.config.label_interpolation.get(target_name)
        if not conf:
            return False
        if isinstance(conf, dict):
            if plane in conf:
                return bool(conf[plane])
            if "__all__" in conf:
                return bool(conf["__all__"])
        return bool(conf)

    def _normalize_plane_mask_volume(self, mask_volume: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if self.config.plane_mask_mode != "volume":
            return mask_volume
        if self._mask_volume_shape is None:
            raise RuntimeError("Mask volume shape requested but not initialised")
        if mask_volume is None:
            return np.zeros(self._mask_volume_shape, dtype=np.uint8)
        if tuple(mask_volume.shape) == tuple(self._mask_volume_shape) and mask_volume.dtype == np.uint8:
            return mask_volume
        mask_uint8 = mask_volume.astype(np.uint8, copy=False)
        return pad_or_crop_3d(mask_uint8, self._mask_volume_shape)

    def _build_axis_aligned_plane_mask_2d(
        self,
        spatial_shape: Tuple[int, int, int],
        plane: str,
        slice_idx: int,
        pos0: int,
        pos1: int,
        patch_size: Tuple[int, int],
    ) -> np.ndarray:
        ph, pw = patch_size
        mask = np.zeros((ph, pw), dtype=np.uint8)

        depth_z, depth_y, depth_x = spatial_shape

        if plane == "z":
            if not (0 <= slice_idx < depth_z):
                return mask
            axis0_min, axis0_max = 0, depth_y
            axis1_min, axis1_max = 0, depth_x
        elif plane == "y":
            if not (0 <= slice_idx < depth_y):
                return mask
            axis0_min, axis0_max = 0, depth_z
            axis1_min, axis1_max = 0, depth_x
        elif plane == "x":
            if not (0 <= slice_idx < depth_x):
                return mask
            axis0_min, axis0_max = 0, depth_z
            axis1_min, axis1_max = 0, depth_y
        else:
            raise KeyError(f"Unsupported plane axis '{plane}'")

        g0_start = max(pos0, axis0_min)
        g0_end = min(pos0 + ph, axis0_max)
        g1_start = max(pos1, axis1_min)
        g1_end = min(pos1 + pw, axis1_max)

        if g0_end <= g0_start or g1_end <= g1_start:
            return mask

        local0_start = g0_start - pos0
        local1_start = g1_start - pos1

        mask[
            local0_start : local0_start + (g0_end - g0_start),
            local1_start : local1_start + (g1_end - g1_start),
        ] = 1

        return mask

    def _build_axis_aligned_plane_mask(
        self,
        array: np.ndarray,
        plane: str,
        slice_idx: int,
        pos0: int,
        pos1: int,
        patch_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        spatial_shape = self._extract_spatial_shape(array)
        if spatial_shape is None or len(spatial_shape) != 3:
            return None

        depth_z, depth_y, depth_x = spatial_shape
        mask = np.zeros(spatial_shape, dtype=np.uint8)

        if plane == "z":
            if 0 <= slice_idx < depth_z:
                y0 = max(0, pos0)
                y1 = min(depth_y, pos0 + patch_size[0])
                x0 = max(0, pos1)
                x1 = min(depth_x, pos1 + patch_size[1])
                if y0 < y1 and x0 < x1:
                    mask[slice_idx, y0:y1, x0:x1] = 1
        elif plane == "y":
            if 0 <= slice_idx < depth_y:
                z0 = max(0, pos0)
                z1 = min(depth_z, pos0 + patch_size[0])
                x0 = max(0, pos1)
                x1 = min(depth_x, pos1 + patch_size[1])
                if z0 < z1 and x0 < x1:
                    mask[z0:z1, slice_idx, x0:x1] = 1
        elif plane == "x":
            if 0 <= slice_idx < depth_x:
                z0 = max(0, pos0)
                z1 = min(depth_z, pos0 + patch_size[0])
                y0 = max(0, pos1)
                y1 = min(depth_y, pos1 + patch_size[1])
                if z0 < z1 and y0 < y1:
                    mask[z0:z1, y0:y1, slice_idx] = 1
        else:
            raise KeyError(f"Unsupported plane axis '{plane}'")

        return mask

    def _prepare_plane_mask(
        self,
        mask_2d: Optional[np.ndarray],
        mask_3d: Optional[np.ndarray],
        array: np.ndarray,
        plane: str,
        slice_idx: int,
        pos0: int,
        pos1: int,
        patch_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if not self.config.save_plane_masks:
            return None

        if self.config.plane_mask_mode == "plane":
            if mask_2d is None:
                spatial_shape = self._extract_spatial_shape(array)
                if spatial_shape is None:
                    raise ValueError("Unable to determine spatial shape while building plane mask")
                mask_2d = self._build_axis_aligned_plane_mask_2d(
                    spatial_shape=spatial_shape,
                    plane=plane,
                    slice_idx=slice_idx,
                    pos0=pos0,
                    pos1=pos1,
                    patch_size=patch_size,
                )
            mask_plane = mask_2d.astype(np.uint8, copy=False)
            return pad_or_crop_2d(mask_plane, patch_size)

        if mask_3d is None:
            mask_3d = self._build_axis_aligned_plane_mask(
                array,
                plane=plane,
                slice_idx=slice_idx,
                pos0=pos0,
                pos1=pos1,
                patch_size=patch_size,
            )
        return self._normalize_plane_mask_volume(mask_3d)

    def _extract_slice_patch(self, volume: PlaneSliceVolume, patch: PlaneSlicePatch) -> PlaneSliceResult:
        plane = patch.plane
        slice_idx = patch.slice_index
        pos0, pos1 = patch.position
        patch_size = patch.patch_size

        yaw_angle = self._sample_yaw_angle(plane)
        tilt_angles = self._sample_tilt_angles(plane)

        orientation = None
        image_mask_2d = None
        image_mask_3d = None

        image_array_full = self._get_volume_image(volume)

        if abs(yaw_angle) > EPSILON or any(abs(v) > EPSILON for v in tilt_angles.values()):
            orientation = self._compute_plane_orientation(
                plane=plane,
                slice_idx=slice_idx,
                pos0=pos0,
                pos1=pos1,
                patch_size=patch_size,
                yaw_angle=yaw_angle,
                tilt_angles=tilt_angles,
            )
            img_patch, (image_mask_2d, image_mask_3d) = self._sample_rotated_plane(
                image_array_full,
                plane=plane,
                slice_idx=slice_idx,
                pos0=pos0,
                pos1=pos1,
                patch_size=patch_size,
                orientation=orientation,
                interpolation="linear",
                return_mask=True,
            )
            if img_patch is None:
                orientation = None
                image_mask_2d = None
                image_mask_3d = None
        else:
            img_patch = None

        if img_patch is None:
            img_patch = self._slice_array_patch(image_array_full, plane, slice_idx, pos0, pos1, patch_size)
            if img_patch is None:
                raise RuntimeError("PlaneSlicer failed to extract image patch")
            if image_mask_2d is None:
                if self.config.save_plane_masks and self.config.plane_mask_mode == "plane":
                    spatial_shape = self._extract_spatial_shape(image_array_full)
                    if spatial_shape is None:
                        raise ValueError("Unable to compute spatial shape for plane mask")
                    image_mask_2d = self._build_axis_aligned_plane_mask_2d(
                        spatial_shape=spatial_shape,
                        plane=plane,
                        slice_idx=slice_idx,
                        pos0=pos0,
                        pos1=pos1,
                        patch_size=patch_size,
                    )
                if image_mask_2d is None:
                    image_mask_2d = np.ones(patch_size, dtype=np.uint8)
            if image_mask_3d is None and self.config.save_plane_masks and self.config.plane_mask_mode == "volume":
                image_mask_3d = self._build_axis_aligned_plane_mask(
                    image_array_full,
                    plane=plane,
                    slice_idx=slice_idx,
                    pos0=pos0,
                    pos1=pos1,
                    patch_size=patch_size,
                )

        mask_for_norm = None
        if image_mask_2d is not None:
            mask_for_norm = image_mask_2d.astype(np.float32, copy=False)
            if img_patch.ndim == 2:
                img_patch = img_patch * mask_for_norm
            else:
                img_patch = img_patch * mask_for_norm[np.newaxis, ...]

        if self.normalizer is not None:
            img_patch = self.normalizer.run(img_patch, mask=mask_for_norm)
        else:
            img_patch = img_patch.astype(np.float32)

        if mask_for_norm is not None:
            if img_patch.ndim == 2:
                img_patch = img_patch * mask_for_norm
            else:
                img_patch = img_patch * mask_for_norm[np.newaxis, ...]

        if img_patch.ndim == 2:
            img_patch = img_patch[np.newaxis, ...]

        img_patch = np.ascontiguousarray(img_patch, dtype=np.float32)

        label_patches: Dict[str, np.ndarray] = {}
        is_unlabeled = True

        for target_name in self.target_names:
            label_arr = self._get_volume_label(volume, target_name)
            label_patch = None
            label_mask_2d = None
            label_mask_3d = None
            use_linear = self._use_linear_label_interp(target_name, plane)
            if label_arr is not None:
                if orientation is not None:
                    rotated_label, (label_mask_2d, label_mask_3d) = self._sample_rotated_plane(
                        label_arr,
                        plane=plane,
                        slice_idx=slice_idx,
                        pos0=pos0,
                        pos1=pos1,
                        patch_size=patch_size,
                        orientation=orientation,
                        interpolation="linear" if use_linear else "nearest",
                        return_mask=self.config.save_plane_masks,
                    )
                    if rotated_label is not None:
                        label_patch = rotated_label
                if label_patch is None:
                    label_patch = self._slice_array_patch(label_arr, plane, slice_idx, pos0, pos1, patch_size)
                if label_patch is None:
                    raise RuntimeError("PlaneSlicer failed to extract label patch")
                label_patch = np.asarray(label_patch, dtype=np.float32)
                if label_patch.ndim == 2:
                    label_patch = label_patch[np.newaxis, ...]
                else:
                    label_patch = np.ascontiguousarray(label_patch, dtype=np.float32)
                is_unlabeled = False
            else:
                label_patch = np.zeros((1,) + patch_size, dtype=np.float32)

            label_patches[target_name] = label_patch

        plane_mask = self._prepare_plane_mask(
            mask_2d=image_mask_2d,
            mask_3d=image_mask_3d,
            array=volume.image,
            plane=plane,
            slice_idx=slice_idx,
            pos0=pos0,
            pos1=pos1,
            patch_size=patch_size,
        )

        mesh_payloads: Dict[str, Dict[str, object]] = {}
        for mesh_id, handle in volume.meshes.items():
            payload = handle.read()
            mesh_payloads[mesh_id] = {
                "payload": payload,
                "metadata": handle.metadata,
            }

        result = PlaneSliceResult(
            image=img_patch,
            labels=label_patches,
            is_unlabeled=is_unlabeled,
            plane_mask=plane_mask,
            patch_info={
                "plane": plane,
                "slice_index": slice_idx,
                "position": list(patch.position),
                "patch_size": list(patch.patch_size),
                "angles": {
                    "yaw_rad": float(yaw_angle),
                    "tilt_x_rad": float(tilt_angles.get("x", 0.0)),
                    "tilt_y_rad": float(tilt_angles.get("y", 0.0)),
                    "tilt_z_rad": float(tilt_angles.get("z", 0.0)),
                },
                "global_position": [int(slice_idx), int(pos0), int(pos1)] if plane == "z" else (
                    [int(pos0), int(slice_idx), int(pos1)] if plane == "y" else [int(pos0), int(pos1), int(slice_idx)]
                ),
                "global_end": [
                    int(slice_idx + 1) if plane == "z" else int(pos0 + patch_size[0]),
                    int(pos0 + patch_size[0]) if plane == "z" else (int(slice_idx + 1) if plane == "y" else int(pos1 + patch_size[1])),
                    int(pos1 + patch_size[1]) if plane == "z" or plane == "y" else int(slice_idx + 1),
                ],
                "source_path": str(getattr(volume.image, "path", None)) if hasattr(volume.image, "path") else None,
                "label_source_paths": {
                    name: str(getattr(handle, "path", None)) if hasattr(handle, "path") else None
                    for name, handle in volume.labels.items()
                },
            },
            meshes=mesh_payloads,
        )

        if mesh_payloads:
            result.patch_info['meshes'] = {
                mesh_id: {
                    "path": str(handle.path),
                    "source_volume": handle.metadata.source_volume_id,
                }
                for mesh_id, handle in volume.meshes.items()
            }

        return result
