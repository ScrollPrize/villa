from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber, load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace.geometry import (
    classify_voxels,
    construct_up_vector,
    decode_lasagna_normals_xyz,
    perturb_direction,
    random_unit_vector,
    tangent_at_point,
    xyz_to_zyx,
    zyx_to_xyz,
)
from vesuvius.neural_tracing.fiber_trace.labels import POSITIVE_LABEL
from vesuvius.neural_tracing.datasets.common import (
    _read_volume_crop_from_patch as _common_read_volume_crop_from_patch,
    open_zarr as _common_open_zarr,
    open_zarr_group as _common_open_zarr_group,
)


@dataclass(frozen=True)
class FiberTraceBatch:
    volume: torch.Tensor
    valid_mask: torch.Tensor
    labels: torch.Tensor
    target_fw_xyz: torch.Tensor
    target_up_xyz: torch.Tensor
    target_up_valid: torch.Tensor
    cond_fw_xyz: torch.Tensor
    cond_up_xyz: torch.Tensor
    crop_origin_zyx: torch.Tensor
    crop_kinds: tuple[str, ...]
    fiber_paths: tuple[str, ...]
    direction_kinds: tuple[str, ...]

    def to(self, device: torch.device | str) -> "FiberTraceBatch":
        return FiberTraceBatch(
            volume=self.volume.to(device),
            valid_mask=self.valid_mask.to(device),
            labels=self.labels.to(device),
            target_fw_xyz=self.target_fw_xyz.to(device),
            target_up_xyz=self.target_up_xyz.to(device),
            target_up_valid=self.target_up_valid.to(device),
            cond_fw_xyz=self.cond_fw_xyz.to(device),
            cond_up_xyz=self.cond_up_xyz.to(device),
            crop_origin_zyx=self.crop_origin_zyx.to(device),
            crop_kinds=self.crop_kinds,
            fiber_paths=self.fiber_paths,
            direction_kinds=self.direction_kinds,
        )


@dataclass(frozen=True)
class _FiberRecord:
    fiber: Vc3dFiber
    volume: Any
    mask: Any
    nx: Any | None
    ny: Any | None
    volume_scale: int
    dataset_config: dict[str, Any]
    valid_threshold: float


def _as_zyx3(value: Any, *, key: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        size = int(value)
        return size, size, size
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be an int or length-3 sequence, got {value!r}")
    return tuple(int(v) for v in value)


def _open_zarr_array(
    path: str | Path, *, scale: int, auth_json_path: str | None, config: dict[str, Any]
):
    return _common_open_zarr(
        path, scale=scale, auth_json_path=auth_json_path, config=config
    )


def _open_zarr_group(
    path: str | Path, *, auth_json_path: str | None, config: dict[str, Any]
):
    return _common_open_zarr_group(path, auth_json_path=auth_json_path, config=config)


def _read_raw_crop(
    array: Any, crop_shape: tuple[int, int, int], min_corner: np.ndarray
) -> np.ndarray:
    max_corner = min_corner + np.asarray(crop_shape, dtype=np.int64)
    out = np.zeros(crop_shape, dtype=np.asarray(array[0:1, 0:1, 0:1]).dtype)
    shape = np.asarray(array.shape, dtype=np.int64)
    src_starts = np.maximum(min_corner, 0)
    src_ends = np.minimum(max_corner, shape)
    dst_starts = src_starts - min_corner
    dst_ends = dst_starts + (src_ends - src_starts)
    if np.all(src_ends > src_starts):
        out[
            dst_starts[0] : dst_ends[0],
            dst_starts[1] : dst_ends[1],
            dst_starts[2] : dst_ends[2],
        ] = array[
            src_starts[0] : src_ends[0],
            src_starts[1] : src_ends[1],
            src_starts[2] : src_ends[2],
        ]
    return out


def _read_image_crop_from_patch(
    patch: Any,
    crop_shape: tuple[int, int, int],
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    *,
    image_normalization: str,
) -> np.ndarray:
    return _common_read_volume_crop_from_patch(
        patch,
        crop_shape,
        min_corner,
        max_corner,
        image_normalization=image_normalization,
    )


def _resolve_fiber_paths(dataset_config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for item in dataset_config.get("fiber_paths", []) or []:
        paths.append(Path(item))
    fiber_glob = dataset_config.get("fiber_glob")
    if fiber_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(str(fiber_glob))))
    if not paths:
        raise ValueError("dataset entry must provide fiber_paths or fiber_glob")
    return paths


class FiberTraceBatchBuilder:
    """Sample single-fiber training batches from VC3D fiber JSON records."""

    def __init__(
        self, config: dict[str, Any], *, rng: np.random.Generator | None = None
    ) -> None:
        self.config = dict(config)
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng(self.config.get("seed", None))
        )
        self.crop_size = _as_zyx3(self.config.get("crop_size", 64), key="crop_size")
        self.batch_size = int(self.config.get("batch_size", 2))
        if self.batch_size <= 0 or self.batch_size % 2 != 0:
            raise ValueError(
                f"batch_size must be a positive even integer, got {self.batch_size}"
            )

        self.image_normalization = str(self.config.get("image_normalization", "zscore"))
        self.positive_direction_probability = float(
            self.config.get("positive_direction_probability", 0.5)
        )
        self.positive_direction_jitter_degrees = float(
            self.config.get("positive_direction_jitter_degrees", 10.0)
        )
        self.positive_radius = float(self.config.get("positive_radius", 1.5))
        self.ignore_radius = float(
            self.config.get("ignore_radius", max(3.0, self.positive_radius))
        )
        self.positive_cosine = float(
            self.config.get("positive_cosine", np.cos(np.deg2rad(30.0)))
        )
        self.negative_cosine = float(
            self.config.get("negative_cosine", np.cos(np.deg2rad(75.0)))
        )
        self.random_valid_max_attempts = int(
            self.config.get("random_valid_max_attempts", 256)
        )
        self.allow_arbitrary_up_fallback = bool(
            self.config.get("allow_arbitrary_up_fallback", False)
        )
        self.degenerate_up_policy = str(
            self.config.get("degenerate_up_policy", "invalid")
        )
        if self.degenerate_up_policy not in {"invalid", "raise"}:
            raise ValueError(
                "degenerate_up_policy must be 'invalid' or 'raise', "
                f"got {self.degenerate_up_policy!r}"
            )

        datasets = self.config.get("datasets")
        self.records: list[_FiberRecord] = []
        array_records = self.config.get("_array_records")
        if array_records is not None:
            if not isinstance(array_records, list) or not array_records:
                raise ValueError(
                    "_array_records must be a non-empty list when provided"
                )
            for record_raw in array_records:
                record_config = dict(record_raw)
                volume = np.asarray(record_config["volume"])
                mask = np.asarray(record_config["mask"])
                if volume.ndim != 3 or mask.ndim != 3:
                    raise ValueError("_array_records volume and mask must be 3D arrays")
                if tuple(volume.shape) != tuple(mask.shape):
                    raise ValueError(
                        "_array_records mask shape must match volume shape: "
                        f"volume={tuple(volume.shape)} mask={tuple(mask.shape)}"
                    )
                nx = record_config.get("nx")
                ny = record_config.get("ny")
                if nx is None or ny is None:
                    if not self.allow_arbitrary_up_fallback:
                        raise ValueError(
                            "_array_records entries must provide Lasagna nx and ny "
                            "normal channels"
                        )
                    nx_arr = None
                    ny_arr = None
                else:
                    nx_arr = np.asarray(nx)
                    ny_arr = np.asarray(ny)
                    if nx_arr.ndim != 3 or ny_arr.ndim != 3:
                        raise ValueError("_array_records nx and ny must be 3D arrays")
                    if tuple(nx_arr.shape) != tuple(volume.shape) or tuple(
                        ny_arr.shape
                    ) != tuple(volume.shape):
                        raise ValueError(
                            "_array_records normal channel shapes must match volume shape: "
                            f"volume={tuple(volume.shape)} nx={tuple(nx_arr.shape)} "
                            f"ny={tuple(ny_arr.shape)}"
                        )
                fiber = record_config.get("fiber")
                if fiber is None:
                    fiber = load_vc3d_fiber(record_config["fiber_path"])
                self.records.append(
                    _FiberRecord(
                        fiber=fiber,
                        volume=volume,
                        mask=mask,
                        nx=nx_arr,
                        ny=ny_arr,
                        volume_scale=0,
                        dataset_config=record_config,
                        valid_threshold=float(
                            record_config.get("valid_mask_threshold", 0.0)
                        ),
                    )
                )
        else:
            if not isinstance(datasets, list) or not datasets:
                raise ValueError("config must contain a non-empty datasets list")

        for dataset_config_raw in datasets or []:
            if array_records is not None:
                break
            dataset_config = dict(dataset_config_raw)
            volume_path = dataset_config.get("volume_path")
            if not volume_path:
                raise ValueError("dataset entry missing volume_path")
            mask_path = dataset_config.get("mask_path") or dataset_config.get(
                "grad_mag_path"
            )
            if not mask_path:
                raise ValueError(
                    "dataset entry must provide mask_path or grad_mag_path"
                )
            nx_path = dataset_config.get("nx_path")
            ny_path = dataset_config.get("ny_path")
            if (not nx_path or not ny_path) and not self.allow_arbitrary_up_fallback:
                raise ValueError(
                    "dataset entry must provide explicit Lasagna nx_path and ny_path "
                    "normal channels"
                )

            volume_scale = int(dataset_config.get("volume_scale", 0))
            mask_scale = int(
                dataset_config.get(
                    "mask_scale", dataset_config.get("grad_mag_scale", volume_scale)
                )
            )
            normal_scale = int(dataset_config.get("normal_scale", volume_scale))
            volume = _open_zarr_array(
                volume_path,
                scale=volume_scale,
                auth_json_path=dataset_config.get("volume_auth_json"),
                config=self.config,
            )
            mask = _open_zarr_array(
                mask_path,
                scale=mask_scale,
                auth_json_path=dataset_config.get(
                    "mask_auth_json", dataset_config.get("volume_auth_json")
                ),
                config=self.config,
            )
            if nx_path and ny_path:
                nx = _open_zarr_array(
                    nx_path,
                    scale=int(dataset_config.get("nx_scale", normal_scale)),
                    auth_json_path=dataset_config.get(
                        "nx_auth_json",
                        dataset_config.get(
                            "normal_auth_json", dataset_config.get("volume_auth_json")
                        ),
                    ),
                    config=self.config,
                )
                ny = _open_zarr_array(
                    ny_path,
                    scale=int(dataset_config.get("ny_scale", normal_scale)),
                    auth_json_path=dataset_config.get(
                        "ny_auth_json",
                        dataset_config.get(
                            "normal_auth_json", dataset_config.get("volume_auth_json")
                        ),
                    ),
                    config=self.config,
                )
            else:
                nx = None
                ny = None
            if tuple(volume.shape) != tuple(mask.shape):
                raise ValueError(
                    "mask/grad-mag shape must match volume shape at selected levels: "
                    f"volume={tuple(volume.shape)} mask={tuple(mask.shape)}"
                )
            if nx is not None and tuple(volume.shape) != tuple(nx.shape):
                raise ValueError(
                    "nx normal shape must match volume shape at selected levels: "
                    f"volume={tuple(volume.shape)} nx={tuple(nx.shape)}"
                )
            if ny is not None and tuple(volume.shape) != tuple(ny.shape):
                raise ValueError(
                    "ny normal shape must match volume shape at selected levels: "
                    f"volume={tuple(volume.shape)} ny={tuple(ny.shape)}"
                )

            valid_threshold = float(
                dataset_config.get(
                    "valid_mask_threshold", self.config.get("valid_mask_threshold", 0.0)
                )
            )
            for fiber_path in _resolve_fiber_paths(dataset_config):
                self.records.append(
                    _FiberRecord(
                        fiber=load_vc3d_fiber(fiber_path),
                        volume=volume,
                        mask=mask,
                        nx=nx,
                        ny=ny,
                        volume_scale=volume_scale,
                        dataset_config=dataset_config,
                        valid_threshold=valid_threshold,
                    )
                )
        if not self.records:
            raise ValueError("no fiber records were loaded")

    def __len__(self) -> int:
        return len(self.records)

    def _sample_record(self, record_index: int | None) -> _FiberRecord:
        if record_index is not None:
            return self.records[int(record_index)]
        return self.records[int(self.rng.integers(0, len(self.records)))]

    def _normal_at_zyx(
        self, record: _FiberRecord, center_zyx: np.ndarray
    ) -> tuple[np.ndarray | None, bool]:
        if record.nx is None or record.ny is None:
            if self.allow_arbitrary_up_fallback:
                return None, True
            raise ValueError("Lasagna nx/ny normal channels are required")
        center = np.asarray(center_zyx, dtype=np.int64)
        shape = np.asarray(record.nx.shape, dtype=np.int64)
        if np.any(center < 0) or np.any(center >= shape):
            return np.zeros(3, dtype=np.float32), False
        nx_vox = np.asarray(
            record.nx[
                center[0] : center[0] + 1,
                center[1] : center[1] + 1,
                center[2] : center[2] + 1,
            ]
        )
        ny_vox = np.asarray(
            record.ny[
                center[0] : center[0] + 1,
                center[1] : center[1] + 1,
                center[2] : center[2] + 1,
            ]
        )
        normal_xyz, valid = decode_lasagna_normals_xyz(nx_vox, ny_vox)
        return normal_xyz.reshape(-1, 3)[0], bool(valid.reshape(-1)[0])

    def _sample_random_valid_center(self, record: _FiberRecord) -> np.ndarray:
        shape = np.asarray(record.mask.shape, dtype=np.int64)
        for _ in range(self.random_valid_max_attempts):
            center = np.asarray(
                [self.rng.integers(0, int(size)) for size in shape], dtype=np.int64
            )
            value = np.asarray(
                record.mask[
                    center[0] : center[0] + 1,
                    center[1] : center[1] + 1,
                    center[2] : center[2] + 1,
                ]
            )
            if value.size and float(value.reshape(-1)[0]) > record.valid_threshold:
                _, normal_valid = self._normal_at_zyx(record, center)
                if normal_valid:
                    return center

        mask_np = np.asarray(record.mask[:])
        valid_mask = mask_np > record.valid_threshold
        if record.nx is not None and record.ny is not None:
            _, normal_valid = decode_lasagna_normals_xyz(
                np.asarray(record.nx[:]), np.asarray(record.ny[:])
            )
            valid_mask = valid_mask & normal_valid
        valid_coords = np.argwhere(valid_mask)
        if valid_coords.size == 0:
            raise ValueError("mask/grad-mag and nx/ny volumes contain no valid voxels")
        return valid_coords[int(self.rng.integers(0, valid_coords.shape[0]))].astype(
            np.int64, copy=False
        )

    def _sample_conditioning(
        self, tangent_xyz: np.ndarray, normal_xyz: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if float(self.rng.random()) < self.positive_direction_probability:
            fw = perturb_direction(
                tangent_xyz,
                max_angle_degrees=self.positive_direction_jitter_degrees,
                rng=self.rng,
            )
            direction_kind = "gt_tangent"
        else:
            fw = None
            for _ in range(16):
                candidate = random_unit_vector(self.rng)
                try:
                    construct_up_vector(
                        candidate,
                        normal_xyz,
                        allow_arbitrary_up_fallback=self.allow_arbitrary_up_fallback,
                    )
                except ValueError:
                    continue
                fw = candidate
                break
            if fw is None:
                raise ValueError(
                    "could not sample a random conditioning direction with a valid "
                    "Lasagna normal up vector"
                )
            direction_kind = "random"
        up = construct_up_vector(
            fw,
            normal_xyz,
            allow_arbitrary_up_fallback=self.allow_arbitrary_up_fallback,
        )
        return (
            fw.astype(np.float32, copy=False),
            up.astype(np.float32, copy=False),
            direction_kind,
        )

    def _sample_one_crop(
        self, record: _FiberRecord, *, crop_kind: str
    ) -> dict[str, Any]:
        fiber = record.fiber
        if crop_kind == "gt_control":
            controls_xyz = fiber.control_points_xyz
            center_xyz = controls_xyz[int(self.rng.integers(0, controls_xyz.shape[0]))]
            center = np.rint(xyz_to_zyx(center_xyz)).astype(np.int64)
        elif crop_kind == "random_valid":
            center = self._sample_random_valid_center(record)
        else:
            raise ValueError(f"unsupported crop_kind {crop_kind!r}")
        center_normal_xyz, center_normal_valid = self._normal_at_zyx(record, center)
        if not center_normal_valid:
            raise ValueError(f"sampled {crop_kind} center has invalid Lasagna nx/ny")

        crop_size_np = np.asarray(self.crop_size, dtype=np.int64)
        min_corner = center - (crop_size_np // 2)
        max_corner = min_corner + crop_size_np
        patch = SimpleNamespace(volume=record.volume, scale=record.volume_scale)
        volume_crop = _read_image_crop_from_patch(
            patch,
            self.crop_size,
            min_corner,
            max_corner,
            image_normalization=self.image_normalization,
        )
        valid_crop = (
            _read_raw_crop(record.mask, self.crop_size, min_corner)
            > record.valid_threshold
        )
        if record.nx is None or record.ny is None:
            if not self.allow_arbitrary_up_fallback:
                raise ValueError("Lasagna nx/ny normal channels are required")
            normal_xyz_crop = None
            normal_valid_crop = np.ones(self.crop_size, dtype=bool)
        else:
            nx_crop = _read_raw_crop(record.nx, self.crop_size, min_corner)
            ny_crop = _read_raw_crop(record.ny, self.crop_size, min_corner)
            normal_xyz_crop, normal_valid_crop = decode_lasagna_normals_xyz(
                nx_crop, ny_crop
            )
            valid_crop = valid_crop & normal_valid_crop
        if not bool(valid_crop.any()):
            raise ValueError(
                f"sampled {crop_kind} crop has no valid mask/grad-mag and nx/ny voxels"
            )

        tangent_xyz = tangent_at_point(
            fiber.line_points_xyz, zyx_to_xyz(center.astype(np.float32))
        )
        cond_fw_xyz, cond_up_xyz, direction_kind = self._sample_conditioning(
            tangent_xyz, center_normal_xyz
        )
        classified = classify_voxels(
            crop_origin_zyx=min_corner,
            crop_shape=self.crop_size,
            line_points_xyz=fiber.line_points_xyz,
            cond_fw_xyz=cond_fw_xyz,
            valid_mask=valid_crop,
            normal_xyz=normal_xyz_crop,
            normal_valid_mask=normal_valid_crop,
            allow_arbitrary_up_fallback=self.allow_arbitrary_up_fallback,
            degenerate_up_policy=self.degenerate_up_policy,
            positive_radius=self.positive_radius,
            ignore_radius=self.ignore_radius,
            positive_cosine=self.positive_cosine,
            negative_cosine=self.negative_cosine,
        )
        positive_labels = classified["labels"] == POSITIVE_LABEL
        if bool(positive_labels.any()) and not bool(
            (positive_labels & classified["target_up_valid"]).any()
        ):
            raise ValueError(
                f"sampled {crop_kind} crop has no valid up-vector supervision"
            )
        return {
            "volume": volume_crop.astype(np.float32, copy=False),
            "valid_mask": valid_crop,
            "labels": classified["labels"],
            "target_fw_xyz": classified["target_fw_xyz"],
            "target_up_xyz": classified["target_up_xyz"],
            "target_up_valid": classified["target_up_valid"],
            "cond_fw_xyz": cond_fw_xyz,
            "cond_up_xyz": cond_up_xyz,
            "origin": min_corner.astype(np.int64, copy=False),
            "crop_kind": crop_kind,
            "direction_kind": direction_kind,
        }

    def sample_batch(self, *, record_index: int | None = None) -> FiberTraceBatch:
        record = self._sample_record(record_index)
        half = self.batch_size // 2
        crop_kinds = ("gt_control",) * half + ("random_valid",) * half
        samples = [self._sample_one_crop(record, crop_kind=kind) for kind in crop_kinds]
        fiber_path = str(record.fiber.path) if record.fiber.path is not None else ""

        return FiberTraceBatch(
            volume=torch.from_numpy(
                np.stack([s["volume"] for s in samples], axis=0)[:, None]
            ).to(torch.float32),
            valid_mask=torch.from_numpy(
                np.stack([s["valid_mask"] for s in samples], axis=0)
            ).to(torch.bool),
            labels=torch.from_numpy(
                np.stack([s["labels"] for s in samples], axis=0)
            ).to(torch.long),
            target_fw_xyz=torch.from_numpy(
                np.stack([s["target_fw_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            target_up_xyz=torch.from_numpy(
                np.stack([s["target_up_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            target_up_valid=torch.from_numpy(
                np.stack([s["target_up_valid"] for s in samples], axis=0)
            ).to(torch.bool),
            cond_fw_xyz=torch.from_numpy(
                np.stack([s["cond_fw_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            cond_up_xyz=torch.from_numpy(
                np.stack([s["cond_up_xyz"] for s in samples], axis=0)
            ).to(torch.float32),
            crop_origin_zyx=torch.from_numpy(
                np.stack([s["origin"] for s in samples], axis=0)
            ).to(torch.long),
            crop_kinds=tuple(str(s["crop_kind"]) for s in samples),
            fiber_paths=tuple(fiber_path for _ in samples),
            direction_kinds=tuple(str(s["direction_kind"]) for s in samples),
        )
