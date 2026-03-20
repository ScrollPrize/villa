from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from ink.recipes.stitch.ops import gaussian_weights, resolve_buffer_crop


def _coerce_xyxy_rows(xyxys: Any) -> list[tuple[int, int, int, int]]:
    if isinstance(xyxys, torch.Tensor):
        if xyxys.ndim == 1:
            if int(xyxys.numel()) != 4:
                raise ValueError(f"stitch xyxys tensor must have 4 values, got shape={tuple(xyxys.shape)}")
            return [tuple(int(v) for v in xyxys.tolist())]
        if xyxys.ndim != 2 or int(xyxys.shape[1]) != 4:
            raise ValueError(f"stitch xyxys tensor must have shape (N,4), got shape={tuple(xyxys.shape)}")
        return [tuple(int(v) for v in row.tolist()) for row in xyxys]

    if not isinstance(xyxys, (list, tuple)):
        raise TypeError(f"unsupported stitch xyxys value: {xyxys!r}")
    if len(xyxys) == 4 and all(not isinstance(value, (list, tuple)) for value in xyxys):
        return [tuple(int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in xyxys)]

    out = []
    for item in xyxys:
        if isinstance(item, torch.Tensor):
            flat = item.detach().reshape(-1)
            if int(flat.numel()) != 4:
                raise ValueError("stitch xyxy tensor entries must have 4 values")
            out.append(tuple(int(v.item()) for v in flat))
            continue
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("stitch xyxy entries must have 4 values")
        out.append(tuple(int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in item))
    return out


def _segment_store_key(segment_id: str) -> str:
    return str(segment_id).replace("/", "__")


def _downsample_shape(shape: tuple[int, int], *, downsample: int) -> tuple[int, int]:
    h, w = [int(v) for v in shape]
    ds = max(1, int(downsample))
    return ((h + ds - 1) // ds, (w + ds - 1) // ds)


def _sigmoid_numpy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


@dataclass
class ZarrStitchStore:
    root_dir: str | Path = ".tmp/stitch_eval"
    downsample: int = 1
    gaussian_sigma_scale: float = 1.0 / 8.0
    gaussian_min_weight: float = 1e-6

    _segment_shapes: dict[str, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _sum_arrays: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _weight_arrays: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _gaussian_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def build(self, *, segment_shapes: dict[str, tuple[int, int]], downsample: int | None = None) -> ZarrStitchStore:
        built = ZarrStitchStore(
            root_dir=self.root_dir,
            downsample=int(self.downsample if downsample is None else downsample),
            gaussian_sigma_scale=float(self.gaussian_sigma_scale),
            gaussian_min_weight=float(self.gaussian_min_weight),
        )
        built._init_arrays(segment_shapes)
        return built

    def _init_arrays(self, segment_shapes: dict[str, tuple[int, int]]) -> None:
        self._segment_shapes = {
            str(segment_id): tuple(int(v) for v in shape)
            for segment_id, shape in dict(segment_shapes).items()
        }
        root_dir = Path(self.root_dir).resolve()
        root_dir.mkdir(parents=True, exist_ok=True)

        self._sum_arrays = {}
        self._weight_arrays = {}
        for segment_id, full_shape in self._segment_shapes.items():
            ds_shape = _downsample_shape(full_shape, downsample=int(self.downsample))
            key = _segment_store_key(segment_id)
            sum_path = root_dir / f"{key}__logit_sum.zarr"
            weight_path = root_dir / f"{key}__weight_sum.zarr"
            sum_arr = zarr.open(str(sum_path), mode="w", shape=ds_shape, dtype=np.float32)
            weight_arr = zarr.open(str(weight_path), mode="w", shape=ds_shape, dtype=np.float32)
            sum_arr[:] = 0.0
            weight_arr[:] = 0.0
            self._sum_arrays[segment_id] = sum_arr
            self._weight_arrays[segment_id] = weight_arr

    def reset(self) -> None:
        for segment_id in self._sum_arrays:
            self._sum_arrays[segment_id][:] = 0.0
            self._weight_arrays[segment_id][:] = 0.0

    def segment_ds_shape(self, segment_id: str) -> tuple[int, int]:
        segment_id = str(segment_id)
        arr = self._sum_arrays.get(segment_id)
        if arr is None:
            raise KeyError(f"unknown stitched segment id {segment_id!r}")
        return tuple(int(v) for v in arr.shape)

    def add_batch(self, *, logits: torch.Tensor, xyxys: Any, segment_ids: tuple[str, ...] | list[str]) -> None:
        if logits.ndim != 4 or int(logits.shape[1]) != 1:
            raise ValueError(f"stitch logits must have shape (N,1,H,W), got {tuple(logits.shape)}")
        xyxy_rows = _coerce_xyxy_rows(xyxys)
        segment_ids = [str(segment_id) for segment_id in segment_ids]
        if int(logits.shape[0]) != len(xyxy_rows) or len(segment_ids) != len(xyxy_rows):
            raise ValueError(
                "stitched batch size mismatch between logits, xyxys, and segment ids: "
                f"{int(logits.shape[0])} vs {len(xyxy_rows)} vs {len(segment_ids)}"
            )

        logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32)
        for batch_idx, segment_id in enumerate(segment_ids):
            sum_arr = self._sum_arrays.get(segment_id)
            weight_arr = self._weight_arrays.get(segment_id)
            if sum_arr is None or weight_arr is None:
                raise KeyError(f"unknown stitched segment id {segment_id!r}")
            patch_logits = logits_cpu[batch_idx : batch_idx + 1]
            crop = resolve_buffer_crop(
                xyxy=xyxy_rows[batch_idx],
                downsample=int(self.downsample),
                offset=(0, 0),
                buffer_shape=sum_arr.shape,
            )
            if crop is None:
                continue

            if patch_logits.shape[-2:] != (crop["target_h"], crop["target_w"]):
                patch_logits = F.interpolate(
                    patch_logits,
                    size=(crop["target_h"], crop["target_w"]),
                    mode="bilinear",
                    align_corners=False,
                )

            weights = gaussian_weights(
                self._gaussian_cache,
                h=crop["target_h"],
                w=crop["target_w"],
                sigma_scale=float(self.gaussian_sigma_scale),
                min_weight=float(self.gaussian_min_weight),
            )
            weight_crop = weights[crop["py0"]:crop["py1"], crop["px0"]:crop["px1"]]
            patch_crop = patch_logits[..., crop["py0"]:crop["py1"], crop["px0"]:crop["px1"]]
            patch_crop_np = patch_crop.squeeze(0).squeeze(0).numpy() * weight_crop

            y_slice = slice(crop["y1"], crop["y2"])
            x_slice = slice(crop["x1"], crop["x2"])
            current_sum = np.asarray(sum_arr[y_slice, x_slice], dtype=np.float32)
            current_weight = np.asarray(weight_arr[y_slice, x_slice], dtype=np.float32)
            sum_arr[y_slice, x_slice] = current_sum + patch_crop_np
            weight_arr[y_slice, x_slice] = current_weight + weight_crop

    def read_region_logits_and_coverage(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        segment_id = str(segment_id)
        sum_arr = self._sum_arrays.get(segment_id)
        weight_arr = self._weight_arrays.get(segment_id)
        if sum_arr is None or weight_arr is None:
            raise KeyError(f"unknown stitched segment id {segment_id!r}")

        if bbox is None:
            y0, y1, x0, x1 = 0, int(sum_arr.shape[0]), 0, int(sum_arr.shape[1])
        else:
            y0, y1, x0, x1 = [int(v) for v in bbox]
            y0 = max(0, min(y0, int(sum_arr.shape[0])))
            y1 = max(0, min(y1, int(sum_arr.shape[0])))
            x0 = max(0, min(x0, int(sum_arr.shape[1])))
            x1 = max(0, min(x1, int(sum_arr.shape[1])))
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool)

        stitched_sum = np.asarray(sum_arr[y0:y1, x0:x1], dtype=np.float32)
        stitched_weight = np.asarray(weight_arr[y0:y1, x0:x1], dtype=np.float32)
        coverage = stitched_weight > 0.0
        stitched_logits = np.divide(
            stitched_sum,
            stitched_weight,
            out=np.zeros_like(stitched_sum, dtype=np.float32),
            where=coverage,
        )
        return stitched_logits, coverage

    def read_region_probs_and_coverage(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        stitched_logits, coverage = self.read_region_logits_and_coverage(
            segment_id=segment_id,
            bbox=bbox,
        )
        stitched_probs = np.zeros_like(stitched_logits, dtype=np.float32)
        if bool(coverage.any()):
            stitched_probs[coverage] = _sigmoid_numpy(stitched_logits[coverage])
        return stitched_probs, coverage

    def write_full_segment_probs(self, *, segment_id: str, out_path: str | Path | None = None) -> str:
        segment_id = str(segment_id)
        if out_path is None:
            out_path = Path(self.root_dir).resolve() / f"{_segment_store_key(segment_id)}__prob.zarr"
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        probs, _coverage = self.read_region_probs_and_coverage(segment_id=segment_id)
        prob_arr = zarr.open(str(out_path), mode="w", shape=tuple(probs.shape), dtype=np.float32)
        prob_arr[:] = probs
        return str(out_path)
