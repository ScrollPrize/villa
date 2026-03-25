from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from ink.recipes.stitch.ops import (
    gaussian_weights,
    normalize_xyxy_rows,
    resolve_buffer_crop,
    stitch_logit_map,
    stitch_prob_map,
)


def segment_store_key(segment_id: str) -> str:
    return str(segment_id).replace("/", "__")


def _downsample_shape(shape: tuple[int, int], *, downsample: int) -> tuple[int, int]:
    h, w = [int(v) for v in shape]
    ds = max(1, int(downsample))
    return ((h + ds - 1) // ds, (w + ds - 1) // ds)


def _chunk_shape(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = [max(1, int(v)) for v in shape]
    return (min(h, 256), min(w, 256))


def _to_u8(img_float: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img_float, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def downsample_preview_for_media(
    img: np.ndarray,
    *,
    source_downsample: int,
    media_downsample: int,
) -> np.ndarray:
    """Match the legacy stitched-media downsample behavior exactly."""
    source_downsample = int(source_downsample)
    media_downsample = int(media_downsample)
    if source_downsample < 1:
        raise ValueError(f"source_downsample must be >= 1, got {source_downsample}")
    if media_downsample < 1:
        raise ValueError(f"media_downsample must be >= 1, got {media_downsample}")
    if source_downsample > 1 or media_downsample == 1:
        return np.ascontiguousarray(img)

    in_h, in_w = img.shape[:2]
    factor = int(media_downsample)
    if (in_h % factor) == 0 and (in_w % factor) == 0:
        out_h = in_h // factor
        out_w = in_w // factor
        if img.ndim == 2:
            if img.dtype == np.uint8:
                reduced = img.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3), dtype=np.float32)
            else:
                reduced = img.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3))
        elif img.ndim == 3:
            channels = int(img.shape[2])
            if img.dtype == np.uint8:
                reduced = img.reshape(out_h, factor, out_w, factor, channels).mean(
                    axis=(1, 3),
                    dtype=np.float32,
                )
            else:
                reduced = img.reshape(out_h, factor, out_w, factor, channels).mean(axis=(1, 3))
        else:
            raise ValueError(f"unsupported image ndim for stitched preview downsample: {img.ndim}")
        if img.dtype == np.uint8:
            reduced = np.clip(np.rint(reduced), 0.0, 255.0).astype(np.uint8, copy=False)
        else:
            reduced = reduced.astype(img.dtype, copy=False)
        return np.ascontiguousarray(reduced)

    out_h = max(1, (int(in_h) + factor - 1) // factor)
    out_w = max(1, (int(in_w) + factor - 1) // factor)
    if out_h == in_h and out_w == in_w:
        return np.ascontiguousarray(img)

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stitched preview downsample fallback requires OpenCV (cv2)"
        ) from exc

    resized = cv2.resize(
        img,
        (out_w, out_h),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized)


def write_preview_png(*, out_path: str | Path, image_u8: np.ndarray) -> str:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_u8 = np.ascontiguousarray(np.asarray(image_u8, dtype=np.uint8))
    if image_u8.ndim not in {2, 3}:
        raise ValueError(f"preview image must be 2D or 3D uint8 array, got shape {tuple(image_u8.shape)}")

    try:
        from PIL import Image  # type: ignore

        Image.fromarray(image_u8).save(out_path, format="PNG")
        return str(out_path)
    except ModuleNotFoundError:
        pass

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "writing stitched preview PNGs requires Pillow or OpenCV (cv2)"
        ) from exc

    to_write = image_u8
    if image_u8.ndim == 3 and int(image_u8.shape[2]) == 3:
        to_write = image_u8[..., ::-1]
    if not bool(cv2.imwrite(str(out_path), to_write)):
        raise RuntimeError(f"failed to write preview PNG to {str(out_path)!r}")
    return str(out_path)


def _normalize_rois(
    rois: Any,
    *,
    full_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    full_h, full_w = [int(v) for v in full_shape]
    if rois is None:
        return ((0, full_h, 0, full_w),)

    normalized = []
    for item in tuple(rois):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("segment_rois entries must have 4 values")
        y0, y1, x0, x1 = [int(v) for v in item]
        y0 = max(0, min(y0, full_h))
        y1 = max(0, min(y1, full_h))
        x0 = max(0, min(x0, full_w))
        x1 = max(0, min(x1, full_w))
        if y1 > y0 and x1 > x0:
            normalized.append((y0, y1, x0, x1))
    if not normalized:
        return ((0, full_h, 0, full_w),)
    return tuple(normalized)


def _open_sparse_array(path: Path, shape: tuple[int, int]) -> Any:
    return zarr.open(
        str(path),
        mode="w",
        shape=tuple(int(v) for v in shape),
        chunks=_chunk_shape(shape),
        dtype=np.float32,
        fill_value=0.0,
        write_empty_chunks=False,
    )


def _clamp_bbox(
    bbox: tuple[int, int, int, int] | None,
    *,
    full_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    full_h, full_w = [int(v) for v in full_shape]
    if bbox is None:
        return (0, full_h, 0, full_w)
    y0, y1, x0, x1 = [int(v) for v in bbox]
    return (
        max(0, min(y0, full_h)),
        max(0, min(y1, full_h)),
        max(0, min(x0, full_w)),
        max(0, min(x1, full_w)),
    )


def _segment_array_paths(root_dir: Path, segment_id: str) -> tuple[Path, Path]:
    key = segment_store_key(segment_id)
    return (
        root_dir / f"{key}__logit_sum.zarr",
        root_dir / f"{key}__weight_sum.zarr",
    )


def _segment_preview_path(root_dir: Path, segment_id: str) -> Path:
    return root_dir / f"{segment_store_key(segment_id)}__prob.png"


def _prepare_patch_logits(patch_logits: torch.Tensor, *, target_h: int, target_w: int) -> torch.Tensor:
    if patch_logits.shape[-2:] == (int(target_h), int(target_w)):
        return patch_logits
    return F.interpolate(
        patch_logits,
        size=(int(target_h), int(target_w)),
        mode="bilinear",
        align_corners=False,
    )


def _roi_patch_slices(
    *,
    crop: dict[str, int],
    roi_bbox: tuple[int, int, int, int],
) -> tuple[slice, slice, slice, slice] | None:
    roi_y0, roi_y1, roi_x0, roi_x1 = [int(v) for v in roi_bbox]
    inter_y0 = max(int(crop["y1"]), roi_y0)
    inter_y1 = min(int(crop["y2"]), roi_y1)
    inter_x0 = max(int(crop["x1"]), roi_x0)
    inter_x1 = min(int(crop["x2"]), roi_x1)
    if inter_y1 <= inter_y0 or inter_x1 <= inter_x0:
        return None

    patch_y0 = int(inter_y0 - int(crop["y1"]))
    patch_y1 = int(patch_y0 + (inter_y1 - inter_y0))
    patch_x0 = int(inter_x0 - int(crop["x1"]))
    patch_x1 = int(patch_x0 + (inter_x1 - inter_x0))
    return (
        slice(inter_y0, inter_y1),
        slice(inter_x0, inter_x1),
        slice(patch_y0, patch_y1),
        slice(patch_x0, patch_x1),
    )


@dataclass
class ZarrStitchStore:
    root_dir: str | Path | None = None
    downsample: int = 1
    gaussian_sigma_scale: float = 1.0 / 8.0
    gaussian_min_weight: float = 1e-6

    _segment_shapes: dict[str, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _segment_rois: dict[str, tuple[tuple[int, int, int, int], ...]] = field(default_factory=dict, init=False, repr=False)
    _sum_arrays: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _weight_arrays: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _gaussian_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def build(
        self,
        *,
        segment_shapes: dict[str, tuple[int, int]],
        downsample: int | None = None,
        segment_rois: dict[str, tuple[tuple[int, int, int, int], ...]] | None = None,
    ) -> ZarrStitchStore:
        built = ZarrStitchStore(
            root_dir=self.root_dir,
            downsample=int(self.downsample if downsample is None else downsample),
            gaussian_sigma_scale=float(self.gaussian_sigma_scale),
            gaussian_min_weight=float(self.gaussian_min_weight),
        )
        built._configure_segments(segment_shapes, segment_rois=segment_rois)
        if built.root_dir is not None:
            built.reset()
        return built

    def _configure_segments(
        self,
        segment_shapes: dict[str, tuple[int, int]],
        *,
        segment_rois: dict[str, tuple[tuple[int, int, int, int], ...]] | None = None,
    ) -> None:
        self._segment_shapes = {
            str(segment_id): _downsample_shape(tuple(int(v) for v in shape), downsample=int(self.downsample))
            for segment_id, shape in dict(segment_shapes).items()
        }
        self._segment_rois = {
            str(segment_id): _normalize_rois(
                dict(segment_rois or {}).get(segment_id),
                full_shape=ds_shape,
            )
            for segment_id, ds_shape in self._segment_shapes.items()
        }
        self._sum_arrays = {}
        self._weight_arrays = {}

    def _resolved_root_dir(self) -> Path:
        root_dir = self.root_dir
        if root_dir is None:
            raise ValueError("stitch store root_dir must be set before stitched inference begins")
        return Path(root_dir).resolve()

    def _require_configured_segments(self) -> None:
        if self._segment_shapes:
            return
        raise ValueError("stitch store must be built with segment shapes before use")

    def _open_root_arrays(self, root_dir: Path) -> None:
        for segment_id, ds_shape in self._segment_shapes.items():
            self._sum_arrays[segment_id], self._weight_arrays[segment_id] = self._open_segment_arrays(
                root_dir,
                segment_id=segment_id,
                shape=ds_shape,
            )

    def reset(self) -> None:
        self._require_configured_segments()
        root_dir = self._resolved_root_dir()
        root_dir.mkdir(parents=True, exist_ok=True)
        self._sum_arrays = {}
        self._weight_arrays = {}
        self._open_root_arrays(root_dir)

    def _open_segment_arrays(self, root_dir: Path, *, segment_id: str, shape: tuple[int, int]) -> tuple[Any, Any]:
        sum_path, weight_path = _segment_array_paths(root_dir, segment_id)
        return _open_sparse_array(sum_path, shape), _open_sparse_array(weight_path, shape)

    def _segment_arrays(self, segment_id: str) -> tuple[Any, Any]:
        segment_id = str(segment_id)
        if not self._sum_arrays or not self._weight_arrays:
            raise RuntimeError("stitch store arrays are not initialized; set root_dir and call reset() before use")
        sum_arr = self._sum_arrays.get(segment_id)
        weight_arr = self._weight_arrays.get(segment_id)
        if sum_arr is None or weight_arr is None:
            raise KeyError(f"unknown stitched segment id {segment_id!r}")
        return sum_arr, weight_arr

    def segment_ds_shape(self, segment_id: str) -> tuple[int, int]:
        segment_id = str(segment_id)
        full_shape = self._segment_shapes.get(segment_id)
        if full_shape is None:
            raise KeyError(f"unknown stitched segment id {segment_id!r}")
        return tuple(int(v) for v in full_shape)

    def segment_ids(self) -> tuple[str, ...]:
        return tuple(str(segment_id) for segment_id in self._segment_shapes)

    def _accumulate_patch(
        self,
        *,
        segment_id: str,
        crop: dict[str, int],
        patch_crop_np: np.ndarray,
        weight_crop: np.ndarray,
    ) -> None:
        sum_arr, weight_arr = self._segment_arrays(segment_id)
        for roi_bbox in self._segment_rois.get(segment_id, ()):
            slices = _roi_patch_slices(
                crop=crop,
                roi_bbox=roi_bbox,
            )
            if slices is None:
                continue
            y_slice, x_slice, patch_y_slice, patch_x_slice = slices
            current_sum = np.asarray(sum_arr[y_slice, x_slice], dtype=np.float32)
            current_weight = np.asarray(weight_arr[y_slice, x_slice], dtype=np.float32)
            roi_patch = patch_crop_np[patch_y_slice, patch_x_slice]
            roi_weight = weight_crop[patch_y_slice, patch_x_slice]
            sum_arr[y_slice, x_slice] = current_sum + (roi_patch * roi_weight)
            weight_arr[y_slice, x_slice] = current_weight + roi_weight

    def add_batch(self, *, logits: torch.Tensor, xyxys: Any, segment_ids: tuple[str, ...] | list[str]) -> None:
        if logits.ndim != 4 or int(logits.shape[1]) != 1:
            raise ValueError(f"stitch logits must have shape (N,1,H,W), got {tuple(logits.shape)}")
        xyxy_rows = normalize_xyxy_rows(xyxys)
        segment_ids = [str(segment_id) for segment_id in segment_ids]
        if int(logits.shape[0]) != len(xyxy_rows) or len(segment_ids) != len(xyxy_rows):
            raise ValueError(
                "stitched batch size mismatch between logits, xyxys, and segment ids: "
                f"{int(logits.shape[0])} vs {len(xyxy_rows)} vs {len(segment_ids)}"
            )

        logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32)
        for batch_idx, segment_id in enumerate(segment_ids):
            patch_logits = logits_cpu[batch_idx : batch_idx + 1]
            crop = resolve_buffer_crop(
                xyxy=xyxy_rows[batch_idx],
                downsample=int(self.downsample),
                offset=(0, 0),
                buffer_shape=self.segment_ds_shape(segment_id),
            )
            if crop is None:
                continue

            patch_logits = _prepare_patch_logits(
                patch_logits,
                target_h=int(crop["target_h"]),
                target_w=int(crop["target_w"]),
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
            self._accumulate_patch(
                segment_id=segment_id,
                crop=crop,
                patch_crop_np=patch_crop.squeeze(0).squeeze(0).numpy(),
                weight_crop=weight_crop,
            )

    def _read_region_arrays(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        segment_id = str(segment_id)
        sum_arr, weight_arr = self._segment_arrays(segment_id)
        y0, y1, x0, x1 = _clamp_bbox(
            bbox,
            full_shape=self.segment_ds_shape(segment_id),
        )
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        return (
            np.asarray(sum_arr[y0:y1, x0:x1], dtype=np.float32),
            np.asarray(weight_arr[y0:y1, x0:x1], dtype=np.float32),
        )

    def read_region_logits_and_coverage(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        stitched_sum, stitched_weight = self._read_region_arrays(
            segment_id=segment_id,
            bbox=bbox,
        )
        return stitch_logit_map(stitched_sum, stitched_weight)

    def read_region_probs_and_coverage(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        stitched_sum, stitched_weight = self._read_region_arrays(
            segment_id=segment_id,
            bbox=bbox,
        )
        return stitch_prob_map(stitched_sum, stitched_weight)

    def write_full_segment_probs(self, *, segment_id: str, out_path: str | Path | None = None) -> str:
        segment_id = str(segment_id)
        if out_path is None:
            out_path = Path(self.root_dir).resolve() / f"{segment_store_key(segment_id)}__prob.zarr"
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        probs, _coverage = self.read_region_probs_and_coverage(segment_id=segment_id)
        prob_arr = zarr.open(
            str(out_path),
            mode="w",
            shape=tuple(probs.shape),
            chunks=_chunk_shape(tuple(probs.shape)),
            dtype=np.float32,
            fill_value=0.0,
            write_empty_chunks=False,
        )
        prob_arr[:] = probs
        return str(out_path)

    def full_segment_prob_preview_u8(
        self,
        *,
        segment_id: str,
        media_downsample: int = 1,
    ) -> np.ndarray:
        probs, _coverage = self.read_region_probs_and_coverage(segment_id=str(segment_id))
        return downsample_preview_for_media(
            _to_u8(probs),
            source_downsample=int(self.downsample),
            media_downsample=int(media_downsample),
        )

    def write_full_segment_preview_png(
        self,
        *,
        segment_id: str,
        media_downsample: int = 1,
        out_path: str | Path | None = None,
        image_u8: np.ndarray | None = None,
    ) -> str:
        segment_id = str(segment_id)
        root_dir = Path(self.root_dir).resolve()
        if out_path is None:
            out_path = _segment_preview_path(root_dir, segment_id)
        if image_u8 is None:
            image_u8 = self.full_segment_prob_preview_u8(
                segment_id=segment_id,
                media_downsample=int(media_downsample),
            )
        return write_preview_png(out_path=out_path, image_u8=image_u8)
