from __future__ import annotations

from typing import Any

import numpy as np
import zarr

from ink.recipes.data.layout import NestedZarrLayout, resolve_segment_artifact_path
from ink.recipes.data.masks import normalize_mask_names


def parse_layer_range_value(layer_range, *, context: str, total_layers: int | None = None) -> tuple[int, int]:
    """Validate a [start, end) layer range and normalize it to integers."""
    if layer_range is None:
        if total_layers is None:
            raise KeyError(f"{context} requires layer_range")
        return 0, int(total_layers)
    start_idx, end_idx = [int(v) for v in layer_range]
    if end_idx <= start_idx:
        raise ValueError(f"{context} requires end_idx > start_idx")
    return start_idx, end_idx


def _open_zarr_array(path: str):
    root = zarr.open(path, mode="r")
    if hasattr(root, "shape"):
        return root
    return root["0"]


def _read_xy_array(array) -> np.ndarray:
    shape = tuple(int(x) for x in array.shape)
    if len(shape) == 2:
        return np.asarray(array)
    if len(shape) == 3:
        return np.asarray(array[int(shape[0] // 2)])
    raise ValueError(f"expected 2D or 3D zarr array, got shape={shape}")


def _xy_shape_from_array(array) -> tuple[int, int]:
    shape = tuple(int(x) for x in array.shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3:
        return int(shape[1]), int(shape[2])
    raise ValueError(f"expected 2D or 3D zarr array, got shape={shape}")


def _read_xy_array_region(array, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    shape = tuple(int(x) for x in array.shape)
    if len(shape) == 2:
        return np.asarray(array[y0:y1, x0:x1])
    if len(shape) == 3:
        return np.asarray(array[int(shape[0] // 2), y0:y1, x0:x1])
    raise ValueError(f"expected 2D or 3D zarr array, got shape={shape}")


def _segment_array_path(
    layout: NestedZarrLayout,
    segment_id: str,
    *,
    array_name: str,
    suffix: str = "",
    required: bool = True,
):
    segment_dir = layout.resolve_segment_dir(segment_id)
    return resolve_segment_artifact_path(
        segment_dir,
        segment_id,
        artifact_name=array_name,
        suffix=suffix,
        required=required,
    )


def _image_hw(image_shape_hw) -> tuple[int, int]:
    return int(image_shape_hw[0]), int(image_shape_hw[1])


def _validate_xy_shape(*, segment_id: str, image_shape_hw, array, array_name: str) -> None:
    image_hw = _image_hw(image_shape_hw)
    array_hw = _xy_shape_from_array(array)
    if image_hw != array_hw:
        raise ValueError(
            f"{segment_id}: image and {array_name} must have the same shape; "
            f"got image={image_hw} and {array_name}={array_hw}"
        )


def _open_validated_xy_array(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    *,
    array_name: str,
    suffix: str = "",
    required: bool = True,
):
    path = _segment_array_path(
        layout,
        segment_id,
        array_name=array_name,
        suffix=suffix,
        required=required,
    )
    if not path.exists():
        return None
    array = _open_zarr_array(str(path))
    _validate_xy_shape(
        segment_id=segment_id,
        image_shape_hw=image_shape_hw,
        array=array,
        array_name=array_name,
    )
    return array


def _clip_bbox_to_image(bbox, *, image_shape_hw) -> tuple[int, int, int, int]:
    image_h, image_w = _image_hw(image_shape_hw)
    y0, y1, x0, x1 = [int(v) for v in bbox]
    y0 = max(0, min(y0, image_h))
    y1 = max(0, min(y1, image_h))
    x0 = max(0, min(x0, image_w))
    x1 = max(0, min(x1, image_w))
    return y0, y1, x0, x1


def _as_uint8(array) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return array


def _combine_masks(mask_names, *, load_mask) -> np.ndarray | None:
    combined_mask = None
    for current_mask_name in mask_names:
        current_mask = load_mask(current_mask_name)
        if current_mask is None:
            return None
        current_mask = _as_uint8(current_mask)
        combined_mask = current_mask if combined_mask is None else np.maximum(combined_mask, current_mask)
    return None if combined_mask is None else np.asarray(combined_mask, dtype=np.uint8)


def _read_clipped_xy_region(array, bbox, *, image_shape_hw) -> np.ndarray:
    y0, y1, x0, x1 = _clip_bbox_to_image(bbox, image_shape_hw=image_shape_hw)
    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 0), dtype=np.uint8)
    return _as_uint8(_read_xy_array_region(array, (y0, y1, x0, x1)))


def read_supervision_mask_for_shape(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    *,
    mask_suffix: str = "",
    mask_names,
) -> np.ndarray:
    """Load the full-resolution supervision mask after validating its XY shape."""
    combined_mask = _combine_masks(
        normalize_mask_names(mask_names=mask_names),
        load_mask=lambda current_mask_name: _read_xy_array(
            _open_validated_xy_array(
                layout,
                segment_id,
                image_shape_hw,
                array_name=current_mask_name,
                suffix=mask_suffix,
            )
        ),
    )
    return np.asarray(combined_mask, dtype=np.uint8)


def read_optional_supervision_mask_for_shape(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    *,
    mask_suffix: str = "",
    mask_names,
) -> np.ndarray | None:
    """Load a full-resolution supervision mask if present, otherwise return None."""
    return _combine_masks(
        normalize_mask_names(mask_names=mask_names),
        load_mask=lambda current_mask_name: (
            None
            if (
                supervision_mask_array := _open_validated_xy_array(
                    layout,
                    segment_id,
                    image_shape_hw,
                    array_name=current_mask_name,
                    suffix=mask_suffix,
                    required=False,
                )
            )
            is None
            else _read_xy_array(supervision_mask_array)
        ),
    )


def read_label_and_supervision_mask_region(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    bbox,
    *,
    label_suffix: str = "",
    mask_suffix: str = "",
    mask_names,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a clipped label/mask window, returning empty arrays for empty regions."""
    label_array = _open_validated_xy_array(
        layout,
        segment_id,
        image_shape_hw,
        array_name="inklabels",
        suffix=label_suffix,
    )
    combined_mask = _combine_masks(
        normalize_mask_names(mask_names=mask_names),
        load_mask=lambda current_mask_name: _read_clipped_xy_region(
            _open_validated_xy_array(
                layout,
                segment_id,
                image_shape_hw,
                array_name=current_mask_name,
                suffix=mask_suffix,
            ),
            bbox,
            image_shape_hw=image_shape_hw,
        ),
    )
    return (
        _read_clipped_xy_region(label_array, bbox, image_shape_hw=image_shape_hw),
        np.asarray(combined_mask, dtype=np.uint8),
    )


def read_label_region(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    bbox,
    *,
    label_suffix: str = "",
) -> np.ndarray:
    """Load a clipped label window, returning an empty array for empty regions."""
    label_array = _open_validated_xy_array(
        layout,
        segment_id,
        image_shape_hw,
        array_name="inklabels",
        suffix=label_suffix,
    )
    return _read_clipped_xy_region(label_array, bbox, image_shape_hw=image_shape_hw)


def read_supervision_mask_region(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    bbox,
    *,
    mask_suffix: str = "",
    mask_names,
) -> np.ndarray:
    """Load a clipped supervision-mask window, returning an empty array for empty regions."""
    combined_mask = _combine_masks(
        normalize_mask_names(mask_names=mask_names),
        load_mask=lambda current_mask_name: _read_clipped_xy_region(
            _open_validated_xy_array(
                layout,
                segment_id,
                image_shape_hw,
                array_name=current_mask_name,
                suffix=mask_suffix,
            ),
            bbox,
            image_shape_hw=image_shape_hw,
        ),
    )
    return np.asarray(combined_mask, dtype=np.uint8)


def _from_uint16_to_uint8(array: np.ndarray, *, segment_id: str) -> np.ndarray:
    """Downcast uint16 volumes by dropping the low byte for the 8-bit pipeline."""
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.uint16:
        return (array >> 8).astype(np.uint8, copy=False)
    raise TypeError(
        f"{segment_id}: unsupported zarr dtype for 8-bit pipeline: {array.dtype}. "
        "Expected uint8 or uint16."
    )


class ZarrSegmentVolume:
    def __init__(
        self,
        layout: NestedZarrLayout,
        segment_id: str,
        *,
        layer_range,
        reverse_layers: bool = False,
        in_channels: int | None = None,
    ):
        self.layout = layout
        self.segment_id = str(segment_id)
        segment_dir = self.layout.resolve_segment_dir(self.segment_id)
        volume_path = segment_dir / f"{self.segment_id}.zarr"
        if not volume_path.exists():
            raise FileNotFoundError(
                f"Could not resolve zarr volume for {self.segment_id!r} inside {str(segment_dir)!r}. "
                f"Expected {str(volume_path)!r}."
            )
        self.path = str(volume_path)
        self._zarr_array = None
        array = self._open_zarr_array(self.path)
        raw_shape = tuple(int(x) for x in array.shape)
        if len(raw_shape) != 3:
            raise ValueError(f"{self.segment_id}: expected 3D zarr volume, got shape={raw_shape} at {self.path}")
        self.layer_range = parse_layer_range_value(
            layer_range,
            context=f"segments[{self.segment_id!r}].layer_range",
            total_layers=int(raw_shape[0]),
        )
        self.reverse_layers = bool(reverse_layers)
        self._dtype = np.dtype(array.dtype)
        self._base_h = int(raw_shape[1])
        self._base_w = int(raw_shape[2])
        start_idx, end_idx = self.layer_range
        layer_indices = list(range(int(start_idx), int(end_idx)))
        if in_channels is not None:
            requested = int(in_channels)
            if requested <= 0:
                raise ValueError("in_channels must be positive")
            if len(layer_indices) < requested:
                raise ValueError(
                    f"{self.segment_id}: expected at least {requested} layers, got {len(layer_indices)} "
                    f"from range {start_idx}-{end_idx}"
                )
            if len(layer_indices) > requested:
                start = max(0, (len(layer_indices) - requested) // 2)
                layer_indices = layer_indices[start:start + requested]
        if self.reverse_layers:
            layer_indices.reverse()
        self._in_channels = int(len(layer_indices))
        self._reverse_layer_read = bool(self.reverse_layers)
        self._z_slice_start = int(min(layer_indices))
        self._z_slice_stop = int(max(layer_indices)) + 1
        expected_layer_indices = list(range(self._z_slice_start, self._z_slice_stop))
        if self._reverse_layer_read:
            expected_layer_indices.reverse()
        if layer_indices != expected_layer_indices:
            raise ValueError(
                f"{self.segment_id}: selected zarr layers must be contiguous; got {layer_indices}"
            )
        self._zarr_array = array

        pad_h = int((256 - (self._base_h % 256)) % 256)
        pad_w = int((256 - (self._base_w % 256)) % 256)
        self._out_h = int(self._base_h + pad_h)
        self._out_w = int(self._base_w + pad_w)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_zarr_array"] = None
        return state

    @staticmethod
    def _open_zarr_array(path: str):
        return _open_zarr_array(path)

    @property
    def shape(self) -> tuple[int, int, int]:
        return int(self._out_h), int(self._out_w), int(self._in_channels)

    @property
    def image_shape_hw(self) -> tuple[int, int]:
        return int(self._base_h), int(self._base_w)

    def read_patch(self, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
        """Read an HWC patch and zero-pad any area outside the stored volume."""
        y1 = int(y1)
        y2 = int(y2)
        x1 = int(x1)
        x2 = int(x2)
        patch = np.zeros((y2 - y1, x2 - x1, int(self._in_channels)), dtype=self._dtype)
        yy1 = max(0, y1)
        yy2 = min(int(self._base_h), y2)
        xx1 = max(0, x1)
        xx2 = min(int(self._base_w), x2)
        if yy2 > yy1 and xx2 > xx1:
            array = self._zarr_array
            if array is None:
                array = self._zarr_array = self._open_zarr_array(self.path)
            data = array[self._z_slice_start:self._z_slice_stop, yy1:yy2, xx1:xx2]
            if self._reverse_layer_read:
                data = data[::-1]
            block = np.transpose(np.asarray(data), (1, 2, 0))
            patch[yy1 - y1:yy2 - y1, xx1 - x1:xx2 - x1, :] = block
        return _from_uint16_to_uint8(patch, segment_id=self.segment_id)


class ZarrSegmentLabelMaskStore:
    def __init__(
        self,
        *,
        layout: NestedZarrLayout,
        segment_id: str,
        image_shape_hw: tuple[int, int],
        label_suffix: str = "",
        mask_suffix: str = "",
        mask_names,
        bbox_rows=(),
    ):
        self.layout = layout
        self.segment_id = str(segment_id)
        self.image_shape_hw = (int(image_shape_hw[0]), int(image_shape_hw[1]))
        self.label_suffix = str(label_suffix)
        self.mask_suffix = str(mask_suffix)
        self.mask_names = normalize_mask_names(mask_names=mask_names)
        self.mask_name = self.mask_names[-1]
        self.bbox_rows = self._normalize_bbox_rows(bbox_rows)
        self._label_bbox_cache: dict[int, np.ndarray] = {}
        self._bbox_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _normalize_bbox_rows(self, bbox_rows) -> tuple[tuple[int, int, int, int], ...]:
        rows = np.asarray(bbox_rows, dtype=np.int32)
        if rows.size == 0:
            return ()
        if rows.ndim != 2 or int(rows.shape[1]) != 4:
            raise ValueError(f"bbox_rows must have shape (N, 4), got {tuple(rows.shape)}")
        return tuple(tuple(int(value) for value in row) for row in rows.tolist())

    def set_bbox_rows(self, bbox_rows) -> None:
        normalized = self._normalize_bbox_rows(bbox_rows)
        if normalized:
            self.bbox_rows = normalized

    def _bbox_local_coords(
        self,
        *,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        bbox_index: int,
    ) -> tuple[int, int, int, int, int] | None:
        bbox_index = int(bbox_index)
        if bbox_index < 0 or bbox_index >= len(self.bbox_rows):
            return None
        bbox_y0, _bbox_y1, bbox_x0, _bbox_x1 = self.bbox_rows[bbox_index]
        return (
            bbox_index,
            int(y1) - int(bbox_y0),
            int(y2) - int(bbox_y0),
            int(x1) - int(bbox_x0),
            int(x2) - int(bbox_x0),
        )

    def _label_and_mask_bbox_crops(self, bbox_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._bbox_cache.get(int(bbox_index))
        if cached is None:
            bbox = self.bbox_rows[int(bbox_index)]
            cached = read_label_and_supervision_mask_region(
                self.layout,
                self.segment_id,
                self.image_shape_hw,
                bbox,
                label_suffix=self.label_suffix,
                mask_suffix=self.mask_suffix,
                mask_names=self.mask_names,
            )
            self._bbox_cache[int(bbox_index)] = cached
        return cached

    def _label_bbox_crop(self, bbox_index: int) -> np.ndarray:
        cached = self._bbox_cache.get(int(bbox_index))
        if cached is not None:
            return cached[0]

        label_crop = self._label_bbox_cache.get(int(bbox_index))
        if label_crop is None:
            bbox = self.bbox_rows[int(bbox_index)]
            label_crop = read_label_region(
                self.layout,
                self.segment_id,
                self.image_shape_hw,
                bbox,
                label_suffix=self.label_suffix,
            )
            self._label_bbox_cache[int(bbox_index)] = label_crop
        return label_crop

    def read_patch(
        self,
        *,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        bbox_index: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        bbox_query = self._bbox_local_coords(y1=y1, y2=y2, x1=x1, x2=x2, bbox_index=bbox_index)
        if bbox_query is None:
            return read_label_and_supervision_mask_region(
                self.layout,
                self.segment_id,
                self.image_shape_hw,
                (int(y1), int(y2), int(x1), int(x2)),
                label_suffix=self.label_suffix,
                mask_suffix=self.mask_suffix,
                mask_names=self.mask_names,
            )
        bbox_index, local_y1, local_y2, local_x1, local_x2 = bbox_query
        label_crop, supervision_crop = self._label_and_mask_bbox_crops(bbox_index)
        return (
            _read_mask_patch(label_crop, y1=local_y1, y2=local_y2, x1=local_x1, x2=local_x2),
            _read_mask_patch(supervision_crop, y1=local_y1, y2=local_y2, x1=local_x1, x2=local_x2),
        )

    def read_label_patch(
        self,
        *,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        bbox_index: int = -1,
    ) -> np.ndarray:
        bbox_query = self._bbox_local_coords(y1=y1, y2=y2, x1=x1, x2=x2, bbox_index=bbox_index)
        if bbox_query is None:
            return read_label_region(
                self.layout,
                self.segment_id,
                self.image_shape_hw,
                (int(y1), int(y2), int(x1), int(x2)),
                label_suffix=self.label_suffix,
            )
        bbox_index, local_y1, local_y2, local_x1, local_x2 = bbox_query
        label_crop = self._label_bbox_crop(bbox_index)
        return _read_mask_patch(label_crop, y1=local_y1, y2=local_y2, x1=local_x1, x2=local_x2)


def _read_mask_patch(mask, *, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    if y2 <= y1 or x2 <= x1:
        raise ValueError(f"invalid patch coords: {(x1, y1, x2, y2)}")

    mask = np.asarray(mask)
    out = np.zeros((int(y2 - y1), int(x2 - x1)), dtype=mask.dtype)
    yy1 = max(0, int(y1))
    yy2 = min(int(mask.shape[0]), int(y2))
    xx1 = max(0, int(x1))
    xx2 = min(int(mask.shape[1]), int(x2))
    if yy2 > yy1 and xx2 > xx1:
        out[yy1 - int(y1):yy2 - int(y1), xx1 - int(x1):xx2 - int(x1)] = mask[yy1:yy2, xx1:xx2]
    return out


def resolve_segment_volume(
    *,
    layout: NestedZarrLayout,
    segments,
    segment_id: str,
    in_channels: int,
    volume_cache: dict[Any, ZarrSegmentVolume],
) -> ZarrSegmentVolume:
    segment_spec = segments[segment_id]
    raw_layer_range = segment_spec.get("layer_range")
    reverse_layers = bool(segment_spec.get("reverse_layers", False))
    layer_range = None
    if raw_layer_range is not None:
        layer_range = parse_layer_range_value(
            raw_layer_range,
            context=f"segments[{segment_id!r}].layer_range",
        )

    volume_key = (segment_id, layer_range, reverse_layers, in_channels)
    volume = volume_cache.get(volume_key)
    if volume is None:
        volume = ZarrSegmentVolume(
            layout,
            segment_id,
            layer_range=layer_range,
            reverse_layers=reverse_layers,
            in_channels=in_channels,
        )
        volume_cache[volume_key] = volume
    return volume


def resolve_segment_label_mask_store(
    *,
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw: tuple[int, int],
    label_suffix: str,
    mask_suffix: str,
    mask_names,
    label_mask_store_cache: dict[Any, ZarrSegmentLabelMaskStore],
    bbox_rows=(),
) -> ZarrSegmentLabelMaskStore:
    resolved_mask_names = normalize_mask_names(mask_names=mask_names)
    cache_key = (str(segment_id), str(label_suffix), str(mask_suffix), resolved_mask_names)
    label_mask_store = label_mask_store_cache.get(cache_key)
    if label_mask_store is None:
        label_mask_store = ZarrSegmentLabelMaskStore(
            layout=layout,
            segment_id=str(segment_id),
            image_shape_hw=image_shape_hw,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            mask_names=resolved_mask_names,
            bbox_rows=bbox_rows,
        )
        label_mask_store_cache[cache_key] = label_mask_store
        return label_mask_store
    if tuple(label_mask_store.image_shape_hw) != (int(image_shape_hw[0]), int(image_shape_hw[1])):
        raise ValueError(
            f"{segment_id}: label/mask store shape mismatch {label_mask_store.image_shape_hw} vs {tuple(image_shape_hw)}"
        )
    if bbox_rows is not None and int(np.asarray(bbox_rows).size) > 0:
        label_mask_store.set_bbox_rows(bbox_rows)
    return label_mask_store


__all__ = [
    "parse_layer_range_value",
    "read_label_region",
    "read_label_and_supervision_mask_region",
    "read_optional_supervision_mask_for_shape",
    "read_supervision_mask_region",
    "read_supervision_mask_for_shape",
    "resolve_segment_label_mask_store",
    "resolve_segment_volume",
    "ZarrSegmentLabelMaskStore",
    "ZarrSegmentVolume",
]
