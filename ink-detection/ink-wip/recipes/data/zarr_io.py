from __future__ import annotations

import numpy as np
import zarr

from ink.recipes.data.layout import NestedZarrLayout


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


def _read_xy_zarr(path: str) -> np.ndarray:
    """Read a 2D zarr array, or the center slice of a 3D array, as an XY plane."""
    array = _open_zarr_array(path)
    shape = tuple(int(x) for x in array.shape)
    if len(shape) == 2:
        return np.asarray(array)
    if len(shape) == 3:
        return np.asarray(array[int(shape[0] // 2)])
    raise ValueError(f"expected 2D or 3D zarr array at {path!r}, got shape={shape}")


def _xy_shape_from_array(array) -> tuple[int, int]:
    shape = tuple(int(x) for x in array.shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3:
        return int(shape[1]), int(shape[2])
    raise ValueError(f"expected 2D or 3D zarr array, got shape={shape}")


def _read_xy_zarr_region(path: str, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    array = _open_zarr_array(path)
    shape = tuple(int(x) for x in array.shape)
    if len(shape) == 2:
        return np.asarray(array[y0:y1, x0:x1])
    if len(shape) == 3:
        return np.asarray(array[int(shape[0] // 2), y0:y1, x0:x1])
    raise ValueError(f"expected 2D or 3D zarr array at {path!r}, got shape={shape}")


def _resolve_label_and_mask_paths(layout: NestedZarrLayout, segment_id: str, *, label_suffix: str = "", mask_suffix: str = ""):
    return layout.resolve_paths(
        segment_id,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )


def _validate_label_and_mask_shapes(
    *,
    segment_id: str,
    image_shape_hw,
    label_array,
    supervision_mask_array,
) -> None:
    image_h = int(image_shape_hw[0])
    image_w = int(image_shape_hw[1])
    image_hw = (image_h, image_w)
    label_hw = _xy_shape_from_array(label_array)
    supervision_mask_hw = _xy_shape_from_array(supervision_mask_array)
    if image_hw != label_hw or image_hw != supervision_mask_hw:
        raise ValueError(
            f"{segment_id}: image, inklabels, and supervision_mask must have the same shape; "
            f"got image={image_hw}, inklabels={label_hw}, supervision_mask={supervision_mask_hw}"
        )


def read_label_and_supervision_mask_for_shape(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    *,
    label_suffix: str = "",
    mask_suffix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Load full-resolution label and supervision masks after validating their shape."""
    paths = _resolve_label_and_mask_paths(
        layout,
        segment_id,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    label_array = _open_zarr_array(str(paths.inklabels_path))
    supervision_mask_array = _open_zarr_array(str(paths.supervision_mask_path))
    _validate_label_and_mask_shapes(
        segment_id=segment_id,
        image_shape_hw=image_shape_hw,
        label_array=label_array,
        supervision_mask_array=supervision_mask_array,
    )

    label = _read_xy_zarr(str(paths.inklabels_path))
    supervision_mask = _read_xy_zarr(str(paths.supervision_mask_path))

    label = np.asarray(label)
    if label.dtype != np.uint8:
        label = np.clip(label, 0, 255).astype(np.uint8, copy=False)

    supervision_mask = np.asarray(supervision_mask)
    if supervision_mask.dtype != np.uint8:
        supervision_mask = np.clip(supervision_mask, 0, 255).astype(np.uint8, copy=False)
    return label, supervision_mask


def read_optional_supervision_mask_for_shape(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    *,
    mask_suffix: str = "",
) -> np.ndarray | None:
    """Load a full-resolution supervision mask if present, otherwise return None."""
    segment_dir = layout.resolve_segment_dir(segment_id)
    mask_path = segment_dir / f"{str(segment_id)}_supervision_mask{str(mask_suffix)}.zarr"
    if not mask_path.exists():
        return None

    supervision_mask_array = _open_zarr_array(str(mask_path))
    supervision_mask_hw = _xy_shape_from_array(supervision_mask_array)
    image_hw = (int(image_shape_hw[0]), int(image_shape_hw[1]))
    if image_hw != supervision_mask_hw:
        raise ValueError(
            f"{segment_id}: image and supervision_mask must have the same shape; "
            f"got image={image_hw} and supervision_mask={supervision_mask_hw}"
        )

    supervision_mask = _read_xy_zarr(str(mask_path))
    supervision_mask = np.asarray(supervision_mask)
    if supervision_mask.dtype != np.uint8:
        supervision_mask = np.clip(supervision_mask, 0, 255).astype(np.uint8, copy=False)
    return supervision_mask


def read_label_and_supervision_mask_region(
    layout: NestedZarrLayout,
    segment_id: str,
    image_shape_hw,
    bbox,
    *,
    label_suffix: str = "",
    mask_suffix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a clipped label/mask window, returning empty arrays for empty regions."""
    paths = _resolve_label_and_mask_paths(
        layout,
        segment_id,
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
    )
    label_array = _open_zarr_array(str(paths.inklabels_path))
    supervision_mask_array = _open_zarr_array(str(paths.supervision_mask_path))
    _validate_label_and_mask_shapes(
        segment_id=segment_id,
        image_shape_hw=image_shape_hw,
        label_array=label_array,
        supervision_mask_array=supervision_mask_array,
    )

    image_h = int(image_shape_hw[0])
    image_w = int(image_shape_hw[1])
    y0, y1, x0, x1 = [int(v) for v in bbox]
    y0 = max(0, min(y0, image_h))
    y1 = max(0, min(y1, image_h))
    x0 = max(0, min(x0, image_w))
    x1 = max(0, min(x1, image_w))
    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)

    label = _read_xy_zarr_region(str(paths.inklabels_path), (y0, y1, x0, x1))
    supervision_mask = _read_xy_zarr_region(str(paths.supervision_mask_path), (y0, y1, x0, x1))

    label = np.asarray(label)
    if label.dtype != np.uint8:
        label = np.clip(label, 0, 255).astype(np.uint8, copy=False)

    supervision_mask = np.asarray(supervision_mask)
    if supervision_mask.dtype != np.uint8:
        supervision_mask = np.clip(supervision_mask, 0, 255).astype(np.uint8, copy=False)
    return label, supervision_mask


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
        self._layer_indices = np.asarray(layer_indices, dtype=np.int64)
        self._in_channels = int(self._layer_indices.size)
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
        array = self._zarr_array
        if array is None:
            array = self._zarr_array = self._open_zarr_array(self.path)

        patch = np.zeros((y2 - y1, x2 - x1, int(self._in_channels)), dtype=self._dtype)
        yy1 = max(0, y1)
        yy2 = min(int(self._base_h), y2)
        xx1 = max(0, x1)
        xx2 = min(int(self._base_w), x2)
        if yy2 > yy1 and xx2 > xx1:
            block = np.transpose(
                np.asarray(array[self._layer_indices, yy1:yy2, xx1:xx2]),
                (1, 2, 0),
            )
            patch[yy1 - y1:yy2 - y1, xx1 - x1:xx2 - x1, :] = block
        return _from_uint16_to_uint8(patch, segment_id=self.segment_id)


__all__ = [
    "parse_layer_range_value",
    "read_label_and_supervision_mask_for_shape",
    "read_label_and_supervision_mask_region",
    "read_optional_supervision_mask_for_shape",
    "ZarrSegmentVolume",
]
