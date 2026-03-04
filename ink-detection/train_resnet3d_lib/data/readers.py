"""Stable public reader API.

Keep importing from here if you need backwards compatibility.
Prefer direct imports from `image_readers` and `zarr_volume` for new code.
"""

from train_resnet3d_lib.data.image_readers import (
    _assert_bottom_right_pad_compatible_global,
    _clip_intensity_inplace,
    _compute_selected_layer_indices,
    _parse_layer_range,
    _read_gray,
    _require_dict,
    build_group_mappings,
    read_fragment_mask_for_shape,
    read_image_fragment_mask,
    read_image_layers,
    read_image_mask,
    read_label_and_fragment_mask_for_shape,
)
from train_resnet3d_lib.data.zarr_volume import (
    ZarrSegmentVolume,
    _ensure_zarr_v2,
    _from_uint16_to_uint8,
    _looks_like_zarr_store,
    resolve_segment_zarr_path,
)

__all__ = [
    "_require_dict",
    "_read_gray",
    "_parse_layer_range",
    "_compute_selected_layer_indices",
    "_clip_intensity_inplace",
    "read_image_layers",
    "_assert_bottom_right_pad_compatible_global",
    "read_image_mask",
    "read_image_fragment_mask",
    "_looks_like_zarr_store",
    "resolve_segment_zarr_path",
    "_ensure_zarr_v2",
    "_from_uint16_to_uint8",
    "ZarrSegmentVolume",
    "read_label_and_fragment_mask_for_shape",
    "read_fragment_mask_for_shape",
    "build_group_mappings",
]
