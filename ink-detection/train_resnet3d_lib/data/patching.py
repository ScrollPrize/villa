import cv2
import numpy as np
from scipy import ndimage

from train_resnet3d_lib.config import CFG


def _label_tile_is_empty(label_tile) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def extract_patch_coordinates(
    mask,
    fragment_mask,
    *,
    filter_empty_tile,
):
    xyxys = []
    stride = CFG.stride
    x1_list = list(range(0, fragment_mask.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, fragment_mask.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if filter_empty_tile and mask is not None and _label_tile_is_empty(
                mask[a:a + CFG.tile_size, b:b + CFG.tile_size]
            ):
                continue
            tile_has_invalid = bool(np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0))
            if tile_has_invalid and filter_empty_tile:
                continue

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue
                    if tile_has_invalid and (not filter_empty_tile) and np.any(fragment_mask[y1:y2, x1:x2] == 0):
                        continue
                    windows_dict[(y1, y2, x1, x2)] = True
                    xyxys.append([x1, y1, x2, y2])
    if len(xyxys) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(xyxys, dtype=np.int64)


def _component_bboxes(mask, *, connectivity=2):
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8, copy=False)
    if mask_u8.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape={tuple(mask_u8.shape)}")
    if not bool(mask_u8.any()):
        return np.zeros((0, 4), dtype=np.int32)

    cc_conn = 4 if int(connectivity) == 1 else 8
    n_all, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=cc_conn)
    bboxes = []
    for li in range(1, int(n_all)):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        w = int(stats[li, cv2.CC_STAT_WIDTH])
        h = int(stats[li, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        bboxes.append((y, y + h, x, x + w))
    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.int32)
    bboxes.sort(key=lambda b: (int(b[0]), int(b[2]), int(b[1]), int(b[3])))
    return np.asarray(bboxes, dtype=np.int32)


def _build_mask_store_and_patch_index(
    mask,
    fragment_mask,
    *,
    filter_empty_tile,
):
    mask_u8 = np.asarray(mask)
    if mask_u8.ndim != 2:
        raise ValueError(f"expected 2D label mask, got shape={tuple(mask_u8.shape)}")
    if mask_u8.dtype != np.uint8:
        mask_u8 = np.clip(mask_u8, 0, 255).astype(np.uint8, copy=False)

    fragment_mask = np.asarray(fragment_mask)
    if fragment_mask.ndim != 2:
        raise ValueError(f"expected 2D fragment mask, got shape={tuple(fragment_mask.shape)}")
    if fragment_mask.shape != mask_u8.shape:
        raise ValueError(
            f"label/fragment mask shape mismatch: {tuple(mask_u8.shape)} vs {tuple(fragment_mask.shape)}"
        )

    bboxes = _component_bboxes(fragment_mask, connectivity=2)
    if int(bboxes.shape[0]) == 0:
        xyxys = extract_patch_coordinates(mask_u8, fragment_mask, filter_empty_tile=bool(filter_empty_tile))
        bbox_idx = np.full((int(xyxys.shape[0]),), -1, dtype=np.int32)
        return (
            {"mode": "full", "shape": tuple(mask_u8.shape), "mask": mask_u8},
            xyxys,
            bbox_idx,
        )

    mask_crops = []
    kept_bboxes = []
    xy_chunks = []
    bbox_chunks = []
    seen_windows = set()
    for y0, y1, x0, x1 in bboxes.tolist():
        y0 = int(y0)
        y1 = int(y1)
        x0 = int(x0)
        x1 = int(x1)
        if y1 <= y0 or x1 <= x0:
            continue
        mask_c = np.asarray(mask_u8[y0:y1, x0:x1], dtype=np.uint8).copy()
        fragment_mask_c = fragment_mask[y0:y1, x0:x1]
        xy_local = extract_patch_coordinates(mask_c, fragment_mask_c, filter_empty_tile=bool(filter_empty_tile))
        if int(xy_local.shape[0]) == 0:
            continue

        xy_global_rows = []
        for x1_l, y1_l, x2_l, y2_l in np.asarray(xy_local, dtype=np.int64).tolist():
            gx1 = int(x1_l) + int(x0)
            gy1 = int(y1_l) + int(y0)
            gx2 = int(x2_l) + int(x0)
            gy2 = int(y2_l) + int(y0)
            key = (gx1, gy1, gx2, gy2)
            if key in seen_windows:
                continue
            seen_windows.add(key)
            xy_global_rows.append([gx1, gy1, gx2, gy2])
        if len(xy_global_rows) == 0:
            continue

        local_bbox_idx = int(len(mask_crops))
        xy_global = np.asarray(xy_global_rows, dtype=np.int64)

        mask_crops.append(mask_c)
        kept_bboxes.append((int(y0), int(y1), int(x0), int(x1)))
        xy_chunks.append(xy_global)
        bbox_chunks.append(np.full((int(xy_global.shape[0]),), local_bbox_idx, dtype=np.int32))

    if len(xy_chunks) == 0:
        return (
            {"mode": "full", "shape": tuple(mask_u8.shape), "mask": mask_u8},
            np.zeros((0, 4), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )

    xyxys = np.concatenate(xy_chunks, axis=0)
    bbox_idx = np.concatenate(bbox_chunks, axis=0)
    return (
        {
            "mode": "bboxes",
            "shape": tuple(mask_u8.shape),
            "bboxes": np.asarray(kept_bboxes, dtype=np.int32),
            "mask_crops": mask_crops,
        },
        xyxys,
        bbox_idx,
    )


def _mask_store_shape(mask_store):
    if isinstance(mask_store, np.ndarray):
        if mask_store.ndim < 2:
            raise ValueError(f"expected at least 2D mask array, got shape={tuple(mask_store.shape)}")
        return (int(mask_store.shape[0]), int(mask_store.shape[1]))
    if isinstance(mask_store, dict):
        shape = mask_store.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("mask store missing valid 'shape'")
        return int(shape[0]), int(shape[1])
    raise TypeError(f"unsupported mask store type: {type(mask_store).__name__}")


def _read_mask_patch(mask_store, *, y1, y2, x1, x2, bbox_index=None):
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    if y2 <= y1 or x2 <= x1:
        raise ValueError(f"invalid patch coords: {(x1, y1, x2, y2)}")

    if isinstance(mask_store, np.ndarray):
        return np.asarray(mask_store[y1:y2, x1:x2])

    if not isinstance(mask_store, dict):
        raise TypeError(f"unsupported mask store type: {type(mask_store).__name__}")

    mode = str(mask_store.get("mode", "full"))
    if mode == "full":
        if "mask" not in mask_store:
            raise ValueError("mask store mode='full' is missing key 'mask'")
        mask_arr = np.asarray(mask_store["mask"])
        return np.asarray(mask_arr[y1:y2, x1:x2])

    if mode != "bboxes":
        raise ValueError(f"unsupported mask store mode: {mode!r}")

    bboxes = np.asarray(mask_store.get("bboxes"))
    mask_crops = list(mask_store.get("mask_crops", []))
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("mask store mode='bboxes' requires bboxes shape (N, 4)")
    if int(bboxes.shape[0]) != int(len(mask_crops)):
        raise ValueError("mask store mode='bboxes' requires matching bboxes and mask_crops lengths")

    idx = None
    if bbox_index is not None:
        idx_i = int(bbox_index)
        if idx_i >= 0:
            idx = idx_i
    if idx is None:
        for i, bbox in enumerate(bboxes.tolist()):
            by0, by1, bx0, bx1 = [int(v) for v in bbox]
            if y1 >= by0 and y2 <= by1 and x1 >= bx0 and x2 <= bx1:
                idx = int(i)
                break
    if idx is None:
        raise ValueError(f"could not resolve bbox for patch coords {(x1, y1, x2, y2)}")
    if idx < 0 or idx >= int(len(mask_crops)):
        raise ValueError(f"bbox index out of range: {idx}")

    by0, by1, bx0, bx1 = [int(v) for v in bboxes[idx].tolist()]
    ly1 = int(y1 - by0)
    ly2 = int(y2 - by0)
    lx1 = int(x1 - bx0)
    lx2 = int(x2 - bx0)
    if ly1 < 0 or lx1 < 0 or ly2 > (by1 - by0) or lx2 > (bx1 - bx0):
        raise ValueError(
            f"patch {(x1, y1, x2, y2)} is out of bbox bounds {(bx0, by0, bx1, by1)}"
        )
    crop = np.asarray(mask_crops[idx])
    return np.asarray(crop[ly1:ly2, lx1:lx2])


def extract_patches_infer(image, fragment_mask, *, include_xyxys=True):
    images = []
    xyxys = []

    stride = CFG.stride
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            tile_has_invalid = bool(np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0))

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue
                    if tile_has_invalid and np.any(fragment_mask[y1:y2, x1:x2] == 0):
                        continue

                    windows_dict[(y1, y2, x1, x2)] = True
                    images.append(image[y1:y2, x1:x2])
                    if include_xyxys:
                        xyxys.append([x1, y1, x2, y2])
                    assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)

    return images, xyxys


def extract_patches(image, mask, fragment_mask, *, include_xyxys, filter_empty_tile):
    images = []
    masks = []
    xyxys = []

    stride = CFG.stride
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if filter_empty_tile and _label_tile_is_empty(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]):
                continue
            tile_has_invalid = bool(np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0))
            if tile_has_invalid and filter_empty_tile:
                continue

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue
                    if tile_has_invalid and (not filter_empty_tile) and np.any(fragment_mask[y1:y2, x1:x2] == 0):
                        continue

                    windows_dict[(y1, y2, x1, x2)] = True
                    images.append(image[y1:y2, x1:x2])
                    masks.append(mask[y1:y2, x1:x2, None])
                    if include_xyxys:
                        xyxys.append([x1, y1, x2, y2])
                    assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)

    return images, masks, xyxys


def _downsample_bool_mask_any(mask: np.ndarray, ds: int) -> np.ndarray:
    ds = max(1, int(ds))
    if mask is None:
        raise ValueError("mask is None")
    mask_bool = (mask > 0)
    h = int(mask_bool.shape[0])
    w = int(mask_bool.shape[1])
    ds_h = (h + ds - 1) // ds
    ds_w = (w + ds - 1) // ds
    pad_h = int(ds_h * ds - h)
    pad_w = int(ds_w * ds - w)
    if pad_h or pad_w:
        mask_bool = np.pad(mask_bool, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    mask_bool = mask_bool.reshape(ds_h, ds, ds_w, ds)
    return mask_bool.any(axis=(1, 3))


def _mask_component_bboxes_downsample(mask: np.ndarray, ds: int) -> np.ndarray:
    mask_ds = _downsample_bool_mask_any(mask, int(ds))
    if not mask_ds.any():
        return np.zeros((0, 4), dtype=np.int32)
    return _component_bboxes(mask_ds.astype(np.uint8, copy=False), connectivity=2)


def _mask_border(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool is None:
        raise ValueError("mask_bool is None")
    mask_bool = mask_bool.astype(bool, copy=False)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)
    thickness = 5
    eroded = ndimage.binary_erosion(
        mask_bool,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
        iterations=thickness,
    )
    return mask_bool & ~eroded


__all__ = [
    "_label_tile_is_empty",
    "extract_patch_coordinates",
    "_component_bboxes",
    "_build_mask_store_and_patch_index",
    "_mask_store_shape",
    "_read_mask_patch",
    "extract_patches_infer",
    "extract_patches",
    "_downsample_bool_mask_any",
    "_mask_component_bboxes_downsample",
    "_mask_border",
]
