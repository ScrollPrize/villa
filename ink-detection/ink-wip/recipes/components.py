from __future__ import annotations

import numpy as np


def _as_bool_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={tuple(array.shape)}")
    return array.astype(bool, copy=False)


def _cc_connectivity_cv2(connectivity: int) -> int:
    connectivity = int(connectivity)
    if connectivity == 1:
        return 4
    if connectivity == 2:
        return 8
    raise ValueError(f"connectivity must be 1 (4-neighborhood) or 2 (8-neighborhood), got {connectivity}")


def label_components(mask: np.ndarray, *, connectivity: int = 2) -> tuple[np.ndarray, int]:
    """Return connected-component labels and count for a binary 2D mask."""
    import cv2

    mask_bool = _as_bool_2d(mask)
    if not mask_bool.any():
        return np.zeros(mask_bool.shape, dtype=np.int32), 0
    n_all, labels = cv2.connectedComponents(
        mask_bool.astype(np.uint8, copy=False),
        connectivity=_cc_connectivity_cv2(connectivity),
    )
    return labels.astype(np.int32, copy=False), int(max(0, int(n_all) - 1))


def component_bboxes(mask: np.ndarray, *, connectivity: int = 2) -> np.ndarray:
    """Return connected-component bounding boxes as (y0, y1, x0, x1) rows."""
    import cv2

    mask_bool = _as_bool_2d(mask)
    if not mask_bool.any():
        return np.zeros((0, 4), dtype=np.int32)

    n_all, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8, copy=False),
        connectivity=_cc_connectivity_cv2(connectivity),
    )
    rows = []
    for label_idx in range(1, int(n_all)):
        x0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        rows.append((y0, y0 + h, x0, x0 + w))
    if not rows:
        return np.zeros((0, 4), dtype=np.int32)
    rows.sort(key=lambda bbox: (int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])))
    return np.asarray(rows, dtype=np.int32)
