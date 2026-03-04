import numpy as np


def _resolve_roi(shape, bbox, ds, *, use_roi):
    h = int(shape[0])
    w = int(shape[1])
    ds = max(1, int(ds))
    ds_h = (h + ds - 1) // ds
    ds_w = (w + ds - 1) // ds

    if not use_roi or bbox is None:
        return (0, ds_h, 0, ds_w), (ds_h, ds_w)

    y0, y1, x0, x1 = [int(v) for v in bbox]
    y0 = max(0, min(y0, ds_h))
    y1 = max(0, min(y1, ds_h))
    x0 = max(0, min(x0, ds_w))
    x1 = max(0, min(x1, ds_w))
    if y1 <= y0 or x1 <= x0:
        return (0, ds_h, 0, ds_w), (ds_h, ds_w)
    return (y0, y1, x0, x1), (ds_h, ds_w)


def _resolve_rois(shape, bboxes, ds, *, use_roi):
    base_roi, full_shape = _resolve_roi(shape, None, ds, use_roi=use_roi)
    if not use_roi or bboxes is None:
        return [base_roi], full_shape

    bboxes_arr = np.asarray(bboxes, dtype=np.int64)
    if bboxes_arr.ndim == 1:
        if bboxes_arr.shape[0] != 4:
            raise ValueError(f"stitch ROI bbox must have 4 values, got shape={tuple(bboxes_arr.shape)}")
        roi, _ = _resolve_roi(shape, tuple(int(v) for v in bboxes_arr.tolist()), ds, use_roi=use_roi)
        return [roi], full_shape

    if bboxes_arr.ndim != 2 or bboxes_arr.shape[1] != 4:
        raise ValueError(f"stitch ROI bboxes must have shape (N,4), got shape={tuple(bboxes_arr.shape)}")

    ds_h, ds_w = full_shape
    rois = []
    for y0, y1, x0, x1 in bboxes_arr.tolist():
        y0 = max(0, min(int(y0), ds_h))
        y1 = max(0, min(int(y1), ds_h))
        x0 = max(0, min(int(x0), ds_w))
        x1 = max(0, min(int(x1), ds_w))
        if y1 > y0 and x1 > x0:
            rois.append((y0, y1, x0, x1))

    return rois or [base_roi], full_shape


def build_segment_roi_meta(shape, bboxes, ds, *, use_roi):
    rois, (ds_h, ds_w) = _resolve_rois(shape, bboxes, ds, use_roi=use_roi)
    return {
        "full_shape": (int(ds_h), int(ds_w)),
        "rois": [
            {
                "offset": (int(y0), int(x0)),
                "buffer_shape": (max(1, int(y1 - y0)), max(1, int(x1 - x0))),
            }
            for y0, y1, x0, x1 in rois
        ],
    }


def allocate_segment_buffers(roi_meta):
    buffers = []
    for roi in roi_meta.get("rois", []):
        buf_h, buf_w = [int(v) for v in roi["buffer_shape"]]
        offset = tuple(int(v) for v in roi["offset"])
        buffers.append(
            (
                np.zeros((buf_h, buf_w), dtype=np.float32),
                np.zeros((buf_h, buf_w), dtype=np.float32),
                offset,
            )
        )
    return buffers
