import cupy as cp
from cucim.core.operations.morphology import distance_transform_edt


def dilate_label_batch_with_cucim(labels, padding_mask, distance, ignore_label=None):
    if distance in (None, 0):
        return labels

    cp.cuda.Device(labels.device.index).use()
    out = labels.clone()

    if padding_mask.ndim == labels.ndim - 1:
        padding_mask = padding_mask.unsqueeze(1)

    for b in range(out.shape[0]):
        for c in range(out.shape[1]):
            x = cp.from_dlpack(out[b, c].contiguous())
            valid = cp.from_dlpack(padding_mask[b, 0].contiguous())
            source = (x == 1) & (valid > 0)
            dist = distance_transform_edt(~source, return_indices=False, float64_distances=False)
            fill = (x == 0) & (valid > 0) & (dist <= float(distance))
            x[fill] = x.dtype.type(1)

    return out
