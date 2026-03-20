from __future__ import annotations

from ink.core.types import Batch, BatchMeta


def move_batch_to_device(batch: Batch, *, device=None) -> Batch:
    if device is None:
        return batch

    def _to(value):
        if value is None or not hasattr(value, "to"):
            return value
        return value.to(device=device, non_blocking=True)

    return Batch(
        x=_to(batch.x),
        y=_to(batch.y),
        meta=BatchMeta(
            segment_ids=list(batch.meta.segment_ids),
            valid_mask=_to(batch.meta.valid_mask),
            patch_xyxy=batch.meta.patch_xyxy,
            group_idx=_to(batch.meta.group_idx),
        ),
    )
