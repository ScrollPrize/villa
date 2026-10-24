"""Batch tensors: moving nested batches to a device and padding to a fixed row count."""
import torch


def fixed_rows(value, size):
    """Pad with finite copies; the caller discards the extra output rows."""
    if not 0 < len(value) <= size:
        raise ValueError(f'Expected 1..{size} rows, got {len(value)}')
    if len(value) < size:
        value = torch.cat((value, value[:1].expand(size-len(value), *value.shape[1:])))
    stride = 1
    canonical = []
    for extent in reversed(value.shape):
        canonical.append(stride)
        stride *= max(1, extent)
    if value.stride() == tuple(reversed(canonical)):
        return value
    # Singleton dimensions can otherwise retain the stride of an indexed view.
    return torch.empty_like(value, memory_format=torch.contiguous_format).copy_(value)


def move_batch(batch, device):
    # Pinned loader batches copy asynchronously; the compute stream orders later use.
    return {k: move_batch(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True)
            for k, v in batch.items() if not k.startswith('_')}
