import torch
import torch.nn.functional as F

def safe_crop_with_padding(tensor, min_corner, crop_size):
    """Crop tensor around min_corner with zero-padding if crop goes outside bounds.

    Args:
        tensor: Tensor to crop, shape [batch, z, y, x, ...] where ... can be additional dims (e.g., channels)
        min_corner: Minimum corner of crop [batch, 3] or [3] in zyx order
        crop_size: Size of the crop (int)

    Returns:
        Cropped tensor with shape [batch, crop_size, crop_size, crop_size, ...]
    """
    crop_size_int = int(crop_size)
    min_corner = torch.broadcast_to(min_corner, (tensor.shape[0], 3))
    max_corner = min_corner + crop_size_int

    # Get tensor spatial shape (last 3 dims before any trailing dimensions)
    spatial_shape = torch.tensor(tensor.shape[1:4], device=min_corner.device, dtype=min_corner.dtype)

    # Clamp to tensor bounds
    actual_min = torch.maximum(min_corner, torch.zeros_like(min_corner))
    actual_max = torch.minimum(max_corner, spatial_shape.unsqueeze(0).expand_as(actual_min))

    # Extract valid portion and pad if needed
    crops = []
    for iib in range(tensor.shape[0]):
        crop = tensor[
            iib,
            actual_min[iib, 0].int().item() : actual_max[iib, 0].int().item(),
            actual_min[iib, 1].int().item() : actual_max[iib, 1].int().item(),
            actual_min[iib, 2].int().item() : actual_max[iib, 2].int().item()
        ]

        # Pad if needed
        pad_before = (actual_min[iib] - min_corner[iib]).int()
        pad_after = (max_corner[iib] - actual_max[iib]).int()

        if torch.any(pad_before > 0) or torch.any(pad_after > 0):
            # F.pad expects padding in reverse order of dimensions (last dim first)
            # For tensor [z, y, x, ...]: pad last 3 dims (spatial) and any trailing dims
            # Format: (pad_for_last_dim_before, pad_for_last_dim_after, ..., pad_for_first_dim_before, pad_for_first_dim_after)
            trailing_dims = crop.dim() - 3
            # Padding for trailing dimensions (if any) - no padding needed
            trailing_padding = (0, 0) * trailing_dims if trailing_dims > 0 else ()
            # Padding for spatial dimensions: x, y, z (in reverse order)
            spatial_padding = (
                pad_before[2].item(), pad_after[2].item(),  # x (last spatial dim)
                pad_before[1].item(), pad_after[1].item(),  # y
                pad_before[0].item(), pad_after[0].item()   # z (first spatial dim)
            )
            paddings = trailing_padding + spatial_padding
            crop = F.pad(crop, paddings, mode='constant', value=0)

        crops.append(crop)

    return torch.stack(crops, dim=0)


def transform_to_first_crop_space(step_pred, offset, crop_size):
    """Transform predictions from a later step's crop space to the first step's crop space.

    Args:
        step_pred: Predictions in later step's crop space [batch, channels, z, y, x]
        offset: Offset from later step crop min corner to first step crop min corner [batch, 3]
        crop_size: Size of the crops

    Returns:
        step_pred transformed to first step's crop space
    """

    output_shape = step_pred.shape[:2] + (crop_size,) * 3
    step_pred_in_first_crop = torch.full(output_shape, step_pred.amin(), device=step_pred.device, dtype=step_pred.dtype)

    src_start = torch.clamp(-offset, 0, crop_size)
    src_end = torch.clamp(crop_size - offset, 0, crop_size)
    dst_start = torch.clamp(offset, 0, crop_size)
    dst_end = torch.clamp(crop_size + offset, 0, crop_size)

    for iib in range(step_pred.shape[0]):
        src_slice = (slice(iib, iib+1), slice(None),
                    slice(src_start[iib, 0].item(), src_end[iib, 0].item()),
                    slice(src_start[iib, 1].item(), src_end[iib, 1].item()),
                    slice(src_start[iib, 2].item(), src_end[iib, 2].item()))
        dst_slice = (slice(iib, iib+1), slice(None),
                    slice(dst_start[iib, 0].item(), dst_end[iib, 0].item()),
                    slice(dst_start[iib, 1].item(), dst_end[iib, 1].item()),
                    slice(dst_start[iib, 2].item(), dst_end[iib, 2].item()))

        if (dst_end[iib] > dst_start[iib]).all() and (src_end[iib] > src_start[iib]).all():
            step_pred_in_first_crop[dst_slice] = step_pred[src_slice]

    return step_pred_in_first_crop