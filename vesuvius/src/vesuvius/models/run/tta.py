import torch


def _concat_if_multi_task(output, is_multi_task: bool, concat_fn):
    if not is_multi_task:
        return output
    if concat_fn is None:
        raise ValueError("concat_fn must be provided for multi-task models")
    return concat_fn(output)


def infer_with_tta(model,
                   inputs: torch.Tensor,
                   tta_type: str = 'mirroring',
                   *,
                   is_multi_task: bool = False,
                   concat_multi_task_outputs=None,
                   batched: bool = False) -> torch.Tensor:
    """
    Apply TTA for 3D or 2D models.

    - For 3D, inputs: (B, C, D, H, W) → returns (B, C, D, H, W)
    - For 2D, inputs: (B, C, H, W) → returns (B, C, H, W)

    - tta_type: 'mirroring' uses 8 flip combinations
                'rotation' uses axis transpositions to rotate volumes
    - is_multi_task: if True, model returns a dict; provide concat_multi_task_outputs
                     to concatenate dict outputs into a tensor (B, C, ...)
    - batched: if True, run all augmentations in one forward by concatenating along batch
    """
    if tta_type not in ('mirroring', 'rotation'):
        raise ValueError(f"Unsupported tta_type: {tta_type}")

    # Determine number of spatial dims independent of channel count
    ndim = inputs.ndim
    if ndim < 4:
        raise ValueError(f"infer_with_tta expects at least 4D input (B,C,...) got {ndim}D")
    spatial_dims = ndim - 2  # subtract batch and channel dims
    if spatial_dims not in (2, 3):
        raise ValueError(f"infer_with_tta expects 2D or 3D spatial dims, got {spatial_dims}")

    if tta_type == 'mirroring':
        if spatial_dims == 3:
            augments = [
                (lambda t: t, lambda t: t),
                (lambda t: torch.flip(t, dims=[-1]), lambda t: torch.flip(t, dims=[-1])),
                (lambda t: torch.flip(t, dims=[-2]), lambda t: torch.flip(t, dims=[-2])),
                (lambda t: torch.flip(t, dims=[-3]), lambda t: torch.flip(t, dims=[-3])),
                (lambda t: torch.flip(t, dims=[-1, -2]), lambda t: torch.flip(t, dims=[-1, -2])),
                (lambda t: torch.flip(t, dims=[-1, -3]), lambda t: torch.flip(t, dims=[-1, -3])),
                (lambda t: torch.flip(t, dims=[-2, -3]), lambda t: torch.flip(t, dims=[-2, -3])),
                (lambda t: torch.flip(t, dims=[-1, -2, -3]), lambda t: torch.flip(t, dims=[-1, -2, -3])),
            ]
        else:  # 2D flips over H and W
            augments = [
                (lambda t: t, lambda t: t),
                (lambda t: torch.flip(t, dims=[-1]), lambda t: torch.flip(t, dims=[-1])),
                (lambda t: torch.flip(t, dims=[-2]), lambda t: torch.flip(t, dims=[-2])),
                (lambda t: torch.flip(t, dims=[-2, -1]), lambda t: torch.flip(t, dims=[-2, -1])),
            ]
    else:  # rotation
        if spatial_dims == 3:
            augments = [
                (lambda t: t, lambda t: t),
                (lambda t: torch.transpose(t, -3, -1), lambda t: torch.transpose(t, -3, -1)),
                (lambda t: torch.transpose(t, -3, -2), lambda t: torch.transpose(t, -3, -2)),
            ]
        else:  # 2D: use transpose(H,W) as rotation
            augments = [
                (lambda t: t, lambda t: t),
                (lambda t: torch.transpose(t, -2, -1), lambda t: torch.transpose(t, -2, -1)),
            ]

    if not batched:
        outputs = []
        for apply_fn, invert_fn in augments:
            aug_inputs = apply_fn(inputs)
            out = model(aug_inputs)
            out = _concat_if_multi_task(out, is_multi_task, concat_multi_task_outputs)
            outputs.append(invert_fn(out))
        return torch.mean(torch.stack(outputs, dim=0), dim=0)

    base_bs = inputs.shape[0]
    augmented_inputs = [apply_fn(inputs) for apply_fn, _ in augments]
    cat_inputs = torch.cat(augmented_inputs, dim=0)
    merged = model(cat_inputs)
    merged = _concat_if_multi_task(merged, is_multi_task, concat_multi_task_outputs)

    chunks = torch.split(merged, base_bs, dim=0)
    outputs = []
    for idx, (_, invert_fn) in enumerate(augments):
        outputs.append(invert_fn(chunks[idx]))
    return torch.mean(torch.stack(outputs, dim=0), dim=0)
