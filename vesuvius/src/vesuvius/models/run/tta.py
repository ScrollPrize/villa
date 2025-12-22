import torch


def _concat_if_multi_task(output, is_multi_task: bool, concat_fn):
    if not is_multi_task:
        return output
    if concat_fn is None:
        raise ValueError("concat_fn must be provided for multi-task models")
    return concat_fn(output)


def infer_with_tta_batched_3d(model, inputs: torch.Tensor,
                               *, is_multi_task: bool = False,
                               concat_multi_task_outputs=None) -> torch.Tensor:
    """
    Batched TTA for 3D mirroring - processes all 8 flips in a single forward pass.

    This reduces kernel launch overhead from 8 separate model calls to 1.
    Requires 8x GPU memory compared to sequential, but provides ~40-60% speedup.

    Args:
        model: The neural network model
        inputs: Input tensor of shape (B, C, D, H, W)
        is_multi_task: If True, model returns a dict
        concat_multi_task_outputs: Function to concatenate multi-task outputs

    Returns:
        Averaged output tensor of shape (B, C_out, D, H, W)
    """
    B, C, D, H, W = inputs.shape

    # Create all 8 flip combinations efficiently
    # Stack along a new dimension: (8, B, C, D, H, W)
    augmented = torch.stack([
        inputs,                                      # m0: original
        torch.flip(inputs, dims=[-1]),               # m1: flip X
        torch.flip(inputs, dims=[-2]),               # m2: flip Y
        torch.flip(inputs, dims=[-3]),               # m3: flip Z
        torch.flip(inputs, dims=[-1, -2]),           # m4: flip XY
        torch.flip(inputs, dims=[-1, -3]),           # m5: flip XZ
        torch.flip(inputs, dims=[-2, -3]),           # m6: flip YZ
        torch.flip(inputs, dims=[-1, -2, -3]),       # m7: flip XYZ
    ], dim=0)

    # Reshape for batch processing: (8*B, C, D, H, W)
    augmented = augmented.view(8 * B, C, D, H, W)

    # Single forward pass for all augmentations
    outputs = model(augmented)

    # Handle multi-task outputs
    if is_multi_task and concat_multi_task_outputs:
        outputs = concat_multi_task_outputs(outputs)

    # Get output channel count and reshape: (8, B, C_out, D, H, W)
    C_out = outputs.shape[1]
    outputs = outputs.view(8, B, C_out, D, H, W)

    # Reverse the flips on outputs (in-place for memory efficiency)
    outputs[1] = torch.flip(outputs[1], dims=[-1])
    outputs[2] = torch.flip(outputs[2], dims=[-2])
    outputs[3] = torch.flip(outputs[3], dims=[-3])
    outputs[4] = torch.flip(outputs[4], dims=[-1, -2])
    outputs[5] = torch.flip(outputs[5], dims=[-1, -3])
    outputs[6] = torch.flip(outputs[6], dims=[-2, -3])
    outputs[7] = torch.flip(outputs[7], dims=[-1, -2, -3])

    # Average across augmentations
    return outputs.mean(dim=0)


def infer_with_tta_batched_2d(model, inputs: torch.Tensor,
                               *, is_multi_task: bool = False,
                               concat_multi_task_outputs=None) -> torch.Tensor:
    """
    Batched TTA for 2D mirroring - processes all 4 flips in a single forward pass.
    """
    B, C, H, W = inputs.shape

    # Create all 4 flip combinations: (4, B, C, H, W)
    augmented = torch.stack([
        inputs,                            # m0: original
        torch.flip(inputs, dims=[-1]),     # m1: flip W
        torch.flip(inputs, dims=[-2]),     # m2: flip H
        torch.flip(inputs, dims=[-2, -1]), # m3: flip HW
    ], dim=0)

    # Reshape: (4*B, C, H, W)
    augmented = augmented.view(4 * B, C, H, W)

    # Single forward pass
    outputs = model(augmented)

    if is_multi_task and concat_multi_task_outputs:
        outputs = concat_multi_task_outputs(outputs)

    C_out = outputs.shape[1]
    outputs = outputs.view(4, B, C_out, H, W)

    # Reverse flips
    outputs[1] = torch.flip(outputs[1], dims=[-1])
    outputs[2] = torch.flip(outputs[2], dims=[-2])
    outputs[3] = torch.flip(outputs[3], dims=[-2, -1])

    return outputs.mean(dim=0)


def infer_with_tta_sequential(model,
                               inputs: torch.Tensor,
                               tta_type: str = 'mirroring',
                               *,
                               is_multi_task: bool = False,
                               concat_multi_task_outputs=None) -> torch.Tensor:
    """
    Sequential TTA - original implementation with 8 separate model calls.
    Used as fallback when batched TTA runs out of memory.
    """
    # Determine number of spatial dims
    ndim = inputs.ndim
    spatial_dims = ndim - 2

    if tta_type == 'mirroring':
        if spatial_dims == 3:
            m0 = model(inputs)
            m1 = model(torch.flip(inputs, dims=[-1]))
            m2 = model(torch.flip(inputs, dims=[-2]))
            m3 = model(torch.flip(inputs, dims=[-3]))
            m4 = model(torch.flip(inputs, dims=[-1, -2]))
            m5 = model(torch.flip(inputs, dims=[-1, -3]))
            m6 = model(torch.flip(inputs, dims=[-2, -3]))
            m7 = model(torch.flip(inputs, dims=[-1, -2, -3]))

            m0 = _concat_if_multi_task(m0, is_multi_task, concat_multi_task_outputs)
            m1 = _concat_if_multi_task(m1, is_multi_task, concat_multi_task_outputs)
            m2 = _concat_if_multi_task(m2, is_multi_task, concat_multi_task_outputs)
            m3 = _concat_if_multi_task(m3, is_multi_task, concat_multi_task_outputs)
            m4 = _concat_if_multi_task(m4, is_multi_task, concat_multi_task_outputs)
            m5 = _concat_if_multi_task(m5, is_multi_task, concat_multi_task_outputs)
            m6 = _concat_if_multi_task(m6, is_multi_task, concat_multi_task_outputs)
            m7 = _concat_if_multi_task(m7, is_multi_task, concat_multi_task_outputs)

            outputs = [
                m0,
                torch.flip(m1, dims=[-1]),
                torch.flip(m2, dims=[-2]),
                torch.flip(m3, dims=[-3]),
                torch.flip(m4, dims=[-1, -2]),
                torch.flip(m5, dims=[-1, -3]),
                torch.flip(m6, dims=[-2, -3]),
                torch.flip(m7, dims=[-1, -2, -3])
            ]
            return torch.mean(torch.stack(outputs, dim=0), dim=0)
        else:  # 2D flips over H and W
            m0 = model(inputs)
            m1 = model(torch.flip(inputs, dims=[-1]))
            m2 = model(torch.flip(inputs, dims=[-2]))
            m3 = model(torch.flip(inputs, dims=[-2, -1]))

            m0 = _concat_if_multi_task(m0, is_multi_task, concat_multi_task_outputs)
            m1 = _concat_if_multi_task(m1, is_multi_task, concat_multi_task_outputs)
            m2 = _concat_if_multi_task(m2, is_multi_task, concat_multi_task_outputs)
            m3 = _concat_if_multi_task(m3, is_multi_task, concat_multi_task_outputs)

            outputs = [
                m0,
                torch.flip(m1, dims=[-1]),
                torch.flip(m2, dims=[-2]),
                torch.flip(m3, dims=[-2, -1])
            ]
            return torch.mean(torch.stack(outputs, dim=0), dim=0)

    else:  # rotation
        if spatial_dims == 3:
            r0 = model(inputs)
            x_up = torch.transpose(inputs, -3, -1)
            r_x_up = model(x_up)
            z_up = torch.transpose(inputs, -3, -2)
            r_z_up = model(z_up)

            r0 = _concat_if_multi_task(r0, is_multi_task, concat_multi_task_outputs)
            r_x_up = _concat_if_multi_task(r_x_up, is_multi_task, concat_multi_task_outputs)
            r_z_up = _concat_if_multi_task(r_z_up, is_multi_task, concat_multi_task_outputs)

            outputs = [
                r0,
                torch.transpose(r_x_up, -3, -1),
                torch.transpose(r_z_up, -3, -2)
            ]
            return torch.mean(torch.stack(outputs, dim=0), dim=0)
        else:  # 2D
            r0 = model(inputs)
            hw = torch.transpose(inputs, -2, -1)
            r_hw = model(hw)

            r0 = _concat_if_multi_task(r0, is_multi_task, concat_multi_task_outputs)
            r_hw = _concat_if_multi_task(r_hw, is_multi_task, concat_multi_task_outputs)

            outputs = [
                r0,
                torch.transpose(r_hw, -2, -1)
            ]
            return torch.mean(torch.stack(outputs, dim=0), dim=0)


# Global flag to track if batched TTA has failed (to avoid repeated OOM errors)
_batched_tta_disabled = False


def infer_with_tta(model,
                   inputs: torch.Tensor,
                   tta_type: str = 'mirroring',
                   *,
                   is_multi_task: bool = False,
                   concat_multi_task_outputs=None,
                   use_batched: bool = True) -> torch.Tensor:
    """
    Apply TTA for 3D or 2D models with automatic batched/sequential selection.

    - For 3D, inputs: (B, C, D, H, W) → returns (B, C, D, H, W)
    - For 2D, inputs: (B, C, H, W) → returns (B, C, H, W)

    - tta_type: 'mirroring' uses 8 flip combinations (batched by default)
                'rotation' uses axis transpositions (sequential only)
    - use_batched: If True, try batched TTA first for mirroring (40-60% faster).
                   Falls back to sequential on OOM.
    """
    global _batched_tta_disabled

    if tta_type not in ('mirroring', 'rotation'):
        raise ValueError(f"Unsupported tta_type: {tta_type}")

    ndim = inputs.ndim
    if ndim < 4:
        raise ValueError(f"infer_with_tta expects at least 4D input (B,C,...) got {ndim}D")
    spatial_dims = ndim - 2
    if spatial_dims not in (2, 3):
        raise ValueError(f"infer_with_tta expects 2D or 3D spatial dims, got {spatial_dims}")

    # Use batched TTA for mirroring if enabled and not previously failed
    if tta_type == 'mirroring' and use_batched and not _batched_tta_disabled:
        try:
            if spatial_dims == 3:
                return infer_with_tta_batched_3d(
                    model, inputs,
                    is_multi_task=is_multi_task,
                    concat_multi_task_outputs=concat_multi_task_outputs
                )
            else:
                return infer_with_tta_batched_2d(
                    model, inputs,
                    is_multi_task=is_multi_task,
                    concat_multi_task_outputs=concat_multi_task_outputs
                )
        except torch.cuda.OutOfMemoryError:
            # Batched TTA failed due to OOM, disable for rest of session
            _batched_tta_disabled = True
            torch.cuda.empty_cache()
            # Fall through to sequential

    # Sequential TTA (fallback or for rotation)
    return infer_with_tta_sequential(
        model, inputs, tta_type,
        is_multi_task=is_multi_task,
        concat_multi_task_outputs=concat_multi_task_outputs
    )
