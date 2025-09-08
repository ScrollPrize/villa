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
                   concat_multi_task_outputs=None) -> torch.Tensor:
    """
    Apply TTA for 3D or 2D models.

    - For 3D, inputs: (B, C, D, H, W) → returns (B, C, D, H, W)
    - For 2D, inputs: (B, C, H, W) → returns (B, C, H, W)

    - tta_type: 'mirroring' uses 8 flip combinations
                'rotation' uses axis transpositions to rotate volumes
    - is_multi_task: if True, model returns a dict; provide concat_multi_task_outputs
                     to concatenate dict outputs into a tensor (B, C, ...)
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
            m1 = model(torch.flip(inputs, dims=[-1]))  # W
            m2 = model(torch.flip(inputs, dims=[-2]))  # H
            m3 = model(torch.flip(inputs, dims=[-2, -1]))  # HW

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
        else:  # 2D: use transpose(H,W) as rotation
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
