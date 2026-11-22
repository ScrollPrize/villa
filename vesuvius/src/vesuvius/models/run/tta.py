import math

import torch


def _concat_if_multi_task(output, is_multi_task: bool, concat_fn):
    if not is_multi_task:
        return output
    if concat_fn is None:
        raise ValueError("concat_fn must be provided for multi-task models")
    return concat_fn(output)


# Heads whose channels are the components of a direction (x, y, z order, as the auxiliary
# tasks write them) or of the structure tensor. Mirroring or transposing the input does not
# only move such values, it also negates or swaps their components, so the TTA has to undo
# both before averaging; otherwise the variants cancel (normals average to zero).
# Spatial axes are named by their tensor dim: z = -3, y = -2, x = -1.
_AXES = {'z': -3, 'y': -2, 'x': -1}
_VECTOR_TASKS = ('surface_normals', 'nearest_component', 'inplane_direction')
# Vector heads that are axes when their configuration names no direction loss: an in-plane
# direction is an eigenvector, so its sign is arbitrary.
_AXIS_TASKS_BY_DEFAULT = ('inplane_direction',)
# Direction losses that do not depend on the sign of the prediction.
_SIGN_INVARIANT_LOSSES = ('SignInvariantCosineLoss',)
_TENSOR_ORDER = {3: ('zz', 'zy', 'zx', 'yy', 'yx', 'xx'), 2: ('yy', 'yx', 'xx')}
# An axis is written in the half-space 4x + 2y + z >= 0 of the input frame (x, y, z weights), so
# the output has one sign convention. No coordinate axis or axis diagonal lies on its boundary.
_HALF_SPACE = (4.0, 2.0, 1.0)
# The orientation tensors of an axis head are reduced this many voxels (whole rows) at a time,
# so their float32 temporaries have a fixed size instead of growing with the patch or chunk.
_SLAB_VOXELS = 1 << 18
# Squarings in `principal_axis`: the axis error falls like (l2 / l1) ** (2 ** _SQUARINGS).
_SQUARINGS = 6


def _direction_losses(target_config):
    """Names of the losses configured for a vector head's direction channels.

    A `ChannelSelectLoss` counts with its base loss, unless it selects only the last channel
    (the distance of `nearest_component`)."""
    losses = target_config.get('losses')
    if not losses and target_config.get('loss_fn'):
        losses = [{'name': target_config['loss_fn']}]
    names = []
    for loss in losses or ():
        name = loss.get('name')
        if name == 'ChannelSelectLoss':
            kwargs = loss.get('kwargs') or {}
            if kwargs.get('channels', loss.get('channels', 'all_but_last')) == 'last':
                continue
            name = (kwargs.get('base') or loss.get('base') or {}).get('name')
        names.append(name)
    return names


def is_axis_head(target_config, out_channels=None, spatial_dims=3) -> bool:
    """Whether a head's leading channels are an axis: a direction whose sign is arbitrary.

    Read from the losses the head was trained with. When every loss on the direction channels
    is `SignInvariantCosineLoss` (as configured for `inplane_direction` and for the direction
    of `nearest_component`), the network may report either sign, so TTA variants and
    overlapping patches can disagree in sign while they agree on the axis. Any other loss on
    those channels (`CosineSimilarityLoss`, `NormalSmoothnessLoss`, a regression loss) can
    supervise the orientation, so the head is signed and is never flipped, also when it is
    combined with a sign-invariant loss. Without a direction loss in the configuration,
    in-plane directions are axes and the other vector heads signed.
    """
    cfg = target_config or {}
    task = str(cfg.get('task_type', '')).lower()
    count = cfg.get('out_channels') if out_channels is None else out_channels
    if task not in _VECTOR_TASKS or (count is not None and int(count) < spatial_dims):
        return False
    names = _direction_losses(cfg)
    if not names:
        return task in _AXIS_TASKS_BY_DEFAULT
    return all(name in _SIGN_INVARIANT_LOSSES for name in names)


def axis_head_channels(target_info, spatial_dims=3):
    """The (x, y[, z]) channels of each head marked `sign_invariant` in `target_info`."""
    heads = sorted((target_info or {}).values(), key=lambda info: info['start_channel'])
    return [tuple(range(int(info['start_channel']), int(info['start_channel']) + spatial_dims))
            for info in heads if info.get('sign_invariant', False)]


def tta_channel_groups(targets, target_info, spatial_dims: int):
    """Geometric channel groups of a multi-task output, for `infer_with_tta(channel_groups=...)`.

    `targets` is the checkpoint's target configuration (with each auxiliary head's
    `task_type` and losses), `target_info` the channel range of each target in the concatenated
    output. Each group maps a tuple of spatial dims to an output channel: (axis,) for a vector
    component, (axis_a, axis_b) for a structure tensor component. Vector heads are 'axial'
    when `is_axis_head`, otherwise 'vector'.
    """
    groups = []
    for name, info in (target_info or {}).items():
        cfg = (targets or {}).get(name) or {}
        task = str(cfg.get('task_type', '')).lower()
        start, count = int(info['start_channel']), int(info['out_channels'])
        if task in _VECTOR_TASKS and count >= spatial_dims:
            channels = {(_AXES[a],): start + i for i, a in enumerate('xyz'[:spatial_dims])}
            kind = 'axial' if is_axis_head(cfg, count, spatial_dims) else 'vector'
            groups.append({'kind': kind, 'channels': channels})
        elif task == 'structure_tensor' and count == len(_TENSOR_ORDER[spatial_dims]):
            channels = {(_AXES[c[0]], _AXES[c[1]]): start + i
                        for i, c in enumerate(_TENSOR_ORDER[spatial_dims])}
            groups.append({'kind': 'tensor', 'channels': channels})
    return groups


def _undo_component_change(output: torch.Tensor, channel_groups, flipped=(), swapped=None) -> torch.Tensor:
    """Bring the components of an un-flipped / un-transposed variant back to the input frame.

    Works in place on `output` (a TTA variant the caller owns), so no full-size copy is made.
    `flipped`: the spatial dims the input was mirrored over (a component along one of
    them changes sign). `swapped`: the pair of spatial dims the input was transposed over
    (components along them trade places).
    """
    if not channel_groups:
        return output
    flipped = set(flipped)
    sigma = {}
    if swapped is not None:
        a, b = swapped
        sigma = {a: b, b: a}
    targets, sources = [], []
    for group in channel_groups:
        channels = group['channels']
        for key, ch in channels.items():
            if sum(axis in flipped for axis in key) % 2:
                output[:, ch].neg_()
            if sigma:
                source = tuple(sigma.get(axis, axis) for axis in key)
                source = channels[source] if source in channels else channels[source[::-1]]
                if source != ch:
                    targets.append(ch)
                    sources.append(source)
    if targets:
        # copies only the swapped channels
        output[:, targets] = output[:, sources]
    return output


def _pairs(d):
    return [(i, j) for i in range(d) for j in range(i, d)]


def _row_slabs(rows, row_voxels):
    """Slices of whole rows (the first spatial dim) holding about `_SLAB_VOXELS` voxels each."""
    step = max(1, _SLAB_VOXELS // max(1, int(row_voxels)))
    return [slice(s, min(s + step, rows)) for s in range(0, rows, step)]


def principal_axis(m, d):
    """The principal axis of per-voxel orientation tensors.

    `m` maps (i, j), i <= j < d, to float32 tensors: the components of the (weighted) mean of
    v v^T over the predictions v of each voxel. Returns the d components of the eigenvector of
    its largest eigenvalue, scaled to the square root of that eigenvalue and put in the
    half-space `_HALF_SPACE`. Predictions that agree up to sign give their common vector (with
    its length); in general the length is the root mean square of their projections on the
    axis. The result does not depend on the predictions' signs or order.
    """
    def at(p, i, j):
        return p[(i, j) if i <= j else (j, i)]

    def normalized(p):
        trace = sum(p[(i, i)] for i in range(d))
        inv = torch.where(trace > 0, trace.reciprocal(), torch.zeros_like(trace))
        return {key: value * inv for key, value in p.items()}

    # Repeated squaring: (M / tr M) ** (2 ** k), renormalised, tends to the projector on the axis.
    p = normalized(m)
    for _ in range(_SQUARINGS):
        p = normalized({(i, j): sum(at(p, i, k) * at(p, k, j) for k in range(d)) for i, j in _pairs(d)})
    # Its column with the largest diagonal entry is a nonzero multiple of the axis (unless M = 0).
    column, diagonal = [at(p, i, 0) for i in range(d)], p[(0, 0)]
    for j in range(1, d):
        take = p[(j, j)] > diagonal
        diagonal = torch.where(take, p[(j, j)], diagonal)
        column = [torch.where(take, at(p, i, j), c) for i, c in enumerate(column)]
    norm = torch.sqrt(sum(c * c for c in column))
    inv = torch.where(norm > 0, norm.reciprocal(), torch.zeros_like(norm))
    axis = [c * inv for c in column]
    eigenvalue = sum(at(m, i, j) * axis[i] * axis[j] for i in range(d) for j in range(d))
    length = torch.sqrt(eigenvalue.clamp_min(0))
    side = sum(w * c for w, c in zip(_HALF_SPACE, axis))
    length = torch.where(side < 0, -length, length)
    return [c * length for c in axis]


def _average_axis(mean: torch.Tensor, variants, channels) -> None:
    """Write the variants' mean axis into `channels` (x, y[, z]) of `mean` (B, C, *spatial).

    The variants are averaged as orientation tensors v v^T, in float32 (half-precision
    products of small components underflow), and reduced to their principal axis. Signs
    the variants or the network chose do not matter.
    """
    d, count = len(channels), len(variants)
    row_voxels = mean.shape[0] * math.prod(mean.shape[3:])
    for rows in _row_slabs(mean.shape[2], row_voxels):
        m = {}
        for variant in variants:
            v = [variant[:, c, rows].float() for c in channels]
            for i, j in _pairs(d):
                term = v[i] * v[j]
                m[(i, j)] = term if (i, j) not in m else m[(i, j)].add_(term)
        axis = principal_axis({key: value.div_(count) for key, value in m.items()}, d)
        for c, component in zip(channels, axis):
            mean[:, c, rows] = component


def _average_variants(variants, channel_groups) -> torch.Tensor:
    """Mean of the TTA variants (a list, or a tensor stacked on dim 0).

    Axis heads (directions without a sign, see `is_axis_head`) get the principal axis of the
    variants' mean orientation tensor instead of the mean of their components.
    """
    if torch.is_tensor(variants):
        mean = variants.mean(dim=0)
    else:
        mean = torch.mean(torch.stack(list(variants), dim=0), dim=0)
    for group in channel_groups or ():
        if group['kind'] == 'axial':
            channels = group['channels']
            _average_axis(mean, variants, [channels[(a,)] for a in (-1, -2, -3) if (a,) in channels])
    return mean


def orientation_layouts(axis_channels, num_channels):
    """Where a blend accumulates the orientation tensors v v^T of its axis heads.

    `axis_channels` holds the (x, y, z) channels of each axis head (see `axis_head_channels`).
    The diagonal xx, yy, zz goes to the head's own channels, the off-diagonal xy, xz, yz to
    three extra channels after the `num_channels` output channels. Returns one {(i, j): channel}
    per head; the accumulator needs `num_channels + 3 * len(axis_channels)` channels (3D).
    """
    layouts, extra = [], int(num_channels)
    for channels in axis_channels:
        layout = {}
        for i, j in _pairs(len(channels)):
            if i == j:
                layout[(i, i)] = int(channels[i])
            else:
                layout[(i, j)], extra = extra, extra + 1
        layouts.append(layout)
    return layouts


def accumulator_channels(layouts, num_channels):
    """Channel count of a blend accumulator with these `orientation_layouts`."""
    return max([int(num_channels)] + [c + 1 for layout in layouts for c in layout.values()])


def reduce_orientation_tensors(sums: torch.Tensor, layouts) -> None:
    """In blended means `sums` (C_acc, Z, Y, X), replace the orientation tensor of each axis
    head (`orientation_layouts`) by its principal axis, in the head's x, y, z channels."""
    row_voxels = math.prod(sums.shape[2:])
    for layout in layouts:
        d = sum(1 for i, j in layout if i == j)
        channels = [layout[(i, i)] for i in range(d)]
        for rows in _row_slabs(sums.shape[1], row_voxels):
            axis = principal_axis({key: sums[c, rows] for key, c in layout.items()}, d)
            for c, component in zip(channels, axis):
                sums[c, rows] = component


def infer_with_tta_batched_3d(model, inputs: torch.Tensor,
                               *, is_multi_task: bool = False,
                               concat_multi_task_outputs=None,
                               channel_groups=None) -> torch.Tensor:
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

    if channel_groups:
        flips = [(), (-1,), (-2,), (-3,), (-1, -2), (-1, -3), (-2, -3), (-1, -2, -3)]
        for k, f in enumerate(flips):
            _undo_component_change(outputs[k], channel_groups, flipped=f)
        return _average_variants(outputs, channel_groups)

    # Average across augmentations
    return outputs.mean(dim=0)


def infer_with_tta_batched_2d(model, inputs: torch.Tensor,
                               *, is_multi_task: bool = False,
                               concat_multi_task_outputs=None,
                               channel_groups=None) -> torch.Tensor:
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

    if channel_groups:
        flips = [(), (-1,), (-2,), (-2, -1)]
        for k, f in enumerate(flips):
            _undo_component_change(outputs[k], channel_groups, flipped=f)
        return _average_variants(outputs, channel_groups)

    return outputs.mean(dim=0)


def infer_with_tta_sequential(model,
                               inputs: torch.Tensor,
                               tta_type: str = 'mirroring',
                               *,
                               is_multi_task: bool = False,
                               concat_multi_task_outputs=None,
                               channel_groups=None) -> torch.Tensor:
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

            flips = [(), (-1,), (-2,), (-3,), (-1, -2), (-1, -3), (-2, -3), (-1, -2, -3)]
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
            for o, f in zip(outputs, flips):
                _undo_component_change(o, channel_groups, flipped=f)
            return _average_variants(outputs, channel_groups)
        else:  # 2D flips over H and W
            m0 = model(inputs)
            m1 = model(torch.flip(inputs, dims=[-1]))
            m2 = model(torch.flip(inputs, dims=[-2]))
            m3 = model(torch.flip(inputs, dims=[-2, -1]))

            m0 = _concat_if_multi_task(m0, is_multi_task, concat_multi_task_outputs)
            m1 = _concat_if_multi_task(m1, is_multi_task, concat_multi_task_outputs)
            m2 = _concat_if_multi_task(m2, is_multi_task, concat_multi_task_outputs)
            m3 = _concat_if_multi_task(m3, is_multi_task, concat_multi_task_outputs)

            flips = [(), (-1,), (-2,), (-2, -1)]
            outputs = [
                m0,
                torch.flip(m1, dims=[-1]),
                torch.flip(m2, dims=[-2]),
                torch.flip(m3, dims=[-2, -1])
            ]
            for o, f in zip(outputs, flips):
                _undo_component_change(o, channel_groups, flipped=f)
            return _average_variants(outputs, channel_groups)

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
                _undo_component_change(torch.transpose(r_x_up, -3, -1), channel_groups, swapped=(-3, -1)),
                _undo_component_change(torch.transpose(r_z_up, -3, -2), channel_groups, swapped=(-3, -2)),
            ]
            return _average_variants(outputs, channel_groups)
        else:  # 2D
            r0 = model(inputs)
            hw = torch.transpose(inputs, -2, -1)
            r_hw = model(hw)

            r0 = _concat_if_multi_task(r0, is_multi_task, concat_multi_task_outputs)
            r_hw = _concat_if_multi_task(r_hw, is_multi_task, concat_multi_task_outputs)

            outputs = [
                r0,
                _undo_component_change(torch.transpose(r_hw, -2, -1), channel_groups, swapped=(-2, -1)),
            ]
            return _average_variants(outputs, channel_groups)


# Global flag to track if batched TTA has failed (to avoid repeated OOM errors)
_batched_tta_disabled = False


def infer_with_tta(model,
                   inputs: torch.Tensor,
                   tta_type: str = 'mirroring',
                   *,
                   is_multi_task: bool = False,
                   concat_multi_task_outputs=None,
                   use_batched: bool = True,
                   channel_groups=None) -> torch.Tensor:
    """
    Apply TTA for 3D or 2D models with automatic batched/sequential selection.

    - For 3D, inputs: (B, C, D, H, W) → returns (B, C, D, H, W)
    - For 2D, inputs: (B, C, H, W) → returns (B, C, H, W)

    - tta_type: 'mirroring' uses 8 flip combinations (batched by default)
                'rotation' uses axis transpositions (sequential only)
    - use_batched: If True, try batched TTA first for mirroring (40-60% faster).
                   Falls back to sequential on OOM.
    - channel_groups: vector and tensor channels of the output (see `tta_channel_groups`),
                   whose components are negated / swapped back with each variant.
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

    # Use batched TTA for mirroring if enabled and not previously failed.
    # It needs ~8x the activation memory and its out-of-memory fallback only
    # exists on CUDA, so other devices (CPU, MPS) run the flips one at a time.
    if (tta_type == 'mirroring' and use_batched and not _batched_tta_disabled
            and inputs.device.type == 'cuda'):
        try:
            if spatial_dims == 3:
                return infer_with_tta_batched_3d(
                    model, inputs,
                    is_multi_task=is_multi_task,
                    concat_multi_task_outputs=concat_multi_task_outputs,
                    channel_groups=channel_groups,
                )
            else:
                return infer_with_tta_batched_2d(
                    model, inputs,
                    is_multi_task=is_multi_task,
                    concat_multi_task_outputs=concat_multi_task_outputs,
                    channel_groups=channel_groups,
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
        concat_multi_task_outputs=concat_multi_task_outputs,
        channel_groups=channel_groups,
    )
