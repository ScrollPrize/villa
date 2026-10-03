import os

import torch


def pinned_to_device(cpu_tensor, device):
    # Upload through a pinned staging tensor: a pageable H2D copy
    # synchronises the CPU on all queued GPU work, while pin_memory() (a
    # host memcpy into torch's recycling pinned allocator) plus a
    # non_blocking copy overlaps with whatever the GPU is still running.
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != 'cuda' or cpu_tensor.device.type != 'cpu':
        return cpu_tensor.to(device=device)
    return cpu_tensor.pin_memory().to(device=device, non_blocking=True)


_scalar_tensor_cache = {}


def cached_scalar_tensor(value, device, dtype=torch.float32):
    # Device-resident scalar constant without a per-call host-to-device copy
    # (which would stall the CPU behind all queued GPU work).
    key = (float(value), str(device), dtype)
    cached = _scalar_tensor_cache.get(key)
    if cached is None:
        cached = torch.as_tensor(value, device=device, dtype=dtype)
        _scalar_tensor_cache[key] = cached
    return cached


def maybe_compile(fn):
    # torch.compile the hot pure-tensor helpers when FIT_SPIRAL_COMPILE=1.
    # Inductor fuses the elementwise chains (fwd and generated bwd) but uses
    # FMA contraction, so results shift at the last-ulp level; off by default.
    if os.environ.get('FIT_SPIRAL_COMPILE', '0') == '1':
        # The training loop backwards each loss family with retain_graph=True;
        # donated buffers assume single-use backward graphs and hard-error.
        import torch._functorch.config
        torch._functorch.config.donated_buffer = False
        return torch.compile(fn, dynamic=True)
    return fn


@maybe_compile
def expm_2x2(L):
    # Closed-form matrix exponential for (..., 2, 2) matrices:
    # exp(L) = e^m (cosh(s) I + sinh(s)/s (L - m I)), where
    # m = tr(L)/2 and s^2 = ((a - d)/2)^2 + bc.
    a, b = L[..., 0, 0], L[..., 0, 1]
    c, d = L[..., 1, 0], L[..., 1, 1]
    m = 0.5 * (a + d)
    s2 = (0.5 * (a - d)) ** 2 + b * c
    small = s2.abs() < 1e-8
    s = torch.where(small, torch.ones_like(s2), s2).abs().sqrt()
    pos = s2 >= 0
    cosh_term = torch.where(small, 1.0 + s2 / 2.0, torch.where(pos, torch.cosh(s), torch.cos(s)))
    sinc_term = torch.where(small, 1.0 + s2 / 6.0, torch.where(pos, torch.sinh(s), torch.sin(s)) / s)
    em = torch.exp(m)
    f_diag = em * cosh_term
    f_off = em * sinc_term
    # One coalesced stack instead of four strided advanced-indexing writes;
    # the element values are identical.
    return torch.stack([
        f_diag + f_off * (a - m),
        f_off * b,
        f_off * c,
        f_diag + f_off * (d - m),
    ], dim=-1).view(*L.shape)


@maybe_compile
def bilinear_atlas_lookup(zyxs_flat, offsets, widths, patch_indices, ijs, heights=None):
    """Bilinearly sample packed patch grids at fractional ``(i, j)`` coordinates.

    When ``heights`` is given, the four bilinear corners are clamped inside the
    addressed patch. Callers relying on "floor(ij) lies on a valid quad" should
    pass it: jitters drawn as float64 in [0, 1) and cast to float32 can round
    to exactly 1.0, which pushes a sample one cell past the last valid quad
    row/column - the corner gather then silently reads the next patch's first
    row (or trips a device-side assert on the atlas's last patch).
    """
    base = offsets[patch_indices]
    width = widths[patch_indices]
    ijs = ijs.to(torch.float32)
    i0 = ijs[..., 0].floor().to(torch.int64)
    j0 = ijs[..., 1].floor().to(torch.int64)
    if heights is not None:
        height = heights[patch_indices]
        i0 = torch.minimum(i0.clamp(min=0), height - 2)
        j0 = torch.minimum(j0.clamp(min=0), width - 2)
        di = (ijs[..., 0] - i0.to(torch.float32)).unsqueeze(-1).clamp(0., 1.)
        dj = (ijs[..., 1] - j0.to(torch.float32)).unsqueeze(-1).clamp(0., 1.)
    else:
        di = (ijs[..., 0] - i0.to(torch.float32)).unsqueeze(-1)
        dj = (ijs[..., 1] - j0.to(torch.float32)).unsqueeze(-1)

    flat_tl = base + i0 * width + j0
    tl = zyxs_flat[flat_tl]
    tr = zyxs_flat[flat_tl + 1]
    bl = zyxs_flat[flat_tl + width]
    br = zyxs_flat[flat_tl + width + 1]
    top = tl + (tr - tl) * dj
    bottom = bl + (br - bl) * dj
    return top + (bottom - top) * di


def interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='const') -> torch.Tensor:
    # See https://github.com/pytorch/pytorch/issues/50334
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    # indices = torch.sum(x[None, ...] >= xp.view(-1, *[1] * x.ndim), dim=0) - 1
    indices = torch.searchsorted(xp.squeeze(-1), x) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    return m[indices] * x[..., None] + b[indices]


def grid_sample_border(input, grid):
    """``F.grid_sample(input, grid, mode='bilinear', padding_mode='border',
    align_corners=True)`` for 4-D and 5-D inputs.

    MPS implements grid_sample's forward but not its backward, so there, when
    autograd needs the backward, the same interpolation is built from gathers
    (whose backward MPS has). Every other case calls grid_sample itself.
    """
    if (input.device.type != 'mps' or not torch.is_grad_enabled()
            or not (input.requires_grad or grid.requires_grad)):
        return torch.nn.functional.grid_sample(
            input, grid, mode='bilinear', padding_mode='border',
            align_corners=True)
    return gather_grid_sample_border(input, grid)


def gather_grid_sample_border(input, grid):
    """grid_sample_border's interpolation from gathers, on any device."""
    n, channels = input.shape[:2]
    spatial = input.shape[2:]
    # grid's last axis is (x, y[, z]); spatial is ([z,] y, x).
    coords = grid.reshape(n, -1, len(spatial)).flip(-1)
    corners = []
    stride = 1
    for axis in reversed(range(len(spatial))):
        size = spatial[axis]
        position = ((coords[..., axis] + 1) * 0.5 * (size - 1)).clamp(0, size - 1)
        low = torch.nan_to_num(position, nan=0.0).floor()
        high_weight = position - low
        low = low.to(torch.int64)
        high = (low + 1).clamp(max=size - 1)
        corners.append(((low * stride, 1 - high_weight), (high * stride, high_weight)))
        stride *= size
    flat = input.reshape(n, channels, -1)
    out = None
    for choice in range(1 << len(spatial)):
        index, weight = None, None
        for bit, pair in enumerate(corners):
            offset, factor = pair[(choice >> bit) & 1]
            index = offset if index is None else index + offset
            weight = factor if weight is None else weight * factor
        values = flat.gather(2, index[:, None, :].expand(n, channels, -1))
        term = values * weight[:, None, :]
        out = term if out is None else out + term
    return out.reshape(n, channels, *grid.shape[1:-1])
