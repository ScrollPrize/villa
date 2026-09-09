"""Gaussian smoothing of flow-lattice gradients.

Smoothing the velocity gradient before the optimizer step is gradient
descent in a Sobolev metric, the standard preconditioner of diffeomorphic
registration: the loss and its minima are unchanged, but every update to the
flow is smooth at the kernel's scale instead of moving each lattice cell
independently on its own noisy gradient. Both functions smooth in place,
treat the lattices' leading slab axis (the flow stages) and the vector
components independently, and renormalise the kernel at lattice borders so a
constant gradient stays constant.

Widths are in lattice cells of the lattice being smoothed; the caller
converts from scroll voxels (see SpiralAndTransform.smooth_flow_grad_).

On a cylindrical lattice the smoothing is anisotropic: one width along the
sheet (along z and around each ring, which in the flow frame is along the
model spiral's windings) and a separate, normally much smaller, width across
rings (across windings). See smooth_cylindrical_.
"""

import math

import torch
import torch.nn.functional as F

import flow_triton

# Truncation radius of the Gaussian, in standard deviations.
KERNEL_TRUNCATE = 3.0
# Widest kernel the fused kernels unroll (taps = 2 * radius + 1); wider
# kernels take the conv fallback.
MAX_FUSED_RADIUS = 48

if flow_triton._HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _blur_axis_kernel(x_ptr, out_ptr, w_ptr, norm_ptr, L, INNER,
                          RADIUS: tl.constexpr,
                          BLOCK_L: tl.constexpr, BLOCK_I: tl.constexpr):
        # One Gaussian pass along an axis of a contiguous [OUTER, L, INNER]
        # tensor, zero-padded and divided by the per-position border norm.
        # Tiles are [BLOCK_L positions, BLOCK_I inner elements]; along the
        # last axis INNER is 1 and the tile runs along L instead.
        outer = tl.program_id(0).to(tl.int64)
        l = tl.program_id(1) * BLOCK_L + tl.arange(0, BLOCK_L)
        i = tl.program_id(2) * BLOCK_I + tl.arange(0, BLOCK_I)
        ml = l < L
        mi = i < INNER
        base = outer * L * INNER
        acc = tl.zeros([BLOCK_L, BLOCK_I], dtype=tl.float32)
        for k in tl.static_range(2 * RADIUS + 1):
            src = l + (k - RADIUS)
            m = (src >= 0) & (src < L)
            w = tl.load(w_ptr + k)
            vals = tl.load(
                x_ptr + base + src.to(tl.int64)[:, None] * INNER + i[None, :],
                mask=m[:, None] & mi[None, :], other=0.0)
            acc += w * vals
        norm = tl.load(norm_ptr + l, mask=ml, other=1.0)
        acc = acc / norm[:, None]
        tl.store(out_ptr + base + l.to(tl.int64)[:, None] * INNER + i[None, :],
                 acc, mask=ml[:, None] & mi[None, :])

    @triton.jit
    def _blur_rings_kernel(x_ptr, out_ptr, w_ptr, ring_ptr, num_phi_ptr,
                           offsets_ptr, ROWS, TOTAL,
                           RADIUS: tl.constexpr,
                           BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        # Circular Gaussian pass around every ring of a packed [ROWS, TOTAL]
        # plane at once: each packed cell looks up its ring, and its taps
        # wrap within that ring's cells.
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mr = r < ROWS
        mc = c < TOTAL
        ring = tl.load(ring_ptr + c, mask=mc, other=0)
        n = tl.load(num_phi_ptr + ring, mask=mc, other=1)
        offset = tl.load(offsets_ptr + ring, mask=mc, other=0)
        local = c - offset
        rows = r.to(tl.int64)[:, None] * TOTAL
        acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        for k in tl.static_range(2 * RADIUS + 1):
            src = (local + (k - RADIUS)) % n
            src = tl.where(src < 0, src + n, src)
            w = tl.load(w_ptr + k)
            vals = tl.load(x_ptr + rows + (offset + src)[None, :],
                           mask=mr[:, None] & mc[None, :], other=0.0)
            acc += w * vals
        tl.store(out_ptr + rows + c[None, :], acc, mask=mr[:, None] & mc[None, :])

    @triton.jit
    def _blur_radial_kernel(x_ptr, out_ptr, w_ptr, ring_ptr, num_phi_ptr,
                            offsets_ptr, ROWS, TOTAL, NUM_RINGS,
                            RADIUS: tl.constexpr,
                            BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        # Gaussian pass across the rings of a packed [ROWS, TOTAL] plane: each
        # cell's tap k reads ring (r + k) at the cell's own angle, linearly
        # interpolated between that ring's two nearest cells (rings hold
        # different cell counts). Taps falling on the pinned axis ring 0 or
        # beyond the outermost ring are dropped and the weights renormalised,
        # so a constant stays constant at the borders; ring 0 itself passes
        # through unchanged.
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mr = r < ROWS
        mc = c < TOTAL
        ring = tl.load(ring_ptr + c, mask=mc, other=0)
        n = tl.load(num_phi_ptr + ring, mask=mc, other=1)
        offset = tl.load(offsets_ptr + ring, mask=mc, other=0)
        local = c - offset  # cell index within its ring; angle = local / n turns
        rows = r.to(tl.int64)[:, None] * TOTAL
        acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        wsum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for k in tl.static_range(2 * RADIUS + 1):
            src_ring = ring + (k - RADIUS)
            valid = mc & (src_ring >= 1) & (src_ring < NUM_RINGS)
            src_ring_safe = tl.where(valid, src_ring, 0)
            n_src = tl.load(num_phi_ptr + src_ring_safe, mask=valid, other=1)
            off_src = tl.load(offsets_ptr + src_ring_safe, mask=valid, other=0)
            # Position local / n * n_src on the source ring, split exactly in
            # integers (the eager reference does the same), so aligned angles
            # land on one cell and the two paths agree to rounding.
            numerator = local * n_src
            lo = numerator // n
            frac = (numerator - lo * n).to(tl.float32) / n.to(tl.float32)
            hi = (lo + 1) % n_src
            w = tl.load(w_ptr + k)
            v_lo = tl.load(x_ptr + rows + (off_src + lo)[None, :],
                           mask=mr[:, None] & valid[None, :], other=0.0)
            v_hi = tl.load(x_ptr + rows + (off_src + hi)[None, :],
                           mask=mr[:, None] & valid[None, :], other=0.0)
            acc += w * (v_lo + (v_hi - v_lo) * frac[None, :])
            wsum += tl.where(valid, w, 0.0)
        wsum = tl.where(wsum > 0.0, wsum, 1.0)
        out = acc / wsum[None, :]
        own = tl.load(x_ptr + rows + c[None, :], mask=mr[:, None] & mc[None, :], other=0.0)
        out = tl.where((ring == 0)[None, :], own, out)
        tl.store(out_ptr + rows + c[None, :], out, mask=mr[:, None] & mc[None, :])


def _fused(grad, kernel):
    return (flow_triton.rk4_triton_available(grad)
            and kernel.numel() // 2 <= MAX_FUSED_RADIUS)


def _blur_axis_fused(x, out, axis, kernel, norm):
    # x, out :: contiguous, same shape; blur along `axis` (negative index).
    shape = x.shape
    axis = axis % x.dim()
    outer = int(math.prod(shape[:axis]))
    length = int(shape[axis])
    inner = int(math.prod(shape[axis + 1:]))
    radius = kernel.numel() // 2
    if inner == 1:
        block_l, block_i = 256, 1
    else:
        block_l, block_i = 8, 128
    grid = (outer, triton.cdiv(length, block_l), triton.cdiv(inner, block_i))
    _blur_axis_kernel[grid](
        x, out, kernel, norm, length, inner,
        RADIUS=radius, BLOCK_L=block_l, BLOCK_I=block_i)
    return out


def gaussian_kernel(sigma_cells, device=None, dtype=torch.float32):
    """Normalised 1-D Gaussian of width ``sigma_cells`` (radius 3 sigma), or
    None when it would be a delta."""
    sigma = float(sigma_cells)
    if not sigma > 0.0:
        return None
    radius = int(math.ceil(KERNEL_TRUNCATE * sigma))
    if radius < 1:
        return None
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    if float(kernel[radius]) >= 1.0 - 1e-7:
        return None
    return kernel.to(dtype)


def _border_norm(length, kernel, device, dtype):
    # Response of the zero-padded kernel to a constant 1 along one axis: the
    # per-position normaliser that makes the border behave like a truncated,
    # renormalised kernel instead of shrinking toward zero.
    radius = kernel.numel() // 2
    ones = torch.ones(1, 1, length, device=device, dtype=dtype)
    return F.conv1d(ones, kernel.view(1, 1, -1), padding=radius).view(length)


def smooth_cartesian_(grad, sigma_cells):
    """Separable Gaussian blur of ``grad`` ([..., Z, Y, X]) in place."""
    kernel = gaussian_kernel(sigma_cells, grad.device, grad.dtype)
    if kernel is None:
        return grad
    radius = kernel.numel() // 2
    spatial = grad.shape[-3:]
    volumes = grad.reshape(-1, 1, *spatial)  # a view: grads are contiguous
    if _fused(grad, kernel):
        # Three fused passes per (slab, component) volume, ping-ponging with
        # one scratch volume so the transient stays one volume, not a lattice.
        norms = [_border_norm(spatial[axis], kernel, grad.device, grad.dtype)
                 for axis in range(3)]
        scratch = torch.empty_like(volumes[0, 0])
        for index in range(volumes.shape[0]):
            volume = volumes[index, 0]
            _blur_axis_fused(volume, scratch, -3, kernel, norms[0])
            _blur_axis_fused(scratch, volume, -2, kernel, norms[1])
            _blur_axis_fused(volume, scratch, -1, kernel, norms[2])
            volume.copy_(scratch)
        return grad
    for axis in range(3):
        weight_shape = [1, 1, 1, 1, 1]
        weight_shape[2 + axis] = kernel.numel()
        weight = kernel.view(*weight_shape)
        padding = [0, 0, 0]
        padding[axis] = radius
        norm_shape = [1, 1, 1, 1, 1]
        norm_shape[2 + axis] = spatial[axis]
        norm = _border_norm(spatial[axis], kernel, grad.device, grad.dtype).view(*norm_shape)
        # One (slab, component) volume at a time bounds the transient to a
        # single volume rather than the whole lattice.
        for index in range(volumes.shape[0]):
            volume = volumes[index:index + 1]
            volume.copy_(F.conv3d(volume, weight, padding=padding).div_(norm))
    return grad


def smooth_cylindrical_(grad, ring_num_phi, ring_offsets, sigma_cells,
                        across_sigma_cells=0.0):
    """Anisotropic Gaussian blur of a packed cylindrical lattice gradient in place.

    ``grad`` is [..., nz, total_phi] with the rings packed end to end along
    the last axis (``ring_offsets[r]`` .. ``ring_offsets[r + 1]`` holds ring
    r's ``ring_num_phi[r]`` cells). ``sigma_cells`` is the width along the
    sheet: along z (border-renormalised) and around each ring (circular).
    ``across_sigma_cells`` is the width across rings: each tap reads the
    neighbouring ring at the cell's own angle, linearly interpolated between
    that ring's two nearest cells, with the weights renormalised at the
    innermost and outermost rings. Ring 0 is the pinned axis cell: it is left
    alone and never read. Both widths are in cells (the radial spacing, the z
    spacing and the ring arc length are all one cell). The vector components
    are stored in the local (z, radial, tangential) basis, so blurring around
    a ring spreads a radial push as radial pushes at neighbouring angles.
    """
    kernel = gaussian_kernel(sigma_cells, grad.device, grad.dtype)
    radial_kernel = gaussian_kernel(across_sigma_cells, grad.device, grad.dtype)
    if kernel is None and radial_kernel is None:
        return grad
    nz, total = grad.shape[-2], grad.shape[-1]
    planes = grad.reshape(-1, 1, nz, total)  # a view
    ring_num_phi = torch.as_tensor(ring_num_phi, dtype=torch.int64)
    ring_offsets = torch.as_tensor(ring_offsets, dtype=torch.int64)

    if _fused(grad, kernel if kernel is not None else radial_kernel):
        device = grad.device
        num_phi = ring_num_phi.to(device=device, dtype=torch.int32)
        offsets = ring_offsets.to(device=device, dtype=torch.int32)
        cell_ring = torch.repeat_interleave(
            torch.arange(num_phi.numel(), device=device, dtype=torch.int32),
            num_phi.to(torch.int64))
        scratch = torch.empty_like(planes[0, 0])
        block_r, block_c = 8, 128
        grid = (triton.cdiv(nz, block_r), triton.cdiv(total, block_c))
        norm = None if kernel is None else _border_norm(nz, kernel, device, grad.dtype)
        for index in range(planes.shape[0]):
            plane = planes[index, 0]
            # Ping-pong between the plane and one scratch plane; the sequence
            # ends back in the plane whatever subset of passes is active.
            src, dst = plane, scratch
            if kernel is not None:
                _blur_axis_fused(src, dst, -2, kernel, norm)
                src, dst = dst, src
                _blur_rings_kernel[grid](
                    src, dst, kernel, cell_ring, num_phi, offsets, nz, total,
                    RADIUS=kernel.numel() // 2, BLOCK_R=block_r, BLOCK_C=block_c)
                src, dst = dst, src
            if radial_kernel is not None:
                _blur_radial_kernel[grid](
                    src, dst, radial_kernel, cell_ring, num_phi, offsets,
                    nz, total, num_phi.numel(),
                    RADIUS=radial_kernel.numel() // 2, BLOCK_R=block_r, BLOCK_C=block_c)
                src, dst = dst, src
            if src is not plane:
                plane.copy_(src)
        return grad

    num_phi = ring_num_phi.tolist()
    offsets = ring_offsets.tolist()
    lead = grad.shape[:-2]
    if kernel is not None:
        radius = kernel.numel() // 2
        # z pass.
        norm = _border_norm(nz, kernel, grad.device, grad.dtype).view(1, 1, nz, 1)
        weight = kernel.view(1, 1, -1, 1)
        for index in range(planes.shape[0]):
            plane = planes[index:index + 1]
            plane.copy_(F.conv2d(plane, weight, padding=(radius, 0)).div_(norm))

        # phi pass, per ring, circular.
        kernel_1d = kernel.view(1, 1, -1)
        for ring in range(1, len(num_phi)):
            n = num_phi[ring]
            if n < 2:
                continue
            start = offsets[ring]
            cells = grad[..., start:start + n]  # [..., nz, n], a view
            wrapped = torch.arange(-radius, n + radius, device=grad.device) % n
            padded = cells[..., wrapped].reshape(-1, 1, n + 2 * radius)
            cells.copy_(F.conv1d(padded, kernel_1d).view(*lead, nz, n))

    if radial_kernel is not None:
        _blur_radial_eager_(grad, num_phi, offsets, radial_kernel)
    return grad


def _blur_radial_eager_(grad, num_phi, offsets, kernel):
    # Reference radial pass (see _blur_radial_kernel): every ring reads the
    # source rings from an untouched copy, so the pass is one linear operator
    # and not a sweep.
    source = grad.clone()
    radius = kernel.numel() // 2
    weights = kernel.tolist()
    num_rings = len(num_phi)
    for ring in range(1, num_rings):
        n = num_phi[ring]
        start = offsets[ring]
        local = torch.arange(n, device=grad.device, dtype=torch.int64)
        acc = torch.zeros_like(source[..., start:start + n])
        wsum = 0.0
        for k, w in enumerate(weights):
            src_ring = ring + k - radius
            if src_ring < 1 or src_ring >= num_rings:
                continue
            n_src = num_phi[src_ring]
            off_src = offsets[src_ring]
            numerator = local * n_src
            lo = numerator // n  # < n_src since local < n
            frac = (numerator - lo * n).to(grad.dtype) / n
            hi = (lo + 1) % n_src
            v_lo = source[..., off_src + lo]
            v_hi = source[..., off_src + hi]
            acc += w * (v_lo + (v_hi - v_lo) * frac)
            wsum += w
        grad[..., start:start + n] = acc / wsum
    return grad


def describe_widths(along_voxels, across_voxels, cell_voxels, spatial_scale_factor,
                    field_type):
    """One-line report of the effective smoothing widths per lattice.

    Converts the configured scroll-voxel widths to cells of the high- and
    low-resolution lattices and says when a kernel collapses to the identity
    (see gaussian_kernel), so a width that is far below the cell size is
    visible at startup instead of silently doing nothing.
    """
    def per_lattice(voxels):
        parts = []
        for name, scale in (('HR', 1), ('LR', spatial_scale_factor)):
            cells = float(voxels) / (float(cell_voxels) * scale)
            kernel = gaussian_kernel(cells, dtype=torch.float64)
            if kernel is None:
                parts.append(f'{name} {cells:.2f} cells (identity)')
            else:
                parts.append(f'{name} {cells:.2f} cells (radius {kernel.numel() // 2})')
        return ', '.join(parts)

    report = (f'flow gradient smoothing ({field_type}): along-sheet '
              f'{float(along_voxels):g} voxels = {per_lattice(along_voxels)}')
    if field_type == 'cylindrical':
        report += (f'; across rings {float(across_voxels):g} voxels = '
                   f'{per_lattice(across_voxels)}')
    elif float(across_voxels) > 0.0:
        report += ('; across-ring width ignored: a Cartesian lattice is '
                   'smoothed isotropically at the along-sheet width')
    return report
