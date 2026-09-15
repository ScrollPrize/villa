import numpy as np
import torch

import geom_utils


def get_spiral_yxs(num_windings, dr_per_winding, inter_point_spacing, group_by_winding=False, device='cuda'):

    # Note this is not differentiable wrt dr_per_winding nor inter_point_spacing!

    # r = b * theta => b = drpw / 2pi
    # ...so r = dr_per_winding * theta / (2 * pi)

    # Kth winding has average radius (K + 0.5) * dr_per_winding => circumference (K + 0.5) * dr_per_winding * 2 * pi
    # ...so should have (K + 0.5) * dr_per_winding * 2 * pi / inter_point_spacing steps
    # can construct these thetas directly, then r's via formula

    thetas = [
        winding_idx * 2 * torch.pi + torch.arange(
            0, 2 * np.pi,
            step=inter_point_spacing / (winding_idx + 0.5) / float(dr_per_winding),
            device=device
        )
        for winding_idx in range(num_windings)
    ]
    radii = [dr_per_winding * thetas_for_winding / (2 * torch.pi) for thetas_for_winding in thetas]

    yxs = [
        torch.stack([torch.sin(thetas_for_winding), torch.cos(thetas_for_winding)], dim=-1) * radii_for_winding[:, None]
        for thetas_for_winding, radii_for_winding in zip(thetas, radii)
    ]

    if group_by_winding:
        return yxs
    else:
        return torch.cat(yxs, dim=0)


def get_spiral_points(predictions_slice, centre_xy, dr_per_winding=10):

    inter_point_spacing = 4  # pixels; this doesn't affect the shape of the spiral, just where we sample it

    # This only affects how far 'out' we go, it doesn't affect the shape. We set it such that the spiral just
    # touches the most distant-from-umbilicus edge of the slice
    num_windings = int(1 + np.maximum(centre_xy, predictions_slice.shape[::-1] - centre_xy).max() / dr_per_winding)

    yxs = centre_xy[::-1] + get_spiral_yxs(num_windings, dr_per_winding, inter_point_spacing).cpu().numpy()

    yxs = (yxs + 0.5).astype(np.int64)
    yxs = yxs[(0 <= yxs[:, 0]) & (yxs[:, 0] < predictions_slice.shape[0])]
    yxs = yxs[(0 <= yxs[:, 1]) & (yxs[:, 1] < predictions_slice.shape[1])]

    return yxs


def get_winding_xy(winding_idx, theta, dr_per_winding):
    winding_radius = winding_idx * dr_per_winding + theta / (2 * np.pi) * dr_per_winding
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * winding_radius[..., None]


def get_theta(relative_yx):
    relative_yx = torch.stack([
        relative_yx[..., 0],
        torch.where(relative_yx[..., 1].abs() < 1.e-10, 1.e-10, relative_yx[..., 1]),
    ], dim=-1)  # avoid NaN gradients from atan2 / sqrt
    theta = torch.arctan2(relative_yx[..., 0], relative_yx[..., 1]) % (2 * np.pi)  # [0, 2pi]; zero along x-axis
    return theta, relative_yx


@geom_utils.maybe_compile
def get_theta_and_radii(relative_yx, dr_per_winding):
    theta, relative_yx = get_theta(relative_yx)
    radius = torch.linalg.norm(relative_yx, dim=-1)
    # The spiral has radius 0 at winding angle 0 then increases linearly at rate dr_per_winding
    # Note get_fibre_loss assumes this form!
    shifted_radius = radius - theta / (2 * np.pi) * dr_per_winding
    shifted_radius = shifted_radius.clamp(min=0.)
    return theta, radius, shifted_radius


@geom_utils.maybe_compile
def get_theta_crossing_step_adjustments(theta, dr_per_winding, dim=-1):
    """Return shifted-radius corrections for consecutive theta=0 crossings."""
    if theta.shape[dim] <= 1:
        shape = list(theta.shape)
        shape[dim] = 0
        return torch.empty(shape, device=theta.device, dtype=theta.dtype)

    theta_diffs = torch.diff(theta.detach(), dim=dim)
    return (
        (theta_diffs > np.pi).to(theta.dtype)
        - (theta_diffs < -np.pi).to(theta.dtype)
    ) * dr_per_winding.detach()


@geom_utils.maybe_compile
def unwrap_shifted_radii(theta, shifted_radii, dr_per_winding, dim=-1):
    """Unwrap shifted radii along ``dim`` and return their crossing adjustments."""
    if theta.shape[dim] == 0:
        return shifted_radii, torch.zeros_like(shifted_radii)

    step_adjustments = get_theta_crossing_step_adjustments(
        theta, dr_per_winding, dim=dim,
    )
    zero_shape = list(theta.shape)
    zero_shape[dim] = 1
    adjustments = torch.cat([
        torch.zeros(
            zero_shape,
            device=shifted_radii.device,
            dtype=shifted_radii.dtype,
        ),
        torch.cumsum(step_adjustments.to(shifted_radii.dtype), dim=dim),
    ], dim=dim)
    return shifted_radii + adjustments, adjustments


@geom_utils.maybe_compile
def radius_from_unwrapped_shifted(
    theta, unwrapped_shifted_radii, crossing_adjustments, dr_per_winding,
):
    """Convert unwrapped shifted radii back to each point's wrapped-theta radius."""
    raw_shifted_radii = unwrapped_shifted_radii - crossing_adjustments
    return raw_shifted_radii + theta / (2 * np.pi) * dr_per_winding


def get_radial_covector_in_scroll_space(slice_to_spiral_transform, scroll_zyx, spiral_zyx=None, epsilon=6.0):
    """Pull the spiral-space radial unit normal back to scroll space, unnormalised.

    At each scroll-space point the outward radial direction of spiral space,
    normalize(spiral_yx), is a surface normal of the fitted sheet; a normal
    is a covector and transports as ``J^T n`` with ``J = d(spiral)/d(scroll)``
    (unlike a tangent vector, which pushes forward as ``J d``). ``J`` is
    estimated by central differences of ``epsilon`` along the three scroll
    axes. Returns ``J^T n`` (num_points, 3) in zyx, *not* normalised: its
    direction is the scroll-space sheet normal and its length is the local
    stretch of the transform along that normal (see
    get_radial_normal_stretch). Gradient flows through the transform
    parameters via the Jacobian only; the sample positions and the radial
    direction are held fixed. If the forward image ``spiral_zyx`` is supplied
    it is reused for the radial direction (as a constant); otherwise it is
    computed here.
    """
    device = scroll_zyx.device
    num_points = scroll_zyx.shape[0]
    scroll_zyx = scroll_zyx.detach()

    basis_zyx = torch.eye(3, device=device, dtype=scroll_zyx.dtype) * epsilon
    scroll_plus = (scroll_zyx[None, :, :] + basis_zyx[:, None, :]).reshape(-1, 3)
    scroll_minus = (scroll_zyx[None, :, :] - basis_zyx[:, None, :]).reshape(-1, 3)
    if spiral_zyx is None:
        combined_spiral = slice_to_spiral_transform(torch.cat([scroll_zyx, scroll_plus, scroll_minus], dim=0))
        spiral_zyx = combined_spiral[:num_points]
        spiral_plus, spiral_minus = combined_spiral[num_points:].chunk(2, dim=0)
    else:
        spiral_plus, spiral_minus = slice_to_spiral_transform(torch.cat([scroll_plus, scroll_minus], dim=0)).chunk(2, dim=0)

    spiral_outward_yx = torch.nn.functional.normalize(spiral_zyx[:, 1:].detach(), dim=-1)
    spiral_outward_zyx = torch.cat([torch.zeros_like(spiral_outward_yx[:, :1]), spiral_outward_yx], dim=-1)

    spiral_plus = spiral_plus.view(3, num_points, 3)
    spiral_minus = spiral_minus.view(3, num_points, 3)
    jacobian_columns = (spiral_plus - spiral_minus) / (2.0 * epsilon)  # scroll basis axis, point, spiral zyx
    return (jacobian_columns * spiral_outward_zyx[None, :, :]).sum(dim=-1).transpose(0, 1)


# Central-difference step (input-frame voxels) for the normal-stretch estimate
# behind the vertical-fiber radial offset. Matches constraint_baking's
# direction-transport default; the flow field is smooth at this scale.
RADIAL_OFFSET_STRETCH_EPSILON = 2.0


def get_radial_normal_stretch(slice_to_spiral_transform, scroll_zyx, spiral_zyx=None, *,
                              epsilon=RADIAL_OFFSET_STRETCH_EPSILON, device=None,
                              chunk_size=65536):
    """Local stretch of the transform along the fitted sheet's normal, ``|J^T n|``.

    A displacement of ``d`` input-frame voxels along the scroll-space sheet
    normal ``m = J^T n / |J^T n|`` (the scan-space gradient of the fitted
    winding, which is not in general the line to the umbilicus) lands
    ``d * n . J m = d * |J^T n|`` further out in spiral radius, so a physical
    offset measured in the input frame is converted to a spiral-space radial
    offset by this factor (1 under a rigid motion; the gap expander and flow
    make it vary per point). The vertical-fiber offset is applied along
    ``m`` itself, the increasing-winding direction (the sheet's back face,
    away from the umbilicus). Evaluated
    under no_grad, chunked and RNG-free, with each chunk staged to ``device``
    when given (the transform's device) and the result returned on
    ``scroll_zyx``'s device as (num_points,) float32. ``spiral_zyx`` may
    supply the already-computed forward image of ``scroll_zyx``.
    """
    scroll_zyx = torch.as_tensor(scroll_zyx)
    flat = scroll_zyx.reshape(-1, 3)
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=flat.device)
    if flat.shape[0] == 0:
        return out
    spiral_flat = None if spiral_zyx is None else torch.as_tensor(spiral_zyx).reshape(-1, 3)
    target = device if device is not None else flat.device
    with torch.no_grad():
        for start in range(0, flat.shape[0], chunk_size):
            points = flat[start:start + chunk_size].to(device=target, dtype=torch.float32)
            image = None
            if spiral_flat is not None:
                image = spiral_flat[start:start + chunk_size].to(device=target, dtype=torch.float32)
            covector = get_radial_covector_in_scroll_space(
                slice_to_spiral_transform, points, spiral_zyx=image, epsilon=epsilon)
            out[start:start + chunk_size] = torch.linalg.norm(covector, dim=-1).to(
                device=out.device, dtype=out.dtype)
    return out


def get_bounding_windings(relative_yx, dr_per_winding):
    # The spiral has radius 0 at winding angle 0 then increases linearly at rate dr_per_winding
    # Want to find the two windings that bracket yx
    # If theta=+eps, then these are given by floor/ceil of radius / dr_per_winding
    # For other theta, we shift the point radially so 'as if' it were at theta=0
    theta, radius, shifted_radius = get_theta_and_radii(relative_yx, dr_per_winding)
    inner_winding = torch.floor(shifted_radius / dr_per_winding)
    outer_winding = torch.ceil(shifted_radius / dr_per_winding)
    return theta, radius, inner_winding, outer_winding


def get_spiral_density(relative_yx, dr_per_winding=10., sigma=3., winding_range=None):
    if winding_range is None:
        min_w, max_w = float('-inf'), float('inf')
    else:
        min_w, max_w = winding_range
    theta, radius, inner_winding, outer_winding = get_bounding_windings(relative_yx, dr_per_winding)
    def evaluate_kernel(winding_idx):
        winding_xy = get_winding_xy(winding_idx, theta, dr_per_winding)
        distance = torch.linalg.norm(winding_xy.flip(-1) - relative_yx, dim=-1)
        kernel = torch.exp(-distance ** 2 / sigma ** 2)
        kernel = torch.where((winding_idx >= min_w) & (winding_idx < max_w), kernel, torch.zeros_like(kernel))
        return kernel
    result = evaluate_kernel(inner_winding) + evaluate_kernel(outer_winding)
    return result.clip(0., 1.)


def canonical_winding_samples(winding_indices, num_samples, dr_per_winding, device, z_begin, z_end):
    winding_indices_t = geom_utils.pinned_to_device(
        torch.as_tensor(winding_indices, dtype=torch.float32), device)
    theta = torch.rand([len(winding_indices), num_samples], device=device) * (2 * torch.pi)
    z = torch.empty([len(winding_indices), num_samples], device=device).uniform_(float(z_begin), float(z_end - 1))
    radius = (winding_indices_t[:, None] + theta / (2 * torch.pi)) * dr_per_winding
    return torch.stack([
        z,
        torch.sin(theta) * radius,
        torch.cos(theta) * radius,
    ], dim=-1)
