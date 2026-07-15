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
    winding_indices_t = torch.as_tensor(winding_indices, device=device, dtype=torch.float32)
    theta = torch.rand([len(winding_indices), num_samples], device=device) * (2 * torch.pi)
    z = torch.empty([len(winding_indices), num_samples], device=device).uniform_(float(z_begin), float(z_end - 1))
    radius = (winding_indices_t[:, None] + theta / (2 * torch.pi)) * dr_per_winding
    return torch.stack([
        z,
        torch.sin(theta) * radius,
        torch.cos(theta) * radius,
    ], dim=-1)
