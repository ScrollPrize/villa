"""Surface overlap loss utilities for neural tracing training.

The surface overlap loss provides dense supervision for the quadrilateral region
spanned by the four cardinal UV direction points. The region is formed
by two triangles in 3D volume space:
- Triangle 1: u_neg, u_pos, v_pos
- Triangle 2: u_neg, u_pos, v_neg

The model directly predicts a surface overlap segmentation mask, which is compared
to the GT mask using BCE + Dice loss.
"""

import torch
import torch.nn.functional as F

from vesuvius.models.training.loss.nnunet_losses import DC_and_BCE_loss


def _point_in_triangle_3d(points, v0, v1, v2, thickness):
    """
    Check if 3D points lie within `thickness` distance of triangle (v0, v1, v2).

    Uses signed distance to triangle plane + barycentric coordinate check.

    Args:
        points: [*spatial_dims, 3] tensor of query points
        v0, v1, v2: [3] tensors defining triangle vertices
        thickness: Maximum distance from triangle plane to consider "inside"

    Returns:
        Boolean mask of same spatial shape as points
    """
    device = points.device
    dtype = points.dtype

    v0 = v0.to(device=device, dtype=dtype)
    v1 = v1.to(device=device, dtype=dtype)
    v2 = v2.to(device=device, dtype=dtype)

    # Compute triangle edges and normal
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.linalg.cross(edge1, edge2)
    normal_len = torch.linalg.norm(normal)

    if normal_len < 1e-8:
        # Degenerate triangle (collinear points)
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=device)

    normal = normal / normal_len

    # Distance from points to plane
    point_to_v0 = points - v0
    signed_dist = torch.einsum('...d,d->...', point_to_v0, normal)
    dist_to_plane = torch.abs(signed_dist)

    # Project points onto triangle plane for barycentric test
    projected = points - signed_dist[..., None] * normal

    # Barycentric coordinates via dot product method
    # Reference: https://blackpawn.com/texts/pointinpoly/
    v0p = projected - v0

    dot00 = torch.dot(edge2, edge2)
    dot01 = torch.dot(edge2, edge1)
    dot11 = torch.dot(edge1, edge1)

    dot02 = torch.einsum('...d,d->...', v0p, edge2)
    dot12 = torch.einsum('...d,d->...', v0p, edge1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if point is in triangle (with small margin for numerical stability)
    margin = 0.01
    in_triangle = (u >= -margin) & (v >= -margin) & (u + v <= 1.0 + margin)

    # Combine plane distance and barycentric check
    return in_triangle & (dist_to_plane <= thickness)


def render_surf_overlap_mask(u_neg_zyx, u_pos_zyx, v_neg_zyx, v_pos_zyx,
                        min_corner_zyx, crop_size, thickness=2.0):
    """
    Render binary mask for surface overlap region formed by 4 cardinal points.

    The region consists of two triangles in 3D volume space:
    - Triangle 1: u_neg, u_pos, v_pos
    - Triangle 2: u_neg, u_pos, v_neg

    Args:
        u_neg_zyx: [3] tensor - point in negative U direction
        u_pos_zyx: [3] tensor - point in positive U direction
        v_neg_zyx: [3] tensor - point in negative V direction
        v_pos_zyx: [3] tensor - point in positive V direction
        min_corner_zyx: [3] tensor - minimum corner of crop in world coordinates
        crop_size: Integer crop dimension
        thickness: Surface thickness in voxels (default 2.0)

    Returns:
        Binary mask [crop_size, crop_size, crop_size]
    """
    device = u_neg_zyx.device
    dtype = u_neg_zyx.dtype

    # Create coordinate grid in world space
    coords = torch.stack(torch.meshgrid(
        torch.arange(crop_size, device=device, dtype=dtype),
        torch.arange(crop_size, device=device, dtype=dtype),
        torch.arange(crop_size, device=device, dtype=dtype),
        indexing='ij'
    ), dim=-1) + min_corner_zyx.to(device=device, dtype=dtype)  # [Z, Y, X, 3]

    # Check both triangles
    # Triangle 1: u_neg, u_pos, v_pos
    in_tri1 = _point_in_triangle_3d(coords, u_neg_zyx, u_pos_zyx, v_pos_zyx, thickness)
    # Triangle 2: u_neg, u_pos, v_neg
    in_tri2 = _point_in_triangle_3d(coords, u_neg_zyx, u_pos_zyx, v_neg_zyx, thickness)

    return (in_tri1 | in_tri2).float()


# Module-level loss instance (created lazily)
_srf_overlap_loss_fn = None


def _get_srf_overlap_loss_fn():
    """Get or create the surface overlap loss function."""
    global _srf_overlap_loss_fn
    if _srf_overlap_loss_fn is None:
        _srf_overlap_loss_fn = DC_and_BCE_loss(
            bce_kwargs={},
            soft_dice_kwargs={'batch_dice': False, 'ddp': False},
            weight_ce=1.0,
            weight_dice=1.0
        )
    return _srf_overlap_loss_fn


def compute_surf_overlap_loss(pred_surf_overlap, target_surf_overlap_mask, mask=None):
    """
    Compute surface overlap loss using BCE + Dice.

    Args:
        pred_surf_overlap: Predicted surface overlap logits [B, 1, Z, Y, X]
        target_surf_overlap_mask: Binary GT surface overlap [B, 1, Z, Y, X] or [B, Z, Y, X]
        mask: Optional validity mask [B, 1, Z, Y, X]

    Returns:
        Scalar loss tensor
    """
    # Ensure target has channel dim
    if target_surf_overlap_mask.ndim == 4:
        target_surf_overlap_mask = target_surf_overlap_mask.unsqueeze(1)

    if mask is not None and mask.ndim == 4:
        mask = mask.unsqueeze(1)

    target_binary = (target_surf_overlap_mask > 0.5).float()

    loss_fn = _get_srf_overlap_loss_fn()
    return loss_fn(pred_surf_overlap, target_binary, loss_mask=mask)
