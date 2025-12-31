"""UV Equidistance loss - enforces uniform spacing of cardinal predictions.

This loss encourages the predicted cardinal direction points (u+, u-, v+, v-)
to be equidistant from each other, promoting a regular grid pattern in the
surface mesh predictions.
"""

import torch


def differentiable_centroid(heatmap, threshold=0.5, temperature=10.0):
    """
    Differentiable blob centroid extraction mimicking cc3d behavior.

    Uses a soft threshold to create a differentiable approximation of
    connected component centroid extraction.

    Args:
        heatmap: [Z, Y, X] probabilities (after sigmoid)
        threshold: Soft threshold for blob membership
        temperature: Sharpness of threshold (higher = sharper)

    Returns:
        centroid: [3] (z, y, x) coordinates in voxel space
    """
    Z, Y, X = heatmap.shape
    device = heatmap.device
    dtype = heatmap.dtype

    # Soft threshold (differentiable approximation of > threshold)
    weights = torch.sigmoid(temperature * (heatmap - threshold))

    # Create coordinate grids
    z_coords = torch.arange(Z, device=device, dtype=dtype)
    y_coords = torch.arange(Y, device=device, dtype=dtype)
    x_coords = torch.arange(X, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # Weighted centroid
    total_weight = weights.sum() + 1e-8
    cz = (weights * zz).sum() / total_weight
    cy = (weights * yy).sum() / total_weight
    cx = (weights * xx).sum() / total_weight

    return torch.stack([cz, cy, cx])


def _compute_pairwise_distances(u_neg, u_pos, v_neg, v_pos):
    """
    Compute all 6 pairwise Euclidean distances between 4 cardinal points.

    Args:
        u_neg, u_pos, v_neg, v_pos: [3] tensors of (z, y, x) coordinates

    Returns:
        distances: [6] tensor of pairwise distances
    """
    d_u = torch.norm(u_pos - u_neg)      # u+ to u-
    d_v = torch.norm(v_pos - v_neg)      # v+ to v-
    d_pp = torch.norm(u_pos - v_pos)     # u+ to v+
    d_pn = torch.norm(u_pos - v_neg)     # u+ to v-
    d_np = torch.norm(u_neg - v_pos)     # u- to v+
    d_nn = torch.norm(u_neg - v_neg)     # u- to v-

    return torch.stack([d_u, d_v, d_pp, d_pn, d_np, d_nn])


def _coefficient_of_variation_squared(distances, eps=1e-6):
    """
    Compute coefficient of variation squared (scale-invariant variance).

    Args:
        distances: [6] tensor of pairwise distances
        eps: Numerical stability epsilon

    Returns:
        cv_sq: Scalar coefficient of variation squared
    """
    mean_d = distances.mean()
    var_d = ((distances - mean_d) ** 2).mean()
    return var_d / (mean_d ** 2 + eps)


def compute_uv_equidist_loss_slots(
    pred_cardinals,
    cardinal_positions,
    unknown_mask,
    threshold=0.5,
    temperature=10.0,
    eps=1e-6
):
    """
    Equidistance loss for slotted training.

    In slotted training, cardinal directions are predicted in separate slots.
    For "unknown" (predicted) cardinals, we extract centroids from heatmaps.
    For "known" (conditioning) cardinals, we use GT positions.

    Args:
        pred_cardinals: [B, 4, Z, Y, X] predicted cardinal heatmaps (probabilities)
            Channel order: 0=u_neg, 1=u_pos, 2=v_neg, 3=v_pos
        cardinal_positions: [B, 4, 3] GT positions (unperturbed) in crop-local coords
        unknown_mask: [B, 4] bool tensor - True if cardinal is predicted (unknown)
        threshold: Blob threshold for centroid extraction
        temperature: Soft threshold sharpness
        eps: Numerical stability epsilon

    Returns:
        loss: Scalar mean loss over batch
        distances: [B, 6] pairwise distances for logging
    """
    B = pred_cardinals.shape[0]

    losses = []
    all_distances = []

    for i in range(B):
        positions = []
        for c in range(4):  # 0=u_neg, 1=u_pos, 2=v_neg, 3=v_pos
            if unknown_mask[i, c]:
                # This cardinal is predicted - extract centroid from heatmap
                pos = differentiable_centroid(pred_cardinals[i, c], threshold, temperature)
            else:
                # This cardinal is known - use GT position
                pos = cardinal_positions[i, c]
            positions.append(pos)

        u_neg, u_pos, v_neg, v_pos = positions

        # Compute 6 pairwise distances
        distances = _compute_pairwise_distances(u_neg, u_pos, v_neg, v_pos)
        all_distances.append(distances)

        # Coefficient of variation squared
        cv_sq = _coefficient_of_variation_squared(distances, eps)
        losses.append(cv_sq)

    return torch.stack(losses).mean(), torch.stack(all_distances)
