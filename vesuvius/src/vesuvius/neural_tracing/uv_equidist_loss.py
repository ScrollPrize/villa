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
    Compute the 4 edge distances between adjacent cardinal points.

    Only computes single-step edge distances around the diamond perimeter,
    excluding diagonal spans (u_neg↔u_pos, v_neg↔v_pos) which have different
    scale and cannot be equal to edge distances for planar points.

    Args:
        u_neg, u_pos, v_neg, v_pos: [3] tensors of (z, y, x) coordinates

    Returns:
        distances: [4] tensor of edge distances
    """
    d_nn = torch.norm(u_neg - v_neg)     # u- to v-
    d_np = torch.norm(u_neg - v_pos)     # u- to v+
    d_pn = torch.norm(u_pos - v_neg)     # u+ to v-
    d_pp = torch.norm(u_pos - v_pos)     # u+ to v+

    return torch.stack([d_nn, d_np, d_pn, d_pp])


def _coefficient_of_variation_squared(distances, eps=1e-6):
    """
    Compute coefficient of variation squared (scale-invariant variance).

    Args:
        distances: [4] tensor of edge distances
        eps: Numerical stability epsilon

    Returns:
        cv_sq: Scalar coefficient of variation squared
    """
    mean_d = distances.mean()
    # Handle degenerate case: if all points collapsed to same location,
    # distances are ~0 and we should return a penalty to push them apart
    if mean_d < eps:
        return torch.tensor(1.0, device=distances.device, dtype=distances.dtype)
    var_d = ((distances - mean_d) ** 2).mean()
    return var_d / (mean_d ** 2 + eps)


def compute_uv_equidist_loss_slots(
    pred_cardinals,
    cardinal_positions,
    unknown_mask,
    valid_mask=None,
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
        valid_mask: [B] bool tensor - True if sample has all 4 valid cardinals.
            If None, all samples are considered valid.
        threshold: Blob threshold for centroid extraction
        temperature: Soft threshold sharpness
        eps: Numerical stability epsilon

    Returns:
        loss: Scalar mean loss over valid samples in batch (0 if no valid samples)
        distances: [B, 4] edge distances for logging
    """
    B = pred_cardinals.shape[0]

    if valid_mask is None:
        valid_mask = torch.ones(B, dtype=torch.bool, device=pred_cardinals.device)

    losses = []
    all_distances = []

    for i in range(B):
        if not valid_mask[i]:
            # Skip invalid samples - use zeros for distances
            all_distances.append(torch.zeros(4, device=pred_cardinals.device, dtype=pred_cardinals.dtype))
            continue

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

        # Compute 4 edge distances
        distances = _compute_pairwise_distances(u_neg, u_pos, v_neg, v_pos)
        all_distances.append(distances)

        # Coefficient of variation squared
        cv_sq = _coefficient_of_variation_squared(distances, eps)
        losses.append(cv_sq)

    # Return mean over valid samples only
    if len(losses) == 0:
        loss = torch.tensor(0.0, device=pred_cardinals.device, dtype=pred_cardinals.dtype)
    else:
        loss = torch.stack(losses).mean()

    return loss, torch.stack(all_distances)
