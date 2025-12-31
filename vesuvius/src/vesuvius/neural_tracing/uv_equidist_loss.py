"""UV Equidistance loss - encourages regular spacing of cardinal predictions.

This loss encourages the predicted cardinal direction points (u+, u-, v+, v-)
to form a regular grid by keeping adjacent edge distances uniform and
opposite-axis spans consistent.
"""

import torch
import torch.nn.functional as F


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


def _compute_edge_distances(u_neg, u_pos, v_neg, v_pos):
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


def _compute_axis_distances(u_neg, u_pos, v_neg, v_pos):
    """
    Compute the 2 axis-span distances between opposite cardinals.

    Args:
        u_neg, u_pos, v_neg, v_pos: [3] tensors of (z, y, x) coordinates

    Returns:
        distances: [2] tensor of axis distances (u span, v span)
    """
    d_u = torch.norm(u_pos - u_neg)     # u+ to u-
    d_v = torch.norm(v_pos - v_neg)     # v+ to v-

    return torch.stack([d_u, d_v])


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


def _distance_bounds_penalty(distances, min_distance, max_distance):
    """
    Hinged penalty for distances outside allowed bounds.

    - Below min: 20% tolerance, penalty if distance < 0.8 * min_distance
    - Above max: penalty if distance > max_distance

    Penalty grows quadratically as distance falls outside bounds.

    Args:
        distances: [N] distances
        min_distance: Expected minimum distance for this set
        max_distance: Maximum allowed distance for this set

    Returns:
        penalty: Scalar penalty (exactly 0 if all distances within bounds)
    """
    tolerance = 0.2  # Hardcoded 20% tolerance for minimum
    min_threshold = min_distance * (1 - tolerance)

    # Hinge penalties: relu ensures zero when within bounds
    below_min = torch.relu(min_threshold - distances)  # Positive when too close
    above_max = torch.relu(distances - max_distance)   # Positive when too far

    return (below_min ** 2).mean() + (above_max ** 2).mean()


def compute_uv_equidist_loss_slots_cv(
    pred_cardinals,
    cardinal_positions,
    unknown_mask,
    valid_mask=None,
    threshold=0.5,
    temperature=10.0,
    min_distance=None,
    max_distance=None,
    min_axis_distance=None,
    max_axis_distance=None,
    eps=1e-6
):
    """
    Equidistance loss for slotted training.

    In slotted training, cardinal directions are predicted in separate slots.
    For "unknown" (predicted) cardinals, we extract centroids from heatmaps.
    For "known" (conditioning) cardinals, we use GT positions.

    Combines three terms:
    - Edge CV²: enforces uniform spacing of adjacent edge distances
    - Axis CV²: enforces consistent u- and v-axis spans
    - Distance bounds penalties (hinged): optional limits for each distance set

    Args:
        pred_cardinals: [B, 4, Z, Y, X] predicted cardinal heatmaps (probabilities)
            Channel order: 0=u_neg, 1=u_pos, 2=v_neg, 3=v_pos
        cardinal_positions: [B, 4, 3] GT positions (unperturbed) in crop-local coords
        unknown_mask: [B, 4] bool tensor - True if cardinal is predicted (unknown)
        valid_mask: [B] bool tensor - True if sample has all 4 valid cardinals.
            If None, all samples are considered valid.
        threshold: Blob threshold for centroid extraction
        temperature: Soft threshold sharpness
        min_distance: Expected minimum edge distance.
            If provided, adds hinged penalty for edge distances below 0.8 * min_distance.
        max_distance: Maximum allowed edge distance.
            If provided, adds hinged penalty for edge distances above this threshold.
        min_axis_distance: Expected minimum axis-span distance.
            If provided, adds hinged penalty for axis distances below 0.8 * min_axis_distance.
        max_axis_distance: Maximum allowed axis-span distance.
            If provided, adds hinged penalty for axis distances above this threshold.
        eps: Numerical stability epsilon

    Returns:
        loss: Scalar mean loss over valid samples in batch (0 if no valid samples)
        distances: [B, 6] edge distances followed by axis distances for logging
    """
    B = pred_cardinals.shape[0]

    if valid_mask is None:
        valid_mask = torch.ones(B, dtype=torch.bool, device=pred_cardinals.device)

    losses = []
    all_distances = []

    for i in range(B):
        if not valid_mask[i]:
            # Skip invalid samples - use zeros for distances
            all_distances.append(torch.zeros(6, device=pred_cardinals.device, dtype=pred_cardinals.dtype))
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

        # Compute edge and axis distances
        edge_distances = _compute_edge_distances(u_neg, u_pos, v_neg, v_pos)
        axis_distances = _compute_axis_distances(u_neg, u_pos, v_neg, v_pos)
        all_distances.append(torch.cat([edge_distances, axis_distances]))

        # Coefficient of variation squared (uniformity)
        edge_cv_sq = _coefficient_of_variation_squared(edge_distances, eps)
        axis_cv_sq = _coefficient_of_variation_squared(axis_distances, eps)

        # Distance bounds penalties - hinged (if min/max distance provided)
        bounds_penalty = 0.0
        if min_distance is not None and max_distance is not None:
            bounds_penalty = bounds_penalty + _distance_bounds_penalty(
                edge_distances, min_distance, max_distance
            )
        if min_axis_distance is not None and max_axis_distance is not None:
            bounds_penalty = bounds_penalty + _distance_bounds_penalty(
                axis_distances, min_axis_distance, max_axis_distance
            )

        losses.append(edge_cv_sq + axis_cv_sq + bounds_penalty)

    # Return mean over valid samples only
    if len(losses) == 0:
        loss = torch.tensor(0.0, device=pred_cardinals.device, dtype=pred_cardinals.dtype)
    else:
        loss = torch.stack(losses).mean()

    return loss, torch.stack(all_distances)


def compute_uv_equidist_loss_slots(
    pred_cardinals,
    cardinal_positions,
    unknown_mask,
    valid_mask=None,
    threshold=0.5,
    temperature=10.0,
    eps=1e-6
):
    """GT-anchored equidistance loss for slotted training.

    Penalizes deviation of predicted inter-point distances from GT distances.
    Unlike the CV² version (compute_uv_equidist_loss_slots_cv), this anchors
    to actual geometry and prevents trivial solutions where predictions
    collapse to uniform spacing at wrong locations.

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
        distances: [B, 6] predicted edge distances followed by axis distances for logging
    """
    B = pred_cardinals.shape[0]

    if valid_mask is None:
        valid_mask = torch.ones(B, dtype=torch.bool, device=pred_cardinals.device)

    losses = []
    all_distances = []

    for i in range(B):
        if not valid_mask[i]:
            # Skip invalid samples - use zeros for distances
            all_distances.append(torch.zeros(6, device=pred_cardinals.device, dtype=pred_cardinals.dtype))
            continue

        # Extract predicted positions (centroid for unknown, GT for known)
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

        # Compute predicted distances
        pred_edge_distances = _compute_edge_distances(u_neg, u_pos, v_neg, v_pos)
        pred_axis_distances = _compute_axis_distances(u_neg, u_pos, v_neg, v_pos)
        all_distances.append(torch.cat([pred_edge_distances, pred_axis_distances]))

        # Compute GT distances
        gt_u_neg, gt_u_pos, gt_v_neg, gt_v_pos = cardinal_positions[i]
        gt_edge_distances = _compute_edge_distances(gt_u_neg, gt_u_pos, gt_v_neg, gt_v_pos)
        gt_axis_distances = _compute_axis_distances(gt_u_neg, gt_u_pos, gt_v_neg, gt_v_pos)

        # Huber loss - robust to outliers from poor centroid extraction
        # beta=10.0: quadratic for errors < 10 voxels, linear for larger errors
        edge_loss = F.smooth_l1_loss(pred_edge_distances, gt_edge_distances, beta=10.0)
        axis_loss = F.smooth_l1_loss(pred_axis_distances, gt_axis_distances, beta=10.0)
        losses.append(edge_loss + axis_loss)

    # Return mean over valid samples only
    if len(losses) == 0:
        loss = torch.tensor(0.0, device=pred_cardinals.device, dtype=pred_cardinals.dtype)
    else:
        loss = torch.stack(losses).mean()

    return loss, torch.stack(all_distances)
