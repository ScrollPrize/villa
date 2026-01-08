import torch
import torch.nn.functional as F

def _smoothness_reg(
    mask_cosine: torch.Tensor | None = None,
    mask_img: torch.Tensor | None = None,
    *,
    model,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Smoothness penalty on the *offset* field in coarse grid index space.

    We regularize the learnable offsets along grid indices:
    - x index (horizontal neighbors): ||Δ_off||^2 for (u_offset, v_offset),
    - y index (vertical neighbors):   ||Δ_off||^2 for (u_offset, v_offset),

    i.e. both u- and v-offset components are treated the same in both
    directions; the split into smooth_x / smooth_y is only by *index
    direction*, not by coordinate component.

    Horizontal relations use the line-offset connectivity field: for each
    edge between columns (x,x+1) we build two connections, one from the
    left column towards the right and one from the right column towards the
    left, each vertically displaced according to its own offset and
    interpolated in the neighbor column.

    If coarse-grid masks are provided (shape (1,1,gh,gw)), we build per-edge
    weights from:

        mask_img:      image / validity weighting
        mask_cosine:   cosine-domain band in coarse space

    Inside the cosine band we use full weight (mask_img). Outside the band
    we keep only smoothness, but downweighted:

        - smooth_x: 1/10 of its in-band weight
        - smooth_y: 1/10  of its in-band weight
    """
    off = model.offset  # (1,2,gh,gw)  2 = (u_offset, v_offset)
    _, _, gh, gw = off.shape

    # Horizontal smoothness using line-offset connectivity.
    #
    # We need to compare vectors with the SAME anchor (same coarse vertex).
    #
    # _coarse_x_line_pairs gives per-edge directed connections:
    #   dir=0 (lr): anchor at col x     -> interpolated point on col x+1
    #   dir=1 (rl): anchor at col x+1   -> interpolated point on col x
    #
    # For a vertex at column x (interior 1..gw-2), we have:
    #   v_right = lr vector from edge (x,x+1)   anchored at x
    #   v_left  = rl vector from edge (x-1,x)   anchored at x
    #
    # And we want them to be opposite (straightness / C1):
    #   v_right ≈ -v_left  <=>  v_right + v_left ≈ 0.
    if gw >= 2:
        src_x, nbr_x = model._coarse_x_line_pairs(off)  # (1,2,2,gh,gw-1)
        v_right = nbr_x[:, :, 0] - src_x[:, :, 0]  # (1,2,gh,gw-1), anchor col: 0..gw-2
        v_left = nbr_x[:, :, 1] - src_x[:, :, 1]   # (1,2,gh,gw-1), anchor col: 1..gw-1
        if gw >= 3:
            v_sum = v_right[:, :, :, 1:] + v_left[:, :, :, :-1]  # anchors aligned: col 1..gw-2
            sx_mid = (v_sum * v_sum).mean(dim=1, keepdim=True)    # (1,1,gh,gw-2)
            smooth_x = v_right.new_zeros(1, 1, gh, gw - 1)
            smooth_x[:, :, :, 1:] = sx_mid
        else:
            smooth_x = v_right.new_zeros(1, 1, gh, gw - 1)
    else:
        smooth_x = torch.zeros((), device=off.device, dtype=off.dtype)

    # First-order differences along y index: o[..., j+1, :] - o[..., j, :].
    # Again, treat (u_offset, v_offset) symmetrically via ||Δ_off||^2.
    if gh >= 2:
        dy = off[:, :, 1:, :] - off[:, :, :-1, :]  # (1,2,gh-1,gw)
        dy_sq = (dy * dy).sum(dim=1, keepdim=True)
        smooth_y = dy_sq  # (1,1,gh-1,gw)
    else:
        smooth_y = torch.zeros((), device=off.device, dtype=off.dtype)

    # No masks: use global mean.
    if mask_img is None and mask_cosine is None:
        return smooth_x.mean(), smooth_y.mean()

    # Prepare per-vertex masks.
    if mask_img is None:
        m_img = torch.ones((1, 1, gh, gw), device=off.device, dtype=off.dtype)
    else:
        m_img = mask_img.to(device=off.device, dtype=off.dtype)

    if mask_cosine is None:
        m_cos = torch.ones_like(m_img)
    else:
        m_cos = mask_cosine.to(device=off.device, dtype=off.dtype)

    # Horizontal edges: between (y,x) and (y,x+1).
    if isinstance(smooth_x, torch.Tensor) and smooth_x.numel() > 0:
        m_img_h = 0.5 * (m_img[:, :, :, :-1] + m_img[:, :, :, 1:])  # (1,1,gh,gw-1)
        m_cos_h = 0.5 * (m_cos[:, :, :, :-1] + m_cos[:, :, :, 1:])  # (1,1,gh,gw-1)
        alpha_x = 1.0
        w_h = m_img_h * (m_cos_h + alpha_x * (1.0 - m_cos_h))
        wsum_h = w_h.sum()
        if wsum_h > 0:
            loss_x = (smooth_x * w_h).sum() / wsum_h
        else:
            loss_x = smooth_x.mean()
    else:
        loss_x = smooth_x

    # Vertical edges: between (y,x) and (y+1,x).
    if isinstance(smooth_y, torch.Tensor) and smooth_y.numel() > 0:
        m_img_v = 0.5 * (m_img[:, :, :-1, :] + m_img[:, :, 1:, :])  # (1,1,gh-1,gw)
        m_cos_v = 0.5 * (m_cos[:, :, :-1, :] + m_cos[:, :, 1:, :])  # (1,1,gh-1,gw)
        alpha_y = 1.0
        w_v = m_img_v * (m_cos_v + alpha_y * (1.0 - m_cos_v))
        wsum_v = w_v.sum()
        if wsum_v > 0:
            loss_y = (smooth_y * w_v).sum() / wsum_v
        else:
            loss_y = smooth_y.mean()
    else:
        loss_y = smooth_y

    return loss_x, loss_y

def _line_offset_smooth_reg(mask: torch.Tensor | None = None, *, model) -> torch.Tensor:
    """
    Smoothness penalty on the line_offset field along y for each direction.

    line_offset has shape (1,2,gh,gw), channels:
        0: offset towards left neighbor
        1: offset towards right neighbor

    We regularize first-order differences along the coarse y index for each
    direction channel separately:
        dy = line_offset[:, :, j+1, :] - line_offset[:, :, j, :].

    If a coarse-grid mask is provided (1,1,gh,gw), we weight vertical
    differences by the average of their endpoint weights and normalize by
    the sum of weights. The same mask is broadcast to both directions.
    """
    lo = model.line_offset  # (1,2,gh,gw)
    _, _, gh, gw = lo.shape
    if gh < 2:
        return torch.zeros((), device=lo.device, dtype=lo.dtype)

    dy = lo[:, :, 1:, :] - lo[:, :, :-1, :]  # (1,2,gh-1,gw)
    dy_sq = dy * dy                          # (1,2,gh-1,gw)

    if mask is None:
        return dy_sq.mean()

    m = mask.to(device=lo.device, dtype=lo.dtype)  # (1,1,gh,gw)
    m_v = 0.5 * (m[:, :, 1:, :] + m[:, :, :-1, :])  # (1,1,gh-1,gw)
    wsum = m_v.sum()
    if wsum <= 0:
        return dy_sq.mean()

    w = m_v.expand(1, 2, gh - 1, gw)  # broadcast to both directions
    return (dy_sq * w).sum() / wsum

def _mod_smooth_reg(mask: torch.Tensor | None = None, *, model) -> torch.Tensor:
    """
    Smoothness penalty on modulation parameters (amp, bias) on the coarse grid.

    We only regularize variation along y for modulation. If a coarse-grid
    mask is provided, we weight each vertical difference by the average of
    its endpoint weights (after resampling the mask to the modulation grid
    in x) and normalize by the sum of weights so that differences with one
    in-image endpoint still contribute while fully out-of-image connections
    (both endpoints 0) are ignored.
    """
    # Stack amp and bias so both are regularized consistently.
    mods = torch.cat([model.amp_coarse, model.bias_coarse], dim=1)  # (1,2,gh,gw_mod)

    # First-order differences along y in coarse grid index space.
    dy = mods[:, :, 1:, :] - mods[:, :, :-1, :]   # (1,2,gh-1,gw_mod)
    dy_sq = dy * dy

    if dy_sq.numel() == 0:
        base = torch.zeros((), device=mods.device, dtype=mods.dtype)
        return base

    if mask is None:
        return dy_sq.mean()

    gh_m, gw_m = mods.shape[2], mods.shape[3]
    with torch.no_grad():
        # Resample coarse grid mask (defined on coord grid width) to modulation width.
        mask_mod = F.interpolate(
            mask,
            size=(gh_m, gw_m),
            mode="bilinear",
            align_corners=True,
        )
        # Vertical edges: between rows j and j+1; use average of endpoint weights.
        m_v = 0.5 * (mask_mod[:, :, :-1, :] + mask_mod[:, :, 1:, :])  # (1,1,gh-1,gw_mod)
    wsum = m_v.sum()
    if wsum > 0:
        return (dy_sq.mean(dim=1, keepdim=True) * m_v).sum() / wsum
    return dy_sq.mean()

def _step_reg(mask: torch.Tensor | None = None, *, model, w_img: int, h_img: int) -> torch.Tensor:
    """
    Regularization on coarse *rotated* coords in target-image space.

    For each step along the cosine grid we consider the distance between
    neighboring sample positions after mapping into image space.

    We compute:
    - horizontal neighbor distances (along coarse x),
    - vertical neighbor distances (along coarse y).

    Horizontally we use the line-offset connectivity: each edge between
    columns (x,x+1) is represented by two connections (left->right and
    right->left) with vertical displacement given by the corresponding
    offsets and interpolation along the neighbor column.

    Vertically:
    - enforce each distance to be at least 0.5 * the average vertical distance
        and encourage distances to be close to that average.

    If a coarse-grid mask is provided, we weight horizontal/vertical edges
    by the average of their endpoint weights and normalize by the sum of
    weights, so that edges lying fully outside the image (both endpoints 0)
    do not contribute while edges with one in-image endpoint still receive
    a non-zero weight.
    """
    coords = model.base_grid + model.offset  # (1,2,gh,gw)
    u = coords[:, 0:1]
    v = coords[:, 1:2]

    # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
    x_norm, y_norm = model._apply_global_transform(u, v)

    # Map normalized coords to pixel coordinates of the target image.
    x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
    y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))

    # Horizontal neighbor distances using line-offset connectivity. Work in
    # pixel space by treating (x_pix,y_pix) as a 2D coordinate field on the
    # coarse grid and building left/right connections via interpolation.
    coords_pix = torch.cat([x_pix, y_pix], dim=1)  # (1,2,gh,gw)
    src_h, nbr_h = model._coarse_x_line_pairs(coords_pix)  # (1,2,2,gh,gw-1)
    delta_h = nbr_h - src_h
    dist_h = torch.sqrt((delta_h * delta_h).sum(dim=1, keepdim=True) + 1e-12)  # (1,1,2,gh,gw-1)
    # Average over the two directions for each horizontal edge.
    dist_h = dist_h.mean(dim=2)  # (1,1,gh,gw-1)

    # Vertical neighbor distances (steps along coarse y index).
    dx_v = x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]
    dy_v = y_pix[:, :, 1:, :] - y_pix[:, :, :-1, :]
    dist_v = torch.sqrt(dx_v * dx_v + dy_v * dy_v + 1e-12)  # (1,1,gh-1,gw)

    if mask is not None:
        # Per-edge masks via average of endpoint weights.
        m = mask  # (1,1,gh,gw)
        m_h = 0.5 * (m[:, :, :, :-1] + m[:, :, :, 1:])    # (1,1,gh,gw-1)
        m_v = 0.5 * (m[:, :, :-1, :] + m[:, :, 1:, :])    # (1,1,gh-1,gw)

        # Weighted averages for reference distances.
        wsum_h = m_h.sum()
        if wsum_h > 0:
            avg_h = (dist_h * m_h).sum() / wsum_h
        else:
            avg_h = dist_h.mean()

        wsum_v = m_v.sum()
        if wsum_v > 0:
            avg_v = (dist_v * m_v).sum() / wsum_v
        else:
            avg_v = dist_v.mean()
    else:
        # Unweighted averages over all edges.
        avg_h = dist_h.mean()
        avg_v = dist_v.mean()

    avg_h_det = avg_h.detach()
    avg_v_det = avg_v.detach()

    # Horizontal: enforce each distance to be at least 0.1 * avg horizontal.
    min_h = 0.1 * avg_h_det
    if float(min_h) <= 0.0:
        loss_h = torch.zeros((), device=coords.device, dtype=coords.dtype)
    else:
        shortfall_h = torch.clamp(min_h - dist_h, min=0.0) / min_h
        if mask is not None:
            wsum_h = m_h.sum()
            if wsum_h > 0:
                loss_h = ((shortfall_h * shortfall_h) * m_h).sum() / wsum_h
            else:
                loss_h = (shortfall_h * shortfall_h).mean()
        else:
            loss_h = (shortfall_h * shortfall_h).mean()

    # Vertical: encourage each distance to be close to avg distance.
    target_v = avg_v_det
    diff_v = dist_v - target_v
    if mask is not None:
        wsum_v = m_v.sum()
        if wsum_v > 0:
            loss_v_avg = ((diff_v * diff_v) * m_v).sum() / wsum_v
        else:
            loss_v_avg = (diff_v * diff_v).mean()
    else:
        loss_v_avg = (diff_v * diff_v).mean()

    base = 1 * loss_h + 10 * loss_v_avg
    # base = loss_v_avg
    return base

def _angle_symmetry_reg(mask: torch.Tensor | None = None, *, model, w_img: int, h_img: int) -> torch.Tensor:
    """
    Angle-symmetry regularizer on coarse coords in image space.

    For each horizontal edge between neighboring coarse grid columns, we
    compare the horizontal edge direction to the local vertical direction
    (along coarse y). The loss penalizes deviations from orthogonality:

        L = mean( cos(theta)^2 )

    where theta is the angle between the horizontal edge and the vertical
    direction in image space. This encourages the "rungs" that connect
    neighboring vertical lines to be straight relative to the vertical
    grid lines, while still allowing bending along y.

    If a coarse-grid mask is provided, each (row,col) location contributing
    to the comparison is weighted by an average of the corresponding
    horizontal and vertical edge weights, so that configurations with at
    least one in-image vertex still contribute, while locations whose
    participating edges are fully outside the image receive zero weight.
    """
    coords = model.base_grid + model.offset  # (1,2,gh,gw)
    u = coords[:, 0:1]
    v = coords[:, 1:2]

    # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
    x_norm, y_norm = model._apply_global_transform(u, v)

    # Map normalized coords to pixel coordinates.
    x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
    y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))

    # Build 2D coarse coordinate field in pixel space for connectivity.
    coords_pix = torch.cat([x_pix, y_pix], dim=1)  # (1,2,gh,gw)
    _, _, gh, gw = coords_pix.shape
    if gh < 2 or gw < 2:
        return torch.zeros((), device=coords.device, dtype=coords.dtype)

    # Horizontal edge vectors using line-offset connectivity. For each
    # horizontal edge (x,x+1) we have two directions (left->right, right->left).
    src_h, nbr_h = model._coarse_x_line_pairs(coords_pix)  # (1,2,2,gh,gw-1)
    hvec = nbr_h - src_h                             # (1,2,2,gh,gw-1)
    hvx = hvec[:, 0:1]                               # (1,1,2,gh,gw-1)
    hvy = hvec[:, 1:2]                               # (1,1,2,gh,gw-1)

    # Vertical edge vectors between neighboring rows (top -> bottom).
    dx_v = x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]   # (1,1,gh-1,gw)
    dy_v = y_pix[:, :, 1:, :] - y_pix[:, :, :-1, :]   # (1,1,gh-1,gw)

    # We only compare where both directions are defined:
    # rows 0..gh-2 for vertical, and cols 0..gw-2 for horizontal.
    hvx_use = hvx[:, :, :, 0:gh-1, 0:gw-1]           # (1,1,2,gh-1,gw-1)
    hvy_use = hvy[:, :, :, 0:gh-1, 0:gw-1]
    vvx_base = dx_v[:, :, 0:gh-1, 0:gw-1]            # (1,1,gh-1,gw-1)
    vvy_base = dy_v[:, :, 0:gh-1, 0:gw-1]

    # Broadcast vertical vectors across the two horizontal directions.
    vvx = vvx_base.unsqueeze(2).expand_as(hvx_use)   # (1,1,2,gh-1,gw-1)
    vvy = vvy_base.unsqueeze(2).expand_as(hvy_use)

    # Cosine of angle between horizontal connections and vertical direction.
    eps = 1e-12
    h_norm = torch.sqrt(hvx_use * hvx_use + hvy_use * hvy_use + eps)
    v_norm = torch.sqrt(vvx * vvx + vvy * vvy + eps)
    dot = hvx_use * vvx + hvy_use * vvy
    cos_theta = dot / (h_norm * v_norm + eps)

    # Penalize squared cosine -> encourages orthogonality. This implicitly
    # averages over both connectivity directions per horizontal edge.
    base_unweighted = cos_theta * cos_theta  # (1,1,2,gh-1,gw-1)

    if mask is None:
        return base_unweighted.mean()

def _y_straight_reg(mask: torch.Tensor | None = None, *, model, w_img: int, h_img: int) -> torch.Tensor:
    """
    Straightness regularizer along coarse y: penalize changes in the y-step vector.

    This is a second-difference penalty on the mapped (x_pix,y_pix) field:
    - build per-row step vectors v_j = p_{j+1} - p_j,
    - penalize ||v_{j+1} - v_j||^2.

    If a coarse-grid mask is provided, we weight each second-difference term by the
    average of the three involved vertex weights.
    """
    coords = model.base_grid + model.offset  # (1,2,gh,gw)
    u = coords[:, 0:1]
    v = coords[:, 1:2]

    x_norm, y_norm = model._apply_global_transform(u, v)
    x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
    y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))

    _, _, gh, gw = x_pix.shape
    if gh < 3:
        return torch.zeros((), device=coords.device, dtype=coords.dtype)

    # v_j: step from row j to j+1.
    dx_v = x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]
    dy_v = y_pix[:, :, 1:, :] - y_pix[:, :, :-1, :]
    # second difference of step vectors.
    d2x = dx_v[:, :, 1:, :] - dx_v[:, :, :-1, :]
    d2y = dy_v[:, :, 1:, :] - dy_v[:, :, :-1, :]
    base = d2x * d2x + d2y * d2y  # (1,1,gh-2,gw)

    if mask is None:
        return base.mean()

    m = mask.to(device=coords.device, dtype=coords.dtype)
    # Each term involves rows j, j+1, j+2 -> average their per-vertex weights.
    w = (m[:, :, :-2, :] + m[:, :, 1:-1, :] + m[:, :, 2:, :]) / 3.0  # (1,1,gh-2,gw)
    wsum = w.sum()
    if wsum > 0:
        return (base * w).sum() / wsum
    return base.mean()

    # Build per-location weights from coarse mask: combine horizontal and
    # vertical edge masks so that locations involving out-of-image points
    # are downweighted/ignored.
    m = mask  # (1,1,gh,gw)
    m_h = 0.5 * (m[:, :, :, :-1] + m[:, :, :, 1:])     # (1,1,gh,gw-1)
    m_v = 0.5 * (m[:, :, :-1, :] + m[:, :, 1:, :])     # (1,1,gh-1,gw)

    # Restrict to the (gh-1,gw-1) region used above.
    m_h_use = m_h[:, :, 0:gh-1, 0:gw-1]   # (1,1,gh-1,gw-1)
    m_v_use = m_v[:, :, 0:gh-1, 0:gw-1]   # (1,1,gh-1,gw-1)
    # Average horizontal/vertical contributions at each location.
    m_loc = 0.5 * (m_h_use + m_v_use)
    m_loc = m_loc.unsqueeze(2).expand_as(base_unweighted)    # (1,1,2,gh-1,gw-1)

    wsum = m_loc.sum()
    if wsum > 0:
        return (base_unweighted * m_loc).sum() / wsum
    return base_unweighted.mean()
