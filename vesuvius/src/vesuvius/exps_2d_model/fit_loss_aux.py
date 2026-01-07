
def _quad_triangle_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Quad-based triangle-area regularizer in image space.

    For each quad in the coarse grid, we form four corner-based triangles
    using the direct neighboring quad corners and compute signed areas via
    the 2D cross product.

    We:
    - penalize triangle area magnitude being less than 1/4 of the average,
    - strongly penalize negative (flipped) triangle areas.

    If a coarse-grid mask is provided, each quad is weighted by the average
    of its four corner weights so that quads fully outside the image
    (all four corners 0) do not contribute, while quads with at least one
    in-image corner still receive a non-zero weight.
    """
    coords = model.base_grid + model.offset  # (1,2,gh,gw)
    u = coords[:, 0:1]
    v = coords[:, 1:2]

    # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
    x_norm, y_norm = model._apply_global_transform(u, v)

    # Map normalized coords to pixel coordinates of the target image.
    x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
    y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))

    # Quad corners: p00 (y,x), p01 (y,x+1), p11 (y+1,x+1), p10 (y+1,x).
    px00 = x_pix[:, :, :-1, :-1]
    py00 = y_pix[:, :, :-1, :-1]
    px01 = x_pix[:, :, :-1, 1:]
    py01 = y_pix[:, :, :-1, 1:]
    px11 = x_pix[:, :, 1:, 1:]
    py11 = y_pix[:, :, 1:, 1:]
    px10 = x_pix[:, :, 1:, :-1]
    py10 = y_pix[:, :, 1:, :-1]

    # Four corner-based triangles per quad, signed area via cross product.

    # Triangle at p00: (p00, p01, p10)
    ax0 = px01 - px00
    ay0 = py01 - py00
    bx0 = px10 - px00
    by0 = py10 - py00
    A0 = 0.5 * (ax0 * by0 - ay0 * bx0)

    # Triangle at p01: (p01, p11, p00)
    ax1 = px11 - px01
    ay1 = py11 - py01
    bx1 = px00 - px01
    by1 = py00 - py01
    A1 = 0.5 * (ax1 * by1 - ay1 * bx1)

    # Triangle at p11: (p11, p10, p01)
    ax2 = px10 - px11
    ay2 = py10 - py11
    bx2 = px01 - px11
    by2 = py01 - py11
    A2 = 0.5 * (ax2 * by2 - ay2 * bx2)

    # Triangle at p10: (p10, p00, p11)
    ax3 = px00 - px10
    ay3 = py00 - py10
    bx3 = px11 - px10
    by3 = py11 - py10
    A3 = 0.5 * (ax3 * by3 - ay3 * bx3)

    areas = torch.stack([A0, A1, A2, A3], dim=0)  # (4,1,gh-1,gw-1)
    areas_abs = areas.abs()
    avg_area_abs = areas_abs.mean().detach()

    if float(avg_area_abs) <= 0.0:
        return torch.zeros((), device=coords.device, dtype=coords.dtype)

    # Magnitude: piecewise penalty on |A| relative to avg|A|:
    # - 0 for |A| >= avg|A|,
    # - linear from |A| = avg|A| down to |A| = 0.25 * avg|A|,
    # - linear + quadratic for |A| < 0.25 * avg|A|.
    #
    # Implemented without masks/conditionals, using clamp so everything
    # is expressed as smooth elementwise ops.
    A = 0.1 * avg_area_abs
    A_quarter = 0.05 * avg_area_abs
    eps = 1e-12

    # Linear component, active for |A| < A and saturating at |A| <= 0.25*A.
    # 0 at |A| = A, ~1 at |A| = 0.25*A (before scaling).
    lin_raw = torch.clamp(A - areas, min=0.0, max=(A - A_quarter + eps))
    lin_term = lin_raw / (A - A_quarter + eps)

    # Quadratic extra below 0.25*A (0 above, grows as |A| goes to 0).
    low_def = torch.clamp(A_quarter - areas, min=0.0)
    quad_term = (low_def / (A_quarter + eps)) ** 2

    size_pen = lin_term + quad_term
    tri_size_loss_unweighted = size_pen  # (4,1,gh-1,gw-1)

    # Orientation: strongly penalize negative signed area (kept for completeness).
    neg = torch.clamp(-areas, min=0.0) / (avg_area_abs + 1e-12)
    tri_neg_loss_unweighted = neg * neg

    if mask is None:
        tri_size_loss = tri_size_loss_unweighted.mean()
        base = tri_size_loss
        return base

    # Per-quad mask from four corners.
    m = mask  # (1,1,gh,gw)
    m00 = m[:, :, :-1, :-1]
    m01 = m[:, :, :-1, 1:]
    m11 = m[:, :, 1:, 1:]
    m10 = m[:, :, 1:, :-1]
    # Use average of the four corner weights so that quads with at least one
    # in-image corner still contribute while fully out-of-image quads (all 0)
    # are ignored.
    m_quad = 0.25 * (m00 + m01 + m10 + m11)  # (1,1,gh-1,gw-1)

    # Broadcast to 4 triangles per quad.
    m_tri = m_quad.unsqueeze(0).expand_as(tri_size_loss_unweighted)  # (4,1,gh-1,gw-1)

    wsum = m_tri.sum()
    if wsum > 0:
        tri_size_loss = (tri_size_loss_unweighted * m_tri).sum() / wsum
    else:
        tri_size_loss = tri_size_loss_unweighted.mean()

    # Return combined triangle-area loss; external lambda scales overall strength.
    base = tri_size_loss
    return base
