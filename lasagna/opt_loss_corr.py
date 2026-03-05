from __future__ import annotations

import torch

import model as fit_model


def corr_loss(
	*, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""3D correction point loss: point-to-quad nearest surface, collection-coupled winding error."""
	dev = res.xyz_lr.device
	dt = res.xyz_lr.dtype

	pts_c = res.data.corr_points
	if pts_c is None or pts_c.points_xyz_winda.shape[0] == 0:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1,), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	pts = pts_c.points_xyz_winda.to(device=dev, dtype=dt)   # (K, 4)
	col = pts_c.collection_idx.to(device=dev, dtype=torch.int64)  # (K,)
	K = int(pts.shape[0])
	P = pts[:, :3]      # (K, 3)
	winda = pts[:, 3]   # (K,)

	xyz_lr = res.xyz_lr          # (D, Hm, Wm, 3) — has grad
	normals = res.normals        # (D, Hm, Wm, 3) — detached
	D, Hm, Wm, _ = xyz_lr.shape

	# --- Step 2: unit normals (detached) ---
	n_unit = normals / (normals.norm(dim=-1, keepdim=True) + 1e-12)  # (D, Hm, Wm, 3)

	# --- Step 3: point-to-quad distance search ---
	# Build quads: (D, Hm-1, Wm-1) quads
	Qd = D
	Qh = Hm - 1
	Qw = Wm - 1
	if Qh <= 0 or Qw <= 0:
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1,), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	# Quad corners (detached for nearest search)
	xyz_det = xyz_lr.detach()
	v00 = xyz_det[:, :-1, :-1]  # (D, Qh, Qw, 3)
	v10 = xyz_det[:, 1:, :-1]
	v01 = xyz_det[:, :-1, 1:]
	v11 = xyz_det[:, 1:, 1:]

	NQ = Qd * Qh * Qw  # total quads
	v00_f = v00.reshape(NQ, 3)  # (NQ, 3)
	v10_f = v10.reshape(NQ, 3)
	v01_f = v01.reshape(NQ, 3)
	v11_f = v11.reshape(NQ, 3)

	# For each point and each quad, compute closest point on bilinear quad
	# P: (K, 3), v*_f: (NQ, 3)
	# Expand: P -> (K, 1, 3), v*_f -> (1, NQ, 3)
	P_exp = P.unsqueeze(1)          # (K, 1, 3)
	v00_exp = v00_f.unsqueeze(0)    # (1, NQ, 3)
	v10_exp = v10_f.unsqueeze(0)
	v01_exp = v01_f.unsqueeze(0)
	v11_exp = v11_f.unsqueeze(0)

	# Affine projection: e1 = v10 - v00, e2 = v01 - v00, g = P - v00
	e1 = v10_exp - v00_exp  # (1, NQ, 3)
	e2 = v01_exp - v00_exp  # (1, NQ, 3)
	g = P_exp - v00_exp     # (K, NQ, 3)

	# Solve 2x2 system via Cramer's rule
	e1e1 = (e1 * e1).sum(-1)   # (1, NQ)
	e1e2 = (e1 * e2).sum(-1)   # (1, NQ)
	e2e2 = (e2 * e2).sum(-1)   # (1, NQ)
	ge1 = (g * e1).sum(-1)     # (K, NQ)
	ge2 = (g * e2).sum(-1)     # (K, NQ)

	det = e1e1 * e2e2 - e1e2 * e1e2  # (1, NQ)
	det_safe = det + (det.abs() < 1e-20).float() * 1e-20
	u = (ge1 * e2e2 - ge2 * e1e2) / det_safe  # (K, NQ)
	v = (ge2 * e1e1 - ge1 * e1e2) / det_safe  # (K, NQ)

	# Clamp to [0, 1]
	u = u.clamp(0.0, 1.0)
	v = v.clamp(0.0, 1.0)

	# Closest point on quad: Q(u,v) = v00*(1-u)*(1-v) + v10*u*(1-v) + v01*(1-u)*v + v11*u*v
	u1 = u.unsqueeze(-1)  # (K, NQ, 1)
	v1 = v.unsqueeze(-1)
	Q_closest = v00_exp * (1 - u1) * (1 - v1) + v10_exp * u1 * (1 - v1) + v01_exp * (1 - u1) * v1 + v11_exp * u1 * v1  # (K, NQ, 3)

	# Squared distance
	diff = P_exp - Q_closest  # (K, NQ, 3)
	dist_sq = (diff * diff).sum(-1)  # (K, NQ)

	# Find nearest quad per point
	nearest_idx = dist_sq.argmin(dim=1)  # (K,)

	# Recover (d, h, w) indices from flat quad index
	nearest_d = nearest_idx // (Qh * Qw)
	rem = nearest_idx % (Qh * Qw)
	nearest_h = rem // Qw
	nearest_w = rem % Qw

	# Recover u, v at nearest quad
	kidx = torch.arange(K, device=dev)
	u_near = u[kidx, nearest_idx]  # (K,)
	v_near = v[kidx, nearest_idx]  # (K,)

	# --- Step 4: boundary checks ---
	# Vertex normals at 4 corners of nearest quad
	n00 = n_unit[nearest_d, nearest_h, nearest_w]          # (K, 3)
	n10 = n_unit[nearest_d, nearest_h + 1, nearest_w]
	n01 = n_unit[nearest_d, nearest_h, nearest_w + 1]
	n11 = n_unit[nearest_d, nearest_h + 1, nearest_w + 1]

	# Quad corners (detached)
	c00 = v00_f[nearest_idx]  # (K, 3)
	c10 = v10_f[nearest_idx]
	c01 = v01_f[nearest_idx]
	c11 = v11_f[nearest_idx]

	# 4 edge checks: for each edge (va -> vb), wall normal at each endpoint = cross(edge_dir, vertex_normal)
	# Point passes edge check if on interior side of EITHER endpoint's wall normal
	def _edge_check(va: torch.Tensor, vb: torch.Tensor, na: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
		edge = vb - va  # (K, 3)
		# Wall normals at each endpoint
		wn_a = torch.cross(edge, na, dim=-1)  # (K, 3)
		wn_b = torch.cross(edge, nb, dim=-1)  # (K, 3)
		# Check: dot(P - va, wn_a) >= 0 OR dot(P - vb, wn_b) >= 0
		side_a = ((P - va) * wn_a).sum(-1)  # (K,)
		side_b = ((P - vb) * wn_b).sum(-1)  # (K,)
		return (side_a >= 0) | (side_b >= 0)

	# 4 edges: (00->10), (10->11), (11->01), (01->00)
	ok_e0 = _edge_check(c00, c10, n00, n10)
	ok_e1 = _edge_check(c10, c11, n10, n11)
	ok_e2 = _edge_check(c11, c01, n11, n01)
	ok_e3 = _edge_check(c01, c00, n01, n00)

	# 5th boundary: quad surface normal check — point on correct side
	# Average normal of quad
	quad_n = 0.25 * (n00 + n10 + n01 + n11)  # (K, 3)
	quad_center = 0.25 * (c00 + c10 + c01 + c11)  # (K, 3)
	side_surf = ((P - quad_center) * quad_n).sum(-1)  # (K,)
	# "correct side" means within reasonable distance — we just check that the point
	# isn't behind the surface (negative side). Allow both sides by checking magnitude.
	# Actually, for correction points we want them to be on either side,
	# so the surface check is: the signed distance shouldn't be too extreme.
	# Use a generous threshold: just ensure the point is roughly near the quad.
	ok_surf = side_surf.abs() < (dist_sq[kidx, nearest_idx].sqrt() + 1e-6) * 10.0

	valid = ok_e0 & ok_e1 & ok_e2 & ok_e3 & ok_surf  # (K,)

	# --- Step 5: signed normal distance observation (with grad) ---
	# Re-evaluate using xyz_lr (with grad) at nearest quad positions
	v00_g = xyz_lr[nearest_d, nearest_h, nearest_w]          # (K, 3)
	v10_g = xyz_lr[nearest_d, nearest_h + 1, nearest_w]
	v01_g = xyz_lr[nearest_d, nearest_h, nearest_w + 1]
	v11_g = xyz_lr[nearest_d, nearest_h + 1, nearest_w + 1]

	# Bilinear position at (u, v) — u, v detached
	u_d = u_near.detach().unsqueeze(-1)  # (K, 1)
	v_d = v_near.detach().unsqueeze(-1)  # (K, 1)
	Q_g = v00_g * (1 - u_d) * (1 - v_d) + v10_g * u_d * (1 - v_d) + v01_g * (1 - u_d) * v_d + v11_g * u_d * v_d  # (K, 3)

	# Bilinear-interpolated normal at (u, v) — detached
	n_interp = n00 * (1 - u_d) * (1 - v_d) + n10 * u_d * (1 - v_d) + n01 * (1 - u_d) * v_d + n11 * u_d * v_d  # (K, 3)
	n_interp = n_interp / (n_interp.norm(dim=-1, keepdim=True) + 1e-12)

	# Signed normal distance: dot(P - Q, n_interp)
	# Grad flows through Q_g (vertex positions), normals are detached
	obs = ((P - Q_g) * n_interp).sum(-1)  # (K,)

	# --- Step 6: collection-coupled +/- winda error ---
	err = torch.full((K,), float("nan"), device=dev, dtype=dt)
	uc = torch.unique(col)
	for cid in uc.tolist():
		m = (col == int(cid)) & valid
		if not bool(m.any()):
			continue
		obs_m = obs[m]
		wa_m = winda[m]
		avg_pos = (obs_m - wa_m).mean()
		err_pos = obs_m - (avg_pos + wa_m)
		mse_pos = (err_pos * err_pos).mean()
		avg_neg = (obs_m + wa_m).mean()
		err_neg = obs_m - (avg_neg - wa_m)
		mse_neg = (err_neg * err_neg).mean()
		use_neg = bool((mse_neg < mse_pos).item())
		err[m] = err_neg if use_neg else err_pos

	point_valid = torch.isfinite(err)
	if not bool(point_valid.any()):
		z = torch.zeros((), device=dev, dtype=dt)
		em = torch.zeros((1,), device=dev, dtype=dt)
		mk = torch.zeros_like(em)
		return z, (em,), (mk,)

	# --- Step 7: loss ---
	err_sq = torch.where(point_valid, err * err, torch.zeros_like(err))
	loss = err_sq[point_valid].mean()

	mask = point_valid.to(dtype=dt)  # (K,)
	return loss, (err_sq,), (mask,)
