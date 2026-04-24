from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model


def winding_density_loss_maps(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for winding-density period-sum loss.

	Uses connection vectors (xy_conn) to define strips between the current
	vertex and its prev/next winding neighbors. Integrates grad_mag along
	each strip; the integral should equal 1.0.

	Outputs:
	- lm: (D, 1, He, We)
	- mask: (D, 1, He, We)
	"""
	xy_conn = res.xy_conn      # (D, Hm, Wm, 3, 3) — [prev, self, next]
	mask_conn = res.mask_conn  # (D, 1, Hm, Wm, 3)
	D, Hm, Wm, _, _ = xy_conn.shape
	He = int(res.xyz_hr.shape[1])
	We = int(res.xyz_hr.shape[2])
	device = xy_conn.device
	dtype = xy_conn.dtype

	if D < 2:
		return (torch.zeros(D, 1, He, We, device=device, dtype=dtype),
				torch.zeros(D, 1, He, We, device=device, dtype=dtype))

	strip_samples = max(2, int(res.params.subsample_mesh) + 1)

	# Extract prev, center, next — each (D, Hm, Wm, 3)
	prev_pt = xy_conn[:, :, :, :, 0]
	center_pt = xy_conn[:, :, :, :, 1]
	next_pt = xy_conn[:, :, :, :, 2]

	# Upsample connection endpoints to HR: (D, 3, Hm, Wm) -> (D, 3, He, We)
	def _upsample_hw(pts: torch.Tensor) -> torch.Tensor:
		t = pts.permute(0, 3, 1, 2)  # (D, 3, Hm, Wm)
		t = F.interpolate(t, size=(He, We), mode='bilinear', align_corners=True)
		return t.permute(0, 2, 3, 1)  # (D, He, We, 3)

	prev_hr = _upsample_hw(prev_pt)    # (D, He, We, 3)
	center_hr = _upsample_hw(center_pt)
	next_hr = _upsample_hw(next_pt)

	# Upsample mask_conn: (D, 1, Hm, Wm, 3) -> (D, 1, He, We, 3)
	# Reshape to (D*3, 1, Hm, Wm), upsample, reshape back
	mc = mask_conn.permute(0, 4, 1, 2, 3).reshape(D * 3, 1, Hm, Wm)
	mc = F.interpolate(mc, size=(He, We), mode='nearest')
	mask_conn_hr = mc.reshape(D, 3, 1, He, We).permute(0, 2, 3, 4, 1)  # (D, 1, He, We, 3)

	def _strip_loss(start: torch.Tensor, end: torch.Tensor, sign: torch.Tensor,
				   target: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
		"""Compute strip loss between two endpoint sets.

		start, end: (D, H, W, 3)
		sign: (D, H, W) — +1 correct side, -1 wrong side
		target: target integral value (1.0 for inter-winding, arbitrary for ext offset)
		Returns: lm (D, H, W), strip_valid (D, H, W)
		"""
		D_, H_, W_ = start.shape[:3]
		t = torch.linspace(0.0, 1.0, strip_samples, device=device, dtype=dtype)
		diff = end - start  # (D, H, W, 3)
		strip = start.unsqueeze(-2) + t.view(1, 1, 1, -1, 1) * diff.unsqueeze(-2)  # (D, H, W, S, 3)

		# Flatten strip into W for grid_sample
		strip_flat = strip.reshape(D_, H_, W_ * strip_samples, 3)
		sampled = res.data.grid_sample_fullres(strip_flat)
		mag = sampled.grad_mag.squeeze(0).squeeze(0)  # (D, H, W*S)
		mag = mag.reshape(D_, H_, W_, strip_samples)

		# Unsigned strip length (Euclidean distance between endpoints).
		strip_len = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (D, H, W)

		# Apply sign: wrong-side crossings produce negative integral
		signed_len = strip_len * sign

		# Midpoint-rule line integral
		integral = mag.mean(dim=-1) * signed_len  # (D, H, W)

		# Huber loss on (integral - target): handles target=0 and near-zero integrals
		err = integral - target
		lm = torch.where(err.abs() <= 1.0, 0.5 * err * err, err.abs() - 0.5)  # (D, H, W)

		# Strip validity: all sample points must have grad_mag > 0
		sv = (sampled.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=dtype)
		sv = sv.reshape(D_, H_, W_, strip_samples)
		strip_valid = sv.amin(dim=-1)  # (D, H, W)

		return lm, strip_valid

	# Upsample sign_conn: (D, 1, Hm, Wm, 2) -> (D, 1, He, We, 2)
	sign_conn = res.sign_conn  # (D, 1, Hm, Wm, 2)
	sc = sign_conn.permute(0, 4, 1, 2, 3).reshape(D * 2, 1, Hm, Wm)
	sc = F.interpolate(sc, size=(He, We), mode='nearest')
	sign_conn_hr = sc.reshape(D, 2, 1, He, We).permute(0, 2, 3, 4, 1)  # (D, 1, He, We, 2)
	sign_prev_hr = sign_conn_hr[:, 0, :, :, 0]  # (D, He, We)
	sign_next_hr = sign_conn_hr[:, 0, :, :, 1]  # (D, He, We)

	# Prev strip: prev_hr -> center_hr (strip matches ray direction → keep sign)
	lm_prev, sv_prev = _strip_loss(prev_hr, center_hr, sign_prev_hr)
	# Next strip: center_hr -> next_hr (strip opposes ray direction → negate sign)
	lm_next, sv_next = _strip_loss(center_hr, next_hr, -sign_next_hr)

	# Per-direction masks: each direction gated independently
	# mask_conn_hr: (D, 1, He, We, 3) — [prev, center, next]
	m_prev_ep = mask_conn_hr[:, :, :, :, 0] * mask_conn_hr[:, :, :, :, 1]  # prev & center
	m_next_ep = mask_conn_hr[:, :, :, :, 1] * mask_conn_hr[:, :, :, :, 2]  # center & next
	mask_prev = m_prev_ep * sv_prev.unsqueeze(1)
	mask_next = m_next_ep * sv_next.unsqueeze(1)

	return lm_prev.unsqueeze(1), mask_prev, lm_next.unsqueeze(1), mask_next


def winding_density_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Winding density loss: grad_mag integral between adjacent windings should be 1.0."""
	lm_prev, mask_prev, lm_next, mask_next = winding_density_loss_maps(res=res)
	# Each direction contributes independently
	wsum = mask_prev.sum() + mask_next.sum()
	if float(wsum.detach().cpu()) > 0.0:
		loss = ((lm_prev * mask_prev).sum() + (lm_next * mask_next).sum()) / wsum
	else:
		loss = 0.5 * (lm_prev.mean() + lm_next.mean())
	# Combined lm/mask for visualization
	lm = 0.5 * (lm_prev + lm_next)
	mask = (mask_prev + mask_next).clamp(max=1.0)
	return loss, (lm,), (mask,)


def ext_offset_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""External offset loss via quad-corner proxy targets at LR.

	For each model vertex, computes the signed winding error (integral - target),
	then builds 4 proxy targets at the model position shifted by winding_error
	along each ext surface quad corner's normal (constant, from ext mesh).
	Per-corner L2 losses are weighted by bilinear (u, v) fractions.
	"""
	if res.ext_conn is None or not res.ext_conn:
		device = res.xyz_lr.device
		z = torch.zeros((), device=device)
		return z, (z.unsqueeze(0),), (z.unsqueeze(0),)

	D, Hm, Wm, _ = res.xyz_lr.shape
	device = res.xyz_lr.device
	dtype = res.xyz_lr.dtype
	strip_samples = max(2, int(res.params.subsample_mesh) + 1)

	xyz_det = res.xyz_lr.detach()

	total_loss = torch.zeros((), device=device, dtype=dtype)
	total_wsum = 0.0
	all_lm = []
	all_mask = []

	for (ext_mask, offset,
		 P00, P10, P01, P11,
		 N00, N10, N01, N11,
		 u_frac, v_frac) in res.ext_conn:
		with torch.no_grad():
			# Bilinear ext surface point (for strip integral)
			uf = u_frac.unsqueeze(-1)
			vf = v_frac.unsqueeze(-1)
			ext_pt = (1-uf)*(1-vf)*P00 + uf*(1-vf)*P10 + (1-uf)*vf*P01 + uf*vf*P11

			# Grad_mag along strip from model to ext surface point
			diff = ext_pt - xyz_det
			t = torch.linspace(0.0, 1.0, strip_samples, device=device, dtype=dtype)
			strip = xyz_det.unsqueeze(-2) + t.view(1, 1, 1, -1, 1) * diff.unsqueeze(-2)
			strip_flat = strip.reshape(D, Hm, Wm * strip_samples, 3)
			sampled = res.data.grid_sample_fullres(strip_flat)
			mag = sampled.grad_mag.squeeze(0).squeeze(0).reshape(D, Hm, Wm, strip_samples)
			mean_mag = mag.mean(dim=-1).clamp(min=1e-4)

			# Strip validity
			sv = (sampled.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=dtype)
			sv = sv.reshape(D, Hm, Wm, strip_samples).amin(dim=-1)
			mask = (ext_mask.squeeze(1) * sv).unsqueeze(1)  # (D, 1, Hm, Wm)

			# Signed winding integral: positive on +normal side, negative on -normal side
			strip_len = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
			ext_n = (1-uf)*(1-vf)*N00 + uf*(1-vf)*N10 + (1-uf)*vf*N01 + uf*vf*N11
			side = ((xyz_det - ext_pt) * ext_n).sum(dim=-1).sign()
			side = torch.where(side == 0, torch.ones_like(side), side)
			signed_integral = mean_mag * strip_len * side

			# Winding error: how many windings off from target
			err = signed_integral - offset  # positive = too far on +n side

			# Sample GT normal at intersection point, align with ext surface normal
			gt_sampled = res.data.grid_sample_fullres(ext_pt)
			gt_nx = gt_sampled.nx.squeeze(0).permute(1, 0, 2, 3)  # (D, 1, Hm, Wm) → need (D, Hm, Wm)
			gt_ny = gt_sampled.ny.squeeze(0).permute(1, 0, 2, 3)
			gt_nx = gt_nx.squeeze(1)  # (D, Hm, Wm)
			gt_ny = gt_ny.squeeze(1)
			gt_nz = torch.sqrt((1.0 - gt_nx * gt_nx - gt_ny * gt_ny).clamp(min=0.0))
			gt_n = torch.stack([gt_nx, gt_ny, gt_nz], dim=-1)  # (D, Hm, Wm, 3)
			gt_n = gt_n / (gt_n.norm(dim=-1, keepdim=True) + 1e-8)
			# Flip GT normal to align with ext surface normal
			flip = (gt_n * ext_n).sum(dim=-1, keepdim=True).sign()
			flip = torch.where(flip == 0, torch.ones_like(flip), flip)
			gt_n = gt_n * flip

			# Proxy per corner: model shifted by winding error along GT normal
			e = err.unsqueeze(-1)  # (D, Hm, Wm, 1)
			proxy00 = P00 - e * gt_n
			proxy10 = P10 - e * gt_n
			proxy01 = P01 - e * gt_n
			proxy11 = P11 - e * gt_n

			# Bilinear weights from intersection position within quad
			w00 = ((1 - u_frac) * (1 - v_frac)).unsqueeze(-1)  # (D, Hm, Wm, 1)
			w10 = (u_frac * (1 - v_frac)).unsqueeze(-1)
			w01 = ((1 - u_frac) * v_frac).unsqueeze(-1)
			w11 = (u_frac * v_frac).unsqueeze(-1)

		# Weighted sum of per-corner L2 losses
		d00 = (res.xyz_lr - proxy00).square().sum(dim=-1, keepdim=True)
		d10 = (res.xyz_lr - proxy10).square().sum(dim=-1, keepdim=True)
		d01 = (res.xyz_lr - proxy01).square().sum(dim=-1, keepdim=True)
		d11 = (res.xyz_lr - proxy11).square().sum(dim=-1, keepdim=True)
		lm = (w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11).permute(0, 3, 1, 2)  # (D, 1, Hm, Wm)

		wsum = float(mask.sum().detach().cpu())
		if wsum > 0.0:
			total_loss = total_loss + (lm * mask).sum() / wsum
			total_wsum += wsum
		all_lm.append(lm)
		all_mask.append(mask)

	n_ext = len(res.ext_conn)
	if n_ext > 1:
		total_loss = total_loss / n_ext

	lm_avg = sum(all_lm) / n_ext
	mask_avg = sum(all_mask).clamp(max=1.0) / n_ext
	return total_loss, (lm_avg,), (mask_avg,)
