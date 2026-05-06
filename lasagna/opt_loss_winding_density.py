from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model


WINDING_DENSITY_BARRIER_MARGIN = 0.2
WINDING_DENSITY_BARRIER_SCALE = 10.0


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
		sampled = res.data.grid_sample_fullres(strip_flat, channels={"grad_mag"})
		mag = sampled.grad_mag.squeeze(0).squeeze(0)  # (D, H, W*S)
		mag = mag.reshape(D_, H_, W_, strip_samples)

		# Unsigned strip length (Euclidean distance between endpoints).
		strip_len = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (D, H, W)

		# Apply sign: wrong-side crossings produce negative integral
		signed_len = strip_len * sign

		# Midpoint-rule line integral
		integral = mag.mean(dim=-1) * signed_len  # (D, H, W)

		# L2 loss on signed integral residual plus a squared hinge barrier for
		# near-zero / wrong-side signed integrals.
		err = integral - target
		barrier_err = torch.relu(WINDING_DENSITY_BARRIER_MARGIN - integral) * WINDING_DENSITY_BARRIER_SCALE
		lm = err * err + barrier_err * barrier_err  # (D, H, W)

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


EXT_OFFSET_USE_GT_NORMALS = False


def ext_offset_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""External offset loss via per-ext-corner proxy targets.

	For each ext surface corner, finds the model quad it projects onto,
	computes the signed winding error (strip integral - target offset),
	then builds 4 proxy targets at each model quad corner shifted by
	proxy_n * winding_error.  Per-corner L2 losses are weighted by bilinear
	(u, v) fractions from the ray-quad intersection.

	Ext_conn data is upsampled for better coverage between sparse corners.
	"""
	if res.ext_conn is None or not res.ext_conn:
		device = res.xyz_lr.device
		z = torch.zeros((), device=device)
		return z, (z.unsqueeze(0),), (z.unsqueeze(0),)

	device = res.xyz_lr.device
	dtype = res.xyz_lr.dtype
	strip_samples = max(2, int(res.params.subsample_mesh) + 1)
	upsample = max(1, int(res.params.subsample_mesh))

	def _cell_interp_hw(t: torch.Tensor) -> torch.Tensor:
		"""Sample inside external cells; invalid cells are filtered separately."""
		D_, H_, W_, C_ = t.shape
		if H_ < 2 or W_ < 2:
			return torch.empty(D_, 0, 0, C_, device=t.device, dtype=t.dtype)
		a = torch.linspace(0.0, 1.0, upsample, device=t.device, dtype=t.dtype).view(1, 1, 1, upsample, 1, 1)
		b = torch.linspace(0.0, 1.0, upsample, device=t.device, dtype=t.dtype).view(1, 1, 1, 1, upsample, 1)
		q00 = t[:, :-1, :-1].unsqueeze(3).unsqueeze(4)
		q10 = t[:, 1:, :-1].unsqueeze(3).unsqueeze(4)
		q01 = t[:, :-1, 1:].unsqueeze(3).unsqueeze(4)
		q11 = t[:, 1:, 1:].unsqueeze(3).unsqueeze(4)
		out = (1 - a) * (1 - b) * q00 + a * (1 - b) * q10 + (1 - a) * b * q01 + a * b * q11
		return out.reshape(D_, (H_ - 1) * upsample, (W_ - 1) * upsample, C_)

	def _cell_interp_scalar(t: torch.Tensor) -> torch.Tensor:
		return _cell_interp_hw(t.unsqueeze(-1)).squeeze(-1)

	def _cell_mask(corner_mask: torch.Tensor, quad_mask: torch.Tensor, full_h: torch.Tensor, full_w: torch.Tensor) -> torch.Tensor:
		"""Return repeated valid-cell mask. A cell is valid only if all corners are valid."""
		if full_h.shape[1] < 2 or full_h.shape[2] < 2:
			return torch.empty(full_h.shape[0], 1, 0, 0, device=full_h.device, dtype=full_h.dtype)
		cm = corner_mask.squeeze(1).bool()
		finite = torch.isfinite(full_h) & torch.isfinite(full_w)
		cell = (
			quad_mask.squeeze(1).bool() &
			cm[:, :-1, :-1] & cm[:, 1:, :-1] & cm[:, :-1, 1:] & cm[:, 1:, 1:] &
			finite[:, :-1, :-1] & finite[:, 1:, :-1] & finite[:, :-1, 1:] & finite[:, 1:, 1:]
		)
		cell = cell.repeat_interleave(upsample, dim=1).repeat_interleave(upsample, dim=2)
		return cell.unsqueeze(1).to(dtype=full_h.dtype)

	total_loss = torch.zeros((), device=device, dtype=dtype)
	total_wsum = 0.0
	all_lm = []
	all_mask = []

	Hm = int(res.xyz_lr.shape[1])
	Wm = int(res.xyz_lr.shape[2])

	for item in res.ext_conn:
		if len(item) == 7:
			ext_mask, offset, ext_P, ext_N, full_h, full_w, ext_quad_mask = item
		else:
			ext_mask, offset, ext_P, ext_N, full_h, full_w = item
			ext_quad_mask = (
				ext_mask[:, :, :-1, :-1] *
				ext_mask[:, :, 1:, :-1] *
				ext_mask[:, :, :-1, 1:] *
				ext_mask[:, :, 1:, 1:]
			)

		# Sample inside valid external quads only; never interpolate across holes.
		ext_mask_up = _cell_mask(ext_mask, ext_quad_mask, full_h, full_w)
		ext_P_up = _cell_interp_hw(ext_P)
		ext_N_up = _cell_interp_hw(ext_N)
		ext_N_up = ext_N_up / (ext_N_up.norm(dim=-1, keepdim=True) + 1e-8)
		full_h_up = _cell_interp_scalar(full_h)
		full_w_up = _cell_interp_scalar(full_w)

		# Derive per-upsampled-point model quad from interpolated grid position
		D = full_h_up.shape[0]
		He = full_h_up.shape[1]
		We = full_h_up.shape[2]
		if He == 0 or We == 0:
			continue

		# In-bounds: upsampled point must map to valid model quad
		in_bounds = (full_h_up >= 0) & (full_h_up < Hm - 1) & (full_w_up >= 0) & (full_w_up < Wm - 1)
		ext_mask_up = ext_mask_up * in_bounds.unsqueeze(1).to(dtype=dtype)
		sample_valid = ext_mask_up.squeeze(1) > 0

		# Clamp for safe indexing (out-of-bounds already masked)
		fh_safe = torch.where(sample_valid, full_h_up, torch.zeros_like(full_h_up))
		fw_safe = torch.where(sample_valid, full_w_up, torch.zeros_like(full_w_up))
		fh_c = fh_safe.clamp(0, Hm - 1)
		fw_c = fw_safe.clamp(0, Wm - 1)
		row = fh_c.floor().clamp(0, Hm - 2).long()
		col = fw_c.floor().clamp(0, Wm - 2).long()
		u_frac = fh_c - row.float()
		v_frac = fw_c - col.float()

		# Gather model quad corners from xyz_lr (WITH gradients)
		d_idx = torch.arange(D, device=device).view(D, 1, 1).expand(D, He, We)
		M00 = res.xyz_lr[d_idx, row, col]
		M10 = res.xyz_lr[d_idx, row + 1, col]
		M01 = res.xyz_lr[d_idx, row, col + 1]
		M11 = res.xyz_lr[d_idx, row + 1, col + 1]

		with torch.no_grad():
			M00_det = M00.detach()
			M10_det = M10.detach()
			M01_det = M01.detach()
			M11_det = M11.detach()

			# Bilinear model point (detached) for strip sampling
			uf = u_frac.unsqueeze(-1)
			vf = v_frac.unsqueeze(-1)
			M_bilin = (1-uf)*(1-vf)*M00_det + uf*(1-vf)*M10_det + (1-uf)*vf*M01_det + uf*vf*M11_det

			ext_P_safe = torch.where(sample_valid.unsqueeze(-1), ext_P_up, torch.zeros_like(ext_P_up))
			ext_N_safe = torch.where(sample_valid.unsqueeze(-1), ext_N_up, torch.zeros_like(ext_N_up))

			# Strip: ext_P → M_bilin, sample grad_mag
			diff = M_bilin - ext_P_safe
			t = torch.linspace(0.0, 1.0, strip_samples, device=device, dtype=dtype)
			strip = ext_P_safe.unsqueeze(-2) + t.view(1, 1, 1, -1, 1) * diff.unsqueeze(-2)
			strip_flat = strip.reshape(D, He, We * strip_samples, 3)
			sampled = res.data.grid_sample_fullres(strip_flat, channels={"grad_mag"})
			mag = sampled.grad_mag.squeeze(0).squeeze(0).reshape(D, He, We, strip_samples)
			mean_mag = mag.mean(dim=-1).clamp(min=1e-4)

			# Strip validity: all sample points must have grad_mag > 0
			sv = (sampled.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=dtype)
			sv = sv.reshape(D, He, We, strip_samples).amin(dim=-1)
			mask = (ext_mask_up.squeeze(1) * sv).unsqueeze(1)

			# Sign: which side of ext surface (+1 or -1)
			signed_normal_disp = ((M_bilin - ext_P_safe) * ext_N_safe).sum(dim=-1)
			int_sign = torch.sign(signed_normal_disp)

			# Magnitude: unsigned winding count from strip integral
			strip_len = diff.square().sum(dim=-1).sqrt().clamp(min=1e-8)
			unsigned_windings = strip_len * mean_mag

			# Signed windings: sign from intersection, magnitude from integral
			signed_windings = int_sign * unsigned_windings
			winding_err = signed_windings - offset

			# Proxy normal: GT sampled at upsampled ext positions, or ext_N
			if EXT_OFFSET_USE_GT_NORMALS:
				gt_n = res.data.grid_sample_fullres(ext_P_safe, channels={"nx", "ny"}).normal_3d
				dot = (gt_n * ext_N_safe).sum(dim=-1, keepdim=True)
				proxy_n = torch.where(dot >= 0, gt_n, -gt_n)
			else:
				proxy_n = ext_N_safe

			# 4 proxies: model quad corner - proxy_n * winding_err
			we = winding_err.unsqueeze(-1)
			proxy00 = M00_det - proxy_n * we
			proxy10 = M10_det - proxy_n * we
			proxy01 = M01_det - proxy_n * we
			proxy11 = M11_det - proxy_n * we

		# 4 weighted L2 losses (M_i gathered from xyz_lr, has gradients)
		w00 = ((1 - u_frac) * (1 - v_frac)).unsqueeze(1)
		w10 = (u_frac * (1 - v_frac)).unsqueeze(1)
		w01 = ((1 - u_frac) * v_frac).unsqueeze(1)
		w11 = (u_frac * v_frac).unsqueeze(1)

		lm00 = (M00 - proxy00).square().sum(dim=-1).unsqueeze(1)
		lm10 = (M10 - proxy10).square().sum(dim=-1).unsqueeze(1)
		lm01 = (M01 - proxy01).square().sum(dim=-1).unsqueeze(1)
		lm11 = (M11 - proxy11).square().sum(dim=-1).unsqueeze(1)

		lm = w00 * lm00 + w10 * lm10 + w01 * lm01 + w11 * lm11

		wsum = float(mask.sum().detach().cpu())
		if wsum > 0.0:
			total_loss = total_loss + torch.where(mask > 0, lm, torch.zeros_like(lm)).sum() / wsum
			total_wsum += wsum
		all_lm.append(torch.where(mask > 0, lm, torch.full_like(lm, float("nan"))))
		all_mask.append(mask)

	n_ext = len(all_lm)
	if n_ext == 0:
		z = torch.zeros((), device=device, dtype=dtype)
		return z, (z.unsqueeze(0),), (z.unsqueeze(0),)
	if n_ext > 1:
		total_loss = total_loss / n_ext

	lm_stack = torch.stack(all_lm)
	lm_avg = torch.nanmean(lm_stack, dim=0)
	lm_avg = torch.where(torch.isfinite(lm_avg), lm_avg, torch.full_like(lm_avg, float("nan")))
	mask_avg = sum(all_mask).clamp(max=1.0) / n_ext
	return total_loss, (lm_avg,), (mask_avg,)
