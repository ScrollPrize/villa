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

	def _strip_loss(start: torch.Tensor, end: torch.Tensor, sign: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Compute strip loss between two HR endpoint sets.

		start, end: (D, He, We, 3)
		sign: (D, He, We) — +1 correct side, -1 wrong side
		Returns: lm (D, He, We), strip_valid (D, He, We)
		"""
		t = torch.linspace(0.0, 1.0, strip_samples, device=device, dtype=dtype)
		diff = end - start  # (D, He, We, 3)
		strip = start.unsqueeze(-2) + t.view(1, 1, 1, -1, 1) * diff.unsqueeze(-2)  # (D, He, We, S, 3)

		# Flatten strip into W for grid_sample
		strip_flat = strip.reshape(D, He, We * strip_samples, 3)
		sampled = res.data.grid_sample_fullres(strip_flat)
		mag = sampled.grad_mag.squeeze(0).squeeze(0)  # (D, He, We*S)
		mag = mag.reshape(D, He, We, strip_samples)

		# Unsigned strip length (Euclidean distance between endpoints).
		strip_len = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (D, He, We)

		# Apply sign: wrong-side crossings produce negative integral (~-1.0 vs target +1.0)
		signed_len = strip_len * sign

		# mag_n = grad_mag * signed_strip_length; target is 1.0 (one winding traversed)
		mag_n = mag * signed_len.unsqueeze(-1)
		lm = ((mag_n - 1.0) ** 2).mean(dim=-1)  # (D, He, We)

		# Strip validity: all sample points must have grad_mag > 0
		sv = (sampled.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=dtype)
		sv = sv.reshape(D, He, We, strip_samples)
		strip_valid = sv.amin(dim=-1)  # (D, He, We)

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
