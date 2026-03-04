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

	# Per-vertex unit normals for signed strip length.
	# For the arc init, normals point inward (toward center of curvature).
	# Both strip directions (prev→center, center→next) should go anti-normal
	# (outward, toward the adjacent winding).  If a connection point is on the
	# wrong side of the surface, the signed length flips negative, making the
	# density negative and creating a strong penalty.
	xyz_lr = res.xyz_lr  # (D, Hm, Wm, 3)
	edge_h = torch.zeros_like(xyz_lr)
	edge_h[:, 1:-1] = xyz_lr[:, 2:] - xyz_lr[:, :-2]
	edge_h[:, 0] = xyz_lr[:, 1] - xyz_lr[:, 0]
	edge_h[:, -1] = xyz_lr[:, -1] - xyz_lr[:, -2]
	edge_w = torch.zeros_like(xyz_lr)
	edge_w[:, :, 1:-1] = xyz_lr[:, :, 2:] - xyz_lr[:, :, :-2]
	edge_w[:, :, 0] = xyz_lr[:, :, 1] - xyz_lr[:, :, 0]
	edge_w[:, :, -1] = xyz_lr[:, :, -1] - xyz_lr[:, :, -2]
	normals_lr = torch.cross(edge_h, edge_w, dim=-1)  # (D, Hm, Wm, 3)
	normals_lr = normals_lr / torch.sqrt((normals_lr * normals_lr).sum(dim=-1, keepdim=True) + 1e-12)
	normals_hr = _upsample_hw(normals_lr)  # (D, He, We, 3)
	normals_hr = normals_hr / torch.sqrt((normals_hr * normals_hr).sum(dim=-1, keepdim=True) + 1e-12)

	# Upsample mask_conn: (D, 1, Hm, Wm, 3) -> (D, 1, He, We, 3)
	# Reshape to (D*3, 1, Hm, Wm), upsample, reshape back
	mc = mask_conn.permute(0, 4, 1, 2, 3).reshape(D * 3, 1, Hm, Wm)
	mc = F.interpolate(mc, size=(He, We), mode='nearest')
	mask_conn_hr = mc.reshape(D, 3, 1, He, We).permute(0, 2, 3, 4, 1)  # (D, 1, He, We, 3)

	def _strip_loss(start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Compute strip loss between two HR endpoint sets.

		start, end: (D, He, We, 3)
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

		# Signed strip length: projection of displacement onto anti-normal.
		# Positive when strip goes anti-normal (correct: toward adjacent winding).
		# Negative when strip goes along-normal (wrong side / self-intersection).
		# Differentiable — gradient flows through the dot product to mesh positions.
		signed_len = -(diff * normals_hr).sum(dim=-1)  # (D, He, We)

		# mag_n = grad_mag * signed_strip_length
		mag_n = mag * signed_len.unsqueeze(-1)
		lm = ((mag_n - 1.0) ** 2).mean(dim=-1)  # (D, He, We)

		# Strip validity: all sample points must be valid
		if sampled.valid is not None:
			sv = (sampled.valid.squeeze(0).squeeze(0) > 0.5).to(dtype=dtype)
			sv = sv.reshape(D, He, We, strip_samples)
			strip_valid = sv.amin(dim=-1)  # (D, He, We)
		else:
			strip_valid = torch.ones(D, He, We, device=device, dtype=dtype)

		return lm, strip_valid

	# Prev strip: prev_hr -> center_hr
	lm_prev, sv_prev = _strip_loss(prev_hr, center_hr)
	# Next strip: center_hr -> next_hr
	lm_next, sv_next = _strip_loss(center_hr, next_hr)

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
