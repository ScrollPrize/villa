from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model


def _grid_segment_length_x_px(*, xy_px: torch.Tensor) -> torch.Tensor:
	"""Per-sample distance along horizontal index direction of `xy_px`.

	Returns (N,1,H,W), using the average of adjacent x-segments for interior samples.
	"""
	if xy_px.ndim != 4 or int(xy_px.shape[-1]) != 2:
		raise ValueError("xy_px must be (N,H,W,2)")
	if int(xy_px.shape[2]) <= 1:
		return torch.ones((int(xy_px.shape[0]), 1, int(xy_px.shape[1]), int(xy_px.shape[2])), device=xy_px.device, dtype=xy_px.dtype)

	x = xy_px[..., 0]
	y = xy_px[..., 1]
	dx = x[:, :, 1:] - x[:, :, :-1]
	dy = y[:, :, 1:] - y[:, :, :-1]
	seg = torch.sqrt(dx * dx + dy * dy + 1e-12)

	dist = torch.zeros_like(x)
	dist[:, :, 1:-1] = 0.5 * (seg[:, :, 1:] + seg[:, :, :-1])
	dist[:, :, 0] = seg[:, :, 0]
	dist[:, :, -1] = seg[:, :, -1]
	return dist.unsqueeze(1)


def gradmag_period_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for gradient-magnitude period-sum loss.

	This integrates `grad_mag` along the *connection* polyline (left→mid→right)
	from [`FitResult.xy_conn`](model.py:51), i.e. along the observed normal.

	Outputs:
	- lm: (N,1,Hm,Wm)
	- mask: (N,1,Hm,Wm) from MIN mask over (left,mid,right)
	"""
	xyc = res.xy_conn
	if xyc.ndim != 5 or int(xyc.shape[-2]) != 3 or int(xyc.shape[-1]) != 2:
		raise ValueError("xy_conn must be (N,Hm,Wm,3,2)")

	# Upsample the strip along both dimensions to match HR sampling density.
	n, hm, wm, _k3, _c2 = (int(v) for v in xyc.shape)
	he = int(res.xy_hr.shape[1])
	we = int(res.xy_hr.shape[2])
	xy_strip_lr = xyc.reshape(n, hm, wm * 3, 2)
	xy_strip_lr_nchw = xy_strip_lr.permute(0, 3, 1, 2).contiguous()
	xy_strip_hr_nchw = F.interpolate(xy_strip_lr_nchw, size=(he, int(we * 3)), mode="bilinear", align_corners=True)
	xy_strip_hr = xy_strip_hr_nchw.permute(0, 2, 3, 1).contiguous()

	mag_flat = res.data.grid_sample_px(xy_px=xy_strip_hr).grad_mag
	mag3 = mag_flat.view(n, 1, he, we, 3)
	mag_l = mag3[..., 0]
	mag_m = mag3[..., 1]
	mag_r = mag3[..., 2]

	xy3 = xy_strip_hr.view(n, he, we, 3, 2)
	left = xy3[..., 0, :]
	mid = xy3[..., 1, :]
	right = xy3[..., 2, :]

	dl = left - mid
	dr = right - mid
	eps = 1e-12
	len_lm = torch.sqrt((dl * dl).sum(dim=-1) + eps) * float(res.data.downscale)
	len_mr = torch.sqrt((dr * dr).sum(dim=-1) + eps) * float(res.data.downscale)

	# Trapezoidal rule along the two segments.
	integ = 0.5 * (mag_l + mag_m) * len_lm + 0.5 * (mag_m + mag_r) * len_mr
	# Conn strip integrates two segments (left-mid & mid-right). Empirically this
	# corresponds to ~2x the intended per-period integral, so target half.
	lm = (integ - 0.5) * (integ - 0.5)

	mc_lr = res.mask_conn
	mc_strip_lr = mc_lr.reshape(n, 1, hm, wm * 3)
	mc_strip_hr = F.interpolate(mc_strip_lr, size=(he, int(we * 3)), mode="nearest")
	mc3 = mc_strip_hr.view(n, 1, he, we, 3)
	mask = torch.minimum(torch.minimum(mc3[..., 0], mc3[..., 1]), mc3[..., 2])
	return lm, mask


def _masked_mean(lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	wsum = mask.sum()
	if float(wsum) > 0.0:
		return (lm * mask).sum() / wsum
	return lm.mean()


def gradmag_period_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = gradmag_period_loss_map(res=res)
	return _masked_mean(lm, mask)
