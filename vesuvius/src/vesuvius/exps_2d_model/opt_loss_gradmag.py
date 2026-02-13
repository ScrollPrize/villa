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

	# xyc shape is z * h * w * l/m/r * xy-loc
	xyc = res.xy_conn
	if xyc.ndim != 5 or int(xyc.shape[-2]) != 3 or int(xyc.shape[-1]) != 2:
		raise ValueError("xy_conn must be (N,Hm,Wm,3,2)")

	# print("xyc size", xyc.shape, "xy-hr", res.xy_hr.shape)

	# Resize strips with the same HR/LR scale factor as xy_hr vs xy_lr.
	n, hm, wm, _k3, _c2 = (int(v) for v in xyc.shape)
	strip_scale = int(res.subsample_mesh)
	strip_scale = max(1, strip_scale)
	strip_samples = int(strip_scale) + 1

	# Upsample (Hm,Wm) to (Hm*scale+1,Wm*scale+1), keeping strips separate:
	# - left strip: (L,C)
	# - right strip: (C,R)
	he = int(hm) * int(strip_scale) + 1
	we = int(wm) * int(strip_scale) + 1
	# print("strip_scale", strip_scale, "strip_samples", strip_samples, "he/we", (he, we))

	xy_l_lr = xyc[..., 0:1, :]
	xy_c_lr = xyc[..., 1:2, :]
	xy_r_lr = xyc[..., 2:3, :]
	xy_l_nchw = xy_l_lr.permute(0, 4, 3, 1, 2).reshape(n, 2, hm, wm).contiguous()
	xy_c_nchw = xy_c_lr.permute(0, 4, 3, 1, 2).reshape(n, 2, hm, wm).contiguous()
	xy_r_nchw = xy_r_lr.permute(0, 4, 3, 1, 2).reshape(n, 2, hm, wm).contiguous()
	xy_l_hr = F.interpolate(xy_l_nchw, size=(he, we), mode="bilinear", align_corners=True).reshape(n, 2, he, we).permute(0, 2, 3, 1).contiguous()
	xy_c_hr = F.interpolate(xy_c_nchw, size=(he, we), mode="bilinear", align_corners=True).reshape(n, 2, he, we).permute(0, 2, 3, 1).contiguous()
	xy_r_hr = F.interpolate(xy_r_nchw, size=(he, we), mode="bilinear", align_corners=True).reshape(n, 2, he, we).permute(0, 2, 3, 1).contiguous()

	left = xy_l_hr
	center = xy_c_hr
	right = xy_r_hr

	t = torch.linspace(0.0, 1.0, strip_samples, device=xyc.device, dtype=xyc.dtype)
	vec_lc = (center - left).unsqueeze(-2)
	vec_cr = (right - center).unsqueeze(-2)
	xy_lc = left.unsqueeze(-2) + t.view(1, 1, 1, -1, 1) * vec_lc
	xy_cr = center.unsqueeze(-2) + t.view(1, 1, 1, -1, 1) * vec_cr
	xy_lc_strip = xy_lc.reshape(n, he, we * strip_samples, 2)
	xy_cr_strip = xy_cr.reshape(n, he, we * strip_samples, 2)

	mag_lc = res.data.grid_sample_px(xy_px=xy_lc_strip).grad_mag.reshape(n, 1, he, we, strip_samples)
	mag_cr = res.data.grid_sample_px(xy_px=xy_cr_strip).grad_mag.reshape(n, 1, he, we, strip_samples)

	# Validity mask for the strip samples (not just endpoints).
	mask_lc_strip = fit_model.xy_img_mask(res=res, xy=xy_lc_strip, loss_name="gradmag").reshape(n, he, we, strip_samples).unsqueeze(1)
	mask_cr_strip = fit_model.xy_img_mask(res=res, xy=xy_cr_strip, loss_name="gradmag").reshape(n, he, we, strip_samples).unsqueeze(1)
	mask_lc_strip = mask_lc_strip.amin(dim=-1)
	mask_cr_strip = mask_cr_strip.amin(dim=-1)

	# (dbg dump removed)

	eps = 1e-12
	len_lc = torch.sqrt(((center - left) * (center - left)).sum(dim=-1) + eps) * float(res.data.downscale)
	len_cr = torch.sqrt(((right - center) * (right - center)).sum(dim=-1) + eps) * float(res.data.downscale)
	ds_lc = (len_lc / float(strip_samples)).unsqueeze(1).unsqueeze(-1)
	ds_cr = (len_cr / float(strip_samples)).unsqueeze(1).unsqueeze(-1)

	mag_lc_n = mag_lc * ds_lc * strip_samples
	mag_cr_n = mag_cr * ds_cr * strip_samples
	# print("mag_lc_n mean", float(mag_lc_n.mean().detach().cpu()))
	# print("mag_cr_n mean", float(mag_cr_n.mean().detach().cpu()))

	lm_lc = (mag_lc_n - 1.0) * (mag_lc_n - 1.0)
	lm_cr = (mag_cr_n - 1.0) * (mag_cr_n - 1.0)
	lm = 0.5 * (lm_lc.mean(dim=-1) + lm_cr.mean(dim=-1))

	# print("xy_conn", res.xy_conn.shape)
	# print("lm", lm.shape)

	mc_lr = res.mask_conn
	mc_lc_lr = mc_lr[..., 0:2].permute(0, 1, 4, 2, 3).reshape(n, 2, hm, wm).contiguous()
	mc_cr_lr = mc_lr[..., 1:3].permute(0, 1, 4, 2, 3).reshape(n, 2, hm, wm).contiguous()
	mc_lc_hr = F.interpolate(mc_lc_lr, size=(he, we), mode="nearest").reshape(n, 1, 2, he, we).permute(0, 1, 3, 4, 2).contiguous()
	mc_cr_hr = F.interpolate(mc_cr_lr, size=(he, we), mode="nearest").reshape(n, 1, 2, he, we).permute(0, 1, 3, 4, 2).contiguous()
	mask_lc = torch.minimum(mc_lc_hr[..., 0], mc_lc_hr[..., 1])
	mask_cr = torch.minimum(mc_cr_hr[..., 0], mc_cr_hr[..., 1])
	mask = torch.minimum(mask_lc, mask_cr)
	mask = torch.minimum(mask, mask_lc_strip)
	mask = torch.minimum(mask, mask_cr_strip)
	return lm, mask


def _masked_mean(lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	wsum = mask.sum()
	if float(wsum) > 0.0:
		return (lm * mask).sum() / wsum
	return lm.mean()


def gradmag_period_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	lm, mask = gradmag_period_loss_map(res=res)
	return _masked_mean(lm, mask), (lm,), (mask,)
