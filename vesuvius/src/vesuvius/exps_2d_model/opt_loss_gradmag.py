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

	- Uses `res.data_s.grad_mag` sampled at `res.xy_hr`.
	- Groups the x-axis into `periods = Wm-1` periods (base-mesh winding intervals).
	- For each (row,period), enforces that the distance-weighted sum of magnitudes is ~1.

	Outputs:
	- lm: (N,1,He,periods)
	- mask: (N,1,He,periods) from MIN mask over each period interval.
	"""
	mag_hr = res.data_s.grad_mag
	mask_sample = res.mask_hr
	m = float(res.data_s.downscale)
	periods_int = max(1, int(res.xy_lr.shape[2]) - 1)
	if int(mag_hr.shape[-1]) < periods_int:
		base = torch.zeros((), device=mag_hr.device, dtype=mag_hr.dtype)
		mask0 = torch.zeros((), device=mag_hr.device, dtype=mag_hr.dtype)
		return base, mask0

	dist_x_hr = _grid_segment_length_x_px(xy_px=res.xy_hr) * m
	if dist_x_hr.shape[-2:] != mag_hr.shape[-2:]:
		raise RuntimeError("dist_x_hr must match mag_hr spatial shape")

	ww = int(mag_hr.shape[-1])
	samples_per = ww // periods_int
	if samples_per <= 0:
		base = torch.zeros((), device=mag_hr.device, dtype=mag_hr.dtype)
		mask0 = torch.zeros((), device=mag_hr.device, dtype=mag_hr.dtype)
		return base, mask0
	max_cols = samples_per * periods_int

	mag_use = mag_hr
	if mask_sample is not None and mask_sample.shape[-2:] == mag_hr.shape[-2:]:
		mag_use = mag_use * mask_sample

	# Apply direction sign: use LR mesh geometry to determine winding orientation.
	#
	# We compute a single sign per (row,period) (where period is one LR x-segment),
	# and apply it uniformly to all HR samples belonging to that period.
	xy_lr = res.xy_lr
	if int(xy_lr.shape[1]) >= 2 and int(xy_lr.shape[2]) >= 2:
		md = xy_lr.new_zeros(xy_lr.shape)
		if int(xy_lr.shape[1]) >= 3:
			md[:, 1:-1, :, :] = xy_lr[:, 2:, :, :] - xy_lr[:, :-2, :, :]
			md[:, 0, :, :] = xy_lr[:, 1, :, :] - xy_lr[:, 0, :, :]
			md[:, -1, :, :] = xy_lr[:, -1, :, :] - xy_lr[:, -2, :, :]
		else:
			md[:, 0, :, :] = xy_lr[:, 1, :, :] - xy_lr[:, 0, :, :]
			md[:, 1, :, :] = md[:, 0, :, :]

		xyc = res.xy_conn
		left = xyc[..., 0, :]
		mid = xyc[..., 1, :]
		right = xyc[..., 2, :]
		v_l = left - mid
		v_r = right - mid

		mdx = md[..., 0]
		mdy = md[..., 1]
		cross_r = mdx * v_r[..., 1] - mdy * v_r[..., 0]
		cross_l = mdx * v_l[..., 1] - mdy * v_l[..., 0]
		ok = (cross_r > 0.0) & (cross_l < 0.0)

		# period sign from the right vertex of each LR segment (col 1..Wm-1)
		ok_period_lr = ok[:, :, 1:]
		one = torch.ones_like(ok_period_lr, dtype=mag_hr.dtype)
		sign_period_lr = torch.where(ok_period_lr, -one, one)
		w_period_lr = torch.where(ok_period_lr, one, 10.0 * one)
		sign_period_lr = sign_period_lr.unsqueeze(1)
		w_period_lr = w_period_lr.unsqueeze(1)
		sign_period_hr = F.interpolate(sign_period_lr, size=(int(mag_hr.shape[-2]), int(sign_period_lr.shape[-1])), mode="nearest")
		w_period_hr = F.interpolate(w_period_lr, size=(int(mag_hr.shape[-2]), int(w_period_lr.shape[-1])), mode="nearest")

		# expand periods to HR columns
		sign_hr = sign_period_hr.repeat_interleave(samples_per, dim=-1)
		w_hr = w_period_hr.repeat_interleave(samples_per, dim=-1)
		if int(sign_hr.shape[-1]) < int(mag_hr.shape[-1]):
			pad = int(mag_hr.shape[-1]) - int(sign_hr.shape[-1])
			sign_hr = torch.cat([sign_hr, sign_hr[..., -1:].expand(*sign_hr.shape[:-1], pad)], dim=-1)
			w_hr = torch.cat([w_hr, w_hr[..., -1:].expand(*w_hr.shape[:-1], pad)], dim=-1)
		elif int(sign_hr.shape[-1]) > int(mag_hr.shape[-1]):
			sign_hr = sign_hr[..., : int(mag_hr.shape[-1])]
			w_hr = w_hr[..., : int(mag_hr.shape[-1])]
		mag_use = mag_use * sign_hr

	weighted = (mag_use[:, :, :, :max_cols] * dist_x_hr[:, :, :, :max_cols]).view(
		int(mag_hr.shape[0]),
		1,
		int(mag_hr.shape[-2]),
		periods_int,
		samples_per,
	)
	sum_period = weighted.sum(dim=-1)
	lm = (sum_period - 1.0) * (sum_period - 1.0)
	if int(xy_lr.shape[1]) >= 2 and int(xy_lr.shape[2]) >= 2:
		w_period = w_period_hr
		if w_period.shape[-2:] != lm.shape[-2:]:
			w_period = w_period.expand_as(lm)
		lm = lm * w_period

	if mask_sample is None or mask_sample.shape[-2:] != mag_hr.shape[-2:]:
		mask = torch.ones_like(lm)
		return lm, mask

	mask_reshaped = mask_sample[:, :, :, :max_cols].view(
		int(mag_hr.shape[0]),
		1,
		int(mag_hr.shape[-2]),
		periods_int,
		samples_per,
	)
	mask, _ = mask_reshaped.min(dim=-1)
	return lm, mask


def _masked_mean(lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	wsum = mask.sum()
	if float(wsum) > 0.0:
		return (lm * mask).sum() / wsum
	return lm.mean()


def gradmag_period_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	lm, mask = gradmag_period_loss_map(res=res)
	return _masked_mean(lm, mask)
