from __future__ import annotations

import torch

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

	mag_use = mag_hr
	if mask_sample is not None and mask_sample.shape[-2:] == mag_hr.shape[-2:]:
		mag_use = mag_use * mask_sample

	ww = int(mag_hr.shape[-1])
	samples_per = ww // periods_int
	if samples_per <= 0:
		base = torch.zeros((), device=mag_hr.device, dtype=mag_hr.dtype)
		mask0 = torch.zeros((), device=mag_hr.device, dtype=mag_hr.dtype)
		return base, mask0
	max_cols = samples_per * periods_int

	weighted = (mag_use[:, :, :, :max_cols] * dist_x_hr[:, :, :, :max_cols]).view(
		int(mag_hr.shape[0]),
		1,
		int(mag_hr.shape[-2]),
		periods_int,
		samples_per,
	)
	sum_period = weighted.sum(dim=-1)
	lm = (sum_period - 1.0) * (sum_period - 1.0)

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
