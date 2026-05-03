from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dense_batch_flow
import model as fit_model
from opt_loss_dir import _vertex_normals

_INNER_FACTOR = 0.25  # penalty reduction for points inside the predicted surface

_flow_gate_cfg: dict | None = None
_flow_gate_stage: str = "stage"
_flow_gate_seed_xyz: tuple[float, float, float] | None = None
_flow_gate_out_dir: Path | None = None


def configure_flow_gate(
	*,
	cfg: dict | None,
	stage_name: str,
	seed_xyz: tuple[float, float, float] | None,
	out_dir: str | None,
) -> None:
	global _flow_gate_cfg, _flow_gate_stage, _flow_gate_seed_xyz, _flow_gate_out_dir
	if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
		_flow_gate_cfg = None
		_flow_gate_stage = str(stage_name)
		_flow_gate_seed_xyz = seed_xyz
		_flow_gate_out_dir = Path(out_dir) if out_dir else None
		return
	_flow_gate_cfg = dict(cfg)
	_flow_gate_stage = str(stage_name)
	_flow_gate_seed_xyz = seed_xyz
	_flow_gate_out_dir = Path(out_dir) if out_dir else None


def _write_flow_gate_debug(
	*,
	stage_name: str,
	pred_u8: np.ndarray,
	weight_hr: np.ndarray,
	out_dir: Path | None,
	threshold: float,
) -> None:
	if out_dir is None:
		return
	try:
		import tifffile
	except Exception as exc:
		print(f"[pred_dt_flow_gate] debug write skipped: tifffile import failed: {exc}", flush=True)
		return
	out_dir.mkdir(parents=True, exist_ok=True)
	thresholded = (pred_u8 > threshold).astype(np.float32) * 255.0
	weights = np.asarray(weight_hr, dtype=np.float32)
	with tifffile.TiffWriter(str(out_dir / f"pred_dt_flow_gate_{stage_name}_layers.tif")) as tw:
		tw.write(thresholded, photometric="minisblack", metadata={"Name": "thresholded_pred_dt"})
		tw.write(weights, photometric="minisblack", metadata={"Name": "flow_gate_weight"})
	tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{stage_name}_thresholded.tif"), thresholded)
	tifffile.imwrite(str(out_dir / f"pred_dt_flow_gate_{stage_name}_weight.tif"), weights)


def _flow_gate_weight(res: fit_model.FitResult3D) -> torch.Tensor | None:
	cfg = _flow_gate_cfg
	if cfg is None:
		return None
	if res.xyz_lr.shape[0] != 1:
		raise RuntimeError("pred_dt_flow_gate currently supports only single-winding models (D == 1)")
	if _flow_gate_seed_xyz is None:
		raise RuntimeError("pred_dt_flow_gate requires a fit seed so it can project the flow source")
	if res.data_s.pred_dt is None:
		raise RuntimeError("pred_dt_flow_gate requires pred_dt to be loaded")

	threshold = 127.0
	flow_zero = float(cfg.get("flow_zero", 50.0))
	flow_one = float(cfg.get("flow_one", 300.0))
	if flow_one <= flow_zero:
		raise ValueError("pred_dt_flow_gate requires flow_one > flow_zero")
	debug = bool(cfg.get("debug", True))

	with torch.no_grad():
		pred_hr = res.data_s.pred_dt.squeeze(0).squeeze(0)
		if pred_hr.ndim != 3 or pred_hr.shape[0] != 1:
			raise RuntimeError(f"pred_dt_flow_gate expected pred_dt render shape (1,H,W), got {tuple(pred_hr.shape)}")
		pred_img = pred_hr[0].detach().clamp(0, 255).round().to(torch.uint8).cpu().numpy()
		He, We = pred_img.shape

		xyz_hr = res.xyz_hr[0].detach()
		seed = torch.tensor(_flow_gate_seed_xyz, device=xyz_hr.device, dtype=xyz_hr.dtype)
		dist2 = ((xyz_hr - seed.view(1, 1, 3)) ** 2).sum(dim=-1)
		source_flat = int(torch.argmin(dist2).detach().cpu())
		source_y = source_flat // We
		source_x = source_flat % We

		Hm = int(res.xyz_lr.shape[1])
		Wm = int(res.xyz_lr.shape[2])
		sub_h = int(res.params.subsample_mesh)
		sub_w = int(res.params.subsample_winding)
		yy, xx = np.meshgrid(
			np.arange(Hm, dtype=np.float32) * float(sub_h),
			np.arange(Wm, dtype=np.float32) * float(sub_w),
			indexing="ij",
		)
		query_xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)

		print(
			f"[pred_dt_flow_gate] {_flow_gate_stage}: render={We}x{He} "
			f"grid={Wm}x{Hm} source=({source_x},{source_y})",
			flush=True,
		)
		query_flow, dense_flow = dense_batch_flow.compute_flow_grid(
			pred_img,
			source_xy=(source_x, source_y),
			query_xy=query_xy,
			verbose=debug,
		)
		flow_lr = torch.as_tensor(
			query_flow.reshape(Hm, Wm),
			device=res.xyz_lr.device,
			dtype=torch.float32,
		).view(1, 1, Hm, Wm)
		weight = ((flow_lr - flow_zero) / (flow_one - flow_zero)).clamp(0.0, 1.0)

		if debug:
			weight_hr = F.interpolate(
				weight,
				size=(He, We),
				mode="bilinear",
				align_corners=True,
			)[0, 0].detach().cpu().numpy().astype(np.float32)
			_write_flow_gate_debug(
				stage_name=_flow_gate_stage,
				pred_u8=pred_img,
				weight_hr=weight_hr,
				out_dir=_flow_gate_out_dir,
				threshold=threshold,
			)
			print(
				f"[pred_dt_flow_gate] {_flow_gate_stage}: "
				f"flow_min={float(np.min(query_flow)):.3f} "
				f"flow_max={float(np.max(query_flow)):.3f} "
				f"weight_mean={float(weight.mean().detach().cpu()):.4f}",
				flush=True,
			)
		return weight


def pred_dt_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Pred-DT loss: two clamped L1 terms pushing mesh into the prediction.

	Encoding: outside=[80,127], inside=[128,175], boundary at 127.5.
	lm_out = clamp(127 - raw, min=0)      — active outside, zero inside
	lm_in  = clamp(255 - raw, max=127)    — active inside, constant (no grad) outside
	lm = lm_out + 0.25 * lm_in
	"""
	# Project gradients onto surface normal only (prevents tangential crimping)
	n = _vertex_normals(res.xyz_lr.detach())
	proj_len = (res.xyz_lr * n).sum(dim=-1, keepdim=True)
	xyz_normal = proj_len * n
	xyz_tangential = res.xyz_lr - xyz_normal
	xyz = xyz_normal + xyz_tangential.detach()

	# Sample pred_dt using common sampling with per-channel spacing and diff gradients
	sampled = res.data.grid_sample_fullres(xyz, diff=True)
	sampled_raw = sampled.pred_dt.squeeze(0).permute(1, 0, 2, 3)  # (D, 1, Hm, Wm)

	lm_out = (127.0 - sampled_raw).clamp(min=0)      # outside: 1–47, inside: 0 (no grad)
	lm_in = (255.0 - sampled_raw).clamp(max=127.0)    # inside: 80–127, outside: 127 (constant, no grad)
	lm = lm_out + _INNER_FACTOR * lm_in

	mask = res.mask_lr
	flow_gate = _flow_gate_weight(res)
	wsum = mask.sum()
	if float(wsum) > 0.0:
		if flow_gate is not None:
			loss = (lm * mask * flow_gate).sum() / wsum
		else:
			loss = (lm * mask).sum() / wsum
	else:
		loss = lm.mean()
	if flow_gate is not None:
		return loss, (lm,), (mask * flow_gate,)
	return loss, (lm,), (mask,)
