from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
import cv2

import fit_data
import opt_loss_dir
import opt_loss_step


def _to_uint8(arr: "np.ndarray") -> "np.ndarray":
	if arr.dtype == np.uint8:
		return arr
	arr_f = arr.astype("float32")
	vmin = float(arr_f.min())
	vmax = float(arr_f.max())
	if vmax > vmin:
		norm = (arr_f - vmin) / (vmax - vmin)
	else:
		norm = np.zeros_like(arr_f, dtype="float32")
	return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")


def _draw_grid_vis(
	*,
	scale: int,
	h_img: int,
	w_img: int,
	background: torch.Tensor | None,
	xy_lr: torch.Tensor,
	xy_conn: torch.Tensor | None,
) -> "np.ndarray":
	with torch.no_grad():
		sf = max(1, int(scale))
		h_vis = int(h_img * sf)
		w_vis = int(w_img * sf)

		bg = np.zeros((h_vis, w_vis, 3), dtype="uint8")
		if background is not None:
			img_np = background[0, 0].detach().cpu().numpy()
			img_u8 = _to_uint8(img_np)
			w_im = max(1, w_vis // 2)
			h_im = max(1, h_vis // 2)
			img_resized = cv2.resize(img_u8, (w_im, h_im), interpolation=cv2.INTER_LINEAR)
			img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
			x0 = (w_vis - w_im) // 2
			y0 = (h_vis - h_im) // 2
			bg[y0:y0 + h_im, x0:x0 + w_im, :] = img_resized

		xy_lr_cpu = xy_lr.to(dtype=torch.float32, device="cpu")
		x_norm = xy_lr_cpu[0, 0].numpy()
		y_norm = xy_lr_cpu[0, 1].numpy()

		x_pix = (0.5 + 0.25 * x_norm) * float(w_vis - 1)
		y_pix = (0.5 + 0.25 * y_norm) * float(h_vis - 1)

		gh, gw = x_pix.shape

		def in_bounds(px: int, py: int) -> bool:
			return 0 <= px < w_vis and 0 <= py < h_vis

		for iy in range(gh):
			for ix in range(gw):
				px = int(round(float(x_pix[iy, ix])))
				py = int(round(float(y_pix[iy, ix])))
				if in_bounds(px, py):
					cv2.circle(bg, (px, py), 1, (0, 255, 0), -1)

		for iy in range(gh):
			for ix in range(gw - 1):
				x0 = int(round(float(x_pix[iy, ix])))
				y0 = int(round(float(y_pix[iy, ix])))
				x1 = int(round(float(x_pix[iy, ix + 1])))
				y1 = int(round(float(y_pix[iy, ix + 1])))
				if in_bounds(x0, y0) and in_bounds(x1, y1):
					cv2.line(bg, (x0, y0), (x1, y1), (0, 0, 255), 1)

		if xy_conn is not None:
			xy_conn_cpu = xy_conn.to(dtype=torch.float32, device="cpu")
			w = float(max(1, int(w_img) - 1))
			h = float(max(1, int(h_img) - 1))
			xn = (xy_conn_cpu[0, :, 0] / w) * 2.0 - 1.0
			yn = (xy_conn_cpu[0, :, 1] / h) * 2.0 - 1.0
			xc = (0.5 + 0.25 * xn.numpy()) * float(w_vis - 1)
			yc = (0.5 + 0.25 * yn.numpy()) * float(h_vis - 1)
			for iy in range(gh):
				for ix in range(gw - 1):
					x0_lr = int(round(float(x_pix[iy, ix])))
					y0_lr = int(round(float(y_pix[iy, ix])))
					x1_lr = int(round(float(xc[2, iy, ix])))
					y1_lr = int(round(float(yc[2, iy, ix])))
					if in_bounds(x0_lr, y0_lr) and in_bounds(x1_lr, y1_lr):
						cv2.line(bg, (x0_lr, y0_lr), (x1_lr, y1_lr), (255, 255, 0), 1)

					x0_rl = int(round(float(x_pix[iy, ix + 1])))
					y0_rl = int(round(float(y_pix[iy, ix + 1])))
					x1_rl = int(round(float(xc[0, iy, ix + 1])))
					y1_rl = int(round(float(yc[0, iy, ix + 1])))
					if in_bounds(x0_rl, y0_rl) and in_bounds(x1_rl, y1_rl):
						cv2.line(bg, (x0_rl, y0_rl), (x1_rl, y1_rl), (0, 255, 255), 1)

		for iy in range(gh - 1):
			for ix in range(gw):
				x0 = int(round(float(x_pix[iy, ix])))
				y0 = int(round(float(y_pix[iy, ix])))
				x1 = int(round(float(x_pix[iy + 1, ix])))
				y1 = int(round(float(y_pix[iy + 1, ix])))
				if in_bounds(x0, y0) and in_bounds(x1, y1):
					cv2.line(bg, (x0, y0), (x1, y1), (255, 0, 0), 1)

		return bg


def save(
	*,
	model,
	data: fit_data.FitData,
	postfix: str,
	out_dir: str,
	scale: int,
) -> None:
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)

	h_img, w_img = data.size
	res = model(data)
	grid_xy = res.xy_lr

	grid_vis = _draw_grid_vis(
		scale=scale,
		h_img=h_img,
		w_img=w_img,
		background=data.cos,
		xy_lr=grid_xy,
		xy_conn=res.xy_conn,
	)
	grid_path = out / f"res_grid_{postfix}.jpg"
	cv2.imwrite(str(grid_path), np.flip(grid_vis, -1))

	loss_maps = {
		"dir": {
			"fn": lambda: opt_loss_dir.direction_loss_map(res=res)[0],
			"suffix": "dir",
			"reduce": True,
		},
		"dir_v": {
			"fn": lambda: opt_loss_dir.direction_loss_maps(res=res)[0],
			"suffix": "dir_v",
			"reduce": True,
		},
		"dir_conn": {
			"fn": lambda: opt_loss_dir.direction_loss_maps(res=res)[1],
			"suffix": "dir_conn",
			"reduce": True,
		},
		"step_h": {
			"fn": lambda: opt_loss_step.step_loss_maps(res=res)[0],
			"suffix": "step_h",
			"reduce": True,
		},
		"step_v": {
			"fn": lambda: opt_loss_step.step_loss_maps(res=res)[1],
			"suffix": "step_v",
			"reduce": True,
		},
	}
	for _k, spec in loss_maps.items():
		m = spec["fn"]().detach().cpu()
		if bool(spec["reduce"]) and m.ndim == 4:
			m = m[0, 0]
		out_path = out / f"res_loss_{spec['suffix']}_{postfix}.tif"
		tifffile.imwrite(str(out_path), m.numpy().astype("float32"), compression="lzw")

	tgt = data.cos[0, 0].detach().cpu().numpy()
	tgt_t = model.target_cos()
	tgt = tgt_t[0, 0].detach().cpu().numpy()
	tgt_path = out / f"res_tgt_{postfix}.tif"
	tifffile.imwrite(str(tgt_path), tgt.astype("float32"), compression="lzw")
