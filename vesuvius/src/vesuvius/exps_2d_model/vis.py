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
	mask_lr: torch.Tensor | None,
	mask_conn: torch.Tensor | None,
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
		x_pix = xy_lr_cpu[0, :, :, 0].numpy() * float(sf)
		y_pix = xy_lr_cpu[0, :, :, 1].numpy() * float(sf)

		gh, gw = x_pix.shape

		def in_bounds(px: int, py: int) -> bool:
			return 0 <= px < w_vis and 0 <= py < h_vis

		mask_lr_cpu = None
		if mask_lr is not None:
			mask_lr_cpu = mask_lr.to(dtype=torch.float32, device="cpu")

		for iy in range(gh):
			for ix in range(gw):
				px = int(round(float(x_pix[iy, ix])))
				py = int(round(float(y_pix[iy, ix])))
				if in_bounds(px, py):
					ok = True
					if mask_lr_cpu is not None:
						ok = float(mask_lr_cpu[0, 0, iy, ix]) > 0.0
					col = (0, 255, 0) if ok else (128, 128, 128)
					cv2.circle(bg, (px, py), 1, col, -1)
					label = f"{int(round(float(xy_lr_cpu[0, iy, ix, 0])))}:{int(round(float(xy_lr_cpu[0, iy, ix, 1])))}"
					cv2.putText(bg, label, (px + 2, py + 2), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)

		for iy in range(gh):
			for ix in range(gw - 1):
				x0 = int(round(float(x_pix[iy, ix])))
				y0 = int(round(float(y_pix[iy, ix])))
				x1 = int(round(float(x_pix[iy, ix + 1])))
				y1 = int(round(float(y_pix[iy, ix + 1])))
				if in_bounds(x0, y0) and in_bounds(x1, y1):
					ok = True
					if mask_lr_cpu is not None:
						ok = float(mask_lr_cpu[0, 0, iy, ix]) > 0.0 and float(mask_lr_cpu[0, 0, iy, ix + 1]) > 0.0
					col = (0, 0, 255) if ok else (128, 128, 128)
					cv2.line(bg, (x0, y0), (x1, y1), col, 1)

		mask_conn_cpu = None
		if mask_conn is not None:
			mask_conn_cpu = mask_conn.to(dtype=torch.float32, device="cpu")

		if xy_conn is not None:
			xy_conn_cpu = xy_conn.to(dtype=torch.float32, device="cpu")
			xc = xy_conn_cpu[0, :, :, 2, 0].numpy() * float(sf)
			yc = xy_conn_cpu[0, :, :, 2, 1].numpy() * float(sf)
			xc_l = xy_conn_cpu[0, :, :, 0, 0].numpy() * float(sf)
			yc_l = xy_conn_cpu[0, :, :, 0, 1].numpy() * float(sf)
			for iy in range(gh):
				for ix in range(gw - 1):
					x0_lr = int(round(float(x_pix[iy, ix])))
					y0_lr = int(round(float(y_pix[iy, ix])))
					x1_lr = int(round(float(xc[iy, ix])))
					y1_lr = int(round(float(yc[iy, ix])))
					if in_bounds(x0_lr, y0_lr) and in_bounds(x1_lr, y1_lr):
						ok = True
						if mask_conn_cpu is not None:
							ok = float(mask_conn_cpu[0, 0, iy, ix, 1]) > 0.0 and float(mask_conn_cpu[0, 0, iy, ix, 2]) > 0.0
						col = (255, 255, 0) if ok else (128, 128, 128)
						cv2.line(bg, (x0_lr, y0_lr), (x1_lr, y1_lr), col, 1)

					x0_rl = int(round(float(x_pix[iy, ix + 1])))
					y0_rl = int(round(float(y_pix[iy, ix + 1])))
					x1_rl = int(round(float(xc_l[iy, ix + 1])))
					y1_rl = int(round(float(yc_l[iy, ix + 1])))
					if in_bounds(x0_rl, y0_rl) and in_bounds(x1_rl, y1_rl):
						ok = True
						if mask_conn_cpu is not None:
							ok = float(mask_conn_cpu[0, 0, iy, ix + 1, 1]) > 0.0 and float(mask_conn_cpu[0, 0, iy, ix + 1, 0]) > 0.0
						col = (0, 255, 255) if ok else (128, 128, 128)
						cv2.line(bg, (x0_rl, y0_rl), (x1_rl, y1_rl), col, 1)

		for iy in range(gh - 1):
			for ix in range(gw):
				x0 = int(round(float(x_pix[iy, ix])))
				y0 = int(round(float(y_pix[iy, ix])))
				x1 = int(round(float(x_pix[iy + 1, ix])))
				y1 = int(round(float(y_pix[iy + 1, ix])))
				if in_bounds(x0, y0) and in_bounds(x1, y1):
					ok = True
					if mask_lr_cpu is not None:
						ok = float(mask_lr_cpu[0, 0, iy, ix]) > 0.0 and float(mask_lr_cpu[0, 0, iy + 1, ix]) > 0.0
					col = (255, 0, 0) if ok else (128, 128, 128)
					cv2.line(bg, (x0, y0), (x1, y1), col, 1)

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
		mask_lr=res.mask_lr,
		mask_conn=res.mask_conn,
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
