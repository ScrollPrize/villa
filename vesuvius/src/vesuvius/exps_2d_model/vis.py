from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
import cv2

import fit_data
import opt_loss_dir
import opt_loss_geom
import opt_loss_gradmag
import opt_loss_step
import inv_map


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


def _loss_concat_vis(
	*,
	loss_maps_2d: list[tuple[str, np.ndarray]],
	border_px: int,
	label_px: int,
) -> np.ndarray:
	if len(loss_maps_2d) == 0:
		return np.zeros((1, 1), dtype="float32")

	border = int(max(0, border_px))
	label_h = int(max(0, label_px))
	label_h = max(1, label_h // 2) if label_h > 0 else 0
	max_h = 1
	out_w = 0
	for _name, arr in loss_maps_2d:
		max_h = max(max_h, int(arr.shape[0]))
		out_w += int(arr.shape[1])
	if len(loss_maps_2d) > 1:
		out_w += border * (len(loss_maps_2d) - 1)
	max_h = int(max_h)
	out = np.full((label_h + max_h, int(out_w)), 0.5, dtype="float32")

	x = 0
	for i, (_name, arr) in enumerate(loss_maps_2d):
		p = arr.astype("float32", copy=False)
		h, w = int(p.shape[0]), int(p.shape[1])
		out[label_h:label_h + h, x:x + w] = p
		x += w
		if border > 0 and i + 1 < len(loss_maps_2d):
			out[:, x:x + border] = 0.5
			x += border

	if label_h > 0:
		s = 8
		lab_u8 = np.full((label_h * s, int(out.shape[1]) * s), 128, dtype="uint8")
		x = 0
		for i, (name, m) in enumerate(loss_maps_2d):
			y0 = int(2 * s)
			if (i % 2) == 1:
				y0 = int(6 * s)
			cv2.putText(
				lab_u8,
				str(name),
				(int(x * s), y0),
				cv2.FONT_HERSHEY_PLAIN,
				1.0,
				0,
				1,
				lineType=cv2.LINE_8,
			)
			x += int(m.shape[1])
			if border > 0 and i + 1 < len(loss_maps_2d):
				x += border
		lab_f = (lab_u8.astype("float32") / 255.0)
		lab_f = cv2.resize(lab_f, (int(out.shape[1]), label_h), interpolation=cv2.INTER_AREA)
		out[0:label_h, :] = lab_f
	return out


def _loss_concat_vis_u8(
	*,
	loss_maps_2d: list[tuple[str, np.ndarray]],
	border_px: int,
	label_px: int,
	scale: int,
) -> np.ndarray:
	if len(loss_maps_2d) == 0:
		return np.zeros((1, 1, 3), dtype="uint8")

	border = int(max(0, border_px))
	label_h = int(max(0, label_px))
	s = int(max(1, scale))

	# Build float mosaic first (this is the canonical layout), then convert to u8 and label.
	concat_f = _loss_concat_vis(loss_maps_2d=loss_maps_2d, border_px=border, label_px=label_h)
	u8 = (np.clip(concat_f, 0.0, 1.0) * 255.0).astype("uint8")
	out = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
	if s != 1:
		out = cv2.resize(out, (int(out.shape[1]) * s, int(out.shape[0]) * s), interpolation=cv2.INTER_NEAREST)

	if label_h <= 0:
		return out

	x = 0
	for i, (name, m) in enumerate(loss_maps_2d):
		y0 = int(2 * s)
		if (i % 2) == 1:
			y0 = int(6 * s)
		cv2.putText(
			out,
			str(name),
			(int(x * s), y0),
			cv2.FONT_HERSHEY_PLAIN,
			1.0,
			(0, 0, 0),
			1,
			lineType=cv2.LINE_8,
		)
		x += int(m.shape[1])
		if border > 0 and i + 1 < len(loss_maps_2d):
			x += border

	return out


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
		x_off = 0
		y_off = 0
		w_im = w_vis
		h_im = h_vis
		if background is not None:
			img_np = background[0, 0].detach().cpu().numpy()
			img_u8 = _to_uint8(img_np)
			w_im = max(1, w_vis // 2)
			h_im = max(1, h_vis // 2)
			img_resized = cv2.resize(img_u8, (w_im, h_im), interpolation=cv2.INTER_LINEAR)
			img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
			x_off = (w_vis - w_im) // 2
			y_off = (h_vis - h_im) // 2
			bg[y_off:y_off + h_im, x_off:x_off + w_im, :] = img_resized

		xy_lr_cpu = xy_lr.to(dtype=torch.float32, device="cpu")
		wx = float(max(1, int(w_img) - 1))
		hy = float(max(1, int(h_img) - 1))
		sx = float(max(1, int(w_im) - 1)) / wx
		sy = float(max(1, int(h_im) - 1)) / hy
		x_pix = x_off + xy_lr_cpu[0, :, :, 0].numpy() * sx
		y_pix = y_off + xy_lr_cpu[0, :, :, 1].numpy() * sy

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
					# label = f"{int(round(float(xy_lr_cpu[0, iy, ix, 0])))}:{int(round(float(xy_lr_cpu[0, iy, ix, 1])))}"
					# cv2.putText(bg, label, (px + 2, py + 2), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)

		for iy in range(gh):
			for ix in range(gw - 1):
				xp0 = int(round(float(x_pix[iy, ix])))
				yp0 = int(round(float(y_pix[iy, ix])))
				xp1 = int(round(float(x_pix[iy, ix + 1])))
				yp1 = int(round(float(y_pix[iy, ix + 1])))
				if in_bounds(xp0, yp0) and in_bounds(xp1, yp1):
					ok = True
					if mask_lr_cpu is not None:
						ok = float(mask_lr_cpu[0, 0, iy, ix]) > 0.0 and float(mask_lr_cpu[0, 0, iy, ix + 1]) > 0.0
					col = (0, 0, 255) if ok else (128, 128, 128)
					cv2.line(bg, (xp0, yp0), (xp1, yp1), col, 1)

		mask_conn_cpu = None
		if mask_conn is not None:
			mask_conn_cpu = mask_conn.to(dtype=torch.float32, device="cpu")

		if xy_conn is not None:
			xy_conn_cpu = xy_conn.to(dtype=torch.float32, device="cpu")
			# Draw both conn segments per mesh vertex (left-mid & mid-right).
			x_l = x_off + xy_conn_cpu[0, :, :, 0, 0].numpy() * sx
			y_l = y_off + xy_conn_cpu[0, :, :, 0, 1].numpy() * sy
			x_m = x_off + xy_conn_cpu[0, :, :, 1, 0].numpy() * sx
			y_m = y_off + xy_conn_cpu[0, :, :, 1, 1].numpy() * sy
			x_r = x_off + xy_conn_cpu[0, :, :, 2, 0].numpy() * sx
			y_r = y_off + xy_conn_cpu[0, :, :, 2, 1].numpy() * sy
			for iy in range(gh):
				for ix in range(gw):
					xm = int(round(float(x_m[iy, ix])))
					ym = int(round(float(y_m[iy, ix])))
					xl = int(round(float(x_l[iy, ix])))
					yl = int(round(float(y_l[iy, ix])))
					xr = int(round(float(x_r[iy, ix])))
					yr = int(round(float(y_r[iy, ix])))
					if in_bounds(xm, ym) and in_bounds(xl, yl):
						ok = True
						if mask_conn_cpu is not None:
							ok = float(mask_conn_cpu[0, 0, iy, ix, 0]) > 0.0 and float(mask_conn_cpu[0, 0, iy, ix, 1]) > 0.0
						col = (0, 255, 255) if ok else (128, 128, 128)
						cv2.line(bg, (xm, ym), (xl, yl), col, 1)
					if in_bounds(xm, ym) and in_bounds(xr, yr):
						ok = True
						if mask_conn_cpu is not None:
							ok = float(mask_conn_cpu[0, 0, iy, ix, 1]) > 0.0 and float(mask_conn_cpu[0, 0, iy, ix, 2]) > 0.0
						col = (255, 255, 0) if ok else (128, 128, 128)
						cv2.line(bg, (xm, ym), (xr, yr), col, 1)

		for iy in range(gh - 1):
			for ix in range(gw):
				xp0 = int(round(float(x_pix[iy, ix])))
				yp0 = int(round(float(y_pix[iy, ix])))
				xp1 = int(round(float(x_pix[iy + 1, ix])))
				yp1 = int(round(float(y_pix[iy + 1, ix])))
				if in_bounds(xp0, yp0) and in_bounds(xp1, yp1):
					ok = True
					if mask_lr_cpu is not None:
						ok = float(mask_lr_cpu[0, 0, iy, ix]) > 0.0 and float(mask_lr_cpu[0, 0, iy + 1, ix]) > 0.0
					col = (255, 0, 0) if ok else (128, 128, 128)
					cv2.line(bg, (xp0, yp0), (xp1, yp1), col, 1)

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
	out_grids = out / "grids"
	out_loss = out / "loss_maps"
	out_tgt = out / "targets"
	out_vis = out / "vis"
	out_grids.mkdir(parents=True, exist_ok=True)
	out_loss.mkdir(parents=True, exist_ok=True)
	out_tgt.mkdir(parents=True, exist_ok=True)
	out_vis.mkdir(parents=True, exist_ok=True)

	h_img, w_img = data.size
	res = model(data)
	# with torch.no_grad():
	# 	xy = res.xy_conn[0].detach().cpu().numpy()
	# 	gh, gw = xy.shape[0], xy.shape[1]
	# 	print(f"xy_conn dbg: grid={gh}x{gw}")
	# 	for iy in range(gh):
	# 		for ix in range(gw):
	# 			l = xy[iy, ix, 0]
	# 			p = xy[iy, ix, 1]
	# 			r = xy[iy, ix, 2]
	# 			dl = float(((p[0] - l[0]) ** 2 + (p[1] - l[1]) ** 2) ** 0.5)
	# 			dr = float(((r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2) ** 0.5)
	# 			print(
	# 				f"({iy:02d},{ix:02d}) p=({p[0]:.2f},{p[1]:.2f}) "
	# 				f"l=({l[0]:.2f},{l[1]:.2f}) dl={dl:.2f} "
	# 				f"r=({r[0]:.2f},{r[1]:.2f}) dr={dr:.2f}"
	# 			)
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
	grid_path = out_grids / f"res_grid_{postfix}.jpg"
	cv2.imwrite(str(grid_path), np.flip(grid_vis, -1))

	uv_img, uv_mask = inv_map.inverse_map_autograd(xy_lr=res.xy_lr, h_out=h_img, w_out=w_img)

	dir_lm_v, _dir_mask_v = opt_loss_dir.dir_v_loss_maps(res=res)
	dir_lm_conn_l, dir_lm_conn_r, dir_mask_conn_l, dir_mask_conn_r = opt_loss_dir.dir_conn_loss_maps(res=res)
	inv_conn_l = (1.0 - dir_mask_conn_l).to(dtype=dir_lm_conn_l.dtype)
	inv_conn_r = (1.0 - dir_mask_conn_r).to(dtype=dir_lm_conn_r.dtype)
	dir_lm_conn_l = dir_lm_conn_l * dir_mask_conn_l + 0.5 * inv_conn_l
	dir_lm_conn_r = dir_lm_conn_r * dir_mask_conn_r + 0.5 * inv_conn_r

	smooth_x_lm, _smooth_x_mask = opt_loss_geom.smooth_x_loss_map(res=res)
	smooth_y_lm, _smooth_y_mask = opt_loss_geom.smooth_y_loss_map(res=res)
	meshoff_sy_lm, _meshoff_sy_mask = opt_loss_geom.meshoff_smooth_y_loss_map(res=res)
	conn_sy_l_lm, _conn_sy_l_mask = opt_loss_geom.conn_y_smooth_l_loss_map(res=res)
	conn_sy_r_lm, _conn_sy_r_mask = opt_loss_geom.conn_y_smooth_r_loss_map(res=res)
	angle_lm, _angle_mask = opt_loss_geom.angle_symmetry_loss_map(res=res)
	y_straight_lm, _y_straight_mask = opt_loss_geom.y_straight_loss_map(res=res)
	gradmag_lm, _gradmag_mask = opt_loss_gradmag.gradmag_period_loss_map(res=res)

	loss_maps = {
		"dir_v": {
			"fn": lambda: dir_lm_v,
			"suffix": "dir_v",
			"reduce": True,
		},
		"dir_conn_l": {
			"fn": lambda: dir_lm_conn_l,
			"suffix": "dir_conn_l",
			"reduce": True,
		},
		"dir_conn_r": {
			"fn": lambda: dir_lm_conn_r,
			"suffix": "dir_conn_r",
			"reduce": True,
		},
		"step_v": {
			"fn": lambda: opt_loss_step.step_loss_maps(res=res),
			"suffix": "step_v",
			"reduce": True,
		},
		"gradmag": {
			"fn": lambda: gradmag_lm,
			"suffix": "gradmag",
			"reduce": True,
		},
		"smooth_x": {
			"fn": lambda: smooth_x_lm,
			"suffix": "smooth_x",
			"reduce": True,
		},
		"smooth_y": {
			"fn": lambda: smooth_y_lm,
			"suffix": "smooth_y",
			"reduce": True,
		},
		"meshoff_sy": {
			"fn": lambda: meshoff_sy_lm,
			"suffix": "meshoff_sy",
			"reduce": True,
		},
		"conn_sy_l": {
			"fn": lambda: conn_sy_l_lm,
			"suffix": "conn_sy_l",
			"reduce": True,
		},
		"conn_sy_r": {
			"fn": lambda: conn_sy_r_lm,
			"suffix": "conn_sy_r",
			"reduce": True,
		},
		"angle": {
			"fn": lambda: angle_lm,
			"suffix": "angle",
			"reduce": True,
		},
		"y_straight": {
			"fn": lambda: y_straight_lm,
			"suffix": "y_straight",
			"reduce": True,
		},
	}
	for _k, spec in loss_maps.items():
		m = spec["fn"]().detach().cpu()
		if bool(spec["reduce"]) and m.ndim == 4:
			m = m[0, 0]
		out_path = out_loss / f"res_loss_{spec['suffix']}_{postfix}.tif"
		tifffile.imwrite(str(out_path), m.numpy().astype("float32"), compression="lzw")

		mt = spec["fn"]().detach()
		if mt.ndim == 4 and int(mt.shape[2]) > 1 and int(mt.shape[3]) > 1:
			im = inv_map.warp_nchw_from_uv(src=mt, uv=uv_img, uv_mask=uv_mask, fill=0.5)
			out_path_img = out_loss / f"res_loss_img_{spec['suffix']}_{postfix}.tif"
			tifffile.imwrite(str(out_path_img), im[0, 0].detach().cpu().numpy().astype("float32"), compression="lzw")

	# One horizontally-concatenated float tif for quick inspection.
	loss_2d: list[tuple[str, np.ndarray]] = []
	gradmag_2d: np.ndarray | None = None
	for _k, spec in loss_maps.items():
		m = spec["fn"]().detach().cpu()
		if m.ndim == 4:
			m2 = m[0, 0].numpy().astype("float32")
		elif m.ndim == 2:
			m2 = m.numpy().astype("float32")
		else:
			continue
		name = str(spec["suffix"])
		if name == "gradmag":
			gradmag_2d = m2
			continue
		loss_2d.append((name, m2))
	if gradmag_2d is not None:
		max_h = 1
		for _name, arr in loss_2d:
			max_h = max(max_h, int(arr.shape[0]))
		gradmag_w = int(gradmag_2d.shape[1])
		loss_2d.insert(2, ("gradmag", np.full((int(max_h), gradmag_w), 0.5, dtype="float32")))
	# Float tif with a top label band (label pixels are visual-only floats in [0,1]).
	concat_f = _loss_concat_vis(loss_maps_2d=loss_2d, border_px=6, label_px=16)
	concat_f = np.repeat(np.repeat(concat_f, 8, axis=0), 8, axis=1)
	label_h = max(1, 16 // 2)
	border = 6
	s = 8

	# Paste gradmag after resize, scaling in height to match the other (full-res) maps.
	if gradmag_2d is not None:
		tgt_h = int(max_h) * s
		x = 0
		for i, (name, m) in enumerate(loss_2d):
			if name == "gradmag":
				w = int(m.shape[1])
				gm = gradmag_2d
				gm_resized = cv2.resize(gm, (int(w) * s, tgt_h), interpolation=cv2.INTER_NEAREST)
				concat_f[label_h * s:label_h * s + tgt_h, int(x) * s:int(x) * s + int(w) * s] = gm_resized.astype("float32", copy=False)
				break
			x += int(m.shape[1])
			if border > 0 and i + 1 < len(loss_2d):
				x += border

	lab_u8 = np.full((label_h * s, int(concat_f.shape[1])), 128, dtype="uint8")
	x = 0
	for i, (name, m) in enumerate(loss_2d):
		y0 = int(2 * s)
		if (i % 2) == 1:
			y0 = int(6 * s)
		cv2.putText(
			lab_u8,
			str(name),
			(int(x * s), y0),
			cv2.FONT_HERSHEY_PLAIN,
			1.0,
			0,
			1,
			lineType=cv2.LINE_8,
		)
		x += int(m.shape[1])
		if border > 0 and i + 1 < len(loss_2d):
			x += border
	concat_f[0:label_h * s, :] = lab_u8.astype("float32") / 255.0
	concat_path = out_vis / f"res_loss_concat_{postfix}.tif"
	tifffile.imwrite(str(concat_path), concat_f.astype("float32"), compression="lzw")

	tgt = data.cos[0, 0].detach().cpu().numpy()
	tgt_t = model.target_cos()
	tgt = tgt_t[0, 0].detach().cpu().numpy()
	tgt_path = out_tgt / f"res_tgt_{postfix}.tif"
	tifffile.imwrite(str(tgt_path), tgt.astype("float32"), compression="lzw")
