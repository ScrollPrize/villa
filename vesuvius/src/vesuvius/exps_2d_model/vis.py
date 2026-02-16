from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
import cv2

import fit_data
import inv_map


def _write_ply_mesh(
	*,
	out_path: Path,
	x: np.ndarray,
	y: np.ndarray,
	z: np.ndarray,
	mask: np.ndarray | None,
) -> None:
	"""Write a simple connected grid mesh as ASCII PLY.

	Grid layout is (Z,H) where Z is depth and H is the winding height.
	Faces are created as two triangles per quad.
	"""
	z_size, h_size = (int(x.shape[0]), int(x.shape[1]))
	if x.shape != (z_size, h_size) or y.shape != (z_size, h_size) or z.shape != (z_size, h_size):
		raise ValueError("x/y/z must all be (Z,H)")

	msk = None
	if mask is not None:
		msk = mask.astype("bool", copy=False)
		if msk.shape != (z_size, h_size):
			raise ValueError("mask must be (Z,H)")

	verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
	faces: list[tuple[int, int, int]] = []

	def vid(zi: int, hi: int) -> int:
		return int(zi * h_size + hi)

	for zi in range(z_size - 1):
		for hi in range(h_size - 1):
			if msk is not None:
				if not (msk[zi, hi] and msk[zi, hi + 1] and msk[zi + 1, hi] and msk[zi + 1, hi + 1]):
					continue
			v00 = vid(zi, hi)
			v01 = vid(zi, hi + 1)
			v10 = vid(zi + 1, hi)
			v11 = vid(zi + 1, hi + 1)
			faces.append((v00, v10, v11))
			faces.append((v00, v11, v01))

	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", encoding="utf-8") as f:
		f.write("ply\n")
		f.write("format ascii 1.0\n")
		f.write(f"element vertex {int(verts.shape[0])}\n")
		f.write("property float x\n")
		f.write("property float y\n")
		f.write("property float z\n")
		f.write(f"element face {int(len(faces))}\n")
		f.write("property list uchar int vertex_indices\n")
		f.write("end_header\n")
		for vx, vy, vz in verts:
			f.write(f"{float(vx):.6f} {float(vy):.6f} {float(vz):.6f}\n")
		for a, b, c in faces:
			f.write(f"3 {int(a)} {int(b)} {int(c)}\n")


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


def save_corr_points(
	*,
	data: fit_data.FitData,
	xy_lr: torch.Tensor,
	xy_conn: torch.Tensor,
	points_xyz_winda: torch.Tensor,
	idx_left: torch.Tensor,
	valid_left: torch.Tensor,
	idx_right: torch.Tensor,
	valid_right: torch.Tensor,
	winding_avg: torch.Tensor,
	winding_err: torch.Tensor,
	out_dir: str,
	scale: int,
) -> None:
	with torch.no_grad():
		if int(points_xyz_winda.shape[0]) <= 0:
			return
		out = Path(out_dir)
		out_corr = out / "corr_vis"
		out_corr.mkdir(parents=True, exist_ok=True)
		h_img, w_img = data.size
		sf = max(1, int(scale))
		h_vis = int(h_img * sf)
		w_vis = int(w_img * sf)
		rad = max(2, int(round((2.0 * float(sf)) / 3.0)))
		seg_th = max(2, int(round(float(sf))))
		conn_th = max(1, seg_th - 1)

		n = int(xy_lr.shape[0])
		z_pts = points_xyz_winda[:, 2].to(dtype=torch.float32)
		z_idx = torch.round(z_pts).to(dtype=torch.int64).clamp(0, max(0, n - 1))
		z_unique = torch.unique(z_idx, sorted=True)

		for zt in z_unique.tolist():
			z = int(zt)
			bg = _draw_grid_vis(
				scale=sf,
				h_img=h_img,
				w_img=w_img,
				background=data.cos[z:z + 1],
				xy_lr=xy_lr[z:z + 1],
				xy_conn=xy_conn[z:z + 1],
				mask_lr=None,
				mask_conn=None,
			)

			x_off = 0
			y_off = 0
			w_im = w_vis
			h_im = h_vis
			if data.cos is not None:
				w_im = max(1, w_vis // 2)
				h_im = max(1, h_vis // 2)
				x_off = (w_vis - w_im) // 2
				y_off = (h_vis - h_im) // 2
			wx = float(max(1, int(w_img) - 1))
			hy = float(max(1, int(h_img) - 1))
			sx = float(max(1, int(w_im) - 1)) / wx
			sy = float(max(1, int(h_im) - 1)) / hy

			xy_lr_z = xy_lr[z].to(dtype=torch.float32, device="cpu")
			k_sel = torch.nonzero(z_idx == z, as_tuple=False).reshape(-1)
			for kk in k_sel.tolist():
				pt = points_xyz_winda[int(kk)].to(dtype=torch.float32, device="cpu")
				px = int(round(float(x_off + pt[0].item() * sx)))
				py = int(round(float(y_off + pt[1].item() * sy)))
				vl = bool(valid_left[int(kk)].item()) if int(valid_left.numel()) > int(kk) else False
				vr = bool(valid_right[int(kk)].item()) if int(valid_right.numel()) > int(kk) else False
				is_partial = not (vl and vr)

				l = idx_left[int(kk)].to(dtype=torch.int64, device="cpu")
				r = idx_right[int(kk)].to(dtype=torch.int64, device="cpu")
				r0l = int(l[1].item())
				c0l = int(l[2].item())
				r0r = int(r[1].item())
				c0r = int(r[2].item())
				if vl and r0l >= 0 and c0l >= 0 and (r0l + 1) < int(xy_lr_z.shape[0]) and c0l < int(xy_lr_z.shape[1]):
					a0 = xy_lr_z[r0l, c0l]
					a1 = xy_lr_z[r0l + 1, c0l]
					a0p = (int(round(float(x_off + a0[0].item() * sx))), int(round(float(y_off + a0[1].item() * sy))))
					a1p = (int(round(float(x_off + a1[0].item() * sx))), int(round(float(y_off + a1[1].item() * sy))))
					cv2.line(bg, a0p, a1p, (0, 200, 255), seg_th)
					ab = a1p[0] - a0p[0], a1p[1] - a0p[1]
					ap = px - a0p[0], py - a0p[1]
					den = float(ab[0] * ab[0] + ab[1] * ab[1])
					t = 0.0 if den <= 1e-12 else max(0.0, min(1.0, float(ap[0] * ab[0] + ap[1] * ab[1]) / den))
					cx = int(round(float(a0p[0] + t * ab[0])))
					cy = int(round(float(a0p[1] + t * ab[1])))
					cv2.line(bg, (px, py), (cx, cy), (0, 200, 255), conn_th)
				if vr and r0r >= 0 and c0r >= 0 and (r0r + 1) < int(xy_lr_z.shape[0]) and c0r < int(xy_lr_z.shape[1]):
					b0 = xy_lr_z[r0r, c0r]
					b1 = xy_lr_z[r0r + 1, c0r]
					b0p = (int(round(float(x_off + b0[0].item() * sx))), int(round(float(y_off + b0[1].item() * sy))))
					b1p = (int(round(float(x_off + b1[0].item() * sx))), int(round(float(y_off + b1[1].item() * sy))))
					cv2.line(bg, b0p, b1p, (255, 220, 0), seg_th)
					ab = b1p[0] - b0p[0], b1p[1] - b0p[1]
					ap = px - b0p[0], py - b0p[1]
					den = float(ab[0] * ab[0] + ab[1] * ab[1])
					t = 0.0 if den <= 1e-12 else max(0.0, min(1.0, float(ap[0] * ab[0] + ap[1] * ab[1]) / den))
					cx = int(round(float(b0p[0] + t * ab[0])))
					cy = int(round(float(b0p[1] + t * ab[1])))
					cv2.line(bg, (px, py), (cx, cy), (255, 220, 0), conn_th)

				if is_partial:
					overlay = bg.copy()
					cv2.circle(overlay, (px, py), rad, (255, 0, 255), -1)
					bg = cv2.addWeighted(overlay, 0.35, bg, 0.65, 0.0)
					cv2.circle(bg, (px, py), rad + 1, (200, 200, 200), 1)
				else:
					cv2.circle(bg, (px, py), rad, (255, 0, 255), -1)
					cv2.circle(bg, (px, py), rad + 1, (255, 255, 255), 1)

				wa = float("nan")
				we = float("nan")
				if int(winding_avg.numel()) > int(kk):
					wa = float(winding_avg[int(kk)].item())
				if int(winding_err.numel()) > int(kk):
					we = float(winding_err[int(kk)].item())
				if np.isfinite(wa) or np.isfinite(we):
					wa_s = "nan" if not np.isfinite(wa) else f"{wa:.2f}"
					we_s = "nan" if not np.isfinite(we) else f"{we:+.2f}"
					lbl = f"{wa_s}/{we_s}"
					cv2.putText(
						bg,
						lbl,
						(px + rad + 2, py - rad - 2),
						cv2.FONT_HERSHEY_PLAIN,
						0.45,
						(255, 255, 255),
						1,
						lineType=cv2.LINE_8,
					)

			out_path = out_corr / f"corr_grid_z{int(z):04d}.jpg"
			cv2.imwrite(str(out_path), np.flip(bg, -1))


def save(
	*,
	model,
	data: fit_data.FitData,
	res=None,
	vis_losses=None,
	postfix: str,
	out_dir: str,
	scale: int,
) -> None:
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	out_grids = out / "grids"
	out_loss = out / "loss_maps"
	out_img_loss = out / "img_loss_vis"
	out_tgt = out / "targets"
	out_vis = out / "vis"
	out_grids.mkdir(parents=True, exist_ok=True)
	out_loss.mkdir(parents=True, exist_ok=True)
	out_img_loss.mkdir(parents=True, exist_ok=True)
	out_tgt.mkdir(parents=True, exist_ok=True)
	out_vis.mkdir(parents=True, exist_ok=True)

	h_img, w_img = data.size
	if res is None:
		print("FIXME - save without res is deprecated!")
		res = model(data)
	z_i = int(getattr(model, "_last_grow_insert_z", 0) or 0)
	z_i = max(0, min(z_i, int(data.cos.shape[0]) - 1))
	bg = data.cos[z_i:z_i + 1]
	xy_lr = res.xy_lr[z_i:z_i + 1]
	xy_conn = res.xy_conn[z_i:z_i + 1]
	mask_lr = res.mask_lr[z_i:z_i + 1]
	mask_conn = res.mask_conn[z_i:z_i + 1]
	h2 = int(h_img) * 2
	w2 = int(w_img) * 2

	grid_xy = xy_lr
	grid_vis = _draw_grid_vis(
		scale=scale,
		h_img=h_img,
		w_img=w_img,
		background=bg,
		xy_lr=grid_xy,
		xy_conn=xy_conn,
		mask_lr=mask_lr,
		mask_conn=mask_conn,
	)
	grid_path = out_grids / f"res_grid_{postfix}.jpg"
	cv2.imwrite(str(grid_path), np.flip(grid_vis, -1))

	# Also dump the (optionally blurred) grad_mag tensor.
	mag_np = data.grad_mag[z_i, 0].detach().cpu().numpy().astype("float32")
	tifffile.imwrite(str(out_vis / f"res_grad_mag_{postfix}.tif"), mag_np, compression="lzw")

	# Dump stage img masks (if present) for debugging.
	if res.stage_img_masks is not None:
		for lbl, m in res.stage_img_masks.items():
			m0 = m
			if m0.ndim == 4 and int(m0.shape[0]) > 1:
				m0 = m0[z_i:z_i + 1]
			if m0.ndim == 4 and int(m0.shape[1]) == 1:
				m_np = m0[0, 0].detach().cpu().numpy().astype("float32")
				out_path = out_vis / f"res_stage_img_mask_{str(lbl)}_{postfix}.tif"
				tifffile.imwrite(str(out_path), m_np, compression="lzw")

				# Overlay mask on top of the grid visualization.
				grid_ov = grid_vis.copy()
				h_vis, w_vis = int(grid_ov.shape[0]), int(grid_ov.shape[1])
				w_im = w_vis
				h_im = h_vis
				x_off = 0
				y_off = 0
				if bg is not None:
					w_im = max(1, w_vis // 2)
					h_im = max(1, h_vis // 2)
					x_off = (w_vis - w_im) // 2
					y_off = (h_vis - h_im) // 2
				m_rs = cv2.resize(m_np, (w_im, h_im), interpolation=cv2.INTER_LINEAR)
				m_rs = np.clip(m_rs, 0.0, 1.0)
				alpha = (m_rs * 0.6).astype("float32")
				roi = grid_ov[y_off:y_off + h_im, x_off:x_off + w_im, :].astype("float32")
				col = np.zeros((h_im, w_im, 3), dtype="float32")
				col[:, :, 2] = 255.0
				roi = roi * (1.0 - alpha[:, :, None]) + col * alpha[:, :, None]
				grid_ov[y_off:y_off + h_im, x_off:x_off + w_im, :] = np.clip(roi, 0.0, 255.0).astype("uint8")
				out_path_ov = out_vis / f"res_stage_img_mask_overlay_{str(lbl)}_{postfix}.jpg"
				cv2.imwrite(str(out_path_ov), np.flip(grid_ov, -1))

	def _save_img_loss_vis(*, iters: int | None = None, postfix2: str | None = None) -> None:
		it_label = "default" if iters is None else f"it{int(iters)}"
		p2 = str(postfix2) if postfix2 is not None else f"{postfix}_{it_label}"
		inv_kwargs: dict[str, object] = {"xy_lr": xy_lr, "h_out": h_img, "w_out": w_img}
		if iters is not None:
			inv_kwargs["iters"] = int(iters)
		uv_img, uv_mask = inv_map.inverse_map_autograd(**inv_kwargs)
		uv_img_nchw = uv_img.permute(0, 3, 1, 2).contiguous()
		uv_img_nchw = torch.nn.functional.interpolate(uv_img_nchw, size=(h2, w2), mode="bilinear", align_corners=True)
		uv_img = uv_img_nchw.permute(0, 2, 3, 1).contiguous()
		uv_mask = torch.nn.functional.interpolate(uv_mask, size=(h2, w2), mode="nearest")

		def _scale_uv_for_src(*, uv_lr: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
			_hm0, _wm0 = (int(xy_lr.shape[1]), int(xy_lr.shape[2]))
			_hm1, _wm1 = (int(src.shape[2]), int(src.shape[3]))
			fx = float(max(1, _wm1 - 1)) / float(max(1, _wm0 - 1))
			fy = float(max(1, _hm1 - 1)) / float(max(1, _hm0 - 1))
			uv2 = uv_lr.clone()
			uv2[..., 0] = uv2[..., 0] * fx
			uv2[..., 1] = uv2[..., 1] * fy
			return uv2
		img_loss_layers: list[np.ndarray] = []
		img_loss_names: list[str] = []

		grid_vis2 = grid_vis
		grid_gray = (cv2.cvtColor(grid_vis2, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0)
		hg, wg = int(grid_gray.shape[0]), int(grid_gray.shape[1])
		ch, cw = int(uv_mask.shape[2]), int(uv_mask.shape[3])
		y0 = (hg - ch) // 2
		x0 = (wg - cw) // 2
		crop = grid_gray[y0:y0 + ch, x0:x0 + cw].copy()
		img_loss_layers.append(crop)
		img_loss_names.append("grid_crop")

		tgt_plain = res.target_plain[z_i:z_i + 1]
		tgt_mod = res.target_mod[z_i:z_i + 1]
		uv_tgt = _scale_uv_for_src(uv_lr=uv_img, src=tgt_plain)
		tgt_plain_img = inv_map.warp_nchw_from_uv(src=tgt_plain, uv=uv_tgt, uv_mask=uv_mask, fill=0.5)
		tgt_mod_img = inv_map.warp_nchw_from_uv(src=tgt_mod, uv=uv_tgt, uv_mask=uv_mask, fill=0.5)
		img_loss_layers.append(tgt_plain_img[0, 0].detach().cpu().numpy().astype("float32"))
		img_loss_names.append("target_plain")
		img_loss_layers.append(tgt_mod_img[0, 0].detach().cpu().numpy().astype("float32"))
		img_loss_names.append("target_mod")

		for _k, lm, mask in loss_maps:
			mt = lm.detach()
			mk = mask.detach() if mask is not None else None
			if mt.ndim != 4:
				continue
			if int(mt.shape[0]) != int(uv_img.shape[0]):
				if int(uv_img.shape[0]) == 1 and int(mt.shape[0]) > z_i:
					mt = mt[z_i:z_i + 1]
					if mk is not None and mk.ndim == 4 and int(mk.shape[0]) > z_i:
						mk = mk[z_i:z_i + 1]
				else:
					continue
			if mk is not None:
				if mk.ndim == 4 and int(mk.shape[0]) == int(mt.shape[0]):
					mt = mt * mk.to(dtype=mt.dtype)
				elif mk.ndim == 2 and int(mt.shape[0]) == 1:
					mt = mt * mk[None, None].to(dtype=mt.dtype)
			if int(mt.shape[2]) > 1 and int(mt.shape[3]) > 1:
				uv_m = _scale_uv_for_src(uv_lr=uv_img, src=mt)
				im = inv_map.warp_nchw_from_uv(src=mt, uv=uv_m, uv_mask=uv_mask, fill=0.5)
				img_loss_layers.append(im[0, 0].detach().cpu().numpy().astype("float32"))
				img_loss_names.append(str(_k))

		if not img_loss_layers:
			return
		out_path_img = out_img_loss / f"res_img_loss_{p2}.tif"
		with tifffile.TiffWriter(str(out_path_img), bigtiff=False) as tw:
			for name, layer in zip(img_loss_names, img_loss_layers, strict=True):
				page_name = str(name)
				tw.write(
					layer,
					compression="lzw",
					description=page_name,
					extratags=[(285, "s", 0, page_name, False)],
				)
		return
	loss_maps: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []
	if vis_losses is not None:
		for name, (lms, masks) in vis_losses.loss_maps.items():
			for i, lm in enumerate(lms):
				mask = masks[i] if i < len(masks) else None
				suffix = str(name) if len(lms) == 1 else f"{name}_{i}"
				loss_maps.append((suffix, lm, mask))

	_save_img_loss_vis()

	for _k, lm, mask in loss_maps:
		mt0 = lm.detach()
		if mask is not None:
			mt0 = mt0 * mask.to(dtype=mt0.dtype)
		m = mt0.detach().cpu()
		if m.ndim == 4:
			m = m[:, 0]
		out_path = out_loss / f"res_loss_{_k}_{postfix}.tif"
		tifffile.imwrite(str(out_path), m.numpy().astype("float32"), compression="lzw")

	# One horizontally-concatenated float tif for quick inspection.
	loss_2d: list[tuple[str, np.ndarray]] = []
	for _k, lm, mask in loss_maps:
		m = lm.detach()
		if mask is not None:
			m = m * mask.to(dtype=m.dtype)
		m = m.detach().cpu()
		if m.ndim == 4:
			m2 = m[0, 0].detach().numpy().astype("float32")
		elif m.ndim == 2:
			m2 = m.detach().numpy().astype("float32")
		else:
			continue
		loss_2d.append((str(_k), m2))
	# Float tif with a top label band (label pixels are visual-only floats in [0,1]).
	if not loss_2d:
		loss_2d = [("none", np.zeros((1, 1), dtype="float32"))]
	concat_f = _loss_concat_vis(loss_maps_2d=loss_2d, border_px=6, label_px=16)
	concat_f = np.repeat(np.repeat(concat_f, 8, axis=0), 8, axis=1)
	label_h = max(1, 16 // 2)
	border = 6
	s = 8

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
	label_rows = min(int(label_h * s), int(concat_f.shape[0]))
	concat_f[0:label_rows, :] = (lab_u8.astype("float32") / 255.0)[0:label_rows, :]
	concat_path = out_vis / f"res_loss_concat_{postfix}.tif"
	tifffile.imwrite(str(concat_path), concat_f.astype("float32"), compression="lzw")

	# Save the mesh-domain target (no image-size dependence).
	tgt = res.target_plain[0, 0].detach().cpu().numpy().astype("float32")
	tgt_path = out_tgt / f"res_tgt_{postfix}.tif"
	tifffile.imwrite(str(tgt_path), tgt, compression="lzw")

	# Write connected PLY meshes (one per winding) using all z slices (no skipping).
	# Grid layout per winding: (Z,Hm), where Hm is mesh height (winding direction).
	xy_all = res.xy_lr.detach().to(dtype=torch.float32, device="cpu")
	mask_all = None
	if res.mask_lr is not None:
		mask_all = res.mask_lr.detach().to(dtype=torch.float32, device="cpu")
	meta_sf = float(getattr(data, "downscale", 1.0) or 1.0)
	nz, hm, wm, _c2 = (int(v) for v in xy_all.shape)
	out_ply = out / "ply"
	for wi in range(wm):
		x = (xy_all[:, :, wi, 0].numpy().astype("float32") * meta_sf)
		y = (xy_all[:, :, wi, 1].numpy().astype("float32") * meta_sf)
		z = np.broadcast_to(np.arange(nz, dtype="float32")[:, None], (nz, hm))
		msk = None
		if mask_all is not None:
			msk = (mask_all[:, 0, :, wi].numpy().astype("float32") > 0.0)
		out_path = out_ply / f"winding_{wi:04d}" / f"{postfix}.ply"
		_write_ply_mesh(out_path=out_path, x=x, y=y, z=z, mask=msk)
