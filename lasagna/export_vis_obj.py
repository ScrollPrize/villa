"""Export multi-layer OBJ files for MeshLab visualization.

Produces separate .obj files for mesh, connections, volume slices, and loss maps.
Each file can be loaded as a separate layer in MeshLab.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

import fit_data
import model as fit_model
import opt_loss_dir
import opt_loss_smooth
import opt_loss_step
import opt_loss_winding_density


# ---------------------------------------------------------------------------
# OBJ / MTL writers
# ---------------------------------------------------------------------------

def _write_mtl(path: Path, material_name: str, texture_file: str) -> None:
	lines = [
		f"newmtl {material_name}",
		"Ka 1.0 1.0 1.0",
		"Kd 1.0 1.0 1.0",
		"Ks 0.0 0.0 0.0",
		"d 1.0",
		"illum 1",
		f"map_Kd {texture_file}",
	]
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_obj_mesh(path: Path, verts: np.ndarray, faces: np.ndarray,
					uvs: np.ndarray, mtl_name: str | None) -> None:
	"""Write mesh OBJ with optional material/UVs.

	verts: (N, 3)
	faces: (F, 3) 0-indexed triangle indices
	uvs: (N, 2) or None
	"""
	lines: list[str] = []
	stem = path.stem
	if mtl_name is not None:
		lines.append(f"mtllib {stem}.mtl")
		lines.append(f"usemtl {mtl_name}")

	for v in verts:
		lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

	if uvs is not None:
		for uv in uvs:
			lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
		for f in faces:
			i0, i1, i2 = f[0] + 1, f[1] + 1, f[2] + 1
			lines.append(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}")
	else:
		for f in faces:
			lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_obj_lines(path: Path, verts: np.ndarray,
					 lines_idx: list[tuple[int, int]]) -> None:
	"""Write OBJ with line elements.

	verts: (N, 3)
	lines_idx: list of (v0, v1) 0-indexed pairs
	"""
	out: list[str] = []
	for v in verts:
		out.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
	for a, b in lines_idx:
		out.append(f"l {a+1} {b+1}")
	path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _write_obj_quad(path: Path, corners: np.ndarray, mtl_name: str) -> None:
	"""Write a single textured quad as OBJ.

	corners: (4, 3) — quad vertices in order (0,0), (1,0), (1,1), (0,1)
	"""
	stem = path.stem
	lines = [
		f"mtllib {stem}.mtl",
		f"usemtl {mtl_name}",
	]
	for c in corners:
		lines.append(f"v {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
	lines.append("vt 0.0 0.0")
	lines.append("vt 1.0 0.0")
	lines.append("vt 1.0 1.0")
	lines.append("vt 0.0 1.0")
	lines.append("f 1/1 2/2 3/3")
	lines.append("f 1/1 3/3 4/4")
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PNG writer (minimal, no PIL dependency)
# ---------------------------------------------------------------------------

def _write_png(path: Path, img: np.ndarray) -> None:
	"""Write an RGB uint8 image as PNG using only zlib (no PIL/matplotlib)."""
	import zlib
	h, w, c = img.shape
	assert c == 3

	def _chunk(ctype: bytes, data: bytes) -> bytes:
		return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

	raw = b""
	for y in range(h):
		raw += b"\x00" + img[y].tobytes()

	out = b"\x89PNG\r\n\x1a\n"
	out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
	out += _chunk(b"IDAT", zlib.compress(raw, 6))
	out += _chunk(b"IEND", b"")
	path.write_bytes(out)


# ---------------------------------------------------------------------------
# Colormapping
# ---------------------------------------------------------------------------

# Viridis LUT (64 entries, linearly interpolated)
_VIRIDIS_ANCHORS = np.array([
	[0.267004, 0.004874, 0.329415],
	[0.282327, 0.140926, 0.457517],
	[0.253935, 0.265254, 0.529983],
	[0.206756, 0.371758, 0.553117],
	[0.163625, 0.471133, 0.558148],
	[0.127568, 0.566949, 0.550556],
	[0.134692, 0.658636, 0.517649],
	[0.266941, 0.748751, 0.440573],
	[0.477504, 0.821444, 0.318195],
	[0.741388, 0.873449, 0.149561],
	[0.993248, 0.906157, 0.143936],
], dtype=np.float32)


def _viridis(vals: np.ndarray) -> np.ndarray:
	"""Map [0,1] floats to RGB uint8 using a viridis-like colormap."""
	t = np.clip(vals, 0.0, 1.0) * (len(_VIRIDIS_ANCHORS) - 1)
	idx = np.floor(t).astype(np.int32)
	idx = np.clip(idx, 0, len(_VIRIDIS_ANCHORS) - 2)
	frac = (t - idx).astype(np.float32)
	c = _VIRIDIS_ANCHORS[idx] * (1 - frac[..., None]) + _VIRIDIS_ANCHORS[idx + 1] * frac[..., None]
	return (c * 255).clip(0, 255).astype(np.uint8)


def _loss_to_png(path: Path, lm_2d: np.ndarray, mask_2d: np.ndarray | None) -> None:
	"""Colormap a 2D loss map and write as PNG."""
	# Normalize to [0, 1] using robust percentile
	valid = lm_2d[mask_2d > 0.5] if mask_2d is not None else lm_2d.ravel()
	if len(valid) == 0:
		vmin, vmax = 0.0, 1.0
	else:
		vmin = float(np.percentile(valid, 2))
		vmax = float(np.percentile(valid, 98))
	if vmax - vmin < 1e-8:
		vmax = vmin + 1.0
	normed = (lm_2d - vmin) / (vmax - vmin)
	rgb = _viridis(normed)
	# Dim masked-out regions
	if mask_2d is not None:
		dark = (mask_2d < 0.5)
		rgb[dark] = (rgb[dark] * 0.3).astype(np.uint8)
	_write_png(path, rgb)


# ---------------------------------------------------------------------------
# Mesh geometry helpers
# ---------------------------------------------------------------------------

def _triangulate_grid(D: int, H: int, W: int) -> np.ndarray:
	"""Triangulate a D-layer (H, W) grid into triangle faces.

	Returns (F, 3) 0-indexed face array. All D layers combined.
	"""
	faces = []
	for d in range(D):
		base = d * H * W
		for h in range(H - 1):
			for w in range(W - 1):
				v00 = base + h * W + w
				v10 = base + (h + 1) * W + w
				v11 = base + (h + 1) * W + (w + 1)
				v01 = base + h * W + (w + 1)
				faces.append([v00, v10, v11])
				faces.append([v00, v11, v01])
	return np.array(faces, dtype=np.int32)


def _mesh_uvs(D: int, H: int, W: int) -> np.ndarray:
	"""Generate UV coordinates for a D-layer (H, W) grid.

	Slices stacked vertically in texture space.
	Returns (D*H*W, 2).
	"""
	uvs = np.zeros((D * H * W, 2), dtype=np.float32)
	for d in range(D):
		for h in range(H):
			for w in range(W):
				idx = d * H * W + h * W + w
				uvs[idx, 0] = w / max(1, W - 1)
				uvs[idx, 1] = 1.0 - (d * H + h) / max(1, D * H - 1)
	return uvs


# ---------------------------------------------------------------------------
# Volume slice helpers
# ---------------------------------------------------------------------------

def _slice_texture(data: fit_data.FitData3D, plane: str,
				   channel: str) -> np.ndarray:
	"""Read a volume slice directly at native voxel resolution.

	plane: "xy", "xz", or "yz"
	Returns (H, W, 3) uint8 at exact data resolution (no interpolation).
	"""
	vol = getattr(data, channel)  # (1, 1, Z, Y, X) uint8
	arr = vol.squeeze().cpu().numpy()  # (Z, Y, X)
	Z, Y, X = arr.shape
	cz, cy, cx = Z // 2, Y // 2, X // 2
	if plane == "xy":
		slc = arr[cz, :, :]      # (Y, X)
	elif plane == "xz":
		slc = arr[:, cy, :]      # (Z, X)
	elif plane == "yz":
		slc = arr[:, :, cx]      # (Z, Y)
	else:
		raise ValueError(f"unknown plane: {plane}")
	gray = slc.astype(np.uint8)
	return np.stack([gray, gray, gray], axis=-1)[::-1]


def _slice_corners(plane: str, bbox_min: np.ndarray, bbox_max: np.ndarray,
				   center: np.ndarray) -> np.ndarray:
	"""Return 4 corners of a slice quad in 3D fullres coords.

	Order: (0,0), (1,0), (1,1), (0,1) for UV mapping.
	"""
	if plane == "xy":
		z = center[2]
		return np.array([
			[bbox_min[0], bbox_min[1], z],
			[bbox_max[0], bbox_min[1], z],
			[bbox_max[0], bbox_max[1], z],
			[bbox_min[0], bbox_max[1], z],
		], dtype=np.float32)
	elif plane == "xz":
		y = center[1]
		return np.array([
			[bbox_min[0], y, bbox_min[2]],
			[bbox_max[0], y, bbox_min[2]],
			[bbox_max[0], y, bbox_max[2]],
			[bbox_min[0], y, bbox_max[2]],
		], dtype=np.float32)
	elif plane == "yz":
		x = center[0]
		return np.array([
			[x, bbox_min[1], bbox_min[2]],
			[x, bbox_max[1], bbox_min[2]],
			[x, bbox_max[1], bbox_max[2]],
			[x, bbox_min[1], bbox_max[2]],
		], dtype=np.float32)


# ---------------------------------------------------------------------------
# Loss term registry
# ---------------------------------------------------------------------------

_LOSS_FUNCS = {
	"step": opt_loss_step.step_loss,
	"smooth": opt_loss_smooth.smooth_loss,
	"winding_density": opt_loss_winding_density.winding_density_loss,
	"normal": opt_loss_dir.normal_loss,
}


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_vis_obj(
	model_path: str,
	data_path: str,
	output_dir: str,
	*,
	slices: list[str],
	channels: list[str],
	losses: list[str],
	include_mesh: bool,
	include_connections: bool,
	device: str = "cuda",
) -> None:
	"""Export multi-layer OBJ visualization files."""
	dev = torch.device(device)
	out = Path(output_dir)
	out.mkdir(parents=True, exist_ok=True)

	# Load model
	print(f"[export_vis] loading model from {model_path}", flush=True)
	st = torch.load(model_path, map_location=dev, weights_only=False)
	mdl = fit_model.Model3D.from_checkpoint(st, device=dev)

	# Load data (auto-crop around mesh bbox)
	print(f"[export_vis] loading data from {data_path}", flush=True)
	data = fit_data.load_3d_for_model(path=data_path, device=dev, model=mdl,
									cuda_gridsample=dev.type == "cuda")

	# Forward pass
	print("[export_vis] running forward pass", flush=True)
	with torch.no_grad():
		res = mdl(data)

	xyz_lr = res.xyz_lr.detach().cpu().numpy()  # (D, Hm, Wm, 3)
	D, Hm, Wm, _ = xyz_lr.shape
	print(f"[export_vis] mesh shape: D={D}, Hm={Hm}, Wm={Wm}", flush=True)

	# Mesh bounding box in fullres coords
	bbox_min = xyz_lr.reshape(-1, 3).min(axis=0)
	bbox_max = xyz_lr.reshape(-1, 3).max(axis=0)
	center = 0.5 * (bbox_min + bbox_max)

	# Data extent in fullres coords (for slice textures)
	Z, Y, X = data.size
	data_min = np.array(data.origin_fullres, dtype=np.float32)
	data_max = data_min + np.array([
		(X - 1) * data.spacing[0],
		(Y - 1) * data.spacing[1],
		(Z - 1) * data.spacing[2],
	], dtype=np.float32)
	data_center = 0.5 * (data_min + data_max)

	# ------ Mesh ------
	if include_mesh:
		print("[export_vis] writing mesh.obj", flush=True)
		verts = xyz_lr.reshape(-1, 3)
		faces = _triangulate_grid(D, Hm, Wm)
		_write_obj_mesh(out / "mesh.obj", verts, faces, uvs=None, mtl_name=None)

	# ------ Connections ------
	if include_connections:
		print("[export_vis] writing connections.obj", flush=True)
		xy_conn = res.xy_conn.detach().cpu().numpy()  # (D, Hm, Wm, 3, 3)
		mask_conn = res.mask_conn.detach().cpu().numpy()  # (D, 1, Hm, Wm, 3)
		conn_verts: list[np.ndarray] = []
		conn_lines: list[tuple[int, int]] = []
		vi = 0
		for d in range(D):
			for h in range(Hm):
				for w in range(Wm):
					center_pt = xy_conn[d, h, w, :, 1]  # (3,) xyz of self
					# prev connection
					if mask_conn[d, 0, h, w, 0] > 0.5 and mask_conn[d, 0, h, w, 1] > 0.5:
						prev_pt = xy_conn[d, h, w, :, 0]
						conn_verts.append(center_pt)
						conn_verts.append(prev_pt)
						conn_lines.append((vi, vi + 1))
						vi += 2
					# next connection
					if mask_conn[d, 0, h, w, 2] > 0.5 and mask_conn[d, 0, h, w, 1] > 0.5:
						next_pt = xy_conn[d, h, w, :, 2]
						conn_verts.append(center_pt)
						conn_verts.append(next_pt)
						conn_lines.append((vi, vi + 1))
						vi += 2
		if conn_verts:
			_write_obj_lines(out / "connections.obj",
							 np.array(conn_verts, dtype=np.float32), conn_lines)
		else:
			print("[export_vis] no valid connections to write", flush=True)

	# ------ Volume slices ------
	for plane in slices:
		for channel in channels:
			vol = getattr(data, channel, None)
			if vol is None:
				print(f"[export_vis] skipping slice {plane}/{channel} (not in data)", flush=True)
				continue
			name = f"slice_{plane}_{channel}"
			print(f"[export_vis] writing {name}", flush=True)
			# Texture (native voxel resolution, no scaling)
			tex = _slice_texture(data, plane, channel)
			png_name = f"{name}.png"
			_write_png(out / png_name, tex)
			# MTL
			_write_mtl(out / f"{name}.mtl", name, png_name)
			# OBJ quad
			corners = _slice_corners(plane, data_min, data_max, data_center)
			_write_obj_quad(out / f"{name}.obj", corners, name)

	# ------ Loss maps ------
	if losses:
		verts_flat = xyz_lr.reshape(-1, 3)
		faces = _triangulate_grid(D, Hm, Wm)
		uvs = _mesh_uvs(D, Hm, Wm)

		for loss_name in losses:
			print(f"[export_vis] computing loss: {loss_name}", flush=True)
			loss_fn = _LOSS_FUNCS[loss_name]
			with torch.no_grad():
				_lv, lms, masks = loss_fn(res=res)

			lm = lms[0].detach().cpu().numpy()  # (D, 1, H', W')
			mask = masks[0].detach().cpu().numpy()

			# Loss maps may be smaller than mesh grid (e.g. Hm-1, Wm-1 for face-based losses)
			# Resize all to (D*Hm, Wm) for the texture
			lm_sq = lm.squeeze(1)  # (D, H', W')
			mask_sq = mask.squeeze(1)  # (D, H', W')

			# Stack along height for texture: (D*H', W') -> resize to (D*Hm, Wm)
			lm_stacked = lm_sq.reshape(-1, lm_sq.shape[-1])  # (D*H', W')
			mask_stacked = mask_sq.reshape(-1, mask_sq.shape[-1])

			# Resize via bilinear to match mesh UV layout
			tex_h, tex_w = D * Hm, Wm
			if lm_stacked.shape != (tex_h, tex_w):
				lm_t = torch.from_numpy(lm_stacked).unsqueeze(0).unsqueeze(0).float()
				lm_t = torch.nn.functional.interpolate(lm_t, size=(tex_h, tex_w),
													   mode="bilinear", align_corners=True)
				lm_stacked = lm_t.squeeze().numpy()
				mask_t = torch.from_numpy(mask_stacked).unsqueeze(0).unsqueeze(0).float()
				mask_t = torch.nn.functional.interpolate(mask_t, size=(tex_h, tex_w),
														 mode="nearest")
				mask_stacked = mask_t.squeeze().numpy()

			obj_name = f"loss_{loss_name}"
			png_name = f"{obj_name}.png"
			_loss_to_png(out / png_name, lm_stacked, mask_stacked)
			_write_mtl(out / f"{obj_name}.mtl", obj_name, png_name)
			_write_obj_mesh(out / f"{obj_name}.obj", verts_flat, faces, uvs, obj_name)

	print(f"[export_vis] done. Output: {output_dir}", flush=True)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Export OBJ visualization of a fitted model")
	parser.add_argument("--model", required=True, help="Path to model.pt checkpoint")
	parser.add_argument("--input", required=True, help="Path to lasagna normals zarr")
	parser.add_argument("--output-dir", required=True, help="Output directory for OBJ/MTL/PNG files")
	parser.add_argument("--slices", nargs="*", default=["xy", "xz", "yz"],
						choices=["xy", "xz", "yz"], help="Volume slice planes (default: xy xz yz)")
	parser.add_argument("--channels", nargs="*", default=["cos", "pred_dt"],
						help="Volume channels to slice (default: cos pred_dt)")
	parser.add_argument("--losses", nargs="*", default=["normal", "step"],
						help="Loss maps to export (default: normal step)")
	parser.add_argument("--no-mesh", action="store_true", help="Skip mesh export")
	parser.add_argument("--no-connections", action="store_true", help="Skip connection lines")
	parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
	args = parser.parse_args()

	export_vis_obj(
		model_path=args.model,
		data_path=args.input,
		output_dir=args.output_dir,
		slices=args.slices,
		channels=args.channels,
		losses=args.losses,
		include_mesh=not args.no_mesh,
		include_connections=not args.no_connections,
		device=args.device,
	)
