"""Export multi-layer OBJ files for MeshLab visualization.

Produces separate .obj files for mesh, connections, volume slices, and loss maps.
Each file can be loaded as a separate layer in MeshLab.
"""
from __future__ import annotations

import json
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
import opt_loss_pred_dt
import opt_loss_winding_volume


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


def _write_obj_multi_quad(path: Path,
						  quads: list[tuple[np.ndarray, str]]) -> None:
	"""Write multiple textured quads into a single OBJ, each with its own material.

	quads: list of (corners(4,3), material_name) pairs.
	Each material must have a corresponding .mtl file already written.
	"""
	stem = path.stem
	lines = [f"mtllib {stem}.mtl"]
	vi = 0  # running vertex count
	for corners, mtl_name in quads:
		lines.append(f"usemtl {mtl_name}")
		for c in corners:
			lines.append(f"v {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
		lines.append(f"vt 0.0 0.0")
		lines.append(f"vt 1.0 0.0")
		lines.append(f"vt 1.0 1.0")
		lines.append(f"vt 0.0 1.0")
		v1 = vi + 1  # OBJ is 1-indexed
		lines.append(f"f {v1}/{v1} {v1+1}/{v1+1} {v1+2}/{v1+2}")
		lines.append(f"f {v1}/{v1} {v1+2}/{v1+2} {v1+3}/{v1+3}")
		vi += 4
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mtl_multi(path: Path,
					 materials: list[tuple[str, str]]) -> None:
	"""Write an MTL file with multiple materials.

	materials: list of (material_name, texture_filename) pairs.
	"""
	blocks = []
	for mat_name, tex_file in materials:
		blocks.append(
			f"newmtl {mat_name}\n"
			f"Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n"
			f"d 1.0\nillum 1\nmap_Kd {tex_file}"
		)
	path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PNG writer (minimal, no PIL dependency)
# ---------------------------------------------------------------------------

def _write_pfm(path: Path, data: np.ndarray) -> None:
	"""Write a 2D float32 array as a grayscale PFM (Portable FloatMap) file."""
	h, w = data.shape[:2]
	with open(path, 'wb') as f:
		f.write(b'Pf\n')
		f.write(f'{w} {h}\n'.encode())
		f.write(b'-1.0\n')  # negative = little-endian
		# PFM stores bottom-to-top
		f.write(np.ascontiguousarray(data[::-1]).astype(np.float32).tobytes())


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
	"""Colormap a 2D loss map as grayscale and write as PNG.

	White = high loss, black = zero loss or masked out.
	"""
	# Normalize to [0, 1] using robust percentile of valid region
	valid = lm_2d[mask_2d > 0.5] if mask_2d is not None else lm_2d.ravel()
	if len(valid) == 0:
		vmin, vmax = 0.0, 1.0
	else:
		vmin = float(np.percentile(valid, 2))
		vmax = float(np.percentile(valid, 98))
	if vmax - vmin < 1e-8:
		vmax = vmin + 1.0
	normed = np.clip((lm_2d - vmin) / (vmax - vmin), 0.0, 1.0)
	gray = (normed * 255).astype(np.uint8)
	# Black out masked regions
	if mask_2d is not None:
		gray[mask_2d < 0.5] = 0
	_write_png(path, np.stack([gray, gray, gray], axis=-1))


# ---------------------------------------------------------------------------
# Mesh geometry helpers
# ---------------------------------------------------------------------------

def _filter_mesh(verts: np.ndarray, faces: np.ndarray,
				 valid: np.ndarray, uvs: np.ndarray | None = None,
				 ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
	"""Remove faces with any invalid vertex and compact the index space.

	verts: (V, 3), faces: (F, 3) 0-indexed, valid: (V,) bool, uvs: (V, 2)|None
	Returns (verts', faces', uvs') with only valid geometry.
	"""
	# Keep faces where all 3 vertices are valid
	face_ok = valid[faces[:, 0]] & valid[faces[:, 1]] & valid[faces[:, 2]]
	faces = faces[face_ok]
	if len(faces) == 0:
		empty3 = np.zeros((0, 3), dtype=verts.dtype)
		empty2 = np.zeros((0, 2), dtype=np.float32) if uvs is not None else None
		return empty3, np.zeros((0, 3), dtype=np.int32), empty2
	# Compact: remap old vertex indices → new contiguous indices
	used = np.unique(faces.ravel())
	remap = np.full(len(verts), -1, dtype=np.int32)
	remap[used] = np.arange(len(used), dtype=np.int32)
	return (
		verts[used],
		remap[faces],
		uvs[used] if uvs is not None else None,
	)


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

def _slice_index(plane: str, shape_zyx: tuple[int, int, int],
				 frac: float) -> int:
	"""Return the voxel index along the sliced axis for a given fraction [0, 1]."""
	Z, Y, X = shape_zyx
	if plane == "xy":
		return int(np.clip(frac * (Z - 1), 0, Z - 1))
	elif plane == "xz":
		return int(np.clip(frac * (Y - 1), 0, Y - 1))
	elif plane == "yz":
		return int(np.clip(frac * (X - 1), 0, X - 1))
	raise ValueError(f"unknown plane: {plane}")


def _take_slice(arr: np.ndarray, plane: str, idx: int) -> np.ndarray:
	"""Extract a 2D slice from a (Z, Y, X) array at the given index."""
	if plane == "xy":
		return arr[idx, :, :]      # (Y, X)
	elif plane == "xz":
		return arr[:, idx, :]      # (Z, X)
	elif plane == "yz":
		return arr[:, :, idx]      # (Z, Y)
	raise ValueError(f"unknown plane: {plane}")


_CHANNEL_VIS_MAX: dict[str, float] = {
	"pred_dt": 10.0,
}


def _slice_texture(data: fit_data.FitData3D, plane: str,
				   channel: str, frac: float = 0.5) -> np.ndarray:
	"""Read a volume slice directly at native voxel resolution.

	plane: "xy", "xz", or "yz"
	frac: position along the sliced axis, 0.0 = start, 1.0 = end
	Returns (H, W, 3) uint8 at exact data resolution (no interpolation).
	"""
	vol = getattr(data, channel)  # (1, 1, Z, Y, X) uint8
	arr = vol.squeeze().cpu().numpy()  # (Z, Y, X)
	idx = _slice_index(plane, arr.shape, frac)
	slc = _take_slice(arr, plane, idx).astype(np.float32)
	vis_max = _CHANNEL_VIS_MAX.get(channel)
	if vis_max is not None:
		gray = np.clip(slc / vis_max * 255.0, 0.0, 255.0).astype(np.uint8)
	else:
		gray = slc.astype(np.uint8)
	return np.stack([gray, gray, gray], axis=-1)[::-1]


def _slice_winding_volume(data: fit_data.FitData3D, plane: str,
						  frac: float = 0.5) -> np.ndarray:
	"""Read a slice from the winding volume tensor and colormap to RGB."""
	wv = data.winding_volume  # (1, 1, Z, Y, X) float32
	arr = wv.squeeze().cpu().numpy()  # (Z, Y, X)
	idx = _slice_index(plane, arr.shape, frac)
	slc = _take_slice(arr, plane, idx)
	# Normalize: winding values are typically in [0, D+1] range
	vmin, vmax = float(slc.min()), float(slc.max())
	if vmax - vmin < 1e-8:
		vmax = vmin + 1.0
	normed = (slc - vmin) / (vmax - vmin)
	return _viridis(normed)[::-1]



def _slice_corners(plane: str, bbox_min: np.ndarray, bbox_max: np.ndarray,
				   frac: float = 0.5) -> np.ndarray:
	"""Return 4 corners of a slice quad in 3D fullres coords.

	frac: position along the sliced axis, 0.0 = bbox_min, 1.0 = bbox_max.
	Order: (0,0), (1,0), (1,1), (0,1) for UV mapping.
	"""
	pos = bbox_min + frac * (bbox_max - bbox_min)
	if plane == "xy":
		z = pos[2]
		return np.array([
			[bbox_min[0], bbox_min[1], z],
			[bbox_max[0], bbox_min[1], z],
			[bbox_max[0], bbox_max[1], z],
			[bbox_min[0], bbox_max[1], z],
		], dtype=np.float32)
	elif plane == "xz":
		y = pos[1]
		return np.array([
			[bbox_min[0], y, bbox_min[2]],
			[bbox_max[0], y, bbox_min[2]],
			[bbox_max[0], y, bbox_max[2]],
			[bbox_min[0], y, bbox_max[2]],
		], dtype=np.float32)
	elif plane == "yz":
		x = pos[0]
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
	"winding_vol": opt_loss_winding_volume.winding_volume_loss,
	"normal": opt_loss_dir.normal_loss,
	"pred_dt": lambda *, res: opt_loss_pred_dt.pred_dt_loss(res=res, sqrt=False),
}


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_vis_obj(
	model_path: str,
	data_path: str,
	output_dir: str | None = None,
	*,
	slices: list[str],
	channels: list[str],
	losses: list[str],
	include_mesh: bool,
	include_connections: bool,
	winding_volume_path: str | None = None,
	stats_json: str | None = None,
	erode_valid_mask: int = 0,
	device: str = "cuda",
) -> None:
	"""Export multi-layer OBJ visualization files and/or loss statistics."""
	dev = torch.device(device)
	out = Path(output_dir) if output_dir is not None else None
	if out is not None:
		out.mkdir(parents=True, exist_ok=True)

	# Load model
	print(f"[export_vis] loading model from {model_path}", flush=True)
	st = torch.load(model_path, map_location=dev, weights_only=False)
	mdl = fit_model.Model3D.from_checkpoint(st, device=dev)

	# Restore winding auto-offset from checkpoint if available
	if "_winding_offset_" in st:
		opt_loss_winding_volume._winding_offset = float(st["_winding_offset_"])
		opt_loss_winding_volume._winding_direction = int(st["_winding_direction_"])
		print(f"[export_vis] restored winding auto_offset={opt_loss_winding_volume._winding_offset}, "
			  f"direction={opt_loss_winding_volume._winding_direction}", flush=True)

	# Load data (auto-crop around mesh bbox)
	print(f"[export_vis] loading data from {data_path}", flush=True)
	data = fit_data.load_3d_streaming(path=data_path, device=dev)

	# Load winding volume if provided
	if winding_volume_path is not None:
		import dataclasses
		print(f"[export_vis] loading winding volume from {winding_volume_path}", flush=True)
		# Derive crop from data's origin/spacing/size to match the loaded CT crop
		Z_d, Y_d, X_d = data.size
		ox, oy, oz = data.origin_fullres
		sx, sy, sz = data.spacing
		crop_wv = (int(ox), int(oy), int(oz),
				   int(X_d * sx), int(Y_d * sy), int(Z_d * sz))
		wv_t, wv_min, wv_max = fit_data.load_winding_volume(
			path=winding_volume_path, device=dev,
			crop=crop_wv, downscale=sx,
		)
		data = dataclasses.replace(data, winding_volume=wv_t,
					winding_min=wv_min, winding_max=wv_max)
		wv_np = wv_t.squeeze().cpu().numpy()
		print(f"[export_vis] winding volume shape: {wv_t.shape}"
			  f"  min={float(wv_np.min()):.3f} max={float(wv_np.max()):.3f}", flush=True)
		del wv_np
		if "winding_vol" not in losses:
			losses = list(losses) + ["winding_vol"]

	# Prefetch chunks for streaming mode
	if data.sparse_caches:
		with torch.no_grad():
			_xyz_lr_pf = mdl._grid_xyz()
			_xyz_hr_pf = mdl._grid_xyz_hr(_xyz_lr_pf)
		for _cache in data.sparse_caches.values():
			_sp = data._spacing_for(_cache.channels[0])
			_cache.prefetch(_xyz_hr_pf, data.origin_fullres, _sp)
		for _cache in data.sparse_caches.values():
			_cache.sync()

	# Forward pass
	print("[export_vis] running forward pass", flush=True)
	with torch.no_grad():
		res = mdl(data)

	xyz_lr = res.xyz_lr.detach().cpu().numpy()  # (D, Hm, Wm, 3)
	D, Hm, Wm, _ = xyz_lr.shape
	print(f"[export_vis] mesh shape: D={D}, Hm={Hm}, Wm={Wm}", flush=True)

	# Per-vertex validity: grad_mag > 0 at vertex positions
	with torch.no_grad():
		sampled_gm = data.grid_sample_fullres(res.xyz_lr.detach()).grad_mag  # (D,1,Hm,Wm)
	vert_valid = (sampled_gm.squeeze(1).cpu().numpy() > 0).reshape(-1)  # (D*Hm*Wm,)
	n_valid = int(vert_valid.sum())
	print(f"[export_vis] valid vertices: {n_valid}/{len(vert_valid)}", flush=True)

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
	if include_mesh and out is not None:
		print("[export_vis] writing mesh.obj", flush=True)
		verts = xyz_lr.reshape(-1, 3)
		faces = _triangulate_grid(D, Hm, Wm)
		verts_f, faces_f, _ = _filter_mesh(verts, faces, vert_valid)
		_write_obj_mesh(out / "mesh.obj", verts_f, faces_f, uvs=None, mtl_name=None)
		# Unfiltered mesh (all vertices, no validity masking)
		print("[export_vis] writing mesh_full.obj", flush=True)
		_write_obj_mesh(out / "mesh_full.obj", verts, faces, uvs=None, mtl_name=None)

		# Validity mask on full mesh: green = valid (grad_mag > 0), red = invalid
		print("[export_vis] writing mesh_validity.obj", flush=True)
		uvs_full = _mesh_uvs(D, Hm, Wm)
		validity_stacked = vert_valid.reshape(D * Hm, Wm).astype(np.uint8)
		rgb_validity = np.zeros((D * Hm, Wm, 3), dtype=np.uint8)
		rgb_validity[validity_stacked > 0] = [0, 200, 0]     # green = valid
		rgb_validity[validity_stacked == 0] = [200, 0, 0]     # red = invalid
		_write_png(out / "mesh_validity.png", rgb_validity)
		_write_mtl(out / "mesh_validity.mtl", "mesh_validity", "mesh_validity.png")
		_write_obj_mesh(out / "mesh_validity.obj", verts, faces, uvs_full, "mesh_validity")

	# ------ Connections ------
	if include_connections and out is not None:
		print("[export_vis] writing connections.obj", flush=True)
		xy_conn = res.xy_conn.detach().cpu().numpy()  # (D, Hm, Wm, 3, 3)
		mask_conn = res.mask_conn.detach().cpu().numpy()  # (D, 1, Hm, Wm, 3)
		vv = vert_valid.reshape(D, Hm, Wm)
		conn_verts: list[np.ndarray] = []
		conn_lines: list[tuple[int, int]] = []
		vi = 0
		for d in range(D):
			for h in range(Hm):
				for w in range(Wm):
					if not vv[d, h, w]:
						continue
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

	# ------ Volume slices (start, center, end in one OBJ per plane+channel) ------
	_SLICE_POSITIONS = [("start", 0.1), ("mid", 0.5), ("end", 0.9)]
	if out is not None:
		for plane in slices:
			for channel in channels:
				vol = getattr(data, channel, None)
				if vol is None:
					print(f"[export_vis] skipping slice {plane}/{channel} (not in data)", flush=True)
					continue
				obj_name = f"slice_{plane}_{channel}"
				print(f"[export_vis] writing {obj_name} (3 positions)", flush=True)
				quads: list[tuple[np.ndarray, str]] = []
				materials: list[tuple[str, str]] = []
				for pos_label, frac in _SLICE_POSITIONS:
					mat_name = f"{obj_name}_{pos_label}"
					png_name = f"{mat_name}.png"
					tex = _slice_texture(data, plane, channel, frac=frac)
					_write_png(out / png_name, tex)
					vol_arr = getattr(data, channel).squeeze().cpu().numpy()
					sl_idx = _slice_index(plane, vol_arr.shape, frac)
					_write_pfm(out / f"{mat_name}.pfm", _take_slice(vol_arr, plane, sl_idx).astype(np.float32))
					corners = _slice_corners(plane, data_min, data_max, frac=frac)
					quads.append((corners, mat_name))
					materials.append((mat_name, png_name))
				_write_mtl_multi(out / f"{obj_name}.mtl", materials)
				_write_obj_multi_quad(out / f"{obj_name}.obj", quads)

		# ------ Normal direction slices (nx, ny as RGB normal map) ------
		if data.nx is None or data.ny is None:
			print("[export_vis] skipping normal slices (streaming mode, no dense nx/ny)", flush=True)
			nx_arr = ny_arr = None
		else:
			nx_arr = data.nx.squeeze().cpu().numpy().astype(np.float32)   # (Z, Y, X) uint8
			ny_arr = data.ny.squeeze().cpu().numpy().astype(np.float32)   # (Z, Y, X) uint8
		if nx_arr is not None:
			for plane in slices:
				obj_name = f"slice_{plane}_normals"
				print(f"[export_vis] writing {obj_name} (3 positions)", flush=True)
				quads: list[tuple[np.ndarray, str]] = []
				materials: list[tuple[str, str]] = []
				for pos_label, frac in _SLICE_POSITIONS:
					mat_name = f"{obj_name}_{pos_label}"
					png_name = f"{mat_name}.png"
					idx = _slice_index(plane, nx_arr.shape, frac)
					nx_slc = _take_slice(nx_arr, plane, idx)  # (H, W) 0-255, 128=zero
					ny_slc = _take_slice(ny_arr, plane, idx)
					# Decode to [-1, 1]
					nxf = (nx_slc - 128.0) / 127.0
					nyf = (ny_slc - 128.0) / 127.0
					nzf = np.sqrt(np.clip(1.0 - nxf**2 - nyf**2, 0.0, 1.0))
					# Standard normal map encoding: [0,1] → [0,255]
					rgb = np.stack([
						((nxf * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8),
						((nyf * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8),
						((nzf * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8),
					], axis=-1)
					_write_png(out / png_name, rgb[::-1])
					_write_pfm(out / f"{mat_name}_nx.pfm", nxf.astype(np.float32))
					_write_pfm(out / f"{mat_name}_ny.pfm", nyf.astype(np.float32))
					corners = _slice_corners(plane, data_min, data_max, frac=frac)
					quads.append((corners, mat_name))
					materials.append((mat_name, png_name))
				_write_mtl_multi(out / f"{obj_name}.mtl", materials)
				_write_obj_multi_quad(out / f"{obj_name}.obj", quads)

		# ------ Validity mask slices (grad_mag > 0 as green/red) ------
		if data.grad_mag is None:
			print("[export_vis] skipping validity slices (streaming mode, no dense grad_mag)", flush=True)
			gm_arr = None
		else:
			gm_arr = data.grad_mag.squeeze().cpu().numpy()  # (Z, Y, X)
		if gm_arr is not None:
			for plane in slices:
				obj_name = f"slice_{plane}_validity"
				print(f"[export_vis] writing {obj_name} (3 positions)", flush=True)
				quads: list[tuple[np.ndarray, str]] = []
				materials: list[tuple[str, str]] = []
				for pos_label, frac in _SLICE_POSITIONS:
					mat_name = f"{obj_name}_{pos_label}"
					png_name = f"{mat_name}.png"
					idx = _slice_index(plane, gm_arr.shape, frac)
					slc = _take_slice(gm_arr, plane, idx)  # (H, W)
					rgb = np.zeros((*slc.shape, 3), dtype=np.uint8)
					rgb[slc > 0] = [0, 200, 0]      # green = valid
					rgb[slc == 0] = [200, 0, 0]      # red = invalid
					_write_png(out / png_name, rgb[::-1])
					corners = _slice_corners(plane, data_min, data_max, frac=frac)
					quads.append((corners, mat_name))
					materials.append((mat_name, png_name))
				_write_mtl_multi(out / f"{obj_name}.mtl", materials)
				_write_obj_multi_quad(out / f"{obj_name}.obj", quads)

	# ------ Loss maps & stats ------
	loss_stats = {}
	if losses:
		if out is not None:
			verts_flat_loss = xyz_lr.reshape(-1, 3)
			faces_loss = _triangulate_grid(D, Hm, Wm)
			uvs_loss = _mesh_uvs(D, Hm, Wm)
			verts_flat_loss, faces_loss, uvs_loss = _filter_mesh(
				verts_flat_loss, faces_loss, vert_valid, uvs_loss)

		for loss_name in losses:
			print(f"[export_vis] computing loss: {loss_name}", flush=True)
			loss_fn = _LOSS_FUNCS[loss_name]
			with torch.no_grad():
				_, lms, masks = loss_fn(res=res)

			lm = lms[0].detach().cpu().numpy()  # (D, 1, H', W')
			mask = masks[0].detach().cpu().numpy()

			# Collect stats from raw loss map
			lm_raw = lm.squeeze()
			mask_raw = mask.squeeze()
			valid_vals = lm_raw[mask_raw > 0.5]
			if len(valid_vals) > 0:
				loss_stats[loss_name] = {
					"avg": float(valid_vals.mean()),
					"max": float(valid_vals.max()),
					"n_valid": int(len(valid_vals)),
				}
			else:
				loss_stats[loss_name] = {
					"avg": 0.0,
					"max": 0.0,
					"n_valid": 0,
				}

			# Visual output
			if out is not None:
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
				_write_obj_mesh(out / f"{obj_name}.obj", verts_flat_loss, faces_loss, uvs_loss, obj_name)

	# ------ Winding volume diagnostics ------
	if data.winding_volume is not None and out is not None:
		# Winding volume slices (start, center, end in one OBJ per plane)
		for plane in slices:
			obj_name = f"slice_{plane}_winding"
			print(f"[export_vis] writing {obj_name} (3 positions)", flush=True)
			quads: list[tuple[np.ndarray, str]] = []
			materials: list[tuple[str, str]] = []
			for pos_label, frac in _SLICE_POSITIONS:
				mat_name = f"{obj_name}_{pos_label}"
				png_name = f"{mat_name}.png"
				tex = _slice_winding_volume(data, plane, frac=frac)
				_write_png(out / png_name, tex)
				wv_arr = data.winding_volume.squeeze().cpu().numpy()
				sl_idx = _slice_index(plane, wv_arr.shape, frac)
				_write_pfm(out / f"{mat_name}.pfm", _take_slice(wv_arr, plane, sl_idx).astype(np.float32))
				corners = _slice_corners(plane, data_min, data_max, frac=frac)
				quads.append((corners, mat_name))
				materials.append((mat_name, png_name))
			_write_mtl_multi(out / f"{obj_name}.mtl", materials)
			_write_obj_multi_quad(out / f"{obj_name}.obj", quads)

		# Diagnostic meshes: winding value and winding mask
		print("[export_vis] computing winding diagnostics", flush=True)
		with torch.no_grad():
			sampled_wv = opt_loss_winding_volume._sample_winding_volume(res=res)  # (D, 1, Hm, Wm)

		verts_flat = xyz_lr.reshape(-1, 3)
		faces = _triangulate_grid(D, Hm, Wm)
		uvs = _mesh_uvs(D, Hm, Wm)
		verts_diag, faces_diag, uvs_diag = _filter_mesh(verts_flat, faces, vert_valid, uvs)

		# Winding value map: viridis colormapped sampled winding volume
		wv_vals = sampled_wv.squeeze(1).cpu().numpy()  # (D, Hm, Wm)
		wv_stacked = wv_vals.reshape(-1, Wm)  # (D*Hm, Wm)
		tex_h, tex_w = D * Hm, Wm
		vmin_wv, vmax_wv = float(wv_stacked.min()), float(wv_stacked.max())
		if vmax_wv - vmin_wv < 1e-8:
			vmax_wv = vmin_wv + 1.0
		normed_wv = (wv_stacked - vmin_wv) / (vmax_wv - vmin_wv)
		rgb_wv = _viridis(normed_wv)
		_write_png(out / "winding_value.png", rgb_wv)
		_write_mtl(out / "winding_value.mtl", "winding_value", "winding_value.png")
		_write_obj_mesh(out / "winding_value.obj", verts_diag, faces_diag, uvs_diag, "winding_value")
		print(f"[export_vis] winding value range: [{vmin_wv:.2f}, {vmax_wv:.2f}]", flush=True)

		# Winding mask map: green = valid (mask_lr), red = invalid
		mask_combined = res.mask_lr.squeeze(1).cpu().numpy()  # (D, Hm, Wm)
		mask_stacked = mask_combined.reshape(-1, Wm)  # (D*Hm, Wm)
		rgb_mask = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
		rgb_mask[mask_stacked > 0.5] = [0, 200, 0]    # green = valid
		rgb_mask[mask_stacked <= 0.5] = [100, 0, 0]    # dark red = invalid
		_write_png(out / "winding_mask.png", rgb_mask)
		_write_mtl(out / "winding_mask.mtl", "winding_mask", "winding_mask.png")
		_write_obj_mesh(out / "winding_mask.obj", verts_diag, faces_diag, uvs_diag, "winding_mask")

	# ------ Stats JSON ------
	if stats_json is not None:
		stats = {
			"losses": loss_stats,
			"mesh": {
				"D": D,
				"Hm": Hm,
				"Wm": Wm,
				"n_valid_verts": n_valid,
				"n_total_verts": int(D * Hm * Wm),
			},
		}
		stats_path = Path(stats_json)
		stats_path.parent.mkdir(parents=True, exist_ok=True)
		stats_path.write_text(json.dumps(stats, indent=2))
		print(f"[export_vis] wrote stats to {stats_json}", flush=True)

	print(f"[export_vis] done.", flush=True)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Export OBJ visualization of a fitted model")
	parser.add_argument("--model", required=True, help="Path to model.pt checkpoint")
	parser.add_argument("--input", required=True, help="Path to lasagna normals zarr")
	parser.add_argument("--output-dir", default=None, help="Output directory for OBJ/MTL/PNG files")
	parser.add_argument("--stats-json", default=None, help="Write per-loss statistics to this JSON file")
	parser.add_argument("--slices", nargs="*", default=["xy", "xz", "yz"],
						choices=["xy", "xz", "yz"], help="Volume slice planes (default: xy xz yz)")
	parser.add_argument("--channels", nargs="*", default=["cos", "pred_dt"],
						help="Volume channels to slice (default: cos pred_dt)")
	parser.add_argument("--losses", nargs="*", default=["normal", "step", "pred_dt"],
						help="Loss maps to export (default: normal step pred_dt)")
	parser.add_argument("--winding-volume", default=None,
						help="Path to winding volume zarr (enables winding diagnostics)")
	parser.add_argument("--no-mesh", action="store_true", help="Skip mesh export")
	parser.add_argument("--no-connections", action="store_true", help="Skip connection lines")
	parser.add_argument("--erode-valid-mask", default="auto",
						help="Erode grad_mag validity mask by N voxels (default: auto = read from checkpoint)")
	parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
	args = parser.parse_args()

	if args.output_dir is None and args.stats_json is None:
		parser.error("At least one of --output-dir or --stats-json must be given")

	# Resolve erode-valid-mask: "auto" reads from checkpoint, integer overrides
	erode_val = args.erode_valid_mask
	if erode_val == "auto":
		st = torch.load(args.model, map_location="cpu", weights_only=False)
		fit_cfg = st.get("_fit_config_", {})
		fit_args = fit_cfg.get("args", {}) if isinstance(fit_cfg, dict) else {}
		erode_val = int(fit_args.get("erode-valid-mask", fit_args.get("erode_valid_mask", 0)))
		if erode_val > 0:
			print(f"[export_vis] auto-detected erode-valid-mask={erode_val} from checkpoint", flush=True)
	else:
		erode_val = int(erode_val)

	export_vis_obj(
		model_path=args.model,
		data_path=args.input,
		output_dir=args.output_dir,
		slices=args.slices,
		channels=args.channels,
		losses=args.losses,
		include_mesh=not args.no_mesh,
		include_connections=not args.no_connections,
		winding_volume_path=args.winding_volume,
		stats_json=args.stats_json,
		erode_valid_mask=erode_val,
		device=args.device,
	)
