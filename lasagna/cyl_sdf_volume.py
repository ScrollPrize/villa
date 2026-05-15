from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_CYL_OUTSIDE_GRID_STEP = 64.0
DEFAULT_CYL_OUTSIDE_CHUNK_SIZE = 8
DEFAULT_CYL_OUTSIDE_DEEP_INTERP_CHUNKS = 10.0
DEFAULT_CYL_OUTSIDE_DEEP_BLEND_CHUNKS = 2.0

_ext_module = None


@dataclass(frozen=True)
class CylOutsideVolume:
	volume: torch.Tensor          # (1, Z, Y, X) uint8
	origin: tuple[float, float, float]
	spacing: tuple[float, float, float]
	shape: tuple[int, int, int]   # (Z, Y, X)
	depth_max: float


def libigl_headers_available() -> bool:
	"""Return whether the common libigl/Eigen header locations are visible."""
	return _find_header("igl/signed_distance.h") is not None and _find_header("Eigen/Dense") is not None


def _include_paths() -> list[str]:
	paths: list[str] = []
	for env_name in ("LIBIGL_INCLUDE_DIR", "IGL_INCLUDE_DIR"):
		value = os.environ.get(env_name)
		if value:
			paths.append(value)
	for env_name in ("EIGEN_INCLUDE_DIR", "EIGEN3_INCLUDE_DIR"):
		value = os.environ.get(env_name)
		if value:
			paths.append(value)
	for path in ("/usr/include", "/usr/local/include", "/opt/homebrew/include", "/usr/include/eigen3", "/usr/local/include/eigen3", "/opt/homebrew/include/eigen3"):
		paths.append(path)
	out: list[str] = []
	seen: set[str] = set()
	for path in paths:
		path = str(Path(path))
		if path not in seen and Path(path).exists():
			seen.add(path)
			out.append(path)
	return out


def _find_header(header: str) -> str | None:
	for path in _include_paths():
		if (Path(path) / header).exists():
			return path
	return None


def _extension_build_dir() -> Path:
	root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/lasagna_torch_extensions"))
	root.mkdir(parents=True, exist_ok=True)
	return root


def _get_ext_module():
	global _ext_module
	if _ext_module is not None:
		return _ext_module
	if not libigl_headers_available():
		raise RuntimeError(
			"cyl_outside requires libigl and Eigen headers to build the previous-shell "
			"inside-depth field. Install libigl headers and Eigen, or set "
			"LIBIGL_INCLUDE_DIR and EIGEN_INCLUDE_DIR so igl/signed_distance.h and "
			"Eigen/Dense are on the include path."
		)
	src = str(Path(__file__).with_name("cyl_sdf_volume_ext.cpp"))
	try:
		from torch.utils.cpp_extension import load as _load_ext

		_ext_module = _load_ext(
			name="cyl_sdf_volume_ext",
			sources=[src],
			extra_include_paths=_include_paths(),
			extra_cflags=["-O3", "-DGLOG_USE_GLOG_EXPORT"],
			build_directory=str(_extension_build_dir()),
			verbose=False,
		)
	except Exception as exc:
		raise RuntimeError(
			"failed to build cyl_outside libigl extension. Ensure libigl and Eigen "
			"headers are installed and visible via LIBIGL_INCLUDE_DIR and "
			"EIGEN_INCLUDE_DIR; the extension includes igl/signed_distance.h and "
			"Eigen/Dense."
		) from exc
	return _ext_module


def capped_shell_mesh(shell_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Build a watertight, outward-oriented capped mesh from a periodic shell.

	The shell is expected as (H, W, 3), with W wrapping around the cylinder. Caps
	are fan-triangulated from one center vertex per end row.
	"""
	if shell_xyz.ndim != 3 or int(shell_xyz.shape[-1]) != 3:
		raise ValueError(f"shell_xyz must have shape (H, W, 3), got {tuple(shell_xyz.shape)}")
	H = int(shell_xyz.shape[0])
	W = int(shell_xyz.shape[1])
	if H < 2 or W < 3:
		raise ValueError(f"capped shell mesh requires H>=2 and W>=3, got H={H} W={W}")
	verts = shell_xyz.detach().to(device="cpu", dtype=torch.float64).contiguous().reshape(H * W, 3)
	bottom_center = shell_xyz[0].detach().to(device="cpu", dtype=torch.float64).mean(dim=0, keepdim=True)
	top_center = shell_xyz[-1].detach().to(device="cpu", dtype=torch.float64).mean(dim=0, keepdim=True)
	verts = torch.cat([verts, bottom_center, top_center], dim=0)
	bottom_i = H * W
	top_i = H * W + 1
	faces: list[list[int]] = []
	for h in range(H - 1):
		for w in range(W):
			p00 = h * W + w
			p01 = h * W + ((w + 1) % W)
			p10 = (h + 1) * W + w
			p11 = (h + 1) * W + ((w + 1) % W)
			faces.append([p00, p11, p10])
			faces.append([p00, p01, p11])
	for w in range(W):
		w1 = (w + 1) % W
		faces.append([bottom_i, w1, w])
		faces.append([top_i, (H - 1) * W + w, (H - 1) * W + w1])
	return verts, torch.tensor(faces, dtype=torch.int64)


def default_shell_bbox(
	shell_xyz: torch.Tensor,
	*,
	grid_step: float = DEFAULT_CYL_OUTSIDE_GRID_STEP,
) -> tuple[float, float, float, float, float, float]:
	step = max(1.0e-6, float(grid_step))
	shell_cpu = shell_xyz.detach().to(device="cpu", dtype=torch.float64)
	lo = shell_cpu.reshape(-1, 3).amin(dim=0) - step
	hi = shell_cpu.reshape(-1, 3).amax(dim=0) + step
	return (
		float(lo[0]), float(lo[1]), float(lo[2]),
		float(hi[0]), float(hi[1]), float(hi[2]),
	)


def shape_for_bbox(
	bbox: tuple[float, float, float, float, float, float],
	*,
	grid_step: float,
) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
	step = max(1.0e-6, float(grid_step))
	x0, y0, z0, x1, y1, z1 = (float(v) for v in bbox)
	if x1 < x0 or y1 < y0 or z1 < z0:
		raise ValueError(f"invalid bbox: {bbox}")
	X = max(1, int(math.ceil((x1 - x0) / step)) + 1)
	Y = max(1, int(math.ceil((y1 - y0) / step)) + 1)
	Z = max(1, int(math.ceil((z1 - z0) / step)) + 1)
	return (x0, y0, z0), (Z, Y, X)


def encode_inside_depth(depth: torch.Tensor, depth_max: float) -> torch.Tensor:
	depth_max = float(depth_max)
	if depth_max <= 0.0 or not math.isfinite(depth_max):
		return torch.zeros_like(depth, dtype=torch.uint8)
	q = torch.sqrt((depth.to(dtype=torch.float32) / depth_max).clamp(min=0.0, max=1.0))
	return torch.round(q * 255.0).clamp(min=0.0, max=255.0).to(dtype=torch.uint8)


def decode_inside_depth(encoded: torch.Tensor, depth_max: float) -> torch.Tensor:
	return (encoded.to(dtype=torch.float32) / 255.0).square() * float(depth_max)


def build_previous_shell_inside_depth_volume(
	shell_xyz: torch.Tensor,
	*,
	grid_step: float = DEFAULT_CYL_OUTSIDE_GRID_STEP,
	bbox: tuple[float, float, float, float, float, float] | None = None,
	device: torch.device | str | None = None,
	progress_label: str | None = None,
	threads: int = 0,
	chunk_size: int = DEFAULT_CYL_OUTSIDE_CHUNK_SIZE,
	deep_interp_chunks: float = DEFAULT_CYL_OUTSIDE_DEEP_INTERP_CHUNKS,
	deep_blend_chunks: float = DEFAULT_CYL_OUTSIDE_DEEP_BLEND_CHUNKS,
) -> CylOutsideVolume:
	"""Build a coarse uint8 inside-depth field for the completed previous shell."""
	step = max(1.0e-6, float(grid_step))
	origin, shape = shape_for_bbox(
		default_shell_bbox(shell_xyz, grid_step=step) if bbox is None else bbox,
		grid_step=step,
	)
	verts, faces = capped_shell_mesh(shell_xyz)
	if progress_label:
		print(
			f"[cyl_outside] {progress_label}: loading libigl extension; first run may compile C++",
			flush=True,
		)
	ext_start = time.perf_counter()
	ext = _get_ext_module()
	if progress_label:
		print(
			f"[cyl_outside] {progress_label}: libigl extension ready "
			f"elapsed={time.perf_counter() - ext_start:.2f}s",
			flush=True,
		)
	volume_cpu, depth_max = ext.build_inside_depth_volume(
		verts.contiguous(),
		faces.contiguous(),
		torch.tensor(origin, dtype=torch.float64),
		torch.tensor((step, step, step), dtype=torch.float64),
		torch.tensor(shape, dtype=torch.int64),
		"" if progress_label is None else str(progress_label),
		int(threads),
		int(chunk_size),
		float(deep_interp_chunks),
		float(deep_blend_chunks),
	)
	depth_max = float(depth_max)
	if device is not None:
		volume = volume_cpu.to(device=device, non_blocking=True)
	else:
		volume = volume_cpu
	return CylOutsideVolume(
		volume=volume.contiguous(),
		origin=origin,
		spacing=(step, step, step),
		shape=shape,
		depth_max=depth_max,
	)
