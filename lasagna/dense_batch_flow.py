from __future__ import annotations

import ctypes
import os
import shlex
import subprocess
import threading
from pathlib import Path

import numpy as np


_LIB = None
_LIB_LOCK = threading.Lock()
_AUTOBUILD_ENV = "LASAGNA_DENSE_BATCH_FLOW_AUTOBUILD"


def _candidate_library_paths() -> list[Path]:
	root = Path(__file__).resolve().parent
	return [
		root / "dense_batch_min_cut" / "build" / "libdense_batch_flow.so",
		root / "dense_batch_min_cut" / "build" / "libdense_batch_flow.dylib",
		root / "dense_batch_min_cut" / "build" / "dense_batch_flow.dll",
	]


def _manual_build_message() -> str:
	return (
		"Build it with:\n"
		"  cmake -S lasagna/dense_batch_min_cut -B lasagna/dense_batch_min_cut/build\n"
		"  cmake --build lasagna/dense_batch_min_cut/build --target dense_batch_flow"
	)


def _format_candidate_paths() -> str:
	return "\n".join(f"  {p}" for p in _candidate_library_paths())


def _find_library_path() -> Path | None:
	for path in _candidate_library_paths():
		if path.exists():
			return path
	return None


def _build_commands() -> list[list[str]]:
	source_dir = Path(__file__).resolve().parent / "dense_batch_min_cut"
	build_dir = source_dir / "build"
	return [
		["cmake", "-S", str(source_dir), "-B", str(build_dir)],
		["cmake", "--build", str(build_dir), "--target", "dense_batch_flow"],
	]


def _autobuild_enabled() -> bool:
	return os.environ.get(_AUTOBUILD_ENV, "1") != "0"


def _run_build_command(cmd: list[str]) -> None:
	try:
		result = subprocess.run(
			cmd,
			check=False,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
		)
	except FileNotFoundError as exc:
		commands = "\n".join(f"  {shlex.join(c)}" for c in _build_commands())
		raise RuntimeError(
			"dense_batch_flow auto-build failed because CMake was not found.\n"
			f"Attempted:\n{commands}\n"
			f"{_manual_build_message()}"
		) from exc
	if result.returncode != 0:
		commands = "\n".join(f"  {shlex.join(c)}" for c in _build_commands())
		raise RuntimeError(
			"dense_batch_flow auto-build failed.\n"
			f"Failed command:\n  {shlex.join(cmd)}\n"
			f"Attempted:\n{commands}\n"
			f"Output:\n{result.stdout}"
		)


def _auto_build_library() -> None:
	for cmd in _build_commands():
		_run_build_command(cmd)


def _load_library_from_path(path: Path) -> ctypes.CDLL:
	lib = ctypes.CDLL(str(path))
	lib.dense_batch_flow_grid_u8.argtypes = [
		ctypes.POINTER(ctypes.c_uint8),
		ctypes.c_int,
		ctypes.c_int,
		ctypes.c_int,
		ctypes.c_int,
		ctypes.POINTER(ctypes.c_float),
		ctypes.c_int,
		ctypes.POINTER(ctypes.c_float),
		ctypes.POINTER(ctypes.c_float),
		ctypes.POINTER(ctypes.c_float),
		ctypes.POINTER(ctypes.c_float),
		ctypes.c_int,
		ctypes.c_float,
		ctypes.POINTER(ctypes.c_int),
		ctypes.POINTER(ctypes.c_int),
		ctypes.POINTER(ctypes.c_float),
		ctypes.c_char_p,
		ctypes.c_int,
		ctypes.c_int,
	]
	lib.dense_batch_flow_grid_u8.restype = ctypes.c_int
	return lib


def _load_library() -> ctypes.CDLL:
	global _LIB
	if _LIB is not None:
		return _LIB
	with _LIB_LOCK:
		if _LIB is not None:
			return _LIB
		path = _find_library_path()
		if path is None and _autobuild_enabled():
			_auto_build_library()
			path = _find_library_path()
		if path is None:
			autobuild_note = (
				f"Automatic build is disabled by {_AUTOBUILD_ENV}=0.\n"
				if not _autobuild_enabled()
				else "Automatic build did not produce a loadable library.\n"
			)
			raise RuntimeError(
				"dense_batch_flow library was not found.\n"
				f"{autobuild_note}"
				f"{_manual_build_message()}\n"
				f"Looked for:\n{_format_candidate_paths()}"
			)
		_LIB = _load_library_from_path(path)
		return _LIB


def compute_flow_grid(
	image_u8: np.ndarray,
	*,
	source_xy: tuple[int, int],
	query_xy: np.ndarray,
	verbose: bool = False,
	return_debug: bool = False,
	return_metadata: bool = False,
	grid_step: int = 50,
	backtrack_distance: float = 10.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Run dense source flow and sample it at explicit image-space points.

	image_u8: (H, W) uint8 pred_dt render.
	query_xy: (N, 2) float32 image coordinates, x then y.
	Returns (query_flow, dense_flow), both float32.
	"""
	if image_u8.ndim != 2:
		raise ValueError(f"image_u8 must be 2D, got shape {image_u8.shape}")
	image = np.ascontiguousarray(image_u8, dtype=np.uint8)
	query = np.ascontiguousarray(query_xy, dtype=np.float32)
	if query.ndim != 2 or query.shape[1] != 2:
		raise ValueError(f"query_xy must have shape (N, 2), got {query.shape}")

	height, width = image.shape
	query_flow = np.zeros((query.shape[0],), dtype=np.float32)
	dense_flow = np.zeros((height, width), dtype=np.float32)
	smooth_grid_flow = np.zeros((height, width), dtype=np.float32) if return_debug else None
	graph_edge_flow = np.zeros((height, width, 3), dtype=np.float32) if return_debug else None
	resolved_source_x = ctypes.c_int(-1)
	resolved_source_y = ctypes.c_int(-1)
	resolved_source_capacity = ctypes.c_float(0.0)
	err = ctypes.create_string_buffer(4096)
	lib = _load_library()
	rc = lib.dense_batch_flow_grid_u8(
		image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
		width,
		height,
		int(source_xy[0]),
		int(source_xy[1]),
		query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
		int(query.shape[0]),
		query_flow.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
		dense_flow.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
		(
			smooth_grid_flow.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
			if smooth_grid_flow is not None
			else None
		),
		(
			graph_edge_flow.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
			if graph_edge_flow is not None
			else None
		),
		int(max(1, grid_step)),
		ctypes.c_float(max(0.0, float(backtrack_distance))),
		ctypes.byref(resolved_source_x),
		ctypes.byref(resolved_source_y),
		ctypes.byref(resolved_source_capacity),
		err,
		ctypes.sizeof(err),
		1 if verbose else 0,
	)
	if rc != 0:
		message = err.value.decode("utf-8", errors="replace")
		raise RuntimeError(f"dense_batch_flow failed: {message}")
	metadata = {
		"source_x": int(resolved_source_x.value),
		"source_y": int(resolved_source_y.value),
		"source_capacity": float(resolved_source_capacity.value),
	}
	if return_debug:
		result = (query_flow, dense_flow, smooth_grid_flow, graph_edge_flow)
	else:
		result = (query_flow, dense_flow)
	if return_metadata:
		return (*result, metadata)
	return result
