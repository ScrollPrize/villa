"""
High-level tracing helpers for Volume Cartographer.
"""

from __future__ import annotations

import collections.abc
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import zarr

from . import vc_tracing as _core
from .quadmesh import (
    QuadMesh,
    load_quadmesh,
    quadmesh_to_numpy,
    quadmesh_to_open3d,
    save_quadmesh,
)

_DEFAULT_VOXEL_SIZE = object()

__all__ = [
    "grow_seg_from_seed",
    "GrowSegResult",
    "GrowSegVoxelization",
    "QuadMesh",
    "load_quadmesh",
    "save_quadmesh",
    "quadmesh_to_numpy",
    "quadmesh_to_open3d",
]


class GrowSegVoxelization:
    """Voxelized representation of a traced surface."""

    __slots__ = ("points", "scale")

    def __init__(self, points: np.ndarray, scale: np.ndarray) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.scale = np.ascontiguousarray(scale, dtype=np.float32)

    def _infer_shape(self) -> tuple[int, int, int]:
        if self.points.size == 0:
            return (0, 0, 0)

        x_coords = np.clip(self.points[:, 0] * self.scale[0], 0.0, None)
        y_coords = np.clip(self.points[:, 1] * self.scale[1], 0.0, None)
        z_coords = np.clip(self.points[:, 2], 0.0, None)

        x_extent = int(np.ceil(x_coords.max())) + 1
        y_extent = int(np.ceil(y_coords.max())) + 1
        z_extent = int(np.ceil(z_coords.max())) + 1

        return (max(1, z_extent), max(1, y_extent), max(1, x_extent))

    def to_volume(
        self,
        *,
        reference_volume: np.ndarray | None = None,
        shape: Sequence[int] | None = None,
        dtype: np.dtype[Any] = np.uint8,
        fill_value: int = 255,
        clamp: bool = True,
    ) -> np.ndarray:
        """
        Rasterise the voxelised point cloud into a dense volume.
        """
        if reference_volume is not None:
            volume = np.zeros_like(reference_volume, dtype=dtype)
            target_shape = volume.shape
        else:
            if shape is None:
                target_shape = self._infer_shape()
            else:
                target_shape = tuple(int(value) for value in shape)
            volume = np.zeros(target_shape, dtype=dtype)

        if self.points.size == 0 or any(dim == 0 for dim in volume.shape):
            return volume

        x_idx = np.rint(self.points[:, 0] * self.scale[0]).astype(np.int64)
        y_idx = np.rint(self.points[:, 1] * self.scale[1]).astype(np.int64)
        z_idx = np.rint(np.clip(self.points[:, 2], 0.0, None)).astype(np.int64)

        if clamp:
            z_idx = np.clip(z_idx, 0, volume.shape[0] - 1)
            y_idx = np.clip(y_idx, 0, volume.shape[1] - 1)
            x_idx = np.clip(x_idx, 0, volume.shape[2] - 1)

        volume[(z_idx, y_idx, x_idx)] = fill_value
        return volume


class GrowSegResult(collections.abc.Mapping):
    """Mapping-like container for tracer results with convenience helpers."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload: dict[str, Any] = dict(payload)

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)

    def voxelize(self) -> GrowSegVoxelization:
        """
        Generate a voxel-aligned point cloud for the traced surface.
        """
        surface = self._payload.get("surface")
        if surface is None:
            raise ValueError("result payload does not contain a surface")

        points = np.asarray(surface.points(), dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("surface.points() must return an array shaped (rows, cols, 3)")

        finite_mask = np.isfinite(points).all(axis=-1)
        populated_mask = np.all(points > -0.5, axis=-1)
        valid_points = points[finite_mask & populated_mask]

        mesh_scale = np.asarray(surface.scale, dtype=np.float32).reshape(-1)
        if mesh_scale.size != 2:
            raise ValueError("surface.scale must contain exactly two elements")

        bounds = surface.bounds
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError("surface.bounds must return a sequence of length 2")

        try:
            dense = surface.gen(
                (int(bounds[0]), int(bounds[1])),
                scale=1.0,
                offset=(-bounds[1] / 2.0, -bounds[0] / 2.0, 0.0),
            )
        except Exception:
            dense_points = np.empty((0, 3), dtype=np.float32)
        else:
            dense_array = np.asarray(dense, dtype=np.float32)
            if dense_array.ndim == 3 and dense_array.shape[-1] == 3:
                dense_mask = np.isfinite(dense_array).all(axis=-1)
                dense_points = dense_array[dense_mask]
            else:
                dense_points = np.empty((0, 3), dtype=np.float32)

        if dense_points.size > 0:
            export_points = dense_points
            export_scale = np.array([1.0, 1.0], dtype=np.float32)
        else:
            export_points = valid_points
            export_scale = mesh_scale

        return GrowSegVoxelization(export_points, export_scale)

try:
    _REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:  # pragma: no cover - fallback for packaged installs
    _REPO_ROOT = Path(__file__).resolve().parent


def _normalize_origin(origin: Sequence[float] | None) -> Sequence[float] | None:
    if origin is None:
        return (0.0, 0.0, 0.0)
    if len(origin) != 3:
        raise ValueError("origin must have exactly three elements")

    # Convert user-provided (z, y, x) coordinates into the tracer's expected (x, y, z).
    z, y, x = (float(value) for value in origin)
    return (x, y, z)


def _normalize_volume_arg(
    volume: Any, voxel_size: float | object | None
) -> tuple[Any, float | None]:
    if isinstance(volume, np.ndarray):
        array = np.asarray(volume)
        if array.ndim != 3:
            raise ValueError("NumPy volume must be 3-dimensional (z, y, x)")
        if array.dtype != np.dtype("uint8"):
            raise TypeError(
                "NumPy volume must have dtype numpy.uint8; convert explicitly before tracing",
            )
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        if voxel_size is _DEFAULT_VOXEL_SIZE or voxel_size is None:
            resolved_voxel_size = 7.91
        else:
            resolved_voxel_size = float(voxel_size)
        if not (resolved_voxel_size > 0.0):
            raise ValueError("voxel_size must be positive")

        return (array, resolved_voxel_size), None

    if isinstance(volume, (str, os.PathLike)):
        if voxel_size is _DEFAULT_VOXEL_SIZE or voxel_size is None:
            return volume, None
        return volume, float(voxel_size)

    if isinstance(volume, tuple):
        raise TypeError("volume tuples are no longer supported; pass a NumPy array instead")

    raise TypeError("volume must be a path-like object or a NumPy array")


def _probe_vc_gen_normalgrids(executable: Any | None) -> Path | None:
    if executable is not None:
        candidate = Path(os.fspath(executable))
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"vc_gen_normalgrids executable '{candidate}' does not exist")

    def _safe_resolve(path: Path) -> Path:
        try:
            return path.resolve(strict=False)
        except Exception:
            return path

    search_roots: list[Path] = []
    seen_roots: set[Path] = set()

    def _add_root(path: Path | None) -> None:
        if path is None:
            return
        resolved = _safe_resolve(path)
        if resolved in seen_roots:
            return
        seen_roots.add(resolved)
        search_roots.append(path)

    _add_root(_REPO_ROOT)

    cwd = Path.cwd()
    _add_root(cwd)
    for parent in cwd.parents:
        _add_root(parent)

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        _add_root(parent)

    try:
        core_path = Path(_core.__file__).resolve()
    except Exception:  # pragma: no cover - mirrors packaging edge cases
        core_path = None
    if core_path is not None:
        for parent in core_path.parents:
            _add_root(parent)

    seen_candidates: set[Path] = set()
    candidate_paths: list[Path] = []

    def _add_candidate(path: Path) -> None:
        resolved = _safe_resolve(path)
        if resolved in seen_candidates:
            return
        seen_candidates.add(resolved)
        candidate_paths.append(path)

    def _enqueue_from_root(root: Path) -> None:
        _add_candidate(root / "vc_gen_normalgrids")
        _add_candidate(root / "bin" / "vc_gen_normalgrids")
        _add_candidate(root / "build" / "bin" / "vc_gen_normalgrids")
        _add_candidate(root / "cmake-build-debug" / "bin" / "vc_gen_normalgrids")

    for root in search_roots:
        _enqueue_from_root(root)

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

    discovered = shutil.which("vc_gen_normalgrids")
    if discovered:
        return Path(discovered)

    return None


def _write_numpy_volume_to_zarr(
    array_like: Any,
    root: Path,
    voxel_size: float,
    chunk_shape: Sequence[int] | None,
) -> None:
    array = np.asarray(array_like, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError("NumPy volume must be 3-dimensional (z, y, x)")
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    if chunk_shape is not None:
        if len(chunk_shape) != 3:
            raise ValueError("chunk_shape must contain exactly three integers")
        chunks = tuple(int(max(1, value)) for value in chunk_shape)
    else:
        chunks = (64, 64, 64)

    root.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(root))
    root_group = zarr.group(store=store, overwrite=True)
    dataset = root_group.require_dataset(
        "0",
        shape=array.shape,
        chunks=chunks,
        dtype=array.dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=1, shuffle=zarr.Blosc.NOSHUFFLE),
    )
    dataset[:] = array

    metadata = {"voxelsize": float(voxel_size)}
    (root / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@contextlib.contextmanager
def _volume_path_context(volume: Any) -> Iterator[Path]:
    if isinstance(volume, (str, os.PathLike)):
        yield Path(os.fspath(volume))
        return

    if not isinstance(volume, tuple):
        raise TypeError(
            "volume must be a path-like object or a NumPy tuple when requesting "
            "automatic normal grid generation",
        )

    if len(volume) < 2:
        raise ValueError("NumPy volume tuples must contain at least (array, voxel_size)")

    array = volume[0]
    voxel_size = float(volume[1])
    chunk_shape = volume[2] if len(volume) >= 3 else None

    temp_dir = Path(tempfile.mkdtemp(prefix="vc_volume_for_normals_"))
    try:
        _write_numpy_volume_to_zarr(array, temp_dir, voxel_size, chunk_shape)
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextlib.contextmanager
def _normal_grid_context(normal_grid: Any, volume: Any) -> Iterator[Any]:
    if normal_grid is None or isinstance(normal_grid, (str, os.PathLike)):
        yield normal_grid
        return

    if not isinstance(normal_grid, Mapping):
        raise TypeError(
            "normal_grid must be None, a path-like object, or a mapping describing "
            "generation parameters",
        )

    config = dict(normal_grid)
    mode = config.get("mode", "generate")
    if mode != "generate":
        raise ValueError(f"Unsupported normal_grid mode '{mode}' (only 'generate' is supported)")

    spiral_step = float(config.get("spiral_step", 20.0))
    grid_step = int(config.get("grid_step", 64))
    extra_args = list(config.get("extra_args", ()))

    keep_output = bool(config.get("keep_output", True))
    output_dir_arg = config.get("output_dir")

    if output_dir_arg is None:
        normal_grid_dir = Path(tempfile.mkdtemp(prefix="vc_normal_grid_"))
        created_output_dir = True
    else:
        normal_grid_dir = Path(os.fspath(output_dir_arg))
        normal_grid_dir.mkdir(parents=True, exist_ok=True)
        created_output_dir = False

    executable = _probe_vc_gen_normalgrids(config.get("executable"))
    if executable is None:
        raise FileNotFoundError(
            "vc_gen_normalgrids executable could not be located. "
            "Build the CLI apps or provide the 'executable' path in the normal_grid mapping.",
        )

    with _volume_path_context(volume) as volume_path:
        command = [
            str(executable),
            "generate",
            "--input",
            str(volume_path),
            "--output",
            str(normal_grid_dir),
            "--spiral-step",
            str(spiral_step),
            "--grid-step",
            str(grid_step),
            *[str(arg) for arg in extra_args],
        ]

        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:  # pragma: no cover - mirrors subprocess behaviour
            raise FileNotFoundError(
                f"Failed to execute vc_gen_normalgrids at '{executable}': {exc}",
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"vc_gen_normalgrids exited with status {exc.returncode}; "
                f"command: {' '.join(command)}",
            ) from exc

    try:
        yield str(normal_grid_dir)
    finally:
        if created_output_dir and not keep_output:
            shutil.rmtree(normal_grid_dir, ignore_errors=True)


def grow_seg_from_seed(
    volume: Any,
    params: Mapping[str, Any] | MutableMapping[str, Any] | None,
    origin: Sequence[float] | None = None,
    *,
    resume: Any | None = None,
    corrections: Any | None = None,
    output_dir: Any | None,
    meta: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
    cache_root: Any | None = None,
    voxel_size: float | None | object = _DEFAULT_VOXEL_SIZE,
    normal_grid: Any | None = None,
) -> dict[str, Any]:
    """
    Run the grow_seg_from_seed tracer.

    Parameters
    ----------
    volume:
        Either a path to an OME-Zarr volume or a NumPy array in Z/Y/X order with
        ``dtype`` exactly ``numpy.uint8``.
    params:
        Algorithm configuration as a JSON-serialisable mapping. Required.
    origin:
        Optional origin in voxels specified as ``(z, y, x)``. Defaults to (0, 0, 0).
    resume:
        Existing surface path to resume from.
    corrections:
        Either a path to a corrections JSON file (same structure used by the CLI)
        or a JSON-serialisable mapping following that schema. Only valid when
        resuming from an existing surface.
    output_dir:
        Target directory where tracer artefacts are written. Required.
    meta:
        Additional metadata stored alongside the generated surface.
    cache_root:
        Override cache root directory utilised by the tracer (takes precedence
        over the tuple's ``cache_dir`` entry when both are provided).
    voxel_size:
        Override the voxel size stored for the result. When ``volume`` is a NumPy
        array this defaults to ``7.91`` unless explicitly overridden. When
        ``volume`` is a path, omit the argument (or pass ``None``) to rely on the
        metadata stored alongside the dataset.
    normal_grid:
        Either a path to a precomputed normal grid volume or a mapping that
        requests on-the-fly generation via ``vc_gen_normalgrids``. Supported
        mapping keys:

        - ``mode``: only ``"generate"`` is recognised (default).
        - ``spiral_step`` (float) and ``grid_step`` (int): override CLI defaults.
        - ``output_dir``: target directory for the generated grids; defaults to a
          temporary directory.
        - ``keep_output`` (bool): keep the temporary directory after tracing
          (defaults to ``True``).
        - ``executable``: explicit path to the ``vc_gen_normalgrids`` binary.
        - ``extra_args``: additional command-line arguments to append.

        When omitted, tracing relies on any value specified inside ``params``.
        When both this argument and ``params['normal_grid_path']`` are absent, a
        temporary normal grid is generated automatically (requires the
        ``vc_gen_normalgrids`` binary to be discoverable).
    """
    if params is None:
        raise ValueError("params must be provided")
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    if corrections is not None and resume is None:
        raise ValueError("corrections can only be supplied when resume is provided")

    normalized_origin = _normalize_origin(origin)
    volume_arg, voxel_size_arg = _normalize_volume_arg(volume, voxel_size)
    try:
        params_view = dict(params)
    except TypeError:
        params_view = {}

    effective_normal_grid = normal_grid
    if effective_normal_grid is None:
        normal_grid_path = params_view.get("normal_grid_path")
        if not normal_grid_path:
            raw_step = params_view.get("step_size", 20.0)
            try:
                step_size = float(raw_step)
            except (TypeError, ValueError):
                step_size = 20.0
            effective_normal_grid = {
                "mode": "generate",
                "spiral_step": step_size,
                "keep_output": False,
            }

    with _normal_grid_context(effective_normal_grid, volume_arg) as resolved_normal_grid:
        result_payload = _core.grow_seg_from_seed(
            volume_arg,
            params,
            normalized_origin,
            resume=resume,
            corrections=corrections,
            output_dir=output_dir,
            meta=meta,
            cache_root=cache_root,
            voxel_size=voxel_size_arg,
            normal_grid=resolved_normal_grid,
        )

    return GrowSegResult(result_payload)
