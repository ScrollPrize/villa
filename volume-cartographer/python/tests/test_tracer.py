import importlib.util
import json
import math
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import numpy as np
import zarr

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

import pytest
try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional dependency only when dumping artifacts
    tifffile = None

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

BUILD_PYTHON = REPO_ROOT / "build" / "python"
if BUILD_PYTHON.exists():
    sys.path.insert(0, str(BUILD_PYTHON))

EXTENSION_DIRS = [
    BUILD_PYTHON / "vc" / "tracing",
    PYTHON_SRC / "vc" / "tracing",
]


def _ensure_tracing_extension_available() -> bool:
    module_name = "vc.tracing.vc_tracing"
    if module_name in sys.modules:
        return True

    for candidate_dir in EXTENSION_DIRS:
        if not candidate_dir.is_dir():
            continue
        shared_objects = sorted(candidate_dir.glob("vc_tracing*.so"))
        cache_tag = getattr(sys.implementation, "cache_tag", None)
        if cache_tag:
            shared_objects.sort(
                key=lambda path: cache_tag in path.stem,
                reverse=True,
            )
        if not shared_objects:
            continue
        spec = importlib.util.spec_from_file_location(module_name, shared_objects[0])
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue
        sys.modules[module_name] = module
        return True

    return False


if not _ensure_tracing_extension_available():
    pytest.skip(
        "vc.tracing.vc_tracing extension is not available; build the Python bindings first",
        allow_module_level=True,
    )


import vc.tracing as tracing  # noqa: E402


def _find_vc_gen_normalgrids() -> Optional[Path]:
    candidates = [
        REPO_ROOT / "build" / "bin" / "vc_gen_normalgrids",
        REPO_ROOT / "cmake-build-debug" / "bin" / "vc_gen_normalgrids",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    discovered = shutil.which("vc_gen_normalgrids")
    return Path(discovered) if discovered else None


def _write_volume_to_zarr(volume: np.ndarray, root: Path, voxel_size: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(root))
    root_group = zarr.group(store=store, overwrite=True)
    arr = root_group.require_dataset(
        "0",
        shape=volume.shape,
        chunks=(64, 64, 64),
        dtype=volume.dtype,
        compressor=zarr.Blosc(cname="zstd", clevel=1, shuffle=zarr.Blosc.NOSHUFFLE),
    )
    arr[:] = volume

    metadata = {"voxelsize": float(voxel_size)}
    (root / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _make_hollow_cylinder_volume(size: int, radius: int, wall_thickness: int) -> np.ndarray:
    """Return a C-contiguous (z, y, x) uint8 volume with a hollow cylinder shell."""
    if wall_thickness <= 0:
        raise ValueError("wall_thickness must be positive")

    center = size // 2
    yy, xx = np.ogrid[:size, :size]
    dist2 = (yy - center) ** 2 + (xx - center) ** 2

    inner_r = max(1, radius - wall_thickness // 2)
    outer_r = radius + math.ceil(wall_thickness / 2)
    mask = (dist2 >= inner_r**2) & (dist2 <= outer_r**2)

    slice_2d = np.zeros((size, size), dtype=np.uint8)
    slice_2d[mask] = 255

    volume = np.zeros((size, size, size), dtype=np.uint8)
    volume[:] = slice_2d
    return volume


def _make_curvilinear_plane_volume(
    size: int,
    thickness: int,
    amplitude: float,
    wavelength: float,
) -> np.ndarray:
    """Return a C-contiguous volume containing a gently curved sheet."""
    if thickness <= 0:
        raise ValueError("thickness must be positive")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")

    center = size / 2.0
    yy, xx = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )
    curved_surface = (
        center
        + amplitude
        * np.sin((xx - center) / wavelength)
        * np.cos((yy - center) / wavelength)
    )

    zz = np.arange(size, dtype=np.float32)[:, None, None]
    mask = np.abs(zz - curved_surface[None, :, :]) <= (thickness / 2.0)

    volume = np.zeros((size, size, size), dtype=np.uint8)
    volume[mask] = 255
    return volume


@pytest.fixture(scope="session")
def tracer_dump_dir() -> Optional[Path]:
    dump_path = os.environ.get("VC_DUMP_TRACER_ARTIFACTS")
    if not dump_path:
        return None
    return Path(dump_path)


def _dump_tracer_debug_artifacts(
    dump_dir: Path,
    volume: np.ndarray,
    valid_points: np.ndarray,
    mesh_scale: np.ndarray,
    label: str,
) -> None:
    dump_dir = dump_dir / label
    dump_dir.mkdir(parents=True, exist_ok=True)

    if tifffile is None:
        warnings.warn(
            "tifffile is not available; skipping tracer artifact dump",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    tifffile.imwrite(dump_dir / f"{label}_volume.tif", volume, compression="zlib")

    voxelized = np.zeros_like(volume, dtype=np.uint8)

    x_idx = np.clip(
        np.rint(valid_points[:, 0] * mesh_scale[0]).astype(int),
        0,
        voxelized.shape[2] - 1,
    )
    y_idx = np.clip(
        np.rint(valid_points[:, 1] * mesh_scale[1]).astype(int),
        0,
        voxelized.shape[1] - 1,
    )
    z_idx = np.clip(
        np.rint(np.maximum(valid_points[:, 2], 0.0)).astype(int),
        0,
        voxelized.shape[0] - 1,
    )
    voxelized[(z_idx, y_idx, x_idx)] = 255

    tifffile.imwrite(dump_dir / f"{label}_mesh_voxelization.tif", voxelized, compression="zlib")


@dataclass
class TracerScenario:
    name: str
    make_volume: Callable[[], np.ndarray]
    origin: Callable[[np.ndarray], tuple[float, float, float]]
    validate: Callable[[object, np.ndarray, dict[str, float]], None]
    params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


def _build_hollow_cylinder_scenario() -> TracerScenario:
    size = 512
    radius = 128
    wall_thickness = 8
    center = size // 2

    def make_volume() -> np.ndarray:
        return _make_hollow_cylinder_volume(size, radius, wall_thickness)

    def origin(_volume: np.ndarray) -> tuple[float, float, float]:
        return (float(center), float(center), float(center + radius))

    def validate(mesh: object, valid_points: np.ndarray, params: dict[str, float]) -> None:
        center_xy = np.array([center, center], dtype=np.float32)
        radii = np.linalg.norm(valid_points[:, :2] - center_xy, axis=1)
        assert radii.size > 0, "No radii computed for hollow cylinder"
        assert np.isfinite(radii).all(), "Non-finite radii encountered"
        assert np.mean(radii) == pytest.approx(radius, rel=0.1), "Average radius deviates too much"
        assert np.std(radii) > params["step_size"] * 0.01, "Cylinder surface lacks variance"

    return TracerScenario(
        name="hollow_cylinder",
        make_volume=make_volume,
        origin=origin,
        validate=validate,
    )


def _build_curvilinear_plane_scenario() -> TracerScenario:
    size = 320
    thickness = 6
    amplitude = size * 0.12
    wavelength = size / 6.0
    base_z = size / 2.0

    def make_volume() -> np.ndarray:
        return _make_curvilinear_plane_volume(size, thickness, amplitude, wavelength)

    def origin(_volume: np.ndarray) -> tuple[float, float, float]:
        center = size / 2.0
        return (float(base_z), float(center), float(center))

    def validate(mesh: object, valid_points: np.ndarray, params: dict[str, float]) -> None:
        z_vals = valid_points[:, 2]
        assert z_vals.size > 0, "No points captured for curvilinear plane"
        assert np.isfinite(z_vals).all(), "Curvilinear plane contains non-finite coordinates"
        assert z_vals.min() >= -1.0, "Curvilinear plane dipped below the volume start"
        assert z_vals.max() <= size + 1.0, "Curvilinear plane exceeded the volume bounds"
        assert np.std(z_vals) > thickness * 0.5, "Curvilinear plane lacks sufficient curvature"

    return TracerScenario(
        name="curvilinear_plane",
        make_volume=make_volume,
        origin=origin,
        validate=validate,
        params={"generations": 35},
    )


TRACER_SCENARIOS = [
    _build_hollow_cylinder_scenario(),
    _build_curvilinear_plane_scenario(),
]


@pytest.mark.slow
@pytest.mark.parametrize("scenario", TRACER_SCENARIOS, ids=lambda s: s.name)
def test_grow_seg_from_seed_on_hollow_cylinder(
    tmp_path: Path, tracer_dump_dir: Optional[Path], scenario: TracerScenario
) -> None:
    volume_data = scenario.make_volume()
    assert volume_data.flags["C_CONTIGUOUS"]

    params: dict[str, float] = {
        "generations": 40,
        "step_size": 20,
        "cache_size": 16_000_000,
        "snapshot-interval": 0,
    }
    params.update(scenario.params)

    vc_gen_normalgrids = _find_vc_gen_normalgrids()
    if vc_gen_normalgrids is None:
        pytest.skip("vc_gen_normalgrids executable is not available; build the project first")

    scenario_name = scenario.name
    out_dir = tmp_path / "tracer_artifacts" / f"{scenario_name}-{uuid4().hex}"

    zarr_path = tmp_path / f"{scenario_name}.zarr"
    _write_volume_to_zarr(volume_data, zarr_path, voxel_size=1.0)

    origin = scenario.origin(volume_data)
    result = tracing.grow_seg_from_seed(
        zarr_path,
        params,
        origin,
        output_dir=out_dir,
    )

    mesh = result["surface"]
    points = mesh.points()

    valid_mask = np.all(points > -0.5, axis=-1)
    valid_points = points[valid_mask]
    assert valid_points.size > 0, f"Tracer did not return any populated quad points for {scenario_name}"
    assert np.isfinite(valid_points).all()

    scale = np.asarray(mesh.scale, dtype=np.float32)
    expected_scale = np.array(
        [1.0 / params["step_size"], 1.0 / params["step_size"]], dtype=np.float32
    )
    assert np.allclose(scale, expected_scale, atol=1e-6)

    grid_rows, grid_cols = mesh.grid_shape
    bounds = mesh.bounds
    assert bounds == (
        grid_rows * params["step_size"],
        grid_cols * params["step_size"],
    )

    dense_coords = mesh.gen((bounds[0], bounds[1]), scale=1.0, offset=(-bounds[1] / 2.0, -bounds[0] / 2.0, 0.0))
    dense_coords = np.asarray(dense_coords)
    dense_valid = np.isfinite(dense_coords).all(axis=-1)
    dense_points = dense_coords[dense_valid]
    if dense_points.size > 0:
        manual_points = dense_points
        manual_scale = np.array([1.0, 1.0], dtype=np.float32)
    else:
        manual_points = valid_points
        manual_scale = scale

    voxelization = result.voxelize()
    assert voxelization.points.shape[1] == 3
    assert np.allclose(voxelization.points, manual_points)
    assert np.allclose(voxelization.scale, manual_scale)

    voxel_volume = voxelization.to_volume(shape=volume_data.shape)
    assert voxel_volume.shape == volume_data.shape
    assert voxel_volume.dtype == np.uint8
    assert np.count_nonzero(voxel_volume) > 0

    scenario.validate(mesh, valid_points, params)

    if tracer_dump_dir is not None:
        scenario_dump_dir = tracer_dump_dir / scenario_name
        _dump_tracer_debug_artifacts(
            scenario_dump_dir,
            volume_data,
            voxelization.points,
            voxelization.scale,
            label=scenario_name,
        )
