"""Infer winding-sheet crossings in paired 3-D TIFF cubes.

This is the TIFF-cube counterpart to :mod:`infer_winding_volume`.  Each input
image is paired with a label volume whose value 1 marks known sheets.  The
label nearest the volume centre supplies two pieces of geometry that a
headless winding-phase model needs:

* a central sheet crossing, used to fix the model's otherwise-free phase;
* a local sheet normal, used as the ray direction.

The normal is oriented from the concave ("inside") side toward the outside.
A ``transverse_size x transverse_size x ray_length`` slab is sampled around
that ray, integer passages of the registered monotone phase are decoded, and
the passage coordinates are transformed back into the input TIFF's ZYX frame.
The output is a binary uint8 TIFF with exactly the label volume's shape.

Example::

    python infer_winding_kaggle.py \
        /home/sean/Desktop/winding_model/runs/winding_model_3d_large_11 \
        /mnt/bigpc/intersection_addtl_train_data/kaggle/images \
        /mnt/bigpc/intersection_addtl_train_data/kaggle/labels \
        /mnt/bigpc/intersection_addtl_train_data/kaggle/winding_model_inference \
        --device cuda:0 --compile
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage
import tifffile


@dataclass(frozen=True)
class RayGeometry:
    """Ray-aligned frame expressed directly in input-TIFF ZYX voxels."""

    centre_zyx: np.ndarray
    anchor_zyx: np.ndarray
    anchor_offset: float
    direction_zyx: np.ndarray
    axis_a_zyx: np.ndarray
    axis_b_zyx: np.ndarray
    inside_zyx: np.ndarray
    curvature: float
    central_label_runs: int


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open starts/ends of true runs in a one-dimensional mask."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def _orthogonal_axes(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match VolumeSlabExtractor's deterministic right-handed frame."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(direction)))] = 1.0
    axis_a = np.cross(reference, direction)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(direction, axis_a)
    return axis_a, axis_b


def _sample_label(
    label: np.ndarray, points_zyx: np.ndarray, *, cval: int = 2
) -> np.ndarray:
    values = scipy.ndimage.map_coordinates(
        label,
        np.asarray(points_zyx, dtype=np.float64).reshape(-1, 3).T,
        order=0,
        mode="constant",
        cval=cval,
        prefilter=False,
    )
    return values.reshape(points_zyx.shape[:-1])


def _local_normal(
    label: np.ndarray, point_zyx: np.ndarray, radius: int, sigma: float
) -> np.ndarray:
    """Dominant label-boundary normal near a central sheet point."""
    shape = np.asarray(label.shape)
    centre = np.rint(point_zyx).astype(int)
    lo = np.maximum(0, centre - radius)
    hi = np.minimum(shape, centre + radius + 1)
    crop = (label[tuple(slice(a, b) for a, b in zip(lo, hi))] == 1).astype(
        np.float32
    )
    smooth = scipy.ndimage.gaussian_filter(crop, sigma=float(sigma))
    gradients = np.stack(np.gradient(smooth), axis=-1).reshape(-1, 3)
    mass = np.einsum("ni,ni->n", gradients, gradients)
    keep = mass > max(float(mass.max()) * 1.0e-4, 1.0e-10)
    if int(keep.sum()) < 16:
        raise ValueError("too little local label boundary to estimate a ray normal")
    gradients = gradients[keep]
    tensor = gradients.T @ gradients
    values, vectors = np.linalg.eigh(tensor)
    normal = vectors[:, int(np.argmax(values))]
    return normal / np.linalg.norm(normal)


def _candidate_directions(normal: np.ndarray, cone_degrees: float) -> np.ndarray:
    """Deterministic cap used to refine a structure-tensor normal."""
    axis_a, axis_b = _orthogonal_axes(normal)
    candidates = [normal]
    for fraction, count in ((0.35, 12), (0.7, 20), (1.0, 28)):
        angle = math.radians(float(cone_degrees) * fraction)
        for azimuth in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False):
            tangent = math.cos(azimuth) * axis_a + math.sin(azimuth) * axis_b
            candidates.append(math.cos(angle) * normal + math.sin(angle) * tangent)
    return np.asarray(candidates, dtype=np.float64)


def _refine_direction(
    label: np.ndarray,
    seed: np.ndarray,
    normal: np.ndarray,
    ray_length: int,
    cone_degrees: float,
) -> tuple[np.ndarray, int]:
    """Prefer many thin sheet runs while staying near the local normal."""
    positions = np.arange(ray_length, dtype=np.float64) - (ray_length - 1) / 2.0
    candidates = _candidate_directions(normal, cone_degrees)
    points = seed[None, None, :] + positions[None, :, None] * candidates[:, None, :]
    sampled = _sample_label(label, points)
    best = None
    for index, values in enumerate(sampled):
        starts, ends = _runs(values == 1)
        lengths = ends - starts
        # A grazing direction can accumulate plenty of positive voxels but
        # does not give the desired normal crossings.  Run count dominates;
        # total thickness and angular tilt only break near-ties.
        score = (
            1000.0 * len(starts)
            - float(lengths.sum())
            - 0.05 * index
        )
        candidate = (score, len(starts), index)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    direction = candidates[best[2]]
    return direction / np.linalg.norm(direction), int(best[1])


def _centre_seed_on_run(
    label: np.ndarray, seed: np.ndarray, direction: np.ndarray, radius: int = 16
) -> np.ndarray:
    positions = np.arange(-radius, radius + 0.25, 0.25, dtype=np.float64)
    values = _sample_label(label, seed[None] + positions[:, None] * direction)
    starts, ends = _runs(values == 1)
    if not len(starts):
        return seed
    centres = 0.5 * (positions[starts] + positions[ends - 1])
    offset = centres[int(np.argmin(np.abs(centres)))]
    return seed + float(offset) * direction


def _nearest_line_crossing(
    label: np.ndarray,
    centre: np.ndarray,
    direction: np.ndarray,
    ray_length: int,
) -> tuple[np.ndarray, int]:
    """Centre of the value-1 run nearest the volume centre on a ray."""
    half = (ray_length - 1) / 2.0
    positions = np.arange(-half, half + 0.25, 0.25, dtype=np.float64)
    values = _sample_label(label, centre[None] + positions[:, None] * direction)
    starts, ends = _runs(values == 1)
    if not len(starts):
        raise ValueError("the fitted central ray does not cross a labeled sheet")
    run_centres = 0.5 * (positions[starts] + positions[ends - 1])
    offset = float(run_centres[int(np.argmin(np.abs(run_centres)))])
    return centre + offset * direction, int(len(starts))


def _curvature_orientation(
    label: np.ndarray,
    seed: np.ndarray,
    normal: np.ndarray,
    transverse_radius: float = 24.0,
) -> tuple[np.ndarray, float]:
    """Orient a normal toward the local sheet's concave/inside side.

    Parallel probes track the label run nearest the seed sheet.  A quadratic
    graph ``s(u, v)`` is fit in the normal coordinate; the sign of its trace
    identifies the side into which the sheet curves away from its tangent
    plane.  The sign is immaterial to crossing positions, but canonicalizing
    it makes the sampled ray start on the requested inside side.
    """
    axis_a, axis_b = _orthogonal_axes(normal)
    uv_values = np.linspace(-transverse_radius, transverse_radius, 9)
    uv = np.stack(np.meshgrid(uv_values, uv_values, indexing="ij"), axis=-1).reshape(-1, 2)
    uv = uv[np.linalg.norm(uv, axis=1) <= transverse_radius + 1.0e-6]
    s_values = np.arange(-18.0, 18.01, 0.5)
    bases = (
        seed[None]
        + uv[:, :1] * axis_a[None]
        + uv[:, 1:] * axis_b[None]
    )
    points = bases[:, None, :] + s_values[None, :, None] * normal[None, None]
    sampled = _sample_label(label, points)

    kept_uv, sheet_s = [], []
    for offset, values in zip(uv, sampled):
        starts, ends = _runs(values == 1)
        if not len(starts):
            continue
        centres = 0.5 * (s_values[starts] + s_values[ends - 1])
        nearest = float(centres[int(np.argmin(np.abs(centres)))])
        if abs(nearest) <= 12.0:
            kept_uv.append(offset)
            sheet_s.append(nearest)
    if len(kept_uv) < 12:
        return normal, 0.0

    uv = np.asarray(kept_uv)
    design = np.column_stack(
        [
            np.ones(len(uv)),
            uv[:, 0],
            uv[:, 1],
            uv[:, 0] ** 2,
            uv[:, 0] * uv[:, 1],
            uv[:, 1] ** 2,
        ]
    )
    coefficients = np.linalg.lstsq(design, np.asarray(sheet_s), rcond=None)[0]
    curvature = float(2.0 * (coefficients[3] + coefficients[5]))
    inside = normal if curvature >= 0.0 else -normal
    return inside, abs(curvature)


def estimate_ray_geometry(
    label: np.ndarray,
    ray_length: int,
    *,
    normal_radius: int = 40,
    normal_sigma: float = 1.5,
    refine_cone_degrees: float = 18.0,
) -> RayGeometry:
    """Estimate an anchored inside-to-outside ray from a label cube."""
    if label.ndim != 3:
        raise ValueError(f"expected a 3-D label, got shape {label.shape}")
    crossing_coords = np.argwhere(label == 1)
    if not len(crossing_coords):
        raise ValueError("label contains no value-1 sheet crossings")
    volume_centre = (np.asarray(label.shape, dtype=np.float64) - 1.0) / 2.0
    nearest = int(np.argmin(np.sum((crossing_coords - volume_centre) ** 2, axis=1)))
    nearest_sheet = crossing_coords[nearest].astype(np.float64)

    normal = _local_normal(label, nearest_sheet, normal_radius, normal_sigma)
    normal, run_count = _refine_direction(
        label, volume_centre, normal, ray_length, refine_cone_degrees
    )
    anchor, run_count = _nearest_line_crossing(
        label, volume_centre, normal, ray_length
    )
    anchor = _centre_seed_on_run(label, anchor, normal)
    inside, curvature = _curvature_orientation(label, anchor, normal)

    # The ray origin will be seed - midpoint * direction.  Choosing direction
    # opposite the inside normal therefore puts the ray's starting half on
    # the concave/inside side of the sheet.
    direction = -inside
    axis_a, axis_b = _orthogonal_axes(direction)
    return RayGeometry(
        centre_zyx=volume_centre,
        anchor_zyx=anchor,
        anchor_offset=float(np.dot(anchor - volume_centre, direction)),
        direction_zyx=direction,
        axis_a_zyx=axis_a,
        axis_b_zyx=axis_b,
        inside_zyx=inside,
        curvature=curvature,
        central_label_runs=run_count,
    )


def _sampling_grid(
    geometry: RayGeometry,
    shape_zyx: tuple[int, int, int],
    transverse_size: int,
    ray_length: int,
    device,
):
    """CUDA grid_sample grid plus a geometrically exact validity mask."""
    import torch

    transverse = torch.arange(transverse_size, device=device, dtype=torch.float32)
    transverse -= (transverse_size - 1) / 2.0
    ray = torch.arange(ray_length, device=device, dtype=torch.float32)
    ray -= (ray_length - 1) / 2.0
    centre = torch.as_tensor(
        geometry.centre_zyx, device=device, dtype=torch.float32
    )
    axis_a = torch.as_tensor(geometry.axis_a_zyx, device=device, dtype=torch.float32)
    axis_b = torch.as_tensor(geometry.axis_b_zyx, device=device, dtype=torch.float32)
    direction = torch.as_tensor(
        geometry.direction_zyx, device=device, dtype=torch.float32
    )
    points = (
        centre
        + transverse[:, None, None, None] * axis_a
        + transverse[None, :, None, None] * axis_b
        + ray[None, None, :, None] * direction
    )
    shape = torch.as_tensor(shape_zyx, device=device, dtype=torch.float32)
    valid = ((points >= 0.0) & (points <= shape - 1.0)).all(dim=-1)
    # grid_sample's final coordinate order is XYZ, while points are ZYX.
    grid = 2.0 * points[..., [2, 1, 0]] / (shape[[2, 1, 0]] - 1.0) - 1.0
    return grid.unsqueeze(0), valid


def extract_slab(image: np.ndarray, geometry: RayGeometry, cfg: dict, device):
    import torch
    import torch.nn.functional as F

    transverse_size = int(cfg.get("transverse_size", 128))
    ray_length = int(cfg.get("ray_length", 384))
    grid, valid = _sampling_grid(
        geometry, image.shape, transverse_size, ray_length, device
    )
    volume = torch.from_numpy(np.ascontiguousarray(image)).to(
        device=device, dtype=torch.float32
    )
    slab = F.grid_sample(
        volume[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    return slab, valid


def decode_crossings(
    phase: np.ndarray,
    valid: np.ndarray,
    geometry: RayGeometry,
    output_shape: tuple[int, int, int],
    *,
    edge_margin: int,
    max_level: int | None,
    output_radius: int,
) -> tuple[np.ndarray, dict]:
    """Decode registered integer phase passages and rasterize in input ZYX."""
    transverse_size, _, ray_length = phase.shape
    centre = int(round((transverse_size - 1) / 2.0))
    anchor = (ray_length - 1) / 2.0 + float(geometry.anchor_offset)
    if not 0.0 <= anchor <= ray_length - 1:
        raise ValueError(f"label anchor lies outside the sampled ray: {anchor}")
    anchor_phase = np.interp(
        anchor,
        np.arange(ray_length, dtype=np.float64),
        np.asarray(phase[centre, centre], dtype=np.float64),
    )
    registered = np.asarray(phase, dtype=np.float32) - np.float32(anchor_phase)
    flat = registered.reshape(-1, ray_length)
    valid_flat = valid.reshape(-1, ray_length)
    lower = np.floor(flat)
    counts = (lower[:, 1:] - lower[:, :-1]).astype(np.int16)
    np.clip(counts, 0, None, out=counts)
    column, segment_index = np.nonzero(counts)
    if not len(column):
        return np.zeros(output_shape, dtype=np.uint8), {"decoded_points": 0}

    repetitions = counts[column, segment_index].astype(np.int64)
    column = np.repeat(column, repetitions)
    index = np.repeat(segment_index, repetitions)
    first = np.cumsum(repetitions) - repetitions
    within = np.arange(int(repetitions.sum())) - np.repeat(first, repetitions)
    levels = np.repeat(lower[np.nonzero(counts)], repetitions) + within + 1.0
    base = flat[column, index]
    delta = np.maximum(flat[column, index + 1] - base, 1.0e-9)
    ray_position = index + np.clip((levels - base) / delta, 0.0, 1.0)

    keep = (ray_position >= edge_margin) & (
        ray_position < ray_length - edge_margin
    )
    if max_level is not None:
        keep &= np.abs(levels) <= int(max_level)
    nearest_ray = np.rint(ray_position).astype(np.int64).clip(0, ray_length - 1)
    keep &= valid_flat[column, nearest_ray]
    column, ray_position, levels = column[keep], ray_position[keep], levels[keep]

    row = column // transverse_size
    col = column % transverse_size
    offsets_a = row.astype(np.float64) - (transverse_size - 1) / 2.0
    offsets_b = col.astype(np.float64) - (transverse_size - 1) / 2.0
    offsets_ray = ray_position.astype(np.float64) - (ray_length - 1) / 2.0
    points = (
        geometry.centre_zyx[None]
        + offsets_a[:, None] * geometry.axis_a_zyx[None]
        + offsets_b[:, None] * geometry.axis_b_zyx[None]
        + offsets_ray[:, None] * geometry.direction_zyx[None]
    )
    voxel = np.rint(points).astype(np.int64)
    shape = np.asarray(output_shape)
    in_bounds = ((voxel >= 0) & (voxel < shape)).all(axis=1)
    voxel, levels = voxel[in_bounds], levels[in_bounds]

    output = np.zeros(output_shape, dtype=np.uint8)
    if len(voxel):
        output[voxel[:, 0], voxel[:, 1], voxel[:, 2]] = 1
    if output_radius:
        output = scipy.ndimage.binary_dilation(
            output, iterations=int(output_radius)
        ).astype(np.uint8)
    return output, {
        "decoded_points": int(len(voxel)),
        "output_voxels": int(output.sum()),
        "minimum_level": None if not len(levels) else int(np.min(levels)),
        "maximum_level": None if not len(levels) else int(np.max(levels)),
    }


def _checkpoint_path(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    checkpoint = model_path / "ckpt_final.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no ckpt_final.pth in {model_path}")
    return checkpoint


def _load_model(model_path: Path, device, *, use_ema: bool, compile_model: bool):
    import torch

    try:
        from vesuvius.neural_tracing.winding_models.winding_model import WindingModel
    except ModuleNotFoundError:
        from winding_model import WindingModel

    checkpoint_path = _checkpoint_path(model_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    cfg = checkpoint["config"]
    if bool((cfg.get("model") or {}).get("use_crossing_head", True)):
        raise ValueError("this script currently expects a headless phase checkpoint")
    if float(cfg.get("spacing", 1.0)) != 1.0:
        raise ValueError("TIFF inference currently requires model spacing == 1")
    if int(cfg.get("column_stride", 1)) != 1:
        raise ValueError("TIFF inference currently requires column_stride == 1")
    model = WindingModel(cfg.get("model"))
    state_key = "model_ema" if use_ema and "model_ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key])
    model.to(device).eval()
    if compile_model:
        model = torch.compile(model)
    return model, cfg, checkpoint_path, state_key


def _jsonable_geometry(geometry: RayGeometry) -> dict:
    values = asdict(geometry)
    for key, value in list(values.items()):
        if isinstance(value, np.ndarray):
            values[key] = [float(item) for item in value]
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="model run directory or .pth")
    parser.add_argument("images", type=Path, help="directory of image TIFFs")
    parser.add_argument("labels", type=Path, help="directory of paired label TIFFs")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ema", action="store_true", help="use EMA weights")
    parser.add_argument("--edge-margin", type=int, default=8)
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="optional maximum winding distance from the anchored sheet",
    )
    parser.add_argument(
        "--output-radius",
        type=int,
        default=0,
        help="optional binary-dilation radius after nearest-voxel rasterization",
    )
    parser.add_argument("--normal-radius", type=int, default=40)
    parser.add_argument("--normal-sigma", type=float, default=1.5)
    parser.add_argument("--refine-cone-degrees", type=float, default=18.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compression", default="zlib", choices=("zlib", "lzma", "zstd", "none")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.edge_margin < 0 or args.output_radius < 0:
        raise ValueError("edge margin and output radius must be non-negative")
    if args.max_level is not None and args.max_level < 0:
        raise ValueError("max level must be non-negative")

    import torch

    device = torch.device(args.device)
    model, cfg, checkpoint_path, state_key = _load_model(
        args.model, device, use_ema=args.ema, compile_model=args.compile
    )
    ray_length = int(cfg.get("ray_length", 384))
    files = sorted(args.images.glob("*.tif"))
    files = files[int(args.start) :]
    if args.limit is not None:
        files = files[: int(args.limit)]
    if not files:
        raise ValueError("no input TIFFs selected")
    missing = [path.name for path in files if not (args.labels / path.name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing paired labels, beginning with {missing[0]}")
    args.output.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression
    manifest_path = args.output / "inference_manifest.jsonl"
    started = time.time()
    completed = skipped = 0
    with manifest_path.open("a", buffering=1) as manifest:
        for sequence, image_path in enumerate(files, 1):
            output_path = args.output / image_path.name
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[{sequence}/{len(files)}] skip {image_path.name}", flush=True)
                continue
            item_started = time.time()
            image = tifffile.imread(image_path)
            label_path = args.labels / image_path.name
            label = tifffile.imread(label_path)
            if image.shape != label.shape:
                raise ValueError(
                    f"shape mismatch for {image_path.name}: {image.shape} != {label.shape}"
                )

            geometry = estimate_ray_geometry(
                label,
                ray_length,
                normal_radius=args.normal_radius,
                normal_sigma=args.normal_sigma,
                refine_cone_degrees=args.refine_cone_degrees,
            )
            slab, valid = extract_slab(image, geometry, cfg, device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                phase = model(slab[None], valid[None])["phase"][0]
            output, stats = decode_crossings(
                phase.float().cpu().numpy(),
                valid.cpu().numpy(),
                geometry,
                label.shape,
                edge_margin=args.edge_margin,
                max_level=args.max_level,
                output_radius=args.output_radius,
            )
            tifffile.imwrite(
                output_path,
                output,
                compression=compression,
                photometric="minisblack",
                metadata={"axes": "ZYX"},
            )
            elapsed = time.time() - item_started
            record = {
                "name": image_path.name,
                "shape_zyx": [int(value) for value in label.shape],
                "checkpoint": str(checkpoint_path.resolve()),
                "state": state_key,
                "geometry": _jsonable_geometry(geometry),
                **stats,
                "elapsed_seconds": elapsed,
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
            completed += 1
            total_elapsed = time.time() - started
            rate = completed / total_elapsed if total_elapsed else 0.0
            remaining = (len(files) - sequence) / rate if rate else math.inf
            print(
                f"[{sequence}/{len(files)}] {image_path.name}: "
                f"{stats['decoded_points']:,} points, {stats['output_voxels']:,} voxels, "
                f"{elapsed:.1f}s, ETA {remaining / 60.0:.1f}m",
                flush=True,
            )

    print(
        f"done: wrote {completed}, skipped {skipped}, "
        f"elapsed {(time.time() - started) / 60.0:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
