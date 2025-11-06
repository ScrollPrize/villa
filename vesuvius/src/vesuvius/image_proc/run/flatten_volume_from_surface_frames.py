#!/usr/bin/env python3
"""Flatten a volumetric parchment region using predicted surface-frame vectors.
THIS IS A WIP , NOT TESTED, NOT GUARANTEED TO BE FUNCTIONAL"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import zarr


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve for a local (U, V) parameterisation from predicted surface-frame vectors "
            "and resample the volume into a flattened UV grid."
        )
    )

    parser.add_argument("--input-volume", required=True, help="Path to source intensity OME-Zarr store")
    parser.add_argument(
        "--input-dataset",
        default="0",
        help="Dataset key inside the intensity store (default: 0 for first pyramid level)",
    )
    parser.add_argument("--surface-frame", required=True, help="Path to surface-frame prediction Zarr")
    parser.add_argument(
        "--surface-frame-dataset",
        default="0",
        help="Dataset key inside the surface-frame store (default: 0)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for the flattened output Zarr store (will be overwritten)",
    )
    parser.add_argument(
        "--z-range",
        type=int,
        nargs=2,
        metavar=("Z_START", "Z_STOP"),
        required=True,
        help="Inclusive/exclusive z-slice bounds [start, stop) to process",
    )
    parser.add_argument(
        "--input-channel",
        type=int,
        default=0,
        help="Channel index to use when the intensity volume is 4D (default: 0)",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Width of the flattened UV grid (default: match input X extent)",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Height of the flattened UV grid (default: match input Y extent)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device to run the solver on (default: cpu)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=400,
        help="Maximum optimisation iterations for the UV fields (default: 400)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for the Adam optimiser (default: 1e-2)",
    )
    parser.add_argument(
        "--lambda-orth",
        type=float,
        default=1e-1,
        help="Penalty weight encouraging orthogonality between grad U and grad V",
    )
    parser.add_argument(
        "--lambda-equal-scale",
        type=float,
        default=5e-2,
        help="Penalty weight encouraging equal scale for |grad U| and |grad V|",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Iterations between optimisation progress prints",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Force interpretation of surface-frame array as (Z, Y, X, 9) instead of (9, Z, Y, X)",
    )
    parser.add_argument(
        "--umbilicus",
        type=str,
        default=None,
        help="Optional path to comma-separated (z,y,x) points describing the spiral centre seam",
    )
    parser.add_argument(
        "--seam-thickness",
        type=int,
        default=1,
        help="Half-width (in voxels) of the seam constraint band around the umbilicus",
    )
    parser.add_argument(
        "--seam-direction",
        choices=["x_min", "x_max"],
        default="x_min",
        help="Direction to extend the seam from the umbilicus towards the volume boundary",
    )

    return parser


def _open_dataset(path: str, dataset: str):
    store = zarr.open(path, mode="r")
    if isinstance(store, zarr.hierarchy.Group):
        if dataset not in store:
            raise KeyError(f"Dataset '{dataset}' not found in {path}")
        return store[dataset]
    return store


def _extract_intensity(
    array: zarr.Array,
    z_start: int,
    z_stop: int,
    channel: int,
) -> np.ndarray:
    if array.ndim == 3:
        return array[z_start:z_stop, :, :]
    if array.ndim == 4:
        return array[channel, z_start:z_stop, :, :]
    raise ValueError(f"Unsupported intensity array shape {array.shape}; expected 3D or 4D")


def _extract_surface_frame(
    array: zarr.Array,
    z_start: int,
    z_stop: int,
    force_channels_last: bool,
) -> np.ndarray:
    if array.ndim == 4:
        if array.shape[0] == 9 and not force_channels_last:
            data = array[:, z_start:z_stop, :, :]
            return np.moveaxis(data, 0, -1)
        if array.shape[-1] == 9:
            return array[z_start:z_stop, :, :, :]
    raise ValueError(
        "Surface-frame dataset must be 4D with 9-channel axis;"
        f" got shape {array.shape}. Set --channels-last if channels are trailing."
    )


def _normalize_frame_vectors(frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tu = frame[..., 0:3]
    tv = frame[..., 3:6]
    n = frame[..., 6:9]

    tu = torch.nn.functional.normalize(tu, dim=-1, eps=1e-6)
    n = torch.nn.functional.normalize(n, dim=-1, eps=1e-6)
    tu = tu - (tu * n).sum(dim=-1, keepdim=True) * n
    tu = torch.nn.functional.normalize(tu, dim=-1, eps=1e-6)
    tv = torch.cross(n, tu, dim=-1)
    tv = torch.nn.functional.normalize(tv, dim=-1, eps=1e-6)
    n = torch.cross(tu, tv, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1, eps=1e-6)

    tu = torch.nan_to_num(tu)
    tv = torch.nan_to_num(tv)
    n = torch.nan_to_num(n)
    return tu, tv, n


def _load_umbilicus_points(path: str) -> np.ndarray:
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [token.strip() for token in stripped.split(",") if token.strip()]
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid umbilicus entry '{raw.rstrip()}': expected three comma-separated values"
                )
            entries.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not entries:
        raise ValueError(f"Umbilicus file '{path}' contained no coordinate triples")

    arr = np.asarray(entries, dtype=np.float32)
    order = np.argsort(arr[:, 0])
    return arr[order]


def _interpolate_umbilicus(
    points: np.ndarray,
    z_start: int,
    z_stop: int,
    height: int,
    width: int,
) -> np.ndarray:
    target_z = np.arange(z_start, z_stop, dtype=np.float32)
    z_vals = points[:, 0]
    y_vals = points[:, 1]
    x_vals = points[:, 2]

    interp_y = np.interp(target_z, z_vals, y_vals, left=y_vals[0], right=y_vals[-1])
    interp_x = np.interp(target_z, z_vals, x_vals, left=x_vals[0], right=x_vals[-1])

    centres = np.stack([interp_y, interp_x], axis=1)
    centres[:, 0] = np.clip(centres[:, 0], 0, height - 1)
    centres[:, 1] = np.clip(centres[:, 1], 0, width - 1)
    return centres


def _build_seam_mask(
    centres: np.ndarray,
    shape: Tuple[int, int, int],
    thickness: int,
    direction: str,
) -> torch.Tensor:
    depth, height, width = shape
    mask = np.zeros(shape, dtype=bool)
    thickness = max(0, int(thickness))

    for z_idx, (y_c, x_c) in enumerate(centres):
        y_i = int(round(float(y_c)))
        x_i = int(round(float(x_c)))
        if not (0 <= y_i < height and 0 <= x_i < width):
            continue

        if direction == "x_min":
            x_iter = range(x_i, -1, -1)
        else:
            x_iter = range(x_i, width)

        for x in x_iter:
            for dy in range(-thickness, thickness + 1):
                y = y_i + dy
                if 0 <= y < height:
                    mask[z_idx, y, x] = True

    return torch.from_numpy(mask)


def _propagate_orientation(
    tu: torch.Tensor,
    tv: torch.Tensor,
    n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def align_axis(vec_u: torch.Tensor, vec_v: torch.Tensor, vec_n: torch.Tensor, axis: int):
        dim = vec_u.shape[axis]
        for idx in range(1, dim):
            slicer_current = [slice(None)] * vec_u.ndim
            slicer_prev = [slice(None)] * vec_u.ndim
            slicer_current[axis] = idx
            slicer_prev[axis] = idx - 1
            u_curr = vec_u[tuple(slicer_current)]
            u_prev = vec_u[tuple(slicer_prev)]
            v_curr = vec_v[tuple(slicer_current)]
            v_prev = vec_v[tuple(slicer_prev)]
            n_curr = vec_n[tuple(slicer_current)]
            n_prev = vec_n[tuple(slicer_prev)]

            dot = (u_curr * u_prev).sum(dim=-1) + (v_curr * v_prev).sum(dim=-1)
            flip_mask = dot < 0
            if torch.any(flip_mask):
                u_curr[flip_mask] *= -1
                v_curr[flip_mask] *= -1
                n_curr[flip_mask] *= -1

    for axis in range(3):
        align_axis(tu, tv, n, axis)

    tu, tv, n = _normalize_frame_vectors(torch.cat((tu, tv, n), dim=-1))
    return tu, tv, n


def _forward_gradient(field: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros(field.shape + (3,), device=field.device, dtype=field.dtype)
    grad[:-1, :, :, 0] = field[1:, :, :] - field[:-1, :, :]
    grad[:, :-1, :, 1] = field[:, 1:, :] - field[:, :-1, :]
    grad[:, :, :-1, 2] = field[:, :, 1:] - field[:, :, :-1]
    return grad


def _enforce_boundaries(
    u: torch.Tensor,
    v: torch.Tensor,
    width: float,
    height: float,
    seam_mask: Optional[torch.Tensor],
) -> None:
    with torch.no_grad():
        u[:, :, 0] = 0.0
        u[:, :, -1] = width
        v[:, 0, :] = 0.0
        v[:, -1, :] = height
        if seam_mask is not None:
            u[seam_mask] = 0.0


def _optimize_uv_fields(
    tu: torch.Tensor,
    tv: torch.Tensor,
    weight: torch.Tensor,
    width: int,
    height: int,
    max_iters: int,
    lr: float,
    lambda_orth: float,
    lambda_equal_scale: float,
    log_interval: int,
    seam_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = tu.device
    shape = tu.shape[:-1]

    u = torch.zeros(shape, device=device, dtype=tu.dtype, requires_grad=True)
    v = torch.zeros_like(u, requires_grad=True)

    optimizer = torch.optim.Adam([u, v], lr=lr)

    target_width = float(max(width - 1, 1))
    target_height = float(max(height - 1, 1))

    seam_mask_local: Optional[torch.Tensor] = None
    if seam_mask is not None:
        seam_mask_local = seam_mask.to(device=device, dtype=torch.bool)

    for step in range(1, max_iters + 1):
        optimizer.zero_grad(set_to_none=True)

        grad_u = _forward_gradient(u)
        grad_v = _forward_gradient(v)

        data_residual = (grad_u - tu) ** 2 + (grad_v - tv) ** 2
        loss = (data_residual.sum(dim=-1) * weight).mean()

        if lambda_orth > 0:
            dot = (grad_u * grad_v).sum(dim=-1)
            loss = loss + lambda_orth * ((dot ** 2) * weight).mean()

        if lambda_equal_scale > 0:
            norm_u = torch.sqrt((grad_u ** 2).sum(dim=-1) + 1e-8)
            norm_v = torch.sqrt((grad_v ** 2).sum(dim=-1) + 1e-8)
            loss = loss + lambda_equal_scale * (((norm_u - norm_v) ** 2) * weight).mean()

        loss.backward()

        if seam_mask_local is not None and u.grad is not None:
            u.grad.masked_fill_(seam_mask_local, 0.0)

        optimizer.step()

        _enforce_boundaries(u, v, target_width, target_height, seam_mask_local)

        with torch.no_grad():
            u -= u.mean()
            v -= v.mean()
        if log_interval and step % log_interval == 0:
            print(f"[UV] iter {step:04d}: loss={loss.item():.6f}")

    _enforce_boundaries(u, v, target_width, target_height, seam_mask_local)
    return u.detach(), v.detach()


def _scatter_to_uv_grid(
    intensity: torch.Tensor,
    u_field: torch.Tensor,
    v_field: torch.Tensor,
    out_width: int,
    out_height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = intensity.device

    u_shifted = u_field - u_field.min()
    v_shifted = v_field - v_field.min()

    u_max = u_shifted.max().item()
    v_max = v_shifted.max().item()
    if u_max < 1e-6 or v_max < 1e-6:
        raise RuntimeError("Degenerate U/V range; check the surface-frame field or seam constraints")

    scale_u = (out_width - 1) / u_max
    scale_v = (out_height - 1) / v_max

    u_idx = torch.clamp(torch.round(u_shifted * scale_u), 0, out_width - 1).long()
    v_idx = torch.clamp(torch.round(v_shifted * scale_v), 0, out_height - 1).long()

    depth, height, width = intensity.shape
    z_idx = (
        torch.arange(depth, device=device)
        .view(depth, 1, 1)
        .expand(depth, height, width)
    )

    flattened_size = depth * out_height * out_width
    linear_indices = u_idx + out_width * (v_idx + out_height * z_idx)

    flat_values = torch.zeros(flattened_size, device=device, dtype=intensity.dtype)
    flat_counts = torch.zeros_like(flat_values)

    flat_values.scatter_add_(0, linear_indices.view(-1), intensity.view(-1))
    flat_counts.scatter_add_(0, linear_indices.view(-1), torch.ones_like(intensity).view(-1))

    flat_values = flat_values.view(depth, out_height, out_width)
    flat_counts = flat_counts.view(depth, out_height, out_width)

    averaged = torch.where(flat_counts > 0, flat_values / flat_counts.clamp_min(1.0), torch.zeros_like(flat_values))
    return averaged, flat_counts


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    z_start, z_stop = args.z_range
    if z_stop <= z_start:
        raise ValueError("z-range must satisfy Z_STOP > Z_START")

    device = torch.device(args.device)

    intensity_arr = _open_dataset(args.input_volume, args.input_dataset)
    surface_arr = _open_dataset(args.surface_frame, args.surface_frame_dataset)

    intensity_np = _extract_intensity(intensity_arr, z_start, z_stop, args.input_channel)
    surface_np = _extract_surface_frame(surface_arr, z_start, z_stop, args.channels_last)

    print(f"Loaded intensity sub-volume: {intensity_np.shape} from {args.input_volume}")
    print(f"Loaded surface-frame block: {surface_np.shape} from {args.surface_frame}")

    intensity = torch.from_numpy(np.asarray(intensity_np, dtype=np.float32)).to(device)
    frames = torch.from_numpy(np.asarray(surface_np, dtype=np.float32)).to(device)

    tu, tv, n = _normalize_frame_vectors(frames)
    tu, tv, n = _propagate_orientation(tu, tv, n)

    weight = torch.ones(intensity.shape, device=device, dtype=torch.float32)

    out_width = args.output_width or intensity.shape[-1]
    out_height = args.output_height or intensity.shape[-2]

    seam_mask = None
    centres = None
    if args.umbilicus is not None:
        umb_points = _load_umbilicus_points(args.umbilicus)
        centres = _interpolate_umbilicus(
            umb_points,
            z_start,
            z_stop,
            height=intensity.shape[1],
            width=intensity.shape[2],
        )
        seam_mask = _build_seam_mask(
            centres,
            shape=intensity.shape,
            thickness=args.seam_thickness,
            direction=args.seam_direction,
        )
        seam_voxels = int(seam_mask.sum().item())
        print(
            f"Loaded umbilicus seam with {seam_voxels} constrained voxels towards {args.seam_direction} boundary"
        )

    print(
        f"Optimising UV fields over block (depth={intensity.shape[0]},"
        f" height={intensity.shape[1]}, width={intensity.shape[2]})"
    )

    u_field, v_field = _optimize_uv_fields(
        tu,
        tv,
        weight,
        width=out_width,
        height=out_height,
        max_iters=args.max_iters,
        lr=args.lr,
        lambda_orth=args.lambda_orth,
        lambda_equal_scale=args.lambda_equal_scale,
        log_interval=args.log_interval,
        seam_mask=seam_mask,
    )

    flattened, counts = _scatter_to_uv_grid(intensity, u_field, v_field, out_width, out_height)

    if device.type == "cuda":
        torch.cuda.synchronize()

    out_path = Path(args.output)
    if out_path.exists():
        print(f"Overwriting existing output directory: {out_path}")
        import shutil

        shutil.rmtree(out_path)

    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=True)

    flat_np = flattened.cpu().numpy()
    counts_np = counts.cpu().numpy()
    u_np = u_field.cpu().numpy()
    v_np = v_field.cpu().numpy()
    n_np = n.cpu().numpy()

    root.create_dataset(
        "flattened",
        data=flat_np,
        chunks=(1, min(out_height, 128), min(out_width, 128)),
        dtype=np.float32,
    )
    root.create_dataset(
        "counts",
        data=counts_np,
        chunks=(1, min(out_height, 128), min(out_width, 128)),
        dtype=np.float32,
    )
    root.create_dataset(
        "u_field",
        data=u_np,
        chunks=(1, min(u_np.shape[1], 128), min(u_np.shape[2], 128)),
        dtype=np.float32,
    )
    root.create_dataset(
        "v_field",
        data=v_np,
        chunks=(1, min(v_np.shape[1], 128), min(v_np.shape[2], 128)),
        dtype=np.float32,
    )
    root.create_dataset(
        "n_field",
        data=n_np,
        chunks=(1, min(n_np.shape[1], 128), min(n_np.shape[2], 128)),
        dtype=np.float32,
    )

    attrs = {
        "z_start": int(z_start),
        "z_stop": int(z_stop),
        "lambda_orth": float(args.lambda_orth),
        "lambda_equal_scale": float(args.lambda_equal_scale),
        "max_iters": int(args.max_iters),
        "lr": float(args.lr),
        "seam_direction": args.seam_direction,
        "seam_thickness": int(args.seam_thickness),
    }
    if centres is not None:
        attrs["umbilicus_points"] = centres.astype(np.float32).tolist()

    root.attrs.update(attrs)

    print(f"Flattened volume written to {out_path}")


if __name__ == "__main__":
    main()
