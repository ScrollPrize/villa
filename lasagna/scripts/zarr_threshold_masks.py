#!/usr/bin/env python3
"""Create threshold-derived zarr masks and apply them to Lasagna grad_mag data."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable

import cv2
import numcodecs
import numpy as np
import tifffile
import torch
import zarr

_LASAGNA_DIR = Path(__file__).resolve().parents[1]
if str(_LASAGNA_DIR) not in sys.path:
    sys.path.insert(0, str(_LASAGNA_DIR))

from lasagna_volume import LASAGNA_VOLUME_VERSION, LasagnaVolume
from omezarr_pyramid import (
    build_normal_omezarr_pyramid,
    build_scalar_omezarr_pyramid,
    clear_coarser_levels,
    print_progress as _print_progress,
)

zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})

_MASKOUT_GRAD_META: dict | None = None
_MASKOUT_MASK_META: dict | None = None
_MASKOUT_CFG: dict | None = None


def _parse_z_range(value: str) -> tuple[int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--z-range must be START,END")
    try:
        z0, z1 = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--z-range values must be integers") from exc
    if z1 <= z0:
        raise argparse.ArgumentTypeError("--z-range END must be greater than START")
    return z0, z1


def _open_zarr_array(path: Path, array: str | None):
    root = zarr.open(str(path), mode="r")
    if hasattr(root, "shape"):
        return root, path

    keys = sorted(str(k) for k in root.keys())
    if not keys:
        raise ValueError(f"zarr group has no arrays/groups: {path}")

    key = array
    if key is None:
        key = "0" if "0" in keys else keys[0]
    if key not in root:
        raise ValueError(f"zarr group {path} has no entry {key!r}; entries={keys}")

    arr = root[key]
    if not hasattr(arr, "shape"):
        raise ValueError(f"zarr entry {key!r} is not an array")
    return arr, path / key


def _attrs_dict(obj) -> dict:
    try:
        return dict(obj.attrs)
    except Exception:
        return {}


def _scale_from_preprocess_params(arr) -> tuple[float, float, float] | None:
    attrs = _attrs_dict(arr)
    params = attrs.get("preprocess_params")
    if isinstance(params, dict) and "scaledown" in params:
        sd = float(params["scaledown"])
        return sd, sd, sd
    if "scaledown" in attrs:
        sd = float(attrs["scaledown"])
        return sd, sd, sd
    return None


def _scale_from_ome_parent(array_path: Path) -> tuple[float, float, float] | None:
    """Infer ZYX scale from parent OME-Zarr multiscales metadata."""

    parent = array_path.parent
    try:
        group = zarr.open_group(str(parent), mode="r")
    except Exception:
        return None

    attrs = _attrs_dict(group)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list):
        return None

    rel = array_path.name
    for ms in multiscales:
        datasets = ms.get("datasets") if isinstance(ms, dict) else None
        if not isinstance(datasets, list):
            continue
        for dataset in datasets:
            if not isinstance(dataset, dict) or str(dataset.get("path", "")) != rel:
                continue
            transforms = dataset.get("coordinateTransformations", [])
            for transform in transforms:
                if not isinstance(transform, dict) or transform.get("type") != "scale":
                    continue
                scale = transform.get("scale")
                if isinstance(scale, (list, tuple)) and len(scale) >= 3:
                    return float(scale[0]), float(scale[1]), float(scale[2])
    return None


def _infer_zyx_scale(arr, array_path: Path) -> tuple[float, float, float]:
    scale = _scale_from_preprocess_params(arr)
    if scale is not None:
        return scale
    scale = _scale_from_ome_parent(array_path)
    if scale is not None:
        return scale
    return 1.0, 1.0, 1.0


def _full_z_range_to_array(
    z_range_full: tuple[int, int],
    *,
    scale_z: float,
    array_z: int,
) -> tuple[int, int]:
    z0f, z1f = z_range_full
    if scale_z <= 0:
        raise ValueError(f"invalid z scale: {scale_z}")
    z0 = int(np.floor(float(z0f) / scale_z))
    z1 = int(np.ceil(float(z1f) / scale_z))
    z0 = max(0, min(z0, array_z))
    z1 = max(z0, min(z1, array_z))
    if z1 <= z0:
        raise ValueError(
            f"full-res z-range {z_range_full} maps to empty zarr range "
            f"[{z0},{z1}) with scale_z={scale_z} array_z={array_z}"
        )
    return z0, z1


def _default_full_z_range(*, array_z: int, scale_z: float) -> tuple[int, int]:
    if scale_z <= 0:
        raise ValueError(f"invalid z scale: {scale_z}")
    return 0, int(np.ceil(float(array_z) * scale_z))


def _read_metadata(array_path: Path) -> dict:
    zarray = array_path / ".zarray"
    if zarray.exists():
        return json.loads(zarray.read_text())
    zarr_json = array_path / "zarr.json"
    if zarr_json.exists():
        return json.loads(zarr_json.read_text())
    return {}


def _chunk_key_candidates(
    array_path: Path,
    chunk_coord: tuple[int, ...],
    metadata: dict,
) -> Iterable[Path]:
    # zarr v2 defaults to "." unless dimension_separator="/" is set.
    sep_v2 = str(metadata.get("dimension_separator", "."))
    if sep_v2 not in {".", "/"}:
        sep_v2 = "."
    yield array_path / sep_v2.join(str(v) for v in chunk_coord)
    yield array_path / ".".join(str(v) for v in chunk_coord)
    yield array_path.joinpath(*(str(v) for v in chunk_coord))

    # zarr v3 stores chunks below c/ by default.
    enc = metadata.get("chunk_key_encoding", {})
    cfg = enc.get("configuration", {}) if isinstance(enc, dict) else {}
    sep_v3 = str(cfg.get("separator", "/"))
    if sep_v3 not in {".", "/"}:
        sep_v3 = "/"
    yield array_path / "c" / sep_v3.join(str(v) for v in chunk_coord)
    yield array_path / "c" / ".".join(str(v) for v in chunk_coord)
    yield array_path / "c" / Path(*[str(v) for v in chunk_coord])


def _chunk_exists(array_path: Path, chunk_coord: tuple[int, ...], metadata: dict) -> bool:
    return any(p.exists() for p in _chunk_key_candidates(array_path, chunk_coord, metadata))


def _missing_chunk_mask_for_slab(
    *,
    array_path: Path,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    z_range: tuple[int, int],
    channel: int | None,
) -> np.ndarray:
    """Return a ZYX bool mask for chunks absent from a local zarr store.

    If the chunk layout cannot be inspected, this returns an all-false mask.
    """

    if not array_path.exists():
        return np.zeros((z_range[1] - z_range[0], shape[-2], shape[-1]), dtype=bool)

    metadata = _read_metadata(array_path)
    if not metadata:
        return np.zeros((z_range[1] - z_range[0], shape[-2], shape[-1]), dtype=bool)

    z0, z1 = z_range
    if len(shape) == 3:
        chunk_prefix: tuple[int, ...] = ()
        cz, cy, cx = chunks
        z_axis = 0
    elif len(shape) == 4:
        ch = 0 if channel is None else int(channel)
        cc, cz, cy, cx = chunks
        chunk_prefix = (ch // cc,)
        z_axis = 1
    else:
        raise ValueError(f"unsupported zarr ndim={len(shape)}")

    del z_axis  # kept above to make axis intent explicit.
    Z, Y, X = shape[-3], shape[-2], shape[-1]
    z0 = max(0, min(z0, Z))
    z1 = max(z0, min(z1, Z))
    out = np.zeros((z1 - z0, Y, X), dtype=bool)

    czi0 = z0 // cz
    czi1 = (z1 - 1) // cz
    cyi1 = (Y - 1) // cy
    cxi1 = (X - 1) // cx
    missing_chunks = 0

    for czi in range(czi0, czi1 + 1):
        gz0 = czi * cz
        gz1 = min(Z, gz0 + cz)
        lz0 = max(0, gz0 - z0)
        lz1 = min(z1 - z0, gz1 - z0)
        if lz1 <= lz0:
            continue
        for cyi in range(cyi1 + 1):
            gy0 = cyi * cy
            gy1 = min(Y, gy0 + cy)
            for cxi in range(cxi1 + 1):
                chunk_coord = chunk_prefix + (czi, cyi, cxi)
                if _chunk_exists(array_path, chunk_coord, metadata):
                    continue
                gx0 = cxi * cx
                gx1 = min(X, gx0 + cx)
                out[lz0:lz1, gy0:gy1, gx0:gx1] = True
                missing_chunks += 1

    if missing_chunks:
        print(f"[mask] marked {missing_chunks} missing zarr chunks as empty", flush=True)
    return out


def _read_slab(
    arr,
    *,
    z_range: tuple[int, int],
    channel: int | None,
) -> np.ndarray:
    z0, z1 = z_range
    Z = int(arr.shape[-3])
    z0 = max(0, min(z0, Z))
    z1 = max(z0, min(z1, Z))
    if z1 <= z0:
        raise ValueError(f"empty clipped z-range: requested={z_range} volume_z={Z}")

    if len(arr.shape) == 3:
        data = np.asarray(arr[z0:z1, :, :])
    elif len(arr.shape) == 4:
        ch = 0 if channel is None else int(channel)
        data = np.asarray(arr[ch, z0:z1, :, :])
    else:
        raise ValueError(f"expected 3D ZYX or 4D CZYX zarr array, got shape={arr.shape}")
    if data.dtype != np.uint8:
        raise ValueError(f"expected uint8 zarr data, got dtype={data.dtype}")
    return data


def _shift_slices(
    ndim: int,
    dz: int,
    dy: int,
    dx: int,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    src = [slice(None)] * ndim
    dst = [slice(None)] * ndim
    for axis, delta in ((ndim - 3, dz), (ndim - 2, dy), (ndim - 1, dx)):
        if delta > 0:
            src[axis] = slice(0, -delta)
            dst[axis] = slice(delta, None)
        elif delta < 0:
            src[axis] = slice(-delta, None)
            dst[axis] = slice(0, delta)
    return tuple(dst), tuple(src)


def _uint8_dilate_aniso(
    image: torch.Tensor,
    iterations: int,
    *,
    radius_z: int,
    radius_y: int,
    radius_x: int,
) -> torch.Tensor:
    if image.dtype != torch.uint8:
        raise TypeError(f"_uint8_dilate_aniso expects uint8 tensor, got {image.dtype}")
    if image.ndim < 3:
        raise ValueError(f"_uint8_dilate_aniso expects at least 3 dimensions, got {image.ndim}")
    if iterations <= 0:
        return image
    if radius_z < 0 or radius_y < 0 or radius_x < 0:
        raise ValueError(
            f"radii must be >= 0, got z={radius_z} y={radius_y} x={radius_x}"
        )
    if radius_z == 0 and radius_y == 0 and radius_x == 0:
        return image

    x = image
    offsets = [
        (dz, dy, dx)
        for dz in range(-radius_z, radius_z + 1)
        for dy in range(-radius_y, radius_y + 1)
        for dx in range(-radius_x, radius_x + 1)
    ]
    for _ in range(iterations):
        out = torch.zeros_like(x)
        for dz, dy, dx in offsets:
            dst, src = _shift_slices(x.ndim, dz, dy, dx)
            out[dst] = torch.maximum(out[dst], x[src])
        x = out
    return x


def _uint8_dilate(image: torch.Tensor, iterations: int, radius: int = 1) -> torch.Tensor:
    return _uint8_dilate_aniso(
        image,
        iterations=iterations,
        radius_z=radius,
        radius_y=radius,
        radius_x=radius,
    )


def _dilate(mask: torch.Tensor, iterations: int, radius: int = 1) -> torch.Tensor:
    return _uint8_dilate(mask, iterations=iterations, radius=radius)


def _erode(mask: torch.Tensor, iterations: int, radius: int = 1) -> torch.Tensor:
    if mask.dtype != torch.uint8:
        raise TypeError(f"_erode expects uint8 mask, got {mask.dtype}")
    if iterations <= 0:
        return mask
    return 1 - _dilate(1 - mask, iterations=iterations, radius=radius)


def _erode_aniso(
    mask: torch.Tensor,
    iterations: int,
    *,
    radius_z: int,
    radius_y: int,
    radius_x: int,
) -> torch.Tensor:
    if mask.dtype != torch.uint8:
        raise TypeError(f"_erode_aniso expects uint8 mask, got {mask.dtype}")
    if iterations <= 0:
        return mask
    return 1 - _uint8_dilate_aniso(
        1 - mask,
        iterations=iterations,
        radius_z=radius_z,
        radius_y=radius_y,
        radius_x=radius_x,
    )


def _grayscale_dilate(image: torch.Tensor, iterations: int, radius: int = 1) -> torch.Tensor:
    return _uint8_dilate(image, iterations=iterations, radius=radius)


def _grayscale_erode(image: torch.Tensor, iterations: int, radius: int = 1) -> torch.Tensor:
    if image.dtype != torch.uint8:
        raise TypeError(f"_grayscale_erode expects uint8 image, got {image.dtype}")
    if iterations <= 0:
        return image
    return 255 - _grayscale_dilate(255 - image, iterations=iterations, radius=radius)


def _grayscale_close(image: torch.Tensor, iterations: int, radius: int = 1) -> torch.Tensor:
    if iterations <= 0:
        return image
    image = _grayscale_dilate(image, iterations=iterations, radius=radius)
    image = _grayscale_erode(image, iterations=iterations, radius=radius)
    return image


def _build_mask(
    volume: np.ndarray,
    missing_mask: np.ndarray,
    *,
    device: torch.device,
    empty_threshold: float,
    grow_threshold: float,
    initial_dilate: int,
    image_close_iterations: int,
    grow_z_iterations: int,
    grow_xy_iterations: int,
    close_iterations: int,
) -> np.ndarray:
    if volume.dtype != np.uint8:
        raise ValueError(f"_build_mask expects uint8 volume, got dtype={volume.dtype}")

    seed_np = (volume <= empty_threshold).astype(np.uint8, copy=False)
    seed_np[missing_mask] = 1
    mask = torch.as_tensor(seed_np, device=device)[None, None, ...]
    del seed_np

    mask = _dilate(mask, iterations=initial_dilate)
    work = torch.as_tensor(volume, device=device)
    if work.device.type == "cpu":
        work = work.clone()
    work.mul_(1 - mask[0, 0])
    work = _grayscale_close(
        work[None, None, ...],
        iterations=image_close_iterations,
    )
    low_value = (work <= grow_threshold).to(dtype=torch.uint8)
    del work

    grow_z_iterations = max(0, int(grow_z_iterations))
    grow_xy_iterations = max(0, int(grow_xy_iterations))
    grow_steps = max(grow_z_iterations, grow_xy_iterations)
    for i in range(grow_steps):
        rz = 1 if i < grow_z_iterations else 0
        rxy = 1 if i < grow_xy_iterations else 0
        expanded = _uint8_dilate_aniso(
            mask,
            iterations=1,
            radius_z=rz,
            radius_y=rxy,
            radius_x=rxy,
        )
        new_mask = expanded & (1 - mask)
        mask = mask | (new_mask & low_value)
        del expanded, new_mask

    mask = _uint8_dilate_aniso(
        mask,
        iterations=close_iterations,
        radius_z=0,
        radius_y=1,
        radius_x=1,
    )
    mask = _erode_aniso(
        mask,
        iterations=close_iterations,
        radius_z=0,
        radius_y=1,
        radius_x=1,
    )
    return (mask[0, 0] > 0).detach().cpu().numpy()


def _to_u8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _gray_to_rgb(image: np.ndarray) -> np.ndarray:
    gray = _to_u8_image(image)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(bool).astype(np.uint8) * 255)[:, :, None], 3, axis=2)


def _overlay_rgb(image: np.ndarray, mask: np.ndarray, *, alpha: float) -> np.ndarray:
    rgb = _gray_to_rgb(image).astype(np.float32)
    m = mask.astype(bool)
    if np.any(m):
        rgb[m] = rgb[m] * (1.0 - alpha) + 255.0 * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _write_layered_tif(
    *,
    out_path: Path,
    original_slice: np.ndarray,
    masked_out_slice: np.ndarray,
    keep_slice: np.ndarray,
    overlay_alpha: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [
        ("original", _gray_to_rgb(original_slice)),
        ("mask", _mask_to_rgb(keep_slice)),
        ("overlay", _overlay_rgb(original_slice, masked_out_slice, alpha=overlay_alpha)),
    ]
    with tifffile.TiffWriter(str(out_path)) as tw:
        for name, image in layers:
            tw.write(
                image,
                photometric="rgb",
                extratags=[(285, "s", 0, name, False)],
            )


def _overlay_mask_on_slice(
    image: np.ndarray,
    masked_out: np.ndarray,
    keep_mask: np.ndarray,
    *,
    alpha: float,
    include_mask_panel: bool,
) -> np.ndarray:
    original = _gray_to_rgb(image)
    overlay = _overlay_rgb(image, masked_out, alpha=alpha)
    if not include_mask_panel:
        return np.concatenate([original, overlay], axis=1)
    keep_panel = _mask_to_rgb(keep_mask)
    return np.concatenate([original, keep_panel, overlay], axis=1)


def _preview_full_z_values(z_range_full: tuple[int, int], step: int) -> list[int]:
    if step <= 0:
        return []
    z0, z1 = z_range_full
    first = ((z0 + step - 1) // step) * step
    return list(range(first, z1, step))


def _preview_slices(
    *,
    z_range_array: tuple[int, int],
    z_range_full: tuple[int, int],
    scale_z: float,
    step_full: int,
) -> list[tuple[int, int, int]]:
    previews = []
    seen: set[int] = set()
    for z_full in _preview_full_z_values(z_range_full, step_full):
        z_arr = int(round(float(z_full) / scale_z))
        z_arr = max(z_range_array[0], min(z_arr, z_range_array[1] - 1))
        if z_arr in seen:
            continue
        seen.add(z_arr)
        local_z = z_arr - z_range_array[0]
        actual_full = int(round(float(z_arr) * scale_z))
        previews.append((actual_full, z_arr, local_z))
    return previews


def _write_overlay_jpgs(
    *,
    out_dir: Path,
    volume: np.ndarray,
    masked_out: np.ndarray,
    keep_mask: np.ndarray,
    previews: list[tuple[int, int, int]],
    alpha: float,
    include_mask_panel: bool,
    quality: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for actual_full, z_arr, local_z in previews:
        overlay = _overlay_mask_on_slice(
            volume[local_z],
            masked_out[local_z],
            keep_mask[local_z],
            alpha=alpha,
            include_mask_panel=include_mask_panel,
        )
        path = out_dir / f"mask_overlay_z{actual_full:06d}_levelz{z_arr:06d}.jpg"
        if not cv2.imwrite(
            str(path),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
        ):
            raise RuntimeError(f"failed to write JPG preview: {path}")
        written += 1
    return written


def _write_preview_layered_tifs(
    *,
    out_dir: Path,
    volume: np.ndarray,
    masked_out: np.ndarray,
    keep_mask: np.ndarray,
    previews: list[tuple[int, int, int]],
    overlay_alpha: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for actual_full, z_arr, local_z in previews:
        path = out_dir / f"mask_layers_z{actual_full:06d}_levelz{z_arr:06d}.tif"
        _write_layered_tif(
            out_path=path,
            original_slice=volume[local_z],
            masked_out_slice=masked_out[local_z],
            keep_slice=keep_mask[local_z],
            overlay_alpha=overlay_alpha,
        )
        written += 1
    return written


def _write_mask_zarr(
    *,
    out_path: Path,
    keep_mask: np.ndarray,
    source_path: Path,
    scale_zyx: tuple[float, float, float],
    z_range_full: tuple[int, int],
    z_range_array: tuple[int, int],
) -> None:
    if keep_mask.ndim != 3:
        raise ValueError(f"expected 3D mask, got shape={keep_mask.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.mkdir(parents=True)

    shape = tuple(int(v) for v in keep_mask.shape)
    chunks = tuple(min(32, int(v)) for v in shape)
    zarray = {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": "|u1",
        "compressor": {"id": "zlib", "level": 1},
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
    }
    zattrs = {
        "source_zarr": str(source_path),
        "scale_zyx": [float(v) for v in scale_zyx],
        "z_range_full": [int(v) for v in z_range_full],
        "z_range_array": [int(v) for v in z_range_array],
        "mask_values": {"masked_out": 0, "keep": 255},
        "lasagna_mask_semantics": "uint8 mask: 255=keep, 0=mask_out",
    }
    (out_path / ".zarray").write_text(json.dumps(zarray, indent=2) + "\n")
    (out_path / ".zattrs").write_text(json.dumps(zattrs, indent=2) + "\n")

    Z, Y, X = shape
    cZ, cY, cX = chunks
    for z0 in range(0, Z, cZ):
        z1 = min(Z, z0 + cZ)
        iz = z0 // cZ
        for y0 in range(0, Y, cY):
            y1 = min(Y, y0 + cY)
            iy = y0 // cY
            for x0 in range(0, X, cX):
                x1 = min(X, x0 + cX)
                ix = x0 // cX
                src = keep_mask[z0:z1, y0:y1, x0:x1]
                if not np.any(src):
                    continue
                chunk = np.zeros(chunks, dtype=np.uint8)
                chunk[: z1 - z0, : y1 - y0, : x1 - x0] = src.astype(np.uint8) * 255
                chunk_path = out_path / str(iz) / str(iy) / str(ix)
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                chunk_path.write_bytes(zlib.compress(chunk.tobytes(order="C"), level=1))


def _axis_power2_mapping(src_dim: int, dst_dim: int) -> tuple[str, int]:
    src_dim = int(src_dim)
    dst_dim = int(dst_dim)
    if src_dim <= 0 or dst_dim <= 0:
        raise ValueError(f"invalid shape mapping src={src_dim} dst={dst_dim}")
    if src_dim == dst_dim:
        return "same", 1
    for k in range(1, 31):
        f = 1 << k
        if src_dim < dst_dim and abs(src_dim * f - dst_dim) < f:
            return "up", f
        if src_dim > dst_dim and abs(dst_dim * f - src_dim) < f:
            return "down", f
    raise ValueError(f"shape ratio is not a power-of-two zarr scale: src={src_dim} dst={dst_dim}")


def _mapped_axis_indices(src_dim: int, dst_dim: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    mode, factor = _axis_power2_mapping(src_dim, dst_dim)
    dst_idx = np.arange(int(start), int(end), dtype=np.int64)
    if mode == "same":
        src_idx = dst_idx
    elif mode == "up":
        src_idx = dst_idx // factor
    else:
        src_idx = dst_idx * factor
    valid = (src_idx >= 0) & (src_idx < int(src_dim))
    return src_idx, valid


def _omezarr_paths_for_group(vol: LasagnaVolume, group) -> tuple[Path, Path, int]:
    grad_path = (vol.path.parent / group.zarr_path).resolve()
    data_level = int(group.scaledown)
    if grad_path.name.isdigit():
        level_path = grad_path
        root_path = grad_path.parent
        data_level = int(grad_path.name)
    else:
        root_path = grad_path
        level_path = root_path / str(data_level)
    return root_path, level_path, data_level


def _grad_mag_omezarr_paths(lasagna_json: Path) -> tuple[LasagnaVolume, Path, Path, int]:
    vol = LasagnaVolume.load(lasagna_json)
    group, _ch_idx = vol.channel_group("grad_mag")
    root_path, level_path, data_level = _omezarr_paths_for_group(vol, group)
    return vol, root_path, level_path, data_level


def _numeric_omezarr_levels(root_path: Path) -> list[int]:
    if not root_path.exists():
        raise ValueError(f"OME-Zarr root does not exist: {root_path}")
    levels = sorted(int(p.name) for p in root_path.iterdir() if p.is_dir() and p.name.isdigit())
    if not levels:
        raise ValueError(f"OME-Zarr root has no numeric levels: {root_path}")
    return levels


def _level_chunks_zyx(level_path: Path) -> tuple[int, int, int]:
    arr = zarr.open(str(level_path), mode="r")
    chunks = tuple(int(v) for v in arr.chunks[-3:])
    if not chunks:
        raise ValueError(f"cannot read chunk size from {level_path}: chunks={arr.chunks}")
    return chunks


def _array_info_from_metadata(array_path: Path) -> tuple[tuple[int, ...], tuple[int, ...], str, dict]:
    metadata = _read_metadata(array_path)
    if not metadata:
        raise ValueError(f"missing zarr metadata at {array_path}")
    try:
        shape = tuple(int(v) for v in metadata["shape"])
        chunks = tuple(int(v) for v in metadata["chunks"])
        dtype = str(metadata["dtype"])
    except KeyError as exc:
        raise ValueError(f"zarr metadata at {array_path} missing {exc.args[0]!r}") from exc
    return shape, chunks, dtype, metadata


def _direct_zarr_meta(array_path: Path, metadata: dict | None = None) -> dict:
    metadata = _read_metadata(array_path) if metadata is None else metadata
    if int(metadata.get("zarr_format", 2)) != 2:
        raise ValueError(f"maskout direct chunk IO only supports zarr v2 arrays: {array_path}")
    comp_cfg = metadata.get("compressor")
    filters_cfg = metadata.get("filters") or []
    return {
        "path": str(array_path),
        "shape": tuple(int(v) for v in metadata["shape"]),
        "chunks": tuple(int(v) for v in metadata["chunks"]),
        "dtype": np.dtype(metadata["dtype"]),
        "fill_value": metadata.get("fill_value", 0),
        "order": str(metadata.get("order", "C")),
        "dim_sep": str(metadata.get("dimension_separator", ".")),
        "compressor": numcodecs.get_codec(comp_cfg) if comp_cfg else None,
        "filters": [numcodecs.get_codec(f) for f in filters_cfg],
    }


def _direct_chunk_path(meta: dict, chunk_coord: tuple[int, ...]) -> Path:
    key = str(meta["dim_sep"]).join(str(int(v)) for v in chunk_coord)
    return Path(str(meta["path"])) / key.replace("/", os.sep)


def _direct_fill_chunk(meta: dict) -> np.ndarray:
    return np.full(tuple(int(v) for v in meta["chunks"]), meta["fill_value"], dtype=meta["dtype"])


def _direct_read_chunk(meta: dict, chunk_coord: tuple[int, ...]) -> np.ndarray:
    path = _direct_chunk_path(meta, chunk_coord)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return _direct_fill_chunk(meta)
    comp = meta["compressor"]
    if comp is not None:
        raw = comp.decode(raw)
    for filt in reversed(meta["filters"]):
        raw = filt.decode(raw)
    return np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["chunks"], order=meta["order"]).copy()


def _direct_write_chunk(meta: dict, chunk_coord: tuple[int, ...], data: np.ndarray) -> None:
    arr = np.asarray(data, dtype=meta["dtype"])
    if tuple(int(v) for v in arr.shape) != tuple(int(v) for v in meta["chunks"]):
        raise ValueError(f"direct chunk write shape mismatch: got={arr.shape} expected={meta['chunks']}")
    raw = (np.ascontiguousarray(arr) if meta["order"] == "C" else np.asfortranarray(arr)).tobytes()
    for filt in meta["filters"]:
        raw = filt.encode(raw)
    comp = meta["compressor"]
    if comp is not None:
        raw = comp.encode(raw)
    path = _direct_chunk_path(meta, chunk_coord)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _read_mask_region_direct(
    *,
    meta: dict,
    mask_channel: int | None,
    rz0: int,
    rz1: int,
    ry0: int,
    ry1: int,
    rx0: int,
    rx1: int,
) -> np.ndarray:
    out = np.zeros((rz1 - rz0, ry1 - ry0, rx1 - rx0), dtype=np.uint8)
    shape = tuple(int(v) for v in meta["shape"])
    chunks = tuple(int(v) for v in meta["chunks"])
    if len(shape) == 3:
        cZ, cY, cX = chunks
        for zc0 in range((rz0 // cZ) * cZ, rz1, cZ):
            zc1 = min(shape[0], zc0 + cZ)
            oz0, oz1 = max(rz0, zc0), min(rz1, zc1)
            if oz1 <= oz0:
                continue
            iz = zc0 // cZ
            for yc0 in range((ry0 // cY) * cY, ry1, cY):
                yc1 = min(shape[1], yc0 + cY)
                oy0, oy1 = max(ry0, yc0), min(ry1, yc1)
                if oy1 <= oy0:
                    continue
                iy = yc0 // cY
                for xc0 in range((rx0 // cX) * cX, rx1, cX):
                    xc1 = min(shape[2], xc0 + cX)
                    ox0, ox1 = max(rx0, xc0), min(rx1, xc1)
                    if ox1 <= ox0:
                        continue
                    ix = xc0 // cX
                    chunk = _direct_read_chunk(meta, (iz, iy, ix))
                    out[oz0 - rz0:oz1 - rz0, oy0 - ry0:oy1 - ry0, ox0 - rx0:ox1 - rx0] = chunk[
                        oz0 - zc0:oz1 - zc0,
                        oy0 - yc0:oy1 - yc0,
                        ox0 - xc0:ox1 - xc0,
                    ]
    elif len(shape) == 4:
        ch = 0 if mask_channel is None else int(mask_channel)
        cC, cZ, cY, cX = chunks
        ic = ch // cC
        oc = ch - ic * cC
        for zc0 in range((rz0 // cZ) * cZ, rz1, cZ):
            zc1 = min(shape[1], zc0 + cZ)
            oz0, oz1 = max(rz0, zc0), min(rz1, zc1)
            if oz1 <= oz0:
                continue
            iz = zc0 // cZ
            for yc0 in range((ry0 // cY) * cY, ry1, cY):
                yc1 = min(shape[2], yc0 + cY)
                oy0, oy1 = max(ry0, yc0), min(ry1, yc1)
                if oy1 <= oy0:
                    continue
                iy = yc0 // cY
                for xc0 in range((rx0 // cX) * cX, rx1, cX):
                    xc1 = min(shape[3], xc0 + cX)
                    ox0, ox1 = max(rx0, xc0), min(rx1, xc1)
                    if ox1 <= ox0:
                        continue
                    ix = xc0 // cX
                    chunk = _direct_read_chunk(meta, (ic, iz, iy, ix))
                    out[oz0 - rz0:oz1 - rz0, oy0 - ry0:oy1 - ry0, ox0 - rx0:ox1 - rx0] = chunk[
                        oc,
                        oz0 - zc0:oz1 - zc0,
                        oy0 - yc0:oy1 - yc0,
                        ox0 - xc0:ox1 - xc0,
                    ]
    else:
        raise ValueError(f"mask zarr must be 3D ZYX or 4D CZYX, got shape={shape}")
    return out


def _read_mask_block_resampled_direct(
    *,
    mask_meta: dict,
    mask_channel: int | None,
    target_shape: tuple[int, int, int],
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    src_shape = tuple(int(v) for v in mask_meta["shape"][-3:])
    zi, zv = _mapped_axis_indices(src_shape[0], target_shape[0], z0, z1)
    yi, yv = _mapped_axis_indices(src_shape[1], target_shape[1], y0, y1)
    xi, xv = _mapped_axis_indices(src_shape[2], target_shape[2], x0, x1)
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
    if not (np.any(zv) and np.any(yv) and np.any(xv)):
        return out

    z_valid = zi[zv]
    y_valid = yi[yv]
    x_valid = xi[xv]
    rz0, rz1 = int(z_valid.min()), int(z_valid.max()) + 1
    ry0, ry1 = int(y_valid.min()), int(y_valid.max()) + 1
    rx0, rx1 = int(x_valid.min()), int(x_valid.max()) + 1
    slab = _read_mask_region_direct(
        meta=mask_meta,
        mask_channel=mask_channel,
        rz0=rz0,
        rz1=rz1,
        ry0=ry0,
        ry1=ry1,
        rx0=rx0,
        rx1=rx1,
    )
    values = slab[np.ix_(z_valid - rz0, y_valid - ry0, x_valid - rx0)] != 0
    out[np.ix_(zv, yv, xv)] = values
    return out


def _init_maskout_worker(
    grad_level_path: str,
    mask_array_path: str,
    grad_metadata: dict,
    mask_metadata: dict,
    mask_shape: tuple[int, ...],
    mask_channel: int | None,
    target_shape: tuple[int, int, int],
) -> None:
    global _MASKOUT_GRAD_META, _MASKOUT_MASK_META, _MASKOUT_CFG
    _MASKOUT_GRAD_META = _direct_zarr_meta(Path(grad_level_path), grad_metadata)
    _MASKOUT_MASK_META = _direct_zarr_meta(Path(mask_array_path), mask_metadata)
    _MASKOUT_CFG = {
        "mask_shape": tuple(int(v) for v in mask_shape),
        "mask_channel": mask_channel,
        "target_shape": tuple(int(v) for v in target_shape),
    }


def _maskout_chunk_worker(item: tuple[int, int, int, int, int, int]) -> tuple[tuple[int, int, int, int, int, int], int]:
    if _MASKOUT_GRAD_META is None or _MASKOUT_MASK_META is None or _MASKOUT_CFG is None:
        raise RuntimeError("maskout worker was not initialized")
    z0, z1, y0, y1, x0, x1 = item
    cZ, cY, cX = tuple(int(v) for v in _MASKOUT_GRAD_META["chunks"])
    chunk_coord = (z0 // cZ, y0 // cY, x0 // cX)
    grad_chunk = _direct_read_chunk(_MASKOUT_GRAD_META, chunk_coord)
    dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
    gm = grad_chunk[:dz, :dy, :dx]
    keep = _read_mask_block_resampled_direct(
        mask_meta=_MASKOUT_MASK_META,
        mask_channel=_MASKOUT_CFG["mask_channel"],
        target_shape=_MASKOUT_CFG["target_shape"],
        z0=z0,
        z1=z1,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
    )
    out = np.where((gm != 0) & keep, gm, 0).astype(np.uint8, copy=False)
    if np.array_equal(out, gm):
        return item, 0
    grad_chunk[:dz, :dy, :dx] = out
    _direct_write_chunk(_MASKOUT_GRAD_META, chunk_coord, grad_chunk)
    return item, 1


def _run_maskout_workers(
    work: list[tuple[int, int, int, int, int, int]],
    *,
    workers: int,
    grad_level_path: Path,
    mask_array_path: Path,
    grad_metadata: dict,
    mask_metadata: dict,
    mask_shape: tuple[int, ...],
    mask_channel: int | None,
    target_shape: tuple[int, int, int],
) -> int:
    if not work:
        return 0
    if workers <= 0:
        workers = max(1, multiprocessing.cpu_count())
    workers = min(max(1, int(workers)), len(work))

    changed = 0
    done_count = 0
    progress_t0 = time.time()
    _print_progress(prefix="[maskout apply]", done=0, total=len(work), t0=progress_t0)
    mp_ctx = multiprocessing.get_context("fork") if sys.platform != "win32" else multiprocessing.get_context("spawn")
    with mp_ctx.Pool(
        processes=workers,
        initializer=_init_maskout_worker,
        initargs=(
            str(grad_level_path),
            str(mask_array_path),
            grad_metadata,
            mask_metadata,
            tuple(int(v) for v in mask_shape),
            mask_channel,
            tuple(int(v) for v in target_shape),
        ),
    ) as pool:
        for _done_item, changed_one in pool.imap_unordered(_maskout_chunk_worker, work, chunksize=1):
            changed += int(changed_one)
            done_count += 1
            _print_progress(prefix="[maskout apply]", done=done_count, total=len(work), t0=progress_t0)
    _print_progress(prefix="[maskout apply]", done=len(work), total=len(work), t0=progress_t0)
    print("", flush=True)
    return changed


def _mark_manifest_v2(vol: LasagnaVolume, *, label: str) -> None:
    if vol.version != LASAGNA_VOLUME_VERSION:
        old_version = vol.version
        vol.version = LASAGNA_VOLUME_VERSION
        vol.save()
        print(
            f"[{label}] upgraded lasagna manifest version {old_version} -> {LASAGNA_VOLUME_VERSION}",
            flush=True,
        )
    else:
        vol.save()


def _apply_external_keep_mask(
    *,
    grad_level_path: Path,
    mask_array_path: Path,
    mask_shape: tuple[int, ...],
    mask_channel: int | None,
    workers: int,
) -> tuple[int, int, int]:
    grad_shape_all, grad_chunks_all, grad_dtype, metadata = _array_info_from_metadata(grad_level_path)
    if len(grad_shape_all) != 3:
        raise ValueError(f"grad_mag level must be 3D ZYX, got shape={grad_shape_all}")
    if np.dtype(grad_dtype) != np.dtype(np.uint8):
        raise ValueError(f"grad_mag dtype must be uint8, got {grad_dtype}")
    if len(mask_shape) not in {3, 4}:
        raise ValueError(f"mask zarr must be 3D ZYX or 4D CZYX, got shape={mask_shape}")
    if len(mask_shape) == 4 and mask_channel is not None:
        if mask_channel < 0 or mask_channel >= int(mask_shape[0]):
            raise ValueError(f"--mask-channel {mask_channel} out of range for mask shape={mask_shape}")

    target_shape = tuple(int(v) for v in grad_shape_all)
    source_shape = tuple(int(v) for v in mask_shape[-3:])
    mappings = [_axis_power2_mapping(s, d) for s, d in zip(source_shape, target_shape)]
    print(
        f"[maskout] grad_mag_shape={target_shape} mask_shape={source_shape} "
        f"mapping_zyx={mappings}",
        flush=True,
    )

    chunks = tuple(int(v) for v in grad_chunks_all)
    skipped_missing = 0
    total = 0
    work: list[tuple[int, int, int, int, int, int]] = []
    Z, Y, X = target_shape
    cZ, cY, cX = chunks
    for z0 in range(0, Z, cZ):
        z1 = min(Z, z0 + cZ)
        iz = z0 // cZ
        for y0 in range(0, Y, cY):
            y1 = min(Y, y0 + cY)
            iy = y0 // cY
            for x0 in range(0, X, cX):
                x1 = min(X, x0 + cX)
                ix = x0 // cX
                total += 1
                if not _chunk_exists(grad_level_path, (iz, iy, ix), metadata):
                    skipped_missing += 1
                    continue
                work.append((z0, z1, y0, y1, x0, x1))
    print(
        f"[maskout] applying mask with direct chunk IO workers={workers or multiprocessing.cpu_count()} "
        f"chunks={len(work)} skipped_missing={skipped_missing} total_chunks={total}",
        flush=True,
    )
    changed = _run_maskout_workers(
        work,
        workers=workers,
        grad_level_path=grad_level_path,
        mask_array_path=mask_array_path,
        grad_metadata=metadata,
        mask_metadata=_read_metadata(mask_array_path),
        mask_shape=mask_shape,
        mask_channel=mask_channel,
        target_shape=target_shape,
    )
    return total, skipped_missing, changed


def run_maskout(args: argparse.Namespace) -> None:
    lasagna_json = Path(args.lasagna_json)
    mask_array_path = Path(args.mask_zarr)
    if args.mask_array is not None:
        mask_array_path = mask_array_path / str(args.mask_array)
    mask_shape, _mask_chunks, mask_dtype, _mask_metadata = _array_info_from_metadata(mask_array_path)
    if np.dtype(mask_dtype) != np.dtype(np.uint8):
        raise ValueError(f"mask dtype must be uint8, got {mask_dtype}")
    mask_channel = None
    if len(mask_shape) == 4:
        mask_channel = int(args.mask_channel)
    _vol, grad_root, grad_level, data_level = _grad_mag_omezarr_paths(lasagna_json)
    levels = _numeric_omezarr_levels(grad_root)
    n_levels = max(levels) + 1
    data_chunks = _level_chunks_zyx(grad_level)
    print(
        f"[maskout] lasagna={lasagna_json} grad_mag={grad_level} "
        f"data_level={data_level} n_levels={n_levels} data_chunks={data_chunks} "
        f"mask={mask_array_path}",
        flush=True,
    )
    t0 = time.perf_counter()
    total, skipped, changed = _apply_external_keep_mask(
        grad_level_path=grad_level,
        mask_array_path=mask_array_path,
        mask_shape=mask_shape,
        mask_channel=mask_channel,
        workers=int(args.workers),
    )
    mask_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[maskout] masked grad_mag chunks changed={changed} skipped_missing={skipped} "
        f"total_chunks={total} in {mask_ms:.1f}ms",
        flush=True,
    )
    pyr_t0 = time.perf_counter()
    print("[maskout] rebuilding grad_mag lower scales ...", flush=True)
    clear_coarser_levels(grad_root, data_level, n_levels)
    build_scalar_omezarr_pyramid(
        grad_root,
        data_level,
        n_levels,
        label="grad_mag",
        workers=int(args.workers),
        force=True,
        zero_overrides=True,
    )
    pyr_ms = (time.perf_counter() - pyr_t0) * 1000.0
    print(
        f"[maskout] rebuilt grad_mag lower scales in {pyr_ms:.1f}ms",
        flush=True,
    )
    _mark_manifest_v2(_vol, label="maskout")


def add_create_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("zarr_path", type=Path, help="Path to a zarr array or group.")
    parser.add_argument(
        "-o",
        "--vis-dir",
        type=Path,
        default=Path("empty_region_mask_vis"),
        help="Visualization output directory. Contains jpg/ and tif/ subdirs.",
    )
    parser.add_argument(
        "--zarr-output",
        type=Path,
        default=None,
        help="Optional plain single-scale zarr output for the uint8 mask volume (255=keep, 0=mask_out).",
    )
    parser.add_argument(
        "--array",
        default=None,
        help="Array key when zarr_path is a group (default: 0, else first entry).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index for 4D CZYX arrays (default: 0).",
    )
    parser.add_argument(
        "--z-range",
        type=_parse_z_range,
        default=None,
        metavar="START,END",
        help="Optional full-resolution Z slab to process, end-exclusive. Default: whole volume.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for morphology (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=0.0,
        help="Initial black/empty voxel threshold (default: 0).",
    )
    parser.add_argument(
        "--grow-threshold",
        type=float,
        default=50.0,
        help="Expansion accepts new voxels with value <= this threshold (default: 50).",
    )
    parser.add_argument(
        "--initial-dilate",
        type=int,
        default=2,
        help="Initial 3D dilation iterations for the black/missing mask (default: 2).",
    )
    parser.add_argument(
        "--grow-iterations",
        type=int,
        default=None,
        help="Compatibility override: set both grow-z and grow-xy iterations to this value.",
    )
    parser.add_argument(
        "--grow-z-iterations",
        type=int,
        default=5,
        help="Threshold-gated growth iterations in Z after the initial mask (default: 5).",
    )
    parser.add_argument(
        "--grow-xy-iterations",
        type=int,
        default=150,
        help="Threshold-gated growth iterations in X/Y after the initial mask (default: 150).",
    )
    parser.add_argument(
        "--image-close-iterations",
        type=int,
        default=1,
        help="Grayscale image closing iterations after blacking out the initial mask (default: 1).",
    )
    parser.add_argument(
        "--close-iterations",
        type=int,
        default=2,
        help="Final XY-only closing iterations: dilate N then erode N (default: 2).",
    )
    parser.add_argument(
        "--jpg-step-full",
        type=int,
        default=1000,
        help="JPG preview stride in full-resolution Z coordinates (default: 1000).",
    )
    parser.add_argument(
        "--jpg-mask-alpha",
        type=float,
        default=0.15,
        help="White mask overlay opacity for JPG previews (default: 0.15).",
    )
    parser.add_argument(
        "--no-jpg-mask-panel",
        action="store_true",
        help="Disable the exact binary mask panel in JPG previews.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=100,
        help="JPEG quality for preview overlays (default: 100).",
    )


def run_create(args: argparse.Namespace) -> None:

    t0 = time.perf_counter()
    arr, array_path = _open_zarr_array(args.zarr_path, args.array)
    if len(arr.shape) not in {3, 4}:
        raise ValueError(f"expected 3D ZYX or 4D CZYX zarr array, got shape={arr.shape}")
    if len(arr.shape) == 3:
        channel = None
    else:
        if args.channel < 0 or args.channel >= int(arr.shape[0]):
            raise ValueError(f"--channel {args.channel} out of range for shape={arr.shape}")
        channel = args.channel
    scale_zyx = _infer_zyx_scale(arr, array_path)
    z_range_full = args.z_range
    if z_range_full is None:
        z_range_full = _default_full_z_range(
            array_z=int(arr.shape[-3]),
            scale_z=scale_zyx[0],
        )
    z_range_array = _full_z_range_to_array(
        z_range_full,
        scale_z=scale_zyx[0],
        array_z=int(arr.shape[-3]),
    )

    print(
        f"[mask] zarr={array_path} shape={tuple(arr.shape)} chunks={tuple(arr.chunks)} "
        f"dtype={arr.dtype} scale_zyx={scale_zyx} "
        f"z_range_full={z_range_full} z_range_array={z_range_array} "
        f"device={args.device}",
        flush=True,
    )

    read_t0 = time.perf_counter()
    volume = _read_slab(arr, z_range=z_range_array, channel=channel)
    missing = _missing_chunk_mask_for_slab(
        array_path=array_path,
        shape=tuple(int(v) for v in arr.shape),
        chunks=tuple(int(v) for v in arr.chunks),
        z_range=z_range_array,
        channel=channel,
    )
    read_ms = (time.perf_counter() - read_t0) * 1000.0

    dev = torch.device(args.device)
    grow_z_iterations = args.grow_z_iterations
    grow_xy_iterations = args.grow_xy_iterations
    if args.grow_iterations is not None:
        grow_z_iterations = args.grow_iterations
        grow_xy_iterations = args.grow_iterations
    mask_t0 = time.perf_counter()
    masked_out = _build_mask(
        volume,
        missing,
        device=dev,
        empty_threshold=args.empty_threshold,
        grow_threshold=args.grow_threshold,
        initial_dilate=args.initial_dilate,
        image_close_iterations=args.image_close_iterations,
        grow_z_iterations=grow_z_iterations,
        grow_xy_iterations=grow_xy_iterations,
        close_iterations=args.close_iterations,
    )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    mask_ms = (time.perf_counter() - mask_t0) * 1000.0
    keep_mask = ~masked_out

    zarr_write_ms = 0.0
    if args.zarr_output is not None:
        zarr_t0 = time.perf_counter()
        _write_mask_zarr(
            out_path=args.zarr_output,
            keep_mask=keep_mask,
            source_path=array_path,
            scale_zyx=scale_zyx,
            z_range_full=z_range_full,
            z_range_array=z_range_array,
        )
        zarr_write_ms = (time.perf_counter() - zarr_t0) * 1000.0

    jpg_dir = args.vis_dir / "jpg"
    preview_tif_dir = args.vis_dir / "tif"
    previews = _preview_slices(
        z_range_array=z_range_array,
        z_range_full=z_range_full,
        scale_z=scale_zyx[0],
        step_full=args.jpg_step_full,
    )
    preview_t0 = time.perf_counter()
    jpg_count = _write_overlay_jpgs(
        out_dir=jpg_dir,
        volume=volume,
        masked_out=masked_out,
        keep_mask=keep_mask,
        previews=previews,
        alpha=args.jpg_mask_alpha,
        include_mask_panel=not args.no_jpg_mask_panel,
        quality=max(1, min(100, args.jpg_quality)),
    )
    preview_tif_count = _write_preview_layered_tifs(
        out_dir=preview_tif_dir,
        volume=volume,
        masked_out=masked_out,
        keep_mask=keep_mask,
        previews=previews,
        overlay_alpha=args.jpg_mask_alpha,
    )
    preview_ms = (time.perf_counter() - preview_t0) * 1000.0

    masked_out_frac = float(masked_out.mean()) if masked_out.size else 0.0
    keep_frac = float(keep_mask.mean()) if keep_mask.size else 0.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[mask] masked_out_frac={masked_out_frac:.4f} keep_frac={keep_frac:.4f} "
        f"read={read_ms:.1f}ms mask={mask_ms:.1f}ms "
        f"zarr={zarr_write_ms:.1f}ms previews={preview_ms:.1f}ms "
        f"total={total_ms:.1f}ms",
        flush=True,
    )
    if args.zarr_output is not None:
        print(
            f"[mask] wrote zarr mask to {args.zarr_output} "
            "(255=keep, 0=mask_out)",
            flush=True,
        )
    print(f"[mask] wrote {jpg_count} JPG previews to {jpg_dir}", flush=True)
    print(
        f"[mask] wrote {preview_tif_count} layered TIFF previews to {preview_tif_dir}",
        flush=True,
    )


def add_maskout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lasagna-json", required=True, type=Path, help="Lasagna .lasagna.json manifest to update in place.")
    parser.add_argument("--mask-zarr", required=True, type=Path, help="3D uint8 mask zarr; 0 masks out, nonzero keeps.")
    parser.add_argument("--mask-array", default=None, help="Optional array key below --mask-zarr.")
    parser.add_argument("--mask-channel", type=int, default=0, help="Channel index for 4D CZYX mask arrays.")
    parser.add_argument("--workers", type=int, default=0, help="Pyramid rebuild workers (default: CPU count).")


def _normal_pair_info(vol: LasagnaVolume) -> tuple[Path, Path, Path, Path, int] | None:
    channels = set(vol.all_channels())
    if "nx" not in channels and "ny" not in channels:
        return None
    if "nx" not in channels or "ny" not in channels:
        raise ValueError("normal pyramid rebuild requires both nx and ny channels")
    nx_group, _ = vol.channel_group("nx")
    ny_group, _ = vol.channel_group("ny")
    nx_root, nx_level, nx_data_level = _omezarr_paths_for_group(vol, nx_group)
    ny_root, ny_level, ny_data_level = _omezarr_paths_for_group(vol, ny_group)
    if nx_data_level != ny_data_level:
        raise ValueError(f"nx/ny data levels differ: nx={nx_data_level} ny={ny_data_level}")
    if nx_root.resolve() == ny_root.resolve():
        raise ValueError("nx/ny normal rebuild expects separate OME-Zarr roots")
    return nx_root, ny_root, nx_level, ny_level, nx_data_level


def run_rebuild_scales(args: argparse.Namespace) -> None:
    lasagna_json = Path(args.lasagna_json)
    vol = LasagnaVolume.load(lasagna_json)
    workers = int(args.workers)
    processed_roots: set[Path] = set()

    normal_info = _normal_pair_info(vol)
    if normal_info is not None:
        nx_root, ny_root, nx_level, _ny_level, data_level = normal_info
        nx_levels = _numeric_omezarr_levels(nx_root)
        ny_levels = _numeric_omezarr_levels(ny_root)
        n_levels = max(max(nx_levels), max(ny_levels)) + 1
        data_chunks = _level_chunks_zyx(nx_level)
        print(
            f"[rebuild-scales] normal nx={nx_root} ny={ny_root} "
            f"data_level={data_level} n_levels={n_levels} data_chunks={data_chunks}",
            flush=True,
        )
        build_normal_omezarr_pyramid(
            nx_root,
            ny_root,
            data_level,
            n_levels,
            label="normal",
            workers=workers,
            force=True,
        )
        processed_roots.add(nx_root.resolve())
        processed_roots.add(ny_root.resolve())

    for group_name, group in vol.groups.items():
        root, level_path, data_level = _omezarr_paths_for_group(vol, group)
        root_key = root.resolve()
        if root_key in processed_roots:
            continue
        if any(ch in {"nx", "ny"} for ch in group.channels):
            raise ValueError(
                f"normal channel group {group_name!r} was not handled as an nx/ny pair"
            )
        levels = _numeric_omezarr_levels(root)
        n_levels = max(levels) + 1
        data_chunks = _level_chunks_zyx(level_path)
        label = ",".join(group.channels) if group.channels else group_name
        print(
            f"[rebuild-scales] scalar {label} root={root} "
            f"data_level={data_level} n_levels={n_levels} data_chunks={data_chunks}",
            flush=True,
        )
        build_scalar_omezarr_pyramid(
            root,
            data_level,
            n_levels,
            label=label,
            workers=workers,
            force=True,
            zero_overrides=("grad_mag" in group.channels),
        )
        processed_roots.add(root_key)

    _mark_manifest_v2(vol, label="rebuild-scales")


def add_rebuild_scales_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lasagna-json", required=True, type=Path, help="Lasagna .lasagna.json manifest to upgrade and rebuild.")
    parser.add_argument("--workers", type=int, default=0, help="Pyramid rebuild workers (default: CPU count).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create threshold-derived zarr masks and apply them to Lasagna grad_mag data.",
        epilog=(
            "commands:\n"
            "  create   Create a mask zarr and previews from an input zarr.\n"
            "  maskout  Apply a mask zarr to grad_mag in a Lasagna manifest.\n"
            "  rebuild-scales  Rebuild lower OME-Zarr scales for every zarr in a Lasagna manifest."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="{create,maskout,rebuild-scales}")
    p_create = sub.add_parser("create", help="Create a mask zarr and previews from an input zarr.")
    add_create_args(p_create)
    p_create.set_defaults(func=run_create)
    p_maskout = sub.add_parser("maskout", help="Apply a mask zarr to grad_mag in a Lasagna manifest.")
    add_maskout_args(p_maskout)
    p_maskout.set_defaults(func=run_maskout)
    p_rebuild = sub.add_parser("rebuild-scales", help="Rebuild lower OME-Zarr scales for every zarr in a Lasagna manifest.")
    add_rebuild_scales_args(p_rebuild)
    p_rebuild.set_defaults(func=run_rebuild_scales)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
