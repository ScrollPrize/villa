"""Pipeline: binary label TIFF → binary zarr → vc_gen_normalgrids →
vc_ngrids --fit-normals → Python lasagna zarr assembly → fit.py.

The lasagna-format zarr is a flat zarr.Array (C, Z, Y, X) uint8 with channels
[cos, grad_mag, nx, ny, pred_dt] and preprocess_params metadata,
ready to be consumed by fit_data.load_3d().

Usage
-----
  python labels_to_lasagna_normals.py \
      --input labels.tif \
      --work-dir ./work \
      --output normals_lasagna.zarr \
      --step 4 \
      --density 128
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import distance_transform_edt


TAG = "[labels_to_lasagna_normals]"


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess, printing its label and streaming output."""
    print(f"{TAG} running: {label}", flush=True)
    print(f"{TAG}   cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with rc={proc.returncode}")


def _write_binary_zarr(vol: np.ndarray, out_path: Path, chunk_size: int) -> None:
    """Write binary uint8 volume as zarr group with dataset '0'."""
    root = zarr.open_group(str(out_path), mode="w", zarr_format=2)
    ds = root.create_array(
        "0",
        shape=vol.shape,
        chunks=(chunk_size, chunk_size, chunk_size),
        dtype=np.uint8,
        overwrite=True,
    )
    ds[:] = vol
    print(f"{TAG} binary zarr written: {out_path}  shape={vol.shape}", flush=True)


def _downsample_any(vol: np.ndarray, step: int) -> np.ndarray:
    """Downsample a 3D binary volume by *step* using max-pooling.

    Pads each axis to be divisible by *step*, then reshapes and takes max
    along the pooling dimensions.
    """
    if step == 1:
        return vol
    z, y, x = vol.shape
    pz = -z % step
    py = -y % step
    px = -x % step
    if pz or py or px:
        vol = np.pad(vol, ((0, pz), (0, py), (0, px)))
    z2, y2, x2 = vol.shape
    return (
        vol.reshape(z2 // step, step, y2 // step, step, x2 // step, step)
        .max(axis=(1, 3, 5))
    )


def _downsample_mean(vol: np.ndarray, step: int) -> np.ndarray:
    """Downsample a 3D float volume by *step* using mean-pooling."""
    if step == 1:
        return vol
    z, y, x = vol.shape
    pz = -z % step
    py = -y % step
    px = -x % step
    if pz or py or px:
        vol = np.pad(vol, ((0, pz), (0, py), (0, px)))
    z2, y2, x2 = vol.shape
    return (
        vol.reshape(z2 // step, step, y2 // step, step, x2 // step, step)
        .mean(axis=(1, 3, 5))
    )


def _compute_pred_dt(binary: np.ndarray, step: int) -> np.ndarray:
    """Compute distance-transform of inverted binary mask at full res, downscale, encode uint8.

    DT = distance from each voxel to the nearest foreground (label=1) surface.
    0 on the surface, increasing away from it.
    Stored as raw distance in voxels, clamped to 255 (matching preprocess_cos_omezarr.py).
    """
    print(f"{TAG} computing distance transform at full res {binary.shape}...", flush=True)
    dt = distance_transform_edt(~binary.astype(bool)).astype(np.float32)
    dt_max = float(dt.max())
    print(f"{TAG} DT max = {dt_max:.1f} voxels", flush=True)

    # Downscale to step resolution via mean-pooling
    dt_ds = _downsample_mean(dt, step).astype(np.float32)

    # Encode: raw distance clamped to 255 (same as preprocess_cos_omezarr.py)
    dt_u8 = np.clip(dt_ds, 0, 255).astype(np.uint8)

    print(f"{TAG} pred_dt downsampled shape={dt_u8.shape}  "
          f"max_dist={float(dt_ds.max()):.1f}", flush=True)
    return dt_u8


def _write_lasagna_zarr(
    ngrids_zarr: Path,
    binary: np.ndarray,
    ignore: np.ndarray,
    output: Path,
    step: int,
    density: int,
    chunk_size: int,
    pred_dt_u8: np.ndarray | None = None,
) -> None:
    """Read ngrids x/0, y/0, z/0 + binary pred + ignore mask → write flat lasagna zarr.Array."""
    # Read normals from ngrids output (raw, NOT hemisphere-encoded)
    ng = zarr.open(str(ngrids_zarr), mode="r")
    nx_vol = np.array(ng["x/0"])  # (Z_ds, Y_ds, X_ds) uint8
    ny_vol = np.array(ng["y/0"])
    nz_vol = np.array(ng["z/0"])

    # Apply +z hemisphere encoding: flip (nx, ny) where nz < 0
    # vc_ngrids stores raw aligned normals; the optimizer expects hemisphere-encoded
    # (nx, ny) with nz = sqrt(1 - nx² - ny²) >= 0.
    nx_f = (nx_vol.astype(np.float32) - 128.0) / 127.0
    ny_f = (ny_vol.astype(np.float32) - 128.0) / 127.0
    nz_f = (nz_vol.astype(np.float32) - 128.0) / 127.0
    flip = np.where(nz_f < 0, np.float32(-1.0), np.float32(1.0))
    nx_f *= flip
    ny_f *= flip
    n_flipped = int((flip < 0).sum())
    print(f"{TAG} hemisphere encoding: flipped {n_flipped}/{flip.size} normals "
          f"({100.0*n_flipped/max(1,flip.size):.1f}%)", flush=True)
    nx_vol = np.clip(np.round(nx_f * 127.0 + 128.0), 0, 255).astype(np.uint8)
    ny_vol = np.clip(np.round(ny_f * 127.0 + 128.0), 0, 255).astype(np.uint8)

    # Downsample binary prediction to step resolution
    pred_ds = _downsample_any(binary, step)
    # Downsample ignore mask with max-pooling (conservative: any ignore voxel → block invalid)
    ignore_ds = _downsample_any(ignore, step)

    # Ensure shapes match (ngrids may be full-volume sized)
    dz, dy, dx = pred_ds.shape
    nx_vol = nx_vol[:dz, :dy, :dx]
    ny_vol = ny_vol[:dz, :dy, :dx]
    ignore_ds = ignore_ds[:dz, :dy, :dx]

    mask = pred_ds > 0
    valid = ignore_ds == 0  # valid where NOT ignore (bg + pred are valid)

    # Build channels
    # cos: 255 where prediction, 0 elsewhere
    cos_ch = np.where(mask, np.uint8(255), np.uint8(0))
    # grad_mag: density where valid (bg + pred), 0 where ignore (encodes invalid)
    grad_mag_ch = np.where(valid, np.uint8(density), np.uint8(0))

    nx_ch = nx_vol.astype(np.uint8)
    ny_ch = ny_vol.astype(np.uint8)

    channels = [cos_ch, grad_mag_ch, nx_ch, ny_ch]
    channel_names = ["cos", "grad_mag", "nx", "ny"]

    if pred_dt_u8 is not None:
        pred_dt_u8 = pred_dt_u8[:dz, :dy, :dx]
        channels.append(pred_dt_u8)
        channel_names.append("pred_dt")

    out_vol = np.stack(channels, axis=0)  # (C, Z, Y, X)

    if output.exists():
        shutil.rmtree(output)

    arr = zarr.open(
        str(output),
        mode="w",
        shape=out_vol.shape,
        chunks=(1, chunk_size, chunk_size, chunk_size),
        dtype=np.uint8,
        zarr_format=2,
    )
    arr[:] = out_vol
    arr.attrs["preprocess_params"] = {
        "scaledown": step,
        "channels": channel_names,
        "grad_mag_encode_scale": float(density),
    }
    print(
        f"{TAG} lasagna zarr written: {output}  shape={out_vol.shape}  "
        f"pred voxels={int(mask.sum())}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Label TIFF → binary zarr → vc normal grids → lasagna normals zarr",
    )
    p.add_argument("--input", required=True,
                    help="Multi-layer label TIFF (ZYX): 0=bg, 1=pred, 2=ignore")
    p.add_argument("--work-dir", required=True,
                    help="Working directory for intermediate files")
    p.add_argument("--output", required=True,
                    help="Output lasagna-format normals zarr (C,Z,Y,X)")
    p.add_argument("--step", type=int, default=4,
                    help="Sample step for vc_ngrids (also becomes scaledown in lasagna zarr, default: 4)")
    p.add_argument("--density", type=int, default=128,
                    help="grad_mag channel value where prediction is active (default: 128)")
    p.add_argument("--sparse-volume", type=int, default=1,
                    help="vc_gen_normalgrids --sparse-volume (default: 1)")
    p.add_argument("--chunk-size", type=int, default=64,
                    help="Zarr chunk size per axis")
    p.add_argument("--skip-gen-normalgrids", action="store_true",
                    help="Skip vc_gen_normalgrids (reuse existing grids in work-dir)")
    p.add_argument("--skip-fit-normals", action="store_true",
                    help="Skip vc_ngrids --fit-normals (reuse existing ngrids zarr)")
    p.add_argument("--no-pred-dt", action="store_true",
                    help="Skip computing pred_dt channel (distance transform)")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    output_path = Path(args.output)
    work_dir.mkdir(parents=True, exist_ok=True)

    binary_zarr_path = work_dir / "binary.zarr"
    grids_path = work_dir / "normalgrids"
    ngrids_zarr_path = work_dir / "ngrids_normals.zarr"

    # -- Step 1: Read TIFF, write binary zarr --------------------------------
    print(f"{TAG} step 1: reading {input_path}", flush=True)
    vol = tifffile.imread(str(input_path))
    if vol.ndim == 2:
        vol = vol[np.newaxis]
    assert vol.ndim == 3, f"expected 3D volume, got shape {vol.shape}"

    n_bg = int((vol == 0).sum())
    n_fg = int((vol == 1).sum())
    n_ign = int((vol == 2).sum())
    print(f"{TAG} volume shape={vol.shape}  bg={n_bg}  pred={n_fg}  ignore={n_ign}", flush=True)

    binary = (vol == 1).astype(np.uint8)
    ignore = (vol == 2).astype(np.uint8)

    if not args.skip_gen_normalgrids:
        _write_binary_zarr(binary, binary_zarr_path, args.chunk_size)

    # -- Step 2: vc_gen_normalgrids ------------------------------------------
    if not args.skip_gen_normalgrids:
        grids_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "vc_gen_normalgrids",
            "-i", str(binary_zarr_path),
            "-o", str(grids_path),
        ]
        if args.sparse_volume > 1:
            cmd += ["--sparse-volume", str(args.sparse_volume)]
        _run(cmd, "vc_gen_normalgrids")
    else:
        print(f"{TAG} step 2: skipping vc_gen_normalgrids (--skip-gen-normalgrids)", flush=True)

    # -- Step 3: vc_ngrids --fit-normals -------------------------------------
    if not args.skip_fit_normals:
        if ngrids_zarr_path.exists():
            shutil.rmtree(ngrids_zarr_path)
        cmd = [
            "vc_ngrids",
            "-i", str(grids_path),
            "--fit-normals",
            "--output-zarr", str(ngrids_zarr_path),
            "--step", str(args.step),
        ]
        _run(cmd, "vc_ngrids --fit-normals")
    else:
        print(f"{TAG} step 3: skipping vc_ngrids --fit-normals (--skip-fit-normals)", flush=True)

    # -- Step 4: Compute pred_dt (distance transform) ------------------------
    if not args.no_pred_dt:
        pred_dt_u8 = _compute_pred_dt(binary, args.step)
    else:
        pred_dt_u8 = None
        print(f"{TAG} step 4: skipping pred_dt (--no-pred-dt)", flush=True)

    # -- Step 5: Assemble lasagna zarr from ngrids + binary ------------------
    _write_lasagna_zarr(
        ngrids_zarr_path, binary, ignore, output_path,
        args.step, args.density, args.chunk_size,
        pred_dt_u8=pred_dt_u8,
    )

    print(f"{TAG} pipeline complete: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
