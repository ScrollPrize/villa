"""Convert fitted.zarr (normal, winding, validity, density) to UNet training labels.

Produces multi-layer TIF files for each target channel at step resolution:
  cos.tif       (Z, Y, X) float32  — cosine-encoded winding position
  grad_mag.tif  (Z, Y, X) float32  — sheet density
  dir_z.tif     (Z, 2, Y, X) float32  — direction encoding in XY plane
  dir_y.tif     (Z, 2, Y, X) float32  — direction encoding in XZ plane
  dir_x.tif     (Z, 2, Y, X) float32  — direction encoding in YZ plane
  validity.tif  (Z, Y, X) float32  — binary validity mask

Usage:
  python fitted_to_unet_labels.py \\
      --input fitted.zarr \\
      --output-dir unet_labels
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import zarr


def _encode_dir(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Double-angle direction encoding from 2D gradient vector.

    Returns (dir0, dir1) each in [0, 1].
    """
    eps = 1e-8
    r2 = gx * gx + gy * gy + eps
    cos2t = (gx * gx - gy * gy) / r2
    sin2t = 2.0 * gx * gy / r2
    dir0 = 0.5 + 0.5 * cos2t
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    dir1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
    return dir0.astype(np.float32), dir1.astype(np.float32)


def convert(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[fitted_to_unet_labels] reading {input_path}", flush=True)
    root = zarr.open(input_path, mode="r")
    normal = np.array(root["normal"])      # (3, Z, Y, X) float32
    winding = np.array(root["winding"])    # (Z, Y, X) float32
    validity = np.array(root["validity"])  # (Z, Y, X) float32
    density = np.array(root["density"])    # (Z, Y, X) float32

    nx, ny, nz = normal[0], normal[1], normal[2]
    Z, Y, X = winding.shape
    print(f"[fitted_to_unet_labels] volume shape: Z={Z} Y={Y} X={X}", flush=True)

    # cos: cosine-encoded winding position
    # 1 on sheet (integer winding), 0 between sheets
    cos_label = (0.5 + 0.5 * np.cos(2.0 * np.pi * winding)).astype(np.float32)
    cos_label *= validity

    # grad_mag: sheet density (directly from fit-data)
    grad_mag_label = (density * validity).astype(np.float32)

    # 3x2 direction encoding from 3D normal
    # Z-slices (XY plane): gradient direction = (nx, ny)
    dir0_z, dir1_z = _encode_dir(nx, ny)
    # Y-slices (XZ plane): gradient direction = (nx, nz)
    dir0_y, dir1_y = _encode_dir(nx, nz)
    # X-slices (YZ plane): gradient direction = (ny, nz)
    dir0_x, dir1_x = _encode_dir(ny, nz)

    # Write TIFs
    tifffile.imwrite(str(out / "cos.tif"), cos_label, compression="lzw")
    print(f"[fitted_to_unet_labels] wrote cos.tif {cos_label.shape}", flush=True)

    tifffile.imwrite(str(out / "grad_mag.tif"), grad_mag_label, compression="lzw")
    print(f"[fitted_to_unet_labels] wrote grad_mag.tif {grad_mag_label.shape}", flush=True)

    # Dir TIFs: (2, Z, Y, X) -> (Z, 2, Y, X) for multi-page TIF
    dir_z = np.stack([dir0_z, dir1_z], axis=0)  # (2, Z, Y, X)
    dir_z_tif = dir_z.transpose(1, 0, 2, 3)     # (Z, 2, Y, X)
    tifffile.imwrite(str(out / "dir_z.tif"), dir_z_tif, compression="lzw")
    print(f"[fitted_to_unet_labels] wrote dir_z.tif {dir_z_tif.shape}", flush=True)

    dir_y = np.stack([dir0_y, dir1_y], axis=0)
    dir_y_tif = dir_y.transpose(1, 0, 2, 3)
    tifffile.imwrite(str(out / "dir_y.tif"), dir_y_tif, compression="lzw")
    print(f"[fitted_to_unet_labels] wrote dir_y.tif {dir_y_tif.shape}", flush=True)

    dir_x = np.stack([dir0_x, dir1_x], axis=0)
    dir_x_tif = dir_x.transpose(1, 0, 2, 3)
    tifffile.imwrite(str(out / "dir_x.tif"), dir_x_tif, compression="lzw")
    print(f"[fitted_to_unet_labels] wrote dir_x.tif {dir_x_tif.shape}", flush=True)

    tifffile.imwrite(str(out / "validity.tif"), validity.astype(np.float32),
                     compression="lzw")
    print(f"[fitted_to_unet_labels] wrote validity.tif {validity.shape}", flush=True)

    print(f"[fitted_to_unet_labels] done: {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert fitted.zarr to UNet training label TIFs")
    p.add_argument("--input", required=True, help="Path to fitted.zarr")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for label TIFs")
    args = p.parse_args()
    convert(args.input, args.output_dir)


if __name__ == "__main__":
    main()
