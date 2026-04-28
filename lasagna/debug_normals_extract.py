#!/usr/bin/env python3
"""Extract normal slices from predict3d output zarr as JPGs.

Reads a prediction zarr (and optionally a reference zarr), extracts XY/XZ/YZ
slices through the center point for cos/nx/ny channels, and saves as grayscale
JPGs. Optionally runs a 2D UNet on the same slices for comparison.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_normal(u8: np.ndarray) -> np.ndarray:
    """uint8 -> float in [-1, 1]."""
    return (u8.astype(np.float32) - 128.0) / 127.0


def _encode_dir(gx: np.ndarray, gy: np.ndarray):
    """Re-encode two normal components to dir0/dir1 in [0,1]. Same as train code."""
    eps = 1e-8
    r2 = gx * gx + gy * gy + eps
    cos2t = (gx * gx - gy * gy) / r2
    sin2t = 2.0 * gx * gy / r2
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    dir0 = 0.5 + 0.5 * cos2t
    dir1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
    return dir0, dir1


def _normals_to_dirs(nx_u8: np.ndarray, ny_u8: np.ndarray):
    """Decode nx/ny uint8 normals and re-encode to per-axis dir0/dir1 uint8."""
    nx = _decode_normal(nx_u8)
    ny = _decode_normal(ny_u8)
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
    dirs = {}
    for axis, gx, gy in [("xy", nx, ny), ("xz", nx, nz), ("yz", ny, nz)]:
        d0, d1 = _encode_dir(gx, gy)
        dirs[f"{axis}_dir0"] = np.clip(d0 * 255.0, 0, 255).astype(np.uint8)
        dirs[f"{axis}_dir1"] = np.clip(d1 * 255.0, 0, 255).astype(np.uint8)
    return dirs


def _print_stats(name: str, u8: np.ndarray):
    f = _decode_normal(u8)
    print(f"  {name:>12s}: uint8 [{u8.min():3d}, {u8.max():3d}]  "
          f"float [{f.min():+.3f}, {f.max():+.3f}]  mean={f.mean():+.4f}")


def _print_stats_raw(name: str, u8: np.ndarray):
    """Stats for non-normal channels (cos, dir0, dir1)."""
    print(f"  {name:>12s}: uint8 [{u8.min():3d}, {u8.max():3d}]  "
          f"mean={u8.mean():.1f}")


def _channel_index(channels: list[str], name: str) -> int:
    try:
        return channels.index(name)
    except ValueError:
        raise SystemExit(f"Channel '{name}' not found in zarr channels: {channels}")


def _open_zarr_array(path: str):
    """Open zarr and return (array, params dict)."""
    store = zarr.open(path, mode="r")
    if isinstance(store, zarr.Group):
        arr = store["0"] if "0" in store else store[list(store.keys())[0]]
    else:
        arr = store
    params = dict(arr.attrs.get("preprocess_params", {}))
    if not params and isinstance(store, zarr.Group):
        params = dict(store.attrs.get("preprocess_params", {}))
    return arr, params


# ---------------------------------------------------------------------------
# 3D prediction extraction (3 orientations)
# ---------------------------------------------------------------------------

def extract_pred(arr, params: dict, px: int, py: int, pz: int,
                 wh: int, zdepth: int, output_dir: Path):
    sd = int(params["scaledown"])
    channels = [str(c) for c in params.get("channels", [])]
    ci_cos = _channel_index(channels, "cos")
    ci_nx = _channel_index(channels, "nx")
    ci_ny = _channel_index(channels, "ny")

    half_wh = wh // 2
    half_z = zdepth // 2

    print(f"Pred zarr: scaledown={sd}, channels={channels}, shape={arr.shape}")

    slices = {}

    # XY slice (fix Z, crop Y and X by wh)
    iz = pz // sd
    y0, y1 = (py - half_wh) // sd, (py + half_wh) // sd
    x0, x1 = (px - half_wh) // sd, (px + half_wh) // sd
    print(f"  XY: z={iz}, y=[{y0}:{y1}], x=[{x0}:{x1}]")
    for name, ci in [("cos", ci_cos), ("nx", ci_nx), ("ny", ci_ny)]:
        s = np.asarray(arr[ci, iz, y0:y1, x0:x1])
        cv2.imwrite(str(output_dir / f"pred3d_xy_{name}.jpg"), s)
        slices[f"xy_{name}"] = s

    # XZ slice (fix Y, crop Z by zdepth, X by wh)
    iy = py // sd
    z0, z1 = (pz - half_z) // sd, (pz + half_z) // sd
    print(f"  XZ: y={iy}, z=[{z0}:{z1}], x=[{x0}:{x1}]")
    for name, ci in [("cos", ci_cos), ("nx", ci_nx), ("ny", ci_ny)]:
        s = np.asarray(arr[ci, z0:z1, iy, x0:x1])
        cv2.imwrite(str(output_dir / f"pred3d_xz_{name}.jpg"), s)
        slices[f"xz_{name}"] = s

    # YZ slice (fix X, crop Z by zdepth, Y by wh)
    ix = px // sd
    print(f"  YZ: x={ix}, z=[{z0}:{z1}], y=[{y0}:{y1}]")
    for name, ci in [("cos", ci_cos), ("nx", ci_nx), ("ny", ci_ny)]:
        s = np.asarray(arr[ci, z0:z1, y0:y1, ix])
        cv2.imwrite(str(output_dir / f"pred3d_yz_{name}.jpg"), s)
        slices[f"yz_{name}"] = s

    # Re-encode normals to per-axis dir0/dir1 for comparison with 2D
    for plane in ["xy", "xz", "yz"]:
        nx_s = slices.get(f"{plane}_nx")
        ny_s = slices.get(f"{plane}_ny")
        if nx_s is not None and ny_s is not None:
            dirs = _normals_to_dirs(nx_s, ny_s)
            d0 = dirs[f"{plane}_dir0"]
            d1 = dirs[f"{plane}_dir1"]
            cv2.imwrite(str(output_dir / f"pred3d_{plane}_dir0.jpg"), d0)
            cv2.imwrite(str(output_dir / f"pred3d_{plane}_dir1.jpg"), d1)
            slices[f"{plane}_dir0"] = d0
            slices[f"{plane}_dir1"] = d1

    print("Pred3D stats:")
    for key, s in sorted(slices.items()):
        if "nx" in key or "ny" in key:
            _print_stats(key, s)
        else:
            _print_stats_raw(key, s)

    return slices


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def extract_ref(ref_arr, ref_params: dict, px: int, py: int, pz: int,
                wh: int, output_dir: Path, pred_slices: dict):
    sd = int(ref_params["scaledown"])
    channels = [str(c) for c in ref_params.get("channels", [])]
    ci_cos = _channel_index(channels, "cos")
    ci_nx = _channel_index(channels, "nx")
    ci_ny = _channel_index(channels, "ny")

    half_wh = wh // 2

    print(f"\nRef zarr: scaledown={sd}, channels={channels}, shape={ref_arr.shape}")

    # XY slice
    iz = pz // sd
    y0, y1 = (py - half_wh) // sd, (py + half_wh) // sd
    x0, x1 = (px - half_wh) // sd, (px + half_wh) // sd
    for name, ci in [("cos", ci_cos), ("nx", ci_nx), ("ny", ci_ny)]:
        s = np.asarray(ref_arr[ci, iz, y0:y1, x0:x1])
        cv2.imwrite(str(output_dir / f"ref_xy_{name}.jpg"), s)
        if "nx" in name or "ny" in name:
            _print_stats(f"ref_xy_{name}", s)
        else:
            _print_stats_raw(f"ref_xy_{name}", s)

    # Diff images for XY nx/ny
    for ch in ["nx", "ny"]:
        pred_s = pred_slices.get(f"xy_{ch}")
        ref_s = np.asarray(ref_arr[ci_nx if ch == "nx" else ci_ny, iz, y0:y1, x0:x1])
        if pred_s is not None:
            ph, pw = pred_s.shape[:2]
            rh, rw = ref_s.shape[:2]
            if (rh, rw) != (ph, pw):
                ref_s = cv2.resize(ref_s, (pw, ph), interpolation=cv2.INTER_NEAREST)
            diff = np.abs(pred_s.astype(np.int16) - ref_s.astype(np.int16)).astype(np.uint8)
            mx = diff.max()
            if mx > 0:
                diff = (diff.astype(np.float32) * 255.0 / mx).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"diff_xy_{ch}.jpg"), diff)
            print(f"  diff_xy_{ch}: max abs diff = {mx} (uint8)")


# ---------------------------------------------------------------------------
# 2D UNet inference on 3 slices
# ---------------------------------------------------------------------------

def run_2d_inference(checkpoint_2d: str, input_zarr_path: str,
                     px: int, py: int, pz: int, wh: int,
                     output_dir: Path):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import load_unet, unet_infer_tiled

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_unet(device, weights=checkpoint_2d, out_channels=4,
                      base_channels=32, num_levels=6, max_channels=1024)
    model.eval()

    vol = zarr.open(input_zarr_path, mode="r")
    if isinstance(vol, zarr.Group):
        vol = vol["0"] if "0" in vol else vol[list(vol.keys())[0]]

    half = wh // 2

    axes = [
        ("xy", lambda: np.asarray(vol[pz, py - half:py + half, px - half:px + half])),
        ("xz", lambda: np.asarray(vol[pz - half:pz + half, py, px - half:px + half])),
        ("yz", lambda: np.asarray(vol[pz - half:pz + half, py - half:py + half, px])),
    ]

    print(f"\n2D UNet inference: checkpoint={checkpoint_2d}")
    print(f"  device={device}, volume shape={vol.shape}")

    for axis_name, read_fn in axes:
        raw_np = read_fn()
        if raw_np.dtype == np.uint16:
            raw_np = (raw_np // 257).astype(np.uint8)
        raw_t = torch.from_numpy(raw_np.astype(np.float32)).to(device)
        mx = raw_t.amax()
        if mx > 0:
            raw_t = raw_t / mx
        raw_t = raw_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        with torch.inference_mode(), torch.autocast(device_type=device.type):
            pred = unet_infer_tiled(model, raw_t, tile_size=2048, overlap=128, border=32)
        # pred: (1, 4, H, W), sigmoid already applied, [0, 1]

        cos_u8 = np.clip(pred[0, 0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        dir0_u8 = np.clip(pred[0, 2].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        dir1_u8 = np.clip(pred[0, 3].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(str(output_dir / f"pred2d_{axis_name}_cos.jpg"), cos_u8)
        cv2.imwrite(str(output_dir / f"pred2d_{axis_name}_dir0.jpg"), dir0_u8)
        cv2.imwrite(str(output_dir / f"pred2d_{axis_name}_dir1.jpg"), dir1_u8)

        print(f"  {axis_name}: shape={raw_np.shape}")
        _print_stats_raw(f"{axis_name}_cos", cos_u8)
        _print_stats_raw(f"{axis_name}_dir0", dir0_u8)
        _print_stats_raw(f"{axis_name}_dir1", dir1_u8)

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract normal slices from predict3d zarr.")
    p.add_argument("--pred-zarr", required=True, help="Prediction zarr path.")
    p.add_argument("--point", type=int, nargs=3, required=True,
                   metavar=("PX", "PY", "PZ"), help="Center point in fullres coords.")
    p.add_argument("--wh", type=int, required=True, help="XY crop size in fullres.")
    p.add_argument("--zdepth", type=int, required=True, help="Z crop depth in fullres.")
    p.add_argument("--output-dir", default="./debug_normals_out", help="Output directory.")
    p.add_argument("--ref-zarr", default=None, help="Optional reference zarr for comparison.")
    p.add_argument("--checkpoint-2d", default=None, help="2D UNet checkpoint for comparison.")
    p.add_argument("--input-zarr", default=None, help="Raw input volume zarr (needed for 2D inference).")
    args = p.parse_args(argv)

    px, py, pz = args.point
    wh = args.wh
    zdepth = args.zdepth
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3D prediction extraction
    pred_arr, pred_params = _open_zarr_array(args.pred_zarr)
    if not pred_params:
        print("WARNING: No preprocess_params found in pred zarr, assuming defaults")
        pred_params = {"scaledown": 4, "channels": ["cos", "grad_mag", "nx", "ny"]}

    pred_slices = extract_pred(pred_arr, pred_params, px, py, pz, wh, zdepth, output_dir)

    # Reference comparison
    if args.ref_zarr:
        ref_arr, ref_params = _open_zarr_array(args.ref_zarr)
        if not ref_params:
            raise SystemExit("Reference zarr has no preprocess_params — cannot determine channels/scaledown")
        extract_ref(ref_arr, ref_params, px, py, pz, wh, output_dir, pred_slices)

    # 2D UNet comparison
    if args.checkpoint_2d:
        if not args.input_zarr:
            raise SystemExit("--input-zarr is required when using --checkpoint-2d")
        run_2d_inference(args.checkpoint_2d, args.input_zarr, px, py, pz, wh, output_dir)

    print(f"\nOutput written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
