"""Evaluate a 3D UNet checkpoint on all training samples.

Runs full-volume tiled inference on every CT+fitted.zarr pair, computes
per-channel masked MSE against supervision targets, and saves predictions
(zarr) + metrics (json) per sample plus a summary ranking.

Usage:
  python lasagna/eval_unet_3d.py \
      --checkpoint runs/unet3d/.../model_best.pt \
      --images-dir /path/to/images \
      --label-dir /path/to/label_dir \
      --output-dir eval_out \
      --patch-size 128
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import zarr
from zarr.codecs import BloscCodec

from train_unet_3d import (
    FittedZarrDataset,
    build_model,
    compute_targets_3d,
    _encode_dir,
    MaskedMSE,
    _CHANNEL_NAMES,
)


# ---------------------------------------------------------------------------
# 3D tiled inference
# ---------------------------------------------------------------------------

def _blend_ramp_1d(length: int, overlap: int, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
    """1D linear blend ramp: rises over `overlap` from edge, flat=1 in centre."""
    ramp = torch.ones(length, device=device, dtype=dtype)
    if overlap > 0:
        ov = min(overlap, length // 2)
        if ov > 0:
            edges = torch.linspace(0.0, 1.0, steps=ov + 1,
                                   device=device, dtype=dtype)[1:]
            ramp[:ov] = edges
            ramp[-ov:] = edges.flip(0)
    return ramp


def tiled_infer_3d(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int,
    overlap: int = 32,
    device: str = "cuda",
    amp_dtype: torch.dtype = torch.bfloat16,
    use_autocast: bool = True,
    output_sigmoid: bool = True,
) -> torch.Tensor:
    """Sliding-window 3D inference with linear blending.

    Args:
        model: 3D UNet in eval mode.
        image: (1, 1, D, H, W) float32 input volume.
        tile_size: cube side length for each tile.
        overlap: overlap in voxels between neighbouring tiles.

    Returns:
        (1, 8, D, H, W) float32 prediction.
    """
    assert image.ndim == 5 and image.shape[0] == 1
    _, _, D, H, W = image.shape
    stride = max(1, tile_size - overlap)
    dtype = torch.float32

    # Build 3D weight kernel from outer product of 1D ramps
    ramp_z = _blend_ramp_1d(tile_size, overlap, image.device, dtype)
    ramp_y = _blend_ramp_1d(tile_size, overlap, image.device, dtype)
    ramp_x = _blend_ramp_1d(tile_size, overlap, image.device, dtype)
    weight_3d = (ramp_z[:, None, None] * ramp_y[None, :, None]
                 * ramp_x[None, None, :])  # (T, T, T)

    def _positions(size: int, tile: int, s: int) -> List[int]:
        if size <= tile:
            return [0]
        pos = list(range(0, size - tile + 1, s))
        last = size - tile
        if pos[-1] != last:
            pos.append(last)
        return pos

    zs = _positions(D, tile_size, stride)
    ys = _positions(H, tile_size, stride)
    xs = _positions(W, tile_size, stride)

    acc = torch.zeros(1, 8, D, H, W, device=image.device, dtype=dtype)
    wsum = torch.zeros(1, 1, D, H, W, device=image.device, dtype=dtype)

    total_tiles = len(zs) * len(ys) * len(xs)
    tile_idx = 0

    with torch.no_grad():
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    z1 = min(z0 + tile_size, D)
                    y1 = min(y0 + tile_size, H)
                    x1 = min(x0 + tile_size, W)
                    patch = image[:, :, z0:z1, y0:y1, x0:x1]

                    # Pad to tile_size if at boundary
                    pz, py, px = z1 - z0, y1 - y0, x1 - x0
                    if pz < tile_size or py < tile_size or px < tile_size:
                        padded = torch.zeros(1, 1, tile_size, tile_size,
                                             tile_size, device=image.device,
                                             dtype=image.dtype)
                        padded[:, :, :pz, :py, :px] = patch
                        patch = padded

                    with torch.amp.autocast(device_type=device,
                                            dtype=amp_dtype,
                                            enabled=use_autocast):
                        result = model(patch)
                        pred = result["output"]
                        if output_sigmoid:
                            pred = torch.sigmoid(pred)
                        else:
                            pred = pred.clamp(0, 1)
                    pred = pred.float()

                    # Crop back to actual patch extent
                    pred = pred[:, :, :pz, :py, :px]
                    w = weight_3d[:pz, :py, :px].unsqueeze(0).unsqueeze(0)

                    acc[:, :, z0:z1, y0:y1, x0:x1] += pred * w
                    wsum[:, :, z0:z1, y0:y1, x0:x1] += w

                    tile_idx += 1
                    if tile_idx % 10 == 0 or tile_idx == total_tiles:
                        print(f"  tile {tile_idx}/{total_tiles}", end="\r",
                              flush=True)

    print()  # newline after progress
    return acc / wsum.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_vis(
    pred: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    ct_slice: np.ndarray,
    out_path: Path,
) -> None:
    """Save center-Z-slice comparison of pred vs GT directions as multi-page TIF.

    pred, targets: (8, Z, Y, X)  mask: (1, Z, Y, X)  ct_slice: (Y, X)

    Pages: CT, pred_cos, pred_mag, pred_dir_xy, validity, gt_cos, gt_mag, gt_dir_xy
    All saved as uint8 RGB (direction pages) or grayscale.
    """
    midz = pred.shape[1] // 2
    m = mask[0, midz]  # (Y, X)

    def _to_u8(a: np.ndarray) -> np.ndarray:
        return np.clip(a * 255, 0, 255).astype(np.uint8)

    pages = [
        _to_u8(ct_slice),                    # CT
        _to_u8(m),                           # validity
        _to_u8(pred[0, midz] * m),           # pred cos
        _to_u8(targets[0, midz] * m),        # GT cos
        _to_u8(pred[1, midz] * m),           # pred mag
        _to_u8(targets[1, midz] * m),        # GT mag
        _to_u8(pred[2, midz] * m),           # pred dir0_z
        _to_u8(targets[2, midz] * m),        # GT dir0_z
        _to_u8(pred[3, midz] * m),           # pred dir1_z
        _to_u8(targets[3, midz] * m),        # GT dir1_z
    ]
    page_names = [
        "CT", "validity",
        "pred_cos", "gt_cos",
        "pred_mag", "gt_mag",
        "pred_dir0_z", "gt_dir0_z",
        "pred_dir1_z", "gt_dir1_z",
    ]

    with tifffile.TiffWriter(str(out_path)) as tw:
        for page, name in zip(pages, page_names):
            # Tag 285 = PageName, read by GIMP as layer name
            tw.write(page, extratags=[(285, 's', 0, name, False)])


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    model: torch.nn.Module,
    img_path: Path,
    zarr_path: Path,
    name: str,
    step: int,
    tile_size: int,
    overlap: int,
    device: str,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    output_sigmoid: bool,
    output_dir: Path,
    w_cos: float = 1.0,
    w_mag: float = 1.0,
    w_dir: float = 1.0,
) -> Dict:
    """Run inference on one full sample and compute metrics."""

    # Load zarr labels (same logic as FittedZarrDataset.__getitem__, no crop)
    root = zarr.open(str(zarr_path), mode="r")
    origin = root.attrs["origin_fullres"]
    normal = np.array(root["normal"])       # (3, Z, Y, X)
    winding = np.array(root["winding"])     # (Z, Y, X)
    validity = np.array(root["validity"])   # (Z, Y, X)
    density = np.array(root["density"])     # (Z, Y, X)

    Z, Y, X = winding.shape
    x0 = int(round(origin[0]))
    y0 = int(round(origin[1]))
    z0 = int(round(origin[2]))

    # Load CT, crop to zarr extent
    image = tifffile.imread(str(img_path))
    if image.dtype == np.uint16:
        image = (image // 257).astype(np.uint8)
    image = image[z0:z0 + Z * step, y0:y0 + Y * step, x0:x0 + X * step]

    # Clamp labels to available image extent
    Z_eff = image.shape[0] // step
    Y_eff = image.shape[1] // step
    X_eff = image.shape[2] // step
    if Z_eff < Z or Y_eff < Y or X_eff < X:
        normal = normal[:, :Z_eff, :Y_eff, :X_eff]
        winding = winding[:Z_eff, :Y_eff, :X_eff]
        validity = validity[:Z_eff, :Y_eff, :X_eff]
        density = density[:Z_eff, :Y_eff, :X_eff]
        Z, Y, X = Z_eff, Y_eff, X_eff

    # Trim CT to exact step-aligned extent
    image = image[:Z * step, :Y * step, :X * step]

    print(f"  volume: {image.shape} -> labels: ({Z},{Y},{X}), step={step}")

    # Run tiled inference on full-res volume
    image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    image_t = image_t.to(device)

    pred_fullres = tiled_infer_3d(
        model, image_t, tile_size=tile_size, overlap=overlap,
        device=device, amp_dtype=amp_dtype, use_autocast=use_autocast,
        output_sigmoid=output_sigmoid,
    )  # (1, 8, D, H, W)

    # Avg-pool prediction down to step resolution
    if step > 1:
        pred_step = F.avg_pool3d(pred_fullres, kernel_size=step, stride=step)
    else:
        pred_step = pred_fullres
    # pred_step: (1, 8, Z, Y, X) at label resolution

    # Compute targets on GPU
    normal_t = torch.from_numpy(normal.copy()).float().unsqueeze(0).to(device)
    winding_t = torch.from_numpy(winding.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
    validity_t = torch.from_numpy(validity.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
    density_t = torch.from_numpy(density.copy()).float().unsqueeze(0).unsqueeze(0).to(device)

    targets, mask, dir_weight = compute_targets_3d(
        normal_t, winding_t, validity_t, density_t,
    )
    # targets: (1, 8, Z, Y, X), mask: (1, 1, Z, Y, X), dir_weight: (1, 6, Z, Y, X)

    # Ensure shapes match (pred may be slightly off due to pooling rounding)
    _, _, pZ, pY, pX = pred_step.shape
    _, _, tZ, tY, tX = targets.shape
    mZ, mY, mX = min(pZ, tZ), min(pY, tY), min(pX, tX)
    pred_step = pred_step[:, :, :mZ, :mY, :mX]
    targets = targets[:, :, :mZ, :mY, :mX]
    mask = mask[:, :, :mZ, :mY, :mX]
    dir_weight = dir_weight[:, :, :mZ, :mY, :mX]

    mask_bin = (mask > 0.5).float()

    # Per-channel masked MSE
    channel_mses = {}
    denom = mask_bin.sum().clamp(min=1.0)
    for i, ch_name in enumerate(_CHANNEL_NAMES):
        diff2 = (pred_step[:, i:i+1] - targets[:, i:i+1]) ** 2
        ch_mse = (diff2 * mask_bin).sum() / denom
        channel_mses[ch_name] = ch_mse.item()

    # Grouped losses (matching training logic but simple MSE, no multi-scale)
    mse_fn = MaskedMSE()
    loss_cos = mse_fn(pred_step[:, 0:1], targets[:, 0:1], mask=mask_bin).item()
    loss_mag = mse_fn(pred_step[:, 1:2], targets[:, 1:2], mask=mask_bin).item()
    loss_dir = mse_fn(pred_step[:, 2:8], targets[:, 2:8], mask=mask_bin,
                      weight=dir_weight).item()
    loss_combined = w_cos * loss_cos + w_mag * loss_mag + w_dir * loss_dir

    valid_frac = mask_bin.sum().item() / max(mask_bin.numel(), 1)

    metrics = {
        "name": name,
        "channel_mse": channel_mses,
        "loss_cos": loss_cos,
        "loss_mag": loss_mag,
        "loss_dir": loss_dir,
        "loss_combined": loss_combined,
        "valid_voxel_fraction": valid_frac,
        "volume_shape_fullres": list(image.shape),
        "label_shape": [Z, Y, X],
        "step": step,
    }

    # Save outputs
    sample_dir = output_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # pred.zarr at step resolution
    pred_np = pred_step[0].cpu().numpy().astype(np.float32)  # (8, Z, Y, X)
    zout = zarr.open_group(str(sample_dir / "pred.zarr"), mode="w")
    zout.create_array(
        "prediction", data=pred_np, chunks=(8, 32, 32, 32),
        compressors=BloscCodec(cname="zstd", clevel=3),
    )
    zout.attrs["channel_names"] = _CHANNEL_NAMES
    zout.attrs["step"] = step

    # metrics.json
    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Center-slice visualization
    targets_np = targets[0].cpu().numpy()  # (8, Z, Y, X)
    mask_np = mask_bin[0].cpu().numpy()    # (1, Z, Y, X)
    # CT at step resolution for vis (avg-pool to match label grid)
    ct_step = F.avg_pool3d(
        torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0,
        kernel_size=step, stride=step,
    )
    midz = pred_np.shape[1] // 2
    ct_slice = ct_step[0, 0, midz].numpy()
    save_vis(pred_np, targets_np, mask_np, ct_slice, sample_dir / "vis.tif")

    # Free GPU memory
    del pred_fullres, pred_step, image_t, targets, mask, dir_weight
    del normal_t, winding_t, validity_t, density_t
    torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 3D UNet checkpoint on all training samples.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory with CT image TIFFs.")
    parser.add_argument("--label-dir", type=str, required=True,
                        help="Directory with per-sample fitted.zarr labels.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write per-sample results.")
    parser.add_argument("--patch-size", type=int, default=192,
                        help="Architecture patch_size (default: 192).")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Inference tile size (default: patch-size).")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Overlap between tiles (default: 64).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto).")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--stats-filter", type=str, default="",
                        help="Dotted JSON path for sample filtering "
                             "(empty = no filter, evaluate ALL).")
    parser.add_argument("--stats-threshold", type=float, default=5.0)
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample names to evaluate "
                             "(default: all). E.g. 'sample_00001,sample_00033'.")
    parser.add_argument("--w-cos", type=float, default=1.0)
    parser.add_argument("--w-mag", type=float, default=1.0)
    parser.add_argument("--w-dir", type=float, default=1.0)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tile_size = args.tile_size or args.patch_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Precision
    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_autocast = True
    elif args.precision == "fp16":
        amp_dtype = torch.float16
        use_autocast = True
    else:
        amp_dtype = torch.float32
        use_autocast = False

    # Build model from checkpoint
    model, norm_type, upsample_mode, output_sigmoid = build_model(
        args.patch_size, device, weights=args.checkpoint,
    )
    model.eval()
    print(f"[eval] output_sigmoid={output_sigmoid}, norm={norm_type}, "
          f"upsample={upsample_mode}, precision={args.precision}")

    # Discover samples via FittedZarrDataset (reuse its matching/filtering)
    stats_filter = args.stats_filter or None
    ds = FittedZarrDataset(
        args.images_dir, args.label_dir, patch_size=None, random_crop=False,
        stats_filter=stats_filter, stats_threshold=args.stats_threshold,
    )
    step = ds.step

    # Filter to specific samples if requested
    if args.samples:
        names = set(s.strip() for s in args.samples.split(","))
        ds.samples = [(ip, zp, n) for ip, zp, n in ds.samples if n in names]
        missing = names - {n for _, _, n in ds.samples}
        if missing:
            print(f"[eval] WARNING: samples not found: {missing}")
        if not ds.samples:
            print("[eval] ERROR: no matching samples")
            sys.exit(1)

    print(f"[eval] {len(ds.samples)} samples, tile_size={tile_size}, "
          f"overlap={args.overlap}, step={step}")

    # Table header
    ch_hdrs = " ".join(f"{c:>7s}" for c in _CHANNEL_NAMES)
    hdr = (f"{'#':>3s}  {'sample':30s}  {'loss':>7s}  {'cos':>7s}  "
           f"{'mag':>7s}  {'dir':>7s}  {'valid':>6s}  {ch_hdrs}")
    sep = "-" * len(hdr)

    all_metrics: List[Dict] = []

    for i, (img_path, zarr_path, name) in enumerate(ds.samples):
        print(f"\n[{i+1}/{len(ds.samples)}] {name}")
        metrics = evaluate_sample(
            model, img_path, zarr_path, name, step,
            tile_size=tile_size, overlap=args.overlap,
            device=device, amp_dtype=amp_dtype, use_autocast=use_autocast,
            output_sigmoid=output_sigmoid, output_dir=output_dir,
            w_cos=args.w_cos, w_mag=args.w_mag, w_dir=args.w_dir,
        )
        all_metrics.append(metrics)

        # Print header (once after first sample, so it appears after loading noise)
        if i == 0:
            print(f"\n{sep}\n{hdr}\n{sep}")

        ch_vals = " ".join(f"{metrics['channel_mse'][c]:7.4f}"
                           for c in _CHANNEL_NAMES)
        print(f"{i+1:3d}  {name:30s}  {metrics['loss_combined']:7.4f}  "
              f"{metrics['loss_cos']:7.4f}  {metrics['loss_mag']:7.4f}  "
              f"{metrics['loss_dir']:7.4f}  {metrics['valid_voxel_fraction']:5.1%}  "
              f"{ch_vals}")

    # Summary sorted by loss descending (worst first)
    all_metrics.sort(key=lambda m: m["loss_combined"], reverse=True)
    summary = {
        "n_samples": len(all_metrics),
        "samples": all_metrics,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{sep}\nSummary (worst first):\n{sep}\n{hdr}\n{sep}")
    for rank, m in enumerate(all_metrics, 1):
        ch_vals = " ".join(f"{m['channel_mse'][c]:7.4f}" for c in _CHANNEL_NAMES)
        print(f"{rank:3d}  {m['name']:30s}  {m['loss_combined']:7.4f}  "
              f"{m['loss_cos']:7.4f}  {m['loss_mag']:7.4f}  "
              f"{m['loss_dir']:7.4f}  {m['valid_voxel_fraction']:5.1%}  "
              f"{ch_vals}")
    print(f"{sep}\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
