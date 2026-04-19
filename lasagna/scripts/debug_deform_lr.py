"""DEBUG: Test cage deformation inner loop on several batches.

Usage:
  python lasagna/scripts/debug_deform_lr.py --config lasagna/configs/tifxyz_train_s3.json \
      --weights path/to/model_best.pt --patch-size 192 --n-batches 5
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numba as _numba
import numpy as _np

@_numba.njit
def _numba_warmup():
    return _np.zeros(1, dtype=_np.int32)
_numba_warmup()
del _numba, _np, _numba_warmup

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_LASAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
if _LASAGNA_DIR not in sys.path:
    sys.path.insert(0, _LASAGNA_DIR)

from train_tifxyz import (
    build_model,
    compute_batch_targets,
    MaskedMSE,
    MaskedSmoothL1,
    ScaleSpaceLoss3D,
    _cage_deform_inner_loop,
)
from tifxyz_lasagna_dataset import (
    TifxyzLasagnaDataset,
    collate_variable_surfaces,
)

TAG = "[debug_deform_lr]"


def _save_separate_tifs(slices_dict, output_dir, prefix):
    """Save each slice as a separate TIFF."""
    try:
        from PIL import Image
    except ImportError:
        print("  (PIL not available, skipping images)", flush=True)
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in slices_dict.items():
        arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr_u8, mode="L")
        path = output_dir / f"{prefix}_{name}.tif"
        img.save(str(path))
    print(f"  saved {len(slices_dict)} images to {output_dir}/{prefix}_*.tif",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description="Debug cage deformation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--max-frac", type=float, default=0.3)
    parser.add_argument("--lr-start", type=float, default=1e3)
    parser.add_argument("--lr-end", type=float, default=1e7)
    parser.add_argument("--output-dir", type=str, default="tmp/debug_deform")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    with open(args.config) as f:
        config = json.load(f)
    config["patch_size"] = args.patch_size
    config["label_patch_size"] = args.patch_size

    same_surface_threshold = config.get("same_surface_threshold")
    if same_surface_threshold is not None:
        same_surface_threshold = float(same_surface_threshold)

    print(f"{TAG} building dataset...", flush=True)
    dataset = TifxyzLasagnaDataset(
        config, apply_augmentation=False,
        include_geometry=True,
    )
    print(f"{TAG} {len(dataset)} patches", flush=True)

    n_total = len(dataset)
    sample_indices = [int(round(i)) for i in np.linspace(0, n_total - 1, args.n_batches)]
    subset = Subset(dataset, sample_indices)
    loader = DataLoader(
        subset, batch_size=1, shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_surfaces,
    )

    print(f"{TAG} loading model from {args.weights}...", flush=True)
    model, _, _ = build_model(args.patch_size, device, args.weights)
    model.eval()

    mse_loss = MaskedMSE()
    smooth_l1_loss = MaskedSmoothL1()
    scale_loss_mse = ScaleSpaceLoss3D(mse_loss, num_scales=5)
    scale_loss_l1 = ScaleSpaceLoss3D(smooth_l1_loss, num_scales=5)

    # Override inner loop LR range via lr-end
    # _cage_deform_inner_loop uses inner_lr * 100 as default end;
    # here we set it by adjusting the ratio
    lr_ratio = args.lr_end / args.lr_start

    output_dir = Path(args.output_dir)
    print(f"{TAG} cage deform: n_iters={args.n_iters}, max_frac={args.max_frac}, "
          f"LR {args.lr_start:.0f} -> {args.lr_end:.0f}")
    print(f"{TAG} testing on {args.n_batches} batches, images -> {output_dir}")

    results_summary = []
    n_done = 0
    for batch in loader:
        if batch is None:
            continue
        if n_done >= args.n_batches:
            break

        image = batch["image"].to(device, non_blocking=True)
        (
            targets, validity,
            dir_sparse_mask, dir_dense_mask, dir_axis_weight,
            _mg, _mm, _mc, _dts, _bfrac, _blo, _bhi,
        ) = compute_batch_targets(
            batch, device,
            same_surface_threshold=same_surface_threshold,
        )

        if validity.sum() == 0:
            print(f"  batch {n_done}: skipped (no valid voxels)")
            continue

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            res = model(image)
            pred = res["output"]

        pi = batch["patch_info"][0]
        label = f"batch {n_done}  ds={pi.get('dataset_name','')}  idx={pi.get('idx','')}"

        has_cage = (
            "cage_ctrl_pos" in batch
            and batch["cage_ctrl_pos"] is not None
            and len(batch["cage_ctrl_pos"]) > 0
        )
        if not has_cage:
            print(f"  {label}: skipped (no cage data)")
            n_done += 1
            continue

        n_ctrl = sum(
            cp.numel() // 3 for cp in batch["cage_ctrl_pos"][0]
        )
        n_surf = len(batch["cage_ctrl_pos"][0])

        # Compute loss BEFORE deformation
        pred_f = pred[:, 0:2].float().detach()
        cos_gm_orig = targets[:, 0:2].float()
        val_f = validity.float()
        with torch.no_grad():
            cos_w = cos_gm_orig[:, 1:2] * 20.0
            loss_before = (
                scale_loss_mse(pred_f[:, 0:1], cos_gm_orig[:, 0:1],
                               mask=val_f, weight=cos_w).item()
                + scale_loss_l1(pred_f[:, 1:2], cos_gm_orig[:, 1:2],
                                mask=val_f, weight=cos_w).item()
            )

        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  surfaces={n_surf}, ctrl_points={n_ctrl}, "
              f"valid_frac={val_f.mean():.3f}, loss_before={loss_before:.6f}")
        print(f"{'='*70}")

        print(f"  {'iter':>4s}  {'lr':>10s}  {'loss':>10s}  {'best':>10s}  "
              f"{'|off|_mean':>10s}  {'|off|_max':>10s}  {'grad_mean':>10s}")

        warped_cg, warped_v, improvement = _cage_deform_inner_loop(
            pred_cos_gm=pred_f,
            targets=targets,
            validity=validity,
            batch=batch,
            dts_batch=_dts,
            n_iters=args.n_iters,
            inner_lr=args.lr_start,
            max_frac=args.max_frac,
            scale_loss_mse_fn=scale_loss_mse,
            scale_loss_l1_fn=scale_loss_l1,
            verbose=True,
            bracket_frac=_bfrac,
            bracket_lo=_blo,
            bracket_hi=_bhi,
        )

        # Compute loss AFTER deformation
        with torch.no_grad():
            cos_w_after = warped_cg[:, 1:2] * 20.0
            loss_after = (
                scale_loss_mse(pred_f[:, 0:1], warped_cg[:, 0:1],
                               mask=warped_v, weight=cos_w_after).item()
                + scale_loss_l1(pred_f[:, 1:2], warped_cg[:, 1:2],
                                mask=warped_v, weight=cos_w_after).item()
            )

        print(f"  loss_before={loss_before:.6f}  loss_after={loss_after:.6f}  "
              f"improvement={improvement:.4f}")

        # Build deformed targets for visualization
        with torch.no_grad():
            targets_deformed = targets.clone()
            targets_deformed[:, 0:2] = warped_cg

        # Save images
        b = 0
        Z = targets.shape[2]
        mid_z = Z // 2
        prefix = f"batch_{n_done:03d}"
        slices = {
            "ct": image[b, 0, mid_z].float().cpu().numpy(),
            "pred_cos": pred[b, 0, mid_z].float().cpu().numpy(),
            "pred_mag": pred[b, 1, mid_z].float().cpu().numpy(),
            "gt_cos_orig": targets[b, 0, mid_z].float().cpu().numpy(),
            "gt_mag_orig": targets[b, 1, mid_z].float().cpu().numpy(),
            "gt_cos_deformed": warped_cg[0, 0, mid_z].cpu().numpy(),
            "gt_mag_deformed": warped_cg[0, 1, mid_z].cpu().numpy(),
            "validity_orig": validity[b, 0, mid_z].float().cpu().numpy(),
            "validity_deformed": warped_v[0, 0, mid_z].cpu().numpy(),
        }
        _save_separate_tifs(slices, output_dir, prefix)

        results_summary.append((label, loss_before, loss_after, improvement))
        n_done += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({n_done} batches, {args.n_iters} iters)")
    print(f"{'='*70}")
    print(f"  {'batch':<45s}  {'before':>10s}  {'after':>10s}  {'ratio':>8s}")
    for label, lb, la, imp in results_summary:
        print(f"  {label:<45s}  {lb:10.6f}  {la:10.6f}  {imp:8.4f}")
    if results_summary:
        avg_imp = sum(r[3] for r in results_summary) / len(results_summary)
        print(f"  {'AVERAGE':<45s}  {'':>10s}  {'':>10s}  {avg_imp:8.4f}")
    print()


if __name__ == "__main__":
    main()
