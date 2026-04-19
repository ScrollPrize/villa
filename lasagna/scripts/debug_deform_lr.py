"""DEBUG: Test deformation inner loop with different LRs on several batches.

Usage:
  python lasagna/debug_deform_lr.py --config lasagna/configs/tifxyz_train_s3.json \
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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

_LASAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
if _LASAGNA_DIR not in sys.path:
    sys.path.insert(0, _LASAGNA_DIR)

from train_tifxyz import (
    build_model,
    compute_batch_targets,
    MaskedMSE,
    MaskedSmoothL1,
    ScaleSpaceLoss3D,
    _apply_warp,
    _compute_normal_field,
    _enforce_positive_jacobian,
)
from tifxyz_lasagna_dataset import (
    TifxyzLasagnaDataset,
    collate_variable_surfaces,
)

TAG = "[debug_deform_lr]"


def run_deform_scheduled(
    pred_cos_gm, targets, validity,
    grid_size, n_iters, max_frac,
    lr_start, lr_end,
    scale_loss_mse, scale_loss_l1,
    batch_label,
):
    """Run deformation with log-space LR sweep schedule, keep best result.

    LR ramps from lr_start to lr_end over n_iters on a log scale.
    Tracks best (lowest loss) deformation seen at any iteration.
    """
    device = pred_cos_gm.device
    G = grid_size
    B = pred_cos_gm.shape[0]
    Z, Y, X = targets.shape[2:]

    pred_f = pred_cos_gm.float().detach()
    cos_gm_orig = targets[:, 0:2].float()
    val_f = validity.float()

    with torch.no_grad():
        gm_ds = F.adaptive_avg_pool3d(cos_gm_orig[:, 1:2], (G, G, G))
        max_disp = max_frac / gm_ds.clamp(min=0.02)
        normal_dir = _compute_normal_field(cos_gm_orig[:, 0:1], G)
        val_ds = F.adaptive_avg_pool3d(val_f, (G, G, G))
        valid_mask = (val_ds > 0.01).float()

    print(f"\n{'='*70}")
    print(f"  {batch_label}  |  B={B}, vol={Z}x{Y}x{X}, grid={G}³")
    print(f"  valid frac = {val_f.mean():.3f}")
    print(f"  LR schedule: {lr_start} → {lr_end} (log ramp over {n_iters} its)")
    print(f"{'='*70}")

    deform = torch.zeros(B, 1, G, G, G, device=device, dtype=torch.float32)
    deform.requires_grad_(True)
    opt = torch.optim.SGD([deform], lr=lr_start)

    import math
    log_start = math.log10(lr_start)
    log_end = math.log10(lr_end)

    best_loss = float("inf")
    best_deform = deform.detach().clone()
    init_loss = None

    print(f"  {'iter':>4s}  {'lr':>10s}  {'loss':>10s}  {'l_cos':>10s}  "
          f"{'l_mag':>10s}  {'|d|_mean':>8s}  {'|d|_max':>8s}  {'best':>10s}")

    with torch.amp.autocast("cuda", enabled=False):
        for it in range(n_iters):
            frac = it / max(n_iters - 1, 1)
            lr_now = 10 ** (log_start + frac * (log_end - log_start))
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            opt.zero_grad()
            disp_3d = deform * normal_dir
            deform_full = F.interpolate(
                disp_3d, size=(Z, Y, X), mode="trilinear", align_corners=False,
            )
            warped_cg, warped_v = _apply_warp(cos_gm_orig, val_f, deform_full)
            cos_w = warped_cg[:, 1:2] * 20.0
            l_cos = scale_loss_mse(pred_f[:, 0:1], warped_cg[:, 0:1], mask=warped_v, weight=cos_w)
            l_mag = scale_loss_l1(pred_f[:, 1:2], warped_cg[:, 1:2], mask=warped_v, weight=cos_w)
            loss = l_cos + l_mag
            loss.backward()
            opt.step()

            with torch.no_grad():
                deform.data.clamp_(-max_disp, max_disp)
                deform.data.mul_(valid_mask)
                _enforce_positive_jacobian(deform, normal_dir)

                cur_loss = loss.item()
                if init_loss is None:
                    init_loss = cur_loss
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_deform = deform.detach().clone()

            if it % 10 == 0 or it == n_iters - 1:
                with torch.no_grad():
                    d_abs = deform.abs()
                print(
                    f"  {it:4d}  {lr_now:10.1f}  {cur_loss:10.6f}  {l_cos.item():10.6f}  "
                    f"{l_mag.item():10.6f}  {d_abs.mean().item():8.4f}  {d_abs.max().item():8.4f}  "
                    f"{best_loss:10.6f}"
                )

    d_abs = best_deform.abs()
    print(f"\n  => best loss={best_loss:.6f}  |d|_mean={d_abs.mean().item():.4f}  |d|_max={d_abs.max().item():.4f}")

    # Return initial and best loss for summary
    return init_loss, best_loss


def main():
    parser = argparse.ArgumentParser(description="Debug deformation LR sweep")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--n-batches", type=int, default=5,
                        help="Number of batches to test")
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--deform-stride", type=int, default=8)
    parser.add_argument("--max-frac", type=float, default=0.3)
    parser.add_argument("--lr-start", type=float, default=1e3,
                        help="Starting LR (low end of log sweep)")
    parser.add_argument("--lr-end", type=float, default=1e8,
                        help="Ending LR (high end of log sweep)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Fixed deterministic sample selection (evenly spaced)
    n_total = len(dataset)
    sample_indices = [int(round(i)) for i in np.linspace(0, n_total - 1, args.n_batches)]
    from torch.utils.data import Subset
    subset = Subset(dataset, sample_indices)
    loader = DataLoader(
        subset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=collate_variable_surfaces,
    )

    print(f"{TAG} loading model from {args.weights}...", flush=True)
    model, _, _ = build_model(
        args.patch_size, device, args.weights,
    )
    model.eval()

    mse_loss = MaskedMSE()
    smooth_l1_loss = MaskedSmoothL1()
    scale_loss_mse = ScaleSpaceLoss3D(mse_loss, num_scales=5)
    scale_loss_l1 = ScaleSpaceLoss3D(smooth_l1_loss, num_scales=5)

    grid_size = args.patch_size // args.deform_stride

    print(f"{TAG} LR sweep: {args.lr_start:.0f}→{args.lr_end:.0f} (log ramp), "
          f"n_iters={args.n_iters}, grid={grid_size}³, max_frac={args.max_frac}")
    print(f"{TAG} testing on {args.n_batches} batches")

    results_summary = []  # (label, init_loss, best_loss)
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
            _mg, _mm, _mc,
        ) = compute_batch_targets(
            batch, device,
            same_surface_threshold=same_surface_threshold,
        )

        if validity.sum() == 0:
            continue

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            res = model(image)
            pred = res["output"]

        pi = batch["patch_info"][0]
        label = f"batch {n_done}  ds={pi.get('dataset_name','')}  idx={pi.get('idx','')}"

        init_loss, best_loss = run_deform_scheduled(
            pred[:, 0:2].float(), targets, validity,
            grid_size=grid_size,
            n_iters=args.n_iters,
            max_frac=args.max_frac,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            scale_loss_mse=scale_loss_mse,
            scale_loss_l1=scale_loss_l1,
            batch_label=label,
        )
        results_summary.append((label, init_loss, best_loss))
        n_done += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({n_done} batches, {args.n_iters} iters, "
          f"LR {args.lr_start:.0f}→{args.lr_end:.0f})")
    print(f"{'='*70}")
    print(f"  {'batch':<45s}  {'init':>10s}  {'best':>10s}  {'improv':>8s}")
    total_init = 0.0
    total_best = 0.0
    for label, il, bl in results_summary:
        improv = (il - bl) / il * 100 if il > 0 else 0
        total_init += il
        total_best += bl
        print(f"  {label:<45s}  {il:10.6f}  {bl:10.6f}  {improv:7.1f}%")
    if results_summary:
        avg_init = total_init / len(results_summary)
        avg_best = total_best / len(results_summary)
        avg_improv = (avg_init - avg_best) / avg_init * 100 if avg_init > 0 else 0
        print(f"  {'AVERAGE':<45s}  {avg_init:10.6f}  {avg_best:10.6f}  {avg_improv:7.1f}%")
    print()


if __name__ == "__main__":
    main()
