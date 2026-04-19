"""DEBUG: Test cage deformation — hard-offset sanity check.

Applies a fixed 5-voxel offset along X to all control points and
saves separate TIFF images for each channel (no optimization).

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
    _build_cage_displacement,
    _apply_warp,
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
    parser = argparse.ArgumentParser(description="Debug cage deformation (hard offset)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument("--offset", type=float, default=5.0,
                        help="Hard offset in voxels to apply to all ctrl points")
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

    output_dir = Path(args.output_dir)
    hard_offset = args.offset
    print(f"{TAG} hard offset = {hard_offset} voxels, output → {output_dir}")

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
            pred = res["output"].float()

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

        cage_ctrl_pos = batch["cage_ctrl_pos"]
        cage_ctrl_normals = batch["cage_ctrl_normals"]
        cage_grid_rc = batch["cage_grid_rc"]
        cage_grid_rc_w = batch["cage_grid_rc_w"]

        b = 0
        ctrl_pos_list = [cp.to(device) for cp in cage_ctrl_pos[b]]
        ctrl_normals_list = [cn.to(device) for cn in cage_ctrl_normals[b]]
        grid_rc_list = [rc.to(device) for rc in cage_grid_rc[b]]
        grid_rc_w_list = [w.to(device) for w in cage_grid_rc_w[b]]
        dts_b = _dts[b]
        N_surf = len(ctrl_pos_list)

        # Real surface masks + chain info for bracket routing
        masks_b = batch["surface_masks"][b].to(device)
        N_masks = masks_b.shape[0]
        surf_masks_b = [masks_b[i] > 0.5 for i in range(min(N_masks, N_surf))]
        chain_info_b = batch["surface_chain_info"][b]

        # Hard offset: set all control point offsets to args.offset
        ctrl_offsets = []
        for s in range(N_surf):
            cp = ctrl_pos_list[s]
            n_pts = cp.shape[0] * cp.shape[1] if cp.dim() >= 2 else 0
            ctrl_offsets.append(
                torch.full((n_pts,), hard_offset, device=device, dtype=torch.float32)
            )

        n_ctrl = sum(o.numel() for o in ctrl_offsets)
        print(f"\n  {label}")
        print(f"  surfaces={N_surf}, ctrl_points={n_ctrl}")

        Z, Y, X = targets.shape[2:]
        with torch.no_grad():
            disp = _build_cage_displacement(
                ctrl_offsets, ctrl_pos_list, ctrl_normals_list,
                grid_rc_list, grid_rc_w_list,
                dts_b, validity[b:b+1].float(), (Z, Y, X),
                surface_masks=surf_masks_b,
                bracket_frac=_bfrac[b], bracket_lo=_blo[b], bracket_hi=_bhi[b],
            )
            print(f"  disp nonzero: {(disp.abs() > 1e-6).sum().item()}")
            print(f"  disp abs max: {disp.abs().max().item():.2f}")
            print(f"  disp abs mean (where >0): "
                  f"{disp[disp.abs() > 1e-6].abs().mean().item():.2f}"
                  if (disp.abs() > 1e-6).any() else "  disp is all zero!")

            warped_cg, warped_v = _apply_warp(
                targets[b:b+1, 0:2].float(),
                validity[b:b+1].float(),
                disp,
            )

        # Save separate tifs
        mid_z = Z // 2
        prefix = f"batch_{n_done:03d}"
        slices = {
            "ct": image[b, 0, mid_z].float().cpu().numpy(),
            "pred_cos": pred[b, 0, mid_z].cpu().numpy(),
            "pred_mag": pred[b, 1, mid_z].cpu().numpy(),
            "gt_cos_orig": targets[b, 0, mid_z].float().cpu().numpy(),
            "gt_mag_orig": targets[b, 1, mid_z].float().cpu().numpy(),
            "gt_cos_deformed": warped_cg[0, 0, mid_z].cpu().numpy(),
            "gt_mag_deformed": warped_cg[0, 1, mid_z].cpu().numpy(),
            "validity_orig": validity[b, 0, mid_z].float().cpu().numpy(),
            "validity_deformed": warped_v[0, 0, mid_z].cpu().numpy(),
            "disp_z": disp[0, 0, mid_z].cpu().numpy() / max(hard_offset, 1) * 0.5 + 0.5,
            "disp_y": disp[0, 1, mid_z].cpu().numpy() / max(hard_offset, 1) * 0.5 + 0.5,
            "disp_x": disp[0, 2, mid_z].cpu().numpy() / max(hard_offset, 1) * 0.5 + 0.5,
        }
        _save_separate_tifs(slices, output_dir, prefix)
        n_done += 1

    print(f"\n{TAG} done. {n_done} batches processed.", flush=True)


if __name__ == "__main__":
    main()
