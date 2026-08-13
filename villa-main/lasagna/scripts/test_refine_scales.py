#!/usr/bin/env python3
"""Test multi-scale refinement with random augmentations.

For each sample: random augmentation + random scale chain, using the
exact same code path as training (augment_batch_inplace →
compute_batch_targets → run_refine_chain).

Saves multi-row TIF with one row per chain pass:
  CT | pred_cos | gt_cos | diff | prior_cos (if chain)

Usage:
    python lasagna/scripts/test_refine_scales.py \
        --config config.json --weights model.pt \
        --num-samples 20 --output-dir tmp/refine_test
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vesuvius" / "src"))

from tifxyz_lasagna_dataset import (
    TifxyzLasagnaDataset,
    collate_variable_surfaces,
    augment_batch_inplace,
)
from train_tifxyz import (
    build_model,
    compute_batch_targets,
    run_refine_chain,
)

TAG = "[test_refine_scales]"

ALL_CHAINS = {
    "s0": [(0, False)],
    "m1": [(-1, False)],
    "p1": [(1, False)],
    "m1_0": [(-1, False), (0, True)],
    "0_p1": [(0, False), (1, True)],
    "m1_p1": [(-1, False), (1, True)],
}


def _to_u8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _compute_z_indices(passes, patch_info, patch_size):
    """Compute per-pass z-indices that map to the same world z-position.

    Reference: mid-z of the scale-0 volume.
    """
    P = patch_size
    pi0 = patch_info[0]
    world_min = pi0["world_min"]
    ref_world_z = float(world_min[0]) + P / 2.0  # base mid in world coords

    z_indices = []
    for p in passes:
        offset = p["offset"]
        pZ = p["pred"].shape[2]
        if offset == 0:
            z_idx = pZ // 2
        elif offset == -1:
            coarse_offset = pi0.get("coarse_offset")
            if coarse_offset is not None:
                base_center_z = float(world_min[0]) + P / 2.0
                coarse_min_z = (base_center_z - P) / 2.0 + float(coarse_offset[0]) / 2.0
                # world_z = coarse_min_z * 2 + z_idx * 2
                z_idx = int(round((ref_world_z - coarse_min_z * 2) / 2.0))
            else:
                z_idx = pZ // 2
        elif offset == 1:
            fine_offset = pi0.get("fine_offset")
            if fine_offset is not None:
                # world_z = (world_min[0] + fine_offset[0]) + z_idx * 0.5
                z_idx = int(round((ref_world_z - float(world_min[0]) - float(fine_offset[0])) * 2.0))
            else:
                z_idx = pZ // 2
        else:
            z_idx = pZ // 2
        z_indices.append(max(0, min(z_idx, pZ - 1)))
    return z_indices


def _save_chain_tif(passes, chain_name, aug_str, sample_idx, output_dir,
                    patch_info=None, patch_size=192):
    """Save multi-row TIF: one row per pass in the chain.

    Each row: CT | resampled_prior | gt_cos | pred_cos | pred_diff | prior_diff
    resampled_prior = previous pass's prediction resampled to this scale.
    prior_diff = |resampled_prior - gt| tests cross-scale alignment.
    All rows use the same world z-position for spatial comparability.
    """
    try:
        from PIL import Image
    except ImportError:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute z-indices aligned to the same world position
    if patch_info is not None:
        z_indices = _compute_z_indices(passes, patch_info, patch_size)
    else:
        z_indices = [p["pred"].shape[2] // 2 for p in passes]

    rows = []
    for pi, (p, mid) in enumerate(zip(passes, z_indices)):
        pred = p["pred"]
        gt = p["gt"]
        val = p["validity"]
        ct = p["ct"]
        prior = p.get("prior_resampled")  # resampled prev pred, always computed

        pred_cos = pred[0, 0, mid].float().numpy()
        gt_cos = gt[0, 0, mid].float().numpy()
        val_np = val[0, 0, mid].float().numpy()
        pred_diff = np.abs(pred_cos - gt_cos) * val_np

        ct_slice = ct[0, 0, mid].float().numpy()

        # Resampled prior prediction from previous pass
        if prior is not None:
            prior_cos = prior[0, 0, mid].float().numpy()
        else:
            prior_cos = np.zeros_like(pred_cos)
        prior_diff = np.abs(prior_cos - gt_cos) * val_np

        panels = [
            _to_u8(ct_slice),
            _to_u8(prior_cos),
            _to_u8(gt_cos),
            _to_u8(pred_cos),
            _to_u8(pred_diff),
            _to_u8(prior_diff),
        ]

        row_img = np.concatenate(panels, axis=1)
        rows.append(row_img)

    # Pad rows to same width (prior column may differ)
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1]), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)

    combined = np.concatenate(padded, axis=0)
    img = Image.fromarray(combined, mode="L")
    fname = f"s{sample_idx:04d}_{aug_str}_{chain_name}.tif"
    img.save(str(output_dir / fname))


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-scale refinement with augmentations.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="tmp/refine_test")
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--model-patch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no-refine", action="store_true",
                        help="Use regular 1ch model instead of 11ch.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    patch_size = args.patch_size
    use_refine = not args.no_refine

    amp_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
    }[args.precision]

    with open(args.config) as f:
        config = json.load(f)
    config["patch_size"] = patch_size
    config["refine_mode"] = True

    model, _, _ = build_model(
        patch_size, device, weights=args.weights, refine=use_refine,
        model_patch_size=args.model_patch_size,
    )
    output_sigmoid = False
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        output_sigmoid = ckpt.get("output_sigmoid", False)
    model.eval()

    dataset = TifxyzLasagnaDataset(
        config, apply_augmentation=False, refine_mode=True,
        include_geometry=True,
    )
    same_surface_threshold = float(config.get("same_surface_threshold", 0.0)) or None

    n = len(dataset)
    indices = np.random.choice(n, size=min(args.num_samples, n), replace=False)
    chain_names = list(ALL_CHAINS.keys())

    # Header
    print(f"{'#':>4s}  {'idx':>6s}  {'aug':<20s}  {'chain':<8s}  "
          f"{'pass':>4s}  {'off':>3s}  {'cos_mse':>10s}  {'mag_l1':>10s}",
          flush=True)
    print("-" * 80, flush=True)

    for si, idx in enumerate(indices):
        sample = dataset[idx]
        if sample is None:
            continue

        batch = collate_variable_surfaces([sample])
        if batch is None:
            continue
        batch = copy.deepcopy(batch)

        # Random augmentation — same call as training
        batch = augment_batch_inplace(batch)
        aug = batch.get("_aug", (False, False, False, 0))
        fz, fy, fx, k = aug
        aug_str = f"f{'z' if fz else ''}{'y' if fy else ''}{'x' if fx else ''}_r{k}"
        if not (fz or fy or fx or k):
            aug_str = "identity"

        image = batch["image"].to(device)
        (
            targets, validity,
            dir_sparse_mask, dir_dense_mask, dir_axis_weight,
            _, _, _, _, _, _, _,
        ) = compute_batch_targets(
            batch, device, same_surface_threshold=same_surface_threshold,
        )

        if validity.sum() == 0:
            continue

        # Random chain
        chain_name = np.random.choice(chain_names)
        chain = ALL_CHAINS[chain_name]

        mode_results = run_refine_chain(
            model, batch, image,
            targets, validity, dir_sparse_mask, dir_dense_mask,
            dir_axis_weight,
            chain, {}, patch_size, device, amp_dtype,
            output_sigmoid=output_sigmoid,
            same_surface_threshold=same_surface_threshold,
            refine=use_refine,
        )

        if not mode_results:
            print(f"{si:4d}  {idx:6d}  {aug_str:<20s}  {chain_name:<8s}  "
                  f"  --  (no scale data available)", flush=True)
            continue

        # Print per-pass metrics
        for pi, p in enumerate(mode_results):
            pred = p["pred"].float()
            gt = p["gt"].float()
            v = p["validity"].float()
            if v.sum() > 0:
                mse_cos = F.mse_loss(pred[:, 0:1] * v, gt[:, 0:1] * v).item()
                l1_mag = F.l1_loss(pred[:, 1:2] * v, gt[:, 1:2] * v).item()
            else:
                mse_cos = float("nan")
                l1_mag = float("nan")
            print(f"{si:4d}  {idx:6d}  {aug_str:<20s}  {chain_name:<8s}  "
                  f"{pi:4d}  {p['offset']:>3d}  {mse_cos:10.6f}  {l1_mag:10.6f}",
                  flush=True)

        _save_chain_tif(mode_results, chain_name, aug_str, si, output_dir,
                        patch_info=batch["patch_info"], patch_size=patch_size)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
