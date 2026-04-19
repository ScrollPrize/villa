#!/usr/bin/env python3
"""Test multi-scale refinement: run 3 inference modes on random samples.

Modes:
  1. Scale 0 only (baseline, no prior)
  2. Chain -1 -> 0 (coarse prior feeds into base)
  3. Chain 0 -> +1 (base prior feeds into fine)

For each sample, computes per-channel metrics vs GT and saves a
grid image showing all 3 modes side by side.

Usage:
    python lasagna/scripts/test_refine_scales.py \\
        --config lasagna/configs/tifxyz_train_s3_srv.json \\
        --weights runs/tifxyz3d/.../model_best.pt \\
        --num-samples 5 --output-dir tmp/refine_test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vesuvius" / "src"))

from tifxyz_lasagna_dataset import (
    TifxyzLasagnaDataset,
    collate_variable_surfaces,
)
from tifxyz_labels import compute_batch_targets
from train_tifxyz import (
    build_model,
    _build_refine_input,
    _get_gt_at_offset,
    _read_ct_at_offset,
    _resample_prior,
    _SCALE_CHANNEL_VALUES,
)


def _run_chain(
    model, batch, image, targets_0, validity_0, dsm_0, ddm_0, daw_0,
    chain, vol_groups, patch_size, device, amp_dtype, output_sigmoid,
    same_surface_threshold,
):
    """Run a chain of scale offsets and return per-pass predictions + GT."""
    results = []
    prev_pred = None
    prev_offset = None

    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=True,
    ):
        for offset, has_prior in chain:
            if offset == 0:
                ct = image
            else:
                ct = _read_ct_at_offset(
                    batch["patch_info"], vol_groups, offset, patch_size, device,
                )

            if has_prior and prev_pred is not None:
                prior = _resample_prior(
                    prev_pred.detach(), prev_offset, offset, patch_size,
                )
            else:
                prior = None

            model_in = _build_refine_input(ct, prior, offset)
            raw = model(model_in)["output"]
            pred = torch.sigmoid(raw) if output_sigmoid else raw

            tgt, val, dsm, ddm, daw = _get_gt_at_offset(
                batch, targets_0, validity_0, dsm_0, ddm_0, daw_0,
                offset, device,
                compute_batch_targets_fn=compute_batch_targets,
                same_surface_threshold=same_surface_threshold,
            )

            # Compute metrics
            cos_mask = val
            if cos_mask.sum() > 0:
                mse_cos = F.mse_loss(
                    pred[:, 0:1] * cos_mask,
                    tgt[:, 0:1] * cos_mask,
                ).item()
                l1_mag = F.l1_loss(
                    pred[:, 1:2] * cos_mask,
                    tgt[:, 1:2] * cos_mask,
                ).item()
            else:
                mse_cos = float("nan")
                l1_mag = float("nan")

            results.append({
                "offset": offset,
                "has_prior": has_prior,
                "pred": pred.detach().cpu(),
                "gt": tgt.detach().cpu(),
                "validity": val.detach().cpu(),
                "mse_cos": mse_cos,
                "l1_mag": l1_mag,
            })

            prev_pred = pred
            prev_offset = offset

    return results


def _save_grid(results_per_mode, sample_idx, output_dir, patch_size):
    """Save a grid image showing cos channel for each mode."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available, skipping grid image", flush=True)
        return

    mode_names = ["scale_0_only", "chain_m1_to_0", "chain_0_to_p1"]
    n_modes = len(results_per_mode)

    # Use middle Z slice
    mid_z = patch_size // 2
    cell_size = patch_size
    margin = 4
    header_h = 20

    # Grid: rows = modes, cols = [pred_cos, gt_cos, pred_mag, gt_mag]
    n_cols = 4
    col_labels = ["pred_cos", "gt_cos", "pred_mag", "gt_mag"]
    img_w = n_cols * (cell_size + margin) + margin
    img_h = n_modes * (cell_size + margin + header_h) + margin

    canvas = Image.new("RGB", (img_w, img_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    for row, (mode_name, passes) in enumerate(
        zip(mode_names, results_per_mode)
    ):
        # Use last pass in chain
        last = passes[-1]
        pred = last["pred"]
        gt = last["gt"]
        val = last["validity"]

        y0 = margin + row * (cell_size + margin + header_h)

        # Header text
        score_str = f"cos={last['mse_cos']:.4f} mag={last['l1_mag']:.4f}"
        draw.text(
            (margin, y0), f"{mode_name}: {score_str}",
            fill=(200, 200, 200),
        )
        y0 += header_h

        # Determine the z-slice based on pred spatial size
        pZ = pred.shape[2]
        mid = pZ // 2

        slices = [
            pred[0, 0, mid].numpy(),   # pred cos
            gt[0, 0, mid].numpy(),     # gt cos
            pred[0, 1, mid].numpy(),   # pred mag
            gt[0, 1, mid].numpy(),     # gt mag
        ]

        for col, arr in enumerate(slices):
            x0 = margin + col * (cell_size + margin)
            # Normalize to [0, 255]
            arr_clip = np.clip(arr, 0, 1)
            arr_u8 = (arr_clip * 255).astype(np.uint8)
            # Resize if needed
            cell_img = Image.fromarray(arr_u8, mode="L").resize(
                (cell_size, cell_size), Image.NEAREST,
            )
            canvas.paste(cell_img, (x0, y0))

    out_path = output_dir / f"sample_{sample_idx:04d}.png"
    canvas.save(str(out_path))
    print(f"  saved {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-scale refinement on random samples.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="tmp/refine_test")
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    patch_size = args.patch_size

    amp_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
    }[args.precision]

    # Load config
    with open(args.config) as f:
        config = json.load(f)
    config["patch_size"] = patch_size
    config["refine_mode"] = True

    # Build model (refine=True for 11ch input)
    model, norm_type, upsample_mode = build_model(
        patch_size, device, weights=args.weights,
        refine=True,
    )
    output_sigmoid = False
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        output_sigmoid = ckpt.get("output_sigmoid", False)
    model.eval()

    # Build dataset
    dataset = TifxyzLasagnaDataset(
        config, apply_augmentation=False,
        refine_mode=True,
    )
    vol_groups = dataset.volume_groups
    same_surface_threshold = float(config.get("same_surface_threshold", 0.0))

    # Define chains
    chains = {
        "scale_0_only": [(0, False)],
        "chain_m1_to_0": [(-1, False), (0, True)],
        "chain_0_to_p1": [(0, False), (1, True)],
    }

    # Random sample indices
    n = len(dataset)
    indices = np.random.choice(n, size=min(args.num_samples, n), replace=False)

    all_scores = []

    for si, idx in enumerate(indices):
        print(f"[{si+1}/{len(indices)}] sample {idx}...", flush=True)

        sample = dataset[idx]
        if sample is None:
            print("  skipped (read error)", flush=True)
            continue

        batch = collate_variable_surfaces([sample])
        if batch is None:
            print("  skipped (empty)", flush=True)
            continue

        image = batch["image"].to(device)
        (
            targets, validity,
            dir_sparse_mask, dir_dense_mask, dir_axis_weight,
            _, _, _,
        ) = compute_batch_targets(
            batch, device,
            same_surface_threshold=same_surface_threshold,
        )

        if validity.sum() == 0:
            print("  skipped (no valid voxels)", flush=True)
            continue

        results_per_mode = []
        row = {"sample_idx": int(idx)}

        for mode_name, chain in chains.items():
            mode_results = _run_chain(
                model, batch, image,
                targets, validity, dir_sparse_mask, dir_dense_mask, dir_axis_weight,
                chain, vol_groups, patch_size, device, amp_dtype,
                output_sigmoid, same_surface_threshold,
            )
            results_per_mode.append(mode_results)

            last = mode_results[-1]
            row[f"{mode_name}_mse_cos"] = last["mse_cos"]
            row[f"{mode_name}_l1_mag"] = last["l1_mag"]
            print(f"  {mode_name}: cos={last['mse_cos']:.4f} "
                  f"mag={last['l1_mag']:.4f}", flush=True)

        all_scores.append(row)
        _save_grid(results_per_mode, si, output_dir, patch_size)

    # Write CSV summary
    if all_scores:
        csv_path = output_dir / "scores.csv"
        keys = all_scores[0].keys()
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in all_scores:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"\nScores saved to {csv_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
