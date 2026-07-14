"""Single-GPU label inspector — render N debug PNGs without spinning up DDP.

Run from the package directory; pass the trainer config and how many random
crops to draw. Example::

    python inspect_labels.py --config configs/train.example.json --n 6 --out ./previews

Use this to validate the pseudo-label pipeline (visually inspect the saved
PNGs) before consuming any training compute.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import RandomFiberCropDataset, collate_random_crops  # noqa: E402
from label_generator import (  # noqa: E402
    FiveClassConfig,
    FiveClassLabelGenerator,
)
from visualization import CLASS_NAMES, make_debug_figure  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", type=Path,
                        default=Path(tempfile.gettempdir()) / "fiber4c_inspect")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--mixed_precision", type=str, default=None,
                        help="Override config mixed_precision; one of bf16/fp16/fp32.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    mp = (args.mixed_precision or cfg.get("mixed_precision", "bf16")).lower()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[mp]

    print(f"[inspect] device={device} dtype={dtype}")
    print(f"[inspect] loading teachers...")

    pseudo_cfg = cfg["pseudo"]
    five_cfg = FiveClassConfig(
        fiber_thr=float(pseudo_cfg.get("fiber_thr", 0.5)),
        ink_thr=float(pseudo_cfg.get("ink_thr", 0.5)),
        papyrus_raw_thr=int(pseudo_cfg.get("papyrus_raw_thr", 90)),
        dark_voxel_thr=int(pseudo_cfg.get("dark_voxel_thr", 90)),
        ws_image_mode=str(pseudo_cfg.get("ws_image_mode", "distance")),
        ws_h_merge=int(pseudo_cfg.get("ws_h_merge", 14000)),
        ws_min_voxels=int(pseudo_cfg.get("ws_min_voxels", 400)),
        pca_cos_threshold=float(pseudo_cfg.get("pca_cos_threshold", 0.819)),
    )

    label_gen = FiveClassLabelGenerator(
        fiber_teacher_ckpt=pseudo_cfg["fiber_teacher"],
        ink_teacher_ckpt=pseudo_cfg.get("ink_teacher"),
        config=five_cfg,
        device=device,
        dtype=dtype,
        crop_size=tuple(cfg["patch_size"]),
    )
    print(f"[inspect] teachers loaded "
          f"(fiber {label_gen.fiber_target_name}/{label_gen.fiber_out_channels}ch, "
          f"ink {label_gen.ink_target_name}/{label_gen.ink_out_channels}ch)")

    ds_cfg = cfg["dataset"]
    dataset = RandomFiberCropDataset(
        volume_url=ds_cfg["volume_url"],
        crop_size=tuple(cfg["patch_size"]),
        storage_options=ds_cfg.get("storage_options", {"anon": True}),
        scale=int(ds_cfg.get("scale", 0)),
        min_nonempty_frac=float(ds_cfg.get("min_nonempty_frac", 0.10)),
        dark_threshold=int(ds_cfg.get("dark_threshold", 50)),
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_random_crops,
    )

    summary_rows: list[str] = []
    summary_rows.append("idx | n_inst n_vert | bg vert horiz pap ink (fractions)  | ws_t s | fiber_thr ink_thr h_merge min_vox cos")
    it = iter(loader)
    t0_all = time.time()
    for i in range(args.n):
        batch = next(it)
        image = batch["image"]                                  # (1, 1, Z, Y, X) fp32 [0,1]
        raw = batch["raw_image"]                                # (1, 1, Z, Y, X) uint8
        t0 = time.time()
        label, debug = label_gen.generate(image, raw)
        torch.cuda.synchronize(device)
        dt = time.time() - t0

        # Bring small tensors to CPU for plotting.
        image_np = image[0, 0].cpu().numpy().astype(np.float32)
        raw_np = raw[0, 0].cpu().numpy().astype(np.uint8)
        label_np = label[0, 0].cpu().numpy().astype(np.int32)
        fiber_prob_np = debug.fiber_prob[0, 0].float().cpu().numpy()
        ink_prob_np = debug.ink_prob[0, 0].float().cpu().numpy()
        inst_np = debug.instance_map[0, 0].cpu().numpy().astype(np.int64)
        counts = debug.class_counts[0].cpu().numpy().tolist()
        n_inst = int(debug.n_instances[0].item())
        n_vert = int(debug.n_vert[0].item())
        n_voxels = int(np.prod(label_np.shape))
        fracs = [c / max(1, n_voxels) for c in counts]

        title = (
            f"crop {i}: n_inst={n_inst} n_vert={n_vert} | "
            f"bg={fracs[0]:.3f} v={fracs[1]:.3f} h={fracs[2]:.3f} i={fracs[3]:.3f} | "
            f"ws={dt:.2f}s"
        )
        print(f"[inspect] {title}")
        summary_rows.append(
            f"{i:>3} | {n_inst:>6} {n_vert:>6} | "
            f"{fracs[0]:.3f} {fracs[1]:.3f} {fracs[2]:.3f} {fracs[3]:.3f} | "
            f"{dt:5.2f} | "
            f"{five_cfg.fiber_thr:>9.3f} {five_cfg.ink_thr:>7.3f} {five_cfg.ws_h_merge:>7d} "
            f"{five_cfg.ws_min_voxels:>7d} {five_cfg.pca_cos_threshold:>5.3f}"
        )

        fig = make_debug_figure(
            image_zyx=image_np,
            raw_zyx=raw_np,
            label_zyx=label_np,
            fiber_prob_zyx=fiber_prob_np,
            ink_prob_zyx=ink_prob_np,
            instance_map_zyx=inst_np,
            student_pred_zyx=None,
            title_suffix=title,
        )
        out_path = args.out / f"preview_{i:02d}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[inspect]   wrote {out_path}")

    print(f"[inspect] DONE in {time.time() - t0_all:.1f}s — {args.n} crops")
    print("=" * 96)
    for row in summary_rows:
        print(row)

    summary_path = args.out / "summary.txt"
    summary_path.write_text("\n".join(summary_rows) + "\n")
    print(f"[inspect] summary at {summary_path}")


if __name__ == "__main__":
    main()
