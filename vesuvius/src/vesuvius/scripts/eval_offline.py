import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tifffile
except Exception as e:  # pragma: no cover
    tifffile = None

import torch

# Local imports from the project
from vesuvius.models.run.inference import Inferer
from vesuvius.models.evaluation.iou_dice import IOUDiceMetric
from vesuvius.models.evaluation.hausdorff import HausdorffDistanceMetric


def load_tif(path: Path) -> np.ndarray:
    if tifffile is None:
        raise RuntimeError("tifffile is required but not installed.")
    arr = tifffile.imread(str(path))
    return arr


def save_tif(path: Path, arr: np.ndarray) -> None:
    if tifffile is None:
        raise RuntimeError("tifffile is required but not installed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


def binarize_if_needed(arr: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 2:
        # Convert to {0,1}
        return (arr > 0).astype(np.uint8)
    return arr.astype(np.int32)


def as_torch_batch(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Both pred and gt are either 2D (H, W) or 3D (Z, H, W)
    # Add batch dimension
    p = pred
    g = gt
    if p.ndim == 2:
        p = p[None, ...]
    if g.ndim == 2:
        g = g[None, ...]
    # Convert to tensors
    p_t = torch.from_numpy(p)
    g_t = torch.from_numpy(g)

    # For multi-class, ensure integer labels
    if num_classes > 2:
        p_t = p_t.to(torch.int64)
        g_t = g_t.to(torch.int64)
    else:
        # Binary labels as 0/1 float is fine
        p_t = p_t.to(torch.float32)
        g_t = g_t.to(torch.float32)

    # Expand to shape expected by metrics when necessary
    # Metrics handle various shapes, including (B, D, H, W) or (B, H, W)
    return p_t, g_t


def find_label_for_image(img_path: Path, labels_dir: Optional[Path], label_suffix: Optional[str]) -> Optional[Path]:
    # Priority 1: explicit labels_dir with same stem and any tif extension
    if labels_dir is not None:
        candidate = labels_dir / (img_path.stem + ".tif")
        if candidate.exists():
            return candidate
        candidate = labels_dir / (img_path.stem + ".tiff")
        if candidate.exists():
            return candidate
    # Priority 2: label next to image with suffix
    if label_suffix:
        candidate = img_path.with_name(img_path.stem + label_suffix)
        if candidate.exists():
            return candidate
    # Priority 3: same folder, common label patterns
    for suf in ["_label.tif", "_labels.tif", "_gt.tif", "_mask.tif"]:
        candidate = img_path.with_name(img_path.stem + suf)
        if candidate.exists():
            return candidate
    return None


def collect_images(images: Path) -> List[Path]:
    if images.is_file() and images.suffix.lower() in {".tif", ".tiff"}:
        return [images]
    if images.is_dir():
        return sorted([p for p in images.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    raise FileNotFoundError(f"No TIFFs found at {images}")


def run_inference_for_image(
    model_path: str,
    image_path: Path,
    out_dir: Path,
    device: str,
    normalization: str,
    do_tta: bool,
    patch_size: Optional[Tuple[int, ...]] = None,
    tiff_activation: str = "argmax",
) -> Path:
    # Run Inferer in TIFF mode with per-image output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    inferer = Inferer(
        model_path=model_path,
        input_dir=str(image_path),
        output_dir=str(out_dir),
        input_format="tiff",
        tta_type="rotation",
        do_tta=do_tta,
        batch_size=1,
        patch_size=patch_size,
        save_softmax=False,
        tiff_activation=tiff_activation,
        normalization_scheme=normalization,
        device=device,
        verbose=False,
        skip_empty_patches=True,
    )
    # Force TIFF small/large path resolution inside Inferer
    inferer.intensity_props_json = None
    result = inferer.infer()
    if isinstance(result, tuple) and len(result) == 2 and result[0] == "tiff":
        # result[1] is list of saved output paths; prefer TIFF output
        saved = [Path(p) for p in result[1]]
        if not saved:
            raise RuntimeError("Inference returned no outputs for TIFF input")
        for p in saved:
            if p.suffix.lower() in {".tif", ".tiff"}:
                return p
        # If only Zarr was produced, inform user this evaluator is TIFF-only
        raise RuntimeError(
            "Inferer produced Zarr outputs for a large TIFF. This evaluator currently supports TIFF outputs only. "
            "Consider cropping or downscaling the input, or lower input size.")
    else:
        # Fallback: if not TIFF path, Inferer wrote zarr logits; not expected here
        raise RuntimeError("Expected TIFF inference path, got logits/coords instead")


def evaluate_dataset(
    model_path: str,
    images: Path,
    output_root: Path,
    labels_dir: Optional[Path],
    label_suffix: Optional[str],
    device: str,
    num_classes: int,
    normalization: str,
    do_tta: bool,
    patch_size: Optional[Tuple[int, ...]],
    metrics: List[str],
) -> Dict[str, float]:
    image_list = collect_images(images)
    if not image_list:
        raise FileNotFoundError(f"No TIFF images found in {images}")

    # Prepare metrics
    use_iou_dice = "iou_dice" in metrics
    use_hausdorff = "hausdorff" in metrics
    iou_dice_metric = IOUDiceMetric(num_classes=num_classes) if use_iou_dice else None
    hausdorff_metric = HausdorffDistanceMetric(num_classes=num_classes, percentile=95.0) if use_hausdorff else None

    per_image_stats: List[Dict[str, float]] = []

    for img_path in image_list:
        base_name = img_path.stem
        out_dir = output_root / base_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Locate label
        label_path = find_label_for_image(img_path, labels_dir, label_suffix)
        if label_path is None:
            print(f"Warning: no label found for {img_path.name}; skipping metrics, saving only prediction.")

        # Save a copy of the input
        try:
            img_arr = load_tif(img_path)
            save_tif(out_dir / "input.tif", img_arr)
        except Exception as e:
            print(f"Warning: failed to copy input {img_path.name}: {e}")

        # Run inference and save prediction
        pred_path = run_inference_for_image(
            model_path=model_path,
            image_path=img_path,
            out_dir=out_dir,
            device=device,
            normalization=normalization,
            do_tta=do_tta,
            patch_size=patch_size,
            tiff_activation="argmax",
        )

        # Standardize prediction filename to pred.tif
        try:
            pred_arr = load_tif(pred_path)
            # The inferer maps binary argmax {0,1}->{0,255}; binarize if needed
            pred_arr = binarize_if_needed(pred_arr, num_classes)
            save_tif(out_dir / "pred.tif", pred_arr)
        except Exception as e:
            print(f"Error reading prediction at {pred_path}: {e}")
            continue

        # If label exists, read and evaluate
        if label_path is not None:
            try:
                gt_arr = load_tif(label_path)
                gt_arr = binarize_if_needed(gt_arr, num_classes)
                save_tif(out_dir / "label.tif", gt_arr)
            except Exception as e:
                print(f"Warning: failed to process label {label_path.name}: {e}")
                gt_arr = None

            image_stats: Dict[str, float] = {}
            if gt_arr is not None:
                pred_t, gt_t = as_torch_batch(pred_arr, gt_arr, num_classes)

                if iou_dice_metric is not None:
                    image_stats.update(iou_dice_metric.compute(pred_t, gt_t))
                if hausdorff_metric is not None:
                    image_stats.update(hausdorff_metric.compute(pred_t, gt_t))

            # Save per-image stats
            with (out_dir / "stats.json").open("w") as f:
                json.dump(image_stats, f, indent=2)

            per_image_stats.append(image_stats)

    # Aggregate over images
    summary: Dict[str, float] = {}
    if per_image_stats:
        keys = set().union(*[s.keys() for s in per_image_stats])
        for k in keys:
            vals = [s[k] for s in per_image_stats if k in s]
            if len(vals) > 0:
                summary[k] = float(np.mean(vals))

    with (output_root / "summary.json").open("w") as f:
        json.dump({"num_images": len(image_list), "metrics": summary}, f, indent=2)

    return summary


def parse_patch_size(ps: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not ps:
        return None
    parts = [int(x.strip()) for x in ps.split(",") if x.strip()]
    if not parts:
        return None
    if len(parts) not in (2, 3):
        raise ValueError("patch-size must be 2 or 3 integers, e.g., '192,192' or '192,192,192'")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation on TIFF inputs using project Inferer")
    parser.add_argument("--model", required=True, help="Path to model folder or checkpoint")
    parser.add_argument("--images", required=True, help="Path to a TIFF file or a directory of TIFFs")
    parser.add_argument("--output", required=True, help="Directory to write predictions and stats")
    parser.add_argument("--labels", default=None, help="Optional directory of label TIFFs matching image stems")
    parser.add_argument("--label-suffix", default=None, help="Optional label suffix, e.g., '_label.tif'")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes (default: 2 for binary)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    parser.add_argument("--normalization", default="instance_zscore", help="Normalization scheme for inference")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--patch-size", default=None, help="Optional patch size, e.g., '192,192' or '192,192,192'")
    parser.add_argument(
        "--metrics",
        default="iou_dice,hausdorff",
        help="Comma-separated metrics to compute: iou_dice, hausdorff",
    )

    args = parser.parse_args()

    images_path = Path(args.images)
    output_root = Path(args.output)
    labels_dir = Path(args.labels) if args.labels else None
    label_suffix = args.label_suffix
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    patch_size = parse_patch_size(args.patch_size)

    output_root.mkdir(parents=True, exist_ok=True)

    summary = evaluate_dataset(
        model_path=args.model,
        images=images_path,
        output_root=output_root,
        labels_dir=labels_dir,
        label_suffix=label_suffix,
        device=args.device,
        num_classes=args.num_classes,
        normalization=args.normalization,
        do_tta=not args.no_tta,
        patch_size=patch_size,
        metrics=metrics,
    )

    print("\nEvaluation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
