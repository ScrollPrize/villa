import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import tifffile
except Exception as e:  # pragma: no cover
    tifffile = None

import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

# Local imports from the project
from vesuvius.models.run.inference import Inferer
from vesuvius.models.evaluation.iou_dice import IOUDiceMetric
from vesuvius.models.evaluation.hausdorff import HausdorffDistanceMetric
from vesuvius.models.evaluation.connected_components import ConnectedComponentsMetric
from vesuvius.models.evaluation.critical_components import CriticalComponentsMetric
from vesuvius.models.evaluation.skeleton_branch_points import SkeletonBranchPointsMetric


def _normalize_metric_names(metric_names: List[str]) -> List[str]:
    if not metric_names:
        return []
    aliases = {
        "skeleton": "skeleton_branch_points",
        "skeleton_junctions": "skeleton_branch_points",
        "skeleton-branch-points": "skeleton_branch_points",
        "cc": "connected_components",
        "components": "connected_components",
        "critical": "critical_components",
        "hausdorff95": "hausdorff",
        "iou": "iou_dice",
        "dice": "iou_dice",
    }
    out = []
    for m in metric_names:
        m = m.strip().lower()
        if m in aliases:
            m = aliases[m]
        out.append(m)
    return out


def build_metrics(metric_names: List[str], num_classes: int) -> List[object]:
    """Create metric objects from names. Supports 'all'."""
    supported = [
        "iou_dice",
        "hausdorff",
        "connected_components",
        "critical_components",
        "skeleton_branch_points",
    ]
    names = _normalize_metric_names(metric_names)
    if not names or "all" in names:
        names = supported

    metrics: List[object] = []
    for name in names:
        if name == "iou_dice":
            metrics.append(IOUDiceMetric(num_classes=num_classes))
        elif name == "hausdorff":
            metrics.append(HausdorffDistanceMetric(num_classes=num_classes, percentile=95.0))
        elif name == "connected_components":
            metrics.append(ConnectedComponentsMetric(num_classes=num_classes))
        elif name == "critical_components":
            metrics.append(CriticalComponentsMetric())
        elif name == "skeleton_branch_points":
            metrics.append(SkeletonBranchPointsMetric(num_classes=num_classes))
        else:
            print(f"Warning: unsupported metric '{name}' ignored. Supported: {', '.join(supported)}")
    return metrics


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


def _to_json_safe(obj: Any) -> Any:
    """Convert numpy/torch scalars and arrays to JSON-serializable types recursively."""
    # Scalars
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # Torch scalars
    try:
        import torch as _torch  # local import safe in workers
        if isinstance(obj, _torch.Tensor):
            if obj.dim() == 0:
                return _to_json_safe(obj.item())
            return _to_json_safe(obj.detach().cpu().tolist())
    except Exception:
        pass
    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    # Containers
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def binarize_if_needed(arr: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 2:
        # Convert to {0,1}
        return (arr > 0).astype(np.uint8)
    return arr.astype(np.int32)


def as_torch_batch(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Both pred and gt are either 2D (H, W) or 3D (Z, H, W).
    # Ensure a batch dimension exists for both (B, H, W) or (B, Z, H, W).
    p = pred
    g = gt
    if p.ndim in (2, 3):
        p = p[None, ...]
    if g.ndim in (2, 3):
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
    metric_objs = build_metrics(metrics, num_classes=num_classes)

    per_image_stats: List[Dict[str, float]] = []

    for img_path in tqdm(image_list, desc="Processing images", unit="img"):
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

        # Use existing prediction if available; otherwise run inference
        standardized_pred_path = out_dir / "pred.tif"
        if standardized_pred_path.exists():
            try:
                pred_arr = load_tif(standardized_pred_path)
                pred_arr = binarize_if_needed(pred_arr, num_classes)
            except Exception as e:
                print(f"Error reading existing prediction at {standardized_pred_path}: {e}")
                continue
        else:
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
                save_tif(standardized_pred_path, pred_arr)
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
                for metric in metric_objs:
                    try:
                        image_stats.update(metric.compute(pred_t, gt_t))
                    except Exception as e:
                        print(f"Warning: metric {getattr(metric, 'name', type(metric).__name__)} failed: {e}")

            # Save per-image stats
            with (out_dir / "stats.json").open("w") as f:
                json.dump(_to_json_safe(image_stats), f, indent=2)

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


def _evaluate_one_label_pred_task(
    label_path_str: str,
    pred_path_str: str,
    out_dir_str: str,
    num_classes: int,
    metric_names: List[str],
):
    label_path = Path(label_path_str)
    pred_path = Path(pred_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read arrays
    gt_arr = load_tif(label_path)
    gt_arr = binarize_if_needed(gt_arr, num_classes)
    save_tif(out_dir / "label.tif", gt_arr)

    pred_arr = load_tif(pred_path)
    pred_arr = binarize_if_needed(pred_arr, num_classes)
    save_tif(out_dir / "pred.tif", pred_arr)

    # Build metrics inside process to avoid pickling issues
    metric_objs = build_metrics(metric_names, num_classes=num_classes)
    image_stats: Dict[str, float] = {}
    pred_t, gt_t = as_torch_batch(pred_arr, gt_arr, num_classes)
    for metric in metric_objs:
        try:
            image_stats.update(metric.compute(pred_t, gt_t))
        except Exception as e:
            print(f"Warning: metric {getattr(metric, 'name', type(metric).__name__)} failed for {label_path.name}: {e}")

    with (out_dir / "stats.json").open("w") as f:
        json.dump(_to_json_safe(image_stats), f, indent=2)

    return image_stats


def evaluate_labels_predictions(
    labels_dir: Path,
    predictions_dir: Path,
    output_root: Path,
    num_classes: int,
    metrics: List[str],
    num_workers: int = 1,
) -> Dict[str, float]:
    # Collect label files
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not predictions_dir.exists() or not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    label_list = sorted([p for p in labels_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    if not label_list:
        raise FileNotFoundError(f"No TIFF labels found in {labels_dir}")

    # Ensure all predictions exist with the same filenames
    missing = [p.name for p in label_list if not (predictions_dir / p.name).exists()]
    if missing:
        missing_str = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise FileNotFoundError(
            f"Missing {len(missing)} prediction file(s) in {predictions_dir} matching labels: {missing_str}")

    per_image_stats: List[Dict[str, float]] = []

    if num_workers and num_workers > 1:
        tasks = []
        for label_path in label_list:
            pred_path = predictions_dir / label_path.name
            base_name = label_path.stem
            out_dir = output_root / base_name
            tasks.append((str(label_path), str(pred_path), str(out_dir), num_classes, metrics))

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_evaluate_one_label_pred_task, *t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating predictions (parallel)", unit="img"):
                try:
                    image_stats = fut.result()
                except Exception as e:
                    print(f"Warning: evaluation task failed: {e}")
                    continue
                per_image_stats.append(image_stats)
    else:
        # Serial path
        metric_objs = build_metrics(metrics, num_classes=num_classes)
        for label_path in tqdm(label_list, desc="Evaluating predictions", unit="img"):
            pred_path = predictions_dir / label_path.name
            base_name = label_path.stem
            out_dir = output_root / base_name
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                gt_arr = load_tif(label_path)
                gt_arr = binarize_if_needed(gt_arr, num_classes)
                save_tif(out_dir / "label.tif", gt_arr)
            except Exception as e:
                print(f"Warning: failed to read/process label {label_path.name}: {e}")
                continue

            try:
                pred_arr = load_tif(pred_path)
                pred_arr = binarize_if_needed(pred_arr, num_classes)
                save_tif(out_dir / "pred.tif", pred_arr)
            except Exception as e:
                print(f"Warning: failed to read/process prediction {pred_path.name}: {e}")
                continue

            image_stats: Dict[str, float] = {}
            pred_t, gt_t = as_torch_batch(pred_arr, gt_arr, num_classes)
            for metric in metric_objs:
                try:
                    image_stats.update(metric.compute(pred_t, gt_t))
                except Exception as e:
                    print(f"Warning: metric {getattr(metric, 'name', type(metric).__name__)} failed: {e}")

        with (out_dir / "stats.json").open("w") as f:
            json.dump(_to_json_safe(image_stats), f, indent=2)

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
        json.dump({"num_images": len(per_image_stats), "metrics": summary}, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation for TIFFs: run inference or evaluate existing predictions.")
    # Inference mode arguments
    parser.add_argument("--model", default=None, help="Path to model folder or checkpoint (inference mode)")
    parser.add_argument("--images", default=None, help="Path to a TIFF file or a directory of TIFFs (inference mode)")
    # Evaluation-only mode arguments
    parser.add_argument("--labels", default=None, help="Directory of label TIFFs (evaluation-only mode)")
    parser.add_argument("--predictions", default=None, help="Directory of prediction TIFFs matching label filenames (evaluation-only mode)")
    # Common
    parser.add_argument("--output", required=True, help="Directory to write per-image stats and summary")
    parser.add_argument("--label-suffix", default=None, help="Optional label suffix when inferring, e.g., '_label.tif'")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes (default: 2 for binary)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    parser.add_argument("--normalization", default="instance_zscore", help="Normalization scheme for inference")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--patch-size", default=None, help="Optional patch size, e.g., '192,192' or '192,192,192'")
    parser.add_argument("--num-workers", type=int, default=1, help="Workers for evaluation-only mode (labels+predictions). Use >1 to parallelize.")
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Comma-separated metrics or 'all'. Supported: "
            "iou_dice, hausdorff, connected_components, critical_components, skeleton_branch_points"
        ),
    )

    args = parser.parse_args()

    output_root = Path(args.output)
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    patch_size = parse_patch_size(args.patch_size)

    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, float]
    # Determine mode
    if args.labels and args.predictions:
        labels_dir = Path(args.labels)
        predictions_dir = Path(args.predictions)
        summary = evaluate_labels_predictions(
            labels_dir=labels_dir,
            predictions_dir=predictions_dir,
            output_root=output_root,
            num_classes=args.num_classes,
            metrics=metrics,
            num_workers=max(1, int(args.num_workers or 1)),
        )
    elif args.model and args.images:
        images_path = Path(args.images)
        labels_dir = Path(args.labels) if args.labels else None
        label_suffix = args.label_suffix
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
    else:
        raise SystemExit("Provide either --labels and --predictions (evaluation-only) or --model and --images (inference mode).")

    print("\nEvaluation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
