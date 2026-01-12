#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

from PIL import Image, ImageDraw

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


MODEL_MAP: Dict[str, Type] = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RF-DETR on ALPUB_v2 (COCO format)."
    )
    parser.add_argument("--dataset-dir", required=True, help="RF-DETR dataset directory.")
    parser.add_argument("--output-dir", default="runs/rfdetr_alpub_v2")
    parser.add_argument(
        "--model-size",
        choices=MODEL_MAP.keys(),
        default="nano",
        help="RF-DETR model size to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=0,
        help="Save an extra checkpoint every N steps (0 disables).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("step", "cosine"),
        default="step",
        help="Learning rate schedule.",
    )
    parser.add_argument(
        "--lr-drop",
        type=int,
        default=100,
        help="Epoch to start LR drop when using step scheduler.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=0.0,
        help="Warmup epochs for LR scheduler.",
    )
    parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=0.0,
        help="Minimum LR factor for cosine schedule.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Console logging frequency (steps).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override model input resolution (divisible by 56).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Override device selection (defaults to auto).",
    )
    parser.add_argument(
        "--pretrain-weights",
        default=None,
        help="Override pretrained checkpoint (e.g., rf-detr-nano.pth).",
    )
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Disable loading pretrained weights.",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable multi-scale training (default in RF-DETR).",
    )
    parser.add_argument(
        "--no-multi-scale",
        action="store_true",
        help="Disable multi-scale training.",
    )
    parser.add_argument(
        "--square-resize-div-64",
        action="store_true",
        help="Use square resize transforms (default in RF-DETR).",
    )
    parser.add_argument(
        "--no-square-resize-div-64",
        action="store_true",
        help="Disable square resize transforms.",
    )
    parser.add_argument("--run-test", action="store_true", help="Run test split after training.")
    parser.add_argument("--no-run-test", action="store_true", help="Skip test split evaluation.")
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default=None, help="W&B project name.")
    parser.add_argument("--wandb-run", default=None, help="W&B run name.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team/org).")
    parser.add_argument(
        "--wandb-log-images",
        action="store_true",
        help="Log sample predictions to W&B each N epochs.",
    )
    parser.add_argument(
        "--wandb-log-steps",
        type=int,
        default=0,
        help="Log training metrics to W&B every N steps (0 disables).",
    )
    parser.add_argument(
        "--wandb-image-count",
        type=int,
        default=8,
        help="Number of images to log per W&B update.",
    )
    parser.add_argument(
        "--wandb-image-split",
        choices=("train", "valid", "test", "val"),
        default="valid",
        help="Dataset split to sample images from.",
    )
    parser.add_argument(
        "--wandb-image-every",
        type=int,
        default=5,
        help="Log images every N epochs.",
    )
    parser.add_argument(
        "--wandb-image-steps",
        type=int,
        default=0,
        help="Log images every N training steps (0 disables).",
    )
    parser.add_argument(
        "--wandb-image-threshold",
        type=float,
        default=0.3,
        help="Score threshold for logging predictions.",
    )
    return parser.parse_args()


def validate_dataset_dir(dataset_dir: str) -> None:
    required = [
        os.path.join(dataset_dir, "train", "_annotations.coco.json"),
        os.path.join(dataset_dir, "valid", "_annotations.coco.json"),
        os.path.join(dataset_dir, "test", "_annotations.coco.json"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing RF-DETR dataset files:\n" + "\n".join(missing)
        )


def build_model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if args.resolution is not None:
        kwargs["resolution"] = args.resolution
    if args.device is not None:
        kwargs["device"] = args.device
    if args.no_pretrain:
        kwargs["pretrain_weights"] = None
    elif args.pretrain_weights is not None:
        kwargs["pretrain_weights"] = args.pretrain_weights
    return kwargs


@dataclass(frozen=True)
class SampleItem:
    path: str
    gt_id: int
    gt_name: str


def load_wandb_samples(
    dataset_dir: str,
    split: str,
    count: int,
    seed: int = 1337,
) -> Tuple[List[SampleItem], Dict[int, str]]:
    split_map = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    split_dir = split_map.get(split, split)
    ann_path = os.path.join(dataset_dir, split_dir, "_annotations.coco.json")
    with open(ann_path, "r") as handle:
        data = json.load(handle)

    categories = data.get("categories", [])
    id_to_name = {cat["id"]: cat["name"] for cat in categories}

    image_id_to_name = {img["id"]: img["file_name"] for img in data.get("images", [])}
    image_id_to_gt = {}
    for ann in data.get("annotations", []):
        image_id_to_gt.setdefault(ann["image_id"], ann["category_id"])

    items = []
    for image_id, file_name in image_id_to_name.items():
        if image_id not in image_id_to_gt:
            continue
        gt_id = image_id_to_gt[image_id]
        gt_name = id_to_name.get(gt_id, f"cls_{gt_id}")
        path = os.path.join(dataset_dir, split_dir, file_name)
        items.append(SampleItem(path=path, gt_id=gt_id, gt_name=gt_name))

    if not items:
        return [], id_to_name

    rng = random.Random(seed)
    sample_count = min(count, len(items))
    return rng.sample(items, sample_count), id_to_name


class WandbImageLogger:
    def __init__(
        self,
        model,
        samples: List[SampleItem],
        id_to_name: Dict[int, str],
        log_every: int,
        threshold: float,
        tag: str,
    ):
        self.model = model
        self.samples = samples
        self.id_to_name = id_to_name
        self.log_every = max(1, log_every)
        self.threshold = threshold
        self.tag = tag

    def __call__(self, log_stats: Dict[str, object]) -> None:
        self._maybe_log(epoch=int(log_stats.get("epoch", 0)), step=None)

    def _maybe_log(self, epoch: int, step: int | None) -> None:
        if os.environ.get("RANK", "0") != "0":
            return
        try:
            import wandb
        except Exception:
            return
        if not wandb.run:
            return

        if epoch % self.log_every != 0:
            return

        log_wandb_images(
            model=self.model,
            samples=self.samples,
            id_to_name=self.id_to_name,
            threshold=self.threshold,
            tag=self.tag,
            epoch=epoch,
            step=step,
        )


class WandbImageStepLogger:
    def __init__(
        self,
        model,
        samples: List[SampleItem],
        id_to_name: Dict[int, str],
        log_steps: int,
        threshold: float,
        tag: str,
    ):
        self.model = model
        self.samples = samples
        self.id_to_name = id_to_name
        self.log_steps = log_steps
        self.threshold = threshold
        self.tag = tag

    def __call__(self, callback_dict: Dict[str, object]) -> None:
        if self.log_steps <= 0:
            return
        if os.environ.get("RANK", "0") != "0":
            return
        step = int(callback_dict.get("step", 0))
        if step == 0 or step % self.log_steps != 0:
            return
        epoch = int(callback_dict.get("epoch", 0))
        log_wandb_images(
            model=self.model,
            samples=self.samples,
            id_to_name=self.id_to_name,
            threshold=self.threshold,
            tag=self.tag,
            epoch=epoch,
            step=step,
        )


def log_wandb_images(
    model,
    samples: List[SampleItem],
    id_to_name: Dict[int, str],
    threshold: float,
    tag: str,
    epoch: int | None,
    step: int | None,
) -> None:
    try:
        import wandb
    except Exception:
        return
    if not wandb.run:
        return

    images = []
    training_module = model.model.model
    was_training = training_module.training
    training_module.eval()
    try:
        for sample in samples:
            image = Image.open(sample.path).convert("RGB")
            detections = model.predict(image, threshold=threshold)
            pred_label = "none"
            pred_score = 0.0
            if len(detections) > 0:
                best_idx = int(detections.confidence.argmax())
                pred_id = int(detections.class_id[best_idx])
                pred_label = id_to_name.get(pred_id, f"cls_{pred_id}")
                pred_score = float(detections.confidence[best_idx])

                box = detections.xyxy[best_idx]
                draw = ImageDraw.Draw(image)
                draw.rectangle(box.tolist(), outline="red", width=2)
            caption = f"gt={sample.gt_name} pred={pred_label} score={pred_score:.2f}"
            images.append(wandb.Image(image, caption=caption))
    finally:
        if was_training:
            training_module.train()

    if images:
        log_dict = {f"examples/{tag}": images}
        if epoch is not None:
            log_dict["epoch"] = epoch
        if step is not None:
            log_dict["step"] = step
        wandb.log(log_dict)


def main() -> int:
    args = parse_args()
    validate_dataset_dir(args.dataset_dir)

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    model_cls = MODEL_MAP[args.model_size]
    model = model_cls(**build_model_kwargs(args))

    train_kwargs: Dict[str, object] = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_steps": args.checkpoint_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "lr_scheduler": args.lr_scheduler,
        "lr_drop": args.lr_drop,
        "warmup_epochs": args.warmup_epochs,
        "lr_min_factor": args.lr_min_factor,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "print_freq": args.print_freq,
        "wandb_log_steps": args.wandb_log_steps,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "wandb": args.wandb,
        "project": args.wandb_project,
        "run": args.wandb_run,
    }

    if args.no_multi_scale:
        train_kwargs["multi_scale"] = False
        train_kwargs["expanded_scales"] = False
    elif args.multi_scale:
        train_kwargs["multi_scale"] = True
        train_kwargs["expanded_scales"] = True

    if args.no_square_resize_div_64:
        train_kwargs["square_resize_div_64"] = False
    elif args.square_resize_div_64:
        train_kwargs["square_resize_div_64"] = True

    if args.no_run_test:
        train_kwargs["run_test"] = False
    elif args.run_test:
        train_kwargs["run_test"] = True

    if args.wandb and args.wandb_log_images:
        samples, id_to_name = load_wandb_samples(
            args.dataset_dir,
            args.wandb_image_split,
            args.wandb_image_count,
        )
        if samples:
            model.callbacks["on_fit_epoch_end"].append(
                WandbImageLogger(
                    model=model,
                    samples=samples,
                    id_to_name=id_to_name,
                    log_every=args.wandb_image_every,
                    threshold=args.wandb_image_threshold,
                    tag=args.wandb_image_split,
                )
            )
            if args.wandb_image_steps:
                model.callbacks["on_train_batch_start"].append(
                    WandbImageStepLogger(
                        model=model,
                        samples=samples,
                        id_to_name=id_to_name,
                        log_steps=args.wandb_image_steps,
                        threshold=args.wandb_image_threshold,
                        tag=args.wandb_image_split,
                    )
                )

    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
