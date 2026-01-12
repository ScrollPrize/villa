#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall


MODEL_MAP: Dict[str, Type] = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


@dataclass(frozen=True)
class CocoImage:
    file_name: str
    image_id: int
    width: int
    height: int
    annotations: List[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize RF-DETR predictions on COCO test samples."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pth file.")
    parser.add_argument("--dataset-dir", required=True, help="RF-DETR dataset directory.")
    parser.add_argument(
        "--split",
        choices=("train", "valid", "val", "test"),
        default="test",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--model-size",
        choices=MODEL_MAP.keys(),
        default="base",
        help="RF-DETR model size used for training.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-dets", type=int, default=5)
    parser.add_argument("--top-k-labels", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/rfdetr_predictions")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Upscale factor for saved images.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Font size for labels (tries to load a truetype font).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Box line width in pixels (after scaling).",
    )
    return parser.parse_args()


def load_coco(ann_path: str) -> Tuple[List[CocoImage], Dict[int, str]]:
    with open(ann_path, "r") as handle:
        data = json.load(handle)

    id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    ann_by_image: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    images: List[CocoImage] = []
    for img in data.get("images", []):
        image_id = img["id"]
        images.append(
            CocoImage(
                file_name=img["file_name"],
                image_id=image_id,
                width=img.get("width", 0),
                height=img.get("height", 0),
                annotations=ann_by_image.get(image_id, []),
            )
        )
    return images, id_to_name


def load_font(size: int | None) -> ImageFont.ImageFont:
    if size is None:
        return ImageFont.load_default()
    for font_name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return draw.textsize(text, font=font)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    color: str,
    label: str,
    font: ImageFont.ImageFont,
    line_width: int,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    if label:
        padding = 2
        text_w, text_h = text_size(draw, font, label)
        text_x = x1 + padding
        text_y = max(0, y1 - text_h - padding * 2)
        draw.rectangle(
            [text_x - padding, text_y - padding, text_x + text_w + padding, text_y + text_h + padding],
            fill="black",
        )
        draw.text((text_x, text_y), label, fill="white", font=font)


def draw_label_block(
    draw: ImageDraw.ImageDraw,
    origin: Tuple[float, float],
    lines: List[str],
    font: ImageFont.ImageFont,
) -> None:
    if not lines:
        return
    padding = 3
    sizes = [text_size(draw, font, line) for line in lines]
    block_w = max(w for w, _ in sizes)
    block_h = sum(h for _, h in sizes) + padding * (len(lines) + 1)
    x, y = origin
    draw.rectangle([x, y, x + block_w + 2 * padding, y + block_h], fill="black")
    cursor_y = y + padding
    for line, (_, h) in zip(lines, sizes):
        draw.text((x + padding, cursor_y), line, fill="white", font=font)
        cursor_y += h + padding


def main() -> int:
    args = parse_args()
    split_dir = "valid" if args.split == "val" else args.split
    ann_path = os.path.join(args.dataset_dir, split_dir, "_annotations.coco.json")
    images, id_to_name = load_coco(ann_path)
    if not images:
        raise RuntimeError(f"No images found in {ann_path}")

    rng = random.Random(args.seed)
    samples = rng.sample(images, min(args.num_samples, len(images)))

    model_kwargs = {}
    if args.device:
        model_kwargs["device"] = args.device
    model_kwargs["pretrain_weights"] = args.checkpoint

    model_cls = MODEL_MAP[args.model_size]
    model = model_cls(**model_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    font_size = args.font_size
    if font_size is None and args.scale >= 1:
        font_size = max(12, int(10 * args.scale))
    font = load_font(font_size)

    for item in samples:
        image_path = os.path.join(args.dataset_dir, split_dir, item.file_name)
        image_original = Image.open(image_path).convert("RGB")
        scale = max(args.scale, 1.0)
        if scale != 1.0:
            new_size = (int(image_original.width * scale), int(image_original.height * scale))
            image = image_original.resize(new_size, resample=Image.NEAREST)
        else:
            image = image_original.copy()
        draw = ImageDraw.Draw(image)

        # Ground-truth boxes
        for ann in item.annotations:
            x, y, w, h = ann["bbox"]
            gt_box = (x * scale, y * scale, (x + w) * scale, (y + h) * scale)
            gt_label = id_to_name.get(ann["category_id"], f"gt_{ann['category_id']}")
            draw_box(draw, gt_box, "green", f"gt:{gt_label}", font, args.line_width)

        # Predictions
        detections = model.predict(image_original, threshold=0.0)
        scores = np.asarray(detections.confidence) if len(detections) else np.array([])
        class_ids = np.asarray(detections.class_id) if len(detections) else np.array([])
        boxes = np.asarray(detections.xyxy) if len(detections) else np.array([])

        if args.top_k_labels > 0 and scores.size > 0:
            order = scores.argsort()[::-1][: args.top_k_labels]
            top_lines = []
            for idx in order:
                pred_id = int(class_ids[idx])
                pred_label = id_to_name.get(pred_id, f"pred_{pred_id}")
                score = float(scores[idx])
                top_lines.append(f"{pred_label} {score:.2f}")
            draw_label_block(
                draw,
                origin=(4, 4),
                lines=[f"top{len(top_lines)}"] + top_lines,
                font=font,
            )

        if scores.size > 0:
            keep = scores >= args.threshold
            keep_idx = np.where(keep)[0]
            if keep_idx.size > 0:
                order = keep_idx[np.argsort(scores[keep_idx])[::-1]]
                order = order[: args.max_dets]
                for idx in order:
                    box = boxes[idx]
                    pred_id = int(class_ids[idx])
                    pred_label = id_to_name.get(pred_id, f"pred_{pred_id}")
                    score = float(scores[idx])
                    box = (box * scale).tolist()
                    draw_box(draw, tuple(box), "red", f"{pred_label} {score:.2f}", font, args.line_width)

        base_name = os.path.basename(item.file_name)
        out_path = os.path.join(args.output_dir, base_name)
        image.save(out_path)

    print(f"Saved {len(samples)} images to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
