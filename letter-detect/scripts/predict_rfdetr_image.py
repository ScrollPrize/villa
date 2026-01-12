#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Tuple, Type

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR on a single image and save visualization."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pth file.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--model-size",
        choices=MODEL_MAP.keys(),
        default="base",
        help="RF-DETR model size used for training.",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-dets", type=int, default=5)
    parser.add_argument("--top-k-labels", type=int, default=3)
    parser.add_argument(
        "--draw-top-boxes",
        type=int,
        default=0,
        help="Draw top-N boxes by score, regardless of threshold (0 disables).",
    )
    parser.add_argument(
        "--labels-json",
        help="Optional COCO annotations file to map class IDs to names.",
    )
    parser.add_argument(
        "--dataset-dir",
        help="Optional RF-DETR dataset dir to load labels from.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "val", "test"),
        default="test",
        help="Split to read labels from when using --dataset-dir.",
    )
    parser.add_argument("--gt-label", help="Optional ground-truth label name for display logic.")
    parser.add_argument("--gt-id", type=int, help="Optional ground-truth label ID for display logic.")
    parser.add_argument("--output", help="Output image path.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Upscale factor for saved image.",
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


def load_label_map(args: argparse.Namespace) -> Dict[int, str]:
    labels_json = args.labels_json
    if labels_json is None and args.dataset_dir:
        split_dir = "valid" if args.split == "val" else args.split
        labels_json = os.path.join(args.dataset_dir, split_dir, "_annotations.coco.json")
    if not labels_json or not os.path.exists(labels_json):
        return {}
    with open(labels_json, "r") as handle:
        data = json.load(handle)
    return {c["id"]: c["name"] for c in data.get("categories", [])}


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
    lines: list[str],
    font: ImageFont.ImageFont,
    text_color: str = "white",
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
        draw.text((x + padding, cursor_y), line, fill=text_color, font=font)
        cursor_y += h + padding


def main() -> int:
    args = parse_args()
    id_to_name = load_label_map(args)
    name_to_id = {v: k for k, v in id_to_name.items()}

    gt_id = args.gt_id
    gt_name = args.gt_label
    if gt_id is None and gt_name:
        gt_id = name_to_id.get(gt_name)
    if gt_name is None and gt_id is not None:
        gt_name = id_to_name.get(gt_id, f"cls_{gt_id}")

    model_kwargs = {"pretrain_weights": args.checkpoint}
    if args.device:
        model_kwargs["device"] = args.device
    model_cls = MODEL_MAP[args.model_size]
    model = model_cls(**model_kwargs)

    image_original = Image.open(args.image).convert("RGB")
    scale = max(args.scale, 1.0)
    if scale != 1.0:
        new_size = (int(image_original.width * scale), int(image_original.height * scale))
        image = image_original.resize(new_size, resample=Image.NEAREST)
    else:
        image = image_original.copy()

    font_size = args.font_size
    if font_size is None and scale >= 1:
        font_size = max(12, int(10 * scale))
    font = load_font(font_size)
    draw = ImageDraw.Draw(image)

    detections = model.predict(image_original, threshold=0.0)
    scores = np.asarray(detections.confidence) if len(detections) else np.array([])
    class_ids = np.asarray(detections.class_id) if len(detections) else np.array([])
    boxes = np.asarray(detections.xyxy) if len(detections) else np.array([])

    if scores.size > 0 and gt_name:
        sorted_idx = scores.argsort()[::-1]
        top1_idx = int(sorted_idx[0])
        top1_id = int(class_ids[top1_idx])
        top1_name = id_to_name.get(top1_id, f"cls_{top1_id}")
        top1_score = float(scores[top1_idx])

        gt_in_topk = False
        gt_score = None
        if gt_id is not None:
            gt_indices = np.where(class_ids == gt_id)[0]
            if gt_indices.size > 0:
                gt_best_idx = int(gt_indices[np.argmax(scores[gt_indices])])
                gt_score = float(scores[gt_best_idx])
                topk_idx = sorted_idx[: max(1, args.top_k_labels)]
                gt_in_topk = gt_best_idx in topk_idx
        else:
            # Fall back to name match if no ID mapping is available.
            pred_names = [id_to_name.get(int(pid), f"cls_{int(pid)}") for pid in class_ids]
            if gt_name in pred_names:
                gt_indices = np.where(np.array(pred_names) == gt_name)[0]
                gt_best_idx = int(gt_indices[np.argmax(scores[gt_indices])])
                gt_score = float(scores[gt_best_idx])
                topk_idx = sorted_idx[: max(1, args.top_k_labels)]
                gt_in_topk = gt_best_idx in topk_idx

        if top1_name == gt_name:
            draw_label_block(
                draw,
                origin=(4, 4),
                lines=[f"{top1_name} {top1_score:.2f}"],
                font=font,
                text_color="green",
            )
        elif gt_in_topk:
            score_text = f" {gt_score:.2f}" if gt_score is not None else ""
            draw_label_block(
                draw,
                origin=(4, 4),
                lines=[f"{gt_name}{score_text}"],
                font=font,
                text_color="red",
            )
        else:
            draw_label_block(
                draw,
                origin=(4, 4),
                lines=[f"{top1_name} {top1_score:.2f}"],
                font=font,
                text_color="red",
            )
    elif args.top_k_labels > 0 and scores.size > 0:
        order = scores.argsort()[::-1][: args.top_k_labels]
        top_lines = []
        for idx in order:
            pred_id = int(class_ids[idx])
            pred_label = id_to_name.get(pred_id, f"cls_{pred_id}")
            top_lines.append(f"{pred_label} {float(scores[idx]):.2f}")
        draw_label_block(draw, origin=(4, 4), lines=[f"top{len(top_lines)}"] + top_lines, font=font)

    if scores.size > 0:
        if args.draw_top_boxes and args.draw_top_boxes > 0:
            order = scores.argsort()[::-1][: args.draw_top_boxes]
        else:
            keep = scores >= args.threshold
            keep_idx = np.where(keep)[0]
            order = keep_idx[np.argsort(scores[keep_idx])[::-1]]
            order = order[: args.max_dets]
        for idx in order:
            box = (boxes[idx] * scale).tolist()
            pred_id = int(class_ids[idx])
            pred_label = id_to_name.get(pred_id, f"cls_{pred_id}")
            score = float(scores[idx])
            draw_box(
                draw,
                tuple(box),
                "red",
                f"{pred_label} {score:.2f}",
                font,
                args.line_width,
            )

    if args.output:
        output_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output_dir, f"{base}_pred.png")

    image.save(output_path)
    print(f"Saved output to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
