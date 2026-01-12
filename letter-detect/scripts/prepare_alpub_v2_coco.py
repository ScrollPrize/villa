#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ALPUB_v2 in COCO format (one box per image)."
    )
    parser.add_argument(
        "--images-root",
        required=True,
        help="Path to ALPUB_v2 images root (folder with class subdirectories).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for COCO annotations and metadata.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--fixed-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Skip image inspection and assume a fixed width/height for all images.",
    )
    parser.add_argument(
        "--skip-bad",
        action="store_true",
        help="Skip unreadable images instead of failing.",
    )
    parser.add_argument(
        "--write-splits-csv",
        action="store_true",
        help="Write a splits.csv file mapping images to splits.",
    )
    parser.add_argument(
        "--export-rfdetr-dir",
        help=(
            "Optional RF-DETR dataset export directory. "
            "Creates train/valid/test folders with _annotations.coco.json."
        ),
    )
    parser.add_argument(
        "--rfdetr-zero-index",
        action="store_true",
        help="Shift category IDs by -1 for RF-DETR export (useful when classes are 1-based).",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to place images into the RF-DETR export directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the RF-DETR export directory.",
    )
    return parser.parse_args()


def list_classes(images_root: str) -> List[str]:
    classes = [d.name for d in os.scandir(images_root) if d.is_dir()]
    classes.sort()
    if not classes:
        raise RuntimeError(f"No class folders found under {images_root}")
    return classes


def list_images(images_root: str, class_name: str) -> List[str]:
    class_dir = os.path.join(images_root, class_name)
    files: list[str] = []
    for entry in os.scandir(class_dir):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(f"{class_name}/{entry.name}")
    files.sort()
    return files


def split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    raw = [n * r for r in ratios]
    counts = [int(math.floor(x)) for x in raw]
    remainder = n - sum(counts)
    fracs = [x - math.floor(x) for x in raw]
    for idx in sorted(range(len(fracs)), key=lambda i: fracs[i], reverse=True):
        if remainder <= 0:
            break
        counts[idx] += 1
        remainder -= 1
    return counts[0], counts[1], counts[2]


def enforce_min_splits(
    n: int, ratios: Tuple[float, float, float], counts: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    train, val, test = counts
    if n >= 3:
        if ratios[1] > 0 and val == 0:
            val = 1
        if ratios[2] > 0 and test == 0:
            test = 1
        train = n - val - test
        if train < 0:
            # Reduce val/test if needed to keep non-negative train.
            deficit = -train
            if val > 1:
                take = min(deficit, val - 1)
                val -= take
                deficit -= take
            if deficit > 0 and test > 1:
                take = min(deficit, test - 1)
                test -= take
                deficit -= take
            train = n - val - test
    return train, val, test


def coco_categories(classes: List[str]) -> List[dict]:
    return [
        {"id": idx + 1, "name": name, "supercategory": "greek_letter"}
        for idx, name in enumerate(classes)
    ]


def load_image_size(
    image_path: str, fixed_size: Optional[Tuple[int, int]], skip_bad: bool
) -> Optional[Tuple[int, int]]:
    if fixed_size:
        return fixed_size
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception as exc:
        if skip_bad:
            print(f"WARNING: failed to read {image_path}: {exc}", file=sys.stderr)
            return None
        raise


def build_coco(
    items: List[Tuple[str, int]],
    images_root: str,
    classes: List[str],
    fixed_size: Optional[Tuple[int, int]],
    skip_bad: bool,
    desc: str,
) -> dict:
    images = []
    annotations = []
    categories = coco_categories(classes)
    image_id = 1
    ann_id = 1
    skipped = 0

    for file_name, class_id in tqdm(items, desc=desc, unit="img"):
        image_path = os.path.join(images_root, file_name)
        size = load_image_size(image_path, fixed_size, skip_bad)
        if size is None:
            skipped += 1
            continue
        width, height = size

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [0, 0, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [[0, 0, width, 0, width, height, 0, height]],
            }
        )
        image_id += 1
        ann_id += 1

    return {
        "info": {
            "description": "ALPUB_v2 character crops (one box per image)",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.utcnow().isoformat() + "Z",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "skipped_images": skipped,
    }


def shift_coco_category_ids(coco: dict, offset: int) -> dict:
    if offset == 0:
        return coco
    shifted = dict(coco)
    shifted["categories"] = [
        {**cat, "id": cat["id"] + offset} for cat in coco.get("categories", [])
    ]
    shifted["annotations"] = [
        {**ann, "category_id": ann["category_id"] + offset}
        for ann in coco.get("annotations", [])
    ]
    return shifted


def link_image(src: str, dst: str, mode: str, overwrite: bool) -> None:
    if os.path.lexists(dst):
        if not overwrite:
            return
        os.remove(dst)

    if mode == "symlink":
        rel_src = os.path.relpath(src, os.path.dirname(dst))
        os.symlink(rel_src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def export_rfdetr_split(
    export_dir: str,
    split_name: str,
    coco: dict,
    images_root: str,
    link_mode: str,
    overwrite: bool,
    category_id_offset: int,
) -> None:
    split_map = {"train": "train", "val": "valid", "test": "test"}
    split_dir = os.path.join(export_dir, split_map[split_name])
    os.makedirs(split_dir, exist_ok=True)

    if category_id_offset != 0:
        coco = shift_coco_category_ids(coco, category_id_offset)

    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(ann_path, "w") as handle:
        json.dump(coco, handle, indent=2)

    for image_info in tqdm(coco["images"], desc=f"Linking {split_name}", unit="img"):
        file_name = image_info["file_name"]
        src = os.path.join(images_root, file_name)
        dst = os.path.join(split_dir, file_name)
        dst_dir = os.path.dirname(dst)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(src):
            print(f"WARNING: missing source image {src}", file=sys.stderr)
            continue
        link_image(src, dst, link_mode, overwrite)


def main() -> int:
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")

    images_root = os.path.abspath(args.images_root)
    output_dir = os.path.abspath(args.output_dir)
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    classes = list_classes(images_root)
    class_to_id = {name: idx + 1 for idx, name in enumerate(classes)}

    rng = random.Random(args.seed)
    split_items = {"train": [], "val": [], "test": []}
    split_counts_by_class: Dict[str, Dict[str, int]] = {}

    for class_name in classes:
        files = list_images(images_root, class_name)
        if not files:
            continue
        rng.shuffle(files)

        train_n, val_n, test_n = split_counts(len(files), ratios)
        train_n, val_n, test_n = enforce_min_splits(len(files), ratios, (train_n, val_n, test_n))

        split_counts_by_class[class_name] = {
            "train": train_n,
            "val": val_n,
            "test": test_n,
        }

        class_id = class_to_id[class_name]
        split_items["train"].extend((f, class_id) for f in files[:train_n])
        split_items["val"].extend((f, class_id) for f in files[train_n : train_n + val_n])
        split_items["test"].extend((f, class_id) for f in files[train_n + val_n :])

    if args.write_splits_csv:
        splits_path = os.path.join(output_dir, "splits.csv")
        with open(splits_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["split", "file_name", "class_id", "class_name"])
            for split_name, items in split_items.items():
                for file_name, class_id in items:
                    writer.writerow([split_name, file_name, class_id, classes[class_id - 1]])

    fixed_size = tuple(args.fixed_size) if args.fixed_size else None
    export_offset = -1 if args.rfdetr_zero_index else 0
    for split_name in ("train", "val", "test"):
        coco = build_coco(
            split_items[split_name],
            images_root,
            classes,
            fixed_size,
            args.skip_bad,
            desc=f"Building {split_name}",
        )
        ann_path = os.path.join(annotations_dir, f"{split_name}.json")
        with open(ann_path, "w") as handle:
            json.dump(coco, handle, indent=2)
        if args.export_rfdetr_dir:
            export_rfdetr_split(
                args.export_rfdetr_dir,
                split_name,
                coco,
                images_root,
                args.link_mode,
                args.overwrite,
                export_offset,
            )

    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w") as handle:
        for name in classes:
            handle.write(name + "\n")

    stats_path = os.path.join(output_dir, "stats.json")
    totals = {
        split: len(split_items[split]) for split in split_items
    }
    with open(stats_path, "w") as handle:
        json.dump(
            {
                "classes": classes,
                "class_to_id": class_to_id,
                "split_counts": totals,
                "split_counts_by_class": split_counts_by_class,
            },
            handle,
            indent=2,
        )

    print("Done.")
    print(f"Images root: {images_root}")
    print(f"Output dir: {output_dir}")
    print("Split counts:", totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
