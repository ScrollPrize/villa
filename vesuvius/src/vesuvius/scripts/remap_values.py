import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


SUFFIX_RE = re.compile(r"(.+?)(?:[_-]\d+)?$")


def stem_base(stem: str) -> str:
    """Return a base key by stripping a trailing _#### or -#### style suffix."""
    m = SUFFIX_RE.match(stem)
    return m.group(1) if m else stem


def discover_tifs(folder: Path):
    return sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))


def pair_files(images_dir: Path, labels_dir: Path):
    """Pair image and label files by filename.

    Strategy:
    - Prefer exact stem matches (e.g., foo_0000.tif -> foo_0000.tif).
    - Fallback: match by base stem after stripping trailing numeric suffix
      (e.g., foo_0000.tif -> foo.tif) if unique.
    """
    image_files = discover_tifs(images_dir)
    label_files = discover_tifs(labels_dir)

    label_by_stem = {p.stem: p for p in label_files}

    # Build base maps and detect duplicates that would cause ambiguity
    labels_by_base = {}
    for p in label_files:
        base = stem_base(p.stem)
        labels_by_base.setdefault(base, []).append(p)

    pairs = []
    missing = []
    ambiguous = []

    for img in image_files:
        # Exact match first
        lbl = label_by_stem.get(img.stem)
        if lbl is not None:
            pairs.append((img, lbl))
            continue

        # Fallback to base match
        base = stem_base(img.stem)
        candidates = labels_by_base.get(base, [])
        if len(candidates) == 1:
            pairs.append((img, candidates[0]))
        elif len(candidates) == 0:
            missing.append(img)
        else:
            ambiguous.append((img, candidates))

    return pairs, missing, ambiguous


def safe_overwrite_tif(path: Path, array: np.ndarray, compression: str = "lzw"):
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tifffile.imwrite(tmp, array, compression=compression)
    os.replace(tmp, path)


def worker(args):
    img_path, lbl_path, mask_value = args

    try:
        img = tifffile.imread(img_path)
        lbl = tifffile.imread(lbl_path)

        if img.shape != lbl.shape:
            return f"Mismatch: {img_path.name} vs {lbl_path.name} shapes {img.shape} != {lbl.shape}"

        # Build mask where label equals the specified value
        mask = (lbl == mask_value)

        if not np.any(mask):
            return f"No-op: {img_path.name} (no voxels == {mask_value})"

        # Zero image where mask true, preserving dtype
        img = img.copy()
        img[mask] = 0

        # Set those label voxels to 0 as well, preserving dtype
        lbl = lbl.copy()
        lbl[mask] = 0

        # Overwrite files in place via safe temp write
        safe_overwrite_tif(img_path, img)
        safe_overwrite_tif(lbl_path, lbl)

        return f"Updated: {img_path.name} | {lbl_path.name}"
    except Exception as e:
        return f"Error: {img_path.name} | {lbl_path.name} -> {e}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pair 3D TIFF images and labels by filename, and wherever the label equals "
            "the specified value, set both image and label to 0 (in place)."
        )
    )
    parser.add_argument("images_dir", type=str, help="Directory containing image TIFFs")
    parser.add_argument("labels_dir", type=str, help="Directory containing label TIFFs")
    parser.add_argument("--mask-value", type=int, required=True, help="Label value to zero out")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Error: images_dir does not exist or is not a directory: {images_dir}")
        return
    if not labels_dir.exists() or not labels_dir.is_dir():
        print(f"Error: labels_dir does not exist or is not a directory: {labels_dir}")
        return

    pairs, missing, ambiguous = pair_files(images_dir, labels_dir)

    if not pairs:
        print("No image/label pairs found."
              f" Images: {len(discover_tifs(images_dir))}, Labels: {len(discover_tifs(labels_dir))}")
        return

    if missing:
        print(f"Warning: {len(missing)} images had no matching label (showing up to 10):")
        for p in missing[:10]:
            print(f"  - {p.name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if ambiguous:
        print(f"Warning: {len(ambiguous)} ambiguous matches (showing up to 5):")
        for img, cands in ambiguous[:5]:
            cand_names = ", ".join(c.name for c in cands[:5])
            if len(cands) > 5:
                cand_names += ", ..."
            print(f"  - {img.name} -> [{cand_names}]")

    tasks = [(img, lbl, args["mask_value"]) for img, lbl in pairs] if isinstance(args, dict) else [
        (img, lbl, args.mask_value) for img, lbl in pairs
    ]

    print(f"Found {len(pairs)} pairs. Using {args.workers} workers. Mask value: {args.mask_value}")
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks), desc="Masking"))

    # Summary
    updated = sum(1 for r in results if r.startswith("Updated"))
    noop = sum(1 for r in results if r.startswith("No-op"))
    mism = sum(1 for r in results if r.startswith("Mismatch"))
    errs = sum(1 for r in results if r.startswith("Error"))

    print("\n=== Summary ===")
    print(f"Updated: {updated}")
    print(f"No-op: {noop}")
    print(f"Shape mismatches: {mism}")
    print(f"Errors: {errs}")


if __name__ == "__main__":
    main()

