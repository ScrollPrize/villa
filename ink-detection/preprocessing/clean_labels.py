#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "0")
from pathlib import Path
import shutil
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import tifffile
from tqdm.auto import tqdm


TARGET_SUFFIXES = (
    "_supervision_mask.tif",
    "_supervision_mask.tiff",
    "_supervision_mask.png",
    "_inklabels.tif",
    "_inklabels.tiff",
    "_inklabels.png",
)
SKIP_DIR_NAMES = {".git", "__pycache__"}
DEFAULT_MIN_COMPONENT_SIZE = 1
DEFAULT_TILE_SHAPE = (256, 256)
SUFFIX_PRIORITY = {
    ".png": 0,
    ".tif": 1,
    ".tiff": 2,
}

Image.MAX_IMAGE_PIXELS = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize label and supervision-mask images, remove small connected "
            "components, optionally fill holes, and write tiled LZW-compressed TIFFs."
        )
    )
    parser.add_argument("root", type=Path, help="Root folder to scan recursively.")
    parser.add_argument(
        "--target-folder",
        type=Path,
        default=None,
        help=(
            "Optional specific folder to process instead of scanning the whole root. "
            "Can be absolute or relative to root."
        ),
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=DEFAULT_MIN_COMPONENT_SIZE,
        help="Remove connected components smaller than this many pixels. Default: 1.",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill all enclosed 2D holes after connected-component filtering.",
    )
    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=0,
        help=(
            "Fill only enclosed holes up to this many pixels. "
            "Set to 0 to disable selective hole filling."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes to use. Defaults to min(CPU count, number of files, 8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output TIFF if present.",
    )
    return parser.parse_args(argv)


def is_target_image(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(TARGET_SUFFIXES)


def find_target_images(root: Path) -> List[Path]:
    matches: List[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if dirname not in SKIP_DIR_NAMES and not dirname.lower().endswith(".zarr")
        )
        per_stem: Dict[str, Path] = {}
        for filename in sorted(filenames):
            candidate = Path(current_root) / filename
            if is_target_image(candidate):
                stem_key = candidate.stem.lower()
                existing = per_stem.get(stem_key)
                if existing is None:
                    per_stem[stem_key] = candidate
                    continue

                existing_rank = SUFFIX_PRIORITY.get(existing.suffix.lower(), 99)
                candidate_rank = SUFFIX_PRIORITY.get(candidate.suffix.lower(), 99)
                if candidate_rank < existing_rank:
                    per_stem[stem_key] = candidate
        matches.extend(per_stem[key] for key in sorted(per_stem))
    return matches


def _read_tiff_first_page(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) == 0:
            raise RuntimeError(f"No TIFF pages found in {path}")
        return tif.pages[0].asarray()


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image = _read_tiff_first_page(path)
    else:
        with Image.open(path) as pil_image:
            image = np.asarray(pil_image)

    if image is None:
        raise RuntimeError(f"Failed to read image data from {path}")
    return np.asarray(image)


def normalize_mask_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim == 0:
        raise ValueError("Expected an image with at least one dimension")

    if image.ndim == 3:
        channels = image.shape[-1]
        if channels in {2, 4}:
            image = image[..., :-1]
        image = np.max(image, axis=-1)
    elif image.ndim > 3:
        raise ValueError(f"Expected a 2D or channel-last image, got shape={tuple(image.shape)}")

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image after normalization, got shape={tuple(image.shape)}")

    binary = image != 0
    return np.ascontiguousarray(binary.astype(np.uint8) * 255)


def remove_small_components(mask_u8: np.ndarray, min_component_size: int) -> np.ndarray:
    if min_component_size <= 1:
        return np.ascontiguousarray(mask_u8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros(num_labels, dtype=bool)
    keep[0] = False
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_component_size)
    filtered = keep[labels]
    return np.ascontiguousarray(filtered.astype(np.uint8) * 255)


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask_u8 != 0)
    return np.ascontiguousarray(filled.astype(np.uint8) * 255)


def fill_small_holes(mask_u8: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0:
        return np.ascontiguousarray(mask_u8)

    background = (mask_u8 == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    if num_labels <= 1:
        return np.ascontiguousarray(mask_u8)

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    touches_border = np.zeros(num_labels, dtype=bool)
    touches_border[border_labels] = True

    fill = np.zeros(num_labels, dtype=bool)
    candidate_labels = np.arange(1, num_labels)
    candidate_areas = stats[1:, cv2.CC_STAT_AREA]
    fill[candidate_labels] = (~touches_border[candidate_labels]) & (candidate_areas <= int(max_hole_area))

    filled = (mask_u8 != 0) | fill[labels]
    return np.ascontiguousarray(filled.astype(np.uint8) * 255)


def clean_mask_image(
    image: np.ndarray,
    *,
    min_component_size: int,
    do_fill_holes: bool,
    max_hole_area: int,
) -> np.ndarray:
    cleaned = normalize_mask_image(image)
    cleaned = remove_small_components(cleaned, min_component_size=min_component_size)
    if max_hole_area > 0:
        cleaned = fill_small_holes(cleaned, max_hole_area=max_hole_area)
    elif do_fill_holes:
        cleaned = fill_holes(cleaned)
    return cleaned


def output_path_for_input(input_path: Path) -> Path:
    suffix = input_path.suffix.lower()
    if suffix == ".png":
        return input_path.with_suffix(".tif")
    if suffix in {".tif", ".tiff"}:
        return input_path
    raise ValueError(f"Unsupported input suffix for {input_path}")


def backup_path_for_input(root: Path, input_path: Path) -> Path:
    relative_path = input_path.relative_to(root)
    return root.parent / "label_backup" / relative_path


def write_tiff(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output_path,
        image,
        compression="lzw",
        tile=DEFAULT_TILE_SHAPE,
        photometric="minisblack",
        metadata=None,
        bigtiff=bool(image.nbytes >= (4 * 1024**3)),
    )


def process_one(
    root: Path,
    input_path: Path,
    *,
    min_component_size: int,
    do_fill_holes: bool,
    max_hole_area: int,
    overwrite: bool,
) -> Dict[str, str]:
    output_path = output_path_for_input(input_path)
    backup_path = backup_path_for_input(root, input_path)
    output_conflict = output_path != input_path and output_path.exists()

    if not overwrite:
        if backup_path.exists() and (output_conflict or output_path == input_path):
            return {
                "status": "skipped",
                "input": str(input_path),
                "output": str(output_path),
                "backup": str(backup_path),
            }
        if backup_path.exists() or output_conflict:
            raise FileExistsError(
                "Refusing to continue with partial prior output: "
                f"backup_exists={backup_path.exists()} output_exists={output_conflict} "
                f"for input {input_path}"
            )

    image = load_image(input_path)
    cleaned = clean_mask_image(
        image,
        min_component_size=min_component_size,
        do_fill_holes=do_fill_holes,
        max_hole_area=max_hole_area,
    )

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and backup_path.exists():
        backup_path.unlink()
    shutil.move(str(input_path), str(backup_path))

    if overwrite and output_path != input_path and output_path.exists():
        output_path.unlink()
    write_tiff(output_path, cleaned)
    return {
        "status": "written",
        "input": str(input_path),
        "output": str(output_path),
        "backup": str(backup_path),
    }


def _process_worker(
    root: str,
    input_path: str,
    min_component_size: int,
    do_fill_holes: bool,
    max_hole_area: int,
    overwrite: bool,
) -> Dict[str, str]:
    return process_one(
        Path(root),
        Path(input_path),
        min_component_size=min_component_size,
        do_fill_holes=do_fill_holes,
        max_hole_area=max_hole_area,
        overwrite=overwrite,
    )


def run_processing(
    root: Path,
    image_paths: Sequence[Path],
    *,
    workers: int,
    min_component_size: int,
    do_fill_holes: bool,
    max_hole_area: int,
    overwrite: bool,
) -> Dict[str, List[Dict[str, str]]]:
    results: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    if workers <= 1:
        iterator = tqdm(image_paths, total=len(image_paths), desc="Cleaning", unit="file")
        for image_path in iterator:
            try:
                results.append(
                    process_one(
                        root,
                        image_path,
                        min_component_size=min_component_size,
                        do_fill_holes=do_fill_holes,
                        max_hole_area=max_hole_area,
                        overwrite=overwrite,
                    )
                )
            except Exception as exc:
                errors.append({"input": str(image_path), "error": str(exc)})
        return {"results": results, "errors": errors}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _process_worker,
                str(root),
                str(image_path),
                min_component_size,
                do_fill_holes,
                max_hole_area,
                overwrite,
            ): image_path
            for image_path in image_paths
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_map),
            total=len(future_map),
            desc="Cleaning",
            unit="file",
        ):
            image_path = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"input": str(image_path), "error": str(exc)})
    return {"results": results, "errors": errors}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")
    if args.min_component_size < 1:
        raise ValueError("--min-component-size must be at least 1")
    if args.max_hole_area < 0:
        raise ValueError("--max-hole-area must be at least 0")

    scan_root = root
    if args.target_folder is not None:
        candidate = args.target_folder.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        scan_root = candidate.resolve()
        if not scan_root.exists():
            raise FileNotFoundError(f"Target folder does not exist: {scan_root}")
        if not scan_root.is_dir():
            raise NotADirectoryError(f"Target folder is not a directory: {scan_root}")
        try:
            scan_root.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Target folder must be inside the root folder: root={root} target={scan_root}"
            ) from exc

    image_paths = find_target_images(scan_root)
    if not image_paths:
        print(f"No matching images found under {scan_root}")
        return 0

    max_workers = args.workers
    if max_workers is None:
        max_workers = min(len(image_paths), os.cpu_count() or 1, 8)
    if max_workers < 1:
        raise ValueError("--workers must be at least 1")

    outcome = run_processing(
        root,
        image_paths,
        workers=max_workers,
        min_component_size=args.min_component_size,
        do_fill_holes=args.fill_holes,
        max_hole_area=args.max_hole_area,
        overwrite=args.overwrite,
    )
    results = outcome["results"]
    errors = outcome["errors"]
    written = sum(1 for result in results if result["status"] == "written")
    skipped = sum(1 for result in results if result["status"] == "skipped")

    print(
        f"Processed {len(image_paths)} image(s): "
        f"{written} written, {skipped} skipped, {len(errors)} failed."
    )
    if errors:
        for error in errors:
            print(f"ERROR {error['input']}: {error['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
