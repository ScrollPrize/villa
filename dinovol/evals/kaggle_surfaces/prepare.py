"""Convert the supplied TIFF train/validation split to verified Zarr v2 arrays."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import tifffile
from tqdm import tqdm
from vc_reader.reader import open_zarr

from dinovol_2.eval.embed_volume import ZarrGroupWriter

FORMAT = "dinovol-kaggle-surfaces/1"


def pairs(source):
    result = {}
    for split, directory in (("train", source), ("val", source / "val")):
        images = {p.stem: p for p in sorted((directory / "images").glob("*.tif"))}
        labels = {p.stem.removesuffix("_surface"): p
                  for p in sorted((directory / "labels").glob("*.tif"))}
        if not images or images.keys() != labels.keys():
            raise ValueError(f"Missing or unmatched images/labels in {directory}")
        result[split] = [(name, image, labels[name]) for name, image in images.items()]
    overlap = {r[0] for r in result["train"]} & {r[0] for r in result["val"]}
    if overlap:
        raise ValueError(f"Train/validation overlap: {sorted(overlap)}")
    return result


def fingerprint(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_array(path, values, chunk, fill):
    writer = ZarrGroupWriter(path.parent, attrs=None, workers=1)
    try:
        writer.create(path.name, values.shape, (chunk,) * 3, values.dtype,
                      attrs={"_ARRAY_DIMENSIONS": ["z", "y", "x"]}, fill_value=fill)
        writer.write(path.name, (0, 0, 0), values)
    finally:
        writer.close()
    # Read every voxel through the actual reader, including all edge chunks.
    with open_zarr(str(path), threads=1, memory_cache=0) as reader:
        for start in range(0, values.shape[0], chunk):
            selection = (slice(start, start + chunk), slice(None), slice(None))
            np.testing.assert_array_equal(reader[selection], values[selection])


def convert_case(job):
    split, name, image_path, label_path, output, chunk = job
    destination = output / split / f"{name}.zarr"
    provenance = {"image": fingerprint(image_path), "label": fingerprint(label_path), "chunk": chunk}
    complete = destination / "case.json"
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["source"] != provenance:
            raise ValueError(f"Source or chunk size changed for {destination}; use a new output directory")
        return record
    if destination.exists():
        raise FileExistsError(f"Incomplete destination {destination}; inspect/remove it before retrying")
    temporary = destination.with_suffix(".zarr.partial")
    # Only remove our own uncommitted conversion from a previous interruption.
    if temporary.exists():
        shutil.rmtree(temporary)
    image = tifffile.imread(image_path, maxworkers=1)
    label = tifffile.imread(label_path, maxworkers=1)
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError(f"Expected paired ZYX volumes: {image_path}, {image.shape}, {label.shape}")
    if image.dtype != np.uint8 or not np.isin(label, [0, 1, 2]).all():
        raise ValueError(f"CUDA reading requires uint8 images and labels in [0, 1, 2]: {name}")
    ZarrGroupWriter(temporary, attrs={"axes": ["z", "y", "x"]}, workers=1).close()
    write_array(temporary / "image", image, chunk, 0)
    write_array(temporary / "label", label.astype(np.uint8), chunk, 2)
    # Coarse occupancy supports foreground/valid crop sampling without loading labels.
    foreground, valid = [], []
    for start in itertools.product(*(range(0, n, chunk) for n in label.shape)):
        end = tuple(min(s + chunk, n) for s, n in zip(start, label.shape))
        block = label[tuple(slice(s, e) for s, e in zip(start, end))]
        bounds = [list(start), list(end)]
        if np.any(block == 1):
            foreground.append(bounds)
        if np.any(block != 2):
            valid.append(bounds)
    if not valid:
        raise ValueError(f"Entire case {name} is ignored")
    record = {"id": name, "split": split, "path": str(destination.relative_to(output)),
              "shape": list(image.shape), "source": provenance,
              "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "label_sha256": hashlib.sha256(label.tobytes()).hexdigest(),
              "foreground_blocks": foreground, "valid_blocks": valid}
    (temporary / "case.json").write_text(json.dumps(record, indent=2) + "\n")
    temporary.rename(destination)
    return record


def prepare(source, output, *, chunk=64, workers=4, limit=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if chunk <= 0 or workers <= 0 or (limit is not None and limit <= 0):
        raise ValueError("chunk, workers and limit must be positive")
    split_pairs = pairs(source)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(split, *case, output, chunk) for split, cases in split_pairs.items()
            for case in (cases[:limit] if limit else cases)]
    for split in split_pairs:
        (output / split).mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(convert_case, jobs), total=len(jobs), desc="TIFF → verified Zarr"))
    manifest = {"format": FORMAT, "source": str(source), "limited": limit is not None,
                "labels": {"background": 0, "surface": 1, "ignore": 2},
                "cases": records}
    temporary = output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/raid_nvme/datasets/kaggle_ves"))
    parser.add_argument("--output", type=Path, default=Path("evals/kaggle_surfaces/data"))
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Smoke test: at most this many cases per split")
    args = parser.parse_args()
    manifest = prepare(**vars(args))
    print(f"Prepared {len(manifest['cases'])} cases in {args.output}")


if __name__ == "__main__":
    main()
