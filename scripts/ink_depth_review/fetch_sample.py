"""Download a bounded official surface-volume sample; retain HTTP and hash provenance."""

import argparse
import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import requests
import tifffile

BASE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/phercparis4/w00_20231016151002/"
SEG = "w00_20231016151002"
MISSING = set()
LOCK = threading.Lock()


def fetch(root, key, missing_ok=False):
    path = root / key
    if not path.exists():
        response = requests.get(BASE + key, timeout=90)
        if response.status_code == 404 and missing_ok:
            with LOCK:
                MISSING.add(key)
            return None
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_bytes(response.content)
        os.replace(temporary, path)
    return path.read_bytes()


def chunk(root, name, level, y, x):
    prefix = f"{name}/{level}/"
    meta = json.loads(fetch(root, prefix + ".zarray"))
    sep = meta.get("dimension_separator", ".")
    data = fetch(
        root,
        prefix + sep.join(map(str, (0, y, x))),
        missing_ok=("inklabels" in name or "supervision_mask" in name),
    )
    if data is None:
        return np.full(meta["chunks"], meta["fill_value"], dtype=meta["dtype"])
    return np.frombuffer(
        numcodecs.get_codec(meta["compressor"]).decode(data), dtype=meta["dtype"]
    ).reshape(meta["chunks"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "downloads"
    # Fixed central coarse tile, used only to locate existing labels, never to score ink.
    coarse = chunk(cache, SEG + "_inklabels.zarr", 5, 3, 6)
    counts = np.count_nonzero(coarse, axis=0)
    yy, xx = np.nonzero(counts)
    if not len(yy):
        raise RuntimeError(
            "Chosen coarse tile has no labels; choose a documented alternative."
        )
    mid = len(yy) // 2
    cy, cx = (3 * 128 + int(yy[mid])) * 32 // 128, (6 * 128 + int(xx[mid])) * 32 // 128
    tasks = [
        (name, y, x)
        for name in (
            SEG + ".zarr",
            SEG + "_inklabels.zarr",
            SEG + "_supervision_mask.zarr",
        )
        for y in range(cy, cy + 2)
        for x in range(cx, cx + 2)
    ]

    def one(t):
        return t, chunk(cache, t[0], 0, t[1], t[2])

    arrays = {name: np.zeros((65, 256, 256), np.uint8) for name, _, _ in tasks}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (name, y, x), arr in pool.map(one, tasks):
            arrays[name][
                :,
                (y - cy) * 128 : (y - cy + 1) * 128,
                (x - cx) * 128 : (x - cx + 1) * 128,
            ] = arr
    for name, arr in arrays.items():
        label = (
            "image"
            if name == SEG + ".zarr"
            else ("prior" if "inklabels" in name else "supervision")
        )
        tifffile.imwrite(
            root / f"{label}.tif",
            arr,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
    manifest = {
        "source": BASE,
        "segment": SEG,
        "coordinate_system": "surface-volume depth,y,x; NOT native scroll ZYX",
        "pyramid_level": 0,
        "origin_dyx": [0, cy * 128, cx * 128],
        "shape": [65, 256, 256],
        "selection": "median nonzero coordinate of coarse ink label tile level 5, y=3 x=6; then 2x2 level-0 chunks",
        "missing_label_chunks_404_fill_zero": sorted(MISSING),
        "files": [],
    }
    for file in sorted(cache.rglob("*")):
        if file.is_file():
            manifest["files"].append(
                {
                    "url": BASE + file.relative_to(cache).as_posix(),
                    "bytes": file.stat().st_size,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    for name, a in arrays.items():
        print(
            name,
            "nonzero",
            np.count_nonzero(a),
            "active depth",
            np.where(np.any(a, axis=(1, 2)))[0].tolist(),
        )


if __name__ == "__main__":
    main()
