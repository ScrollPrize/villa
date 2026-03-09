#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a chunk-aligned level-0 OME-Zarr subset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop", nargs=6, type=int, required=True, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    parser.add_argument("--level", type=int, default=0)
    return parser.parse_args()


def chunk_path(root: Path, delimiter: str, cz: int, cy: int, cx: int) -> Path:
    if delimiter == "/":
        return root / str(cz) / str(cy) / str(cx)
    if delimiter == ".":
        return root / f"{cz}.{cy}.{cx}"
    raise RuntimeError(f"Unsupported dimension separator: {delimiter}")


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    x0, y0, z0, x1, y1, z1 = args.crop
    level_name = str(args.level)
    input_level = input_root / level_name
    output_level = output_root / "0"

    if output_root.exists():
        shutil.rmtree(output_root)

    zarray = json.loads((input_level / ".zarray").read_text())
    shape = zarray["shape"]
    chunks = zarray["chunks"]
    delimiter = zarray.get("dimension_separator", ".")

    crop_zyx = [z0, y0, x0, z1, y1, x1]
    for dim in range(3):
        start = crop_zyx[dim]
        end = crop_zyx[dim + 3]
        if start < 0 or end <= start or end > shape[dim]:
            raise RuntimeError("crop exceeds source shape")
        if (start % chunks[dim]) != 0 or (end % chunks[dim]) != 0:
            raise RuntimeError("crop must be aligned to source chunk boundaries")

    output_level.mkdir(parents=True, exist_ok=True)
    (output_root / ".zgroup").write_text("{\"zarr_format\": 2}\n")
    (output_level / ".zarray").write_text(json.dumps({
        **zarray,
        "shape": [z1 - z0, y1 - y0, x1 - x0],
    }, indent=2) + "\n")

    src_chunk_start = [z0 // chunks[0], y0 // chunks[1], x0 // chunks[2]]
    src_chunk_end = [(z1 // chunks[0]) - 1, (y1 // chunks[1]) - 1, (x1 // chunks[2]) - 1]

    for src_cz in range(src_chunk_start[0], src_chunk_end[0] + 1):
        for src_cy in range(src_chunk_start[1], src_chunk_end[1] + 1):
            for src_cx in range(src_chunk_start[2], src_chunk_end[2] + 1):
                dst_cz = src_cz - src_chunk_start[0]
                dst_cy = src_cy - src_chunk_start[1]
                dst_cx = src_cx - src_chunk_start[2]
                src_path = chunk_path(input_level, delimiter, src_cz, src_cy, src_cx)
                dst_path = chunk_path(output_level, delimiter, dst_cz, dst_cy, dst_cx)
                if not src_path.exists():
                    continue
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

    attrs_path = input_root / ".zattrs"
    if attrs_path.exists():
        attrs = json.loads(attrs_path.read_text())
        if "multiscales" in attrs and attrs["multiscales"]:
            ms = attrs["multiscales"][0]
            datasets = ms.get("datasets", [])
            level_dataset = None
            for ds in datasets:
                if ds.get("path") == level_name:
                    level_dataset = ds
                    break
            if level_dataset is None and datasets:
                level_dataset = datasets[0]
            if level_dataset is not None:
                ms["datasets"] = [{**level_dataset, "path": "0"}]
            attrs["multiscales"] = [ms]
        attrs["subset_origin_zyx"] = [z0, y0, x0]
        attrs["subset_shape_zyx"] = [z1 - z0, y1 - y0, x1 - x0]
        attrs["source_level"] = args.level
        (output_root / ".zattrs").write_text(json.dumps(attrs, indent=2) + "\n")

    meta_path = input_root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["slices"] = z1 - z0
        meta["height"] = y1 - y0
        meta["width"] = x1 - x0
        meta["subset_origin_zyx"] = [z0, y0, x0]
        meta["source_level"] = args.level
        (output_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
