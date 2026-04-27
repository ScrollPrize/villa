#!/usr/bin/env python3
"""Create an OME-Zarr from an existing one by symlinking scale groups at an offset.

Each zarr scale level halves the resolution: scale 0 is full-res, scale 1 is 2x
downsampled, scale 2 is 4x, etc.  The offset shifts which input scale becomes
which output scale.

  new_scale = old_scale - offset

Positive offset  →  downsample simulation (drop finest scales)
--------------------------------------------------------------
  Input has scales 0..4 (0 = full-res, 4 = 16x downsampled).
  --offset 2 produces:

    output 0/  →  input 2/   (was 4x,  now treated as full-res)
    output 1/  →  input 3/   (was 8x,  now treated as 2x)
    output 2/  →  input 4/   (was 16x, now treated as 4x)

  Input scales 0,1 (the two finest) are dropped entirely.

Negative offset  →  upsample simulation (pretend finer scales exist)
--------------------------------------------------------------------
  Input has scales 0..2 (0 = full-res, 2 = 4x downsampled).
  --offset -2 produces:

    output 2/  →  input 0/   (full-res data, now labelled as 4x)
    output 3/  →  input 1/   (2x data,       now labelled as 8x)
    output 4/  →  input 2/   (4x data,       now labelled as 16x)

  Output scales 0,1 don't exist — no directory is created for them.
  VC3D auto-infers dimensions from the first physical scale it finds.

Metadata files (meta.json, .zattrs) are copied and adjusted if present.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path


def discover_scales(zarr_path: Path) -> list[int]:
    """Find numbered subdirectories containing .zarray, return sorted."""
    scales = []
    for entry in zarr_path.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            if (entry / ".zarray").exists():
                scales.append(int(entry.name))
    scales.sort()
    return scales


def make_relative_symlink(target: Path, link: Path) -> None:
    """Create a relative symlink from link → target."""
    rel = os.path.relpath(target.resolve(), link.parent.resolve())
    os.symlink(rel, link)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def ceil_div_pow2(v: int, level: int) -> int:
    """Integer ceil division by 2^level, matching VC3D's ceilDivPow2."""
    denom = 1 << level
    return (v + denom - 1) // denom


def adjust_meta(meta: dict, offset: int) -> dict:
    """Adjust meta.json dimensions and voxelsize for the given offset."""
    meta = copy.deepcopy(meta)
    for dim_key in ("width", "height", "slices"):
        if dim_key not in meta:
            continue
        v = meta[dim_key]
        if offset > 0:
            meta[dim_key] = ceil_div_pow2(v, offset)
        else:
            meta[dim_key] = v * (1 << abs(offset))

    if "voxelsize" in meta and meta["voxelsize"]:
        if offset > 0:
            meta["voxelsize"] = meta["voxelsize"] * (1 << offset)
        else:
            meta["voxelsize"] = meta["voxelsize"] / (1 << abs(offset))

    return meta


def adjust_zattrs(zattrs: dict, scale_mapping: dict[int, int]) -> dict:
    """Rebuild .zattrs multiscales with new scale numbering.

    scale_mapping: {new_scale_num: old_scale_num}
    """
    zattrs = copy.deepcopy(zattrs)
    if "multiscales" not in zattrs:
        return zattrs

    for ms in zattrs["multiscales"]:
        if "datasets" not in ms:
            continue
        new_datasets = []
        for new_scale in sorted(scale_mapping.keys()):
            ds_entry = {"path": str(new_scale)}
            scale_factor = float(1 << new_scale)
            ds_entry["coordinateTransformations"] = [
                {"type": "scale", "scale": [scale_factor, scale_factor, scale_factor]}
            ]
            new_datasets.append(ds_entry)
        ms["datasets"] = new_datasets

    return zattrs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an OME-Zarr from an existing one by symlinking scale "
                    "groups at an offset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path,
                        help="Path to source OME-Zarr directory")
    parser.add_argument("output", type=Path,
                        help="Path to output OME-Zarr directory")
    parser.add_argument("-n", "--offset", required=True, type=int,
                        help="Scale offset (new_scale = old_scale - offset). "
                             "Positive: drop the N finest scales and renumber "
                             "(e.g. 2 makes input scale 2 become output scale 0). "
                             "Negative: shift numbering up, leaving gaps at the top "
                             "(e.g. -2 makes input scale 0 become output scale 2).")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite output if it exists")
    args = parser.parse_args(argv)

    src: Path = args.input.resolve()
    dst: Path = args.output.resolve()
    offset: int = args.offset

    # Validate
    if not src.is_dir():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 1
    if offset == 0:
        print("ERROR: offset must be non-zero", file=sys.stderr)
        return 1

    scales = discover_scales(src)
    if not scales:
        print(f"ERROR: no zarr scale groups found in {src}", file=sys.stderr)
        return 1

    if offset > 0:
        available = [s for s in scales if s >= offset]
        if not available:
            print(f"ERROR: offset {offset} drops all scales {scales}", file=sys.stderr)
            return 1

    if dst.exists():
        if args.force:
            shutil.rmtree(dst)
        else:
            print(f"ERROR: output already exists: {dst} (use --force to overwrite)",
                  file=sys.stderr)
            return 1

    # Compute scale mapping: new_num → old_num
    scale_mapping: dict[int, int] = {}
    if offset > 0:
        for old_scale in scales:
            if old_scale >= offset:
                scale_mapping[old_scale - offset] = old_scale
    else:
        abs_off = abs(offset)
        for old_scale in scales:
            scale_mapping[old_scale + abs_off] = old_scale

    # Create output dir
    dst.mkdir(parents=True)

    # Copy .zgroup if exists
    zgroup_path = src / ".zgroup"
    if zgroup_path.exists():
        shutil.copy2(zgroup_path, dst / ".zgroup")

    # Copy/adjust .zattrs if exists
    zattrs_path = src / ".zattrs"
    if zattrs_path.exists():
        zattrs = read_json(zattrs_path)
        zattrs = adjust_zattrs(zattrs, scale_mapping)
        write_json(dst / ".zattrs", zattrs)

    # Copy/adjust meta.json or metadata.json if present
    for meta_name in ("meta.json", "metadata.json"):
        meta_path = src / meta_name
        if meta_path.exists():
            meta = read_json(meta_path)
            adjusted = adjust_meta(meta, offset)
            write_json(dst / meta_name, adjusted)

    # Create symlinks for scale groups
    for new_scale, old_scale in sorted(scale_mapping.items()):
        make_relative_symlink(src / str(old_scale), dst / str(new_scale))

    # Print summary
    print(f"Offset: {offset} ({'downsample' if offset > 0 else 'upsample'} simulation)")
    print(f"Source: {src}")
    print(f"Output: {dst}")
    print(f"Scales: {scales} → {sorted(scale_mapping.keys())}")
    for new_scale, old_scale in sorted(scale_mapping.items()):
        print(f"  {new_scale}/ → input {old_scale}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
