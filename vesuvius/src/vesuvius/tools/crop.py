"""Copy a region of a volume, local or remote, into a small local OME-Zarr that other tools can open.

    vesuvius.crop https://.../volume.zarr crop.zarr --crop 10400:10656,2900:3156,2300:2556
    vesuvius.crop scroll:1 s1.zarr --crop 5000:5256,3900:4156,4000:4256 --levels 3

The output keeps the source's voxel size, records where the crop came from (a translation
in physical units, plus the source and voxel ranges under "vesuvius_crop"), and gets a
2x mean pyramid of --levels levels. The copy goes one slab of chunks at a time, so the
crop never has to fit in memory, and an interrupted copy picks up where it stopped when
run again with --resume.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from vesuvius.data.utils import create_zarr_array, open_zarr_group
from vesuvius.tools.preview import open_source, parse_crop, read_block, resolve

PROGRESS_KEY = "vesuvius_crop"


def source_scale(spec: str, ndim: int = 3) -> Tuple[List[float], List[Optional[str]]]:
    """Level 0 voxel size and units from the source's multiscales metadata, if it has them."""
    from vesuvius.data.utils import open_zarr

    try:
        store = open_zarr(spec, mode="r", storage_options={"anon": True} if spec.startswith("s3://") else None)
        ms = (store.attrs.get("multiscales") or [None])[0] if not hasattr(store, "shape") else None
    except (KeyError, TypeError, ValueError, OSError):
        ms = None
    scale, units = [1.0] * ndim, [None] * ndim
    if ms:
        for t in ms["datasets"][0].get("coordinateTransformations", []):
            if t.get("type") == "scale":
                scale = [float(v) for v in t["scale"]][-ndim:]
        axes = ms.get("axes", [])
        if len(axes) >= ndim:
            units = [a.get("unit") if isinstance(a, dict) else None for a in axes][-ndim:]
    return scale, units


def downsample2(block: np.ndarray) -> np.ndarray:
    """2x mean in every axis; odd edges keep their last plane."""
    pads = [(0, n % 2) for n in block.shape]
    if any(p[1] for p in pads):
        block = np.pad(block, pads, mode="edge")
    z, y, x = block.shape
    out = block.reshape(z // 2, 2, y // 2, 2, x // 2, 2).astype(np.float32).mean(axis=(1, 3, 5))
    if np.issubdtype(block.dtype, np.integer):
        out = np.rint(out)
    return out.astype(block.dtype)


def copy(args) -> str:
    src = open_source(args.source)
    crop = parse_crop(args.crop, src.shape)
    level0 = src.level(0)
    shape = tuple(hi - lo for lo, hi in crop)
    chunks = tuple(min(c, n) for c, n in zip(args.chunks, shape))
    scale, units = source_scale(resolve(args.source))

    root = open_zarr_group(args.output, mode="a" if args.resume else "w")
    meta = root.attrs.get(PROGRESS_KEY, {}) if args.resume else {}
    if meta and (meta.get("crop") != [list(c) for c in crop] or meta.get("source") != args.source):
        raise SystemExit(f"{args.output} holds a crop of {meta.get('source')} {meta.get('crop')}; "
                         "--resume continues only the same crop")
    done_z = int(meta.get("level0_done_z", 0))
    if "0" not in root:
        create_zarr_array(root, "0", shape=shape, chunks=chunks, dtype=level0.dtype,
                          write_empty_chunks=False)
    out0 = root["0"]
    root.attrs[PROGRESS_KEY] = {"source": args.source, "crop": [list(c) for c in crop],
                                "level0_done_z": done_z, "complete": False}

    started = time.time()
    total = shape[0]
    for z in range(done_z, total, chunks[0]):
        z1 = min(z + chunks[0], total)
        box = [(crop[0][0] + z, crop[0][0] + z1), crop[1], crop[2]]
        out0[z:z1] = read_block(level0, box)
        meta = dict(root.attrs[PROGRESS_KEY])
        meta["level0_done_z"] = z1
        root.attrs[PROGRESS_KEY] = meta
        rate = (z1 - done_z) / max(time.time() - started, 1e-6)
        eta = (total - z1) / rate if rate > 0 else 0
        print(f"level 0: z {z1}/{total}  ({rate:.1f} slices/s, about {eta:.0f} s left)", flush=True)

    prev = out0
    for level in range(1, args.levels):
        if min(prev.shape) < 2:
            break
        lshape = tuple((n + 1) // 2 for n in prev.shape)
        if str(level) in root:
            del root[str(level)]
        out = create_zarr_array(root, str(level), shape=lshape,
                                chunks=tuple(min(c, n) for c, n in zip(chunks, lshape)),
                                dtype=prev.dtype, write_empty_chunks=False)
        step = 2 * chunks[0]
        for z in range(0, prev.shape[0], step):
            block = np.asarray(prev[z:min(z + step, prev.shape[0])])
            small = downsample2(block)
            out[z // 2:z // 2 + small.shape[0]] = small
        print(f"level {level}: {lshape}", flush=True)
        prev = out

    n_levels = len([k for k in root.array_keys() if k.isdigit()])
    offset = [lo * s for (lo, _), s in zip(crop, scale)]
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": f"crop of {args.source}",
        "axes": [{"name": a, "type": "space", **({"unit": u} if u else {})} for a, u in zip("zyx", units)],
        "datasets": [{"path": str(i), "coordinateTransformations": [
            {"type": "scale", "scale": [s * 2 ** i for s in scale]},
            {"type": "translation", "translation": offset},
        ]} for i in range(n_levels)],
    }]
    meta = dict(root.attrs[PROGRESS_KEY])
    meta["complete"] = True
    root.attrs[PROGRESS_KEY] = meta
    return args.output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vesuvius.crop", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("source", help="OME-Zarr or zarr path/URL, or scroll:ID / segment:ID")
    p.add_argument("output", help="local .zarr to write")
    p.add_argument("--crop", required=True, help="Z0:Z1,Y0:Y1,X0:X1 in full-resolution voxels")
    p.add_argument("--levels", type=int, default=3, help="pyramid levels to write (default 3)")
    p.add_argument("--chunks", type=lambda s: tuple(int(v) for v in s.split(",")), default=(64, 128, 128),
                   help="output chunks z,y,x (default 64,128,128)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted copy of the same crop")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = copy(args)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
