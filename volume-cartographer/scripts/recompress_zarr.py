#!/usr/bin/env python3
"""Recompress a local zarr array/tree with the VCZ1 rANS codec.

This is intentionally a small zarr-python example:

    python scripts/recompress_zarr.py \
        /home/sean/Desktop/paris4_level5only.zarr \
        /home/sean/Desktop/paris4_level5only.vcz1.zarr

The output is zarr v2 metadata with compressor id "vcz1".  The chunks are
self-describing VCZ1 payloads written by vc.compression.vcz1, using rANS by
default.
zarr-python 2.x and 3.x both use the numcodecs registration below for v2
arrays; under zarr-python 3.x the script passes zarr_format=2 explicitly.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import numcodecs
    import zarr
except ImportError as e:  # pragma: no cover
    raise SystemExit("install zarr and numcodecs: python -m pip install zarr") from e

try:
    from vc.compression import vcz1
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "install the volume-cartographer Python bindings so "
        "`from vc.compression import vcz1` works"
    ) from e


CODEC_ID = "vcz1"


class Vcz1(numcodecs.abc.Codec):
    """numcodecs wrapper for the C++ VCZ1 chunk codec."""

    codec_id = CODEC_ID

    def __init__(self, codec: str = "rans", quant: int = 1):
        if codec not in {"rans", "zstd"}:
            raise ValueError("codec must be 'rans' or 'zstd'")
        if not 1 <= int(quant) <= 255:
            raise ValueError("quant must be in [1, 255]")
        self.codec = codec
        self.quant = int(quant)

    def encode(self, buf):
        a = np.ascontiguousarray(buf)
        if a.ndim != 3:
            raise ValueError("vcz1 expects 3D chunks")
        if a.dtype not in (np.uint8, np.uint16):
            raise ValueError("vcz1 supports uint8 and uint16 chunks")
        z, y, x = a.shape
        return vcz1.compress(
            a.tobytes(), z, y, x, a.dtype.itemsize, self.quant, self.codec
        )

    def decode(self, buf, out=None):
        payload = bytes(memoryview(buf))
        z, y, x = _vcz1_shape(payload)
        elem_size = payload[5]
        raw = vcz1.decompress(payload, z * y * x * elem_size)
        if out is not None:
            np.frombuffer(out, dtype=np.uint8)[:] = np.frombuffer(raw, dtype=np.uint8)
            return out
        return raw

    def get_config(self):
        return {"id": self.codec_id, "codec": self.codec, "quant": self.quant}


numcodecs.register_codec(Vcz1)


def _vcz1_shape(payload: bytes) -> tuple[int, int, int]:
    if len(payload) < 20 or payload[:4] != b"VCZ1":
        raise ValueError("not a VCZ1 payload")
    return (
        int.from_bytes(payload[8:12], "little"),
        int.from_bytes(payload[12:16], "little"),
        int.from_bytes(payload[16:20], "little"),
    )


def zarr_major() -> int:
    return int(zarr.__version__.split(".", 1)[0])


def create_array(path: Path, src, *, codec: Vcz1):
    kwargs = dict(
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
        compressor=codec,
        fill_value=getattr(src, "fill_value", None),
        overwrite=True,
    )
    if zarr_major() >= 3:
        kwargs["zarr_format"] = 2
    return zarr.create(store=str(path), **kwargs)


def chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...]) -> Iterable[tuple[slice, ...]]:
    grid = [range(math.ceil(s / c)) for s, c in zip(shape, chunks)]
    for idx in np.ndindex(*(len(g) for g in grid)):
        yield tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunks, shape)
        )


def array_dirs(root: Path) -> list[Path]:
    if (root / ".zarray").exists():
        return [root]
    return sorted(p.parent for p in root.rglob(".zarray"))


def copy_group_metadata(src_root: Path, dst_root: Path, arrays: list[Path]) -> None:
    array_meta = {".zarray"}
    for path in src_root.rglob("*"):
        if not path.is_file() or path.name not in {".zattrs", ".zgroup", ".zmetadata"}:
            continue
        if path.name in array_meta:
            continue
        rel = path.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def recompress_array(src_path: Path, dst_path: Path, codec: Vcz1) -> None:
    src = zarr.open(str(src_path), mode="r")
    if len(src.shape) != 3:
        raise SystemExit(f"{src_path}: expected a 3D array, got shape {src.shape}")
    if np.dtype(src.dtype) not in (np.dtype("uint8"), np.dtype("uint16")):
        raise SystemExit(f"{src_path}: expected uint8 or uint16, got {src.dtype}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = create_array(dst_path, src, codec=codec)
    try:
        dst.attrs.update(dict(src.attrs))
    except Exception:
        pass

    print(f"{src_path} -> {dst_path}")
    print(f"  shape={src.shape} chunks={src.chunks} dtype={src.dtype} codec={codec.get_config()}")
    for n, slc in enumerate(chunk_slices(tuple(src.shape), tuple(src.chunks)), 1):
        dst[slc] = src[slc]
        if n % 100 == 0:
            print(f"  {n} chunks", flush=True)
    print(f"  done: {n} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input zarr array or group")
    parser.add_argument("output", type=Path, help="output zarr array or group")
    parser.add_argument("--codec", choices=("rans", "zstd"), default="rans")
    parser.add_argument(
        "--quant",
        type=int,
        default=1,
        help="VCZ1 quantization bin width; 1 is lossless",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete the output path first if it already exists",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input}: does not exist")
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output}: already exists; pass --overwrite")
        shutil.rmtree(args.output)

    arrays = array_dirs(args.input)
    if not arrays:
        raise SystemExit(f"{args.input}: no .zarray found")

    codec = Vcz1(codec=args.codec, quant=args.quant)
    if arrays == [args.input]:
        recompress_array(args.input, args.output, codec)
    else:
        copy_group_metadata(args.input, args.output, arrays)
        for src_array in arrays:
            recompress_array(
                src_array,
                args.output / src_array.relative_to(args.input),
                codec,
            )


if __name__ == "__main__":
    main()
