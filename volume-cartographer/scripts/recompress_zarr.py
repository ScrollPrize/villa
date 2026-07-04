#!/usr/bin/env python3
"""Recompress an OME-Zarr (zarr v2) volume into the vc_delta_zstd format.

Each stored chunk is decoded with its current compressor, filtered with a
delta along z, then y, then x (mod-2^8/2^16 backward differences), and
re-encoded as a self-describing "VCZ1" payload: a 20-byte header followed by
a zstd frame of the filtered voxels. The array's .zarray metadata is updated
to `{"id": "vc_delta_zstd"}` so readers pick the right codec. On scroll CT
data this roughly halves the size that plain zstd achieves, losslessly,
while both encode and decode stay near zstd speed.

Readers:
  - VC3D / volume-cartographer (codec registered as "vc_delta_zstd")
  - Python: import this file (registers a numcodecs codec) and open the
    volume with zarr as usual.

VCZ1 layout (little-endian):
  0..3   magic 'V' 'C' 'Z' '1'
  4      format version (1)
  5      element size in bytes (1 or 2)
  6..7   reserved (0)
  8..19  chunk dims as three uint32: z, y, x (element counts)
  20..   zstd frame of the filtered payload

Usage:
  # Recompress into a new directory tree (metadata + all chunks):
  python recompress_zarr.py /path/to/volume.zarr --out /path/to/out.zarr

  # Recompress in place (resumable; VCZ1 chunks are detected and skipped):
  python recompress_zarr.py /path/to/volume.zarr --in-place

Every chunk is verified (decode(new) == decode(old)) before it replaces or
lands next to the original; a failed verification aborts the run.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    import numcodecs
except ImportError:  # pragma: no cover
    sys.exit("this script requires numcodecs (pip install numcodecs)")

CODEC_ID = "vc_delta_zstd"
MAGIC = b"VCZ1"
HEADER = struct.Struct("<4sBBHIII")  # magic, version, elemsize, reserved, z, y, x
DEFAULT_ZSTD_LEVEL = 3


def encode_chunk(a: np.ndarray, level: int = DEFAULT_ZSTD_LEVEL) -> bytes:
    if a.ndim != 3:
        raise ValueError("vc_delta_zstd expects 3D chunks")
    if a.dtype not in (np.uint8, np.uint16):
        raise ValueError("vc_delta_zstd supports uint8/uint16 only")
    a = np.ascontiguousarray(a)
    d = a.copy()
    d[1:, :, :] -= a[:-1, :, :]
    e = d.copy()
    e[:, 1:, :] -= d[:, :-1, :]
    f = e.copy()
    f[:, :, 1:] -= e[:, :, :-1]
    z, y, x = a.shape
    header = HEADER.pack(MAGIC, 1, a.dtype.itemsize, 0, z, y, x)
    zstd = numcodecs.Zstd(level=level)
    return header + zstd.encode(f.tobytes())


def decode_chunk(buf: bytes) -> np.ndarray:
    magic, version, elemsize, _res, z, y, x = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or version != 1:
        raise ValueError("not a VCZ1 payload")
    if elemsize not in (1, 2):
        raise ValueError(f"unsupported element size {elemsize}")
    dtype = np.uint8 if elemsize == 1 else np.uint16
    raw = numcodecs.Zstd().decode(bytes(buf[HEADER.size:]))
    f = np.frombuffer(raw, dtype=dtype).reshape(z, y, x)
    e = np.cumsum(f, axis=2, dtype=dtype)
    d = np.cumsum(e, axis=1, dtype=dtype)
    return np.cumsum(d, axis=0, dtype=dtype)


class VcDeltaZstd(numcodecs.abc.Codec):
    """numcodecs codec so Python zarr readers can open recompressed volumes."""

    codec_id = CODEC_ID

    def __init__(self, level: int = DEFAULT_ZSTD_LEVEL):
        self.level = level

    def encode(self, buf):
        return encode_chunk(np.asarray(buf), self.level)

    def decode(self, buf, out=None):
        decoded = decode_chunk(bytes(buf))
        if out is not None:
            np.copyto(np.frombuffer(out, dtype=decoded.dtype).reshape(decoded.shape),
                      decoded)
            return out
        return decoded

    def get_config(self):
        return {"id": self.codec_id, "level": self.level}


numcodecs.register_codec(VcDeltaZstd)


# ---------------------------------------------------------------------------
# zarr v2 directory walking (no zarr-python dependency)
# ---------------------------------------------------------------------------

def find_arrays(root: Path) -> list[Path]:
    """Directories containing a .zarray, e.g. OME-Zarr pyramid levels."""
    return sorted(p.parent for p in root.rglob(".zarray"))


def load_meta(array_dir: Path) -> dict:
    meta = json.loads((array_dir / ".zarray").read_text())
    if meta.get("zarr_format") != 2:
        raise SystemExit(f"{array_dir}: only zarr v2 arrays are supported")
    if meta.get("order", "C") != "C":
        raise SystemExit(f"{array_dir}: only C-order arrays are supported")
    if meta.get("filters"):
        raise SystemExit(f"{array_dir}: v2 filters are not supported")
    if len(meta["shape"]) != 3:
        raise SystemExit(f"{array_dir}: only 3D arrays are supported")
    dtype = np.dtype(meta["dtype"])
    if dtype.byteorder == ">":
        raise SystemExit(f"{array_dir}: big-endian dtypes are not supported")
    if dtype.itemsize not in (1, 2) or dtype.kind != "u":
        raise SystemExit(f"{array_dir}: only uint8/uint16 arrays are supported "
                         f"(got {meta['dtype']})")
    comp = meta.get("compressor")
    if comp and comp.get("id") == "c3d":
        raise SystemExit(f"{array_dir}: c3d-coded volumes cannot be recompressed here")
    return meta


def chunk_files(array_dir: Path) -> list[Path]:
    skip = {".zarray", ".zattrs", ".zgroup", ".zmetadata"}
    return [p for p in array_dir.rglob("*")
            if p.is_file() and p.name not in skip and not p.name.endswith(".tmp")]


def _decode_source(buf: bytes, meta: dict) -> np.ndarray:
    dtype = np.dtype(meta["dtype"])
    comp = meta.get("compressor")
    if buf[:4] == MAGIC:
        return decode_chunk(buf)
    if comp is None:
        raw = buf
    else:
        raw = numcodecs.get_codec(dict(comp)).decode(buf)
    return np.frombuffer(raw, dtype=dtype).reshape(meta["chunks"])[:]


def _process_chunk(args: tuple) -> tuple[str, int, int, bool]:
    """Worker: returns (path, in_bytes, out_bytes, converted)."""
    path_s, meta, out_path_s, level = args
    path = Path(path_s)
    out_path = Path(out_path_s)
    buf = path.read_bytes()

    if buf[:4] == MAGIC:  # already converted (resume / idempotence)
        if out_path != path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(buf)
        return (path_s, len(buf), len(buf), False)

    original = _decode_source(buf, meta)
    encoded = encode_chunk(original, level)

    # Verify before anything replaces the source.
    roundtrip = decode_chunk(encoded)
    if not np.array_equal(roundtrip, original):
        raise RuntimeError(f"{path}: verification failed, aborting")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, out_path)
    return (path_s, len(buf), len(encoded), True)


def recompress_array(array_dir: Path, out_dir: Path, level: int, workers: int) -> None:
    meta = load_meta(array_dir)
    files = chunk_files(array_dir)
    print(f"{array_dir}: {len(files)} chunks "
          f"({meta['dtype']}, chunks={meta['chunks']}, "
          f"compressor={meta.get('compressor')})")

    jobs = [(str(p), meta, str(out_dir / p.relative_to(array_dir)), level)
            for p in files]
    total_in = total_out = converted = 0
    if workers <= 1:
        results = map(_process_chunk, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_process_chunk, jobs, chunksize=8)
    for i, (_path, n_in, n_out, was_converted) in enumerate(results, 1):
        total_in += n_in
        total_out += n_out
        converted += was_converted
        if i % 500 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}  {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB "
                  f"({total_out/max(total_in,1):.3f})", flush=True)
    if workers > 1:
        pool.shutdown()

    # Metadata last: an interrupted in-place run keeps the old compressor id,
    # and the resume path recognizes converted chunks by their VCZ1 magic.
    new_meta = dict(meta)
    new_meta["compressor"] = {"id": CODEC_ID, "level": level}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".zarray").write_text(json.dumps(new_meta, indent=4))
    for extra in (".zattrs",):
        src = array_dir / extra
        if src.exists() and not (out_dir / extra).exists():
            (out_dir / extra).write_text(src.read_text())
    print(f"  done: {converted} converted, ratio "
          f"{total_out/max(total_in,1):.3f} vs source encoding")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recompress an OME-Zarr volume into vc_delta_zstd (VCZ1) chunks.")
    ap.add_argument("volume", type=Path, help="zarr root (contains levels or a .zarray)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="write the recompressed tree here")
    group.add_argument("--in-place", action="store_true",
                       help="replace chunks in the source tree (resumable)")
    ap.add_argument("--level", type=int, default=DEFAULT_ZSTD_LEVEL,
                    help="zstd level (default 3)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel worker processes")
    args = ap.parse_args()

    root = args.volume
    if not root.is_dir():
        sys.exit(f"{root}: not a directory")
    arrays = find_arrays(root)
    if not arrays:
        sys.exit(f"{root}: no .zarray found")

    for array_dir in arrays:
        out_dir = array_dir if args.in_place \
            else args.out / array_dir.relative_to(root)
        recompress_array(array_dir, out_dir, args.level, args.workers)

    if not args.in_place:
        # Copy group-level metadata (.zgroup/.zattrs/.zmetadata) for OME-Zarr.
        for extra in root.rglob("*"):
            if extra.is_file() and extra.name in {".zgroup", ".zattrs", ".zmetadata"} \
                    and not any(a in extra.parents for a in arrays):
                dest = args.out / extra.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_text(extra.read_text())
    print("all arrays done")


if __name__ == "__main__":
    main()
