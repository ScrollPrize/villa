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

  # Recompress and rechunk (each axis must stay a power-of-two factor of
  # the source chunk size, so tile boundaries align and every source chunk
  # is read exactly once; requires --out):
  python recompress_zarr.py /path/to/volume.zarr --out out.zarr --chunk-size 256
  python recompress_zarr.py /path/to/volume.zarr --out out.zarr --chunk-size 64,128,128

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

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

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


def chunk_key_path(array_dir: Path, idx: tuple[int, int, int], sep: str) -> Path:
    if sep == "/":
        return array_dir / str(idx[0]) / str(idx[1]) / str(idx[2])
    return array_dir / f"{idx[0]}.{idx[1]}.{idx[2]}"


def parse_chunk_index(array_dir: Path, path: Path, sep: str) -> tuple[int, int, int] | None:
    parts = path.relative_to(array_dir).parts if sep == "/" \
        else tuple(path.name.split("."))
    if len(parts) != 3:
        return None
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def validate_pow2_chunks(old: list[int], new: list[int], where: str) -> None:
    for axis, (o, n) in enumerate(zip(old, new)):
        big, small = max(o, n), min(o, n)
        if n <= 0 or big % small != 0 or ((big // small) & (big // small - 1)) != 0:
            raise SystemExit(
                f"{where}: new chunk size {n} on axis {axis} is not a "
                f"power-of-two factor of the current chunk size {o}")


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


def _process_tile(args: tuple) -> tuple[int, int, int]:
    """Rechunking worker for one supertile of max(old, new) chunks per axis.

    src_entries: [(voxel_offset_in_tile, path)] source chunks to decode.
    dst_entries: [(voxel_offset_in_tile, out_path)] chunks to emit.
    Returns (in_bytes, out_bytes, written).
    """
    meta, src_entries, dst_entries, tile_shape, new_chunks, level = args
    dtype = np.dtype(meta["dtype"])
    fill = int(meta.get("fill_value") or 0)

    tile = np.full(tile_shape, fill, dtype=dtype)
    bytes_in = 0
    for (oz, oy, ox), path_s in src_entries:
        buf = Path(path_s).read_bytes()
        bytes_in += len(buf)
        a = _decode_source(buf, meta)
        tile[oz:oz + a.shape[0], oy:oy + a.shape[1], ox:ox + a.shape[2]] = a

    bytes_out = 0
    written = 0
    nz, ny, nx = new_chunks
    for (oz, oy, ox), out_path_s in dst_entries:
        sub = np.ascontiguousarray(tile[oz:oz + nz, oy:oy + ny, ox:ox + nx])
        encoded = encode_chunk(sub, level)
        if not np.array_equal(decode_chunk(encoded), sub):
            raise RuntimeError(f"{out_path_s}: verification failed, aborting")
        out_path = Path(out_path_s)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_bytes(encoded)
        os.replace(tmp, out_path)
        bytes_out += len(encoded)
        written += 1
    return (bytes_in, bytes_out, written)


def _rechunk_jobs(array_dir: Path, out_dir: Path, meta: dict,
                  new_chunks: list[int], level: int) -> list[tuple]:
    """One job per supertile: max(old, new) voxels per axis, so power-of-two
    alignment lets each job read whole source chunks and write whole output
    chunks with no cross-tile overlap."""
    sep = meta.get("dimension_separator", ".")
    old = meta["chunks"]
    shape = meta["shape"]
    tile = [max(o, n) for o, n in zip(old, new_chunks)]

    by_index: dict[tuple[int, int, int], Path] = {}
    for p in chunk_files(array_dir):
        idx = parse_chunk_index(array_dir, p, sep)
        if idx is None:
            raise SystemExit(f"{array_dir}: unrecognized chunk file {p}")
        by_index[idx] = p

    def grid(span: list[int]) -> list[int]:
        return [(s + t - 1) // t for s, t in zip(shape, span)]

    jobs = []
    tz, ty, tx = grid(tile)
    for gz in range(tz):
        for gy in range(ty):
            for gx in range(tx):
                origin = (gz * tile[0], gy * tile[1], gx * tile[2])
                src_entries = []
                for iz in range(tile[0] // old[0]):
                    for iy in range(tile[1] // old[1]):
                        for ix in range(tile[2] // old[2]):
                            idx = (origin[0] // old[0] + iz,
                                   origin[1] // old[1] + iy,
                                   origin[2] // old[2] + ix)
                            path = by_index.get(idx)
                            if path is not None:
                                src_entries.append(
                                    ((iz * old[0], iy * old[1], ix * old[2]),
                                     str(path)))
                dst_entries = []
                for iz in range(tile[0] // new_chunks[0]):
                    for iy in range(tile[1] // new_chunks[1]):
                        for ix in range(tile[2] // new_chunks[2]):
                            idx = (origin[0] // new_chunks[0] + iz,
                                   origin[1] // new_chunks[1] + iy,
                                   origin[2] // new_chunks[2] + ix)
                            # Skip chunks entirely past the array bounds and,
                            # for resume, ones already written.
                            if any(i * n >= s for i, n, s in
                                   zip(idx, new_chunks, shape)):
                                continue
                            out_path = chunk_key_path(out_dir, idx, sep)
                            if out_path.exists() and \
                                    out_path.read_bytes()[:4] == MAGIC:
                                continue
                            dst_entries.append(
                                ((iz * new_chunks[0], iy * new_chunks[1],
                                  ix * new_chunks[2]), str(out_path)))
                if src_entries and dst_entries:
                    jobs.append((meta, src_entries, dst_entries,
                                 tuple(tile), tuple(new_chunks), level))
    return jobs


def recompress_array(array_dir: Path, out_dir: Path, level: int, workers: int,
                     chunk_size: list[int] | None = None) -> None:
    meta = load_meta(array_dir)

    if chunk_size is not None and list(chunk_size) != list(meta["chunks"]):
        validate_pow2_chunks(meta["chunks"], chunk_size, str(array_dir))
        recompress_array_rechunk(array_dir, out_dir, meta, chunk_size,
                                 level, workers)
        return

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
    progress = tqdm(total=len(jobs), unit="chunk", desc=f"  {array_dir.name}",
                    smoothing=0.05) if tqdm else None
    for i, (_path, n_in, n_out, was_converted) in enumerate(results, 1):
        total_in += n_in
        total_out += n_out
        converted += was_converted
        if progress:
            progress.set_postfix_str(
                f"{total_in/1e9:.2f}->{total_out/1e9:.2f} GB "
                f"({total_out/max(total_in,1):.3f})", refresh=False)
            progress.update(1)
        elif i % 500 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}  {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB "
                  f"({total_out/max(total_in,1):.3f})", flush=True)
    if progress:
        progress.close()
    if workers > 1:
        pool.shutdown()

    # Metadata last: an interrupted in-place run keeps the old compressor id,
    # and the resume path recognizes converted chunks by their VCZ1 magic.
    _write_array_meta(array_dir, out_dir, meta, meta["chunks"], level)
    print(f"  done: {converted} converted, ratio "
          f"{total_out/max(total_in,1):.3f} vs source encoding")


def _write_array_meta(array_dir: Path, out_dir: Path, meta: dict,
                      chunks: list[int], level: int) -> None:
    new_meta = dict(meta)
    new_meta["chunks"] = list(chunks)
    new_meta["compressor"] = {"id": CODEC_ID, "level": level}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".zarray").write_text(json.dumps(new_meta, indent=4))
    for extra in (".zattrs",):
        src = array_dir / extra
        if src.exists() and not (out_dir / extra).exists():
            (out_dir / extra).write_text(src.read_text())


def recompress_array_rechunk(array_dir: Path, out_dir: Path, meta: dict,
                             new_chunks: list[int], level: int,
                             workers: int) -> None:
    jobs = _rechunk_jobs(array_dir, out_dir, meta, new_chunks, level)
    n_out = sum(len(j[2]) for j in jobs)
    print(f"{array_dir}: rechunk {meta['chunks']} -> {list(new_chunks)}, "
          f"{len(jobs)} tiles / {n_out} output chunks "
          f"({meta['dtype']}, compressor={meta.get('compressor')})")

    total_in = total_out = written = 0
    if workers <= 1:
        results = map(_process_tile, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_process_tile, jobs, chunksize=2)
    progress = tqdm(total=len(jobs), unit="tile", desc=f"  {array_dir.name}",
                    smoothing=0.05) if tqdm else None
    for i, (n_in, n_out_bytes, n_written) in enumerate(results, 1):
        total_in += n_in
        total_out += n_out_bytes
        written += n_written
        if progress:
            progress.set_postfix_str(
                f"{total_in/1e9:.2f}->{total_out/1e9:.2f} GB "
                f"({total_out/max(total_in,1):.3f})", refresh=False)
            progress.update(1)
        elif i % 100 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} tiles  {total_in/1e9:.2f} GB -> "
                  f"{total_out/1e9:.2f} GB", flush=True)
    if progress:
        progress.close()
    if workers > 1:
        pool.shutdown()

    _write_array_meta(array_dir, out_dir, meta, new_chunks, level)
    print(f"  done: {written} chunks written, ratio "
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
    ap.add_argument("--chunk-size", type=str, default=None,
                    help="output chunk size: one value or z,y,x; each axis "
                         "must be a power-of-two factor of the source chunk "
                         "size (requires --out)")
    args = ap.parse_args()

    chunk_size = None
    if args.chunk_size:
        parts = [int(v) for v in args.chunk_size.split(",")]
        if len(parts) == 1:
            parts *= 3
        if len(parts) != 3:
            sys.exit("--chunk-size takes one value or z,y,x")
        chunk_size = parts
        if args.in_place:
            sys.exit("--chunk-size requires --out (old and new chunk keys "
                     "would collide in the source tree)")

    root = args.volume
    if not root.is_dir():
        sys.exit(f"{root}: not a directory")
    arrays = find_arrays(root)
    if not arrays:
        sys.exit(f"{root}: no .zarray found")

    for array_dir in arrays:
        out_dir = array_dir if args.in_place \
            else args.out / array_dir.relative_to(root)
        recompress_array(array_dir, out_dir, args.level, args.workers,
                         chunk_size)

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
