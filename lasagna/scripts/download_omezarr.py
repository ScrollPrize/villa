#!/usr/bin/env python3
"""Download regions of S3 OME-Zarr volumes to local disk.

Produces a local mirror of the remote zarr directory structure that can
be opened directly with ``zarr.open()``.  Only copies raw compressed
chunk bytes — no decoding.

Uses boto3 with per-thread sessions and connection pooling for maximum
parallel throughput on high-latency links (e.g. Starlink).

Usage examples::

    # Download scales 0-2, rectangular region around a point
    python download_omezarr.py s3://bucket/vol.zarr ./vol.zarr \\
        --scales 0-2 --center 2000,3000,1000 --radius 256

    # Download a bbox (coordinates at scale 0)
    python download_omezarr.py s3://bucket/vol.zarr ./vol.zarr \\
        --scales 0,1 --bbox 1000,2000,500,3000,4000,1500

    # Download everything at scale 3
    python download_omezarr.py s3://bucket/vol.zarr ./vol.zarr --scales 3
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import botocore.config
import botocore.exceptions
from botocore import UNSIGNED


# ---------------------------------------------------------------------------
# Per-thread boto3 client — each thread gets its own TCP connection pool
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_s3_client(anon: bool, region: str | None = None):
    """Return a boto3 S3 client local to the calling thread.

    Caches per (anon, region) — a change in either creates a new client.
    """
    cache_key = (anon, region)
    cached = getattr(_thread_local, "s3_client_cache", None)
    if cached is not None:
        key, client = cached
        if key == cache_key:
            return client
    cfg = botocore.config.Config(
        max_pool_connections=4,
        retries={"max_attempts": 0},
        connect_timeout=10,
        read_timeout=30,
    )
    if anon:
        cfg = cfg.merge(botocore.config.Config(signature_version=UNSIGNED))
    session = boto3.Session()
    client = session.client("s3", config=cfg, region_name=region)
    _thread_local.s3_client_cache = (cache_key, client)
    return client


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split 's3://bucket/key/path' into ('bucket', 'key/path')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    rest = uri[5:]
    idx = rest.index("/")
    return rest[:idx], rest[idx + 1:].rstrip("/")


# ---------------------------------------------------------------------------
# Stats shared between scanner, downloaders, and progress display
# ---------------------------------------------------------------------------

_RATE_WINDOW = 10.0  # seconds for rolling speed average


class Stats:
    def __init__(self) -> None:
        self.total_chunks = 0
        self.scanned = 0
        self.local = 0
        self.remote = 0
        self.downloaded = 0
        self.failed = 0
        self.bytes_downloaded = 0
        self.noremote = 0
        self.scan_done = False
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._byte_samples: collections.deque[tuple[float, int]] = collections.deque()
        self._first_download_t: float | None = None
        self.noremote_keys: dict[int, set[str]] = {}  # level → set of chunk keys absent on S3

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            cutoff = now - _RATE_WINDOW
            while self._byte_samples and self._byte_samples[0][0] < cutoff:
                self._byte_samples.popleft()
            if self._byte_samples:
                window_start = self._byte_samples[0][0]
                window_bytes = sum(b for _, b in self._byte_samples)
                window_dur = now - window_start
                rolling_rate = window_bytes / max(window_dur, 0.01)
            else:
                rolling_rate = 0.0
            dl_elapsed = (now - self._first_download_t) if self._first_download_t else 0.0
            return {
                "total": self.total_chunks,
                "scanned": self.scanned,
                "local": self.local,
                "remote": self.remote,
                "downloaded": self.downloaded,
                "failed": self.failed,
                "bytes": self.bytes_downloaded,
                "scan_done": self.scan_done,
                "elapsed": now - self._t0,
                "dl_elapsed": dl_elapsed,
                "rate": rolling_rate,
            }

    def inc(self, **kwargs: int) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, getattr(self, k) + v)
            if "bytes_downloaded" in kwargs and kwargs["bytes_downloaded"] > 0:
                now = time.monotonic()
                if self._first_download_t is None:
                    self._first_download_t = now
                self._byte_samples.append((now, kwargs["bytes_downloaded"]))

    def add_noremote(self, level: int, chunk_key: str) -> None:
        with self._lock:
            self.noremote_keys.setdefault(level, set()).add(chunk_key)

    def is_noremote(self, level: int, chunk_key: str) -> bool:
        with self._lock:
            return chunk_key in self.noremote_keys.get(level, set())


def _noremote_path(local_root: str, level: int) -> str:
    return os.path.join(local_root, ".dl_cache", f"{level}.noremote.json")


def _load_noremote(local_root: str, levels: list[int]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for lvl in levels:
        p = _noremote_path(local_root, lvl)
        if os.path.isfile(p):
            with open(p) as f:
                result[lvl] = set(json.load(f))
        else:
            result[lvl] = set()
    return result


def _save_noremote(local_root: str, noremote: dict[int, set[str]]) -> None:
    for lvl, keys in noremote.items():
        if not keys:
            continue
        p = _noremote_path(local_root, lvl)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(sorted(keys), f)
            f.write("\n")


# ---------------------------------------------------------------------------
# S3 helpers using boto3 directly
# ---------------------------------------------------------------------------

def _s3_read_bytes(bucket: str, key: str, anon: bool) -> bytes:
    client = _get_s3_client(anon)
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def _s3_read_json(bucket: str, key: str, anon: bool) -> dict:
    return json.loads(_s3_read_bytes(bucket, key, anon))


def _s3_exists(bucket: str, key: str, anon: bool) -> bool:
    client = _get_s3_client(anon)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def _s3_list_prefix(bucket: str, prefix: str, anon: bool) -> list[str]:
    """List immediate children (CommonPrefixes + Contents) under prefix."""
    client = _get_s3_client(anon)
    if not prefix.endswith("/"):
        prefix += "/"
    result: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            result.append(cp["Prefix"].rstrip("/"))
        for obj in page.get("Contents", []):
            result.append(obj["Key"])
    return result


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _write_local_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _parse_scales_arg(s: str) -> list[int]:
    parts = s.replace(" ", "").split(",")
    out: list[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))


def _parse_coord6(s: str) -> tuple[int, int, int, int, int, int]:
    vals = [int(v) for v in s.replace(" ", "").split(",")]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError(f"expected 6 comma-separated ints, got {len(vals)}")
    return tuple(vals)  # type: ignore[return-value]


def _parse_coord3(s: str) -> tuple[int, int, int]:
    vals = [int(v) for v in s.replace(" ", "").split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 comma-separated ints, got {len(vals)}")
    return tuple(vals)  # type: ignore[return-value]


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# OME-Zarr metadata
# ---------------------------------------------------------------------------

def _discover_levels(bucket: str, prefix: str, anon: bool) -> list[int]:
    entries = _s3_list_prefix(bucket, prefix, anon)
    levels: list[int] = []
    for e in entries:
        name = e.rstrip("/").rsplit("/", 1)[-1]
        try:
            lvl = int(name)
        except ValueError:
            continue
        zarray_key = f"{prefix}/{name}/.zarray"
        if _s3_exists(bucket, zarray_key, anon):
            levels.append(lvl)
    return sorted(levels)


def _parse_multiscales(zattrs: dict) -> dict[int, list[float]]:
    ms_list = zattrs.get("multiscales", [])
    if not ms_list:
        return {}
    ms = ms_list[0]
    n_spatial = 3
    result: dict[int, list[float]] = {}
    for ds in ms.get("datasets", []):
        path = ds.get("path", "")
        try:
            lvl = int(path)
        except ValueError:
            continue
        transforms = ds.get("coordinateTransformations", [])
        scale = [1.0, 1.0, 1.0]
        for t in transforms:
            if t.get("type") == "scale":
                s = t["scale"]
                scale = [float(v) for v in s[-n_spatial:]]
                break
        result[lvl] = scale
    return result


def _level_meta(bucket: str, prefix: str, level: int, anon: bool) -> dict:
    zarray = _s3_read_json(bucket, f"{prefix}/{level}/.zarray", anon)
    shape = tuple(zarray["shape"])
    chunks = tuple(zarray["chunks"])
    dim_sep = zarray.get("dimension_separator", ".")
    return {"shape": shape, "chunks": chunks, "dim_sep": dim_sep, "zarray": zarray}


# ---------------------------------------------------------------------------
# Chunk enumeration
# ---------------------------------------------------------------------------

def _scale_bbox(bbox_zyx: tuple[int, int, int, int, int, int],
                scale_factors: list[float],
                base_scale: list[float]) -> tuple[int, int, int, int, int, int]:
    z0, y0, x0, z1, y1, x1 = bbox_zyx
    rz = scale_factors[0] / base_scale[0]
    ry = scale_factors[1] / base_scale[1]
    rx = scale_factors[2] / base_scale[2]
    return (
        int(math.floor(z0 / rz)),
        int(math.floor(y0 / ry)),
        int(math.floor(x0 / rx)),
        int(math.ceil(z1 / rz)),
        int(math.ceil(y1 / ry)),
        int(math.ceil(x1 / rx)),
    )


def _enumerate_chunks(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    bbox_zyx: tuple[int, int, int, int, int, int] | None = None,
) -> list[tuple[int, int, int]]:
    nz = math.ceil(shape[0] / chunks[0])
    ny = math.ceil(shape[1] / chunks[1])
    nx = math.ceil(shape[2] / chunks[2])

    if bbox_zyx is not None:
        z0, y0, x0, z1, y1, x1 = bbox_zyx
        iz0 = max(0, z0 // chunks[0])
        iy0 = max(0, y0 // chunks[1])
        ix0 = max(0, x0 // chunks[2])
        iz1 = min(nz, math.ceil(z1 / chunks[0]))
        iy1 = min(ny, math.ceil(y1 / chunks[1]))
        ix1 = min(nx, math.ceil(x1 / chunks[2]))
    else:
        iz0, iy0, ix0 = 0, 0, 0
        iz1, iy1, ix1 = nz, ny, nx

    result = []
    for iz in range(iz0, iz1):
        for iy in range(iy0, iy1):
            for ix in range(ix0, ix1):
                result.append((iz, iy, ix))
    return result


def _chunk_key(dim_sep: str, iz: int, iy: int, ix: int) -> str:
    return dim_sep.join(str(i) for i in (iz, iy, ix))


def _chunk_local_path(local_level: str, dim_sep: str, iz: int, iy: int, ix: int) -> str:
    key = _chunk_key(dim_sep, iz, iy, ix)
    return os.path.join(local_level, key.replace("/", os.sep))


def _chunk_s3_key(level_prefix: str, dim_sep: str, iz: int, iy: int, ix: int) -> str:
    key = _chunk_key(dim_sep, iz, iy, ix)
    return f"{level_prefix}/{key}"


# ---------------------------------------------------------------------------
# Scanner thread
# ---------------------------------------------------------------------------

def _scanner(
    bucket: str,
    prefix: str,
    local_root: str,
    levels_meta: dict[int, dict],
    bbox_zyx_level0: tuple[int, int, int, int, int, int] | None,
    multiscales: dict[int, list[float]],
    download_queue: queue.Queue,
    stats: Stats,
    stop_event: threading.Event,
) -> None:
    base_scale = multiscales.get(0, [1.0, 1.0, 1.0])

    # (level, s3_key, local_path, chunk_key_within_level)
    all_items: list[tuple[int, str, str, str]] = []
    for lvl, meta in sorted(levels_meta.items()):
        level_bbox = None
        if bbox_zyx_level0 is not None:
            lvl_scale = multiscales.get(lvl, base_scale)
            level_bbox = _scale_bbox(bbox_zyx_level0, lvl_scale, base_scale)
        chunk_indices = _enumerate_chunks(meta["shape"], meta["chunks"], level_bbox)
        level_prefix = f"{prefix}/{lvl}"
        local_level = os.path.join(local_root, str(lvl))
        dim_sep = meta["dim_sep"]
        for iz, iy, ix in chunk_indices:
            s3key = _chunk_s3_key(level_prefix, dim_sep, iz, iy, ix)
            lpath = _chunk_local_path(local_level, dim_sep, iz, iy, ix)
            ckey = _chunk_key(dim_sep, iz, iy, ix)
            all_items.append((lvl, s3key, lpath, ckey))

    stats.inc(total_chunks=len(all_items))
    random.shuffle(all_items)

    for lvl, s3key, lpath, ckey in all_items:
        if stop_event.is_set():
            break
        if os.path.isfile(lpath) and os.path.getsize(lpath) > 0:
            stats.inc(scanned=1, local=1)
        elif stats.is_noremote(lvl, ckey):
            stats.inc(scanned=1, local=1, noremote=1)
        else:
            stats.inc(scanned=1, remote=1)
            download_queue.put((lvl, s3key, lpath, ckey))

    stats.scan_done = True


# ---------------------------------------------------------------------------
# Downloader — one call per chunk, runs inside ThreadPoolExecutor
# ---------------------------------------------------------------------------

def _download_one(
    bucket: str,
    s3_key: str,
    local_path: str,
    anon: bool,
    stats: Stats,
    level: int = 0,
    chunk_key: str = "",
    max_retries: int = 3,
) -> bool:
    for attempt in range(max_retries):
        try:
            client = _get_s3_client(anon)
            resp = client.get_object(Bucket=bucket, Key=s3_key)
            data = resp["Body"].read()
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            tmp_path = local_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, local_path)
            stats.inc(downloaded=1, bytes_downloaded=len(data))
            return True
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                stats.add_noremote(level, chunk_key)
                stats.inc(downloaded=1)
                return True
            if attempt == max_retries - 1:
                print(f"\n  WARN: failed {s3_key}: {e}", file=sys.stderr)
                stats.inc(failed=1)
                return False
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n  WARN: failed {s3_key}: {e}", file=sys.stderr)
                stats.inc(failed=1)
                try:
                    os.unlink(local_path + ".tmp")
                except OSError:
                    pass
                return False
            time.sleep(0.5 * (attempt + 1))
    return False


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _progress_loop(stats: Stats, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        s = stats.snapshot()
        total = s["total"]
        scanned = s["scanned"]
        local = s["local"]
        remote = s["remote"]
        downloaded = s["downloaded"]
        failed = s["failed"]
        byt = s["bytes"]
        elapsed = s["elapsed"]
        scan_done = s["scan_done"]
        rate = s["rate"]

        dl_elapsed = s["dl_elapsed"]
        avg_rate = byt / max(dl_elapsed, 0.01) if dl_elapsed > 0 else 0.0

        if total > 0 and scanned > 0:
            scan_pct = 100.0 * scanned / total
            if scan_done:
                need = remote - downloaded - failed
                tag = "scan done"
            else:
                est_remote = int(remote * total / scanned)
                need = est_remote - downloaded - failed
                tag = f"scan {scan_pct:4.1f}%"
            need = max(need, 0)

            if rate > 0 and need > 0:
                avg_chunk_bytes = byt / max(downloaded, 1)
                eta_s = (need * avg_chunk_bytes) / rate
                eta_str = f"{eta_s:.0f}s"
            else:
                eta_str = "---"

            tilde = "" if scan_done else "~"
            line = (
                f"\r[{tag}] local: {local:,} | to_dl: {tilde}{need:,} | "
                f"done: {downloaded:,} ({_fmt_bytes(int(rate))}/s, "
                f"avg {_fmt_bytes(int(avg_rate))}/s) | "
                f"fail: {failed} | ETA: {tilde}{eta_str}  "
            )
        else:
            line = "\r[starting scan...]  "

        sys.stderr.write(line)
        sys.stderr.flush()
        stop_event.wait(0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download regions of S3 OME-Zarr volumes to local disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="S3 URI to OME-Zarr root, e.g. s3://bucket/vol.zarr")
    p.add_argument("dest", help="Local destination path")
    p.add_argument("--scales", default=None,
                   help="Scale levels to download: '0-2' or '0,1,3' (default: all)")
    p.add_argument("--bbox", default=None,
                   help="Bounding box at scale 0: x0,y0,z0,x1,y1,z1")
    p.add_argument("--center", default=None,
                   help="Center point at scale 0: x,y,z")
    p.add_argument("--radius", type=int, default=None,
                   help="Rectangular half-size in all dims (used with --center)")
    p.add_argument("--zrange", default=None,
                   help="Z slice range at scale 0: z0,z1 (downloads all X/Y for that Z range)")
    p.add_argument("--workers", type=int, default=64,
                   help="Parallel download threads (default: 64)")
    p.add_argument("--anon", action="store_true",
                   help="Use anonymous S3 access (no credentials)")
    p.add_argument("--region", default=None,
                   help="AWS region (default: auto-detect)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    remote_root = args.source.rstrip("/")
    local_root = args.dest.rstrip("/")

    region_flags = sum(1 for f in [args.bbox, args.center, args.zrange] if f)
    if region_flags > 1:
        print("ERROR: --bbox, --center, and --zrange are mutually exclusive", file=sys.stderr)
        return 1
    if args.center and args.radius is None:
        print("ERROR: --center requires --radius", file=sys.stderr)
        return 1

    bbox_zyx: tuple[int, int, int, int, int, int] | None = None
    if args.bbox:
        x0, y0, z0, x1, y1, z1 = _parse_coord6(args.bbox)
        bbox_zyx = (z0, y0, x0, z1, y1, x1)
    elif args.center:
        cx, cy, cz = _parse_coord3(args.center)
        r = args.radius
        bbox_zyx = (cz - r, cy - r, cx - r, cz + r, cy + r, cx + r)
    elif args.zrange:
        zvals = [int(v) for v in args.zrange.replace(" ", "").split(",")]
        if len(zvals) != 2:
            print("ERROR: --zrange expects z0,z1", file=sys.stderr)
            return 1
        bbox_zyx = (zvals[0], 0, 0, zvals[1], 10_000_000, 10_000_000)

    bucket, prefix = _parse_s3_uri(remote_root)
    anon = args.anon

    # Warm up the main-thread client (also validates credentials/connectivity)
    _get_s3_client(anon, region=args.region)

    print(f"Source: s3://{bucket}/{prefix}", file=sys.stderr)
    print(f"Dest:   {local_root}", file=sys.stderr)

    # --- Metadata ---
    print("Reading metadata...", file=sys.stderr)
    os.makedirs(local_root, exist_ok=True)

    for meta_file in [".zgroup", ".zattrs"]:
        s3key = f"{prefix}/{meta_file}"
        lpath = os.path.join(local_root, meta_file)
        if not os.path.exists(lpath):
            try:
                data = _s3_read_json(bucket, s3key, anon)
                _write_local_json(lpath, data)
            except (botocore.exceptions.ClientError, FileNotFoundError):
                pass

    try:
        zattrs = _s3_read_json(bucket, f"{prefix}/.zattrs", anon)
    except (botocore.exceptions.ClientError, FileNotFoundError):
        zattrs = {}

    multiscales = _parse_multiscales(zattrs)
    available_levels = _discover_levels(bucket, prefix, anon)
    if not available_levels:
        print("ERROR: no OME-Zarr levels found", file=sys.stderr)
        return 1

    if args.scales is not None:
        requested = _parse_scales_arg(args.scales)
        levels = [l for l in requested if l in available_levels]
        missing_lvls = [l for l in requested if l not in available_levels]
        if missing_lvls:
            print(f"WARN: levels {missing_lvls} not found, available: {available_levels}",
                  file=sys.stderr)
    else:
        levels = available_levels

    if not levels:
        print("ERROR: no valid levels to download", file=sys.stderr)
        return 1

    print(f"Levels: {levels}", file=sys.stderr)
    if bbox_zyx:
        z0, y0, x0, z1, y1, x1 = bbox_zyx
        print(f"BBox (ZYX at scale 0): [{z0}:{z1}, {y0}:{y1}, {x0}:{x1}]", file=sys.stderr)

    levels_meta: dict[int, dict] = {}
    for lvl in levels:
        meta = _level_meta(bucket, prefix, lvl, anon)
        levels_meta[lvl] = meta
        lvl_dir = os.path.join(local_root, str(lvl))
        os.makedirs(lvl_dir, exist_ok=True)
        lpath = os.path.join(lvl_dir, ".zarray")
        if not os.path.exists(lpath):
            _write_local_json(lpath, meta["zarray"])
        try:
            lvl_zattrs = _s3_read_json(bucket, f"{prefix}/{lvl}/.zattrs", anon)
            la = os.path.join(lvl_dir, ".zattrs")
            if not os.path.exists(la):
                _write_local_json(la, lvl_zattrs)
        except (botocore.exceptions.ClientError, FileNotFoundError):
            pass
        print(f"  Level {lvl}: shape={meta['shape']} chunks={meta['chunks']} sep='{meta['dim_sep']}'",
              file=sys.stderr)

    if not multiscales:
        for lvl in levels:
            multiscales[lvl] = [1.0, 1.0, 1.0]

    # --- Run ---
    st = Stats()
    st.noremote_keys = _load_noremote(local_root, levels)
    n_cached_noremote = sum(len(v) for v in st.noremote_keys.values())
    if n_cached_noremote:
        print(f"Loaded {n_cached_noremote} cached noremote keys", file=sys.stderr)
    dl_queue: queue.Queue[tuple[int, str, str, str] | None] = queue.Queue(maxsize=10000)
    stop_evt = threading.Event()

    scanner_thread = threading.Thread(
        target=_scanner,
        args=(bucket, prefix, local_root, levels_meta, bbox_zyx,
              multiscales, dl_queue, st, stop_evt),
        daemon=True,
    )
    progress_thread = threading.Thread(
        target=_progress_loop, args=(st, stop_evt), daemon=True,
    )

    scanner_thread.start()
    progress_thread.start()

    n_workers = args.workers

    def _drain_queue() -> None:
        futures: list = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            while True:
                try:
                    item = dl_queue.get(timeout=0.2)
                except queue.Empty:
                    if st.scan_done and dl_queue.empty():
                        break
                    continue
                if item is None:
                    break
                lvl, s3key, lpath, ckey = item
                fut = pool.submit(_download_one, bucket, s3key, lpath, anon, st,
                                  level=lvl, chunk_key=ckey)
                futures.append(fut)
            for f in as_completed(futures):
                f.result()

    try:
        _drain_queue()
    except KeyboardInterrupt:
        print("\nInterrupted — stopping...", file=sys.stderr)
        stop_evt.set()

    stop_evt.set()
    scanner_thread.join(timeout=5)

    _save_noremote(local_root, st.noremote_keys)

    # Store S3 source URI in local .zattrs so predict3d can re-download
    _store_download_meta(local_root, remote_root, anon, args.region)

    snap = st.snapshot()
    sys.stderr.write("\n")
    print(
        f"Complete: {snap['total']:,} chunks "
        f"({snap['local']:,} cached + {snap['downloaded']:,} downloaded"
        + (f", {snap['failed']} failed" if snap["failed"] else "")
        + f", {_fmt_bytes(snap['bytes'])})"
        f" in {snap['elapsed']:.1f}s",
        file=sys.stderr,
    )
    return 1 if snap["failed"] else 0


def _store_download_meta(local_root: str, source_uri: str, anon: bool,
                         region: str | None = None) -> None:
    """Write _download metadata into local .zattrs for later re-download."""
    zattrs_path = os.path.join(local_root, ".zattrs")
    if os.path.isfile(zattrs_path):
        with open(zattrs_path) as f:
            zattrs = json.load(f)
    else:
        zattrs = {}
    dl_meta: dict = {"source": source_uri, "anon": anon}
    if region:
        dl_meta["region"] = region
    zattrs["_download"] = dl_meta
    _write_local_json(zattrs_path, zattrs)


def download(
    source: str,
    dest: str,
    *,
    scales: list[int] | None = None,
    bbox_xyzxyz: tuple[int, int, int, int, int, int] | None = None,
    workers: int = 64,
    anon: bool = False,
    region: str | None = None,
) -> int:
    """Programmatic entry point — same as CLI but with keyword args.

    Returns 0 on success, 1 on failure.
    """
    argv = [source, dest]
    if scales is not None:
        argv += ["--scales", ",".join(str(s) for s in scales)]
    if bbox_xyzxyz is not None:
        x0, y0, z0, x1, y1, z1 = bbox_xyzxyz
        argv += ["--bbox", f"{x0},{y0},{z0},{x1},{y1},{z1}"]
    if anon:
        argv.append("--anon")
    if region:
        argv += ["--region", region]
    argv += ["--workers", str(workers)]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
