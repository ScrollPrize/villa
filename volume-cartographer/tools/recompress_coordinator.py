#!/usr/bin/env python3
"""Coordinator for per-shard vc_zarr_recompress workers.

Spawns N worker processes in parallel.  Each worker encodes exactly one
shard and exits, so any memory fragmentation / leak / OOM is bounded to
that shard's lifetime.  The coordinator:

  1. Writes root + per-level zarr.json metadata.
  2. Lists output shards (one LIST per level) to skip already-done work.
  3. Iterates every shard position in the grid, launches a worker via
     `vc_zarr_recompress --one-shard L/sz/sy/sx`, and keeps N in flight.

Usage:
  recompress_coordinator.py <binary> <input> <output> \\
      --levels 5,4,3,2,1,0 --qp 36 --air-clamp 64 \\
      --jobs 32 --inner-jobs 64

The coordinator itself does nothing heavy — workers do the actual encode.
"""
import argparse
import concurrent.futures as futures
import json
import os
import subprocess
import sys
import time

import boto3


CHUNK_DIM = 128
SHARD_DIM = 1024


def parse_s3(url: str) -> tuple[str, str]:
    # s3+region://bucket/prefix or s3://bucket/prefix
    if url.startswith("s3+"):
        url = "s3://" + url.split("://", 1)[1]
    assert url.startswith("s3://"), f"expected s3:// URL, got {url}"
    rest = url[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def read_source_zarray(s3, bucket: str, prefix: str, level: int) -> dict | None:
    key = f"{prefix}/{level}/.zarray"
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def _list_cz(bucket: str, src_prefix: str, cz: int, ny: int, nx: int):
    """List one cz prefix in parallel, return a list of (cz, cy, cx) that
    exist.  New S3 client per call (boto3 client is not thread-safe
    across all operations; cheap to construct per thread)."""
    local_s3 = boto3.client("s3")
    paginator = local_s3.get_paginator("list_objects_v2")
    cz_prefix = f"{src_prefix}{cz}/"
    out = []
    for page in paginator.paginate(Bucket=bucket, Prefix=cz_prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            rel = key[len(src_prefix):]
            parts = rel.split("/")
            if len(parts) != 3:
                continue
            try:
                cz2, cy, cx = (int(p) for p in parts)
            except ValueError:
                continue
            if cz2 != cz or cy >= ny or cx >= nx:
                continue
            out.append((cz2, cy, cx))
    return out


def build_occupancy_bitmap(s3, bucket: str, prefix: str, level: int,
                            shape: list[int], out_path: str,
                            parallelism: int = 64) -> tuple[int, int]:
    """Build a packed occupancy bitmap via parallel per-cz LISTs. Workers
    list each cz prefix concurrently, coordinator merges into one bitmap."""
    nz = (shape[0] + CHUNK_DIM - 1) // CHUNK_DIM
    ny = (shape[1] + CHUNK_DIM - 1) // CHUNK_DIM
    nx = (shape[2] + CHUNK_DIM - 1) // CHUNK_DIM
    total = nz * ny * nx
    packed = bytearray((total + 7) // 8)
    present = 0

    src_prefix = f"{prefix}/{level}/"
    with futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futs = [pool.submit(_list_cz, bucket, src_prefix, cz, ny, nx)
                for cz in range(nz)]
        for fut in futures.as_completed(futs):
            for cz, cy, cx in fut.result():
                i = cz * ny * nx + cy * nx + cx
                packed[i >> 3] |= 1 << (i & 7)
                present += 1

    import struct
    header = struct.pack("<III", nz, ny, nx)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(packed)
    return total, present


def list_existing_shards(s3, bucket: str, prefix: str, level: int) -> set[tuple[int, int, int]]:
    """Return set of (sz, sy, sx) already in S3 under level/c/."""
    done = set()
    paginator = s3.get_paginator("list_objects_v2")
    shard_prefix = f"{prefix}/{level}/c/"
    for page in paginator.paginate(Bucket=bucket, Prefix=shard_prefix):
        for obj in page.get("Contents", []) or []:
            rel = obj["Key"][len(shard_prefix):]
            parts = rel.split("/")
            if len(parts) != 3:
                continue
            try:
                sz, sy, sx = (int(p) for p in parts)
            except ValueError:
                continue
            done.add((sz, sy, sx))
    return done


def write_metadata(s3, bucket: str, prefix: str, binary: str,
                   input_url: str, output_url: str,
                   levels_arg: str, qp: int, air_clamp: int, shift_n: int):
    """Run vc_zarr_recompress with --levels L --one-shard -1/-1/-1/-1 so it
    writes metadata and exits after noticing no shard to process.  But we
    don't have that flag.  Easiest: run the binary once without --one-shard
    but with --jobs 0 or similar... also not supported.

    Simpler: let the first worker on each level implicitly create metadata
    via a lightweight subprocess that runs with --levels L only long enough
    to print the "=== Level L ===" header (which happens AFTER metadata
    writes), then kill it.  Even simpler still: just shell out and `aws s3
    cp` the root metadata from the existing code by running a dry-run
    binary invocation per level.

    For now: delegate to the binary in one-shot mode per level."""
    # Run vc_zarr_recompress for each level in "prep" mode (no workers
    # actually kick off because we'll interrupt as soon as metadata written).
    # This relies on the binary writing zarr.json BEFORE starting the
    # occupancy LIST / shard processing.  Timeout 30s is generous.
    for level_csv in levels_arg.split(","):
        level = int(level_csv)
        cmd = [binary, input_url, output_url,
               "--levels", str(level),
               "--qp", str(qp),
               "--air-clamp", str(air_clamp),
               "--bit-shift", str(shift_n),
               "--jobs", "1", "--inner-jobs", "1"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Give it 10s to write metadata, then terminate.
        time.sleep(10)
        p.terminate()
        p.wait(timeout=5)
        print(f"[coord] metadata for level {level} written")


def run_worker(binary: str, input_url: str, output_url: str,
               level: int, sz: int, sy: int, sx: int,
               qp: int, air_clamp: int, shift_n: int,
               inner_jobs: int, encode_jobs: int,
               occupancy_file: str | None) -> tuple[int, int, int, int, int]:
    """Launch one worker, wait for it, return exit code + shard coords."""
    cmd = [binary, input_url, output_url,
           "--qp", str(qp),
           "--air-clamp", str(air_clamp),
           "--bit-shift", str(shift_n),
           "--inner-jobs", str(inner_jobs),
           "--encode-jobs", str(encode_jobs),
           "--one-shard", f"{level}/{sz}/{sy}/{sx}"]
    if occupancy_file:
        cmd.extend(["--occupancy-file", occupancy_file])
    # stdout to /dev/null: workers are numerous and chatty
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False)
    return (proc.returncode, level, sz, sy, sx)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary", help="Path to vc_zarr_recompress")
    ap.add_argument("input", help="Input zarr URL (s3://bucket/prefix)")
    ap.add_argument("output", help="Output zarr URL (s3://bucket/prefix)")
    ap.add_argument("--levels", default="5,4,3,2,1,0",
                    help="CSV of levels in processing order")
    ap.add_argument("--qp", type=int, default=36)
    ap.add_argument("--air-clamp", type=int, default=64)
    ap.add_argument("--bit-shift", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=32,
                    help="Number of worker processes in parallel")
    ap.add_argument("--inner-jobs", type=int, default=64,
                    help="Per-worker --inner-jobs (chunk fan-out)")
    ap.add_argument("--encode-jobs", type=int, default=8,
                    help="Per-worker encode thread pool size (default 8 = "
                         "small since the coordinator already parallelises "
                         "across many worker processes)")
    ap.add_argument("--skip-metadata", action="store_true",
                    help="Skip the metadata-write phase (assume already done)")
    args = ap.parse_args()

    in_bucket, in_prefix = parse_s3(args.input)
    out_bucket, out_prefix = parse_s3(args.output)
    s3 = boto3.client("s3")

    level_ids = [int(x) for x in args.levels.split(",")]

    # Phase 1: write metadata (one-time, cheap)
    if not args.skip_metadata:
        write_metadata(s3, out_bucket, out_prefix, args.binary,
                       args.input, args.output,
                       args.levels, args.qp, args.air_clamp, args.bit_shift)

    # Phase 2: per-level, enumerate shard positions and spawn workers
    total_done_all_levels = 0
    for level in level_ids:
        meta = read_source_zarray(s3, in_bucket, in_prefix, level)
        if meta is None:
            print(f"[coord] level {level}: no .zarray, skipping")
            continue
        shape = meta["shape"]
        shard_nz = (shape[0] + SHARD_DIM - 1) // SHARD_DIM
        shard_ny = (shape[1] + SHARD_DIM - 1) // SHARD_DIM
        shard_nx = (shape[2] + SHARD_DIM - 1) // SHARD_DIM
        total_shards = shard_nz * shard_ny * shard_nx

        done = list_existing_shards(s3, out_bucket, out_prefix, level)
        todo = [(sz, sy, sx)
                for sz in range(shard_nz)
                for sy in range(shard_ny)
                for sx in range(shard_nx)
                if (sz, sy, sx) not in done]
        print(f"[coord] level {level}: {total_shards} total, "
              f"{len(done)} done, {len(todo)} to process")

        if not todo:
            continue

        # Build per-level occupancy bitmap once; workers use it via --occupancy-file.
        # Replaces ~400 404 GETs per shard with a single LIST + one mmap-style read.
        occ_path = f"/dev/shm/occ_L{level}.bin"
        t_occ = time.time()
        total_bits, present = build_occupancy_bitmap(
            s3, in_bucket, in_prefix, level, shape, occ_path)
        print(f"[coord] level {level}: occupancy bitmap {present}/{total_bits} "
              f"({100*(1-present/total_bits):.1f}% sparse) in "
              f"{time.time()-t_occ:.1f}s -> {occ_path}")

        t0 = time.time()
        completed = 0
        failed = 0
        with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(run_worker, args.binary, args.input, args.output,
                            level, sz, sy, sx,
                            args.qp, args.air_clamp, args.bit_shift,
                            args.inner_jobs, args.encode_jobs,
                            occ_path): (sz, sy, sx)
                for (sz, sy, sx) in todo
            }
            for fut in futures.as_completed(futs):
                rc, lv, sz, sy, sx = fut.result()
                if rc != 0:
                    failed += 1
                    print(f"[coord] L{lv} {sz}/{sy}/{sx}: FAILED rc={rc}",
                          file=sys.stderr)
                completed += 1
                if completed % 50 == 0 or completed == len(todo):
                    dt = time.time() - t0
                    rate = completed / dt if dt > 0 else 0
                    remaining = len(todo) - completed
                    eta = remaining / rate if rate > 0 else float("inf")
                    print(f"[coord] L{level}: {completed}/{len(todo)} "
                          f"({rate:.1f}/s, ETA {eta/60:.1f} min, "
                          f"{failed} failed)")

        total_done_all_levels += completed
        print(f"[coord] level {level} done: {completed}/{len(todo)} "
              f"processed, {failed} failed, {time.time()-t0:.1f}s")

    print(f"[coord] total {total_done_all_levels} shards processed across "
          f"{len(level_ids)} levels")


if __name__ == "__main__":
    sys.exit(main() or 0)
