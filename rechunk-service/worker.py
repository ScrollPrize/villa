"""Background worker for rechunking zarr volumes with H264 encoding."""

import io
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Any
from urllib.parse import urlparse

import boto3
import numpy as np
import requests

from encoder import encode_h264

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    source_url: str  # zarr root URL (http or s3)
    target_bucket: str
    target_prefix: str
    codec: str = "h264"
    qp: int = 26
    chunk_size: list[int] | None = None  # override chunk shape, or None to keep original
    workers: int = 16
    status: JobStatus = JobStatus.PENDING
    progress_done: int = 0
    progress_total: int = 0
    error: str = ""
    cancel_event: Event = field(default_factory=Event)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_url": self.source_url,
            "target_bucket": self.target_bucket,
            "target_prefix": self.target_prefix,
            "codec": self.codec,
            "qp": self.qp,
            "chunk_size": self.chunk_size,
            "workers": self.workers,
            "status": self.status.value,
            "progress_done": self.progress_done,
            "progress_total": self.progress_total,
            "error": self.error,
            "created_at": self.created_at,
        }


def _fetch_url(url: str) -> bytes:
    """Fetch bytes from HTTP(S) URL."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _fetch_json(url: str) -> Any:
    return json.loads(_fetch_url(url))


def _s3_upload(s3, bucket: str, key: str, data: bytes):
    s3.put_object(Bucket=bucket, Key=key, Body=data)


def _decompress_chunk(raw: bytes, compressor_config: dict | None) -> np.ndarray:
    """Decompress a zarr chunk using its compressor config."""
    if compressor_config is None:
        return np.frombuffer(raw, dtype=np.uint8)

    codec_id = compressor_config.get("id", "")

    if codec_id == "blosc":
        import numcodecs
        codec = numcodecs.Blosc.from_config(compressor_config)
        return np.frombuffer(codec.decode(raw), dtype=np.uint8)
    elif codec_id == "zstd":
        import numcodecs
        codec = numcodecs.Zstd.from_config(compressor_config)
        return np.frombuffer(codec.decode(raw), dtype=np.uint8)
    elif codec_id == "gzip":
        import gzip
        return np.frombuffer(gzip.decompress(raw), dtype=np.uint8)
    elif codec_id == "zlib":
        import zlib
        return np.frombuffer(zlib.decompress(raw), dtype=np.uint8)
    else:
        # Try numcodecs generic
        import numcodecs
        codec = numcodecs.get_codec(compressor_config)
        return np.frombuffer(codec.decode(raw), dtype=np.uint8)


def _discover_levels(source_url: str) -> list[str]:
    """Find pyramid levels (0, 1, 2, ...) under source zarr URL."""
    # Try reading .zgroup at root
    try:
        _fetch_json(f"{source_url}/.zgroup")
    except Exception:
        # Not a group, treat as single-level array
        return [""]

    # Probe for numbered subdirectories
    levels = []
    for i in range(20):
        try:
            _fetch_json(f"{source_url}/{i}/.zarray")
            levels.append(str(i))
        except Exception:
            break

    if not levels:
        # Single array, no pyramid
        return [""]
    return levels


def _process_level(
    job: Job, s3, source_url: str, level: str,
    target_prefix: str, level_progress_offset: int
) -> int:
    """Process one pyramid level. Returns number of chunks processed."""
    level_url = f"{source_url}/{level}" if level else source_url
    level_prefix = f"{target_prefix}/{level}" if level else target_prefix

    # Read .zarray metadata
    zarray = _fetch_json(f"{level_url}/.zarray")
    shape = zarray["shape"]
    chunks = zarray["chunks"]
    compressor = zarray.get("compressor")
    dtype = zarray.get("dtype", "|u1")

    if len(shape) != 3:
        log.warning(f"Skipping level {level}: shape {shape} is not 3D")
        return 0

    nz = (shape[0] + chunks[0] - 1) // chunks[0]
    ny = (shape[1] + chunks[1] - 1) // chunks[1]
    nx = (shape[2] + chunks[2] - 1) // chunks[2]
    total_chunks = nz * ny * nx
    log.info(
        f"Level {level}: shape={shape}, chunks={chunks}, "
        f"grid={nz}x{ny}x{nx} = {total_chunks} chunks"
    )

    # Write zarr metadata to target (with our codec info)
    target_zarray = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": None,  # raw VC3D bytes, no zarr compressor
        "fill_value": 0,
        "order": "C",
        "zarr_format": 2,
        "filters": None,
    }
    _s3_upload(s3, job.target_bucket, f"{level_prefix}/.zarray",
               json.dumps(target_zarray, indent=2).encode())

    # Copy .zattrs if present
    try:
        zattrs = _fetch_url(f"{level_url}/.zattrs")
        _s3_upload(s3, job.target_bucket, f"{level_prefix}/.zattrs", zattrs)
    except Exception:
        pass

    # Build chunk coordinate list
    chunk_coords = [
        (cz, cy, cx)
        for cz in range(nz)
        for cy in range(ny)
        for cx in range(nx)
    ]

    def process_one_chunk(coords: tuple[int, int, int]) -> bool:
        if job.cancel_event.is_set():
            return False

        cz, cy, cx = coords
        chunk_key = f"{cz}.{cy}.{cx}"
        chunk_url = f"{level_url}/{chunk_key}"
        target_key = f"{level_prefix}/{chunk_key}"

        try:
            raw = _fetch_url(chunk_url)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # Missing chunk (sparse array), skip
                return True
            raise

        # Decompress source encoding
        flat = _decompress_chunk(raw, compressor)

        # Determine actual chunk dimensions (handle edge chunks)
        cz_size = min(chunks[0], shape[0] - cz * chunks[0])
        cy_size = min(chunks[1], shape[1] - cy * chunks[1])
        cx_size = min(chunks[2], shape[2] - cx * chunks[2])
        expected = cz_size * cy_size * cx_size

        if flat.size < expected:
            # Pad if needed
            padded = np.zeros(expected, dtype=np.uint8)
            padded[: flat.size] = flat[:expected]
            flat = padded
        else:
            flat = flat[:expected]

        volume = flat.reshape(cz_size, cy_size, cx_size)

        # Encode with H264
        encoded = encode_h264(volume, qp=job.qp)

        # Upload to S3
        _s3_upload(s3, job.target_bucket, target_key, encoded)

        job.progress_done += 1
        return True

    # Process chunks in parallel
    processed = 0
    with ThreadPoolExecutor(max_workers=job.workers) as pool:
        futures = {
            pool.submit(process_one_chunk, c): c for c in chunk_coords
        }
        for future in as_completed(futures):
            if job.cancel_event.is_set():
                break
            try:
                future.result()
                processed += 1
            except Exception:
                coord = futures[future]
                log.error(f"Chunk {coord} failed: {traceback.format_exc()}")
                processed += 1  # count as attempted

    return processed


def run_job(job: Job):
    """Main entry point: process a rechunk job."""
    job.status = JobStatus.RUNNING
    log.info(f"Starting job {job.job_id}: {job.source_url} -> "
             f"s3://{job.target_bucket}/{job.target_prefix}")

    try:
        s3 = boto3.client("s3")
        source_url = job.source_url.rstrip("/")

        # Discover pyramid levels
        levels = _discover_levels(source_url)
        log.info(f"Found {len(levels)} level(s): {levels}")

        # Count total chunks across all levels
        total = 0
        level_infos = []
        for level in levels:
            level_url = f"{source_url}/{level}" if level else source_url
            try:
                zarray = _fetch_json(f"{level_url}/.zarray")
                shape = zarray["shape"]
                chunks = zarray["chunks"]
                n = 1
                for s, c in zip(shape, chunks):
                    n *= (s + c - 1) // c
                total += n
                level_infos.append((level, n))
            except Exception as e:
                log.warning(f"Could not read .zarray for level {level}: {e}")

        job.progress_total = total

        # Write root .zgroup metadata
        if len(levels) > 1 or (len(levels) == 1 and levels[0] != ""):
            zgroup = {"zarr_format": 2}
            _s3_upload(s3, job.target_bucket, f"{job.target_prefix}/.zgroup",
                       json.dumps(zgroup, indent=2).encode())
            # Copy root .zattrs
            try:
                zattrs = _fetch_url(f"{source_url}/.zattrs")
                _s3_upload(s3, job.target_bucket,
                           f"{job.target_prefix}/.zattrs", zattrs)
            except Exception:
                pass

        # Process each level
        offset = 0
        for level in levels:
            if job.cancel_event.is_set():
                break
            _process_level(job, s3, source_url, level,
                          job.target_prefix, offset)

        if job.cancel_event.is_set():
            job.status = JobStatus.CANCELLED
            log.info(f"Job {job.job_id} cancelled")
        else:
            job.status = JobStatus.COMPLETED
            log.info(f"Job {job.job_id} completed: "
                     f"{job.progress_done}/{job.progress_total} chunks")

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        log.error(f"Job {job.job_id} failed: {traceback.format_exc()}")
