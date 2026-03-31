"""FastAPI service for rechunking zarr volumes with H264 encoding.

Also serves as a caching proxy: fetches chunks from S3 on first request,
encodes with H264 QP40, caches on local NVMe, serves from cache thereafter.
"""

import hashlib
import logging
import os
import uuid
from pathlib import Path
from threading import Lock, Thread

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from encoder import encode_h264
from worker import Job, JobStatus, run_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="rechunk-service", version="0.1.0")

# In-memory job store (fine for single-instance)
jobs: dict[str, Job] = {}


class RechunkRequest(BaseModel):
    source_url: str  # HTTP(S) URL to zarr root
    target_bucket: str  # S3 bucket for output
    target_prefix: str  # S3 key prefix for output
    codec: str = "h264"
    qp: int = 26
    chunk_size: list[int] | None = None
    workers: int = 16


@app.post("/rechunk")
def submit_rechunk(req: RechunkRequest):
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        job_id=job_id,
        source_url=req.source_url,
        target_bucket=req.target_bucket,
        target_prefix=req.target_prefix,
        codec=req.codec,
        qp=req.qp,
        chunk_size=req.chunk_size,
        workers=req.workers,
    )
    jobs[job_id] = job

    thread = Thread(target=run_job, args=(job,), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": job.status.value}


@app.get("/jobs")
def list_jobs():
    return [j.to_dict() for j in jobs.values()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    job.cancel_event.set()
    return {"job_id": job_id, "status": "cancelling"}


# ==========================================================================
# Chunk cache proxy — fetch from S3, encode H264, cache on NVMe
# ==========================================================================

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/mnt/nvme-cache/chunks"))
CACHE_QP = int(os.environ.get("CACHE_QP", "40"))
CACHE_MAX_GB = float(os.environ.get("CACHE_MAX_GB", "400"))
_cache_locks: dict[str, Lock] = {}
_global_lock = Lock()


def _cache_path(source_url: str, chunk_path: str) -> Path:
    """Deterministic local path for a cached chunk."""
    url_hash = hashlib.md5(source_url.encode()).hexdigest()[:12]
    return CACHE_DIR / url_hash / chunk_path


def _fetch_and_encode(source_url: str, chunk_path: str) -> bytes | None:
    """Fetch raw chunk from source, encode H264, return encoded bytes."""
    import numcodecs
    import numpy as np

    url = f"{source_url.rstrip('/')}/{chunk_path}"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    raw_compressed = resp.content

    # Try to detect zarr compression and decompress
    # Most zarr uses blosc or zstd
    try:
        codec = numcodecs.Blosc()
        raw = codec.decode(raw_compressed)
    except Exception:
        try:
            codec = numcodecs.Zstd()
            raw = codec.decode(raw_compressed)
        except Exception:
            # Already raw or unknown compression — use as-is
            raw = raw_compressed

    data = np.frombuffer(raw, dtype=np.uint8)

    # Assume cubic chunks — find the cube root
    size = len(data)
    dim = round(size ** (1 / 3))
    if dim * dim * dim != size:
        # Not cubic — just store raw
        return raw_compressed

    volume = data.reshape(dim, dim, dim)
    return encode_h264(volume, qp=CACHE_QP)


@app.get("/chunk/{source_hash}/{chunk_path:path}")
def get_chunk(source_hash: str, chunk_path: str):
    """Serve a cached chunk. Returns 404 if chunk doesn't exist at source."""
    cache_file = CACHE_DIR / source_hash / chunk_path

    # Fast path: already cached
    if cache_file.exists():
        return Response(content=cache_file.read_bytes(),
                       media_type="application/octet-stream")

    raise HTTPException(404, "chunk not cached — use /cache endpoint to populate")


@app.post("/cache")
def register_source(req: dict):
    """Register a source URL for caching. Returns the source hash to use in /chunk/ URLs."""
    source_url = req.get("source_url", "")
    if not source_url:
        raise HTTPException(400, "source_url required")
    url_hash = hashlib.md5(source_url.encode()).hexdigest()[:12]
    cache_dir = CACHE_DIR / url_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Store source URL mapping
    (cache_dir / ".source_url").write_text(source_url)
    return {"source_hash": url_hash, "cache_dir": str(cache_dir)}


@app.get("/proxy/{source_hash}/{chunk_path:path}")
def proxy_chunk(source_hash: str, chunk_path: str):
    """Fetch-through proxy: serve from cache or fetch+encode+cache on miss."""
    cache_file = CACHE_DIR / source_hash / chunk_path

    # Fast path: cached
    if cache_file.exists():
        return Response(content=cache_file.read_bytes(),
                       media_type="application/octet-stream")

    # Get source URL
    source_file = CACHE_DIR / source_hash / ".source_url"
    if not source_file.exists():
        raise HTTPException(404, "source not registered — POST /cache first")
    source_url = source_file.read_text().strip()

    # Lock per chunk to prevent duplicate fetches
    lock_key = f"{source_hash}/{chunk_path}"
    with _global_lock:
        if lock_key not in _cache_locks:
            _cache_locks[lock_key] = Lock()
        lock = _cache_locks[lock_key]

    with lock:
        # Double-check after acquiring lock
        if cache_file.exists():
            return Response(content=cache_file.read_bytes(),
                           media_type="application/octet-stream")

        encoded = _fetch_and_encode(source_url, chunk_path)
        if encoded is None:
            # Write empty marker so we don't re-fetch 404s
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(b"")
            raise HTTPException(404, "chunk not found at source")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(encoded)

        return Response(content=encoded,
                       media_type="application/octet-stream")


@app.get("/cache/stats")
def cache_stats():
    """Return cache disk usage."""
    if not CACHE_DIR.exists():
        return {"total_bytes": 0, "total_files": 0}
    total = 0
    count = 0
    for f in CACHE_DIR.rglob("*"):
        if f.is_file() and f.name != ".source_url":
            total += f.stat().st_size
            count += 1
    return {
        "total_bytes": total,
        "total_gb": round(total / (1024**3), 2),
        "total_files": count,
        "max_gb": CACHE_MAX_GB,
    }
