"""Shared helpers for the open-data scripts (``prepare_unseen.py``, ``prepare_scroll.py``): fetching from the
vesuvius-challenge-open-data bucket, finding a segment's tifxyz, and mirroring zarr chunks locally."""
from __future__ import annotations

import http.client
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

ROOT = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'


def fetch(url: str, tries: int = 6) -> bytes | None:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # absent chunk (masked or empty volume region)
                return None
            if attempt == tries-1:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException):
            if attempt == tries-1:
                raise
        time.sleep(2**attempt)  # dropped connections and throttling: back off and retry


def segment_path(segment: dict, segment_type: str, volume_id: str) -> str | None:
    for entry in segment.get('data') or []:
        if entry['type'] != segment_type:
            continue
        for origin in entry['origins']:
            if segment_type != 'tifxyz-transformed' or f'-on-{volume_id}-' in origin['path']:
                return origin['path'].rstrip('/')
    return None


def read_segment(path: str, coord_scale: float) -> tuple[np.ndarray, dict]:
    """A bucket tifxyz as (rows, cols, 3) float32 ``x, y, z`` times ``coord_scale`` (-1 kept for invalid nodes),
    and its meta.json with ``scale`` (grid nodes per voxel) converted to the scaled coordinates."""
    meta = json.loads(fetch(f'{ROOT}/{path}/meta.json'))
    xyz = np.stack([tifffile.imread(io.BytesIO(fetch(f'{ROOT}/{path}/{c}.tif'))).astype(np.float32) for c in 'xyz'], -1)
    xyz = np.where(xyz == -1, -1, xyz*coord_scale).astype(np.float32)
    meta['scale'] = [s/coord_scale for s in meta['scale']]
    return xyz, meta


def write_segment(dst: Path, xyz: np.ndarray, meta: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate('xyz'):
        tifffile.imwrite(dst/f'{c}.tif', np.ascontiguousarray(xyz[..., i]))
    (dst/'meta.json').write_text(json.dumps(meta))


def mirror_array(src: str, dst: Path, attrs: dict) -> dict:
    """Start a local zarr level ``dst`` mirroring the bucket array ``src`` (its .zarray, with ``attrs`` on the parent
    group); returns the .zarray."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst.parent/'.zgroup').write_text(json.dumps(dict(zarr_format=2)))
    (dst.parent/'.zattrs').write_text(json.dumps(attrs))
    zarray = json.loads(fetch(f'{src}/.zarray'))
    (dst/'.zarray').write_text(json.dumps(zarray))
    return zarray


def chunk_name(zarray: dict, key) -> str:
    return zarray.get('dimension_separator', '.').join(map(str, key))


def download_chunks(jobs: list, workers: int, done=None) -> list:
    """Fetch ``(url, dst)`` jobs into place (absent chunks write nothing); ``done(job, blob)`` is called with each
    result (``blob`` None if absent). Returns the byte counts (0 for absent)."""
    def get(job):
        url, dst = job
        blob = fetch(url)
        if blob is not None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix('.partial')
            tmp.write_bytes(blob)
            tmp.replace(dst)
        if done is not None:
            done(job, blob)
        return len(blob) if blob else 0
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(get, jobs))
