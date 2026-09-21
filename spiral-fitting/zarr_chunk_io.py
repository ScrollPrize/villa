"""Shared read-only local Zarr v2 chunk helpers (including legacy VCZ1)."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np


def _load_delta3d(codec_id: str):
    try:
        import vc_delta3d
    except ImportError as exc:
        raise RuntimeError(
            f'the store uses the {codec_id} codec; install the vc-delta3d package'
        ) from exc
    return vc_delta3d


def _list_chunk_keys(array_dir: Path, separator: str) -> set[tuple[int, int, int]]:
    keys = set()
    if separator == '/':
        for path in glob.iglob(str(array_dir / '*' / '*' / '*')):
            parts = path.rsplit(os.sep, 3)[-3:]
            try:
                keys.add((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    else:
        for name in os.listdir(array_dir):
            parts = name.split('.')
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                keys.add((int(parts[0]), int(parts[1]), int(parts[2])))
    return keys


def _chunk_path(array_dir: Path, key: tuple[int, int, int], separator: str) -> Path:
    if separator == '/':
        return array_dir / str(key[0]) / str(key[1]) / str(key[2])
    return array_dir / f'{key[0]}.{key[1]}.{key[2]}'


def _make_chunk_reader(compressor: dict | None, chunk_voxels: int):
    if compressor is None:
        def read(path: Path) -> np.ndarray:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size != chunk_voxels:
                raise ValueError(f'{path}: raw chunk has {data.size} bytes, expected {chunk_voxels}')
            return data
        return read
    codec_id = compressor.get('id')
    if codec_id == 'vcz1':
        delta3d = _load_delta3d(codec_id)

        def read(path: Path) -> np.ndarray:
            with open(path, 'rb') as f:
                decoded = delta3d.decompress_with_magic(
                    f.read(), chunk_voxels, b'VCZ1')
            decoded = np.frombuffer(decoded, dtype=np.uint8)
            if decoded.size != chunk_voxels:
                raise ValueError(
                    f'{path}: decoded chunk has {decoded.size} bytes, '
                    f'expected {chunk_voxels}')
            return decoded
        return read
    if codec_id == 'vc-delta3d':
        _load_delta3d(codec_id)  # Import registers the numcodecs adapter.
    import numcodecs
    codec = numcodecs.get_codec(compressor)

    def read(path: Path) -> np.ndarray:
        with open(path, 'rb') as f:
            decoded = codec.decode(f.read())
        decoded = np.frombuffer(decoded, dtype=np.uint8)
        if decoded.size != chunk_voxels:
            raise ValueError(f'{path}: decoded chunk has {decoded.size} bytes, expected {chunk_voxels}')
        return decoded
    return read

