"""Minimal streaming OME-Zarr (NGFF 0.4, Zarr v2) writer for reconstructed volumes.

* one multiscale image per group, axes z, y, x in micrometers;
* level 0 is written slab by slab as slices arrive (any z granularity: full z-chunks are flushed);
* optional extra levels (2x mean pooling) are built when the writer is closed;
* chunks are stored raw ("compressor": null), zlib-compressed, or (uint8) lossless vc-delta3d
  ({"id": "vc-delta3d", "quant": 1}, the vc_reader / VC3D codec), written by a thread pool;
* chunks that are entirely the fill value (0) are not written (readers return the fill value);
* coordinateTransformations place the array in the full volume: scale = voxel size,
  translation = origin (z0, y0, x0) * voxel size (+ pooling offset for coarser levels).

No dependency beyond numpy; readable by zarr-python, tensorstore, napari, VC3D...
"""
from __future__ import annotations

import json
import os
import shutil
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class OmeZarrWriter:
    def __init__(self, path, shape, dtype, chunks=(32, 256, 256), voxel_size_um=1.0, origin_zyx=(0, 0, 0),
                 levels=1, compression="none", zlib_level=1, name=None, attrs=None, overwrite=False, threads=16,
                 skip_empty=True, delta3d_preset="fast"):
        if os.path.exists(path):
            if not overwrite:
                raise FileExistsError(f"{path} exists")
            shutil.rmtree(path)
        if compression not in ("none", "zlib", "vc-delta3d"):
            raise ValueError("compression must be 'none', 'zlib' or 'vc-delta3d'")
        if compression == "vc-delta3d":
            from . import _core
            if np.dtype(dtype) != np.uint8:
                raise ValueError("vc-delta3d output is uint8 only")
            if not _core.delta3d_available():
                raise RuntimeError("helirec was built without vc-delta3d (HELIREC_DELTA3D=OFF)")
            self._core = _core
        self.path = path
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.chunks = tuple(int(min(c, s)) for c, s in zip(chunks, self.shape))
        self.levels = max(1, int(levels))
        self.compression = compression
        self.zlib_level = zlib_level
        self.skip_empty = skip_empty
        self.delta3d_preset = delta3d_preset
        self.n_written = self.n_skipped = 0
        self.voxel = float(voxel_size_um)
        self.origin = tuple(int(o) for o in origin_zyx)
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.pending = []
        self.buf = {}            # z-chunk index -> (array, filled mask over z)
        os.makedirs(path)
        self._write_json(".zgroup", {"zarr_format": 2})
        self.level_shapes = [self.shape]
        for _ in range(1, self.levels):
            self.level_shapes.append(tuple((s + 1) // 2 for s in self.level_shapes[-1]))
        for l, shp in enumerate(self.level_shapes):
            self._write_array_meta(l, shp)
        ms = {
            "version": "0.4",
            "name": name or os.path.basename(path.rstrip("/")),
            "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
            "datasets": [{"path": str(l), "coordinateTransformations": [
                {"type": "scale", "scale": [self.voxel * 2 ** l] * 3},
                {"type": "translation", "translation": [(o + (2 ** l - 1) / 2) * self.voxel for o in self.origin]}]}
                for l in range(self.levels)],
            "type": "mean",
        }
        self._write_json(".zattrs", {"multiscales": [ms], **(attrs or {})})

    # ------------------------------------------------------------------ metadata
    def _write_json(self, rel, obj):
        with open(os.path.join(self.path, rel), "w") as fh:
            json.dump(obj, fh, indent=2)

    def _write_array_meta(self, level, shape):
        os.makedirs(os.path.join(self.path, str(level)), exist_ok=True)
        self._write_json(f"{level}/.zarray", {
            "zarr_format": 2, "shape": list(shape), "chunks": list(self.chunks), "dtype": self.dtype.str,
            "compressor": {"none": None, "zlib": {"id": "zlib", "level": self.zlib_level},
                           "vc-delta3d": {"id": "vc-delta3d", "quant": 1}}[self.compression],
            "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/"})

    # ------------------------------------------------------------------ chunk io
    def _chunk_path(self, level, iz, iy, ix):
        return os.path.join(self.path, str(level), str(iz), str(iy), str(ix))

    def _encode(self, a):
        a = np.ascontiguousarray(a, self.dtype)
        if self.compression == "vc-delta3d":
            return self._core.delta3d_compress(a, self.delta3d_preset)
        b = a.tobytes()
        return b if self.compression == "none" else zlib.compress(b, self.zlib_level)

    def _decode(self, raw):
        if self.compression == "vc-delta3d":
            return self._core.delta3d_decompress(raw, list(self.chunks))
        if self.compression == "zlib":
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, self.dtype).reshape(self.chunks)

    def _store_chunk(self, level, iz, iy, ix, block):
        cz, cy, cx = self.chunks
        if block.shape != (cz, cy, cx):  # edge chunks are stored full size, padded with the fill value
            full = np.zeros((cz, cy, cx), self.dtype)
            full[:block.shape[0], :block.shape[1], :block.shape[2]] = block
            block = full
        p = self._chunk_path(level, iz, iy, ix)
        if self.skip_empty and not block.any():   # all fill value: do not store
            if os.path.exists(p):
                os.remove(p)
            self.n_skipped += 1
            return
        self.n_written += 1
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = self._encode(block)
        with open(p, "wb") as fh:
            fh.write(data)

    def _store_slab(self, level, iz, slab):
        """slab: (<= cz, Y, X) for z-chunk iz of `level`."""
        cz, cy, cx = self.chunks
        Y, X = slab.shape[1:]
        for iy in range(0, (Y + cy - 1) // cy):
            for ix in range(0, (X + cx - 1) // cx):
                block = np.array(slab[:, iy * cy:(iy + 1) * cy, ix * cx:(ix + 1) * cx])
                self.pending.append(self.pool.submit(self._store_chunk, level, iz, iy, ix, block))
        self._drain(limit=64)

    def _drain(self, limit=0):
        while len(self.pending) > limit:
            self.pending.pop(0).result()

    def read(self, level, z0, z1):
        """Read back [z0, z1) of a level (whole y, x)."""
        cz, cy, cx = self.chunks
        Z, Y, X = self.level_shapes[level]
        out = np.zeros((z1 - z0, Y, X), self.dtype)
        for iz in range(z0 // cz, (z1 - 1) // cz + 1):
            for iy in range((Y + cy - 1) // cy):
                for ix in range((X + cx - 1) // cx):
                    p = self._chunk_path(level, iz, iy, ix)
                    if not os.path.exists(p):
                        continue
                    blk = self._decode(open(p, "rb").read())
                    a, b = max(z0, iz * cz), min(z1, (iz + 1) * cz)
                    ys, xs = iy * cy, ix * cx
                    h, w = min(cy, Y - ys), min(cx, X - xs)
                    out[a - z0:b - z0, ys:ys + h, xs:xs + w] = blk[a - iz * cz:b - iz * cz, :h, :w]
        return out

    # ------------------------------------------------------------------ public
    def write(self, z0, data):
        """Write slices [z0, z0 + len(data)) of level 0 (any order / granularity)."""
        data = np.asarray(data)
        Z, Y, X = self.shape
        if data.shape[1:] != (Y, X) or z0 < 0 or z0 + data.shape[0] > Z:
            raise ValueError(f"slab {data.shape} at z={z0} does not fit {self.shape}")
        cz = self.chunks[0]
        for k in range(data.shape[0]):
            z = z0 + k
            iz, oz = divmod(z, cz)
            if iz not in self.buf:
                nz = min(cz, Z - iz * cz)
                self.buf[iz] = [np.zeros((nz, Y, X), self.dtype), np.zeros(nz, bool)]
            arr, filled = self.buf[iz]
            arr[oz] = data[k]
            filled[oz] = True
            if filled.all():
                self._store_slab(0, iz, arr)
                del self.buf[iz]

    def close(self):
        for iz, (arr, filled) in sorted(self.buf.items()):
            self._store_slab(0, iz, arr)  # unfilled slices stay at the fill value
        self.buf.clear()
        self._drain()
        cz = self.chunks[0]
        for l in range(1, self.levels):
            Zp = self.level_shapes[l - 1][0]
            Z, Y, X = self.level_shapes[l]
            for iz in range((Z + cz - 1) // cz):
                a, b = iz * cz * 2, min(Zp, (iz + 1) * cz * 2)
                src = self.read(l - 1, a, b).astype(np.float64)
                src = np.pad(src, [(0, (-s) % 2) for s in src.shape], mode="edge")
                ds = src.reshape(src.shape[0] // 2, 2, src.shape[1] // 2, 2, src.shape[2] // 2, 2).mean((1, 3, 5))
                if self.dtype.kind in "ui":
                    ds = np.rint(ds)
                self._store_slab(l, iz, ds.astype(self.dtype)[:, :Y, :X])
            self._drain()
        self.pool.shutdown()


def to_uint8(img, lo=-0.03, hi=0.145, outside=None):
    """float slice(s) -> uint8 with the window [lo, hi] (default: the reference uint8 window).
    `outside`: optional boolean (y, x) mask of pixels without data (outside the reconstruction disk),
    set to 0 (so empty chunks stay empty)."""
    u8 = np.rint(np.clip((np.asarray(img, np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if outside is not None:
        u8[..., outside] = 0
    return u8


def outside_disk(n, roi=None):
    """Pixels of an n x n slice (cropped to roi = (x0, x1, y0, y1)) outside GHBP's reconstruction
    disk (x-c)^2 + (y-c)^2 > (n//2)^2, c = (n-1)/2."""
    x0, x1, y0, y1 = roi if roi else (0, n, 0, n)
    c = (n - 1) / 2
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (xx - c) ** 2 + (yy - c) ** 2 > (n // 2) ** 2
