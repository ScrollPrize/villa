"""128^3 uint8 CT windows read over HTTPS from a public zarr v2 array, optionally box-averaged to a coarser voxel.

`Window` reads a level whose voxels already match the model (7.5 to 10 um). `ResampledWindow` reads a finer level
and box-averages it: each output voxel is the overlap-weighted mean of the source voxels its cell covers, one axis at
a time, with voxel centers at index times voxel size on both grids, so a point at source voxel x lies at
x * src_um / dst_um in the resampled frame. The array must be uint8 with 128^3 chunks (raw or with a numcodecs
compressor such as blosc); chunks outside the array, or missing (HTTP 403 or 404), read as zeros.
"""
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request

import numpy as np

CHUNK = 128


class Window:
    def __init__(self, url, keep=256):
        self.url = url.rstrip("/")
        raw = self._get(".zarray")
        if raw is None:
            raise SystemExit(f"{self.url}/.zarray: not found")
        meta = json.loads(raw)
        self.shape = meta["shape"]
        if meta.get("dtype") not in ("|u1", "uint8") or list(meta.get("chunks", [])) != [CHUNK] * 3:
            raise SystemExit(f"{self.url}: need uint8 128^3 chunks, got {meta.get('dtype')} {meta.get('chunks')}")
        self.sep = meta.get("dimension_separator", ".")
        self.codec = None
        if meta.get("compressor"):
            from numcodecs import get_codec
            self.codec = get_codec(meta["compressor"])
        self.cache, self.keep = {}, keep
        self.fetched = 0

    def _get(self, key, tries=4):
        for k in range(tries):
            try:
                with urllib.request.urlopen(f"{self.url}/{key}", timeout=60) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    return None
                if k == tries - 1:
                    raise
            except Exception:
                if k == tries - 1:
                    raise
            time.sleep(2 ** k)

    def _decode(self, raw):
        if raw is None:
            return np.zeros((CHUNK,) * 3, np.uint8)
        if self.codec is not None:
            raw = self.codec.decode(raw)
        return np.frombuffer(raw, np.uint8).reshape((CHUNK,) * 3)

    def _inside(self, c):
        return all(0 <= q < -(-s // CHUNK) for q, s in zip(c, self.shape))

    def _fetch(self, c):
        if not self._inside(c):
            return None
        self.fetched += 1
        return self._get(self.sep.join(map(str, c)))

    def read(self, lo, size=128):
        """The box [lo, lo + size) in zyx voxels (size: one int or three); voxels outside the array read as 0."""
        lo = np.asarray(lo, int)
        size = np.broadcast_to(np.asarray(size, int), (3,))
        hi = lo + size
        c0, c1 = lo // CHUNK, (hi - 1) // CHUNK
        need = [(z, y, x) for z in range(c0[0], c1[0] + 1) for y in range(c0[1], c1[1] + 1)
                for x in range(c0[2], c1[2] + 1)]
        miss = [c for c in need if c not in self.cache]
        with cf.ThreadPoolExecutor(8) as ex:
            for c, raw in zip(miss, ex.map(self._fetch, miss)):
                self.cache[c] = self._decode(raw)
        out = np.zeros(tuple(size), np.uint8)
        for c in need:
            ch = self.cache[c]
            o = np.asarray(c) * CHUNK
            a0, a1 = np.maximum(lo, o), np.minimum(hi, o + CHUNK)
            out[tuple(slice(s - l, e - l) for s, e, l in zip(a0, a1, lo))] = \
                ch[tuple(slice(s - q, e - q) for s, e, q in zip(a0, a1, o))]
        while len(self.cache) > max(self.keep, len(need)):
            self.cache.pop(next(iter(self.cache)))
        return out


def box_weights(lo, size, ratio):
    """1D box (area) resampling from a fine grid to one ratio times coarser (ratio = dst_um / src_um >= 1).

    Output voxel j covers [(j - 0.5) r, (j + 0.5) r] in source voxels and source voxel i covers [i - 0.5, i + 0.5].
    Returns (i0, W): W[j - lo, i - i0] is the share of output voxel j's length that source voxel i covers, for j in
    [lo, lo + size); every row sums to 1."""
    r = float(ratio)
    if r < 1:
        raise ValueError(f"box_weights only coarsens (ratio {r} < 1)")
    j = np.arange(lo, lo + size, dtype=float)
    a, b = (j - 0.5) * r, (j + 0.5) * r
    i0 = int(np.floor(a[0] + 0.5))
    i1 = int(np.ceil(b[-1] - 0.5))
    i = np.arange(i0, i1 + 1, dtype=float)
    w = np.clip(np.minimum(b[:, None], i[None] + 0.5) - np.maximum(a[:, None], i[None] - 0.5), 0, None) / r
    return i0, w


def resampled_shape(src_shape, ratio):
    """Voxels of the coarse grid whose centers lie inside the source array's extent [-0.5, S - 0.5]."""
    return [int(np.floor((s - 0.5) / ratio)) + 1 for s in src_shape]


class ResampledWindow:
    def __init__(self, url, src_um, dst_um, keep=512):
        self.src_um, self.dst_um = float(src_um), float(dst_um)
        self.ratio = self.dst_um / self.src_um
        if self.ratio < 1:
            raise SystemExit(f"--resample {src_um}:{dst_um}: only a coarser voxel is supported")
        self.src = Window(url, keep=keep)
        self.url = self.src.url
        self.shape = resampled_shape(self.src.shape, self.ratio)

    def read(self, lo, size=128):
        ws = [box_weights(int(v), size, self.ratio) for v in lo]
        slo, ssize = [w[0] for w in ws], [w[1].shape[1] for w in ws]
        wz, wy, wx = (w[1].astype(np.float32) for w in ws)
        v = self.src.read(slo, ssize).astype(np.float32)
        t = np.tensordot(wz, v, axes=(1, 0))            # (z', y, x)
        t = np.tensordot(wy, t, axes=(1, 1))            # (y', z', x)
        t = np.tensordot(wx, t, axes=(1, 2))            # (x', y', z')
        return np.clip(np.rint(t.transpose(2, 1, 0)), 0, 255).astype(np.uint8)


def parse_resample(text):
    """'SRC_UM:DST_UM' to (src, dst) floats, or None for an empty value."""
    if not text:
        return None
    try:
        src, dst = (float(x) for x in text.split(":"))
    except ValueError:
        raise SystemExit(f"--resample {text}: need SRC_UM:DST_UM, such as 3.24:7.91")
    if not (0 < src <= dst):
        raise SystemExit(f"--resample {text}: need 0 < SRC_UM <= DST_UM")
    return src, dst


def open_volume(url, resample=None):
    rs = parse_resample(resample)
    return Window(url) if rs is None else ResampledWindow(url, *rs)
