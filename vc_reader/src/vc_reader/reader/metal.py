"""Apple GPU (Metal) adapter: oriented crops sampled on the GPU from a bounded chunk cache in unified memory

``ZarrReader.metal()`` returns a :class:`MetalReader`. It keeps decoded uint8 chunks of the source array in one
PyTorch MPS tensor (the chunk pool) and samples crops with a Metal kernel compiled at first use
(``torch.mps.compile_shader``), so no Metal toolchain is needed. Chunks come from the reader's own CPU path (its
memory and disk caches), so values match ``ZarrReader.read_affine_many``: trilinear, float32, with the array's
``fill_value`` for missing chunks and outside neighbours.

Requires PyTorch with MPS. Crops of one shape per call (``read_affine_batch``), as ``CudaReader.read_affine_batch``.
"""
from __future__ import annotations

from collections import OrderedDict
import itertools
import operator
import time

import numpy as np

from ._sampling import affine32

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
// The CPU sampler's arithmetic (sampling_math.hpp trilinear, dense_affine.hpp geometry): products rounded before
// each sum, no fused multiply-add, the same corner order and weights
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// One thread per output sample of B crops of D x H x W. A crop's source position is its integer base plus the
// float32 local position ((offset + axes[:, 0] i) + axes[:, 1] j) + axes[:, 2] k (affine32's split), so geometry
// keeps its precision far from the array origin.
// ints: D, H, W, Z, Y, X (array shape), CZ, CY, CX (chunk shape), GZ, GY, GX (chunk grid), B
kernel void affine_linear(device float* out [[buffer(0)]],
                          device const uchar* pool [[buffer(1)]],
                          device const int* table [[buffer(2)]],
                          device const int* bases [[buffer(3)]],
                          device const float* geometry [[buffer(4)]],
                          device const int* ints [[buffer(5)]],
                          constant float& fill [[buffer(6)]],
                          uint gid [[thread_position_in_grid]]) {
    const int D = ints[0], H = ints[1], W = ints[2];
    const int Z = ints[3], Y = ints[4], X = ints[5];
    const int CZ = ints[6], CY = ints[7], CX = ints[8];
    const int GY = ints[10], GX = ints[11], B = ints[12];
    const uint per = uint(D) * uint(H) * uint(W);
    const uint b = gid / per;
    if (b >= uint(B)) return;
    uint r = gid - b * per;
    const int i = int(r / uint(H * W)); r -= uint(i) * uint(H * W);
    const int j = int(r / uint(W));
    const int k = int(r - uint(j) * uint(W));
    device const float* g = geometry + b * 12;  // offset (3), then axes row-major: g[3 + 3 a + c]
    float frac[3];
    int lo[3];
    for (int a = 0; a < 3; ++a) {
        const float ti = g[3 + 3 * a] * float(i);
        const float tj = g[4 + 3 * a] * float(j);
        const float tk = g[5 + 3 * a] * float(k);
        const float q = ((g[a] + ti) + tj) + tk;
        const float f = floor(q);
        lo[a] = int(f) + bases[b * 3 + a];
        frac[a] = q - f;
    }
    const float g0 = 1.f - frac[0], g1 = 1.f - frac[1], g2 = 1.f - frac[2];
    const float w[4] = {g0 * g1, frac[0] * g1, g0 * frac[1], frac[0] * frac[1]};
    const ulong chunk_size = ulong(CZ) * ulong(CY) * ulong(CX);
    float value = 0.f;
    for (int c = 0; c < 8; ++c) {  // corner bit 0: +z, bit 1: +y, bit 2: +x
        const int pz = lo[0] + (c & 1), py = lo[1] + ((c >> 1) & 1), px = lo[2] + ((c >> 2) & 1);
        float v = fill;
        if (pz >= 0 && py >= 0 && px >= 0 && pz < Z && py < Y && px < X) {
            const int qz = pz / CZ, qy = py / CY, qx = px / CX;
            const int slot = table[(qz * GY + qy) * GX + qx];
            if (slot >= 0) {
                const int lz = pz - qz * CZ, ly = py - qy * CY, lx = px - qx * CX;
                v = float(pool[ulong(slot) * chunk_size + ulong((lz * CY + ly) * CX + lx)]);
            }
        }
        const float weight = w[c & 3] * ((c >> 2) & 1 ? frac[2] : g2);
        value += weight * v;
    }
    out[gid] = value;
}
'''

_LIBRARY = None
_SLACK = 1e-3  # voxels; far above the float32 geometry's error at crop sizes in the hundreds


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        import torch
        _LIBRARY = torch.mps.compile_shader(_SOURCE)
    return _LIBRARY


def is_available():
    """True when PyTorch can run Metal kernels on an MPS device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader'))


class MetalReader:
    """Create with reader.metal(). Crops come back as MPS float32 tensors.

    Holds at most ``cache_bytes`` of decoded source chunks on the GPU, least recently used first out. A batch whose
    crops touch more chunks than the cache holds raises ValueError. ``stats`` counts uploads, hits and the seconds
    spent reading missing chunks from the source reader.
    """

    def __init__(self, reader, *, cache_bytes=1 << 30, device='mps'):
        import torch
        reader._ready()
        if reader.ndim != 3 or reader.dtype != np.uint8:
            raise TypeError('Metal sampling supports uint8 ZYX arrays')
        if not is_available():
            raise RuntimeError('Metal reader unavailable: needs PyTorch with MPS and torch.mps.compile_shader')
        self.reader, self.device = reader, torch.device(device)
        self.shape = tuple(int(s) for s in reader.shape)
        self.chunks = tuple(int(c) for c in reader.chunks)
        self.grid = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))
        chunk_bytes = int(np.prod(self.chunks))
        self.slots = operator.index(cache_bytes) // chunk_bytes
        if self.slots < 1:
            raise ValueError(f'cache_bytes must hold at least one {chunk_bytes}-byte chunk')
        fill = reader.fill_value
        self.fill = float(0 if fill is None else fill)
        self.pool = torch.empty((self.slots, chunk_bytes), dtype=torch.uint8, device=self.device)
        self.table = torch.full(self.grid, -1, dtype=torch.int32, device=self.device)
        self.resident = OrderedDict()  # chunk (z, y, x) -> slot, least recently used first
        self.free = list(range(self.slots-1, -1, -1))
        self.counts = dict(hits=0, uploads=0, uploaded_bytes=0, read_seconds=0.)
        self._ints = None

    @property
    def stats(self):
        return dict(self.counts, resident=len(self.resident), slots=self.slots)

    def clear_cache(self):
        self.table.fill_(-1)
        self.resident.clear()
        self.free = list(range(self.slots-1, -1, -1))

    def _chunk_ranges(self, shape, bases, offsets, axes):
        """Per crop, the chunk index range (lo, hi inclusive) of the source box its trilinear taps can read."""
        corners = np.array(list(itertools.product(*[(0, n-1) for n in shape])), np.float64)  # (8, 3)
        points = offsets[:, None, :]+np.einsum('brc,kc->bkr', axes, corners)
        # the shader's float32 geometry can put a corner a few ulps outside this float64 box (127.99998 where the
        # box has 128.00000002, so the shader reads voxel 127), so the box is widened to keep those taps resident
        lo = bases+np.floor(points.min(1)-_SLACK).astype(np.int64)
        hi = bases+np.floor(points.max(1)+_SLACK).astype(np.int64)+1
        size = np.array(self.shape)
        inside = ((hi >= 0) & (lo <= size-1)).all(1)  # crops wholly outside the array read only fill
        lo, hi = np.clip(lo, 0, size-1), np.clip(hi, 0, size-1)
        chunk = np.array(self.chunks)
        return lo//chunk, hi//chunk, inside

    def _ensure(self, needed):
        """Make every chunk of ``needed`` (list of (z, y, x)) resident, evicting the least recently used others."""
        import torch
        missing = []
        for key in needed:
            if key in self.resident:
                self.resident.move_to_end(key)
                self.counts['hits'] += 1
            else:
                missing.append(key)
        if not missing:
            return
        if len(needed) > self.slots:
            raise ValueError(f'a batch touches {len(needed)} chunks; the Metal cache holds {self.slots}')
        protected = set(needed)
        slots = []
        for _ in missing:
            if not self.free:
                victim = next(k for k in self.resident if k not in protected)
                self.free.append(self.resident.pop(victim))
                self.table[victim] = -1
            slots.append(self.free.pop())
        chunk = self.chunks
        selections = []
        for key in missing:
            start = [k*c for k, c in zip(key, chunk)]
            selections.append(tuple(slice(s, min(s+c, n)) for s, c, n in zip(start, chunk, self.shape)))
        start = time.perf_counter()
        try:
            data = self.reader.read_many(selections)
        except BaseException:
            self.free.extend(reversed(slots))  # the slots hold nothing yet, so a retry can take them again
            raise
        self.counts['read_seconds'] += time.perf_counter()-start
        host = np.full((len(missing), *chunk), self.fill, np.uint8)
        for row, values in enumerate(data):
            host[row, :values.shape[0], :values.shape[1], :values.shape[2]] = values
        # one upload, then a slice copy per run of consecutive slots: index_copy_ on the whole pool is far slower
        staging = torch.from_numpy(host.reshape(len(missing), -1)).to(self.device)
        start = 0
        for end in range(1, len(slots)+1):
            if end == len(slots) or slots[end] != slots[end-1]+1:
                self.pool[slots[start]:slots[end-1]+1].copy_(staging[start:end])
                start = end
        keys = np.array(missing, np.int64)
        self.table[tuple(torch.from_numpy(keys[:, a]).to(self.device) for a in range(3))] = \
            torch.tensor(slots, dtype=torch.int32).to(self.device)
        for key, slot in zip(missing, slots):
            self.resident[key] = slot
        self.counts['uploads'] += len(missing)
        self.counts['uploaded_bytes'] += len(missing)*int(np.prod(chunk))

    def read_affine_batch(self, shape, origins, axes, *, order=1, out=None):
        """Crops (B, *shape) float32 on the MPS device: ``source_zyx = origins[b] + axes[b] @ [i, j, k]`` in this
        array's voxels (``ZarrReader.read_affine`` semantics, trilinear). ``out``: an MPS float32 tensor of that
        shape to write into."""
        import torch
        if order != 1:
            raise ValueError('Metal crops support linear sampling (order=1) only')
        shape = tuple(operator.index(s) for s in shape)
        origins, axes = np.asarray(origins, np.float64), np.asarray(axes, np.float64)
        count = len(origins)
        if origins.shape != (count, 3) or axes.shape != (count, 3, 3) or len(shape) != 3:
            raise ValueError('origins must be (B, 3), axes (B, 3, 3) and shape three sizes')
        if out is None:
            out = torch.empty((count, *shape), dtype=torch.float32, device=self.device)
        elif tuple(out.shape) != (count, *shape) or out.dtype != torch.float32 or out.device.type != 'mps' \
                or not out.is_contiguous():
            raise ValueError(f'out must be a contiguous MPS float32 tensor of shape {(count, *shape)}')
        if not count or not int(np.prod(shape)):
            return out
        # the CPU sampler's split of each origin into an integer base and a float32 local origin (affine32), so both
        # run the same float32 geometry
        split = [affine32(shape, o, a) for o, a in zip(origins, axes)]
        bases = np.array([s[1] for s in split], np.int64)
        offsets = np.stack([s[2] for s in split]).astype(np.float64)
        axes = np.stack([s[3] for s in split]).astype(np.float64)
        lo, hi, inside = self._chunk_ranges(shape, bases, offsets, axes)
        needed = set()
        for a, b in zip(lo[inside], hi[inside]):
            needed.update(itertools.product(*(range(int(x), int(y)+1) for x, y in zip(a, b))))
        self._ensure(sorted(needed))
        if np.abs(bases).max() >= 2**31-2**20:
            raise ValueError('crop origins are outside the int32 range')
        geometry = np.concatenate([offsets, axes.reshape(count, 9)], 1).astype(np.float32)
        ints = torch.tensor([*shape, *self.shape, *self.chunks, *self.grid, count], dtype=torch.int32)
        _library().affine_linear(out, self.pool, self.table, torch.from_numpy(bases.astype(np.int32)).to(self.device),
                                 torch.from_numpy(geometry).to(self.device), ints.to(self.device), self.fill,
                                 threads=count*int(np.prod(shape)))
        if not inside.all():
            # a crop wholly outside the array is fill, as on the CPU; the kernel's float32 sum of eight fill taps
            # would add rounding noise that a crop z-score turns into signal
            out[torch.from_numpy(np.flatnonzero(~inside)).to(self.device)] = self.fill
        return out
