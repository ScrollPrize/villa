"""Oriented crops sampled on the GPU from a CT block already on the device: the backend for tracers whose crop
positions come from the model inside a training step (the sheet tracer's rollouts), next to the CPU sampler of the
data workers (crops.py).

Same crop definition (``CropSpec``) and frame convention (columns ``u, v, f``) as the CPU sampler; crops are laid out
``(D, H, W)`` = forward (f), v, u, trilinear with zeros outside the block (``grid_sample``, ``align_corners``). A block
is a ``(B, 1, D, H, W)`` tensor of one pyramid level with its level-voxel ``z, y, x`` origin; level voxel ``i`` is
centred on level-0 coordinate ``(i + 0.5) * 2**level - 0.5`` (mean-pooled pyramid).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

import numpy as np

from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import zscore_crops
from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, crop_local_grid


def crop_offsets(crop: CropSpec, device) -> torch.Tensor:
    """Local ``(u, v, f)`` offsets (D, H, W, 3) of the crop samples."""
    axis = lambda values: torch.as_tensor(values, dtype=torch.float32, device=device)
    f, v, u = torch.meshgrid(axis(crop.forward_coords), axis(crop.vertical_coords), axis(crop.lateral_coords),
                             indexing='ij')
    return torch.stack([u, v, f], -1)


def to_block_grid(points: torch.Tensor, origin: torch.Tensor, level: int, size) -> torch.Tensor:
    """Level-0 ``xyz`` points (B, ..., 3) -> grid_sample coordinates of a block with ``zyx`` origin (B, 3)."""
    scale = float(1 << level)
    idx = (points+.5)/scale-.5-origin.flip(-1).to(points.dtype).view(origin.shape[0], *([1]*(points.ndim-2)), 3)
    return 2*idx/(_extent(tuple(int(v) for v in size), points.device, points.dtype)-1)-1


_EXTENTS: dict = {}


def _extent(size, device, dtype) -> torch.Tensor:
    """Block extent as a cached device tensor: creating it per call is a host-to-device copy that stalls the GPU
    queue, and dividing by Python scalars instead would round differently (CUDA multiplies by the reciprocal)."""
    key = (size, device, dtype)
    if key not in _EXTENTS:
        if len(_EXTENTS) > 4096:  # blocks of their own sizes bring new sizes every batch
            _EXTENTS.clear()
        _EXTENTS[key] = torch.tensor([size[2], size[1], size[0]], device=device, dtype=dtype)
    return _EXTENTS[key]


def sample_block(volume, origin: torch.Tensor, level: int, points: torch.Tensor) -> torch.Tensor:
    """Trilinear samples (B, C, d1, d2, d3) at level-0 points (B, d1, d2, d3, 3) of a batch of blocks: one
    (B, C, D, H, W) tensor, or a list of B (C, D, H, W) blocks of their own sizes, each sampled through its own extent
    (same values; padding to a common size can cost far more memory than the blocks)."""
    if isinstance(volume, torch.Tensor):
        grid = to_block_grid(points, origin, level, volume.shape[2:])
        return F.grid_sample(volume, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return torch.cat([F.grid_sample(v[None], to_block_grid(points[i:i+1], origin[i:i+1], level, v.shape[1:]),
                                    mode='bilinear', padding_mode='zeros', align_corners=True)
                      for i, v in enumerate(volume)])


def oriented_crops(block, origin: torch.Tensor, level: int, pos: torch.Tensor, frames: torch.Tensor,
                   offsets: torch.Tensor, zscore: bool = True) -> torch.Tensor:
    """Crops (B, N, 1, D, H, W), z-scored per crop (``zscore_crops``; raw values with ``zscore=False``), at ``pos``
    (B, N, 3) with frames (B, N, 3, 3) of columns ``u, v, f``; ``offsets`` from ``crop_offsets``, ``block`` as in
    ``sample_block``."""
    B, N = pos.shape[:2]
    d, h, w = offsets.shape[:3]
    world = (pos[:, :, None, None, None]+torch.einsum('bnij,dhwj->bndhwi', frames, offsets)).reshape(B, N*d, h, w, 3)
    crops = sample_block(block, origin, level, world).reshape(B, 1, N, d, h, w).transpose(1, 2)
    return zscore_crops(crops) if zscore else crops


def sample_plane(block: torch.Tensor, origin: torch.Tensor, level: int, center: torch.Tensor, axes: torch.Tensor,
                 extent: tuple[float, float], spacing: float) -> torch.Tensor:
    """Raw intensities on a plane (H, W) of one block (1, D, H, W) through ``center`` (3,) spanned by ``axes`` (2, 3)."""
    h = int(extent[0]/spacing)
    w = int(extent[1]/spacing)
    ii = (torch.arange(h, device=block.device)-(h-1)/2)*spacing
    jj = (torch.arange(w, device=block.device)-(w-1)/2)*spacing
    pts = center+ii[:, None, None]*axes[0]+jj[None, :, None]*axes[1]
    grid = to_block_grid(pts[None, None], origin[None], level, block.shape[1:])
    return F.grid_sample(block[None], grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0, 0]


def scalar_crops_gpu(items, vol, crop: CropSpec, pool=None, *, device='cuda') -> torch.Tensor:
    """GPU counterpart of ``crops.scalar_crops`` (normalized): (B, 1, D, H, W) float32 crops on ``device``.

    Each row's tight source block (``crops.read_tight_blocks``, read on the CPU) is sampled on the GPU: the same
    source-array points (``pos*scale + local @ (frame*scale).T``, computed in float64), trilinear with zeros outside the
    block, CT/255, then ``zscore_crops`` per crop. Values agree with the CPU sampler up to float32 rounding of the
    interpolation weights.
    """
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import read_tight_blocks
    raw, starts = read_tight_blocks(items, vol, crop, pool)
    scale = float(getattr(vol, 'input_scale', 1.))
    local = torch.from_numpy(crop_local_grid(crop)).to(device, torch.float64)  # (D, H, W, 3) u, v, f
    pos = torch.from_numpy(np.stack([np.asarray(i['pos'], np.float64) for i in items])*scale).to(device)
    frame = torch.from_numpy(np.stack([np.asarray(i['frame'], np.float64) for i in items])*scale).to(device)
    world = pos[:, None, None, None]+torch.einsum('dhwj,bij->bdhwi', local, frame)  # source-array xyz
    origin = torch.from_numpy(np.stack(starts).astype(np.float64)).to(device)
    out = []
    for k, block in enumerate(raw):
        b = torch.from_numpy(np.ascontiguousarray(block)).to(device, non_blocking=True).float()/255.
        grid = to_block_grid(world[k:k+1], origin[k:k+1], 0, block.shape[1:]).float()
        out.append(F.grid_sample(b[None], grid, mode='bilinear', padding_mode='zeros', align_corners=True))
    return zscore_crops(torch.cat(out))


class GPUChunkCache:
    """CT chunks resident on the GPU (an LRU pool of chunk slots and a page table over the chunk grid) and oriented
    scalar crops sampled from them: ``crops.scalar_crops`` values (trilinear, zeros outside the array, CT/255, per-crop
    z-score) without assembling per-crop blocks on the CPU.

    Chunks come from the volume's ``ChunkedArray.chunk`` (a chunk absent from the store reads as zeros: slot 0). Each
    batch of rows first makes the chunks under its tight blocks resident, then gathers the eight trilinear corners
    through the page table. Sample positions are float32 offsets from an integer anchor per row and the interpolation
    weights float32 (the CPU kernel works in float64).
    """

    def __init__(self, vol, device='cuda', slots=4096, rows=16):
        self.vol, self.ct, self.device, self.rows = vol, vol.ct, torch.device(device), rows
        self.chunks = tuple(int(c) for c in self.ct.chunks)
        if any(c & (c-1) for c in self.chunks):
            raise ValueError(f'GPU chunk cache needs power-of-two chunks, not {self.chunks}')
        self.shifts = tuple(int(c).bit_length()-1 for c in self.chunks)
        self.size = int(np.prod(self.chunks))
        self.grid = tuple(-(-int(s)//c) for s, c in zip(self.ct.shape, self.chunks))
        self.pool = torch.zeros((slots+1)*self.size, dtype=torch.uint8, device=self.device)  # slot 0: zeros
        # 0 = zeros or not resident; int64, so slot offsets into the pool are int64 (coordinates stay int32)
        self.table = torch.zeros(self.grid, dtype=torch.int64, device=self.device)
        self.bounds = torch.tensor(self.ct.shape, device=self.device, dtype=torch.int32)
        self.resident = {}  # chunk key -> slot (0 for chunks absent from the store), in LRU order
        # Empty chunks occupy no slot and must not lengthen every eviction scan.
        # This is the same LRU order restricted to occupied slots.
        self.occupied = OrderedDict()
        self.free = list(range(slots, 0, -1))
        self.loads = 0
        self.grids = {}  # crop spec -> local (u, v, f) sample offsets on the device

    def grow_slots(self, slots):
        """Increase capacity between crop calls without invalidating resident slots."""
        current = self.pool.numel()//self.size-1
        if slots <= current:
            return
        pool = torch.empty((slots+1)*self.size, dtype=self.pool.dtype, device=self.device)
        pool[:self.pool.numel()].copy_(self.pool)
        self.pool = pool
        self.free = list(range(slots, current, -1))+self.free

    def _ensure(self, keys, pool=None):
        keys = sorted(keys)
        for k in keys:
            if k in self.resident:
                self.resident[k] = self.resident.pop(k)  # most recent last
                if self.resident[k]:
                    self.occupied.move_to_end(k)
        missing = [k for k in keys if k not in self.resident]
        if not missing:
            return
        need = set(keys)
        arrays = list(pool.map(self.ct.chunk, missing) if pool is not None else map(self.ct.chunk, missing))
        for k, a in zip(missing, arrays):
            if a is None or not a.any():
                self.resident[k] = 0
                continue
            if not self.free:
                victim = next((v for v in self.occupied if v not in need), None)
                if victim is None:
                    raise RuntimeError('GPU chunk cache too small for one row group; raise slots or lower rows')
                slot = self.resident.pop(victim)
                del self.occupied[victim]
                self.table[victim] = 0
                self.free.append(slot)
            slot = self.free.pop()
            self.pool[slot*self.size:(slot+1)*self.size] = torch.from_numpy(np.array(a).reshape(-1)).to(
                self.device, non_blocking=True)
            self.table[k] = slot
            self.resident[k] = slot
            self.occupied[k] = None
            self.loads += 1
        # resident-but-unused keys of absent chunks stay mapped to slot 0, like non-resident ones

    def _keys(self, items, crop, scale):
        from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import tight_block
        keys = set()
        ch = np.asarray(self.chunks)
        for it in items:
            start, size = tight_block(it['pos'], it['frame'], crop, scale)
            lo = np.clip(start//ch, 0, np.asarray(self.grid)-1)
            hi = np.clip((start+size-1)//ch, 0, np.asarray(self.grid)-1)
            keys.update((z, y, x) for z in range(lo[0], hi[0]+1) for y in range(lo[1], hi[1]+1)
                        for x in range(lo[2], hi[2]+1))
        return keys

    def _local(self, crop):
        if crop not in self.grids:
            self.grids[crop] = torch.from_numpy(crop_local_grid(crop).astype(np.float32)).to(self.device)
        return self.grids[crop]

    def crops(self, items, vol, crop: CropSpec, pool=None) -> torch.Tensor:
        scale = float(getattr(vol, 'input_scale', 1.))
        local = self._local(crop)
        out = torch.empty((len(items), 1)+tuple(local.shape[:3]), dtype=torch.float32, device=self.device)
        if not items:
            return zscore_crops(out)
        # Stage the small metadata arrays once for the whole call. Previously
        # three blocking .to() calls per row group serialized every gather.
        pos = np.stack([np.asarray(i['pos'], np.float64) for i in items])*scale
        anchor = np.floor(pos)
        frames = np.stack([np.asarray(i['frame'], np.float64) for i in items])*scale
        def transfer(array):
            tensor = torch.from_numpy(array)
            if self.device.type == 'cuda':
                tensor = tensor.pin_memory()
            return tensor.to(self.device, non_blocking=True)
        relatives = transfer((pos-anchor).astype(np.float32))
        frames = transfer(frames.astype(np.float32))
        anchors = transfer(anchor[:, ::-1].astype(np.int32).copy())
        for r0 in range(0, len(items), self.rows):
            part = items[r0:r0+self.rows]
            self._ensure(self._keys(part, crop, scale), pool)
            # float32 offsets from each row's integer anchor (consumer GPUs run float64 at a small fraction of float32)
            take = slice(r0, r0+len(part))
            out[take, 0] = _gather_trilinear(self.pool, self.table, anchors[take], relatives[take], frames[take], local, self.bounds,
                                                        self.shifts)
        return zscore_crops(out)


_GATHERS: dict = {}


def _gather_trilinear(pool, table, anchor, rel, frame, local, bounds, shifts):
    """The compiled gather for these chunk shifts (constants in the kernel)."""
    if shifts not in _GATHERS:
        _GATHERS[shifts] = torch.compile(_gather_kernel(*shifts), dynamic=True)
    return _GATHERS[shifts](pool, table, anchor, rel, frame, local, bounds)


def _gather_kernel(sz, sy, sx):
    mz, my, mx, total = (1 << sz)-1, (1 << sy)-1, (1 << sx)-1, sz+sy+sx

    def gather(pool, table, anchor, rel, frame, local, bounds):
        """Trilinear CT/255 of R oriented crops from chunk slots (zeros outside the array): sample ``(u, v, f)`` of
        ``local`` (D, H, W, 3) lies at array xyz ``anchor`` (R, 3 zyx, int32) + ``rel`` (R, 3) + ``frame`` (R, 3, 3) @ it,
        in float32 (offsets from the anchor stay small). Chunks are powers of two (``shifts``: log2 of z, y, x sizes);
        ``table`` is the flat (z, y, x) chunk grid."""
        gy, gx = table.shape[1], table.shape[2]
        flat_table = table.reshape(-1)
        view = lambda t: t.view(t.shape[0], 1, 1, 1)
        u, v, f = local[..., 0], local[..., 1], local[..., 2]
        offset = [view(rel[:, k])+view(frame[:, k, 0])*u+view(frame[:, k, 1])*v+view(frame[:, k, 2])*f for k in range(3)]
        acc = torch.zeros((rel.shape[0],)+tuple(local.shape[:3]), dtype=torch.float32, device=rel.device)
        base, frac = [], []
        for k, o in zip((0, 1, 2), offset[::-1]):  # z, y, x
            fl = torch.floor(o)
            frac.append(o-fl)
            base.append(fl.int()+view(anchor[:, k]))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    z, y, x = base[0]+dz, base[1]+dy, base[2]+dx
                    valid = (z >= 0) & (z < bounds[0]) & (y >= 0) & (y < bounds[1]) & (x >= 0) & (x < bounds[2])
                    z, y, x = torch.where(valid, z, 0), torch.where(valid, y, 0), torch.where(valid, x, 0)
                    slot = flat_table[((z >> sz)*gy+(y >> sy))*gx+(x >> sx)]
                    inner = ((((z & mz) << sy) | (y & my)) << sx) | (x & mx)
                    value = pool[(slot << total)+inner].float()
                    w = ((frac[0] if dz else 1-frac[0])*(frac[1] if dy else 1-frac[1])*(frac[2] if dx else 1-frac[2]))
                    acc = acc+torch.where(valid, w*value, 0.)
        return acc/255.
    return gather
