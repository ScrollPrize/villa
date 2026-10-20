"""Dense student inference over a level-0 CT array, written as uint8 codes on the level-1 grid (4.8 um).

Geometry (output voxel j <-> level-0 voxels [2j, 2j + 2) per axis; output shape = ceil(level-0 shape / 2)):
  crop origins a per axis: grid_axes(output size, 64), kept where the crop's level-0 parent256 [2a - 64, 2a + 192) lies
  inside the level-0 array. Crop a reads level-0 [2a, 2a + 128)^3, normalizes it with the p1 / p99 (np.percentile,
  linear) of its parent256, average-pools it to 64^3 and predicts the 64^3 logits of output voxels [a, a + 64).
  Crops whose level-0 128^3 region lies entirely in absent chunks are skipped; absent chunks read as 0.
Blend: logit = float32(sum_c float64(g * v_c) / sum_c float64(g)) with g = gaussian(64) (sigma 8 output voxels,
  maximum 1), crops in lexicographic order.
Code: 0 where the voxel's level-0 chunk (index j // 64) is absent or the voxel is outside crop coverage, else
  clip(floor(10 * float64(logit)) + 128, 1, 255). Code c in 2..254 <=> logit in [(c - 128) / 10, (c - 127) / 10).

Work is split into units (tiles of whole output chunks). A unit computes every crop overlapping its tile, in
batches of B crops (the last batch zero-padded), and writes the output chunks of its tile that contain present CT.
"""
import itertools
import hashlib
import math
import os
import queue
import re
import threading

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

PATCH = 64            # output voxels per crop edge (level-0 crop 128)
RING_ROWS = 6         # GPU input window, level-0 chunk rows
ACC_ROWS = 128        # accumulator ring, output rows (>= 64 + the largest crop step)
SLAB = 32             # output rows encoded per GPU -> CPU copy
N_PARENT = 256 ** 3
CODEC = dict(id='blosc', cname='zstd', clevel=1, shuffle=2, blocksize=0)


def grid_axes(size, patch=PATCH):
    count = math.ceil((int(size) - patch) / (patch // 2)) + 1
    s = np.array([round((int(size) - patch) / (count - 1) * i) if count > 1 else 0 for i in range(count)], np.int64)
    return s[(s >= 0) & (s + patch <= size)]


def output_shape(shape0):
    return tuple(int(v) for v in -(-np.asarray(shape0, np.int64) // 2))


def crop_grid(shape0):
    """Crop origins (output voxels) per axis whose level-0 parent256 lies inside the level-0 array."""
    shape1 = output_shape(shape0)
    out = []
    for k in range(3):
        a = grid_axes(shape1[k])
        out.append(a[(2 * a - 64 >= 0) & (2 * a + 192 <= shape0[k])])
    return out


def gaussian(patch=PATCH):
    g = np.zeros((patch,) * 3, np.float32)
    g[(patch // 2,) * 3] = 1
    g = gaussian_filter(g, [patch / 8] * 3, mode='constant', cval=0)
    return g / g.max()


def percentile_table(n=N_PARENT, q=(1., 99.)):
    """For np.percentile(..., method='linear') of n values: the order-statistic indices (below, above) of each
    quantile and the value table T[k, a, b] for order-statistic values a, b in 0..255 (float64, numpy's _lerp)."""
    virtual = (n - 1) * (np.asarray(q, np.float64) / 100)
    below = np.floor(virtual)
    gamma = virtual - below
    lo = np.clip(below.astype(np.intp), 0, n - 1)
    hi = np.clip(below.astype(np.intp) + 1, 0, n - 1)
    a = np.arange(256, dtype=np.float32)[:, None].astype(np.float64)
    b = np.arange(256, dtype=np.float32)[None, :].astype(np.float64)
    table = []
    for t in gamma:
        diff = b - a
        table.append(np.where(t >= .5, b - diff * (1 - t), a + diff * t))
    return lo, hi, np.stack(table)


class Percentiles:
    """(p1, p99) of uint8 parents from their 256-bin histograms, equal to np.percentile of the float32 values."""
    def __init__(self, device):
        lo, hi, table = percentile_table()
        self.table = torch.as_tensor(table, device=device)
        self.lo = torch.as_tensor(lo, dtype=torch.int64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.int64, device=device)

    def __call__(self, counts):
        cum = counts.cumsum(1)
        n = len(counts)
        a = torch.searchsorted(cum, self.lo.expand(n, 2).contiguous(), right=True)
        b = torch.searchsorted(cum, self.hi.expand(n, 2).contiguous(), right=True)
        q = torch.arange(2, device=counts.device).expand(n, 2)
        v = self.table[q, a, b]
        return v[:, 0], v[:, 1]


def normalize_batch(u8, lo, hi):
    """clip to [lo, hi], subtract lo, divide by hi - lo; each step in float64, rounded to float32 (zero if
    hi - lo < 1e-6). Equal to ct.normalize when lo and hi are float32 values (integers, for most parents)."""
    L, H = lo.view(-1, 1, 1, 1), hi.view(-1, 1, 1, 1)
    x = torch.clamp(u8.double(), L, H).float()
    x = (x.double() - L).float()
    x = (x.double() / (H - L)).float()
    return torch.where(((hi - lo) < 1e-6).view(-1, 1, 1, 1), torch.zeros((), device=x.device), x)


def encode(v):
    """float32 logits -> uint8 codes 1..255 (no mask)."""
    return torch.clamp(torch.floor(v.double() * 10) + 128, 1, 255).to(torch.uint8)


def ring(z0, z1, period):
    """[(ring slice, local slice)] covering rows z0..z1 of a ring buffer with `period` rows."""
    out, z = [], z0
    while z < z1:
        r = z % period
        n = min(z1 - z, period - r)
        out.append((slice(r, r + n), slice(z - z0, z - z0 + n)))
        z += n
    return out


def unit_plan(index, unit, shape0, grid, presence_window, chunk=256):
    """Crops and chunks of unit `index` (tile T0..T1 in output voxels). presence_window(c0, c1) returns the bool
    presence of level-0 chunks [c0, c1)."""
    shape1 = np.array(output_shape(shape0))
    T0 = np.asarray(index, np.int64) * unit
    T1 = np.minimum(T0 + unit, shape1)
    plan = dict(index=[int(v) for v in index], T0=T0.tolist(), T1=T1.tolist())
    ax = [g[(g < T1[k]) & (g + PATCH > T0[k])] for k, g in enumerate(grid)]
    if any(len(a) == 0 for a in ax):
        return dict(plan, empty='outside crop coverage', chunks=[])
    nchunks = -(-np.asarray(shape0, np.int64) // 128)
    c0 = np.minimum([(2 * a[0] - 64) // 128 for a in ax], T0 // 64)
    c1 = np.minimum(np.maximum([(2 * a[-1] + 191) // 128 + 1 for a in ax], -(-T1 // 64)), nchunks)
    pres = np.asarray(presence_window(c0, c1), bool)
    assert pres.shape == tuple(c1 - c0)
    lohi = [((2 * a // 128) - c0[k], ((2 * a + 127) // 128) - c0[k]) for k, a in enumerate(ax)]
    present_crop = np.zeros([len(a) for a in ax], bool)
    for bz, by, bx in itertools.product((0, 1), repeat=3):
        present_crop |= pres[np.ix_(lohi[0][bz], lohi[1][by], lohi[2][bx])]
    chunks = []
    for key in itertools.product(*[range(int(a), int(b)) for a, b in zip(T0 // chunk, -(-T1 // chunk))]):
        lo = np.array(key) * chunk // 64 - c0
        hi = -(-np.minimum((np.array(key) + 1) * chunk, shape1) // 64) - c0
        if pres[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
            chunks.append(tuple(int(v) for v in key))
    plan.update(c0=c0.tolist(), c1=c1.tolist(), axes=[a.tolist() for a in ax], crops_all=int(present_crop.size),
                crops=int(present_crop.sum()), chunks=chunks, presence=pres)
    if not chunks or not present_crop.any():
        return dict(plan, empty='no present CT chunk in the tile' if not chunks else 'no crop overlaps a present chunk')
    plan['origins'] = np.stack(np.meshgrid(*ax, indexing='ij'), -1)[present_crop]
    return plan


class RowLoader(threading.Thread):
    """Assembles the level-0 chunk rows c0[0]..c1[0]-1 of a unit window (absent chunks zero) in pinned buffers."""
    def __init__(self, store, plan, buffers=3):
        super().__init__(daemon=True)
        self.store, self.plan = store, plan
        c0, c1 = plan['c0'], plan['c1']
        self.ny, self.nx = c1[1] - c0[1], c1[2] - c0[2]
        size = 128 * self.ny * 128 * self.nx * 128
        self.q = queue.Queue()
        self.free = queue.Queue()
        for _ in range(buffers):
            self.free.put(torch.empty(size, dtype=torch.uint8, pin_memory=torch.cuda.is_available()))
        self.error = None
        self.cancelled = threading.Event()
        self.start()

    def run(self):
        try:
            c0, c1, pres = self.plan['c0'], self.plan['c1'], self.plan['presence']
            for r in range(c0[0], c1[0]):
                buf = self.free.get()
                if buf is None or self.cancelled.is_set():
                    return
                a = buf.numpy().reshape(128, self.ny * 128, self.nx * 128)
                cells = list(itertools.product(range(self.ny), range(self.nx)))
                for y, x in cells:  # only chunks present in the map are read; the rest are zero
                    if not pres[r - c0[0], y, x]:
                        a[:, y * 128:(y + 1) * 128, x * 128:(x + 1) * 128] = 0
                keys = [(r, c0[1] + y, c0[2] + x) for y, x in cells if pres[r - c0[0], y, x]]
                for (_, y, x), chunk in zip(keys, self.store.pool.map(self.store.chunk, keys)):
                    if chunk is None:
                        raise RuntimeError(f'chunk {(r, y, x)} is in the presence map but absent')
                    y, x = y - c0[1], x - c0[2]
                    a[:, y * 128:(y + 1) * 128, x * 128:(x + 1) * 128] = chunk
                self.q.put((r, buf))
        except BaseException as e:  # reported to the consumer
            self.error = e
            self.q.put((None, None))

    def get(self):
        r, buf = self.q.get()
        if r is None:
            raise RuntimeError('chunk loader failed') from self.error
        return r, buf

    def close(self):
        """Stop the producer after its current row (also when the consumer failed)."""
        self.cancelled.set()
        self.free.put(None)
        self.join()


class Engine:
    """Runs the units of one output array on one GPU. low(x): [B, 1, 128^3] normalized crops -> [B, 1, 64^3] float32
    logits (student.Student.low for the released model)."""
    def __init__(self, low, store, writer, batch=16, device='cuda', check=None):
        self.low, self.store, self.writer, self.B, self.device = low, store, writer, batch, device
        self.shape0 = tuple(store.shape)
        self.shape1 = output_shape(self.shape0)
        self.grid = crop_grid(self.shape0)
        self.coverage = [(int(a[0]), int(a[-1]) + PATCH) if len(a) else (0, 0) for a in self.grid]
        self.gauss = torch.as_tensor(gaussian(), device=device)
        self.g64 = self.gauss.double()
        self.pct = Percentiles(device)
        self.parents = torch.empty((batch, 256, 256, 256), dtype=torch.uint8, device=device)
        self.counts = torch.empty((batch, 256), dtype=torch.int64, device=device)
        self.images = torch.empty((batch, 1, 128, 128, 128), dtype=torch.float32, device=device)
        self.check = check  # optional callable(parent uint8 numpy, lo, hi, image numpy) on some crops

    @torch.inference_mode()
    def run_unit(self, plan):
        dev, B = self.device, self.B
        T0, T1 = np.array(plan['T0']), np.array(plan['T1'])
        c0, c1 = plan['c0'], plan['c1']
        ax, origins = plan['axes'], plan['origins']
        ny, nx = c1[1] - c0[1], c1[2] - c0[2]
        win = torch.empty((RING_ROWS * 128, ny * 128, nx * 128), dtype=torch.uint8, device=dev)
        OY, OX = int(T1[1] - T0[1]), int(T1[2] - T0[2])
        S = torch.zeros((ACC_ROWS, OY, OX), dtype=torch.float64, device=dev)
        W = torch.zeros_like(S)
        pres = torch.as_tensor(plan['presence'], device=dev)
        yv = torch.arange(int(T0[1]), int(T1[1]), device=dev)
        xv = torch.arange(int(T0[2]), int(T1[2]), device=dev)
        ymask = (yv >= self.coverage[1][0]) & (yv < self.coverage[1][1])
        xmask = (xv >= self.coverage[2][0]) & (xv < self.coverage[2][1])
        yc, xc = yv // 64 - c0[1], xv // 64 - c0[2]
        st = dict(fin=min(ax[0][0], int(T0[0])), row=None, next_row=c0[0], emitted=int(T0[0]))
        finite = torch.ones((), dtype=torch.bool, device=dev)
        covered = torch.ones((), dtype=torch.bool, device=dev)
        self.writer.begin(plan)

        def emit(z0, z1):
            for rs, ls in ring(z0, z1, ACC_ROWS):
                za = z0 + ls.start
                a, b = max(za, int(T0[0])), min(z0 + ls.stop, int(T1[0]))
                for k in range(a, b, SLAB):
                    e = min(b, k + SLAB)
                    r0 = rs.start + (k - za)
                    zv = torch.arange(k, e, device=dev)
                    zmask = (zv >= self.coverage[0][0]) & (zv < self.coverage[0][1])
                    m = pres[zv // 64 - c0[0]][:, yc][:, :, xc] & zmask.view(-1, 1, 1) & ymask.view(1, -1, 1) & xmask.view(1, 1, -1)
                    s, w = S[r0:r0 + e - k], W[r0:r0 + e - k]
                    covered.logical_and_(((w > 0) | ~m).all())
                    code = torch.where(m, encode((s / w).float()), torch.zeros((), dtype=torch.uint8, device=dev))
                    self.writer.rows(k - int(T0[0]), code.cpu().numpy())
                    st['emitted'] = e
                S[rs] = 0
                W[rs] = 0

        def finalize(z0, z1):
            for z in range(z0, z1, 64):
                emit(z, min(z1, z + 64))

        def need(zlo, zhi):
            first, last = zlo // 128, (zhi - 1) // 128
            assert last - first < RING_ROWS
            while st['next_row'] <= last:
                r, buf = loader.get()
                assert r == st['next_row'] and r - RING_ROWS < first
                s0 = (r % RING_ROWS) * 128
                win[s0:s0 + 128].copy_(buf.view(128, ny * 128, nx * 128))
                loader.free.put(buf)
                st['next_row'] += 1

        def blend(batch, field):
            for j, o in enumerate(batch):
                oz, oy, ox = (int(v) for v in o)
                if oz != st['row']:
                    finalize(st['fin'], oz)
                    st['fin'] = st['row'] = oz
                f = field[j, 0]
                finite.logical_and_(torch.isfinite(f).all())
                prod = (self.gauss * f).double()
                ya, yb = max(oy, int(T0[1])), min(oy + PATCH, int(T1[1]))
                xa, xb = max(ox, int(T0[2])), min(ox + PATCH, int(T1[2]))
                if ya >= yb or xa >= xb:
                    continue
                py, px = slice(ya - oy, yb - oy), slice(xa - ox, xb - ox)
                ty, tx = slice(ya - int(T0[1]), yb - int(T0[1])), slice(xa - int(T0[2]), xb - int(T0[2]))
                for rs, ls in ring(oz, oz + PATCH, ACC_ROWS):
                    S[rs, ty, tx] += prod[ls, py, px]
                    W[rs, ty, tx] += self.g64[ls, py, px]

        loader = RowLoader(self.store, plan)
        try:
            for start in range(0, len(origins), B):
                batch = origins[start:start + B]
                n = len(batch)
                for j, o in enumerate(batch):
                    zlo = 2 * int(o[0]) - 64
                    need(zlo, zlo + 256)
                    y, x = 2 * int(o[1]) - 64 - c0[1] * 128, 2 * int(o[2]) - 64 - c0[2] * 128
                    for rs, ls in ring(zlo, zlo + 256, RING_ROWS * 128):
                        self.parents[j, ls] = win[rs, y:y + 256, x:x + 256]
                    self.counts[j] = torch.bincount(self.parents[j].view(-1), minlength=256)
                lo, hi = self.pct(self.counts[:n])
                self.images[:n, 0] = normalize_batch(self.parents[:n, 64:192, 64:192, 64:192], lo, hi)
                if n < B:
                    self.images[n:].zero_()
                if self.check is not None and start == 0:
                    self.check(self.parents[0].cpu().numpy(), float(lo[0]), float(hi[0]), self.images[0, 0].cpu().numpy())
                field = self.low(self.images)[:n]
                blend(batch, field)
            while st['next_row'] < c1[0]:  # drain rows nobody needed (keeps the loader from blocking)
                loader.free.put(loader.get()[1])
                st['next_row'] += 1
            finalize(st['fin'], max(ax[0][-1] + PATCH, int(T1[0])))
            assert st['emitted'] == int(T1[0]), (st['emitted'], T1)
            if not bool(finite):
                raise RuntimeError(f'unit {plan["index"]}: non-finite logits')
            if not bool(covered):
                raise RuntimeError(f'unit {plan["index"]}: a present voxel has no blend weight')
        finally:
            loader.close()
        return self.writer.end()


def normalize_reference(raw, lo, hi):
    """numpy normalize_batch for one block (the CPU check)."""
    if hi - lo < 1e-6:
        return np.zeros(raw.shape, np.float32)
    x = np.clip(raw.astype(np.float64), lo, hi).astype(np.float32)
    x = (x.astype(np.float64) - lo).astype(np.float32)
    return (x.astype(np.float64) / (hi - lo)).astype(np.float32)


def cpu_check(parent, lo, hi, image):
    """GPU p1 / p99 and normalized crop against np.percentile and normalize_reference (raises on a mismatch)."""
    ref = np.percentile(parent.astype(np.float32), [1, 99])
    if not (ref[0] == lo and ref[1] == hi):
        raise RuntimeError(f'GPU percentiles {(lo, hi)} differ from np.percentile {tuple(ref)}')
    if not np.array_equal(image, normalize_reference(parent[64:192, 64:192, 64:192], lo, hi)):
        raise RuntimeError('GPU normalization differs from the CPU reference')


def zarray(shape1, chunk=256, compressor=CODEC):
    return dict(zarr_format=2, shape=list(shape1), chunks=[chunk] * 3, dtype='|u1', order='C', fill_value=0,
                filters=None, compressor=compressor, dimension_separator='/')


class ZarrWriter:
    """Writes a unit's output chunks (zarr v2 array directory `path`, '/' separator) as their rows arrive. Chunk files
    are written atomically (temporary file, rename); chunks without present CT are never written (fill 0).
    stored / verify / clean inspect one tile's chunk files on disk (done-marker checks, interrupted units)."""
    def __init__(self, path, meta, threads=8, fsync=True):
        from concurrent.futures import ThreadPoolExecutor
        self.path, self.meta, self.fsync = path, meta, fsync
        self.chunk = meta['chunks'][0]
        self.codec = None
        if meta['compressor'] is not None:
            import numcodecs
            self.codec = numcodecs.get_codec(meta['compressor'])
        self.pool = ThreadPoolExecutor(threads)

    def begin(self, plan):
        self.T0, self.T1 = np.array(plan['T0']), np.array(plan['T1'])
        self.shape = tuple(int(v) for v in self.T1 - self.T0)
        assert np.all(self.T0 % self.chunk == 0)
        self.want = set(plan['chunks'])
        padded = tuple(-(-n // self.chunk) * self.chunk for n in self.shape[1:])
        self.bufs = [np.zeros((self.chunk,) + padded, np.uint8) for _ in range(2)]
        self.slot, self.futures, self.files, self.z = 0, [[], []], {}, 0

    def _write(self, key, block):
        data = block.tobytes() if self.codec is None else self.codec.encode(np.ascontiguousarray(block))
        d = self.path / str(key[0]) / str(key[1])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f'.{key[2]}.{os.getpid()}.part'
        try:
            with tmp.open('wb') as f:
                f.write(data)
                if self.fsync:
                    f.flush()
                    os.fsync(f.fileno())
            tmp.replace(d / str(key[2]))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return key, len(data), int(np.count_nonzero(block)), hashlib.sha256(data).hexdigest()

    def rows(self, z, codes):
        """codes: uint8 [n, OY, OX] for tile rows z..z+n (in order)."""
        assert z == self.z, (z, self.z)
        n, C, (OZ, OY, OX) = len(codes), self.chunk, self.shape
        i = 0
        while i < n:
            cz, off = divmod(z + i, C)
            m = min(n - i, C - off)
            buf = self.bufs[self.slot]
            buf[off:off + m] = 0
            buf[off:off + m, :OY, :OX] = codes[i:i + m]
            i += m
            if off + m == C or z + i == OZ:
                buf[off + m:] = 0
                k0 = self.T0 // C
                futures = []
                for oy, ox in itertools.product(range(-(-OY // C)), range(-(-OX // C))):
                    key = (int(k0[0] + cz), int(k0[1] + oy), int(k0[2] + ox))
                    if key in self.want:
                        futures.append(self.pool.submit(self._write, key, buf[:, oy * C:(oy + 1) * C, ox * C:(ox + 1) * C]))
                self.futures[self.slot] = futures
                self.slot ^= 1
                self._collect(self.slot)
        self.z += n

    def _collect(self, slot):
        for f in self.futures[slot]:
            key, size, nonzero, digest = f.result()
            self.files[' '.join(map(str, key))] = dict(bytes=size, nonzero=nonzero, sha256=digest)
        self.futures[slot] = []

    def end(self):
        self._collect(0)
        self._collect(1)
        assert self.z == self.shape[0], (self.z, self.shape)
        assert set(self.files) == {' '.join(map(str, k)) for k in self.want}, 'written chunks differ from the plan'
        return dict(files=self.files, bytes=sum(v['bytes'] for v in self.files.values()))

    def stored(self, T0, T1):
        """Chunk files ('z y x' -> path) and temporary files of tile T0..T1 on disk. The directories also hold
        neighbouring tiles' chunks; only this tile's keys are returned."""
        k0, k1 = np.asarray(T0) // self.chunk, -(-np.asarray(T1) // self.chunk)
        xs = range(int(k0[2]), int(k1[2]))
        chunks, temps = {}, []
        for z, y in itertools.product(range(int(k0[0]), int(k1[0])), range(int(k0[1]), int(k1[1]))):
            d = self.path / str(z) / str(y)
            try:
                names = os.listdir(d)
            except FileNotFoundError:
                continue
            for name in names:
                m = re.fullmatch(r'(0|[1-9][0-9]*)|\.([0-9]+)\.[0-9]+\.part', name)
                if m is None or int(m.group(1) or m.group(2)) not in xs:
                    continue
                if m.group(1) is not None:
                    chunks[f'{z} {y} {name}'] = d / name
                else:
                    temps.append(d / name)
        return chunks, temps

    def verify(self, T0, T1, files):
        """Differences between tile T0..T1 on disk and a record's files {'z y x': {bytes, sha256}}: missing,
        unrecorded, or changed chunk files. An empty list: the tile holds exactly the recorded chunks."""
        chunks, _ = self.stored(T0, T1)
        problems = [f'chunk {k} missing' for k in sorted(set(files) - set(chunks))]
        problems += [f'chunk {k} not recorded' for k in sorted(set(chunks) - set(files))]

        def check(k):
            try:
                data = chunks[k].read_bytes()
            except OSError as e:
                return f'chunk {k} unreadable ({e})'
            if len(data) != files[k]['bytes'] or hashlib.sha256(data).hexdigest() != files[k]['sha256']:
                return f'chunk {k} differs from the record'
        return problems + [p for p in self.pool.map(check, sorted(set(files) & set(chunks))) if p]

    def clean(self, T0, T1, keep=()):
        """Delete tile T0..T1's chunk files not in `keep` (chunk keys) and its temporary files: the leftovers of an
        interrupted or superseded attempt. Run before a unit is (re)computed."""
        keep = {' '.join(map(str, k)) for k in keep}
        chunks, temps = self.stored(T0, T1)
        for path in temps + [path for k, path in chunks.items() if k not in keep]:
            path.unlink(missing_ok=True)

    def close(self):
        """Cancel queued chunk writes and wait for running ones (also after a failed unit)."""
        self.pool.shutdown(wait=True, cancel_futures=True)
