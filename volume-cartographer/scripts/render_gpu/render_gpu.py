#!/usr/bin/env python3
"""GPU sampling stage for vc_render_tifxyz (see README.md).

render_shim.so, preloaded into vc_render_tifxyz, dumps each output tile's base
points and unit step directions. This script samples them on the GPU with
rgpu_kernel.py, which reproduces ChunkSampler::sampleTrilinear and
sampleTileSlicesImpl, writes level-0 chunks into the zarr the C++ run created
(same layout and compressor, all-zero chunks skipped like writeChunkSkipEmpty)
and builds pyramid levels 1-5 the way renderTiles does.

usage: render_gpu.py DUMP_DIR OUT_ZARR VOLUME_LEVEL_DIR [--device cuda:0]
       render_gpu.py @LIST VOLUME_LEVEL_DIR [--wait --producer-pid PID]
"""
import argparse
import glob
import json
import os
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np
import torch

import rgpu_kernel

CS = 128  # volume chunk edge (checked against .zarray)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_dump(d):
    with open(os.path.join(d, "offsets.bin"), "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        offs = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
    if len(offs) != n:
        raise RuntimeError(f"{d}/offsets.bin: truncated")
    tiles = []
    for p in sorted(glob.glob(os.path.join(d, "t*.bin"))):
        buf = np.fromfile(p, dtype=np.uint8)
        o = 0
        while o < len(buf):
            if o + 16 > len(buf):
                raise RuntimeError(f"{p}: truncated tile header")
            ty, tx, h, w = np.frombuffer(buf[o:o + 16], dtype=np.int32)
            o += 16
            k = int(h) * int(w) * 12
            if o + 2 * k > len(buf):
                raise RuntimeError(f"{p}: truncated tile record")
            b = np.frombuffer(buf[o:o + k], dtype=np.float32).reshape(h, w, 3)
            o += k
            dd = np.frombuffer(buf[o:o + k], dtype=np.float32).reshape(h, w, 3)
            o += k
            tiles.append((int(ty), int(tx), b, dd))
    if not tiles:
        raise RuntimeError(f"{d}: dump has no tiles")
    tiles.sort(key=lambda t: (t[0], t[1]))
    return offs, tiles


class ChunkPool:
    """Volume chunks resident on the GPU; dense LUT chunk-grid -> slot (-1 = absent/zero)."""

    def __init__(self, vol_dir, dev, budget_gb):
        meta = read_json(os.path.join(vol_dir, ".zarray"))
        assert meta["dtype"] == "|u1" and meta["chunks"] == [CS] * 3 and meta["order"] == "C", meta
        assert not meta.get("filters"), meta
        self.codec = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
        self.sep = meta.get("dimension_separator", ".")
        self.dir = vol_dir
        self.dev = dev
        self.shape = meta["shape"]
        self.grid = [(s + CS - 1) // CS for s in self.shape]
        self.lut = torch.full(self.grid, -1, dtype=torch.int32, device=dev)
        self.cap = max(8, int(budget_gb * 2**30) // CS**3)
        self.pool = torch.zeros((self.cap, CS**3), dtype=torch.uint8, device=dev)
        self.slot_of = {}  # key -> slot, or -2 for a missing chunk (fill value 0)
        self.free = list(range(self.cap))
        self.clock = {}
        self.io = ThreadPoolExecutor(16)
        self.tick = 0

    def close(self):
        self.io.shutdown()

    def read_host(self, keys):
        """Read chunk files (any thread). -> (present_keys, uint8 array [n, CS^3], absent_keys)"""
        keys = list(keys)
        buf = np.empty((len(keys), CS**3), np.uint8)
        ok = [False] * len(keys)

        def rd(i):
            p = os.path.join(self.dir, self.sep.join(map(str, keys[i])))
            try:
                with open(p, "rb", buffering=0) as f:
                    if self.codec is None:
                        n = f.readinto(memoryview(buf[i]))
                        if n != CS**3:
                            raise RuntimeError(f"{p}: short read ({n} of {CS**3} bytes)")
                    else:
                        try:
                            self.codec.decode(f.read(), out=buf[i])
                        except Exception as e:
                            raise RuntimeError(f"{p}: decode failed: {e}") from e
            except FileNotFoundError:
                return
            ok[i] = True

        list(self.io.map(rd, range(len(keys))))
        idx = [i for i in range(len(keys)) if ok[i]]
        absent = [keys[i] for i in range(len(keys)) if not ok[i]]
        return [keys[i] for i in idx], buf if not absent else buf[idx], absent

    def missing(self, keys):
        return [k for k in keys if k not in self.slot_of]

    def ensure(self, keys, pre=None):
        """Make every key resident (or marked absent). pre: a read_host() result to use first."""
        self.tick += 1
        for k in keys:
            if k in self.slot_of:
                self.clock[k] = self.tick
        if len(keys) > self.cap:
            raise RuntimeError(f"batch needs {len(keys)} chunks > pool {self.cap}; raise --pool-gb")
        need = set(self.missing(keys))
        got_k, got_a, absent = [], [], []
        if pre is not None:
            pk, pa, pab = pre
            sel = [i for i, k in enumerate(pk) if k in need]
            got_k += [pk[i] for i in sel]
            got_a.append(pa if len(sel) == len(pk) else pa[sel])
            absent += [k for k in pab if k in need]
            need -= set(got_k) | set(absent)
        if need:
            k2, a2, ab2 = self.read_host(need)
            got_k += k2
            got_a.append(a2)
            absent += ab2
        for k in absent:  # missing chunk = fill value 0: LUT stays -1
            self.slot_of[k] = -2
            self.clock[k] = self.tick
        if not got_k:
            return
        if len(self.free) < len(got_k):  # evict the least recently used slots not in this batch
            old = sorted((k for k in self.slot_of if self.clock[k] < self.tick and self.slot_of[k] >= 0),
                         key=self.clock.get)
            ev = old[:len(got_k) - len(self.free)]
            if len(ev) < len(got_k) - len(self.free):
                raise RuntimeError(f"chunk pool {self.cap} too small for one batch; raise --pool-gb")
            for k in ev:
                self.free.append(self.slot_of.pop(k))
                self.clock.pop(k)
            e = torch.tensor(ev, device=self.dev, dtype=torch.long)
            self.lut[e[:, 0], e[:, 1], e[:, 2]] = -1
        slots = [self.free.pop() for _ in got_k]
        host = torch.from_numpy(np.concatenate(got_a) if len(got_a) > 1 else got_a[0])
        self.pool[torch.tensor(slots, device=self.dev)] = host.to(self.dev)
        ks = torch.tensor(got_k, device=self.dev, dtype=torch.long)
        self.lut[ks[:, 0], ks[:, 1], ks[:, 2]] = torch.tensor(slots, device=self.dev, dtype=torch.int32)
        for k, sl in zip(got_k, slots):
            self.slot_of[k] = sl
            self.clock[k] = self.tick


def render_one(pool, kernel, dump, out_zarr, a, wpool):
    """Sample one dump into out_zarr. Writes are left in flight on wpool; returns finish()."""
    dev = pool.dev
    t0 = time.time()
    offs, tiles = dump
    zmeta = read_json(os.path.join(out_zarr, "0", ".zarray"))
    CZ, CH, CW = zmeta["chunks"]
    S = len(offs)
    assert zmeta["dtype"] == "|u1" and CZ >= S, zmeta
    comp = numcodecs.get_codec(zmeta["compressor"]) if zmeta["compressor"] else None
    sep = zmeta.get("dimension_separator", ".")
    offs_t = torch.from_numpy(offs).to(dev)
    futs = []

    H, W = zmeta["shape"][1], zmeta["shape"][2]
    levels = sorted(int(d) for d in os.listdir(out_zarr)
                    if d.isdigit() and int(d) > 0 and os.path.exists(os.path.join(out_zarr, d, ".zarray")))
    # level 0 is kept on the host for the pyramid so it doesn't sit on the GPU next to the chunk pool
    img = torch.zeros((S, H, W), dtype=torch.uint8) if levels else None

    def write_chunk(ty, tx, arr, lvl=0, chunks=(CZ, CH, CW)):
        buf = np.zeros(chunks, np.uint8)
        buf[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr
        if not buf.any():  # writeChunkSkipEmpty
            return 0
        data = comp.encode(buf) if comp else buf.tobytes()
        key = sep.join(["0", str(ty), str(tx)])  # chunk key cz/cy/cx (one z chunk)
        p = os.path.join(out_zarr, str(lvl), key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return 1

    batches = [tiles[i:i + a.batch_tiles] for i in range(0, len(tiles), a.batch_tiles)]

    def prep(tb):
        base = torch.from_numpy(np.concatenate([t[2].reshape(-1, 3) for t in tb])).to(dev)
        dirs = torch.from_numpy(np.concatenate([t[3].reshape(-1, 3) for t in tb])).to(dev)
        g = pool.grid
        mark = torch.zeros(g[0] * g[1] * g[2], dtype=torch.uint8, device=dev)
        kernel.mark_cuda(base, dirs, offs_t, *pool.shape, *g, mark)
        u = torch.nonzero(mark).squeeze(1)
        z, y, x = u // (g[1] * g[2]), (u // g[2]) % g[1], u % g[2]
        return (base, dirs), [tuple(r) for r in torch.stack([z, y, x], 1).cpu().tolist()]

    # chunk reads for batch i+1 overlap the upload and sampling of batch i;
    # chunks batch i loads itself are left out of that read
    with ThreadPoolExecutor(1) as pf:
        v, keys = prep(batches[0])
        pending = pf.submit(pool.read_host, pool.missing(keys))
        for bi, tb in enumerate(batches):
            pre = pending.result()
            pending = None
            if bi + 1 < len(batches):
                v2, keys2 = prep(batches[bi + 1])
                cur = set(keys)
                pending = pf.submit(pool.read_host, [k for k in pool.missing(keys2) if k not in cur])
            pool.ensure(keys, pre)
            base, dirs = v
            og = torch.empty((S, base.shape[0]), dtype=torch.uint8, device=dev)
            kernel.sample_cuda(base, dirs, offs_t, pool.lut, pool.pool, *pool.shape, og)
            out = og.cpu().numpy()
            o = 0
            for ty, tx, b, _ in tb:
                h, w = b.shape[:2]
                if img is not None:
                    img[:, ty * CH:ty * CH + h, tx * CW:tx * CW + w] = \
                        torch.from_numpy(out[:, o:o + h * w].reshape(S, h, w))
                futs.append(wpool.submit(write_chunk, ty, tx, out[:, o:o + h * w].reshape(S, h, w)))
                o += h * w
            if bi + 1 < len(batches):
                v, keys = v2, keys2
    n_l0 = len(futs)

    # Pyramid L1..L5, as renderTiles builds it: L1 = rounded 2x2 mean of L0 with partial
    # windows at the image edge; L2+ cascade from zero-padded chunk buffers (count 4).
    prev = img
    del img
    for li, lvl in enumerate(levels):
        m = read_json(os.path.join(out_zarr, str(lvl), ".zarray"))
        _, lh, lw = m["shape"]
        cz, ch, cw = m["chunks"]
        ph, pw = prev.shape[1] + prev.shape[1] % 2, prev.shape[2] + prev.shape[2] % 2
        # int16 window sums (max 4*255) hold the same values as int32 in less memory
        p = torch.nn.functional.pad(prev, (0, pw - prev.shape[2], 0, ph - prev.shape[1]))
        ssum = p.view(S, ph // 2, 2, pw // 2, 2).to(torch.int16).sum((2, 4), dtype=torch.int16)
        del p
        if li == 0:
            one = torch.zeros((ph, pw), dtype=torch.int32)
            one[:prev.shape[1], :prev.shape[2]] = 1
            cnt = one.view(ph // 2, 2, pw // 2, 2).sum((1, 3)).clamp(min=1)
            cur = ((ssum + cnt // 2) // cnt).to(torch.uint8)
        else:
            cur = ((ssum + 2) // 4).to(torch.uint8)
        cur = cur[:, :lh, :lw].contiguous()
        cn = cur.cpu().numpy()
        for cy in range(0, lh, ch):
            for cx in range(0, lw, cw):
                futs.append(wpool.submit(write_chunk, cy // ch, cx // cw,
                                         cn[:, cy:cy + ch, cx:cx + cw], lvl, (cz, ch, cw)))
        prev = cur
    del prev

    def finish():
        written = sum(f.result() for f in futs[:n_l0])
        for f in futs[n_l0:]:
            f.result()
        return dict(out=out_zarr, tiles=len(tiles), written=written, skipped_empty=n_l0 - written,
                    pyramid_levels=len(levels), total_s=round(time.time() - t0, 2))
    return finish


def producer_alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="shim dump dir, or @LIST with lines 'DUMP_DIR OUT_ZARR' "
                                 "(one GPU chunk pool shared across the list)")
    ap.add_argument("out_zarr", nargs="?", help="output zarr (omit with @LIST)")
    ap.add_argument("vol", help="volume level dir, e.g. VOLUME.zarr/0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-tiles", type=int, default=64)
    ap.add_argument("--pool-gb", type=float, default=None,
                    help="GPU memory for volume chunks (default: free memory minus 2 GB)")
    ap.add_argument("--writers", type=int, default=8)
    ap.add_argument("--rm-dump", action="store_true", help="delete each dump dir once its zarr is written")
    ap.add_argument("--wait", action="store_true",
                    help="wait for DUMP/.done or DUMP/.fail (dumps produced concurrently)")
    ap.add_argument("--producer-pid", type=int,
                    help="with --wait: fail if this process exits before a dump is marked")
    a = ap.parse_args()
    if not a.device.startswith("cuda"):
        ap.error("--device must be a CUDA/ROCm device (cuda or cuda:N)")
    if a.dump.startswith("@"):
        if a.out_zarr:
            ap.error("OUT_ZARR comes from the list in @LIST mode")
        with open(a.dump[1:]) as f:
            jobs = [line.split() for line in f if line.strip()]
    else:
        if not a.out_zarr:
            ap.error("OUT_ZARR is required")
        jobs = [(a.dump, a.out_zarr)]

    dev = torch.device(a.device)
    torch.cuda.set_device(dev)  # the kernels launch on the current device
    if a.pool_gb is None:
        a.pool_gb = torch.cuda.mem_get_info(dev)[0] / 2**30 - 2
        if a.pool_gb < 1:
            ap.error(f"only {a.pool_gb + 2:.1f} GB free on {a.device}; pass --pool-gb")
    pool = ChunkPool(a.vol, dev, a.pool_gb)
    kernel = rgpu_kernel.load(dev)
    t0 = time.time()

    def fetch(d):
        while a.wait and not os.path.exists(os.path.join(d, ".done")):
            if os.path.exists(os.path.join(d, ".fail")):
                break
            if a.producer_pid and not producer_alive(a.producer_pid):
                if os.path.exists(os.path.join(d, ".done")) or os.path.exists(os.path.join(d, ".fail")):
                    break
                raise RuntimeError(f"{d}: dump producer {a.producer_pid} exited without marking the dump")
            time.sleep(0.1)
        if os.path.exists(os.path.join(d, ".fail")):
            return None
        return load_dump(d)

    failed = 0
    # the next dump loads while the current one renders; writes overlap the next render
    with ThreadPoolExecutor(1) as lpool, ThreadPoolExecutor(a.writers) as wpool:
        nxt = lpool.submit(fetch, jobs[0][0])
        pend = None

        def flush(p):
            fin, d = p
            print(json.dumps(fin()), flush=True)
            if a.rm_dump:
                shutil.rmtree(d)

        for i, (d, o) in enumerate(jobs):
            cur = nxt.result()
            if i + 1 < len(jobs):
                nxt = lpool.submit(fetch, jobs[i + 1][0])
            if cur is None:
                failed += 1
                print(json.dumps(dict(out=o, skipped="dump failed")), flush=True)
                continue
            fin = render_one(pool, kernel, cur, o, a, wpool)
            if pend:
                flush(pend)
            pend = (fin, d)
        if pend:
            flush(pend)
    pool.close()
    print(json.dumps(dict(surfaces=len(jobs), failed=failed, total_s=round(time.time() - t0, 2))), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
