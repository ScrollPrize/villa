"""Host-RAM offload of fit_spiral's lasagna resident brick pools (see POOL_OFFLOAD.md).

FIT_SPIRAL_POOL_OFFLOAD=1 (default OFF): lasagna_data builds its brick pools with OffloadBrickPool instead of ResidentBrickPool (sparse_cuda_cache.resident_brick_pool_class()).
- The brick table, the index arithmetic and the output stay ON THE CARD, exactly as in ResidentBrickPool.gather.
- The uint8 pool (channels x bricks x 32^3) lives in host RAM that is page-locked and mapped into the card's address space
  (cuMemHostRegister DEVICEMAP).
- A tiny gather kernel reads the requested voxels over PCIe straight from that host memory: zero-copy, no CPU gather, and
  no host synchronisation.
- Values are bit-identical to the resident gather (the same bytes, the same index arithmetic).
The pool load is the original one: the same sidecar memmaps and the same ROI brick selection, only the destination differs.
Applies to every ResidentBrickPool the fit creates (normals, gradient magnitude, surface SDT). A pool built with device='cpu'
(the unit tests) is left resident.
FIT_SPIRAL_POOL_OFFLOAD_MIN_GIB (default 0): only pools at least this large are offloaded."""
import ctypes, os, sys, time, weakref
import numpy as np
import torch

_KSRC = r'''
extern "C" __global__ void gather_host(const unsigned char* __restrict__ pool, const long long* __restrict__ slots,
                                       const long long* __restrict__ lin, long long n, int C, long long rows, long long bv,
                                       unsigned char* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long s = slots[i], l = lin[i];
    for (int c = 0; c < C; c++) out[i * C + c] = pool[((long long)c * rows + s) * bv + l];
}
'''
_MODS = {}


def _module(dev_index):
    if dev_index not in _MODS:
        from host_pool_cuda import Module
        _MODS[dev_index] = Module(_KSRC, device=dev_index)
    return _MODS[dev_index]


def _register(arr, dev_index):
    """page-lock + map a numpy array; returns its device pointer (int)"""
    from host_pool_cuda import _libs, _chk
    _, cuda = _libs(); torch.cuda.set_device(dev_index); torch.zeros(1, device=f"cuda:{dev_index}")
    p = ctypes.c_void_p(arr.ctypes.data)
    _chk(cuda.cuMemHostRegister_v2(p, ctypes.c_size_t(arr.nbytes), ctypes.c_uint(0x02)), "cuMemHostRegister(DEVICEMAP)")
    dptr = ctypes.c_uint64()
    try:
        _chk(cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), p, ctypes.c_uint(0)), "cuMemHostGetDevicePointer")
    except BaseException:
        _unregister(arr)                                                 # registered but unusable: do not leave the pages page-locked
        raise
    return int(dptr.value)


def _unregister(arr):
    """Undo _register (idempotent enough: a second call is refused by the driver and ignored). Also the weakref finalizer of an offloaded pool: it holds `arr`
    strongly, so the registered memory is only released after it was unregistered, whichever way the pool ends (close(), a failed construction, or garbage)."""
    try:
        from host_pool_cuda import _libs
        _libs()[1].cuMemHostUnregister(ctypes.c_void_p(arr.ctypes.data))
    except Exception:
        pass


def offload_class(scc):
    """The host-RAM subclass of scc.ResidentBrickPool (scc = the sparse_cuda_cache module); see sparse_cuda_cache.resident_brick_pool_class()."""
    Base = scc.ResidentBrickPool
    open_pool = scc.open_pool
    min_gib = float(os.environ.get("FIT_SPIRAL_POOL_OFFLOAD_MIN_GIB", "0"))

    class OffloadBrickPool(Base):
        def __init__(self, sidecar_dir, *, origin_zyx=(0, 0, 0), z_roi=None, device="cuda", label, expected_channels=None,
                     expected_shape_zyx=None, progress_callback=None):
            dev = torch.device(device)
            if dev.type == "cuda" and dev.index is None: dev = torch.device("cuda", torch.cuda.current_device())   # resolve a bare 'cuda' now: a later current-device change must not move this pool
            meta, table_np, coords_np, memmaps = open_pool(sidecar_dir)
            rows = int(meta["rows"]); bvox = int(np.prod([int(v) for v in meta["brick_shape"]]))
            if z_roi is not None:
                brick0 = int(meta["brick_shape"][0]); bz = coords_np[:, 0].astype(np.int64)
                nkeep = int(((bz * brick0 < int(z_roi[1])) & ((bz + 1) * brick0 > int(z_roi[0]))).sum()) + 1
            else: nkeep = rows
            gib = len(memmaps) * nkeep * bvox / 2 ** 30
            self._offloaded = dev.type == "cuda" and gib >= min_gib
            if not self._offloaded:
                Base.__init__(self, sidecar_dir, origin_zyx=origin_zyx, z_roi=z_roi, device=device, label=label, expected_channels=expected_channels,
                              expected_shape_zyx=expected_shape_zyx, progress_callback=progress_callback)
                return
            # ---- the original __init__ (sparse_cuda_cache.ResidentBrickPool), with the pool in mapped host RAM instead of on the card
            started = time.perf_counter(); self.sidecar_dir = str(sidecar_dir); self.device = dev; self.label = str(label); self.meta = meta
            self.shape_zyx = tuple(int(v) for v in meta["array_shape"])
            if expected_shape_zyx is not None and tuple(int(v) for v in expected_shape_zyx) != self.shape_zyx:
                raise ValueError(f"{label}: sidecar {sidecar_dir} covers array shape {self.shape_zyx}, expected {tuple(expected_shape_zyx)}")
            self.channels = len(memmaps)
            if expected_channels is not None and self.channels != expected_channels:
                raise ValueError(f"{label}: sidecar {sidecar_dir} has {self.channels} channel(s), expected {expected_channels}")
            self.origin_zyx = tuple(int(v) for v in origin_zyx); brick = tuple(int(v) for v in meta["brick_shape"])
            keep = np.ones(rows, dtype=bool)
            if z_roi is not None:
                z_lo, z_hi = int(z_roi[0]), int(z_roi[1]); brick_z = coords_np[:, 0].astype(np.int64)
                keep = (brick_z * brick[0] < z_hi) & ((brick_z + 1) * brick[0] > z_lo); keep[0] = True
            kept_ids = np.flatnonzero(keep); remap = np.zeros(rows, dtype=np.int32); remap[kept_ids] = np.arange(len(kept_ids), dtype=np.int32)
            self.pool_bytes = self.channels * len(kept_ids) * bvox
            self.host_pool = np.empty((self.channels, len(kept_ids), bvox), dtype=np.uint8)
            slab = max(1, (256 << 20) // bvox); total = self.channels * len(kept_ids); done = 0
            for channel, mm in enumerate(memmaps):
                for lo in range(0, len(kept_ids), slab):
                    ids = kept_ids[lo:lo + slab]; self.host_pool[channel, lo:lo + len(ids)] = mm[ids]; done += len(ids)
                    if progress_callback is not None: progress_callback(done, total, f"channel {channel + 1}/{self.channels} (host RAM)")
            di = dev.index if dev.index is not None else torch.cuda.current_device()
            with torch.cuda.device(di):                                  # _register() sets the device: leave the caller's current device as it was
                self._dptr = _register(self.host_pool, di)
            self._finalizer = weakref.finalize(self, _unregister, self.host_pool)   # ownership of the registration starts here and ends in close(), a failed construction or garbage collection
            try:
                with torch.cuda.device(di):
                    self._mod = _module(di)
                self._dev_index = di
                self.pool = None                                             # nothing of the pool on the card
                self._verify_left = int(os.environ.get("FIT_SPIRAL_POOL_OFFLOAD_VERIFY", "0")); self._verify = dict(gathers=0, lookups=0, mismatched_values=0)
                if self._verify_left:                                        # VERIFY mode (identity test only): a resident copy on the card, every offloaded gather compared with it
                    self._shadow = torch.from_numpy(self.host_pool).to(self.device)
                    import atexit; atexit.register(lambda s_=self: print(f"rb_offload VERIFY {s_.label} FINAL: {s_._verify['gathers']} gathers, {s_._verify['lookups']:,} lookups x {s_.channels} channels compared with the resident pool: {s_._verify['mismatched_values']} mismatched values", flush=True))
                self.table = torch.from_numpy(np.ascontiguousarray(remap[table_np])).to(self.device)
                self._brick = torch.tensor(brick, dtype=torch.long, device=self.device); self._origin = torch.tensor(self.origin_zyx, dtype=torch.long, device=self.device)
                self._shape = torch.tensor(self.shape_zyx, dtype=torch.long, device=self.device); self._stride_z = brick[1] * brick[2]; self._stride_y = brick[2]
                self._bounds_check = os.environ.get("FIT_SPIRAL_RESIDENT_BOUNDS_CHECK") == "1"
                self.resident_bricks = len(kept_ids); self.total_bricks = rows; self._gathers = 0; self._gather_seconds = 0.0
                self._rows = len(kept_ids); self._bvox = bvox
                self.load_seconds = time.perf_counter() - started; self.last_timings = {}
                print(f"{self.label}: OFFLOADED pool {self.resident_bricks:,}/{rows:,} bricks of {brick} x{self.channels} channel(s) "
                      f"({self.pool_bytes / 1024**3:.2f} GiB) in mapped host RAM (FIT_SPIRAL_POOL_OFFLOAD=1), loaded in {self.load_seconds:.1f}s", flush=True)
            except BaseException:
                self._finalizer()                                              # unregister before the host array is released
                self.host_pool = None
                raise

        def gather(self, indices_zyx):
            if not self._offloaded: return Base.gather(self, indices_zyx)
            with torch.cuda.device(self.device):                              # allocations, stream and the raw driver launch all belong to the pool's own card
                started = time.perf_counter(); original_shape = tuple(indices_zyx.shape[:-1]); flat = indices_zyx.detach().reshape(-1, 3)
                if flat.shape[0] == 0: return torch.empty((*original_shape, self.channels), dtype=torch.uint8, device=self.device)
                flat = flat.to(device=self.device, dtype=torch.long); source = flat + self._origin
                if self._bounds_check and bool(((source < 0) | (source >= self._shape)).any()): raise IndexError(f"{self.label} gather received an out-of-bounds index")
                brick_idx = torch.div(source, self._brick, rounding_mode="floor")
                slots = self.table[brick_idx[:, 0], brick_idx[:, 1], brick_idx[:, 2]].to(torch.long).contiguous()
                local = source - brick_idx * self._brick
                linear = (local[:, 0] * self._stride_z + local[:, 1] * self._stride_y + local[:, 2]).contiguous()
                n = int(slots.shape[0]); out = torch.empty((n, self.channels), dtype=torch.uint8, device=self.device)
                st = torch.cuda.current_stream(self.device).cuda_stream
                self._mod.launch("gather_host", ((n + 255) // 256,), (256,), [ctypes.c_uint64(self._dptr), slots, linear, ctypes.c_longlong(n), ctypes.c_int(self.channels),
                                                                               ctypes.c_longlong(self._rows), ctypes.c_longlong(self._bvox), out], stream=st)
                if getattr(self, "_verify_left", 0) > 0:
                    ref = self._shadow[:, slots, linear].transpose(0, 1); bad = int((ref != out).sum()); self._verify_left -= 1
                    v = self._verify; v["gathers"] += 1; v["lookups"] += n; v["mismatched_values"] += bad
                    if self._verify_left == 0 or bad or v["gathers"] % 100 == 0:
                        print(f"rb_offload VERIFY {self.label}: {v['gathers']} gathers, {v['lookups']:,} lookups x {self.channels} channels compared with the resident pool: {v['mismatched_values']} mismatched values", flush=True)
                    if self._verify_left == 0: self._shadow = None; torch.cuda.empty_cache()
                elapsed = time.perf_counter() - started; self._gather_seconds += elapsed; self._gathers += 1
                self.last_timings = {"gather_seconds": elapsed, "resident_bricks": self.resident_bricks, "resident_mib": 0.0, "offloaded_mib": self.pool_bytes / 1024 ** 2}
                return out.reshape(*original_shape, self.channels)

        def close(self):
            finalizer = getattr(self, "_finalizer", None)
            if finalizer is not None: finalizer()                        # unregisters once; later calls (and garbage collection) do nothing
            self.host_pool = None
            self.pool = None; self.table = None

    print(f"brick_pool_offload: lasagna brick pools of at least {min_gib} GiB go to mapped host RAM (FIT_SPIRAL_POOL_OFFLOAD=1)", flush=True)
    return OffloadBrickPool
