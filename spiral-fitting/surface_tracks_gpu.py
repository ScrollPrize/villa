"""Multi-GPU Brook backend of ``extract_surface_tracks.py``.

``run`` writes the DBM that ``extract_surface_tracks.py`` writes with Kimimaro: the same keys
(``h:{z}``, then ``vy:{w}``, then ``vx:{w}``, inserted in that order), the same value format
(``pickle.dumps`` of a list of (N, 3) int32 ZYX tracks in full-resolution coordinates, tracks of
fewer than 10 vertices dropped, ``pickle.dumps([])`` for empty slabs) and the same resume rule
(a key already in the DBM is not computed again). The skeletons come from Brook
(``brook.skeletonize_batch``) instead of Kimimaro, so the tracks are Brook's.

Design:

  input     A local zarr v2 array in the layout of the surface predictions (uint8, blosc, '/'
            chunk keys) is decoded straight from its chunk files by parent threads; any other
            array is read through the caller's ``open_array``. Either way the z-range is
            binarized once (``> 0``) into one bit-packed block in POSIX shared memory,
            (z1 - z0) * Y * ceil(X / 8) bytes, which every GPU worker maps. The same pass
            records the foreground bounding box at the slices ``find_yx_range`` samples.
  workers   One spawned process per GPU, started first so that their start-up (imports, CUDA
            context, Brook and cuCIM warm-ups) overlaps the parent's decoding. CUDA contexts
            created by several processes at once contend in the driver and all become ready
            late, so the workers create theirs one at a time in GPU order (a shared ticket);
            each does its cuInit in a thread while its main thread imports.
  NUMA      When every selected GPU's NUMA node is known, each worker and its track threads run
            on (and allocate from) the GPU's node; the decode threads and the block use the
            first node. Otherwise nothing is bound.
  batches   A worker prepares batch k + 1 (gather into pinned memory, upload, unpack the bits,
            then ``prepare_cc_labels`` on the GPU: cuCIM connected components with
            6-connectivity, max-pool by the downsample factor, keep the components of
            [dust, max-area) / ds**3 voxels) on its own stream while Brook skeletonizes batch
            k; a batch that runs out of device memory is split in halves.
            The parent sizes batches with a static memory model and the worker's free memory,
            caps ribbon calls at 128 slabs, cuts the vertical passes so that the GPUs are
            predicted to finish together, and hands a busy GPU its next batch just before its
            running call is predicted to end.
  tracks    The skeletons return as raw arrays over a pipe; parent threads turn them into
            tracks with ``fast_tracks.tracks_from_packed`` (the ``'interjoint'`` walk of
            ``extract_surface_tracks.get_skeleton_tracks``) and pickle them.
  writes    Values are queued strictly in the key order above and stored by a DBM writer
            process, so that the writes do not hold the parent's GIL.

Linux only (NUMA syscalls, /proc and /sys, POSIX shared memory, libcuda). Needs an NVIDIA GPU
of compute capability 8.0 or newer and the brook, cupy and cucim packages; numba is optional
(its kernels are replaced by numpy when it is missing). ``uv sync`` installs numba on Linux
x86-64; the numpy code covers environments set up without it.
"""
from __future__ import annotations

import ctypes
import dbm
import errno
import functools
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EMPTY = pickle.dumps([])
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DRIVER = 545                # brook-cu12 needs a driver that supports CUDA 12.3 (R545)
GEOMETRY_KEYS = ('downsample_factor', 'z_chunk_depth_h', 'z_chunk_stride_h', 'yx_chunk_thickness_v',
                 'yx_stride_v', 'dust_threshold', 'max_area_threshold')
# The scheduling constants below were measured on one NVIDIA H100 80GB with Scroll 1 surface
# predictions at the default slab geometry. They set only the batch sizes and the dispatch
# timing, never the output: too large a batch costs an out-of-memory halving, too small a batch
# costs speed.
#
# Device memory of one skeletonize_batch call: base_bytes + bytes_per_voxel * voxels, and at most
# max_voxels per call. The worker's free memory, less a reserve for the next batch's
# preprocessing, bounds the voxels of each batch; a GPU with less free memory than about
# base_bytes plus that reserve gets one slab per call.
STATIC_MODEL = dict(base_bytes=12e9, bytes_per_voxel=50.0, max_voxels=1.07e9)
# Seconds per skeletonize_batch call against slabs per call (horizontal calls grow with the
# volume's Y x X, vertical ones with the z-range height). Only a scheduling prior: the parent
# rescales it with the call times it observes.
CALL_SECONDS_PRIOR = dict(
    h=[(1, 0.51), (2, 0.71), (4, 1.04), (8, 1.58), (16, 2.39), (32, 3.84), (48, 4.77), (64, 6.65),
       (96, 7.93), (128, 11.0), (256, 21.3)],
    vy=[(1, 0.30), (2, 0.41), (4, 0.56), (6, 0.69), (8, 0.81), (12, 1.03), (16, 1.18), (24, 1.57),
        (32, 1.91), (48, 2.59), (94, 4.28)],
    vx=[(1, 0.35), (2, 0.47), (4, 0.66), (6, 0.79), (8, 0.97), (12, 1.19), (16, 1.43), (24, 1.84),
        (32, 2.26), (48, 3.08), (96, 5.06)])
CALL_SECONDS_PRIOR_Z = 4096     # z height of the vertical slabs behind CALL_SECONDS_PRIOR
# preprocessing seconds per slab before any is observed
PREP_SECONDS_PRIOR = dict(h=0.025, vy=0.013, vx=0.012)
RIBBON_CAP = 128                # horizontal ribbons per call at most
PREFETCH_LEAD_S = 0.3           # margin before a running call's predicted end for the next batch
TRACK_THREADS_PER_GPU = 4
# CuPy caches freed device blocks, which Brook (allocating outside CuPy) cannot use; they are
# released when the free memory drops below this (on cards of 24 GB or less, after every batch).
FREE_BELOW_BYTES = 24e9
MONITOR_INTERVAL_S = 0.001      # device-memory sampling of the preprocessing peak (every 1 ms)
CONTEXT_TURN_TIMEOUT_S = 120    # a worker that never passes the ticket does not block the others


# ------------------------------------------------------------------------------------------------
# Availability and GPU selection

@functools.cache
def _nvidia_smi_gpus():
    """``{index: dict(cc, driver, name, bus)}`` from nvidia-smi (queried once per process)."""
    exe = shutil.which('nvidia-smi')
    if exe is None:
        raise OSError('nvidia-smi was not found (no NVIDIA driver?)')
    res = subprocess.run([exe, '--query-gpu=index,compute_cap,pci.bus_id,driver_version,name',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise OSError(f'nvidia-smi failed: {(res.stderr or res.stdout).strip()}')
    gpus = {}
    for line in res.stdout.strip().splitlines():
        idx, cc, bus, driver, name = [s.strip() for s in line.split(',', 4)]
        try:
            cc = tuple(int(x) for x in cc.split('.'))
        except ValueError:
            cc = None
        gpus[idx] = dict(cc=cc, driver=driver, name=name, bus=bus)
    return gpus


def _select_gpus(gpus, available):
    """GPU indices (nvidia-smi numbering, as strings): ``gpus``, else CUDA_VISIBLE_DEVICES, else
    every GPU in ``available``. Raises ValueError for a malformed, unknown or repeated index."""
    source = 'gpus'
    if gpus is None:
        env = os.environ.get('CUDA_VISIBLE_DEVICES')
        if env is None:
            return sorted(available, key=int)
        gpus, source = env, 'CUDA_VISIBLE_DEVICES'
    if isinstance(gpus, str):
        gpus = gpus.split(',')
    selected = []
    for item in (str(g).strip() for g in gpus):
        if not item:
            continue
        if not item.isdigit():
            raise ValueError(f'{source}: {item!r} is not a GPU index (nvidia-smi numbering)')
        g = str(int(item))
        if g not in available:
            known = ', '.join(sorted(available, key=int))
            raise ValueError(f'{source}: there is no GPU {g} (GPUs: {known})')
        if g in selected:
            raise ValueError(f'{source}: GPU {g} is listed twice')
        selected.append(g)
    if not selected:
        raise ValueError(f'{source}: no GPU selected')
    return selected


def check_available(gpus=None):
    """``(ok, reason)``: whether the GPU backend can run here, and why or why not.

    Needs Linux x86-64; the brook, cupy and cucim packages (numba is optional); and nvidia-smi
    listing every selected GPU (``gpus``, else CUDA_VISIBLE_DEVICES, else all) with compute
    capability 8.0 or newer and an NVIDIA driver R545 or newer (a version nvidia-smi reports in
    an unexpected form does not block). Creates no CUDA context.
    """
    machine = platform.machine()
    if sys.platform != 'linux' or machine != 'x86_64':
        return False, f'the GPU backend runs on Linux x86-64 only, not {sys.platform} {machine}'
    missing = [name for name in ('brook', 'cupy', 'cucim') if importlib.util.find_spec(name) is None]
    if missing:
        return False, f'Python packages not installed: {", ".join(missing)}'
    try:
        table = _nvidia_smi_gpus()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return False, str(exc)
    if not table:
        return False, 'nvidia-smi lists no GPU'
    try:
        selected = _select_gpus(gpus, table)
    except ValueError as exc:
        return False, str(exc)
    need = '.'.join(map(str, MIN_COMPUTE_CAPABILITY))
    for g in selected:
        cc = table[g]['cc']
        if cc is None or cc < MIN_COMPUTE_CAPABILITY:
            have = 'compute capability ' + '.'.join(map(str, cc)) if cc else 'an unknown compute capability'
            return False, (f'GPU {g} ({table[g]["name"]}) has {have}; Brook needs {need} or newer '
                           '(select other GPUs with --gpus)')
    driver = table[selected[0]]['driver']
    try:
        major = int(driver.split('.')[0])
    except ValueError:
        major = None
    if major is not None and major < MIN_DRIVER:
        return False, f'NVIDIA driver {driver} is older than R{MIN_DRIVER}, which brook-cu12 needs'
    return True, ', '.join(f'GPU {g} ({table[g]["name"]}, compute capability '
                           f'{".".join(map(str, table[g]["cc"]))})' for g in selected)


# ------------------------------------------------------------------------------------------------
# NUMA placement (Linux syscalls through ctypes)

MPOL_BIND = 2
_SYSCALLS = {'x86_64': dict(mbind=237, set_mempolicy=238, get_mempolicy=239),
             'aarch64': dict(mbind=235, get_mempolicy=236, set_mempolicy=237)}
_MASK_WORDS = 16


@functools.cache
def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _syscall(name, *args):
    number = _SYSCALLS.get(platform.machine(), {}).get(name)
    if number is None:
        raise OSError(errno.ENOSYS, f'{name}: unsupported on {platform.machine()}')
    r = _libc().syscall(ctypes.c_long(number), *args)
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, f'{name}: {os.strerror(e)}')
    return r


def _nodemask(nodes):
    mask = (ctypes.c_ulong * _MASK_WORDS)()
    for n in nodes:
        mask[n // 64] |= 1 << (n % 64)
    return mask, ctypes.c_ulong(64 * _MASK_WORDS + 1)


def _set_mempolicy(mode, nodes=()):
    mask, maxnode = _nodemask(nodes)
    _syscall('set_mempolicy', ctypes.c_int(mode), mask if nodes else None,
             maxnode if nodes else ctypes.c_ulong(0))


def _get_mempolicy():
    mode = ctypes.c_int(0)
    mask, maxnode = _nodemask(())
    _syscall('get_mempolicy', ctypes.byref(mode), mask, maxnode, None, ctypes.c_ulong(0))
    return mode.value, [i for i in range(64 * _MASK_WORDS) if mask[i // 64] >> (i % 64) & 1]


def _node_cpus(node):
    cpus = set()
    for part in Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip().split(','):
        if part:
            lo, _, hi = part.partition('-')
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _pci_numa_node(bus_id):
    bus = bus_id.lower()
    bus = bus[-12:] if len(bus) > 12 else bus          # nvidia-smi pads the PCI domain to 8 digits
    try:
        node = int(Path(f'/sys/bus/pci/devices/{bus}/numa_node').read_text())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def _proc_gpus():
    """``{index: dict(bus, node)}`` from /proc/driver/nvidia/gpus, or None when it is missing.

    The directory lists the GPUs by PCI bus id; sorted, that is nvidia-smi's numbering and CUDA's
    with CUDA_DEVICE_ORDER=PCI_BUS_ID. Reading it is much faster than starting nvidia-smi."""
    try:
        buses = sorted(p.name for p in Path('/proc/driver/nvidia/gpus').iterdir())
    except OSError:
        return None
    return {str(i): dict(bus=b, node=_pci_numa_node(b)) for i, b in enumerate(buses)} or None


def _bind_thread(node):
    """Run the calling thread on a NUMA node's CPUs and allocate its memory there."""
    cpus = _node_cpus(node) & os.sched_getaffinity(0) or _node_cpus(node)
    os.sched_setaffinity(0, cpus)
    _set_mempolicy(MPOL_BIND, [node])


def _try_bind_thread(node):
    """``_bind_thread``, best effort: the placement only affects speed."""
    try:
        _bind_thread(node)
    except OSError:
        pass


def _place(array, node):
    """Bind the pages of ``array`` (not yet touched) to a NUMA node, best effort."""
    page = os.sysconf('SC_PAGE_SIZE')
    length = -(-array.nbytes // page) * page
    mask, maxnode = _nodemask([node])
    try:
        _syscall('mbind', ctypes.c_void_p(array.ctypes.data), ctypes.c_ulong(length),
                 ctypes.c_int(MPOL_BIND), mask, maxnode, ctypes.c_uint(0))
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------
# Input: the direct zarr v2 reader and the bit-packing kernels

def _direct_meta(path):
    """The ``.zarray`` metadata of a local zarr v2 array the direct reader decodes, else None.

    The direct reader handles the layout of the surface predictions: 3-D uint8 in C order,
    blosc without filters, fill value 0, '/'-separated chunk keys, and a chunk x size that is a
    whole number of bytes of the bit-packed block (a multiple of 8). This one predicate decides
    between the direct reader and ``open_array``."""
    text = os.fspath(path)
    if '://' in text:
        return None
    try:
        meta = json.loads((Path(text) / '.zarray').read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    shape, chunks = meta.get('shape'), meta.get('chunks')
    ok = (meta.get('zarr_format') == 2 and isinstance(shape, list) and len(shape) == 3
          and isinstance(chunks, list) and len(chunks) == 3 and chunks[2] % 8 == 0
          and (meta.get('compressor') or {}).get('id') == 'blosc' and not meta.get('filters')
          and meta.get('order') == 'C' and meta.get('dtype') in ('|u1', '<u1')
          and meta.get('dimension_separator') == '/' and meta.get('fill_value') in (0, None))
    return meta if ok else None


def _open_input(path, open_array):
    """``(array, direct)``: the metadata as an array stand-in when the direct reader applies,
    else ``open_array(path)``."""
    meta = _direct_meta(path)
    if meta is not None:
        return SimpleNamespace(shape=tuple(meta['shape']), chunks=tuple(meta['chunks'])), True
    arr = open_array(path)
    if len(arr.shape) != 3:
        raise ValueError(f'{path}: expected a 3-D array, not shape {tuple(arr.shape)}')
    return arr, False


def _packbits_nz_py(src, dst):
    """numba kernel source: ``np.packbits(src != 0, axis=-1)`` into ``dst``."""
    nz, ny, nx = src.shape
    full = nx // 8
    rest = nx - 8 * full
    for z in range(nz):
        for y in range(ny):
            for k in range(full):
                b = 0
                for t in range(8):
                    b = (b << 1) | (src[z, y, 8 * k + t] != 0)
                dst[z, y, k] = b
            if rest:
                b = 0
                for t in range(rest):
                    b |= (src[z, y, 8 * full + t] != 0) << (7 - t)
                dst[z, y, full] = b


def _vx_columns_py(block, cols, out, z_lo, z_hi):
    """numba kernel source: ``out[i, z, y] = block[z, y, cols[i]]`` for z in [z_lo, z_hi)."""
    ny = block.shape[1]
    for z in range(z_lo, z_hi):
        for y in range(ny):
            for i in range(cols.shape[0]):
                out[i, z, y] = block[z, y, cols[i]]


_KERNELS = {}
_KERNELS_LOCK = threading.Lock()


def _packbits_kernel():
    """``(pack, name)``: bit-packing of uint8 chunks along x, compiled with numba (which releases
    the GIL, so that decode threads run in parallel) or ``np.packbits`` without numba."""
    with _KERNELS_LOCK:
        if 'pack' not in _KERNELS:
            try:
                import numba
            except ImportError:
                _KERNELS['pack'] = (functools.partial(np.packbits, axis=-1), 'numpy')
            else:
                kernel = numba.njit(nogil=True, cache=True)(_packbits_nz_py)
                kernel(np.zeros((1, 1, 9), np.uint8), np.zeros((1, 1, 2), np.uint8))

                def pack(chunk):
                    out = np.empty(chunk.shape[:-1] + (-(-chunk.shape[-1] // 8),), np.uint8)
                    kernel(chunk, out)
                    return out
                _KERNELS['pack'] = (pack, 'numba')
    return _KERNELS['pack']


def _vx_kernel():
    """The numba kernel of the vx column block, or None without numba."""
    _, name = _packbits_kernel()
    with _KERNELS_LOCK:
        if 'vx' not in _KERNELS:
            _KERNELS['vx'] = None
            if name == 'numba':
                import numba
                vx = numba.njit(nogil=True, cache=True)(_vx_columns_py)
                vx(np.zeros((1, 1, 2), np.uint8), np.zeros(1, np.int64), np.zeros((1, 1, 1), np.uint8),
                   0, 1)
                _KERNELS['vx'] = vx
    return _KERNELS['vx']


def _warm_numba():
    """Load the parent's other numba kernels (vx columns, fast_tracks) in the background."""
    import fast_tracks
    _vx_kernel()
    fast_tracks.warmup()


class _Decoder:
    """Binarizes z0..z1 of the input into the bit-packed block, one task per (chunk row, chunk
    column in y), and records the foreground's bounding box at the sampled slices."""

    def __init__(self, path, arr, direct, z0, z1, sampled):
        self.path, self.arr, self.direct = os.fspath(path), arr, direct
        self.z0, self.z1, self.sampled = z0, z1, sampled
        self.C = tuple(int(c) for c in arr.chunks)
        self.block = None
        self.boxes = []
        self.pack, self.pack_name = _packbits_kernel()

    def rows(self):
        return list(range(self.z0 // self.C[0], (self.z1 - 1) // self.C[0] + 1))

    def tasks(self, row):
        return [(row, cy) for cy in range(-(-self.arr.shape[1] // self.C[1]))]

    def _store(self, z, y0, x0, chunk):
        """Store ``chunk`` (nonzero is foreground), whose first slice is z, at y0, x0 (x0 a
        multiple of 8)."""
        for s in self.sampled:
            if z <= s < z + chunk.shape[0]:
                sl = chunk[s - z] > 0
                if sl.any():
                    ys, xs = np.flatnonzero(sl.any(axis=1)), np.flatnonzero(sl.any(axis=0))
                    self.boxes.append((y0 + ys[0], x0 + xs[0], y0 + ys[-1], x0 + xs[-1]))
        bits = self.pack(chunk)
        self.block[z - self.z0:z - self.z0 + chunk.shape[0], y0:y0 + chunk.shape[1],
                   x0 // 8:x0 // 8 + bits.shape[2]] = bits

    def decode(self, row, cy):
        cz, cyy, cxx = self.C
        Z, Y, X = self.arr.shape
        zr0 = row * cz
        lo, hi = max(self.z0 - zr0, 0), min(self.z1 - zr0, cz, Z - zr0)
        y0 = cy * cyy
        ny = min(cyy, Y - y0)
        if not self.direct:
            # The whole x extent in one piece, so that any chunk x size keeps the bytes aligned;
            # binarized with > 0 as the Kimimaro path does (signed and float inputs included).
            strip = np.asarray(self.arr[zr0 + lo:zr0 + hi, y0:y0 + ny, :]) > 0
            self._store(zr0 + lo, y0, 0, strip.view(np.uint8))
            return row
        from numcodecs import blosc
        buf = np.empty(cz * cyy * cxx, np.uint8)
        for cx in range(-(-X // cxx)):
            try:
                with open(f'{self.path}/{row}/{cy}/{cx}', 'rb') as f:
                    raw = f.read()
            except FileNotFoundError:            # an absent chunk is all fill value 0: the block is zeroed
                continue
            blosc.decompress(raw, buf)
            chunk = buf.reshape(cz, cyy, cxx)[lo:hi, :ny, :min(cxx, X - cx * cxx)]
            self._store(zr0 + lo, y0, cx * cxx, chunk)
        return row


# ------------------------------------------------------------------------------------------------
# GPU worker process

# Out-of-memory errors come as different types: CuPy's OutOfMemoryError, and CUDA's
# cudaErrorMemoryAllocation ('out of memory') raised by CuPy, cuCIM or Brook (brook.CudaError, a
# RuntimeError); so the text is matched.
_OOM_TEXT = ('out of memory', 'outofmemory', 'memoryallocation')


def _is_oom(exc):
    text = f'{type(exc).__name__}: {exc}'.lower()
    return isinstance(exc, MemoryError) or any(s in text for s in _OOM_TEXT)


def _gpu_worker(device, node, cfg, tasks, results, data, turn):
    """One GPU. In on ``tasks``: ('job', config), ('batch', bid, slabs, extra), ('stop',). Out on
    ``results``: ('ready', info), ('fatal', traceback), ('skel_start', bid, t), ('result', bid,
    stats, empty flags), ('error', bid, traceback). Out on ``data``: each batch's skeletons as raw
    arrays (see send_packed)."""
    if node is not None:
        _try_bind_thread(node)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    # Brook 0.1.0 reads BROOK_BATCH_BUDGET as the memory budget of one skeletonize_batch call
    # (default: half the free memory, splitting larger calls). This pipeline sizes its calls with
    # its own memory model and halves a batch on out-of-memory, so Brook's split is turned off.
    os.environ.setdefault('BROOK_BATCH_BUDGET', '1e15')
    send_lock = threading.Lock()
    data_lock = threading.Lock()

    def send(msg):
        with send_lock:
            results.send(msg)

    def take_turn():
        with turn['cond']:
            turn['cond'].wait_for(lambda: turn['value'].value >= cfg['rank'],
                                  timeout=CONTEXT_TURN_TIMEOUT_S)

    def pass_turn():
        with turn['cond']:
            turn['value'].value += 1
            turn['cond'].notify_all()

    def driver_context(box):
        # cuInit and the primary context through the driver API, in a thread while the main
        # thread imports (the driver work runs without the GIL); CuPy then attaches to the same
        # primary context.
        try:
            take_turn()
            lib = ctypes.CDLL('libcuda.so.1')
            r = lib.cuInit(0)
            if r:
                raise RuntimeError(f'cuInit: CUresult {r}')
            d = ctypes.c_int()
            r = lib.cuDeviceGet(ctypes.byref(d), 0)
            if r:
                raise RuntimeError(f'cuDeviceGet: CUresult {r}')
            c = ctypes.c_void_p()
            r = lib.cuDevicePrimaryCtxRetain(ctypes.byref(c), d)
            if r:
                raise RuntimeError(f'cuDevicePrimaryCtxRetain: CUresult {r}')
        except BaseException:
            box['error'] = traceback.format_exc()
        finally:
            pass_turn()

    try:
        box = {}
        early = threading.Thread(target=driver_context, args=(box,), daemon=True)
        early.start()
        # cuCIM imports its optional image-I/O extension (cucim.clara, a large native library)
        # when the package loads and skips it on ImportError, which this entry produces. Only
        # the label function is used here, and skipping the extension shortens every start-up.
        sys.modules.setdefault('cucim.clara', None)
        import cupy as cp
        import cupyx
        import brook
        from brook.device import memory_info
        from cucim.skimage.measure import label as gpu_label
        early.join()
        if 'error' in box:
            raise RuntimeError(box['error'])
        shifts = cp.asarray(np.arange(7, -1, -1).astype(np.uint8))
        cp.cuda.get_current_stream().synchronize()
        gather_threads = cfg['gather_threads']
        gather_pool = ThreadPoolExecutor(gather_threads)
        pool = cp.get_default_memory_pool()
        job = {}
        mapped = {}

        def attach(name, shape):
            if name not in mapped:
                s = shared_memory.SharedMemory(name=name, track=False)     # the parent unlinks it
                mapped[name] = (s, np.ndarray(shape, np.uint8, buffer=s.buf))
            return mapped[name][1]

        def split_copy(dst, src):
            ax = 0 if src.shape[0] >= gather_threads or src.ndim == 1 else 1
            step = -(-src.shape[ax] // gather_threads)

            def part(i):
                sl = (slice(i, i + step),) if ax == 0 else (slice(None), slice(i, i + step))
                dst[sl] = src[sl]
            list(gather_pool.map(part, range(0, src.shape[ax], step)))

        def to_device(src):
            stage = cupyx.empty_pinned(src.shape, np.uint8)        # gathered by several threads
            split_copy(stage, src)
            dev = cp.asarray(stage)
            cp.cuda.get_current_stream().synchronize()
            return dev

        def unpack(kind, a, b, col):
            """The binarized slab as a bool array on the device."""
            block, X = job['block'], job['X']
            if kind == 'h':
                dev, lo, hi = to_device(block[a:b]), 0, X
            elif kind == 'vy':
                dev, lo, hi = to_device(block[:, a:b, :]), 0, X
            else:                                               # vx: rows of the column block
                name, shape, i = col
                b0, b1 = a // 8, (b - 1) // 8 + 1
                lo, hi = a - 8 * b0, b - 8 * b0
                dev = to_device(attach(name, shape)[i:i + (b1 - b0)]).transpose(1, 2, 0)
            bits = ((dev[..., None] >> shifts) & 1).reshape(dev.shape[:-1] + (dev.shape[-1] * 8,))
            return bits[..., lo:hi].astype(cp.bool_)

        def prepare(pred, ds, lo_count, hi_count):
            """prepare_cc_labels on the GPU: 6-connected components, max-pool by ds, and the
            components of [lo_count, hi_count) voxels, as Fortran-ordered int32."""
            cc = gpu_label(pred, connectivity=1)
            shape = tuple(s - s % ds for s in cc.shape)
            cc = cc[:shape[0], :shape[1], :shape[2]]
            cc = cc.reshape(shape[0] // ds, ds, shape[1] // ds, ds, shape[2] // ds, ds).max(axis=(1, 3, 5))
            counts = cp.bincount(cc.ravel())
            keep = (counts >= lo_count) & (counts < hi_count)
            keep[0] = False
            return cp.asfortranarray(cp.where(keep[cc], cc, 0).astype(cp.int32))

        brook.warmup()
        warm_stream = cp.cuda.Stream(non_blocking=True)
        with warm_stream:                                       # loads the cuCIM and CuPy kernels
            prepare(cp.random.random((8, 64, 64)) > 0.5, 4, 1, 10 ** 6)
            warm_stream.synchronize()
        pool.free_all_blocks()

        class Monitor(threading.Thread):
            """Peak device memory in use, sampled; measures what preprocessing needs."""

            def __init__(self):
                super().__init__(daemon=True)
                self.peak = self.used()
                self.running = True

            @staticmethod
            def used():
                free, total = cp.cuda.runtime.memGetInfo()
                return total - free

            def reset(self):
                self.peak = self.used()
                return self.peak

            def run(self):
                while self.running:
                    u = self.used()
                    if u > self.peak:
                        self.peak = u
                    time.sleep(MONITOR_INTERVAL_S)

        mon = Monitor()
        mon.start()
        send(('ready', dict(free=int(memory_info()[0]))))
    except BaseException:
        send(('fatal', traceback.format_exc()))
        return

    def skeletonize(samples, stats):
        """Brook on a batch; after out-of-memory, on its halves."""
        try:
            return [brook.skeletonize_batch(samples, output='packed', **job['skeletonize_kwargs'])]
        except Exception as exc:
            if not _is_oom(exc) or len(samples) == 1:
                raise
            stats['oom_halvings'] += 1
            pool.free_all_blocks()
            half = len(samples) // 2
            return skeletonize(samples[:half], stats) + skeletonize(samples[half:], stats)

    sender = ThreadPoolExecutor(1)

    def send_packed(bid, packed):
        """A batch's skeletons to the parent: a small header, then every array as raw bytes, on a
        pipe of their own so that control messages do not wait behind them."""
        try:
            arrays = [[np.ascontiguousarray(a) for a in arrs] for _, arrs in packed]
            head = (bid, [(key, [(a.dtype.str, a.shape) for a in arrs])
                          for (key, _), arrs in zip(packed, arrays)])
            with data_lock:
                data.send(head)
                for arrs in arrays:
                    for a in arrs:
                        data.send_bytes(memoryview(a).cast('B') if a.size else b'')
        except BaseException:
            send(('error', bid, traceback.format_exc()))

    def maybe_free():
        if memory_info()[0] < FREE_BELOW_BYTES:
            pool.free_all_blocks()

    def prep_batch(bid, slabs, extra):
        """Unpack and preprocess a batch's slabs on the current stream."""
        t0 = time.perf_counter()
        stats = dict(oom_halvings=0)
        stream = cp.cuda.get_current_stream()
        mon.reset()
        prep_peak = 0
        samples, keys, out = [], [], {}
        vx_col = (extra or {}).get('vx_col', {})
        ds, lo_count, hi_count = job['ds'], job['lo_count'], job['hi_count']
        for key, kind, a, b in slabs:
            before = mon.reset()
            col = (extra['vx_name'], extra['vx_shape'], vx_col[key]) if key in vx_col else None
            pred = unpack(kind, a, b, col)
            if not bool(pred.any()):
                out[key] = None
                continue
            samples.append(prepare(pred, ds, lo_count, hi_count))
            keys.append(key)
            del pred
            stream.synchronize()
            prep_peak = max(prep_peak, mon.peak - before)
        stats['prep_s'] = time.perf_counter() - t0
        stats['voxels'] = int(sum(s.size for s in samples))
        stats['prep_peak'] = int(prep_peak)
        return dict(bid=bid, slabs=slabs, samples=samples, keys=keys, out=out, stats=stats)

    def run_batch(p):
        bid, slabs, keys, out, stats = p['bid'], p['slabs'], p['keys'], p['out'], p['stats']
        t0 = time.time()
        send(('skel_start', bid, t0))
        mon.reset()
        parts = skeletonize(p['samples'], stats) if p['samples'] else []
        stats['skel_s'] = time.time() - t0
        i = 0
        for part in parts:
            for j in range(len(part)):
                q = part[j]
                out[keys[i]] = (q.v_off, q.e_off, q.vertices, q.edges)
                i += 1
        assert i == len(keys)
        del parts
        p['samples'] = None
        maybe_free()
        stats['free_after'] = int(memory_info()[0])
        send(('result', bid, stats, [(key, out[key] is None) for key, *_ in slabs]))
        packed = [(key, out[key]) for key, *_ in slabs if out[key] is not None]
        if packed:
            sender.submit(send_packed, bid, packed)

    # Two threads per worker: the prep thread prepares batch k + 1 on its own stream while the
    # skel thread runs Brook on batch k.
    prep_q, skel_q = queue.Queue(), queue.Queue()
    prep_stream = cp.cuda.Stream(non_blocking=True)

    def prep_loop():
        with prep_stream:
            while True:
                item = prep_q.get()
                if item is None:
                    skel_q.put(None)
                    return
                try:
                    p = prep_batch(*item)
                    prep_stream.synchronize()                   # Brook reads the inputs on its own stream
                    skel_q.put(p)
                except BaseException:
                    send(('error', item[0], traceback.format_exc()))

    def skel_loop():
        while True:
            p = skel_q.get()
            if p is None:
                return
            try:
                run_batch(p)
            except BaseException:
                send(('error', p['bid'], traceback.format_exc()))

    threads = [threading.Thread(target=loop, daemon=True) for loop in (prep_loop, skel_loop)]
    for th in threads:
        th.start()
    while True:
        try:
            msg = tasks.recv()
        except EOFError:
            break
        if msg[0] == 'stop':
            break
        if msg[0] == 'job':
            job.clear()
            job.update(msg[1])
            job['block'] = attach(job['block_name'], tuple(job['block_shape']))
            job['lo_count'] = max(1, job['dust'] // job['ds'] ** 3)
            job['hi_count'] = max(1, job['max_area'] // job['ds'] ** 3)
            continue
        _, bid, slabs, extra = msg
        prep_q.put((bid, slabs, extra))
    prep_q.put(None)
    for th in threads:
        th.join()
    sender.shutdown(wait=True)
    mon.running = False
    job.clear()
    shms = [s for s, _ in mapped.values()]
    mapped.clear()
    for s in shms:
        try:
            s.close()
        except BufferError:
            pass


# ------------------------------------------------------------------------------------------------
# DBM writer process

def _dbm_writer(conn, reply):
    """Stores the parent's values in its own process. In: ('open', path), ('put', key, value),
    ('close',), None (exit). Out: ('closed', count) or ('error', traceback)."""
    db = None
    n = 0
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            if msg[0] == 'put':
                db[msg[1]] = msg[2]
                n += 1
            elif msg[0] == 'open':
                db, n = dbm.open(msg[1], 'w'), 0
            elif msg[0] == 'close':
                db.close()
                db = None
                reply.send(('closed', n))
    except BaseException:
        try:
            reply.send(('error', traceback.format_exc()))
        except OSError:
            pass
    finally:
        if db is not None:
            db.close()


# ------------------------------------------------------------------------------------------------
# The parent: processes, threads and shared blocks

class _Pool:
    """The GPU workers, the DBM writer, the parent's thread pools and the shared blocks."""

    def __init__(self, gpus, gpu_node, numa, nodes):
        self.gpus, self.gpu_node, self.numa = gpus, gpu_node, numa
        self.events = queue.Queue()
        self.gstate = [dict(ready=False, free=0) for _ in gpus]
        self.fatal = [None] * len(gpus)                 # a worker's start-up traceback
        self.reader_done = [threading.Event() for _ in gpus]
        self.workers, self.task_conns = [], []
        self.writer = None
        self.decode_pool = None
        self.track_pools = {}
        self.shms = []
        self.block_node = nodes[0] if numa else None
        self.closed = False
        ctx = mp.get_context('spawn')                   # Brook refuses CUDA state inherited by fork
        per_node = Counter(gpu_node.values()) if numa else Counter({None: len(gpus)})
        affinity = os.sched_getaffinity(0)
        ncpu = len(affinity)
        node_ncpu = {n: (len(_node_cpus(n) & affinity) if n is not None else ncpu) for n in nodes}
        gather = {n: max(1, min(16, node_ncpu[n] // (2 * per_node[n]))) for n in nodes}
        # the CUDA-context ticket; kept on the pool because a child unpickles it only when it runs
        self.turn = dict(cond=ctx.Condition(), value=ctx.Value('i', 0))
        # OpenBLAS starts a thread per CPU in every process that loads numpy; the children do no
        # BLAS work.
        blas = 'OPENBLAS_NUM_THREADS' not in os.environ
        if blas:
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
        try:
            result_conns, data_conns = [], []
            for rank, g in enumerate(gpus):
                n = gpu_node[g] if numa else None
                task_r, task_w = ctx.Pipe(duplex=False)
                res_r, res_w = ctx.Pipe(duplex=False)
                data_r, data_w = ctx.Pipe(duplex=False)
                cfg = dict(rank=rank, gather_threads=gather[n])
                p = ctx.Process(target=_gpu_worker, args=(g, n, cfg, task_r, res_w, data_w, self.turn),
                                daemon=True)
                p.start()
                for c in (task_r, res_w, data_w):
                    c.close()
                self.workers.append(p)
                self.task_conns.append(task_w)
                result_conns.append(res_r)
                data_conns.append(data_r)
            w_r, self.w_w = ctx.Pipe(duplex=False)
            self.r_r, r_w = ctx.Pipe(duplex=False)
            self.writer = ctx.Process(target=_dbm_writer, args=(w_r, r_w), daemon=True)
            self.writer.start()
            w_r.close()
            r_w.close()
        except BaseException:
            self.close()
            raise
        finally:
            if blas:
                os.environ.pop('OPENBLAS_NUM_THREADS', None)
        for i, (res, dat) in enumerate(zip(result_conns, data_conns)):
            threading.Thread(target=self._reader, args=(i, res), daemon=True).start()
            threading.Thread(target=self._data_reader, args=(i, dat), daemon=True).start()
        self.decode_threads = max(2, min(32, ncpu // 3))
        if numa:
            self.decode_pool = ThreadPoolExecutor(self.decode_threads, initializer=_try_bind_thread,
                                                  initargs=(self.block_node,))
        else:
            self.decode_pool = ThreadPoolExecutor(self.decode_threads)
        for n in nodes:
            k = max(1, TRACK_THREADS_PER_GPU * per_node[n])
            if n is None:
                self.track_pools[n] = ThreadPoolExecutor(k)
            else:
                self.track_pools[n] = ThreadPoolExecutor(k, initializer=_try_bind_thread, initargs=(n,))

    def send_task(self, i, msg):
        try:
            self.task_conns[i].send(msg)
        except OSError:
            # a worker that failed to start sent its traceback before it exited
            self.reader_done[i].wait(5)
            self.workers[i].join(1)
            if self.fatal[i]:
                raise RuntimeError(
                    f'GPU worker {self.gpus[i]} failed to start:\n{self.fatal[i]}') from None
            code = self.workers[i].exitcode
            raise RuntimeError(f'GPU worker {self.gpus[i]} exited (exit code {code})') from None

    def _reader(self, i, conn):
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                self.reader_done[i].set()
                self.events.put(('gpu_closed', i, None))
                return
            if msg[0] == 'fatal':
                self.fatal[i] = msg[1]
            self.events.put(('gpu', i, msg))

    def _data_reader(self, i, conn):
        """Rebuilds the skeleton arrays a worker sends raw (see send_packed)."""
        while True:
            try:
                bid, items = conn.recv()
                out = [(key, [np.frombuffer(conn.recv_bytes(), np.dtype(dt)).reshape(shape)
                              for dt, shape in specs]) for key, specs in items]
            except (EOFError, OSError):
                return
            self.events.put(('packed', i, out))

    def new_block(self, shape):
        """``(name, array)``: a zero-filled uint8 block in POSIX shared memory."""
        size = int(np.prod(shape))
        free = shutil.disk_usage('/dev/shm').free if os.path.isdir('/dev/shm') else None
        if free is not None and free < size:
            raise RuntimeError(f'the decoded block needs {size / 1e9:.2f} GB of shared memory, but '
                               f'/dev/shm has {free / 1e9:.2f} GB free: enlarge /dev/shm or extract a '
                               'shorter z-range')
        shm = shared_memory.SharedMemory(create=True, size=max(1, size))
        self.shms.append(shm)
        arr = np.ndarray(shape, np.uint8, buffer=shm.buf)
        if self.block_node is not None:
            _place(arr, self.block_node)
        return shm.name, arr

    def close(self):
        if self.closed:
            return
        self.closed = True
        for c in self.task_conns:
            try:
                c.send(('stop',))
            except OSError:
                pass
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join()
        for pool_ in [self.decode_pool, *self.track_pools.values()]:
            if pool_ is not None:
                pool_.shutdown(cancel_futures=True)
        for shm in self.shms:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            try:
                shm.close()
            except BufferError:              # an array still views it; the mapping goes with it
                pass
        self.shms.clear()
        if self.writer is not None:
            try:
                self.w_w.send(None)
            except OSError:
                pass
            self.writer.join(timeout=60)
            if self.writer.is_alive():
                self.writer.terminate()


# ------------------------------------------------------------------------------------------------
# One z-range into one DBM

class _Job:
    """One z-range into one DBM: decodes the input, plans the slabs, dispatches batches to the GPU
    workers, turns the skeletons into tracks and writes them in key order."""

    def __init__(self, pool, path, arr, direct, z_min, z_max, out, geometry, skeletonize_kwargs, say):
        self.pool, self.arr, self.say = pool, arr, say
        self.out = Path(out)
        self.geometry = geometry
        self.skeletonize_kwargs = dict(skeletonize_kwargs, progress=False)
        Z, Y, X = arr.shape
        self.z_min, self.z_max = z_min, z_max
        self.z0, self.z1 = min(z_min, Z), min(z_max, Z)
        step = max(1, (self.z1 - self.z0) // 20)             # the slices find_yx_range samples
        self.sampled = list(range(self.z0, self.z1, step))
        self.block_shape = (self.z1 - self.z0, Y, (X + 7) // 8)
        self.decoder = _Decoder(path, arr, direct, self.z0, self.z1, self.sampled)

    def start_decode(self):
        self.block_name, self.decoder.block = self.pool.new_block(self.block_shape)
        self.rows = self.decoder.rows()
        self.row_left = {r: len(self.decoder.tasks(r)) for r in self.rows}
        self.decode_started = time.time()
        for r in self.rows:
            for task in self.decoder.tasks(r):
                fut = self.pool.decode_pool.submit(self.decoder.decode, *task)
                fut.add_done_callback(lambda f: self.pool.events.put(('decoded', None, f)))

    def run(self):
        """The event loop: decoding, planning, dispatch, results, tracks and ordered writes.
        Returns the number of keys written.

        Events on ``pool.events``, as (kind, GPU position or None, payload):
          ('decoded', None, future)    a decode task finished; its result is the chunk row
          ('gpu', i, message)          from worker i: ('ready', info), ('fatal', traceback),
                                       ('error', bid, traceback), ('skel_start', bid, t) or
                                       ('result', bid, stats, empty flags)
          ('packed', i, items)         a batch's skeletons, [(key, (v_off, e_off, vertices, edges))]
          ('tracks', None, future)     a track job finished; its result is (key, pickled tracks)
          ('vxpart', None, future)     a part of the vx column block is built
          ('gpu_closed', i, None)      worker i closed its result pipe
        When no event arrives within 0.02 s, the loop queues the writes it can, dispatches, and
        checks that the workers and the DBM writer are alive."""
        pool, say, arr, geo = self.pool, self.say, self.arr, self.geometry
        gpus, n = pool.gpus, len(pool.gpus)
        Z, Y, X = arr.shape
        z0, z1, z_min, z_max = self.z0, self.z1, self.z_min, self.z_max
        ds = geo['downsample_factor']
        cz = self.decoder.C[0]
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with dbm.open(str(self.out), 'c') as db:
            existing = {k.decode() for k in db.keys()}
        pool.w_w.send(('open', str(self.out)))
        cfg = dict(block_name=self.block_name, block_shape=self.block_shape, X=X, ds=ds,
                   dust=geo['dust_threshold'], max_area=geo['max_area_threshold'],
                   skeletonize_kwargs=self.skeletonize_kwargs)
        for i in range(n):
            pool.send_task(i, ('job', cfg))

        write_q = queue.Queue()
        wbox = {}

        def writer():
            try:
                while True:
                    item = write_q.get()
                    if item is None:
                        break
                    pool.w_w.send(('put',) + item)
                pool.w_w.send(('close',))
                wbox['reply'] = pool.r_r.recv()
            except (OSError, EOFError) as exc:
                wbox['reply'] = ('error', f'{type(exc).__name__}: {exc}')
                try:
                    if pool.r_r.poll(1):
                        wbox['reply'] = pool.r_r.recv()
                except (OSError, EOFError):
                    pass
        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()

        order, slabs = [], {}

        def add(key, kind, a, b, offset, shape):
            if key in existing:
                return
            order.append(key)
            slabs[key] = dict(kind=kind, a=a, b=b, offset=offset, voxels=int(np.prod(shape)))
        for z in range(z_min, z_max, geo['z_chunk_stride_h']):
            hi = min(z + geo['z_chunk_depth_h'], z_max, Z)
            if hi <= z:
                continue
            add(f'h:{z}', 'h', z - z0, hi - z0, (z, 0, 0), ((hi - z) // ds, Y // ds, X // ds))
        ready = {p: deque() for p in ('h', 'vy', 'vx')}
        total = Counter(h=len(order))
        h_waiting = deque(order)
        rows_done = set()
        decode_left = sum(self.row_left.values())
        say(f'z {z0}..{z1}: {len(order)} horizontal ribbons to compute '
            f'({len(existing)} keys already in the DBM)')

        # device memory kept free for the next batch's preprocessing: a default per pass (16 bytes
        # per full-resolution voxel of one slab) until the workers report the largest
        # preprocessing peak of that pass
        default_reserve = {'h': 16 * geo['z_chunk_depth_h'] * Y * X}
        default_reserve['vy'] = default_reserve['vx'] = (16 * (z1 - z0) * max(Y, X)
                                                         * geo['yx_chunk_thickness_v'])
        observed_reserve = dict(h=None, vy=None, vx=None)
        inflight = {}                                   # bid -> (GPU position, keys, pass, slabs)
        by_gpu = [[] for _ in gpus]                     # in-flight batch ids per GPU, oldest first
        started = {}                                    # bid -> time Brook started on it
        tracks_done = {}
        pending_tracks = 0
        nw = 0
        v_planned = False
        vxb = dict(ready=False, name=None, shape=None, col=None, left=0)
        next_bid = 0

        def budget(i, p):
            """Voxels GPU i may take in one call of pass p."""
            avail = pool.gstate[i]['free']
            reserve = observed_reserve[p] if observed_reserve[p] is not None else default_reserve[p]
            # a margin of 5 % plus 256 MB for what the model does not count (allocator rounding
            # and fragmentation)
            usable = avail - reserve - (0.05 * avail + 256e6)
            return max(0.0, min(STATIC_MODEL['max_voxels'],
                                (usable - STATIC_MODEL['base_bytes']) / STATIC_MODEL['bytes_per_voxel']))

        # Brook's time per call against slabs per call: CALL_SECONDS_PRIOR interpolated log-log,
        # times the median observed/table ratio of this run's calls of at least half the size
        # (before any, 1.1, a margin on the prior; vertical calls scaled by the z-range height,
        # clamped to 0.1..1 so that the prior is not scaled far outside the measured height)
        table = {p: (np.log([x for x, _ in v]), np.log([y for _, y in v]))
                 for p, v in CALL_SECONDS_PRIOR.items()}
        obs_skel = {k: [] for k in table}
        obs_prep = {k: [] for k in table}
        vscale = min(1.0, max(0.1, (z1 - z0) / CALL_SECONDS_PRIOR_Z))

        def table_s(p, k):
            xs, ys = table[p]
            lk = math.log(max(k, 1))
            if lk <= xs[-1]:
                y = float(np.interp(lk, xs, ys))
            else:
                y = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (lk - xs[-1])
            return math.exp(y)

        def predict_skel(p, k):
            # Only calls of at least half this size correct the table: a small call that ran
            # beside the next batch's preprocessing is slower per slab than a large one.
            o = [(m, t) for m, t in obs_skel[p] if m and m >= k / 2]
            if o:
                ratio = float(np.median([t / table_s(p, m) for m, t in o]))
            else:
                ratio = 1.1 * (vscale if p != 'h' else 1.0)
            return ratio * table_s(p, k)

        def predict_prep(p, k):
            o = obs_prep[p]
            return (float(np.median([t / m for m, t in o if m])) if o else PREP_SECONDS_PRIOR[p]) * k

        def can_take(i, p_next=None, n_next=0):
            """Whether GPU i may get a batch now: when idle; when busy, once Brook runs its only
            batch and that call is predicted to end within the next batch's preparation."""
            if not pool.gstate[i]['ready']:
                return False
            q = by_gpu[i]
            if not q:
                return True
            if not (len(q) == 1 and q[0] in started):
                return False
            if p_next is None:
                return True
            _, _, p_run, k_run = inflight[q[0]]
            end = started[q[0]] + predict_skel(p_run, k_run)
            return time.time() >= end - predict_prep(p_next, n_next) - PREFETCH_LEAD_S

        def free_time(j, now):
            """Predicted time at which GPU j finishes what it holds."""
            t = now
            for b_ in by_gpu[j]:
                _, _, p_, k_ = inflight[b_]
                d = predict_skel(p_, k_)
                if b_ in started:
                    t = started[b_] + d
                else:
                    t = max(t, now + predict_prep(p_, k_)) + d
            return max(t, now)

        def balanced_share(i, q):
            """Slabs of vertical pass q for GPU i such that, if every ready GPU took its share of
            the pass's remaining slabs after what it holds, all would finish together
            (water-filling on predicted free times, with the call time linearized around an equal
            share)."""
            now = time.time()
            fs = {j: free_time(j, now) for j in range(n) if pool.gstate[j]['ready']}
            R = sum(len(b_) for b_ in ready[q])
            k0 = max(1, R // max(1, len(fs)))
            k1 = max(k0 + 1, 2 * k0)
            slope = max((predict_skel(q, k1) - predict_skel(q, k0)) / (k1 - k0), 1e-3)
            c0 = max(0.0, predict_skel(q, k0) - slope * k0)
            f = sorted(fs.values())
            acc, T = 0.0, None
            for k in range(1, len(f) + 1):
                acc += f[k - 1] + c0
                T = (R * slope + acc) / k
                if k == len(f) or T <= f[k] + c0:
                    break
            share = int(round((T - fs[i] - c0) / slope))
            if share <= 0 and fs[i] <= min(fs.values()) + 1e-6:
                share = 1      # the earliest free GPU takes at least one slab: small shares round to 0
            return share

        def dispatch():
            nonlocal next_bid
            for i in sorted(range(n), key=lambda i: len(by_gpu[i])):      # idle GPUs first
                if not can_take(i):
                    continue
                p = share = None
                for q in ('h', 'vy', 'vx'):
                    if not ready[q] or (q == 'vx' and not vxb['ready']):
                        continue
                    if q != 'h':
                        m = balanced_share(i, q)
                        if m <= 0:                              # other GPUs finish this pass sooner
                            continue
                        share = m
                    p = q
                    break
                if p is None:
                    continue
                even_h = max(1, math.ceil(total['h'] / n))      # an equal share of the ribbons
                fair_h = min(even_h, RIBBON_CAP)
                if by_gpu[i]:                                   # a prefetch: only when it is time
                    if p == 'h' and decode_left > 0 and len(ready['h']) < fair_h:
                        continue                                # rows still arriving: full batches only
                    n_est = share if share is not None else min(len(ready['h']), even_h)
                    if not can_take(i, p, n_est):
                        continue
                b = budget(i, p)
                take, vox = [], 0
                if share is not None:                           # an interleaved pick of `share` slabs
                    rem = ready[p][0]
                    m = min(share, len(rem))
                    pick = set()
                    for j in sorted(set(np.linspace(0, len(rem) - 1, m).round().astype(int).tolist())):
                        k = rem[j]
                        if take and vox + slabs[k]['voxels'] > b:
                            break
                        take.append(k)
                        pick.add(j)
                        vox += slabs[k]['voxels']
                    rest = [k for j, k in enumerate(rem) if j not in pick]
                    if rest:
                        ready[p][0] = rest
                    else:
                        ready[p].popleft()
                else:
                    while ready['h'] and len(take) < fair_h:
                        k = ready['h'][0]
                        if take and vox + slabs[k]['voxels'] > b:
                            break
                        take.append(ready['h'].popleft())
                        vox += slabs[k]['voxels']
                bid = next_bid
                next_bid += 1
                inflight[bid] = (i, take, p, len(take))
                by_gpu[i].append(bid)
                extra = None
                vx_keys = [k for k in take if slabs[k]['kind'] == 'vx']
                if vx_keys:
                    extra = dict(vx_name=vxb['name'], vx_shape=vxb['shape'],
                                 vx_col={k: vxb['col'][slabs[k]['a'] // 8] for k in vx_keys})
                batch = [(k, *(slabs[k][f] for f in ('kind', 'a', 'b'))) for k in take]
                pool.send_task(i, ('batch', bid, batch, extra))

        def flush():
            nonlocal nw
            while nw < len(order) and order[nw] in tracks_done:
                k = order[nw]
                write_q.put((k, tracks_done.pop(k)))
                nw += 1

        def plan_vertical():
            nonlocal v_planned
            min_yx, max_yx = np.array([Y, X]), np.array([0, 0])
            for y_lo, x_lo, y_hi, x_hi in self.decoder.boxes:
                min_yx = np.minimum(min_yx, [y_lo, x_lo])
                max_yx = np.maximum(max_yx, [y_hi, x_hi])
            say(f'yx range (full-res): {min_yx} .. {max_yx}')
            for axis, prefix in (('y', 'vy'), ('x', 'vx')):
                lo, hi = (min_yx[0], max_yx[0]) if axis == 'y' else (min_yx[1], max_yx[1])
                along = Y if axis == 'y' else X
                keys = []
                for w in range(lo, hi, geo['yx_stride_v']):
                    key = f'{prefix}:{w}'
                    w_max = min(w + geo['yx_chunk_thickness_v'], along)
                    if w_max - w < ds:
                        if key not in existing:
                            order.append(key)
                            tracks_done[key] = EMPTY
                        continue
                    if axis == 'y':
                        shape, off = ((z1 - z0) // ds, (w_max - w) // ds, X // ds), (z0, w, 0)
                    else:
                        shape, off = ((z1 - z0) // ds, Y // ds, (w_max - w) // ds), (z0, 0, w)
                    add(key, prefix, w, w_max, off, shape)
                    if key in slabs:
                        keys.append(key)
                if keys:
                    ready[prefix].append(list(keys))            # one list; batches are cut at dispatch
                    total[prefix] = len(keys)
            v_planned = True
            say(f'{total["vy"] + total["vx"]} vertical slabs to compute '
                f'({total["vy"]} vy, {total["vx"]} vx)')
            pairs = [(s['a'], s['b']) for s in slabs.values() if s['kind'] == 'vx']
            if pairs:
                build_vx_block(pairs)
            else:
                vxb['ready'] = True

        def build_vx_block(pairs):
            """The byte columns the vx slabs need, transposed to (column, z, y) in shared memory,
            so that a worker uploads each vx slab as one contiguous piece."""
            need = sorted({c for a, b in pairs for c in range(a // 8, (b - 1) // 8 + 1)})
            shape = (len(need), z1 - z0, Y)
            name, out = pool.new_block(shape)
            vxb.update(col={c: i for i, c in enumerate(need)}, shape=shape, name=name)
            src = self.decoder.block
            kernel = _vx_kernel()
            cols = np.asarray(need, np.int64)

            def part(z_lo, z_hi):
                if kernel is not None:
                    kernel(src, cols, out, z_lo, z_hi)
                    return
                for z in range(z_lo, z_hi):
                    out[:, z, :] = src[z][:, cols].T
            parts = [(zl, min(zl + 16, z1 - z0)) for zl in range(0, z1 - z0, 16)]
            vxb['left'] = len(parts)
            for zl, zh in parts:
                fut = pool.decode_pool.submit(part, zl, zh)
                fut.add_done_callback(lambda f: pool.events.put(('vxpart', None, f)))

        def writer_error():
            # The writer thread alone reads the reply pipe: stopping it collects the writer
            # process's traceback into wbox (its send fails, then it reads the reply).
            write_q.put(None)
            writer_thread.join(5)
            reply = wbox.get('reply')
            code = f'exit code {pool.writer.exitcode}'
            return f'{code}\n{reply[1]}' if reply else code

        while True:
            if v_planned and nw == len(order) and not inflight and pending_tracks == 0:
                break
            try:
                event = pool.events.get(timeout=0.02)
            except queue.Empty:
                event = None
            if event is None:
                flush()
                dispatch()
                dead = [gpus[j] for j, p in enumerate(pool.workers) if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f'GPU worker(s) {", ".join(dead)} died')
                if not pool.writer.is_alive():
                    raise RuntimeError(f'the DBM writer process died: {writer_error()}')
                continue
            kind, i, payload = event
            if kind == 'decoded':
                row = payload.result()
                decode_left -= 1
                self.row_left[row] -= 1
                if self.row_left[row] == 0:
                    rows_done.add(row)
                    while h_waiting:                            # ribbons whose rows are all decoded
                        s = slabs[h_waiting[0]]
                        need = range((z0 + s['a']) // cz, (z0 + s['b'] - 1) // cz + 1)
                        if not all(r in rows_done for r in need):
                            break
                        ready['h'].append(h_waiting.popleft())
                if decode_left == 0:
                    say(f'decoded {len(self.rows)} chunk rows in {time.time() - self.decode_started:.2f} s')
            elif kind == 'gpu':
                msg = payload
                if msg[0] == 'ready':
                    pool.gstate[i].update(ready=True, **msg[1])
                    say(f'GPU {gpus[i]} ready ({msg[1]["free"] / 1e9:.1f} GB free)')
                elif msg[0] == 'fatal':
                    raise RuntimeError(f'GPU worker {gpus[i]} failed to start:\n{msg[1]}')
                elif msg[0] == 'error':
                    raise RuntimeError(f'batch {msg[1]} failed on GPU {gpus[i]}:\n{msg[2]}')
                elif msg[0] == 'skel_start':
                    started[msg[1]] = msg[2]
                elif msg[0] == 'result':
                    bid, stats, empties = msg[1], msg[2], msg[3]
                    gi, take, p, k = inflight.pop(bid)
                    by_gpu[gi].remove(bid)
                    started.pop(bid, None)
                    if not stats['oom_halvings']:
                        obs_skel[p].append((k, stats['skel_s']))
                        obs_prep[p].append((k, stats['prep_s']))
                    pool.gstate[gi]['free'] = stats['free_after']
                    if stats['voxels']:
                        observed_reserve[p] = max(observed_reserve[p] or 0, stats['prep_peak'])
                    say(f'batch {bid} GPU {gpus[gi]}: {k} {p} slabs, {stats["voxels"] / 1e6:.1f} Mvox, '
                        f'prep {stats["prep_s"]:.2f} s, skeletonize {stats["skel_s"]:.2f} s'
                        + (f', {stats["oom_halvings"]} out-of-memory halvings'
                           if stats['oom_halvings'] else ''))
                    for key, empty in empties:
                        if empty:
                            tracks_done[key] = EMPTY
                        else:
                            pending_tracks += 1
            elif kind == 'packed':
                pool_ = pool.track_pools[pool.gpu_node[gpus[i]] if pool.numa else None]
                for key, (v_off, e_off, vertices, edges) in payload:
                    fut = pool_.submit(_track_job, key, slabs[key]['offset'], v_off, e_off, vertices,
                                       edges)
                    fut.add_done_callback(lambda f: pool.events.put(('tracks', None, f)))
            elif kind == 'tracks':
                key, blob = payload.result()
                tracks_done[key] = blob
                pending_tracks -= 1
            elif kind == 'vxpart':
                payload.result()
                vxb['left'] -= 1
                if vxb['left'] == 0:
                    vxb['ready'] = True
            elif kind == 'gpu_closed':
                if pool.workers[i].exitcode not in (None, 0) or inflight:
                    raise RuntimeError(f'GPU worker {gpus[i]} exited')
            if decode_left == 0 and not v_planned:
                plan_vertical()
            flush()
            dispatch()
        write_q.put(None)
        writer_thread.join()
        reply = wbox.get('reply')
        if not reply or reply[0] != 'closed':
            raise RuntimeError(f'the DBM writer failed: {reply[1] if reply else "no reply"}')
        return nw


def _track_job(key, offset, v_off, e_off, vertices, edges):
    """The pickled tracks of one slab (``get_skeleton_tracks``' value for the DBM)."""
    import fast_tracks
    tracks = fast_tracks.tracks_from_packed(v_off, e_off, vertices, edges, offset, copy=False)
    return key, pickle.dumps(tracks)


# ------------------------------------------------------------------------------------------------

def run(predictions_path, tracks_dbm_path, *, z_min, z_max, geometry, skeletonize_kwargs, open_array,
        gpus=None, log=print):
    """Extract the tracks of z_min..z_max into the DBM at ``tracks_dbm_path`` on the GPUs.

    ``geometry``: dict(downsample_factor, z_chunk_depth_h, z_chunk_stride_h, yx_chunk_thickness_v,
    yx_stride_v, dust_threshold, max_area_threshold), in full-resolution voxels.
    ``skeletonize_kwargs``: the options that determine the skeletons (teasar_params, anisotropy,
    dust_threshold, fix_branching, fix_borders, fill_holes), passed to Brook.
    ``open_array(path)``: opens the input when the direct zarr v2 reader does not apply.
    ``gpus``: nvidia-smi indices (default: CUDA_VISIBLE_DEVICES, else every GPU).
    Keys already in the DBM are skipped. Returns dict(keys_written, seconds, gpus).
    """
    t_start = time.time()
    if log is print:
        log = functools.partial(print, flush=True)
    if z_min < 0:
        raise ValueError(f'z_min must be non-negative, not {z_min}')

    def say(msg):
        log(f'[{time.time() - t_start:7.2f} s] {msg}')
    missing = [k for k in GEOMETRY_KEYS if k not in geometry]
    if missing:
        raise ValueError(f'geometry lacks {", ".join(missing)}')
    ds = geometry['downsample_factor']
    if geometry['z_chunk_depth_h'] < ds or geometry['yx_chunk_thickness_v'] < ds:
        raise ValueError('z_chunk_depth_h and yx_chunk_thickness_v must be at least downsample_factor')
    arr, direct = _open_input(predictions_path, open_array)
    Z = arr.shape[0]
    if min(z_max, Z) <= min(z_min, Z):
        # nothing to extract (as in the Kimimaro path, the DBM is still created)
        Path(tracks_dbm_path).parent.mkdir(parents=True, exist_ok=True)
        with dbm.open(os.fspath(tracks_dbm_path), 'c'):
            pass
        say(f'z {z_min}..{z_max} holds no slice of the input (z size {Z}): nothing to extract')
        return dict(keys_written=0, seconds=time.time() - t_start, gpus=[])

    table = _proc_gpus()
    if table is None:
        table = {g: dict(bus=r['bus'], node=_pci_numa_node(r['bus']))
                 for g, r in _nvidia_smi_gpus().items()}
    selected = _select_gpus(gpus, table)
    gpu_node = {g: table[g]['node'] for g in selected}
    numa = all(v is not None for v in gpu_node.values())
    nodes = sorted(set(gpu_node.values())) if numa else [None]
    # The parent (the thread that runs this, and the threads it starts) works on the GPUs'
    # NUMA node(s); the caller's placement is restored at the end.
    saved_affinity, saved_policy = os.sched_getaffinity(0), None
    if numa:
        try:
            if len(nodes) == 1:
                saved_policy = _get_mempolicy()
                _bind_thread(nodes[0])
            else:
                os.sched_setaffinity(0, set().union(*(_node_cpus(n) for n in nodes)) & saved_affinity)
        except OSError as exc:
            say(f'NUMA binding failed ({exc}); continuing without it')
            numa, nodes = False, [None]

    pool = job = None
    try:
        pool = _Pool(selected, gpu_node, numa, nodes)
        # numba's import and the bit-packing kernel's compilation (or cache load) run here, after
        # the workers were spawned, so they overlap the workers' start-up; the decode threads use
        # the kernel, which releases the GIL, from their first chunk
        _packbits_kernel()
        job = _Job(pool, predictions_path, arr, direct, z_min, z_max, tracks_dbm_path, geometry,
                   skeletonize_kwargs, say)
        job.start_decode()
        threading.Thread(target=_warm_numba, daemon=True).start()
        say(f'input {os.fspath(predictions_path)} {tuple(arr.shape)} chunks {tuple(arr.chunks)}, '
            f'{"direct" if direct else "open_array"} reader ({job.decoder.pack_name} bit packing); '
            f'GPUs {",".join(selected)}; NUMA {"nodes " + ",".join(map(str, nodes)) if numa else "off"}; '
            f'{pool.decode_threads} decode threads')
        nw = job.run()
        seconds = time.time() - t_start
        say(f'{nw} keys written to {os.fspath(tracks_dbm_path)} in {seconds:.1f} s '
            f'on {len(selected)} GPU(s)')
        return dict(keys_written=nw, seconds=seconds, gpus=selected)
    finally:
        job = None                      # drop the job's views of the shared blocks before closing them
        if pool is not None:
            pool.close()
        try:
            if os.sched_getaffinity(0) != saved_affinity:
                os.sched_setaffinity(0, saved_affinity)
            if saved_policy is not None:
                _set_mempolicy(*saved_policy)
        except OSError:
            pass
