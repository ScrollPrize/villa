"""Causal native CT observations shared by training, preview and inference."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import os
import numpy as np
from ..geometry import arclength, interp_at, frame_from_heading, CropSpec, crop_local_grid
from ..volume import NativeCT, sample_supported

SLICE_VERSION = 'native_slices_v1'


@dataclass(frozen=True)
class SliceConfig:
    source: str | None = None  # resolved from the follower CT configuration
    level: int = 0
    grid_scale: float = 4.
    origin: tuple = (0., 0., 0.)  # source voxel zero in base-grid xyz
    trace_scale: float = 8.
    cache: str | None = None
    cache_bytes: int = 256 << 20
    pixels: int = 257
    spacing: float | None = None  # one selected-source voxel per pixel by default
    path_step: float = 4.
    history_length: float = 128.
    references: int = 4
    photometric_gain: float = .1
    photometric_bias: float = .02

    def __post_init__(self):
        if not np.isfinite([self.grid_scale, self.trace_scale]).all() or min(self.grid_scale, self.trace_scale) <= 0:
            raise ValueError('Invalid CT coordinate scale')
        native_spacing = self.grid_scale/self.trace_scale
        if self.spacing is None:
            object.__setattr__(self, 'spacing', native_spacing)
        if not np.isfinite(self.spacing) or self.spacing < native_spacing-1e-12:
            raise ValueError(f'Judge pixel spacing must be at least {native_spacing:g} trace voxels; CT upsampling is disabled')
        if self.pixels < 5 or self.pixels % 2 != 1 or min(self.spacing, self.path_step, self.grid_scale, self.trace_scale) <= 0:
            raise ValueError('Invalid native slice geometry')
        if self.history_length < 12 or not 1 <= self.references <= 4 or self.cache_bytes <= 0:
            raise ValueError('Invalid slice memory settings')
        if len(self.origin) != 3 or not np.isfinite(self.origin).all():
            raise ValueError('Only aligned isotropic scale/origin transforms are supported')
        if not 0 <= self.photometric_gain < 1 or not 0 <= self.photometric_bias < 1:
            raise ValueError('Invalid sequence photometric augmentation')

    def to_dict(self):
        return asdict(self)

    def open(self):
        if not self.source:
            raise ValueError('Judge CT source must be resolved from the follower or supplied explicitly')
        return NativeCT(self.source, self.level, self.cache, self.cache_bytes)


def plane_frames(frame):
    u, v, f = frame.T
    return np.stack([np.stack(axes, 1) for axes in ((u, v, f), (u, f, -v), (v, f, u))])


def transported_frames(path, seed_frame, arcs):
    """Prefix invariant reference implementation on fixed transport/sample grids."""
    path, arcs = np.asarray(path), np.asarray(arcs)
    length = arclength(path)
    frame = np.asarray(seed_frame).copy()
    wanted, result = set(arcs.tolist()), {}
    for s in np.unique(np.r_[0., np.arange(1., arcs[-1]+1e-9), arcs]):
        if s > 0:
            a, b = interp_at(path, length, np.array([max(0., s-4), s]))
            if np.linalg.norm(b-a) > 1e-10:
                frame = frame_from_heading(b-a, frame[:, 0])
        if s in wanted:
            result[s] = frame.copy()
    return np.stack([result[s] for s in arcs])


@lru_cache(maxsize=8)
def _plane_geometry(pixels, spacing, grid_scale, trace_scale):
    crop = CropSpec(depth=1, width=pixels, behind=0, spacing=spacing)
    grid = crop_local_grid(crop).reshape(-1, 3)
    yy, xx = np.mgrid[:pixels, :pixels]-pixels//2
    sigma = 2*grid_scale/(trace_scale*spacing)
    marker = np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma)).astype(np.float32)
    grid.flags.writeable = marker.flags.writeable = False
    return grid, marker


_plane_pool = None, None


def _plane_threads():
    """Per-process pool; the three planes' samplers release the GIL."""
    global _plane_pool
    if _plane_pool[1] != os.getpid():  # a forked child cannot use its parent's threads
        _plane_pool = ThreadPoolExecutor(3, thread_name_prefix='ct-planes'), os.getpid()
    return _plane_pool[0]


def sample_planes(reader, center, frame, cfg):
    grid, marker = _plane_geometry(cfg.pixels, cfg.spacing, cfg.grid_scale, cfg.trace_scale)
    def sample(view):
        xyz = ((center+grid @ view.T)*cfg.trace_scale-np.asarray(cfg.origin))/cfg.grid_scale
        ct, support = sample_supported(reader, xyz)
        return np.stack((ct.reshape(marker.shape), marker, support.reshape(marker.shape)))
    return np.asarray(list(_plane_threads().map(sample, plane_frames(frame))), np.float32)


class SliceStream:
    def __init__(self, reader, cfg, seed_frame):
        self.reader, self.cfg, self.seed_frame = reader, cfg, np.asarray(seed_frame).copy()
        self.path = None
        self.records = {}
        self.transport_arc = 0.
        self.frame = self.seed_frame.copy()
        self.next_regular = 0
        self.next_transport = 1
        self.endpoint_record = None

    def update(self, path, accepted=0.):
        path = np.asarray(path, float)
        if path.ndim != 2 or path.shape[1] != 3 or not len(path) or not np.isfinite(path).all():
            raise ValueError('Slice sampling needs a finite contiguous observed path')
        if self.path is not None and (len(path) < len(self.path) or not np.array_equal(path[:len(self.path)], self.path)):
            raise ValueError('Observed sampling state cannot be rewritten')
        self.path = path.copy()
        arc = arclength(path)
        end = float(arc[-1])
        while min(self.next_regular*self.cfg.path_step, self.next_transport) <= end+1e-8:
            s = min(self.next_regular*self.cfg.path_step, self.next_transport)
            self.frame = self._frame(path, arc, s, self.frame)
            self.transport_arc = s
            if abs(s-self.next_regular*self.cfg.path_step) < 1e-8:
                center = interp_at(path, arc, np.array([s]))[0]
                if s >= end-self.cfg.history_length-1e-8 or s < self.cfg.references*self.cfg.path_step:
                    self.records[s] = self._record(s, center, self.frame, True)
                self.next_regular += 1
            if s == self.next_transport:
                self.next_transport += 1
        recent = [s for s in self.records if s >= end-self.cfg.history_length-1e-8]
        refs = [s for s in self.records if s < self.cfg.references*self.cfg.path_step]
        chosen = sorted(set(recent+refs))
        result = [dict(self.records[s], reference=s in refs, query=s in recent) for s in chosen]
        if not result or abs(result[-1]['arc']-end) > 1e-7:
            frame = self._frame(path, arc, end, self.frame)  # temporary; no transport mutation
            if self.endpoint_record is None or self.endpoint_record['arc'] != end:
                self.endpoint_record = self._record(end, path[-1], frame, False)
            result.append(dict(self.endpoint_record, reference=False, query=True))
        else:
            self.endpoint_record = None
        # Keep bounded raw memory. Policy checks whether an overlap exists before accepting.
        self.records = {s: self.records[s] for s in chosen}
        return result

    @staticmethod
    def _frame(path, arc, s, previous):
        if s == 0:
            return previous.copy()
        a, b = interp_at(path, arc, np.array([max(0., s-4), s]))
        return frame_from_heading(b-a, previous[:, 0]) if np.linalg.norm(b-a) > 1e-10 else previous.copy()

    def _record(self, arc, center, frame, regular):
        images = sample_planes(self.reader, center, frame, self.cfg)
        identity = getattr(self.reader, 'identity', getattr(self.reader, 'path', 'array'))
        key = hashlib.sha256((str(identity)+str(self.cfg)).encode()+center.tobytes()+frame.tobytes()).hexdigest()
        return dict(arc=float(arc), center=center.copy(), frame=frame.copy(), images=images,
                    support=bool(images[:, 2].all()), regular=regular, key=key)


def footprint_allowed(records, cfg, band):
    if band is None:
        return True
    half = (cfg.pixels-1)*cfg.spacing/2
    halo = cfg.grid_scale/cfg.trace_scale
    for r in records:
        for f in plane_frames(r['frame']):
            extent = half*(abs(f[2, 0])+abs(f[2, 1]))+halo
            if r['center'][2]+extent >= band.lo and r['center'][2]-extent < band.hi:
                return False
    return True
