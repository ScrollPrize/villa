"""Direct reader of raw ESRF (BM18) Bliss tomography scans.

Reads the Bliss master file (``<name>_0001.h5``) and the Lima detector files
(``scanNNNN/det3_XXXX.h5``) without nxtomomill / nabu / etfscan. Only numpy and
h5py are used. Everything the reference pipeline (``nxtomomill h52nx`` followed by
``nx2etf``) derives from the raw scan is reproduced here, so that the C++ side can
read frames straight from the Lima files:

=====================================  ===============================================
``BlissScan`` field                    reference (ETF ``projections.h5`` etc.)
=====================================  ===============================================
``projections`` (FrameRefs, order p)   ``projections.h5:/data`` (VDS) frame p
``angles_deg``                         ``framewise/angles_deg`` = +mrsrot (Bliss wrap)
``unwrapped_angles_deg``               ``framewise/unwrapped_angles_deg``
``z_translation_m``                    ``framewise/z_translation_m``
``control_A``                          ``framewise/control`` (machine current, A)
``flats_control_A``                    NX ``control/data`` of the flat frames
``flats_unwrapped_angles_deg``         ``flats.h5:/framewise/unwrapped_angles_deg``
``y_translation_m``                    ``framewise/y_translation_m`` (constant)
``estimated_cor_from_motor_pix``       ``globals/estimated_cor_from_motor_pix``
``distance_m``                         ``globals/distance_m`` (effective propagation)
``energy_kev``, ``pixel_size_m``       ``globals/beam_energy_kev``, ``x_pixel_size_m``
=====================================  ===============================================

These agree bit-exactly with the ETF files written by the reference (nxtomomill / nx2etf) conversion.

Calibration reductions are *not* done here (the C++ side reduces frames). The
reference formulas, for the record (float32 arithmetic, ``np.mean`` semantics)::

    dark   = mean_j float32(dark_j)                               (all dark frames)
    flat_k = mean_j [ dark + (float32(F_j) - dark) * (0.2 / float32(I_j)) ]
             for each flat series k, I_j = flats_control_A[k][j]
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import h5py
import numpy as np

__all__ = ["FrameRefs", "BlissScan", "open_scan", "discover_scans", "pair_companions",
           "group_companions", "parse_tetra_name", "FLAT_NOMINAL_CONTROL_A"]

# nx2etf rescales every flat frame to this nominal machine current (A).
FLAT_NOMINAL_CONTROL_A = 0.2

# Title prefixes used by nxtomomill (settings.Tomo.H5) to type the Bliss entries.
_INIT_TITLES = ("tomo:", "zseries", "multitomo", "backandforth", "back_and_forth")
_DARK_TITLES = ("dark images", "dark")
_FLAT_TITLES = ("flat", "reference images", "ref", "refend")
_PROJ_TITLES = ("projections", "ascan rot 0", "ascan diffrz 0 180 1600 0.1")
_ALIGN_TITLES = ("static images", "ascan diffrz 180 0 4 0.1")

# pint-like multiplicative factors (value_in_SI = value * factor, as pint does).
_LENGTH_TO_M = {"m": 1.0, "meter": 1.0, "mm": 1e-3, "millimeter": 1e-3, "um": 1e-6, "µm": 1e-6,
                "μm": 1e-6, "micrometer": 1e-6, "micron": 1e-6, "nm": 1e-9, "cm": 1e-2}
_CURRENT_TO_A = {"a": 1.0, "ampere": 1.0, "ma": 1e-3, "milliampere": 1e-3}


# ----------------------------------------------------------------------------------------
# data classes
# ----------------------------------------------------------------------------------------
@dataclass
class FrameRefs:
    """Location of a sequence of frames in the raw Lima files (not the master VDS).

    Frame ``i`` of the sequence is ``files[file_index[i]][dataset][frame_index[i]]``.
    """
    files: list[str]
    dataset: str
    file_index: np.ndarray
    frame_index: np.ndarray

    def __post_init__(self):
        self.file_index = np.ascontiguousarray(self.file_index, dtype=np.int32)
        self.frame_index = np.ascontiguousarray(self.frame_index, dtype=np.int32)
        if self.file_index.shape != self.frame_index.shape or self.file_index.ndim != 1:
            raise ValueError("file_index and frame_index must be 1-D arrays of equal length")

    def __len__(self) -> int:
        return int(self.file_index.size)

    def locate(self, i: int) -> tuple[str, int]:
        """(absolute file path, frame index inside it) of frame ``i``."""
        return self.files[int(self.file_index[i])], int(self.frame_index[i])

    def subset(self, idx) -> "FrameRefs":
        """Frames ``idx`` (slice, index array or boolean mask) as a new FrameRefs (same file list)."""
        return FrameRefs(list(self.files), self.dataset, self.file_index[idx], self.frame_index[idx])

    @staticmethod
    def concatenate(parts: list["FrameRefs"]) -> "FrameRefs":
        """Concatenate several FrameRefs (merging their file lists)."""
        if not parts:
            return FrameRefs([], "", np.zeros(0, np.int32), np.zeros(0, np.int32))
        datasets = {p.dataset for p in parts if len(p)}
        if len(datasets) > 1:
            raise ValueError(f"cannot concatenate FrameRefs with different datasets {datasets}")
        files: list[str] = []
        pos: dict[str, int] = {}
        fidx, fr = [], []
        for p in parts:
            remap = np.empty(len(p.files), np.int32)
            for k, fn in enumerate(p.files):
                if fn not in pos:
                    pos[fn] = len(files)
                    files.append(fn)
                remap[k] = pos[fn]
            fidx.append(remap[p.file_index] if len(p) else p.file_index)
            fr.append(p.frame_index)
        return FrameRefs(files, datasets.pop() if datasets else parts[0].dataset,
                         np.concatenate(fidx), np.concatenate(fr))

    def runs(self):
        """Iterate over maximal runs of consecutive frames in one file:
        yields (start_in_sequence, file_index, first_frame_in_file, count)."""
        n = len(self)
        if n == 0:
            return
        brk = np.flatnonzero((np.diff(self.file_index) != 0) | (np.diff(self.frame_index) != 1)) + 1
        starts = np.concatenate([[0], brk])
        ends = np.concatenate([brk, [n]])
        for s, e in zip(starts, ends):
            yield int(s), int(self.file_index[s]), int(self.frame_index[s]), int(e - s)


@dataclass
class BlissScan:
    """One Bliss tomography scan (one ``<name>_0001.h5`` master)."""
    name: str                               # scan name, e.g. "..._HA_0001"
    master: str                             # absolute path of the master .h5
    directory: str                          # directory holding the master and scanNNNN/
    frame_shape: tuple[int, int]            # (rows, cols)
    dtype: np.dtype
    projections: FrameRefs                  # ETF order p = 0..n_proj-1
    darks: FrameRefs
    flats: list[FrameRefs]                  # one per contiguous flat series, acquisition order
    angles_deg: np.ndarray                  # +mrsrot (deg), wrapped as stored by Bliss
    unwrapped_angles_deg: np.ndarray        # np.unwrap(angles_deg, period=360)
    z_translation_m: np.ndarray             # sz / mrsz (m)
    control_A: np.ndarray                   # per-projection machine current (A)
    flats_control_A: list[np.ndarray]       # per flat-frame machine current (A), one per series
    flats_unwrapped_angles_deg: np.ndarray  # (n_series,) angle attached to each flat series
    darks_control_A: np.ndarray             # per dark-frame machine current (A) (unused by ETF)
    y_translation_m: float                  # lateral motor (yrot), m
    estimated_cor_from_motor_pix: float     # y_translation_m / pixel_size_m (offset from (W-1)/2)
    energy_kev: float
    distance_m: float                       # effective propagation distance z1*z2/(z1+z2)
    pixel_size_m: float                     # sample pixel size (x == y)
    detector_pixel_size_m: float | None = None
    count_time_s: float | None = None
    # continuous rotation: fraction of the angular step between projections swept during one exposure (the recorded
    # <rotation>_delta / <rotation>_period, else exposure / (exposure + latency)); None: unknown or a step scan
    exposure_arc: float | None = None
    detector: str = "det3"
    sample_name: str = ""
    start_time: str = ""
    end_time: str = ""
    entries: dict = field(default_factory=dict)   # Bliss entry -> role / n frames (diagnostics)

    # -- basic properties ------------------------------------------------------------
    @property
    def n_proj(self) -> int:
        return len(self.projections)

    @property
    def n_flat_series(self) -> int:
        return len(self.flats)

    @property
    def cor_absolute_pix(self) -> float:
        """Rotation axis column from the motor: (W - 1) / 2 + offset."""
        return (self.frame_shape[1] - 1) / 2.0 + self.estimated_cor_from_motor_pix

    @property
    def tetra(self) -> dict | None:
        """Parsed HA/TA name info (see :func:`parse_tetra_name`) or None."""
        return parse_tetra_name(self.name)

    # -- I/O ---------------------------------------------------------------------------
    def all_refs(self) -> list[FrameRefs]:
        return [self.projections, self.darks, *self.flats]

    def referenced_files(self) -> list[str]:
        out, seen = [], set()
        for r in self.all_refs():
            used = np.unique(r.file_index) if len(r) else []
            for k in used:
                fn = r.files[int(k)]
                if fn not in seen:
                    seen.add(fn)
                    out.append(fn)
        return out

    def check_files(self, deep: bool = False) -> list[str]:
        """Check that every referenced Lima file exists (a missing VDS source silently reads 0).

        With ``deep=True`` each file is also opened and the dataset shape/dtype and the
        frame range used are checked. Returns the list of problems (empty if all good);
        raises FileNotFoundError listing the missing files if any is missing.
        """
        missing = [fn for fn in self.referenced_files() if not os.path.isfile(fn)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} Lima files missing for {self.name}: "
                                    f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        problems = []
        if deep:
            need: dict[str, int] = {}
            for r in self.all_refs():
                for s, fi, f0, cnt in r.runs():
                    fn = r.files[fi]
                    need[fn] = max(need.get(fn, 0), f0 + cnt)
            for fn, nmax in need.items():
                with h5py.File(fn, "r") as f:
                    ds = f.get(self.projections.dataset)
                    if ds is None:
                        problems.append(f"{fn}: no dataset {self.projections.dataset}")
                    elif ds.shape[1:] != tuple(self.frame_shape) or ds.dtype != self.dtype:
                        problems.append(f"{fn}: shape {ds.shape} dtype {ds.dtype}")
                    elif ds.shape[0] < nmax:
                        problems.append(f"{fn}: holds {ds.shape[0]} frames, {nmax} needed")
        return problems

    def read_frames(self, refs: FrameRefs, idx=None, rows=None, cols=None) -> np.ndarray:
        """Slow reference reader (h5py, one file at a time) for tests / debugging.

        ``idx``: int, slice, or index array into ``refs`` (default: all). ``rows``/``cols``:
        optional slices (or index arrays) restricting the pixels read. Returns
        ``(n, rows, cols)`` in the stored dtype (``(rows, cols)`` if ``idx`` is an int).
        """
        scalar = np.isscalar(idx) or (isinstance(idx, np.ndarray) and idx.ndim == 0)
        if idx is None:
            sel = np.arange(len(refs))
        elif isinstance(idx, slice):
            sel = np.arange(len(refs))[idx]
        else:
            sel = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        rsel = slice(None) if rows is None else rows
        csel = slice(None) if cols is None else cols
        nr = len(range(*rsel.indices(self.frame_shape[0]))) if isinstance(rsel, slice) else len(rsel)
        nc = len(range(*csel.indices(self.frame_shape[1]))) if isinstance(csel, slice) else len(csel)
        out = np.empty((len(sel), nr, nc), dtype=self.dtype)
        fi = refs.file_index[sel]
        fr = refs.frame_index[sel]
        for k in np.unique(fi):
            where = np.flatnonzero(fi == k)
            with h5py.File(refs.files[int(k)], "r") as f:
                ds = f[refs.dataset]
                for w in where:
                    a = ds[int(fr[w]), rsel, :] if not isinstance(rsel, slice) else ds[int(fr[w]), rsel]
                    out[w] = a[:, csel]
        return out[0] if scalar else out

    def summary(self) -> str:
        return (f"{self.name}: {self.n_proj} proj, {len(self.darks)} darks, "
                f"flats {[len(f) for f in self.flats]}, frame {self.frame_shape}, "
                f"yrot {self.y_translation_m * 1e3:.4f} mm, cor offset {self.estimated_cor_from_motor_pix:.4f} px, "
                f"angles {self.unwrapped_angles_deg[0]:.5f}..{self.unwrapped_angles_deg[-1]:.5f} deg, "
                f"z {self.z_translation_m[0] * 1e3:.4f}..{self.z_translation_m[-1] * 1e3:.4f} mm")


# ----------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------
def _text(v) -> str:
    if isinstance(v, h5py.Dataset):
        v = v[()]
    if isinstance(v, np.ndarray):
        v = v.item() if v.size == 1 else v.tolist()
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def _units(ds) -> str:
    u = ds.attrs.get("units", None)
    return "" if u is None else _text(u).strip()


def _to_si(value, unit: str, table: dict, default: str):
    u = (unit or default)
    factor = table.get(u, table.get(u.lower()))
    if factor is None:
        raise ValueError(f"unknown unit {unit!r}")
    return value * factor if factor != 1.0 else value


def _iso_to_ms(s: str) -> np.datetime64:
    """nxtomomill ``str_datetime_to_numpy_datetime64``: ISO string -> UTC datetime64[ms]."""
    d = datetime.fromisoformat(s)
    if d.tzinfo is not None:
        d = d.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(d).astype("<M8[ms]")


def _linspace_ms(t0: np.datetime64, t1: np.datetime64, n: int) -> np.ndarray:
    """nxtomomill time stamps: ``linspace(float128(t0), float128(t1), n, dtype='<M8[ms]')``."""
    return np.linspace(t0.astype(np.longdouble), t1.astype(np.longdouble), num=n, endpoint=True,
                       dtype="<M8[ms]")


def _entry_key(name: str):
    try:
        a, b = name.split(".")
        return (int(a), int(b))
    except ValueError:
        return (1 << 30, 0)


def _title_role(title: str) -> str | None:
    for role, titles in (("init", _INIT_TITLES), ("dark", _DARK_TITLES), ("flat", _FLAT_TITLES),
                         ("proj", _PROJ_TITLES), ("align", _ALIGN_TITLES)):
        if any(title.startswith(t) for t in titles):
            return role
    return None


def _entry_role(entry: h5py.Group) -> str | None:
    """nxtomomill ScanTypeFinder: technique/image_key, then scan_category, then title."""
    tech = entry.get("technique")
    if tech is not None and "image_key" in tech:
        try:
            k = int(np.asarray(tech["image_key"][()]).ravel()[0])
            return {0: "proj", 1: "flat", 2: "dark", -1: "align"}.get(k)
        except Exception:
            pass
    if tech is not None and "scan_category" in tech:
        return "init"
    if "title" not in entry:
        return None
    return _title_role(_text(entry["title"]))


def _find_detector(f: h5py.File, entries: list[str]) -> str:
    """Name of the tomo detector (technique/tomoconfig/detector, else first NXdetector with 3-D image)."""
    for e in entries:
        for p in ("technique/tomoconfig/detector", "technique/detector"):
            if p in f[e]:
                obj = f[e][p]
                if isinstance(obj, h5py.Dataset):
                    v = obj[()]
                    v = v[0] if isinstance(v, np.ndarray) and v.size else v
                    return _text(v)
                if isinstance(obj, h5py.Group) and len(obj.keys()) == 1:
                    return list(obj.keys())[0]
    for e in entries:
        inst = f[e].get("instrument")
        if inst is None:
            continue
        for k in inst:
            g = inst.get(k)
            if isinstance(g, h5py.Group) and _text(g.attrs.get("NX_class", "")) == "NXdetector":
                d = g.get("image", g.get("data"))
                if isinstance(d, h5py.Dataset) and d.ndim == 3:
                    return k
    raise RuntimeError("no tomography detector found")


def _frame_dataset(entry: h5py.Group, det: str) -> h5py.Dataset | None:
    inst = entry.get("instrument")
    if inst is None or det not in inst:
        return None
    for k in ("image", "data"):
        d = inst[det].get(k)
        if isinstance(d, h5py.Dataset) and d.ndim == 3 and d.shape[0] > 0:
            return d
    return None


def _vds_refs(ds: h5py.Dataset, master_dir: str, entry: h5py.Group, det: str) -> FrameRefs:
    """Map every frame of a Bliss image dataset to (Lima file, frame) through its VDS layout."""
    n = ds.shape[0]
    fpix = int(ds.shape[1]) * int(ds.shape[2])
    if not ds.is_virtual:
        return _naming_refs(ds, master_dir, entry, det)
    files: list[str] = []
    pos: dict[str, int] = {}
    fidx = np.full(n, -1, np.int32)
    fr = np.full(n, -1, np.int32)
    dsets = set()
    for src in ds.virtual_sources():
        fn = src.file_name
        if fn == ".":
            fn = ds.file.filename
        elif not os.path.isabs(fn):
            fn = os.path.join(master_dir, fn)
        fn = os.path.normpath(fn)
        dsets.add(src.dset_name)
        v0, vcnt = _hyperslab_frames(src.vspace, fpix)
        s0, scnt = _hyperslab_frames(src.src_space, fpix)
        if vcnt != scnt:
            raise RuntimeError(f"irregular VDS mapping in {ds.name}: {vcnt} vs {scnt} frames")
        if fn not in pos:
            pos[fn] = len(files)
            files.append(fn)
        fidx[v0:v0 + vcnt] = pos[fn]
        fr[v0:v0 + vcnt] = np.arange(s0, s0 + scnt, dtype=np.int32)
    if (fidx < 0).any():
        raise RuntimeError(f"VDS {ds.file.filename}:{ds.name} does not cover {int((fidx < 0).sum())} frames")
    if len(dsets) != 1:
        raise RuntimeError(f"VDS {ds.name} uses several source datasets {dsets}")
    dset = dsets.pop()
    if not dset.startswith("/"):
        dset = "/" + dset
    return FrameRefs(files, dset, fidx, fr)


def _hyperslab_frames(space, fpix: int) -> tuple[int, int]:
    """(first frame, number of frames) of a whole-frame selection of a 3-D dataspace."""
    shape = space.shape
    if space.get_select_type() == h5py.h5s.SEL_ALL:
        return 0, int(shape[0])
    start, stride, count, block = space.get_regular_hyperslab()
    if tuple(start[1:]) != (0, 0):
        raise RuntimeError("VDS selection is not made of whole frames")
    nfr = int(count[0]) * int(block[0])
    if int(count[0]) > 1 and int(stride[0]) != int(block[0]):
        raise RuntimeError("strided VDS frame selection is not supported")
    if space.get_select_npoints() != nfr * fpix:
        raise RuntimeError("VDS selection is not made of whole frames")
    return int(start[0]), nfr


def _naming_refs(ds, master_dir: str, entry: h5py.Group, det: str) -> FrameRefs:
    """Fallback (no VDS): Lima naming convention scanNNNN/<prefix>XXXX.h5, frames_per_file each."""
    acq = entry[f"instrument/{det}/acq_parameters"]
    ctrl = entry[f"instrument/{det}/ctrl_parameters"]
    prefix = _text(acq["saving_prefix"])
    fpf = int(ctrl["saving_frame_per_file"][()])
    sub = os.path.basename(_text(acq["saving_directory"]).rstrip("/"))
    n = ds.shape[0]
    k = np.arange(n)
    nfiles = (n + fpf - 1) // fpf
    files = [os.path.join(master_dir, sub, f"{prefix}{i:04d}.h5") for i in range(nfiles)]
    return FrameRefs(files, f"/entry_0000/instrument/{det}/data", k // fpf, k % fpf)


def _per_frame(entry: h5py.Group, names: list[str], n: int, kind: str):
    """Motor value per frame: ``measurement/<name>`` (len n) or scalar ``instrument/positioners/<name>``.

    ``names`` are tried in order for measurements first, then for positioners (the Bliss
    tomo config lists the fast/master motor first, e.g. [mrsz, sz])."""
    meas = entry.get("measurement")
    for nm in names:
        if meas is not None and nm in meas:
            d = meas[nm]
            if isinstance(d, h5py.Dataset) and d.ndim == 1 and d.shape[0] == n:
                return np.asarray(d[()], dtype=np.float64), _units(d), f"measurement/{nm}"
    pos = entry.get("instrument/positioners")
    for nm in names:
        if pos is not None and nm in pos:
            d = pos[nm]
            v = d[()]
            if isinstance(v, (bytes, str)) or (isinstance(v, np.ndarray) and v.dtype.kind in "OSU"):
                continue
            v = float(np.asarray(v).ravel()[0])
            return np.full(n, v), _units(d), f"positioners/{nm}"
    raise KeyError(f"no {kind} motor among {names} in {entry.name}")


def _config_names(f: h5py.File, init: str | None, key: str, default: list[str]) -> list[str]:
    if init is not None:
        for grp in ("technique/tomoconfig", "instrument/mrtomo_config"):
            p = f"{init}/{grp}/{key}"
            if p in f:
                v = f[p][()]
                v = v.tolist() if isinstance(v, np.ndarray) else [v]
                return [_text(x) for x in v]
    return list(default)


def _scalar_with_units(group, path: str):
    if group is None or path not in group:
        return None, ""
    d = group[path]
    v = d[()]
    v = float(np.asarray(v).ravel()[0])
    return v, _units(d)


# ----------------------------------------------------------------------------------------
# machine current (nxtomomill deduce_machine_current)
# ----------------------------------------------------------------------------------------
def _entry_currents(entry: h5py.Group):
    """nxtomomill ``_get_machine_current``: key 'current' in the entry, its measurement group,
    then instrument/machine. Returns (values in A, None) or None."""
    for grp_path in ("", "measurement", "instrument/machine"):
        grp = entry if grp_path == "" else entry.get(grp_path)
        if grp is None or "current" not in grp:
            continue
        d = grp["current"]
        if not isinstance(d, h5py.Dataset):
            continue
        v = d[()]
        if isinstance(v, (bytes, str)):
            continue
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        return _to_si(v, _units(d), _CURRENT_TO_A, "mA")
    return None


def _machine_current(sequence, f: h5py.File) -> np.ndarray:
    """Per-frame current for all frames of the NX sequence (``sequence`` = list of
    (entry name, role, n_frames or 0), in nxtomomill processing order)."""
    known: dict = {}
    frame_t = []
    for e, role, nfr in sequence:
        g = f[e]
        st = _iso_to_ms(_text(g["start_time"])) if "start_time" in g else None
        en = _iso_to_ms(_text(g["end_time"])) if "end_time" in g else None
        if nfr:
            if st is None or en is None:
                t = st if st is not None else en
                if t is None:
                    raise RuntimeError(f"no start/end time in {e}")
                frame_t.append(np.full(nfr, t, dtype="<M8[ms]"))
            else:
                frame_t.append(_linspace_ms(st, en, nfr))
        cur = _entry_currents(g)
        if cur is None or len(cur) == 0:
            continue
        new = {}
        if st is None or en is None:
            t = st if st is not None else en
            if t is not None:
                new[t] = cur[0]
        elif len(cur) == 1:
            new[st] = cur[0]
        else:
            for t, c in zip(_linspace_ms(st, en, len(cur)), cur):
                new[t] = c
        known.update({k: v for k, v in new.items() if not np.isnan(v)})
    if not known:
        raise RuntimeError("no machine current found")
    items = sorted(known.items())
    kt = np.array([k for k, _ in items], dtype="datetime64[ns]").astype(np.float64)
    kc = np.array([v for _, v in items], dtype=np.float64)
    ft = np.concatenate(frame_t).astype("datetime64[ns]")
    order = np.argsort(ft)
    vals = np.interp(ft[order].astype(np.float64), kt, kc)
    out = np.empty_like(vals)
    out[order] = vals
    return out


# ----------------------------------------------------------------------------------------
# open / discover
# ----------------------------------------------------------------------------------------
def _resolve_master(path: str) -> str:
    path = os.path.abspath(os.fspath(path))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    base = os.path.basename(path.rstrip("/"))
    cand = os.path.join(path, base + ".h5")
    if os.path.isfile(cand):
        return cand                                # scan directory <name>_0001/<name>_0001.h5
    subs = sorted(d for d in os.listdir(path)
                  if os.path.isfile(os.path.join(path, d, d + ".h5")))
    if len(subs) == 1:
        return os.path.join(path, subs[0], subs[0] + ".h5")   # parent directory
    if len(subs) > 1:
        raise ValueError(f"{path} holds several scans {subs}; open one of them")
    raise FileNotFoundError(f"no Bliss master <dir>/<dir>.h5 found in {path}")


def open_scan(path) -> BlissScan:
    """Open a Bliss tomography scan.

    ``path`` is the scan directory ``<name>_0001`` (holding ``<name>_0001.h5`` and
    ``scanNNNN/``), its parent directory (if it holds a single scan) or the master file.
    """
    master = _resolve_master(path)
    mdir = os.path.dirname(master)
    name = os.path.splitext(os.path.basename(master))[0]
    with h5py.File(master, "r") as f:
        # entries in nxtomomill order: stable sort of h5py keys by start_time
        keys = [k for k in f.keys() if isinstance(f.get(k), h5py.Group)]

        def _t(k):
            g = f[k]
            if "start_time" in g:
                return _iso_to_ms(_text(g["start_time"])).astype(np.int64)
            return np.int64(int(_entry_key(k)[0]) if _entry_key(k)[0] < (1 << 30) else 0)
        keys.sort(key=_t)
        roles = {k: _entry_role(f[k]) for k in keys}
        inits = [k for k in keys if roles[k] == "init"]
        if not inits:
            raise ValueError(f"{master}: no tomo initialisation entry (not a Bliss tomo scan?)")
        if len(inits) > 1:
            raise NotImplementedError(f"{master}: several tomo sequences {inits} (z-series?) not supported")
        init = inits[0]
        steps = [k for k in keys[keys.index(init) + 1:] if roles[k] in ("dark", "flat", "proj", "align")]
        det = _find_detector(f, [init] + steps)
        rot_names = _config_names(f, init, "rotation", ["mrsrot", "srot"])
        z_names = _config_names(f, init, "translation_z", ["mrsz", "sz"])
        y_names = _config_names(f, init, "translation_y", ["mryrot", "yrot"])

        sequence = []   # (entry, role, n_frames) in NX order, incl. frame-less current entries
        refs = {"dark": [], "flat": [], "proj": [], "align": []}
        kinds = []      # per NX frame: role
        flat_chunks = []  # FrameRefs per flat entry, with NX start index
        angles, zs, zunits, aunits = [], [], set(), set()
        arc_delta, arc_period = [], []
        entries_info = {}
        frame_shape, dtype = None, None
        nx_index = 0
        proj_nx_index = []
        for e in steps:
            g = f[e]
            ds = _frame_dataset(g, det)
            role = roles[e]
            nfr = 0 if ds is None else int(ds.shape[0])
            sequence.append((e, role, nfr))
            entries_info[e] = dict(role=role, title=_text(g["title"]) if "title" in g else "", n_frames=nfr)
            if not nfr:
                continue
            if frame_shape is None:
                frame_shape, dtype = (int(ds.shape[1]), int(ds.shape[2])), ds.dtype
            elif (int(ds.shape[1]), int(ds.shape[2])) != frame_shape or ds.dtype != dtype:
                raise RuntimeError(f"{e}: frame shape/dtype {ds.shape[1:]} {ds.dtype} differs")
            r = _vds_refs(ds, mdir, g, det)
            refs[role].append(r)
            kinds.extend([role] * nfr)
            if role == "flat":
                flat_chunks.append((nx_index, r))
            if role == "proj":
                a, au, _ = _per_frame(g, rot_names, nfr, "rotation")
                z, zu, _ = _per_frame(g, z_names, nfr, "translation_z")
                angles.append(a)
                zs.append(_to_si(z, zu, _LENGTH_TO_M, "mm"))
                aunits.add(au or "deg")
                proj_nx_index.append(np.arange(nx_index, nx_index + nfr))
                meas = g.get("measurement")
                for nm in rot_names:   # the angle swept during / between exposures, as the encoder recorded it
                    if meas is not None and f"{nm}_delta" in meas and f"{nm}_period" in meas:
                        arc_delta.append(np.abs(np.asarray(meas[f"{nm}_delta"][()], np.float64)))
                        arc_period.append(np.abs(np.asarray(meas[f"{nm}_period"][()], np.float64)))
                        break
            nx_index += nfr
        if not refs["proj"]:
            raise ValueError(f"{master}: no projection frames")
        if aunits - {"deg", "degree", "degrees"}:
            raise ValueError(f"unexpected rotation units {aunits}")

        # ---- per-frame machine current over the whole NX sequence
        try:
            current = _machine_current(sequence, f)
        except RuntimeError:
            current = np.full(nx_index, np.nan)
        kinds = np.array(kinds)
        proj_nx_index = np.concatenate(proj_nx_index)

        # ---- angles (ETF: angle_sign(-1) * NX(-mrsrot) = +mrsrot) and unwrap
        angles_deg = np.concatenate(angles)
        unwrapped = np.unwrap(angles_deg, period=360)

        # ---- flats: contiguous runs of flat frames in the NX sequence
        flat_mask = kinds == "flat"
        flats, flats_ctl, flats_ang = [], [], []
        if flat_mask.any():
            fidx = np.flatnonzero(flat_mask)
            runs = np.split(fidx, np.flatnonzero(np.diff(fidx) != 1) + 1)
            all_flat = FrameRefs.concatenate([r for _, r in flat_chunks])
            # position of each NX flat frame inside all_flat
            flat_pos = np.cumsum(flat_mask) - 1
            for run in runs:
                flats.append(all_flat.subset(flat_pos[run]))
                flats_ctl.append(current[run])
                mid = 0.5 * (run[0] + run[-1])
                flats_ang.append(np.interp(mid, proj_nx_index, unwrapped))
        darks = FrameRefs.concatenate(refs["dark"])
        darks_ctl = current[kinds == "dark"]

        # ---- scalars
        tech = f[init].get("technique")
        scan = tech.get("scan") if tech is not None else None
        optic = tech.get("optic") if tech is not None else None
        px, pxu = _scalar_with_units(optic, "sample_pixel_size")
        if px is None:
            px, pxu = _scalar_with_units(scan, "sample_pixel_size")
        pixel_size_m = float(_to_si(px, pxu, _LENGTH_TO_M, "um"))
        dpx, dpxu = _scalar_with_units(optic, "optics_pixel_size")
        detector_pixel_size_m = None if dpx is None else float(_to_si(dpx, dpxu, _LENGTH_TO_M, "um"))
        en, _ = _scalar_with_units(scan, "energy")
        energy_kev = float("nan") if en is None else float(en)
        prop, propu = _scalar_with_units(scan, "effective_propagation_distance")
        if prop is not None:
            distance_m = float(_to_si(prop, propu, _LENGTH_TO_M, "mm"))
        else:   # nxtomomill: (-z1*z2)/(-z1+z2), z1 = -source_sample (NX source distance)
            z2, z2u = _scalar_with_units(scan, "sample_detector_distance")
            z1, z1u = _scalar_with_units(scan, "source_sample_distance")
            if z2 is None:
                distance_m = float("nan")
            elif z1 is None:
                distance_m = float(_to_si(z2, z2u, _LENGTH_TO_M, "mm"))
            else:
                z1 = -float(_to_si(z1, z1u, _LENGTH_TO_M, "mm"))
                z2 = float(_to_si(z2, z2u, _LENGTH_TO_M, "mm"))
                distance_m = (-z1 * z2) / (-z1 + z2)
        et, _ = _scalar_with_units(scan, "exposure_time")
        count_time_s = None if et is None else float(et) * 1e-3
        exposure_arc = None
        continuous = scan is not None and "scan_type" in scan and _text(scan["scan_type"]).upper() == "CONTINUOUS"
        if continuous and arc_delta:
            period = float(np.median(np.concatenate(arc_period)))
            if period > 0:
                exposure_arc = float(np.median(np.concatenate(arc_delta))) / period
        if continuous and exposure_arc is None and et:
            lt, _ = _scalar_with_units(scan, "latency_time")
            exposure_arc = float(et) / (float(et) + float(lt or 0.0))
        if exposure_arc is not None and not 0.0 < exposure_arc <= 1.0:
            exposure_arc = None
        # lateral motor: nxtomomill translation_y[0] (first frame of the sequence), mm -> m
        g0 = f[steps[0]]
        yv, yu, _ = _per_frame(g0, y_names, 1, "translation_y")
        y_translation_m = float(_to_si(yv[0], yu, _LENGTH_TO_M, "mm"))
        estimated_cor = y_translation_m / pixel_size_m
        sample_name = _text(f[init]["sample/name"]) if "sample/name" in f[init] else ""
        start_time = _text(f[init]["start_time"]) if "start_time" in f[init] else ""
        end_time = _text(f[init]["end_time"]) if "end_time" in f[init] else ""

    return BlissScan(
        name=name, master=master, directory=mdir, frame_shape=frame_shape, dtype=np.dtype(dtype),
        projections=FrameRefs.concatenate(refs["proj"]), darks=darks, flats=flats,
        angles_deg=angles_deg, unwrapped_angles_deg=unwrapped,
        z_translation_m=np.concatenate(zs),
        control_A=current[kinds == "proj"], flats_control_A=flats_ctl,
        flats_unwrapped_angles_deg=np.asarray(flats_ang, dtype=np.float64),
        darks_control_A=darks_ctl,
        y_translation_m=y_translation_m, estimated_cor_from_motor_pix=float(estimated_cor),
        energy_kev=energy_kev, distance_m=float(distance_m), pixel_size_m=pixel_size_m,
        detector_pixel_size_m=detector_pixel_size_m, count_time_s=count_time_s, exposure_arc=exposure_arc,
        detector=det,
        sample_name=sample_name, start_time=start_time, end_time=end_time, entries=entries_info)


def _is_tomo_master(fn: str) -> bool:
    try:
        with h5py.File(fn, "r") as f:
            for k in f.keys():
                g = f.get(k)
                if isinstance(g, h5py.Group) and _entry_role(g) == "init":
                    return True
    except (OSError, KeyError):
        return False
    return False


def discover_scans(root, max_depth: int = 3, require_files: bool = False) -> list[BlissScan]:
    """All Bliss tomo scans under ``root``, sorted by name.

    Bliss layout: ``<root>/<sample>/<sample>_NNNN/<sample>_NNNN.h5`` (+ ``scanNNNN/``);
    the proposal-level ``<root>/<sample>/<proposal>_<sample>.h5`` files are not scans and
    are ignored. A master is recognised as a directory ``D`` holding ``D.h5`` whose
    entries contain a tomo initialisation entry (e.g. title ``tomo:helical``). The walk
    does not descend into a scan directory (its ``scanNNNN`` sub-folders only hold Lima
    files). Unreadable or non-tomo files are skipped.
    """
    root = os.path.abspath(os.fspath(root))
    found = []

    def walk(d, depth):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return
        base = os.path.basename(d)
        master = os.path.join(d, base + ".h5")
        if base + ".h5" in names and os.path.isfile(master):
            if _is_tomo_master(master):
                found.append(master)
            return
        if depth >= max_depth:
            return
        for n in names:
            p = os.path.join(d, n)
            if os.path.isdir(p) and not n.startswith("."):
                walk(p, depth + 1)

    walk(root, 0)
    scans = []
    for m in sorted(found, key=lambda p: (os.path.basename(p), p)):
        try:
            s = open_scan(m)
        except (ValueError, NotImplementedError, KeyError, RuntimeError, OSError):
            continue
        if require_files:
            try:
                s.check_files()
            except FileNotFoundError:
                continue
        scans.append(s)
    return scans


# ----------------------------------------------------------------------------------------
# HA / TA companion pairing
# ----------------------------------------------------------------------------------------
_TETRA_KINDS = ["HA", "TA", *[f"TA{i}" for i in range(2, 10)]]
_TETRA_RE = re.compile(r"^(.+?)(?:_(HA|TA[2-9]?))?((?:_(?:fc|cs)_)*?)_(\d{4})(?:_(\d{4}))?$")


def parse_tetra_name(name: str) -> dict | None:
    """night_rail_bm18 ``analyse_tetra_and_crashes`` name rule.

    ``<base>_<KIND>[_fc_|_cs_]_NNNN[_NNNN]`` with KIND in HA < TA < TA2 < ... < TA9.
    Returns {"base", "kind", "rank", "suffix", "crash"} or None if no kind tag.
    """
    m = _TETRA_RE.match(name)
    if not m or m.group(2) is None:
        return None
    suffix = f"_{m.group(4)}" + (f"_{m.group(5)}" if m.group(5) else "")
    return dict(base=m.group(1), kind=m.group(2), rank=_TETRA_KINDS.index(m.group(2)),
                suffix=suffix, crash=bool(m.group(3)))


def _compatible(a: BlissScan, b: BlissScan) -> bool:
    return (a.frame_shape == b.frame_shape and a.n_proj == b.n_proj
            and np.isclose(a.pixel_size_m, b.pixel_size_m, rtol=1e-6)
            and np.isclose(a.energy_kev, b.energy_kev, rtol=1e-6)
            and abs(a.unwrapped_angles_deg[-1] - a.unwrapped_angles_deg[0]
                    - (b.unwrapped_angles_deg[-1] - b.unwrapped_angles_deg[0])) < 1.0)


def group_companions(scans: list[BlissScan]) -> list[list[BlissScan]]:
    """Extended-FOV families, each ordered main first (HA, TA, TA2, ...).

    Rule (night_rail_bm18, ``analyse_tetra_and_crashes`` / ``resolve_companion_scans``):
    scans whose names are ``<base>_<KIND>_<NNNN>`` with the same ``<base>`` and
    ``<NNNN>`` (crash-postfixed ``_fc_``/``_cs_`` names are skipped) form one family;
    members are ordered HA < TA < TA2 < ... < TA9. night_rail uses the highest kind
    (TA) as its Bliss2Nx "master" and HA as companion_01, but the z-concatenated helical
    ETF read by nabuxx has **HA as main** (first ``--filename_template``) and TA as
    ``companion_01``; that is the order returned here.

    Scans without a kind tag are grouped by a geometric rule instead: same frame shape,
    pixel size, energy, n_proj and angular range, different lateral motor (yrot); main =
    smallest ``|estimated_cor_from_motor_pix|`` (axis closest to the detector centre,
    as HA is: +421 px vs TA +1264 px), others by increasing |offset|.
    """
    fams: dict = {}
    untagged = []
    for s in scans:
        t = parse_tetra_name(s.name)
        if t is None:
            untagged.append(s)
        elif not t["crash"]:
            fams.setdefault((t["base"], t["suffix"]), []).append((t["rank"], s))
    out = []
    for key in sorted(fams):
        mem = sorted(fams[key], key=lambda x: x[0])
        if len({r for r, _ in mem}) != len(mem):
            raise ValueError(f"duplicate HA/TA kinds for {key}")
        if len(mem) >= 2:
            out.append([s for _, s in mem])
    # geometric grouping of untagged scans
    used = set()
    for i, a in enumerate(untagged):
        if i in used:
            continue
        grp = [a]
        for j in range(i + 1, len(untagged)):
            b = untagged[j]
            if j not in used and _compatible(a, b) and not np.isclose(
                    a.y_translation_m, b.y_translation_m, atol=1e-6) and \
                    all(not np.isclose(c.y_translation_m, b.y_translation_m, atol=1e-6) for c in grp):
                grp.append(b)
                used.add(j)
        if len(grp) >= 2:
            out.append(sorted(grp, key=lambda s: abs(s.estimated_cor_from_motor_pix)))
    return out


def pair_companions(scans: list[BlissScan]) -> dict:
    """Identify the extended-FOV pair: ``{"main": HA scan, "companion": TA scan}``.

    See :func:`group_companions` for the rule. Exactly one family is expected; for a
    family with more than two members (TA2...), ``"companions"`` lists them all in order
    (``"companion"`` is the first). Raises ValueError if no or several families exist,
    or if the members are geometrically incompatible.
    """
    groups = group_companions(list(scans))
    if len(groups) != 1:
        raise ValueError(f"expected one HA/TA family, found {len(groups)}: "
                         f"{[[s.name for s in g] for g in groups]}")
    g = groups[0]
    for c in g[1:]:
        if not _compatible(g[0], c):
            raise ValueError(f"{g[0].name} and {c.name} are not compatible companions")
    return {"main": g[0], "companion": g[1], "companions": g[1:]}
