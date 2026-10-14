"""
Cross-level chunk occupancy check for OME-Zarr multiscale pyramids.

A coarse pyramid level is derived from the next finer one, so wherever the
finer level has chunk objects the coarser level must have them too. When a
coarse chunk is missing, zarr silently returns ``fill_value`` and viewers,
training loaders and annotators see blank papyrus although the data exists one
level down.

This check uses only object *listings* (via
:func:`vesuvius.data.zarr_chunk_index.build_chunk_occupancy`) plus the small
``.zattrs`` / ``.zarray`` JSON files. No chunk data is downloaded, so a full
scroll volume can be checked in seconds to minutes.

Only zarr v2 levels are supported (same limitation as ``zarr_chunk_index``);
levels that cannot be indexed are reported as skipped, never as holes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from vesuvius.data.zarr_chunk_index import _read_zarray, _url_to_fs, build_chunk_occupancy


@dataclass
class LevelPairResult:
    fine: str
    coarse: str
    expected: int = 0  # coarse chunks that must exist given fine occupancy
    missing: int = 0  # of those, how many have no object
    examples: List[Tuple[int, ...]] = field(default_factory=list)
    skipped: Optional[str] = None  # reason, if the pair could not be checked

    @property
    def ok(self) -> bool:
        return self.skipped is None and self.missing == 0


def expected_coarse_occupancy(
    fine: np.ndarray,
    fine_chunks: Sequence[int],
    coarse_chunks: Sequence[int],
    coarse_grid: Sequence[int],
    factors: Sequence[float],
) -> np.ndarray:
    """Coarse chunks that overlap at least one occupied fine chunk.

    Fine chunk ``i`` on an axis covers fine voxels ``[i*cf, (i+1)*cf)``, i.e.
    coarse voxels ``[i*cf/f, (i+1)*cf/f)``; every coarse chunk overlapping
    that interval is expected to exist.
    """
    fine = np.asarray(fine, dtype=bool)
    ndim = fine.ndim
    if not (len(fine_chunks) == len(coarse_chunks) == len(coarse_grid) == len(factors) == ndim):
        raise ValueError("rank mismatch between bitmap, chunks, grid and factors")
    # Per axis, the [lo, hi] coarse chunk range touched by each fine chunk index.
    ranges = []
    for ax in range(ndim):
        idx = np.arange(fine.shape[ax])
        span = coarse_chunks[ax] * factors[ax]
        lo = np.floor(idx * fine_chunks[ax] / span).astype(np.int64)
        hi = np.ceil((idx + 1) * fine_chunks[ax] / span).astype(np.int64) - 1
        ranges.append((np.minimum(lo, coarse_grid[ax] - 1), np.clip(hi, 0, coarse_grid[ax] - 1)))
    out = np.zeros(tuple(int(g) for g in coarse_grid), dtype=bool)
    for f_idx in np.argwhere(fine):
        sl = tuple(slice(ranges[ax][0][i], ranges[ax][1][i] + 1) for ax, i in enumerate(f_idx))
        out[sl] = True
    return out


def _read_json(url: str, *, anon: bool) -> Optional[dict]:
    """Parse a small JSON file; ``None`` only if it does not exist (other errors propagate)."""
    fs, path = _url_to_fs(url, anon=anon)
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as f:
        return json.loads(f.read())


def _levels_and_scales(store_url: str, *, anon: bool) -> List[Tuple[str, Optional[List[float]]]]:
    attrs = _read_json(store_url.rstrip("/") + "/.zattrs", anon=anon) or {}
    ms = (attrs.get("multiscales") or [{}])[0]
    out = []
    for ds in ms.get("datasets", []):
        scale = None
        for t in ds.get("coordinateTransformations", []) or []:
            if t.get("type") == "scale":
                scale = [float(s) for s in t["scale"]]
        out.append((str(ds["path"]), scale))
    return out


def _spatial(seq: Sequence) -> Tuple:
    return tuple(seq[-3:]) if len(seq) == 4 else tuple(seq)


def check_pyramid(store_url: str, *, anon: bool = False, max_examples: int = 10) -> List[LevelPairResult]:
    """Check every adjacent (finer -> coarser) level pair of an OME-Zarr store."""
    store_url = store_url.rstrip("/")
    levels = _levels_and_scales(store_url, anon=anon)
    if len(levels) < 2:
        return []

    info = {}
    for path, scale in levels:
        url = f"{store_url}/{path}"
        za = _read_zarray(url, anon=anon)
        occ = None
        if za is not None:
            occ = build_chunk_occupancy(
                url, tuple(za["chunks"]), tuple(za["shape"]), use_cache=False, anon=anon
            )
        info[path] = (za, occ, scale)

    results = []
    for (fpath, _), (cpath, _) in zip(levels, levels[1:]):
        res = LevelPairResult(fine=fpath, coarse=cpath)
        fza, focc, fscale = info[fpath]
        cza, cocc, cscale = info[cpath]
        if fza is None or cza is None:
            res.skipped = "missing .zarray"
        elif focc is None:
            res.skipped = f"level {fpath} has no indexable chunks"
        else:
            fshape, cshape = _spatial(fza["shape"]), _spatial(cza["shape"])
            fch, cch = _spatial(fza["chunks"]), _spatial(cza["chunks"])
            if fscale and cscale and len(fscale) == len(cscale):
                factors = [c / f for f, c in zip(_spatial(fscale), _spatial(cscale))]
            else:  # derive from shapes
                factors = [f / c for f, c in zip(fshape, cshape)]
            cgrid = [int(np.ceil(s / c)) for s, c in zip(cshape, cch)]
            expected = expected_coarse_occupancy(focc, fch, cch, cgrid, factors)
            have = cocc if cocc is not None else np.zeros(cgrid, dtype=bool)
            holes = expected & ~have
            res.expected = int(expected.sum())
            res.missing = int(holes.sum())
            res.examples = [tuple(int(v) for v in h) for h in np.argwhere(holes)[:max_examples]]
        results.append(res)
    return results
