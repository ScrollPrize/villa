"""Files: tifxyz grids in and out, single-band grid TIFFs, the scroll axis, and VC3D point collections."""

import json
from pathlib import Path

import numpy as np
import tifffile

VC3D_POINTS_VERSION = "1"   # volume-cartographer core/src/PointCollections.cpp


class MeshIOError(ValueError):
    pass


def load_mesh(folder):
    """(xs, ys, zs, valid) of a tifxyz folder as float32 grids; valid where every coordinate is > -0.5 and
    finite (VC3D writes -1 where a grid point has no vertex)."""
    d = Path(folder)
    if not (d / "x.tif").is_file():
        raise MeshIOError(f"{d}: no x.tif (not a tifxyz folder)")
    xs, ys, zs = (np.asarray(tifffile.imread(str(d / f"{a}.tif"))).astype(np.float32) for a in "xyz")
    if not (xs.ndim == 2 and xs.shape == ys.shape == zs.shape):
        raise MeshIOError(f"{d}: x, y and z tifs must be 2-D grids of one size")
    valid = (xs > -0.5) & (ys > -0.5) & (zs > -0.5) & np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    return xs, ys, zs, valid


def write_grid_tif(path, arr):
    """One 2-D grid as an uncompressed single-strip TIFF, dtype kept (VC3D memory-maps this layout)."""
    a = np.ascontiguousarray(arr)
    tifffile.imwrite(str(path), a, photometric="minisblack", compression=None, rowsperstrip=a.shape[0])


def read_grid_tif(path):
    return np.asarray(tifffile.imread(str(path)))


def write_mesh(src, out, xs, ys, zs, valid, record, note):
    """Write x/y/z.tif (float32, -1 where invalid) and meta.json (the source's, with a new bbox, the note appended
    to its source and the snap record) to out."""
    meta_path = Path(src) / "meta.json" if src is not None else None
    try:                                   # read it before writing anything, so a bad one leaves no half mesh
        meta = json.loads(meta_path.read_text()) if meta_path is not None and meta_path.is_file() else {
            "format": "tifxyz", "type": "seg"}
    except json.JSONDecodeError as exc:
        raise MeshIOError(f"{meta_path}: not JSON ({exc})") from exc
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in (("x", xs), ("y", ys), ("z", zs)):
        a = arr.astype(np.float32)
        a[~valid] = -1.0
        write_grid_tif(out / f"{name}.tif", a)
    if valid.any():
        v = valid
        meta["bbox"] = [[float(xs[v].min()), float(ys[v].min()), float(zs[v].min())],
                        [float(xs[v].max()), float(ys[v].max()), float(zs[v].max())]]
    meta["source"] = meta.get("source", "") + note
    meta["snap"] = record
    (out / "meta.json").write_text(json.dumps(meta, indent=4))
    return out


def vc3d_points(points, name="seeds", color=(1.0, 0.5, 0.0)):
    """A VC3D point collection document with one collection (PointCollections.cpp's JSON layout)."""
    pts = {str(i + 1): {"p": list(p), "creation_time": 0, "wind_a": None} for i, p in enumerate(points)}
    return {"vc_pointcollections_json_version": VC3D_POINTS_VERSION,
            "collections": {"1": {"name": name, "points": pts, "metadata": {"winding_is_absolute": False},
                                  "color": list(color)}}}


def parse_umbilicus(data):
    """Control points [(z, y, x)] sorted by z: JSON (an array of [z, y, x] or of objects with x, y, z, or an object
    with a "points" or "control_points" array), or text with one "z, y, x" line per point."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    if isinstance(data, str):
        text = data.strip()
        if text[:1] in ("[", "{"):
            data = json.loads(text)
        else:
            pts = []
            for n, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = [v.strip() for v in line.split(",") if v.strip()]
                if len(vals) != 3:
                    raise MeshIOError(f"umbilicus line {n}: expected z, y, x")
                pts.append(tuple(float(v) for v in vals))
            data = pts
    if isinstance(data, dict):
        for key in ("points", "control_points"):
            if key in data:
                data = data[key]
                break
        else:
            raise MeshIOError("umbilicus JSON needs an array, or a 'points' or 'control_points' array")
    pts = []
    for i, e in enumerate(data or []):
        if isinstance(e, dict):
            if not all(k in e for k in "xyz"):
                raise MeshIOError(f"umbilicus entry {i} lacks x, y or z")
            pts.append((float(e["z"]), float(e["y"]), float(e["x"])))
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            pts.append((float(e[0]), float(e[1]), float(e[2])))
        else:
            raise MeshIOError(f"umbilicus entry {i} is neither [z, y, x] nor an object with x, y, z")
    if not pts:
        raise MeshIOError("umbilicus has no points")
    return sorted(pts)


def load_umbilicus(source, scale=1.0):
    """Control points from a file or parsed data, every coordinate multiplied by scale."""
    if isinstance(source, (str, Path)) and Path(source).is_file():
        source = Path(source).read_text()
    pts = parse_umbilicus(source)
    return [(z * scale, y * scale, x * scale) for z, y, x in pts] if scale != 1 else pts
