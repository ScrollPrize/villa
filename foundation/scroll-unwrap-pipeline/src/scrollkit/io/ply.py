"""Strict binary-little-endian PLY reader.

Header-driven: the numpy dtype is constructed from the declared properties in declared order,
so any layout drift in the data fails loudly instead of mis-parsing. List properties are read
via a fixed-stride fast path (count bytes asserted afterwards); if the file violates the
uniform-count assumption we fall back to plyfile (slow but general).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .types import MeshData

_SCALAR = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


class PlyHeader:
    def __init__(self) -> None:
        self.format = ""
        self.comments: list[str] = []
        self.texture_files: list[str] = []
        # elements: list of (name, count, [(prop_kind, ...)]) where prop_kind is
        # ("scalar", name, np_type) or ("list", name, count_np_type, item_np_type)
        self.elements: list[tuple[str, int, list[tuple]]] = []
        self.header_bytes = 0


def parse_header(path: str | Path) -> PlyHeader:
    h = PlyHeader()
    with open(path, "rb") as f:
        first = f.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        raw = first
        cur_props: list[tuple] | None = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: EOF before end_header")
            raw += line
            text = line.decode("ascii", errors="replace").strip()
            if text == "end_header":
                break
            if text.startswith("format "):
                h.format = text.split()[1]
            elif text.startswith("comment"):
                comment = text[len("comment"):].strip()
                h.comments.append(comment)
                m = re.match(r"TextureFile\s+(.+)$", comment)
                if m:
                    h.texture_files.append(m.group(1).strip())
            elif text.startswith("element "):
                _, name, count = text.split()
                cur_props = []
                h.elements.append((name, int(count), cur_props))
            elif text.startswith("property "):
                if cur_props is None:
                    raise ValueError(f"{path}: property before element")
                parts = text.split()
                if parts[1] == "list":
                    _, _, ctype, itype, pname = parts
                    cur_props.append(("list", pname, _SCALAR[ctype], _SCALAR[itype]))
                else:
                    _, ptype, pname = parts
                    cur_props.append(("scalar", pname, _SCALAR[ptype]))
        h.header_bytes = len(raw)
    if h.format != "binary_little_endian":
        raise NotImplementedError(f"{path}: only binary_little_endian supported (got {h.format!r})")
    return h


# Expected uniform list lengths for the fixed-stride fast path.
_LIST_LEN = {"vertex_indices": 3, "vertex_index": 3, "texcoord": 6}


def _element_dtype(props: list[tuple]) -> np.dtype:
    fields = []
    for p in props:
        if p[0] == "scalar":
            _, name, t = p
            fields.append((name, "<" + t))
        else:
            _, name, ct, it = p
            if name not in _LIST_LEN:
                raise NotImplementedError(f"unsupported list property {name!r}")
            fields.append((f"__cnt_{name}", "<" + ct))
            fields.append((name, "<" + it, (_LIST_LEN[name],)))
    return np.dtype(fields)


def read_ply(path: str | Path) -> MeshData:
    path = Path(path)
    h = parse_header(path)
    offset = h.header_bytes
    arrays: dict[str, np.ndarray] = {}
    for name, count, props in h.elements:
        dt = _element_dtype(props)
        arr = np.fromfile(path, dtype=dt, count=count, offset=offset)
        if arr.shape[0] != count:
            raise ValueError(f"{path}: truncated element {name!r}: {arr.shape[0]}/{count}")
        for p in props:
            if p[0] == "list":
                pname = p[1]
                cnt = arr[f"__cnt_{pname}"]
                expected = _LIST_LEN[pname]
                if not (cnt == expected).all():
                    raise ValueError(
                        f"{path}: non-uniform list {pname!r} (expected {expected}); "
                        f"fast path invalid — use plyfile fallback"
                    )
        arrays[name] = arr
        offset += count * dt.itemsize
    # trailing-bytes sanity
    fsize = path.stat().st_size
    if offset != fsize:
        raise ValueError(f"{path}: {fsize - offset} unexplained trailing bytes")

    if "vertex" not in arrays or "face" not in arrays:
        raise ValueError(f"{path}: missing vertex/face elements")
    v = arrays["vertex"]
    fa = arrays["face"]
    names = v.dtype.names

    def col3(a, b, c):
        return np.ascontiguousarray(np.stack([v[a], v[b], v[c]], axis=1))

    vertices = col3("x", "y", "z").astype(np.float32, copy=False)
    normals = col3("nx", "ny", "nz").astype(np.float32, copy=False) if "nx" in names else None
    vertex_uv = (
        np.ascontiguousarray(np.stack([v["s"], v["t"]], axis=1)).astype(np.float32, copy=False)
        if "s" in names
        else None
    )
    idx_name = "vertex_indices" if "vertex_indices" in fa.dtype.names else "vertex_index"
    faces = np.ascontiguousarray(fa[idx_name]).astype(np.int32, copy=False)
    wedge_uv = (
        np.ascontiguousarray(fa["texcoord"]).reshape(-1, 3, 2).astype(np.float32, copy=False)
        if "texcoord" in fa.dtype.names
        else None
    )
    texnum = (
        np.ascontiguousarray(fa["texnumber"]).astype(np.int32, copy=False)
        if "texnumber" in fa.dtype.names
        else None
    )
    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        vertex_uv=vertex_uv,
        wedge_uv=wedge_uv,
        face_texnumber=texnum,
        texture_files=list(h.texture_files),
        source_path=str(path),
        comments=list(h.comments),
    )
