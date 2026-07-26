"""Strict OBJ + MTL writer/reader.

Float formatting is %.9g everywhere: 9 significant decimal digits round-trip IEEE-754
binary32 exactly. Faces are written 1-based, order-preserving. Two UV layouts:

- "shared": per-vertex UV, `f v/vt[/vn]` with identical indices (Groups A/B after the
  audit proves wedge == vertex UV bit-exactly).
- "wedge": per-corner UV deduplicated bit-exactly into a vt table, `f v/vt` (Group C).

Reading parses floats as float64 then casts to float32 — the guaranteed round-trip path.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np

from .types import MeshData


def _fmt_block(prefix: str, arr: np.ndarray) -> str:
    """Format an (n,k) float32 array as '<prefix> a b c' lines with %.9g."""
    a = np.asarray(arr, dtype=np.float32)
    cols = a.shape[1]
    pat = prefix + " " + " ".join(["%.9g"] * cols) + "\n"
    return "".join(pat % tuple(row) for row in a)


def dedup_wedge_uv(wedge_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact dedup of (m,3,2) wedge UVs.

    Returns (vt, corner_idx): vt is (k,2) float32 unique rows, corner_idx is (m,3) int64
    indices into vt. Dedup compares raw bits (void view) so distinct float encodings never merge.
    """
    flat = np.ascontiguousarray(wedge_uv.reshape(-1, 2).astype(np.float32, copy=False))
    as_void = flat.view([("", "V8")]).ravel()
    _, first_idx, inverse = np.unique(as_void, return_index=True, return_inverse=True)
    vt = flat[first_idx]
    return vt, inverse.reshape(-1, 3)


def write_mtl(path: str | Path, materials: list[tuple[str, str | None]]) -> None:
    """materials: list of (material_name, texture_filename_or_None).

    map_d references the same RGBA image so OBJ viewers honor the texture's alpha
    (lacunae render as holes, matching our alpha-test renderer). Without it the
    MTL declares d=1.0 and viewers paint alpha-0 regions with the PNG's undefined
    RGB — solid garbage patches instead of holes."""
    lines = ["# scrollkit material library\n"]
    for name, tex in materials:
        lines.append(f"newmtl {name}\n")
        lines.append("Ka 0.200000 0.200000 0.200000\n")
        lines.append("Kd 1.000000 1.000000 1.000000\n")
        lines.append("Ks 0.000000 0.000000 0.000000\n")
        lines.append("d 1.000000\nillum 1\n")
        if tex:
            lines.append(f"map_Kd {tex}\n")
            lines.append(f"map_d {tex}\n")
    Path(path).write_text("".join(lines), encoding="ascii")


def write_obj(
    path: str | Path,
    mesh: MeshData,
    *,
    texture_file: str | None = None,
    mtl_path: str | Path | None = None,
    material_name: str = "material_0",
    uv_mode: str = "auto",
    header_comment: str | None = None,
) -> None:
    """Write mesh to OBJ (+MTL if texture_file given). Order- and bit-preserving."""
    path = Path(path)
    if uv_mode == "auto":
        if mesh.vertex_uv is not None and mesh.wedge_equals_vertex_uv() in (True, None):
            uv_mode = "shared"
        elif mesh.wedge_uv is not None:
            uv_mode = "wedge"
        else:
            uv_mode = "none"

    parts: list[str] = []
    parts.append(f"# scrollkit exact export of {Path(mesh.source_path).name or 'mesh'}\n")
    if header_comment:
        parts.append(f"# {header_comment}\n")
    if texture_file:
        mtl_path = Path(mtl_path) if mtl_path else path.with_suffix(".mtl")
        write_mtl(mtl_path, [(material_name, texture_file)])
        parts.append(f"mtllib {mtl_path.name}\n")

    parts.append(_fmt_block("v", mesh.vertices))

    f1 = mesh.faces.astype(np.int64) + 1  # 1-based
    if uv_mode == "shared":
        parts.append(_fmt_block("vt", mesh.vertex_uv))
        if mesh.normals is not None:
            parts.append(_fmt_block("vn", mesh.normals))
        if texture_file:
            parts.append(f"usemtl {material_name}\n")
        if mesh.normals is not None:
            pat = "f %d/%d/%d %d/%d/%d %d/%d/%d\n"
            tri = np.repeat(f1, 3, axis=1)  # v/vt/vn all share the index
        else:
            pat = "f %d/%d %d/%d %d/%d\n"
            tri = np.repeat(f1, 2, axis=1)
        parts.append("".join(pat % tuple(row) for row in tri))
    elif uv_mode == "wedge":
        vt, corner = dedup_wedge_uv(mesh.wedge_uv)
        parts.append(_fmt_block("vt", vt))
        if mesh.normals is not None:
            parts.append(_fmt_block("vn", mesh.normals))
        if texture_file:
            parts.append(f"usemtl {material_name}\n")
        c1 = corner + 1
        if mesh.normals is not None:
            pat = "f %d/%d/%d %d/%d/%d %d/%d/%d\n"
            tri = np.stack([f1[:, 0], c1[:, 0], f1[:, 0],
                            f1[:, 1], c1[:, 1], f1[:, 1],
                            f1[:, 2], c1[:, 2], f1[:, 2]], axis=1)
        else:
            pat = "f %d/%d %d/%d %d/%d\n"
            tri = np.stack([f1[:, 0], c1[:, 0],
                            f1[:, 1], c1[:, 1],
                            f1[:, 2], c1[:, 2]], axis=1)
        parts.append("".join(pat % tuple(row) for row in tri))
    else:  # no UV
        if mesh.normals is not None:
            parts.append(_fmt_block("vn", mesh.normals))
            pat = "f %d//%d %d//%d %d//%d\n"
            tri = np.repeat(f1, 2, axis=1)
        else:
            pat = "f %d %d %d\n"
            tri = f1
        parts.append("".join(pat % tuple(row) for row in tri))

    path.write_text("".join(parts), encoding="ascii")


class ObjFrameWriter:
    """Fast per-frame OBJ export: vt/vn/f/usemtl bytes are constant per mesh and cached;
    only the `v` block is reformatted each frame."""

    def __init__(
        self,
        mesh: MeshData,
        *,
        texture_file: str | None,
        material_name: str = "material_0",
        mtl_name: str | None = None,
        header_comment: str = "scrollkit unwrap animation frame",
    ) -> None:
        self.mesh = mesh
        head = [f"# {header_comment}\n"]
        if texture_file:
            head.append(f"mtllib {mtl_name}\n")
        self._head = "".join(head).encode("ascii")

        tail: list[str] = []
        f1 = mesh.faces.astype(np.int64) + 1
        if mesh.vertex_uv is not None:
            tail.append(_fmt_block("vt", mesh.vertex_uv))
            if texture_file:
                tail.append(f"usemtl {material_name}\n")
            pat = "f %d/%d %d/%d %d/%d\n"
            tri = np.repeat(f1, 2, axis=1)
            tail.append("".join(pat % tuple(row) for row in tri))
        else:
            if texture_file:
                tail.append(f"usemtl {material_name}\n")
            tail.append("".join("f %d %d %d\n" % tuple(row) for row in f1))
        self._tail = "".join(tail).encode("ascii")

    def write_frame(self, path: str | Path, vertices: np.ndarray) -> None:
        v = np.asarray(vertices, dtype=np.float32)
        body = _fmt_block("v", v).encode("ascii")
        with open(path, "wb") as f:
            f.write(self._head)
            f.write(body)
            f.write(self._tail)


class ShellObjFrameWriter:
    """Per-frame two-sided THIN-SHELL OBJ: OBJ/MTL cannot express per-side textures
    on one surface, so the ink variant ships a shell — the recto surface (ink
    texture) and the verso surface (plain papyrus) offset ±delta along per-frame
    vertex normals so any viewer renders both sides correctly without z-fighting.

    Layout: v block = [recto verts; verso verts] (2n, per frame), vt written once
    (both surfaces share UVs), faces in two usemtl groups; recto winding is chosen
    so its outward normal points to the inner-face side (inner_sign), verso the
    opposite."""

    def __init__(
        self,
        mesh: MeshData,
        *,
        inner_sign: float,
        recto_material: str = "ink_recto",
        verso_material: str = "plain_verso",
        mtl_name: str = "mesh.mtl",
        delta_frac: float = 0.10,
        header_comment: str = "scrollkit unwrap frame (two-sided shell)",
    ) -> None:
        self.F = mesh.faces.astype(np.int64)
        self.n = mesh.vertices.shape[0]
        e = mesh.vertices[self.F[:, 1]] - mesh.vertices[self.F[:, 0]]
        self.delta = float(delta_frac * np.median(np.linalg.norm(e, axis=1)))
        self.inner_sign = 1.0 if inner_sign >= 0 else -1.0
        self._head = (f"# {header_comment}\nmtllib {mtl_name}\n").encode("ascii")

        f1 = self.F + 1
        flipped = f1[:, ::-1]
        # outward normal of the RECTO surface must equal the inner-face direction:
        # original winding has outward +n, flipped has -n.
        recto_f, verso_f = (f1, flipped) if self.inner_sign > 0 else (flipped, f1)
        verso_f = verso_f + np.array([[self.n, self.n, self.n]])
        tail = [_fmt_block("vt", mesh.vertex_uv)]
        pat = "f %d/%d %d/%d %d/%d\n"
        tail.append(f"usemtl {recto_material}\n")
        tail.append("".join(pat % tuple(np.repeat(r, 2)) for r in recto_f))
        tail.append(f"usemtl {verso_material}\n")
        # verso vt indices reference the shared vt block (1..n)
        vv = np.empty((len(verso_f), 6), np.int64)
        vv[:, 0::2] = verso_f
        vv[:, 1::2] = verso_f - self.n
        tail.append("".join(pat % tuple(r) for r in vv))
        self._tail = "".join(tail).encode("ascii")

    def write_frame(self, path: str | Path, vertices: np.ndarray) -> None:
        V = np.asarray(vertices, dtype=np.float64)
        fn = np.cross(V[self.F[:, 1]] - V[self.F[:, 0]], V[self.F[:, 2]] - V[self.F[:, 0]])
        vn = np.zeros_like(V)
        for j in range(3):
            np.add.at(vn, self.F[:, j], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)
        off = (self.inner_sign * self.delta) * vn
        both = np.concatenate([V + off, V - off]).astype(np.float32)
        body = _fmt_block("v", both).encode("ascii")
        with open(path, "wb") as f:
            f.write(self._head)
            f.write(body)
            f.write(self._tail)


def read_obj(path: str | Path) -> MeshData:
    """Strict OBJ reader for files we wrote (v/vt/vn/f, single object, one material).

    Floats parsed as float64 then cast float32 (exact round-trip for %.9g output).
    """
    path = Path(path)
    v_rows: list[str] = []
    vt_rows: list[str] = []
    vn_rows: list[str] = []
    f_rows: list[str] = []
    mtllib: str | None = None
    usemtl: list[str] = []
    with open(path, "r", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("v "):
                v_rows.append(line)
            elif line.startswith("vt "):
                vt_rows.append(line)
            elif line.startswith("vn "):
                vn_rows.append(line)
            elif line.startswith("f "):
                f_rows.append(line)
            elif line.startswith("mtllib"):
                mtllib = line.split(None, 1)[1].strip()
            elif line.startswith("usemtl"):
                usemtl.append(line.split(None, 1)[1].strip())

    def parse_floats(rows: list[str], k: int) -> np.ndarray | None:
        if not rows:
            return None
        out = np.empty((len(rows), k), dtype=np.float64)
        for i, r in enumerate(rows):
            parts = r.split()
            out[i] = [float(x) for x in parts[1 : 1 + k]]
        return out.astype(np.float32)

    vertices = parse_floats(v_rows, 3)
    vt = parse_floats(vt_rows, 2)
    vn = parse_floats(vn_rows, 3)
    if vertices is None or not f_rows:
        raise ValueError(f"{path}: no geometry")

    m = len(f_rows)
    fv = np.empty((m, 3), dtype=np.int64)
    ft = np.full((m, 3), -1, dtype=np.int64)
    fn = np.full((m, 3), -1, dtype=np.int64)
    for i, r in enumerate(f_rows):
        corners = r.split()[1:]
        if len(corners) != 3:
            raise ValueError(f"{path}: non-triangle face at row {i}")
        for j, c in enumerate(corners):
            sub = c.split("/")
            fv[i, j] = int(sub[0])
            if len(sub) > 1 and sub[1]:
                ft[i, j] = int(sub[1])
            if len(sub) > 2 and sub[2]:
                fn[i, j] = int(sub[2])
    if (fv <= 0).any():
        raise ValueError(f"{path}: negative/zero OBJ indices unsupported")
    faces = (fv - 1).astype(np.int32)

    vertex_uv = None
    wedge_uv = None
    if vt is not None and (ft > 0).all():
        ft0 = ft - 1
        if np.array_equal(ft0, fv - 1) and vt.shape[0] == vertices.shape[0]:
            vertex_uv = vt
        else:
            wedge_uv = np.ascontiguousarray(vt[ft0]).astype(np.float32)
    normals = None
    if vn is not None and (fn > 0).all():
        fn0 = fn - 1
        if np.array_equal(fn0, fv - 1) and vn.shape[0] == vertices.shape[0]:
            normals = vn
        else:
            raise ValueError(f"{path}: per-corner normals unsupported by strict reader")

    textures: list[str] = []
    if mtllib:
        mtl_file = path.parent / mtllib
        if mtl_file.exists():
            for line in mtl_file.read_text(encoding="ascii").splitlines():
                line = line.strip()
                if line.startswith("map_Kd"):
                    textures.append(line.split(None, 1)[1].strip())
    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        vertex_uv=vertex_uv,
        wedge_uv=wedge_uv,
        texture_files=textures,
        source_path=str(path),
    )


def copy_texture(src: str | Path, dst: str | Path) -> str:
    """Byte-copy a texture and return its SHA256 (verified equal on both sides)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    h_src = hashlib.sha256(src.read_bytes()).hexdigest()
    h_dst = hashlib.sha256(dst.read_bytes()).hexdigest()
    if h_src != h_dst:
        raise IOError(f"texture copy corrupted: {src} -> {dst}")
    return h_src
