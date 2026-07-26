"""Core mesh container. All arrays keep their on-disk dtypes (float32/int32) — never silently upcast."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshData:
    """A triangle mesh as stored on disk.

    vertices: (n,3) float32
    normals: (n,3) float32 or None
    vertex_uv: (n,2) float32 or None        # per-vertex s,t when present in the file
    faces: (m,3) int32
    wedge_uv: (m,3,2) float32 or None       # per-corner texcoords (authoritative for texturing)
    face_texnumber: (m,) int32 or None
    texture_files: texture names from header comments / mtl, in declaration order
    source_path: provenance
    comments: full original header comments (PLY) for provenance
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    vertex_uv: np.ndarray | None = None
    wedge_uv: np.ndarray | None = None
    face_texnumber: np.ndarray | None = None
    texture_files: list[str] = field(default_factory=list)
    source_path: str = ""
    comments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        v, f = self.vertices, self.faces
        if v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices must be (n,3) float32, got {v.dtype} {v.shape}")
        if f.dtype != np.int32 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must be (m,3) int32, got {f.dtype} {f.shape}")
        if self.normals is not None and (self.normals.dtype != np.float32 or self.normals.shape != v.shape):
            raise ValueError("normals must be (n,3) float32 matching vertices")
        if self.vertex_uv is not None and (self.vertex_uv.dtype != np.float32 or self.vertex_uv.shape != (v.shape[0], 2)):
            raise ValueError("vertex_uv must be (n,2) float32")
        if self.wedge_uv is not None and (self.wedge_uv.dtype != np.float32 or self.wedge_uv.shape != (f.shape[0], 3, 2)):
            raise ValueError("wedge_uv must be (m,3,2) float32")
        if self.face_texnumber is not None and (self.face_texnumber.dtype != np.int32 or self.face_texnumber.shape != (f.shape[0],)):
            raise ValueError("face_texnumber must be (m,) int32")
        if f.size and (f.min() < 0 or f.max() >= v.shape[0]):
            raise ValueError("face indices out of range")

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def wedge_equals_vertex_uv(self) -> bool | None:
        """True iff per-wedge UVs are bit-identical to indexed per-vertex UVs. None if either is absent."""
        if self.wedge_uv is None or self.vertex_uv is None:
            return None
        gathered = self.vertex_uv[self.faces]  # (m,3,2)
        return bool(np.array_equal(gathered.view(np.uint32), self.wedge_uv.view(np.uint32)))
