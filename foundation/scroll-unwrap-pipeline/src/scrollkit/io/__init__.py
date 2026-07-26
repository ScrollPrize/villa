"""Strict, convention-pinned mesh IO. Every byte that touches disk goes through here."""

from .obj import ObjFrameWriter, copy_texture, dedup_wedge_uv, read_obj, write_mtl, write_obj
from .ply import parse_header, read_ply
from .types import MeshData

__all__ = [
    "MeshData",
    "ObjFrameWriter",
    "copy_texture",
    "dedup_wedge_uv",
    "parse_header",
    "read_obj",
    "read_ply",
    "write_mtl",
    "write_obj",
]
