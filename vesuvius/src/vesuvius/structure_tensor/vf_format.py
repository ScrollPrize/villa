#vf_format.py
from __future__ import annotations
import numpy as np
from vc_reader.reader import ZarrWriteGroup, ZarrWriter, create_group
from typing import Any, Dict, Optional, Tuple

_DEFAULT_CODEC = {"cname": "zstd", "level": 1, "shuffle": 1}

def zarr_codec(name: str, level: int) -> Dict[str, Any]:
    """``ZarrWriter.create`` arguments for ``--zarr-compressor``: an unknown name is zstd at level 1."""
    name = name.lower()
    if name == "none":
        return {"compressor": None}
    if name in ("zstd", "lz4", "zlib"):
        return {"cname": name, "level": level, "shuffle": 1}
    return dict(_DEFAULT_CODEC)

def require_array(group: ZarrWriteGroup, name: str, *, shape, **create) -> ZarrWriter:
    """The array ``name`` when it exists with this shape, keeping what it holds; otherwise a new one."""
    if name in group:
        existing = group[name]
        if not isinstance(existing, ZarrWriter):
            raise TypeError(f"{name!r} exists in {group.path!r} but is a group, not an array")
        if existing.shape != tuple(int(n) for n in shape):
            raise ValueError(f"array {name!r} in {group.path!r} has shape {existing.shape}, expected {tuple(shape)}")
        return existing
    return group.create_array(name, shape=shape, **create)

def encode_dir_to_u8(d: np.ndarray) -> np.ndarray:
    """
    Map float direction component in [-1, 1] -> uint8 via round(d*127 + 128), clipped.
    NaN -> 128, +Inf -> 255, -Inf -> 0.
    """
    out = np.round(d * 127.0 + 128.0)
    out = np.nan_to_num(out, nan=128.0, posinf=255.0, neginf=0.0)
    return out.clip(0.0, 255.0).astype(np.uint8, copy=False)

def encode_conf_to_u8(c: np.ndarray) -> np.ndarray:
    """
    Map scalar confidence to uint8. Values are clamped to [0,1] then scaled by 255.
    NaN -> 0.
    """
    c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    return np.round(c * 255.0).astype(np.uint8, copy=False)

class OMEU8VectorWriter:
    """
    Write an OME-ish uint8 layout for a 3D vector field:

      <root>/<group_name>/{z,y,x}/<scale_name>   -> uint8 [Zds, Yds, Xds]
      <root>/confidence/<scale_name>             -> uint8 [Zds, Yds, Xds] (optional)

    Components are ordered (z=0, y=1, x=2) to match your eigenvector layout.
    """
    def __init__(
        self,
        output_path: str,
        group_name: str,
        vol_shape_zyx: Tuple[int, int, int],
        *,
        chunks_zyx: Tuple[int, int, int] = (128, 128, 128),
        codec: Optional[Dict[str, Any]] = None,
        scale_name: str = "0",
        downsample: int = 1,
        make_confidence: bool = False,
    ):
        codec = _DEFAULT_CODEC if codec is None else codec
        self.output_path = output_path
        self.group_name = group_name
        self.scale_name = str(scale_name)
        self.ds = int(downsample)
        self.make_conf = bool(make_confidence)
        if self.ds <= 0:
            raise ValueError("downsample must be >= 1")

        Z, Y, X = vol_shape_zyx
        Zds = (Z + self.ds - 1) // self.ds
        Yds = (Y + self.ds - 1) // self.ds
        Xds = (X + self.ds - 1) // self.ds
        self.shape_ds = (Zds, Yds, Xds)

        root = create_group(output_path)
        grp = root.create_group(group_name)
        self.ds_z = self._require_scale(grp.create_group("z"), chunks_zyx, codec)
        self.ds_y = self._require_scale(grp.create_group("y"), chunks_zyx, codec)
        self.ds_x = self._require_scale(grp.create_group("x"), chunks_zyx, codec)

        self.ds_conf = None
        if self.make_conf:
            self.ds_conf = self._require_scale(root.create_group("confidence"), chunks_zyx, codec)

    def _require_scale(self, g: ZarrWriteGroup, chunks, codec):
        array = require_array(g, self.scale_name, shape=self.shape_ds, chunks=chunks, dtype=np.uint8, **codec)
        array.skip_empty = True
        return array

    def _down_bounds(self, z0, z1, y0, y1, x0, x1):
        s = self.ds
        return (z0 // s, z1 // s, y0 // s, y1 // s, x0 // s, x1 // s)

    def write_block(
        self,
        *,
        z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
        directions_block_zyx: np.ndarray,   # [Dz,Dy,Dx,3] comp=(z,y,x)
        confidence_block: Optional[np.ndarray] = None,  # [Dz,Dy,Dx]
    ):
        if directions_block_zyx.ndim != 4 or directions_block_zyx.shape[-1] != 3:
            raise ValueError(f"directions_block_zyx must be [Dz,Dy,Dx,3], got {directions_block_zyx.shape}")
        Dz, Dy, Dx, _ = directions_block_zyx.shape
        assert Dz == (z1 - z0) and Dy == (y1 - y0) and Dx == (x1 - x0)

        # nearest-neighbour DS for now (consistent with // indexing)
        dsub = directions_block_zyx[::self.ds, ::self.ds, ::self.ds, :] if self.ds > 1 else directions_block_zyx
        z_u8 = encode_dir_to_u8(dsub[..., 0])
        y_u8 = encode_dir_to_u8(dsub[..., 1])
        x_u8 = encode_dir_to_u8(dsub[..., 2])

        z0d, z1d, y0d, y1d, x0d, x1d = self._down_bounds(z0, z1, y0, y1, x0, x1)
        self.ds_z[z0d:z1d, y0d:y1d, x0d:x1d] = z_u8
        self.ds_y[z0d:z1d, y0d:y1d, x0d:x1d] = y_u8
        self.ds_x[z0d:z1d, y0d:y1d, x0d:x1d] = x_u8

        if self.ds_conf is not None:
            if confidence_block is None:
                raise ValueError("make_confidence=True but confidence_block=None")
            csub = confidence_block[::self.ds, ::self.ds, ::self.ds] if self.ds > 1 else confidence_block
            self.ds_conf[z0d:z1d, y0d:y1d, x0d:x1d] = encode_conf_to_u8(csub)

    def close(self):
        pass
