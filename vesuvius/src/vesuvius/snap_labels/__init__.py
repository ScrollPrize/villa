"""Snap tifxyz surface labels onto the papyrus recto face of the CT, with a per-vertex abstain mask.

Entry point: ``vesuvius.snap_labels`` (see ``cli.py`` and README.md). numpy, tifffile and zarr only.
"""

from .face import FACE_REASONS, normals, self_check, snap_face
from .labels import check_outputs, label_snap, write_outputs, write_surface_label
from .meshio import load_mesh, load_umbilicus, write_mesh
from .volume import Sampler, Volume

__all__ = ["FACE_REASONS", "Sampler", "Volume", "check_outputs", "label_snap", "load_mesh", "load_umbilicus",
           "normals", "self_check", "snap_face", "write_mesh", "write_outputs", "write_surface_label"]
