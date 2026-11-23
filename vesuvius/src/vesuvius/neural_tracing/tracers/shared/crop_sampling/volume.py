"""The tracers' CT volume settings.

Arrays are indexed ``z, y, x``. Tracing uses the trace grid (``grid_scale`` base voxels per trace voxel); CT is read
at a finer native grid, and ``input_scale`` maps trace coordinates into the selected CT array. Fiber JSON uses
base-voxel xyz.

The CT is read with vc_reader: ``vc_reader.reader.open_zarr(spec.ct_zarr, level=spec.ct_level,
cache_dir=spec.cache_dir, memory_cache=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vesuvius.neural_tracing.tracers.shared.retired import ANY, retire


# Removed volume fields and their only supported value (ANY: every value), as recorded in older checkpoints: the
# fiber prediction zarrs (read only for the presence channel), presence inputs and a separate model-crop CT level.
RETIRED_SPEC_FIELDS = dict(fiber_zarr_dir=ANY, fiber_level=ANY, inputs='ct', load_presence=False,
                           crop_ct_level=None, crop_ct_grid_scale=None)


@dataclass
class CTVolumeSpec:
    ct_zarr: str
    ct_level: int = 1
    ct_grid_scale: float = 8.0  # base voxels per selected CT voxel; s1_ds2 level 0 = 4
    grid_scale: float = 8.0  # base voxels per trace-grid voxel
    cache_dir: str | None = None  # vc_reader disk cache of remote CT: a directory, or 'vc3d' (VC3D's cache root).
    ct_normalization: dict | None = None  # Bound at startup, persisted in run JSON/checkpoints.

    @property
    def input_scale(self) -> float:
        """CT voxels per trace voxel (a whole number)."""
        if min(self.grid_scale, self.ct_grid_scale) <= 0:
            raise ValueError('Invalid voxel scale')
        scale = self.grid_scale/self.ct_grid_scale
        if not scale.is_integer():
            raise ValueError('CT sampling currently requires an integer number of CT voxels per trace voxel')
        return scale

    @classmethod
    def from_dict(cls, values):
        """A recorded spec (``to_dict``, checkpoints' ``vol_spec``), including one written before fields were removed."""
        return cls(**retire(values, RETIRED_SPEC_FIELDS, 'volume'))

    def to_dict(self) -> dict:
        result = dict(self.__dict__)
        # Omit an unspecified cache location from portable volume metadata.
        if self.cache_dir is None:
            result.pop('cache_dir')
        return result
