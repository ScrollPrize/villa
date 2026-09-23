"""Nearest-cell patch normals with a cached exclusion halo for Lasagna fallback."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class PatchNormals:
    def __init__(self, cache, cell_size, shape, z_roi):
        self.cache = cache
        self.cell_size = cell_size
        self.shape = shape
        self.z_roi = z_roi

    def sample(self, points, *, return_exclusion=False):
        # Export cells are floor(position / cell_size), not rounded OME indices.
        index = torch.floor(points.detach() / self.cell_size).long()
        shape = torch.tensor(self.shape, device=points.device)
        valid = ((index >= 0) & (index < shape)).all(dim=-1)
        valid &= (index[..., 0] >= self.z_roi[0]) & (index[..., 0] < self.z_roi[1])
        safe = torch.minimum(index.clamp(min=0), shape - 1)
        if return_exclusion:
            encoded, near = self.cache.gather_with_exclusion(safe)
            near = near & valid
        else:
            encoded = self.cache.gather(safe)
        valid &= (encoded != 0).any(dim=-1)
        normal = F.normalize((encoded.flip(-1).float() - 128) / 127, dim=-1)
        return (normal, valid, near) if return_exclusion else (normal, valid)

    def close(self):
        self.cache.close()


def load_patch_normals(path, *, z_begin, z_end, progress=None, device='cuda',
                       exclusion_radius=8., cache_directory=None):
    from compact_patch_normals import CompactPatchNormalPool
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('artifact_type') != 'signed_patch_normal_volume'
            or manifest.get('format_version') != 1 or not manifest.get('complete')):
        raise ValueError(f'{path}: expected a complete signed patch-normal export')
    cell_size = float(manifest['cell_size_fitter_voxels'])
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError(f'{path}: invalid patch-normal cell size')
    sidecar = path / 'signed_normals_u8.respool'
    meta = json.loads((sidecar / 'meta.json').read_text())
    encoding = meta.get('normal_encoding', {})
    if (encoding.get('name') != 'signed_unit_vector_u8'
            or encoding.get('components') != ['nx', 'ny', 'nz']
            or encoding.get('offset') != 128 or encoding.get('scale') != 127
            or encoding.get('missing_vector') != [0, 0, 0]):
        raise ValueError(f'{path}: unsupported patch-normal encoding')
    if meta.get('source_metadata', {}).get('cell_size_fitter_voxels') != cell_size:
        raise ValueError(f'{path}: patch-normal sidecar grid differs from manifest')
    shape = tuple(int(v) for v in manifest['shape_zyx'])
    z_roi = (max(0, int(np.floor(z_begin / cell_size))),
             min(shape[0], int(np.ceil(z_end / cell_size))))
    if progress is not None:
        progress.begin('loading', 'Loading patch normals onto GPU',
                       step=0, total_steps=0, unit='bricks')
    cache = CompactPatchNormalPool(
        str(sidecar), z_roi=z_roi, device=device,
        expected_shape_zyx=shape,
        exclusion_radius_cells=float(exclusion_radius) / cell_size,
        cache_directory=cache_directory,
        progress_callback=(lambda current, total, detail: progress.update(
            current, total_steps=total, detail=detail)) if progress else None)
    return PatchNormals(cache, cell_size, shape, z_roi)


def override_patch_normals(volume, points, fallback, fallback_weight, *, return_presence=False):
    normal, present, near = volume['patch_normals'].sample(points, return_exclusion=True)
    fallback_weight = torch.where(near, torch.zeros_like(fallback_weight), fallback_weight)
    result = (torch.where(present[..., None], normal, fallback),
              torch.where(present, torch.ones_like(fallback_weight), fallback_weight))
    return (*result, present) if return_presence else result
