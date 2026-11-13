
import numpy as np
from uuid import uuid4
from vc_reader import tifxyz


def get_area(zyxs, step_size, voxel_size_um):
    valid_vertices = ~(zyxs == -1.).all(-1)
    valid_quads = valid_vertices[:-1, :-1] & valid_vertices[:-1, 1:] & valid_vertices[1:, :-1] & valid_vertices[1:, 1:]
    area_vx2 = int(valid_quads.sum()) * step_size ** 2
    area_cm2 = area_vx2 * voxel_size_um ** 2 / 1.e8
    return area_vx2, area_cm2


def save_tifxyz(zyxs, path, uuid, step_size, voxel_size_um, source, additional_metadata={},
                surface_uuid=None):
    """Write a tifxyz surface to ``path/uuid``.

    ``uuid`` names the output directory only. meta.json's ``uuid`` is the
    surface's persistent identity: pass ``surface_uuid`` to keep an existing
    identity (in-place rewrite); by default a new surface gets a fresh uuid4.
    """
    if surface_uuid is None:
        surface_uuid = str(uuid4())
    if hasattr(zyxs, "detach"):
        zyxs = zyxs.detach().cpu().numpy()
    else:
        zyxs = np.asarray(zyxs)
    zyxs = zyxs.astype(np.float32, copy=False)

    area_vx2, area_cm2 = get_area(zyxs, step_size, voxel_size_um)
    surface = tifxyz.Surface(zyxs[..., ::-1], scale=(1 / step_size, 1 / step_size), uuid=surface_uuid, extra={
        'area_vx2': area_vx2,
        'area_cm2': area_cm2,
        'source': source,
        **additional_metadata,
    })
    tifxyz.write(f'{path}/{uuid}', surface, compression='none', tile=None, mask=False, overwrite=True)
    return True
