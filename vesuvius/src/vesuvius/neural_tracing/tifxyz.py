
import os
import cv2
import json
import numpy as np


def get_area(zyxs, step_size, voxel_size_um):
    valid_vertices = ~(zyxs == -1.).all(-1)
    valid_quads = valid_vertices[:-1, :-1] & valid_vertices[:-1, 1:] & valid_vertices[1:, :-1] & valid_vertices[1:, 1:]
    area_vx2 = int(valid_quads.sum()) * step_size ** 2
    area_cm2 = area_vx2 * voxel_size_um ** 2 / 1.e8
    return area_vx2, area_cm2


def save_tifxyz(zyxs, path, uuid, step_size, voxel_size_um, source, additional_metadata={}):
    if hasattr(zyxs, "detach"):
        zyxs = zyxs.detach().cpu().numpy()
    else:
        zyxs = np.asarray(zyxs)
    zyxs = zyxs.astype(np.float32)

    path = f'{path}/{uuid}'
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f'{path}/x.tif', zyxs[..., 2])
    cv2.imwrite(f'{path}/y.tif', zyxs[..., 1])
    cv2.imwrite(f'{path}/z.tif', zyxs[..., 0])
    area_vx2, area_cm2 = get_area(zyxs, step_size, voxel_size_um)
    with open(f'{path}/meta.json', 'w') as f:
        json.dump({
            'scale': [1 / step_size, 1 / step_size],
            'bbox': [zyxs.min(axis=(0, 1))[::-1].tolist(), zyxs.max(axis=(0, 1))[::-1].tolist()],
            'area_vx2': area_vx2,
            'area_cm2': area_cm2,
            'format': 'tifxyz',
            'type': 'seg',
            'uuid': uuid,
            'source': source,
            **additional_metadata
        }, f, indent=4)
    return True
