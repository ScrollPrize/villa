
import os
import cv2
import json


def save_tifxyz(zyxs, path, uuid, step_size, voxel_size_um, source):
    path = f'{path}/{uuid}'
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f'{path}/x.tif', zyxs[..., 2])
    cv2.imwrite(f'{path}/y.tif', zyxs[..., 1])
    cv2.imwrite(f'{path}/z.tif', zyxs[..., 0])
    area_vx2 = (zyxs.shape[0] - 1) * (zyxs.shape[1] - 1) * step_size ** 2
    with open(f'{path}/meta.json', 'w') as f:
        json.dump({
            'scale': [1 / step_size, 1 / step_size],
            'bbox': [zyxs.min(axis=(0, 1))[::-1].tolist(), zyxs.max(axis=(0, 1))[::-1].tolist()],
            'area_vx2': area_vx2,
            'area_cm2': area_vx2 * voxel_size_um ** 2 / 1.e8,
            'format': 'tifxyz',
            'type': 'seg',
            'uuid': uuid,
            'source': source,
        }, f, indent=4)
    return True

