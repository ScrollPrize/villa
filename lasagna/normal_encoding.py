from __future__ import annotations

import numpy as np


def decode_dir_angle(dir0: np.ndarray, dir1: np.ndarray) -> np.ndarray:
    """Decode Lasagna's two-channel ambiguous direction angle."""

    cos2t = 2.0 * dir0 - 1.0
    sin2t = cos2t - np.sqrt(2.0) * (2.0 * dir1 - 1.0)
    return np.arctan2(sin2t, cos2t) * 0.5


def estimate_normal(
    dir0_z: np.ndarray,
    dir1_z: np.ndarray,
    dir0_y: np.ndarray,
    dir1_y: np.ndarray,
    dir0_x: np.ndarray,
    dir1_x: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a Lasagna 3D normal from three ambiguous two-cos projections.

    Returns ``(w_z, w_y, w_x, nx_n, ny_n, nz_n)`` matching the historical
    predict3d implementation.
    """

    theta_z = decode_dir_angle(dir0_z, dir1_z)
    theta_y = decode_dir_angle(dir0_y, dir1_y)
    theta_x = decode_dir_angle(dir0_x, dir1_x)

    sz, cz = np.sin(theta_z), np.cos(theta_z)
    sy, cy = np.sin(theta_y), np.cos(theta_y)
    sx, cx = np.sin(theta_x), np.cos(theta_x)

    n1_x = cz * cy
    n1_y = sz * cy
    n1_z = cz * sy

    n2_x = cz * cx
    n2_y = sz * cx
    n2_z = sz * sx

    n3_x = cy * sx
    n3_y = sy * cx
    n3_z = sy * sx

    dot2 = n1_x * n2_x + n1_y * n2_y + n1_z * n2_z
    sign2 = np.where(dot2 >= 0, 1.0, -1.0)
    n2_x = n2_x * sign2
    n2_y = n2_y * sign2
    n2_z = n2_z * sign2

    dot3 = n1_x * n3_x + n1_y * n3_y + n1_z * n3_z
    sign3 = np.where(dot3 >= 0, 1.0, -1.0)
    n3_x = n3_x * sign3
    n3_y = n3_y * sign3
    n3_z = n3_z * sign3

    def _enc(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2 = gx * gx + gy * gy + eps
        c2 = (gx * gx - gy * gy) / r2
        s2 = 2.0 * gx * gy / r2
        isq2 = 1.0 / np.sqrt(2.0)
        return 0.5 + 0.5 * c2, 0.5 + 0.5 * (c2 - s2) * isq2

    scores = []
    for ncx, ncy, ncz in (
        (n1_x, n1_y, n1_z),
        (n2_x, n2_y, n2_z),
        (n3_x, n3_y, n3_z),
    ):
        pz0, pz1 = _enc(ncx, ncy)
        py0, py1 = _enc(ncx, ncz)
        px0, px1 = _enc(ncy, ncz)
        err_z = (pz0 - dir0_z) ** 2 + (pz1 - dir1_z) ** 2
        err_y = (py0 - dir0_y) ** 2 + (py1 - dir1_y) ** 2
        err_x = (px0 - dir0_x) ** 2 + (px1 - dir1_x) ** 2
        wz_c = ncx**2 + ncy**2
        wy_c = ncx**2 + ncz**2
        wx_c = ncy**2 + ncz**2
        total_err = wz_c * err_z + wy_c * err_y + wx_c * err_x
        scores.append(1.0 / (total_err + eps))

    s1, s2_s, s3_s = scores
    est_x = s1 * n1_x + s2_s * n2_x + s3_s * n3_x
    est_y = s1 * n1_y + s2_s * n2_y + s3_s * n3_y
    est_z = s1 * n1_z + s2_s * n2_z + s3_s * n3_z
    norm_e = np.sqrt(est_x**2 + est_y**2 + est_z**2) + eps
    est_x = est_x / norm_e
    est_y = est_y / norm_e
    est_z = est_z / norm_e

    wz2 = np.sqrt(est_x**2 + est_y**2 + eps)
    wy2 = np.sqrt(est_x**2 + est_z**2 + eps)
    wx2 = np.sqrt(est_y**2 + est_z**2 + eps)

    wzy = wz2 * wy2
    wzx = wz2 * wx2
    wyx = wy2 * wx2

    rn1_x = wzy * n1_x
    rn1_y = wzy * n1_y
    rn1_z = wzy * n1_z
    rn2_x = wzx * n2_x
    rn2_y = wzx * n2_y
    rn2_z = wzx * n2_z
    rn3_x = wyx * n3_x
    rn3_y = wyx * n3_y
    rn3_z = wyx * n3_z

    dot2r = rn1_x * rn2_x + rn1_y * rn2_y + rn1_z * rn2_z
    s2r = np.where(dot2r >= 0, 1.0, -1.0)
    rn2_x = rn2_x * s2r
    rn2_y = rn2_y * s2r
    rn2_z = rn2_z * s2r

    dot3r = rn1_x * rn3_x + rn1_y * rn3_y + rn1_z * rn3_z
    s3r = np.where(dot3r >= 0, 1.0, -1.0)
    rn3_x = rn3_x * s3r
    rn3_y = rn3_y * s3r
    rn3_z = rn3_z * s3r

    nx_f = rn1_x + rn2_x + rn3_x
    ny_f = rn1_y + rn2_y + rn3_y
    nz_f = rn1_z + rn2_z + rn3_z
    norm_f = np.sqrt(nx_f**2 + ny_f**2 + nz_f**2) + eps
    nx_n = nx_f / norm_f
    ny_n = ny_f / norm_f
    nz_n = nz_f / norm_f

    w_z = np.sqrt(nx_n * nx_n + ny_n * ny_n + eps)
    w_y = np.sqrt(nx_n * nx_n + nz_n * nz_n + eps)
    w_x = np.sqrt(ny_n * ny_n + nz_n * nz_n + eps)

    return w_z, w_y, w_x, nx_n, ny_n, nz_n

def encode_normal_nxny_u8(
    dir0_z: np.ndarray,
    dir1_z: np.ndarray,
    dir0_y: np.ndarray,
    dir1_y: np.ndarray,
    dir0_x: np.ndarray,
    dir1_x: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode Lasagna compact normal ``nx``/``ny`` uint8 channels."""

    _, _, _, nx_n, ny_n, nz_n = estimate_normal(
        dir0_z,
        dir1_z,
        dir0_y,
        dir1_y,
        dir0_x,
        dir1_x,
        eps=eps,
    )
    flip = np.where(nz_n < 0.0, -1.0, 1.0)
    nx_u8 = np.clip(np.round(nx_n * flip * 127.0 + 128.0), 0.0, 255.0).astype(
        np.uint8
    )
    ny_u8 = np.clip(np.round(ny_n * flip * 127.0 + 128.0), 0.0, 255.0).astype(
        np.uint8
    )
    return nx_u8, ny_u8


__all__ = [
    "decode_dir_angle",
    "encode_normal_nxny_u8",
    "estimate_normal",
]
