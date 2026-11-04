"""Betti numbers of a single connected binary component (CPU)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def conn_to_skimage(conn: int, ndim: int) -> int:
    """Voxel connectivity (6/18/26 in 3D, 4/8 in 2D) to the skimage/scipy rank."""
    if ndim == 3:
        mapping = {6: 1, 18: 2, 26: 3}
    elif ndim == 2:
        mapping = {4: 1, 8: 2}
    else:
        raise ValueError(f"Unsupported ndim={ndim}")
    if conn not in mapping:
        raise ValueError(f"Invalid connectivity {conn} for ndim={ndim}")
    return mapping[conn]


def component_betti(mask: np.ndarray, fg_connectivity: int = 26, bg_connectivity: int = 6) -> tuple[int, int, int]:
    """(b1, b2, euler) of one connected component, via b1 = b0 + b2 - chi with b0 = 1.

    b2 counts background components that do not reach the padded border. Use complementary
    connectivities (26/6 or 6/26 in 3D) so the Euler number and the cavity count agree. 2D: b2 = 0.
    """
    from skimage.measure import euler_number

    m = np.pad(mask.astype(bool), 1)
    chi = int(euler_number(m, connectivity=conn_to_skimage(fg_connectivity, m.ndim)))
    if m.ndim == 2:
        return 1 - chi, 0, chi
    st = ndi.generate_binary_structure(3, conn_to_skimage(bg_connectivity, 3))
    _, n = ndi.label(~m, structure=st)
    b2 = n - 1  # the padding shell joins all outside background into one component
    return 1 + b2 - chi, b2, chi
