import numpy as np

def vectorized_point_to_polyline_distance(point, polyline):
    """
    Compute the minimum distance from a point (2,) to a polyline.
    Uses vectorized operations.
    """
    p1 = polyline[:-1]
    p2 = polyline[1:]
    v = p2 - p1
    w = point - p1
    dot_wv = np.einsum('ij,ij->i', w, v)
    dot_vv = np.einsum('ij,ij->i', v, v)
    t = np.divide(dot_wv, dot_vv, out=np.zeros_like(dot_wv), where=dot_vv != 0)
    t = np.clip(t, 0, 1)
    proj = p1 + (t[:, None] * v)
    dists = np.linalg.norm(point - proj, axis=1)
    return np.min(dists)