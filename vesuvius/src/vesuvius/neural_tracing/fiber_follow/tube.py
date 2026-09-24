"""Metric Gaussian targets around the annotated polyline, including bends.

Distances are to segments, not a rasterized set of vertices. Unknown annotation
ends censor their outward half-space when they enter the crop; known nearby
curve bodies retain supervision even if they fold across that half-space.
"""
import numba
import numpy as np

from vesuvius.neural_tracing.fiber_follow.geometry import crop_local_grid, normalize


def tube_geometry(points, pos, frame, crop, sigma, endpoint_stop, offtrack=False):
    local = (points-pos) @ frame
    lower = np.array([crop.lateral_coords[0], crop.lateral_coords[0], crop.forward_coords[0]])
    upper = np.array([crop.lateral_coords[-1], crop.lateral_coords[-1], crop.forward_coords[-1]])
    segments = np.stack([local[:-1], local[1:]], 1)
    keep = ((segments.max(1) >= lower-4*sigma) & (segments.min(1) <= upper+4*sigma)).all(-1)
    caps = np.zeros((len(segments), 2), bool)
    caps[0, 0], caps[-1, 1] = not endpoint_stop[0], not endpoint_stop[1]
    endpoints = local[[0, -1]]
    tangents = normalize(np.stack([local[1]-local[0], local[-1]-local[-2]]), axis=-1)
    unknown = ~np.asarray(endpoint_stop, bool)
    unknown &= ((endpoints >= lower-4*sigma) & (endpoints <= upper+4*sigma)).all(-1)
    return dict(tube_segments=segments[keep], tube_caps=caps[keep],
                tube_endpoints=endpoints, tube_tangents=tangents, tube_unknown=unknown,
                tube_supervised=not offtrack, tube_sigma=sigma)


@numba.njit(cache=True)
def _rasterize(segments, caps, depth, width, behind, spacing, sigma):
    target = np.zeros((depth, width, width), np.float32)
    body = np.zeros((depth, width, width), np.bool_)
    origin = np.array([-(width-1)*spacing/2, -(width-1)*spacing/2, -behind*spacing])
    limits = np.array([width, width, depth])
    for i in range(len(segments)):
        a, b = segments[i, 0], segments[i, 1]
        delta = b-a
        norm2 = (delta*delta).sum()
        if norm2 < 1e-20:
            continue
        lo = np.maximum(0, np.ceil((np.minimum(a, b)-4*sigma-origin)/spacing).astype(np.int64))
        hi = np.minimum(limits-1, np.floor((np.maximum(a, b)+4*sigma-origin)/spacing).astype(np.int64))
        for z in range(lo[2], hi[2]+1):
            for y in range(lo[1], hi[1]+1):
                for x in range(lo[0], hi[0]+1):
                    q = origin + np.array([x, y, z])*spacing
                    t = min(1., max(0., ((q-a)*delta).sum()/norm2))
                    residual = q-(a+t*delta)
                    distance2 = (residual*residual).sum()
                    if distance2 > 16*sigma*sigma:
                        continue
                    value = np.exp(-distance2/(2*sigma*sigma))
                    is_body = not ((t == 0 and caps[i, 0]) or (t == 1 and caps[i, 1]))
                    if value > target[z, y, x]:
                        target[z, y, x] = value
                        body[z, y, x] = is_body
                    elif value >= target[z, y, x] and is_body:
                        body[z, y, x] = True
    return target, body


def render_tube(item, crop, sigma):
    target, body = _rasterize(item['tube_segments'], item['tube_caps'], crop.depth,
                              crop.width, crop.behind, crop.spacing, sigma)
    mask = np.ones_like(target)
    grid = crop_local_grid(crop)
    for i, sign in ((0, -1), (1, 1)):
        if item['tube_unknown'][i]:
            beyond = ((grid-item['tube_endpoints'][i])*item['tube_tangents'][i]).sum(-1)*sign > 0
            mask[beyond & ~body] = 0
    if not item['tube_supervised']:
        mask[:] = 0
    return target, mask
