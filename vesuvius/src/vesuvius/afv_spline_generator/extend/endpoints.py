"""Frozen endpoint tangent estimator from the validated slab merge-gap publisher."""
import numpy as np

def arc(p):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def interpolate(p, s, t):
    return np.column_stack([np.interp(t, s, p[:, k]) for k in range(3)])


def endpoints(curves):
    tips, tangents, reliable = [], [], []
    # Compare the direction at three scales; very short/noisy tips abstain.
    for p in curves:
        for q in (p, p[::-1]):
            s = arc(q)
            inside = interpolate(q, s, np.minimum([4., 12., 24.], s[-1]))
            v = q[0] - inside
            sizes = np.linalg.norm(v, axis=1)
            v /= np.maximum(sizes[:, None], 1e-9)
            ok = (s[-1] >= 16 and sizes[1] >= 10 and
                  np.dot(v[0], v[1]) >= np.cos(np.deg2rad(25)) and
                  np.dot(v[1], v[2]) >= np.cos(np.deg2rad(15)))
            tips.append(q[0]); tangents.append(v[1]); reliable.append(ok)
    return np.asarray(tips), np.asarray(tangents), np.asarray(reliable)


