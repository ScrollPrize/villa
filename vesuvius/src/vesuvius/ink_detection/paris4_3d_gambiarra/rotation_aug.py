"""Teacher stage 4 views: each cube plan view composed with a uniform random rotation and zoom, trilinearly resampled
from the anchor's parent256, then photometrically augmented under a realism cap.

Geometry. For plan transform t (v = A (q - cin) + 63.5) and a uniform rotation R (Shoemake quaternion), the view
is q = M (v - 63.5) + center with M = zoom * A^T R^T, and center = cin clamped per axis to [h, 255 - h],
h = 63.5 * sum_k |M_ik|, so the rotated cube always lies inside the parent256. R is redrawn (at most 64 times)
until the anchor lies in [3.5, 123.5]; there is no identity fallback. Labels are every training point and up to 64
weak sites per class whose rotated position lies in [3.5, 123.5].

The view is resampled from the normalized parent256 and augmented with the Vesuvius transforms in views. Photometric,
per view, in this order (float32): x * U(.6, 1.4) (p .8); x + U(-.15, .15) (p .8); the plan's contrast about the mean;
clamp [0, 1]; the plan's gamma; Gaussian blur sigma U(.4, 1) (p .5, replicate padding); noise N(0, s^2),
s U(.01, .06) (p .2); clamp [0, 1]. Realism cap: compared with the rotated view before photometrics,
the saturated fraction (== 0 or == 1) may grow by at most .10, the mean move by at most .15, and the standard
deviation fall to no less than half; otherwise all photometric parameters are redrawn (at most 50 attempts).

Every draw of view j at update n comes from default_rng([0x5A07, 27, n, j]) (redraws: [0x5A07, 27, n, j, attempt]),
independent of the plan stream, so a resumed run replays the same views.
"""
import math

import numpy as np
import torch

from . import geometry as g
from . import views as views_
from .sampling import select_weak

KEY = 0x5A07
SEED = 27
C = 63.5
LO, HI = 3.5, 123.5
MAX_ATTEMPTS = 64
CAP = dict(saturation_excess=.10, mean_shift=.15, std_ratio=.5, max_attempts=50)
SCALE, SHIFT, BLUR, NOISE = (.8, .6, 1.4), (.8, -.15, .15), (.5, .4, 1.), (.2, .01, .06)  # (p, low, high)
ZOOM = (.5, 1 / 1.15, 1.15)  # p, log-uniform range


class Rejected(RuntimeError):
    """No rotation (or photometric draw) accepted within its attempt budget."""


def uniform_rotation(rng):
    u1, u2, u3 = rng.random(3)
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    x, y = a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2)
    z, w = b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame(t, R, zoom=1.):
    A = np.asarray(t['forward'], np.float64)
    cin = np.asarray(t['input_center'], np.float64)
    R = np.asarray(R, np.float64)
    M = float(zoom) * (A.T @ R.T)
    half = C * np.abs(M).sum(1)
    assert np.all(half <= 127.5), 'zoomed cube exceeds the parent256'
    center = np.clip(cin, half, 255 - half)
    return dict(R=R, M=M, Minv=np.linalg.inv(M), A=A, zoom=float(zoom), center=center, cin=cin)


def to_parent(v, f):
    return (np.asarray(v, np.float64) - C) @ f['M'].T + f['center']


def to_view(v, f):
    """Plan-view coordinates (geometry.points) -> this view's coordinates, through parent coordinates."""
    q = (np.asarray(v, np.float64) - C) @ f['A'] + f['cin']
    return (q - f['center']) @ f['Minv'].T + C


def _coin_value(rng, spec, off):
    p, low, high = spec
    on = bool(rng.random() < p)
    value = float(rng.uniform(low, high))  # drawn whether or not the coin succeeds (fixed-length stream)
    return value if on else off


def photo_params(rng):
    return dict(scale=_coin_value(rng, SCALE, 1.), shift=_coin_value(rng, SHIFT, 0.), blur=_coin_value(rng, BLUR, 0.),
                noise=_coin_value(rng, NOISE, 0.), cuda_seed=int(rng.integers(1 << 62)))


def zoom_param(rng):
    p, low, high = ZOOM
    on = bool(rng.random() < p)
    value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return value if on else 1.


def redraw_params(seed, step, j, attempt):
    """Photometric attempt >= 2: every photometric parameter redrawn (contrast / gamma as in the cube plans)."""
    assert attempt >= 2
    rng = np.random.default_rng([KEY, int(seed), int(step), int(j), int(attempt)])
    photo = photo_params(rng)
    contrast = float(rng.uniform(.65, 1.5))
    gamma = float(np.exp(rng.uniform(np.log(.7), np.log(1.5))))
    if rng.random() >= .8:
        contrast = 1.
    if rng.random() >= .8:
        gamma = 1.
    return dict(photo, contrast=contrast, gamma=gamma)


def draw(seed, step, j, t, anchor_view):
    rng = np.random.default_rng([KEY, int(seed), int(step), int(j)])
    photo = photo_params(rng)
    zoom = zoom_param(rng)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        f = frame(t, uniform_rotation(rng), zoom)
        a = to_view(anchor_view, f)
        if np.all((a >= LO) & (a <= HI)):
            return f, attempt, photo
    raise Rejected(f'update {step} view {j}: no rotation accepted in {MAX_ATTEMPTS} attempts')


def _inside(v):
    return np.all((v >= LO) & (v <= HI), axis=1)


def geometry(data, plan, step, seed=SEED):
    """Per view of a cube plan (CPU): frame, photometric draws and the labels in the rotated crop, plus the label
    accounting against the unrotated plan view. `data` is a ClickData with weak sites."""
    out = []
    for j, item in enumerate(plan['plans']):
        i = item['anchor']
        parent = data.parents[i]
        t = g.construct(item['offset'], item['permutation'], item['flips'], [0, 0, 0])
        anchor_view = g.points((data.q[i] - parent)[None], t)[0]
        f, attempts, photo = draw(seed, step, j, t, anchor_view)
        photo = dict(photo, contrast=float(item['photo']['contrast']), gamma=float(item['photo']['gamma']))
        corners = to_parent(g.CORNERS, f)
        assert corners.min() >= -1e-6 and corners.max() <= 255 + 1e-6, 'rotated crop leaves the parent256'
        lo = np.clip(np.floor(corners.min(0)), 0, 255).astype(np.int64) + parent
        hi = np.clip(np.ceil(corners.max(0)), 0, 255).astype(np.int64) + parent
        assert not g.intersects(lo, hi, data.eval_parents), 'rotated crop touches a held-out parent'
        # training points of the context in the rotated crop
        cand = data.train[data.tree.query_ball_point(parent + 127.5, 159.5, p=np.inf)]
        plan_v = g.points(data.q[cand] - parent, t)
        v = to_view(plan_v, f)
        keep = _inside(v)
        ids, coords = cand[keep].copy(), v[keep].astype(np.float32)
        assert i in ids and not set(ids.tolist()) & set(data.heldout.tolist())
        plan_ids = cand[_inside(plan_v)]
        # weak sites in the rotated crop, then the per-view subsample (seeded by the plan item)
        wc = np.asarray(data.weak_tree.query_ball_point(parent + 127.5, 159.5, p=np.inf), np.int64)
        wplan = g.points(data.weak_q[wc] - parent, t)
        w = to_view(wplan, f)
        wk = _inside(w)
        ii, qq = wc[wk], w[wk].astype(np.float32)
        selected = select_weak(item, ii, data.weak_y)
        ii, qq = ii[selected], qq[selected]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(f['R']) - 1) / 2, -1, 1))))
        out.append(dict(frame=f, attempts=attempts, photo=photo, angle=angle, key=(int(seed), int(step), j),
                        ids=ids, coordinates=coords, weak_ids=ii, weak_coordinates=qq, weak_labels=data.weak_y[ii],
                        labels=len(ids), plan_labels=len(plan_ids), weak=len(ii), tube_plan=int(_inside(wplan).sum()),
                        tube_rotated=int(wk.sum()), zoom=f['zoom']))
    return out


@torch.no_grad()
def photometric(v, q):
    """The photometric chain on a copy of one rotated view v [1, 1, D, H, W], through the Vesuvius transforms in views."""
    x = views_.contrast_gamma(views_.scale_shift(v[0].clone(), q['scale'], q['shift']), q['contrast'], q['gamma'])
    if q['blur']:
        x = views_.blur(x, q['blur'], 'replicate')
    if q['noise']:
        x = views_.add_noise(x, q['noise'], q['cuda_seed'])
    return x.clamp_(0, 1)[None]


def _stats(v):
    return torch.stack([(v <= 0).float().mean(), (v >= 1).float().mean(), v.mean(), v.std()])


def check(clean, aug):
    sat_excess = (aug[0] + aug[1]) - (clean[0] + clean[1])
    shift = abs(aug[2] - clean[2])
    ratio = aug[3] / max(clean[3], 1e-12)
    return bool(sat_excess <= CAP['saturation_excess'] and shift <= CAP['mean_shift'] and ratio >= CAP['std_ratio'])


@torch.no_grad()
def augment(v, photo, key):
    """(augmented view, accepted photo dict, attempts)."""
    clean = _stats(v).tolist()
    for attempt in range(1, CAP['max_attempts'] + 1):
        if attempt > 1:
            photo = redraw_params(*key, attempt)
        x = photometric(v, photo)
        if check(clean, _stats(x).tolist()):
            return x, photo, attempt
        del x
    raise Rejected(f'view {key}: no photometric draw passed the realism cap in {CAP["max_attempts"]} attempts')


@torch.no_grad()
def views(parents, norm, geo, device='cuda'):
    """parents: uint8 [16, 256, 256, 256] raw anchor parents -> (float32 [16, 1, 128^3], photo attempts per view)."""
    raw = torch.as_tensor(parents).to(device, non_blocking=True)
    xs, attempts = [], []
    for j, q in enumerate(geo):
        f = q['frame']
        corners = to_parent(g.CORNERS, f)
        assert corners.min() >= -1e-6 and corners.max() <= 255 + 1e-6, 'rotated crop leaves the parent256'
        v = views_.resample(views_.normalize(raw[j][None, None].float(), [norm[j]])[0], f['M'], f['center'])
        x, _, n = augment(v[None], q['photo'], q['key'])
        xs.append(x)
        attempts.append(n)
    return torch.cat(xs), attempts
