"""Characterise winding_inference rays inside a z slab (base frame).

Each ray: origin + t*step (t in voxels along the ray, anchor at t=192), crossings with
a relative winding level (0 at the anchor). Checks: crossing spacing, level range, and
whether the level increases outward (radius from the umbilicus) or backwards.
"""
import json, sys, glob
import numpy as np

root = sys.argv[1]; zb0, zb1 = float(sys.argv[2]), float(sys.argv[3]); umb = sys.argv[4]
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(umb))['control_points']], float); U = U[np.argsort(U[:, 0])]
O, S, SW, CT, CL, OFF = [], [], [], [], [], []
for sh in sorted(glob.glob(root + '/shard_*')):
    o = np.load(sh + '/ray_origin_zyx.npy'); st = np.load(sh + '/ray_step_zyx.npy')
    off = np.load(sh + '/crossing_offsets.npy'); ct = np.load(sh + '/crossing_t.npy'); cl = np.load(sh + '/crossing_level.npy')
    sw = np.load(sh + '/seed_winding.npy')
    anchor = o + 192 * st
    sel = np.nonzero((anchor[:, 0] >= zb0) & (anchor[:, 0] < zb1))[0]
    for i in sel:
        O.append(o[i]); S.append(st[i]); SW.append(sw[i]); CT.append(ct[off[i]:off[i + 1]]); CL.append(cl[off[i]:off[i + 1]])
print(f'rays with anchor in z [{zb0},{zb1}): {len(O)}; crossings {sum(len(c) for c in CT)}')
print('step norm', np.median([np.linalg.norm(s) for s in S]), ' |step_z| median', np.median([abs(s[0]) for s in S]))
slopes, spac, lev_rng = [], [], []
for o, s, t, l in zip(O, S, CT, CL):
    p = o[None] + t[:, None] * s[None]
    ucy = np.interp(p[:, 0], U[:, 0], U[:, 1]); ucx = np.interp(p[:, 0], U[:, 0], U[:, 2])
    r = np.hypot(p[:, 1] - ucy, p[:, 2] - ucx)
    if len(l) >= 3 and np.ptp(l) > 0:
        slopes.append(np.sign(np.corrcoef(l, r)[0, 1]))
        d = np.diff(t) / np.maximum(np.abs(np.diff(l)), 1)
        spac.extend(d[np.diff(l) != 0])
    lev_rng.append(np.ptp(l))
slopes = np.array(slopes)
print(f'level increases with radius: {np.mean(slopes > 0):.3f}  decreases: {np.mean(slopes < 0):.3f}  (backwards candidates)')
print('crossing spacing per winding (base vx) p10/50/90', np.percentile(spac, [10, 50, 90]).round(1))
print('level span per ray p10/50/90', np.percentile(lev_rng, [10, 50, 90]))
print('seed_winding range', np.min(SW), np.max(SW), ' levels unique sample', np.unique(np.concatenate(CL[:200]))[:20])
steps = np.concatenate([np.diff(l) for l in CL if len(l) > 1])
print('consecutive level steps distribution', {int(k): int(v) for k, v in zip(*np.unique(steps, return_counts=True))})
