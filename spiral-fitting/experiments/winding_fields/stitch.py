"""Stitch full-scroll slabs: integer offset per slab from the overlap, absolute numbering
from winding_inference (seed_winding + crossing level), written to offsets.json.

Each slab field w is a sheet index with an arbitrary integer gauge. Neighbouring slabs
share solved slices; the offset of slab B relative to A is round(median(w_A - w_B)) over
voxels valid in both. The absolute offset is round(median(abs_winding - w)) over
winding_inference crossings that land on a solved slice (input data, not an eval set).
"""
import glob, json, os, sys
import numpy as np

W = '/ephemeral/sean/experiments/winding_fields'
root = sys.argv[1] if len(sys.argv) > 1 else f'{W}/runs/full'
slabs = sorted(glob.glob(f'{root}/slab_*/w.npz'))
print(f'{len(slabs)} slabs')


def load(p):
    d = np.load(p)
    return {k: d[k] for k in ('w', 'mask', 'zs', 'crop', 'cx', 'cy', 'sigma')}


def overlap_offset(A, B):
    common = np.intersect1d(A['zs'], B['zs'])
    if len(common) == 0:
        return None, 0.0, 0
    ya0, ya1, xa0, xa1 = A['crop']; yb0, yb1, xb0, xb1 = B['crop']
    y0, y1, x0, x1 = max(ya0, yb0), min(ya1, yb1), max(xa0, xb0), min(xa1, xb1)
    diffs = []
    for z in common:
        ia = int(np.nonzero(A['zs'] == z)[0][0]); ib = int(np.nonzero(B['zs'] == z)[0][0])
        wa = A['w'][ia, y0 - ya0:y1 - ya0, x0 - xa0:x1 - xa0]; wb = B['w'][ib, y0 - yb0:y1 - yb0, x0 - xb0:x1 - xb0]
        ma = A['mask'][ia, y0 - ya0:y1 - ya0, x0 - xa0:x1 - xa0] & B['mask'][ib, y0 - yb0:y1 - yb0, x0 - xb0:x1 - xb0]
        diffs.append((wa - wb)[ma][::7])
    dd = np.concatenate(diffs)
    off = float(np.rint(np.median(dd)))
    agree = float(np.mean(np.abs(dd - off) < 0.5))
    return off, agree, len(common)


offsets = [0.0]
quality = []
prev = load(slabs[0])
for p in slabs[1:]:
    cur = load(p)
    off, agree, n = overlap_offset(prev, cur)
    if off is None:
        print(f'  {os.path.basename(os.path.dirname(p))}: NO OVERLAP -- offset carried as 0')
        off = 0.0
    offsets.append(offsets[-1] + off)
    quality.append(agree)
    print(f'  {os.path.basename(os.path.dirname(p))}: relative offset {off:+.0f}, overlap agreement {agree:.3f} over {n} slices')
    prev = cur

# absolute numbering from winding_inference crossings that sit on solved slices
absd = []
NO_WI = os.environ.get('NO_WI') == '1'                     # no winding_inference: keep the graph's own numbering
for si, p in (enumerate([]) if NO_WI else enumerate(slabs)):
    d = load(p)
    zs, (y0, y1, x0, x1) = d['zs'], d['crop']
    for sh in sorted(glob.glob(f'{W}/data/winding_inference/shard_*')):
        o = np.load(sh + '/ray_origin_zyx.npy'); st = np.load(sh + '/ray_step_zyx.npy')
        off_ = np.load(sh + '/crossing_offsets.npy'); ct = np.load(sh + '/crossing_t.npy'); cl = np.load(sh + '/crossing_level.npy')
        sw = np.load(sh + '/seed_winding.npy')
        anc = (o + 192 * st) / 2.0
        sel = np.nonzero((anc[:, 0] >= zs[0]) & (anc[:, 0] <= zs[-1]))[0][::20]
        for i in sel:
            t = ct[off_[i]:off_[i + 1]]; lv = cl[off_[i]:off_[i + 1]]
            q = (o[i][None] + t[:, None] * st[i][None]) / 2.0
            k = np.rint((q[:, 0] - zs[0]) / (zs[1] - zs[0])).astype(int)
            on = (k >= 0) & (k < len(zs)) & (np.abs(q[:, 0] - zs[np.clip(k, 0, len(zs) - 1)]) <= 0.5)
            iy = np.rint(q[:, 1]).astype(int) - y0; ix = np.rint(q[:, 2]).astype(int) - x0
            on &= (iy >= 0) & (iy < d['w'].shape[1]) & (ix >= 0) & (ix < d['w'].shape[2])
            if not on.any():
                continue
            on[on] &= d['mask'][k[on], iy[on], ix[on]]
            wv = d['w'][k[on], iy[on], ix[on]] + offsets[si]
            absd.append((sw[i] + lv[on]) - wv)
absd = np.concatenate(absd) if absd else np.zeros(0)
abs_off = float(np.rint(np.median(absd))) if len(absd) else 0.0
agree_abs = float(np.mean(np.abs(absd - abs_off) < 0.5)) if len(absd) else 0.0
print('NO_WI: absolute offset 0 (graph numbering)' if NO_WI else '', end='')
print(f'absolute offset {abs_off:+.0f} from {len(absd)} winding_inference crossings; '
      f'{agree_abs:.3f} of them agree within 0.5 after the shift')
out = {'slabs': [os.path.dirname(p) for p in slabs], 'relative_offsets': offsets, 'overlap_agreement': quality,
       'absolute_offset': abs_off, 'absolute_agreement': agree_abs,
       'note': 'absolute winding = slab w + relative_offsets[i] + absolute_offset; numbering from winding_inference seed_winding+level (input), not from the eval mesh'}
json.dump(out, open(f'{root}/offsets.json', 'w'), indent=1)
print('wrote', f'{root}/offsets.json')
