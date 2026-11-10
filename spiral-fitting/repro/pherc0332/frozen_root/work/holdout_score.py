"""Hold-out scores of a run: satisfied area over the held-out graph patches (from the run's satisfied_fitted.json) and
agreement of the held-out anchor fibers with the fit (|W_fit - W_anchor| < 0.5 after removing the global offset).

    python holdout_score.py RUN_DIR      -> prints "HOLDOUT patches <frac> anchors <frac>"
"""
import os
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.environ.get('SPIRAL_ROOT', '/home/djosey/Documents/villa/spiral-fitting'))
import flatten_spiral_checkpoint as fsc, sample_spiral
from checkpoint_io import load_checkpoint_cpu

D = Path('/media/djosey/nvme4/spiral_0332')
R = Path(sys.argv[1])
hold = json.load(open(os.environ.get('HOLDOUT_JSON', str(D / 'work/holdout.json'))))
sat = {p['id']: p for p in json.load(open(R / 'satisfied_fitted.json'))['patches']}
hp = [sat[d] for d in hold['held_patches'] if d in sat]
patch_frac = sum(p['satisfied_area'] for p in hp) / max(sum(p['total_area'] for p in hp), 1)
ck = load_checkpoint_cpu(R / 'checkpoint_fitted.ckpt'); cfg = fsc._checkpoint_config(ck)
m = fsc._build_model(ck, cfg, fsc.Path(str(D / 'dataset/umbilicus.json')), torch.device('cpu')); t = m.get_slice_to_spiral_transform(); dr = m.get_dr_per_winding().detach()
doc = json.load(open(D / 'work/abs_winding_holdout.json'))
P = []; Wa = []
for c in doc['collections'].values():
    for q in c['points'].values():
        if ck['z_begin'] <= q['p'][2] < ck['z_end']:
            P.append(q['p'][::-1]); Wa.append(q['wind_a'])
P = torch.tensor(np.array(P, dtype=np.float32)); Wa = np.array(Wa)
with torch.inference_mode():
    sp = torch.cat([t(P[i:i + 20000]) for i in range(0, len(P), 20000)])
Wf = (sample_spiral.get_theta_and_radii(sp[:, 1:], dr)[2] / dr).numpy(); dev = Wf - Wa; dev -= np.median(dev)
print(f'HOLDOUT patches {patch_frac:.3f} anchors {np.mean(np.abs(dev) < 0.5):.3f} (n={len(hp)} patches, {len(Wa)} anchor points)')
