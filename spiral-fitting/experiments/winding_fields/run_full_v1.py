"""Full-scroll driver: solve overlapping level-3 slabs with the best sheet-graph config.

Resumable: a slab whose w.npz exists is skipped. Each slab's centre is the umbilicus
at the slab's mid z. Every slab still writes its renders and evals (solve3d report).
"""
import json, os, subprocess, sys, time
import numpy as np

W = '/ephemeral/sean/experiments/winding_fields'
PY = '/ephemeral/sean/villa/spiral-fitting/.venv/bin/python'
# limited to the z band covered by winding_inference rays (level 3; base 7500-17500)
Z_LO, Z_HI, SLAB, STEP, DZ = 3750, 8750, 150, 135, 3
CFG = ('--init none --orient aniso --signal surf --normals las --sigma -1 --level 3 --dz 3 --sheetgraph '
       '--sg_aggregate --sg_prior 1e-6 --sg_drop_back 2 --sg_across_ineq --sg_ineq_up 0.2 --sg_w_across 1 '
       '--sg_across_min_t 3 --sg_across_min_par 0.8 --sg_across_fallback 40 --sg_w_wi 5 --sg_w_z 0.3 '
       '--wi_root data/winding_inference --layer_iters 0 --sg_round_local 20 --sg_round_int --sg_fill_back 5 --no_cache')
U = np.array([[p['z'], p['y'], p['x']] for p in json.load(open(f'{W}/data/umbilicus.json'))['control_points']], float)
U = U[np.argsort(U[:, 0])]
starts = list(range(Z_LO, Z_HI - SLAB // 2, STEP))
env = dict(os.environ, PYTHONPATH=f'{W}/pydeps', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '6'))
only = set(int(a) for a in sys.argv[1:])
print(f'{len(starts)} slabs', flush=True)
for i, z0 in enumerate(starts):
    if only and z0 not in only:
        continue
    z1 = min(z0 + SLAB, Z_HI)
    d = f'{W}/runs/full/slab_{z0:05d}'
    if os.path.exists(f'{d}/w.npz'):
        continue
    os.makedirs(d, exist_ok=True)
    zb = (z0 + z1) / 2 * 2.0                                   # base frame
    cx, cy = np.interp(zb, U[:, 0], U[:, 2]) / 2.0, np.interp(zb, U[:, 0], U[:, 1]) / 2.0
    cmd = f'{PY} -u code/solve3d.py {CFG} --z0 {z0} --z1 {z1} --center {cx:.1f} {cy:.1f} --out {d}/w.npz'
    t = time.time()
    with open(f'{d}/log.txt', 'w') as log:
        r = subprocess.run(cmd.split(), cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f'slab {i + 1}/{len(starts)} z {z0}-{z1} exit {r.returncode} {time.time() - t:.0f}s', flush=True)
print('FULL_DONE', flush=True)
