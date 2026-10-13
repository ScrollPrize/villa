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
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("only", nargs="*", type=int)
ap.add_argument("--mode", default="solve", choices=["solve", "graph", "final", "tracks"])   # graph: save slab graph only; final: fill from joint labels
ap.add_argument("--root", default="runs/full")
ap.add_argument("--part", default="0/1", help="i/n: this process takes every n-th slab starting at i (ordered by --first)")
ap.add_argument("--first", nargs="*", type=int, default=[], help="slab z0s to run before the rest")
ap.add_argument("--targets", default="runs/tracks_full/targets", help="tracks mode: track_targets.py output")
ap.add_argument("--bc_root", default="", help="tracks mode: fix shared slices to the neighbouring slabs' w.npz in this run dir")
ap.add_argument("--pad", type=int, default=0, help="extra level-3 slices of z context each side (multiple of dz); only the slab's own slices are saved")
a = ap.parse_args()
only = set(a.only)
pi_, pn_ = (int(v) for v in a.part.split("/"))
order = [z for z in a.first if z in starts] + [z for z in starts if z not in a.first]
if only:
    order = [z for z in order if z in only]               # split the requested slabs, not all slabs
mine = set(order[pi_::pn_])
print(f'{len(starts)} slabs', flush=True)
for z0 in order:
    i = starts.index(z0)
    if (only and z0 not in only) or z0 not in mine:
        continue
    z1 = min(z0 + SLAB, Z_HI)
    d = f'{W}/{a.root}/slab_{z0:05d}'
    done = f'{d}/graph.npz' if a.mode == 'graph' else f'{d}/w.npz'
    if os.path.exists(done):
        continue
    extra = {'solve': '', 'graph': f' --save_graph {d}/graph.npz --graph_only', 'final': f' --labels_from {d}/labels.npz',
             'tracks': f' --track_targets {W}/{a.targets}'}[a.mode]
    os.makedirs(d, exist_ok=True)
    zb = (z0 + z1) / 2 * 2.0                                   # base frame
    cx, cy = np.interp(zb, U[:, 0], U[:, 2]) / 2.0, np.interp(zb, U[:, 0], U[:, 1]) / 2.0
    if a.bc_root:
        nbs = [f'{W}/{a.bc_root}/slab_{starts[j]:05d}/w.npz' for j in (i - 1, i + 1) if 0 <= j < len(starts)]
        extra += ' --bc_from ' + ' '.join(n for n in nbs if os.path.exists(n))
    cfg = CFG.replace('--sheetgraph ', '').replace('--orient aniso', '--orient none') if a.mode == 'tracks' else CFG     # tracks: no surf sheet graph, fill from track labels
    zp0, zp1 = max(Z_LO, z0 - a.pad), min(Z_HI, z1 + a.pad)
    if a.pad:
        extra += f' --save_z0 {z0} --save_z1 {z1}'
    cmd = f'{PY} -u code/solve3d.py {cfg} --z0 {zp0} --z1 {zp1} --center {cx:.1f} {cy:.1f} --out {d}/w.npz' + extra
    t = time.time()
    with open(f'{d}/log_{a.mode}.txt', 'w') as log:
        r = subprocess.run(cmd.split(), cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f'{a.mode} slab {i + 1}/{len(starts)} z {z0}-{z1} exit {r.returncode} {time.time() - t:.0f}s', flush=True)
print(f'{a.mode.upper()}_DONE', flush=True)
