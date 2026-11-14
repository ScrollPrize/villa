"""Gather equality: OffloadBrickPool (mapped host RAM, zero-copy kernel) vs ResidentBrickPool (on the card), same sidecar, same ROI, random indices.
usage: SPIRAL_FITTING=<spiral-fitting dir> python t_gather.py <pool sidecar dir> <z_lo> <z_hi>      (needs a CUDA card; prints one RESULT json line)"""
import sys, os, time, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SF = os.environ.get("SPIRAL_FITTING", os.path.dirname(HERE)); sys.path.insert(0, SF); sys.path.insert(0, HERE)
import sparse_cuda_cache as scc, rb_offload
Base = scc.ResidentBrickPool; Off = rb_offload.install(scc)
sc = sys.argv[1]; zr = (int(sys.argv[2]), int(sys.argv[3]))
meta = json.load(open(sc + "/meta.json")); shape = [int(v) for v in meta["array_shape"]]
A = Base(sc, z_roi=zr, device="cuda", label="resident"); B = Off(sc, z_roi=zr, device="cuda", label="offload")
g = torch.Generator(device="cpu").manual_seed(0); res = {}
for n in (1_000, 100_000, 1_000_000, 4_000_000):
    z = torch.randint(zr[0], zr[1], (n,), generator=g); y = torch.randint(0, shape[1], (n,), generator=g); x = torch.randint(0, shape[2], (n,), generator=g)
    idx = torch.stack([z, y, x], 1).cuda()
    a = A.gather(idx); b = B.gather(idx); torch.cuda.synchronize()
    eq = bool(torch.equal(a, b)); nz = int((a > 0).any(1).sum())
    ta = []; tb = []
    for _ in range(5):
        torch.cuda.synchronize(); t = time.perf_counter(); A.gather(idx); torch.cuda.synchronize(); ta.append(time.perf_counter() - t)
        torch.cuda.synchronize(); t = time.perf_counter(); B.gather(idx); torch.cuda.synchronize(); tb.append(time.perf_counter() - t)
    res[n] = dict(identical=eq, nonzero_rows=nz, resident_ms=round(1e3 * float(np.median(ta)), 2), offload_ms=round(1e3 * float(np.median(tb)), 2))
    print(n, res[n], flush=True)
print("RESULT", json.dumps(dict(sidecar=sc, z_roi=zr, channels=A.channels, bricks=A.resident_bricks, pool_gib=round(A.pool_bytes / 2**30, 3), gathers=res)))
