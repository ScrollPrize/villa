"""Discrete sheet graph: count windings as objects, not as an integrated gradient.

Nodes are pieces of surf front-band centrelines (per slab slice). Edges carry an
expected sheet-index difference:
  along  : adjacent pieces of the same centreline            -> 0
  z      : overlapping pieces in neighbouring slab slices     -> 0
  across : first other centreline outward along the normal    -> +1
  wi     : consecutive winding_inference crossings on a ray   -> level difference
Every edge also gets the theta branch-cut term sigma*(dtheta_wrapped - wrap(dtheta))/2pi,
because w (sheet index) jumps by one where a sheet crosses the cut.
Indices are solved robustly (L1 via IRLS) on the GPU.
"""
import glob
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

TWO_PI = 2 * np.pi


def _wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


def cut_term(th_a, th_b, sigma):
    d = th_b - th_a
    return sigma * np.rint((d - _wrap(d)) / TWO_PI)


def build_pieces(front, theta, cell=16):
    """front: (D,H,W) bool band mask. Returns piece id map (D,H,W) int32 (-1 = none) and n pieces."""
    D, H, W = front.shape
    pid = np.full((D, H, W), -1, np.int32)
    yy, xx = np.mgrid[0:H, 0:W]
    near_cut = (np.abs(np.abs(theta) - np.pi) < 0.3)
    key_extra = (near_cut & (theta > 0)).astype(np.int64)
    n = 0
    k8 = np.ones((3, 3), int)
    for z in range(D):
        sk = skeletonize(front[z])
        # break at junctions so components are simple curves
        nb = ndi.convolve(sk.astype(np.uint8), k8, mode='constant') - sk
        sk &= ~(sk & (nb > 2))
        lab, nl = ndi.label(sk, structure=k8)
        ys, xs = np.nonzero(lab)
        key = ((lab[ys, xs].astype(np.int64) * 4096 + (ys // cell) * 512 + (xs // cell)) * 2 + key_extra[ys, xs])
        _, inv = np.unique(key, return_inverse=True)
        pid[z, ys, xs] = inv + n
        n += inv.max() + 1 if len(inv) else 0
    return pid, n


def along_edges(pid, theta, sigma):
    """adjacent skeleton pixels (8-neighbourhood) with different piece ids in the same slice."""
    A, B, Dd = [], [], []
    D, H, W = pid.shape
    P = np.pad(pid, ((0, 0), (1, 1), (1, 1)), constant_values=-1)
    a = P[:, 1:H + 1, 1:W + 1]
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        bb = P[:, 1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
        m = (a >= 0) & (bb >= 0) & (a != bb)
        zz, yy, xx = np.nonzero(m)
        A.append(a[m]); B.append(bb[m])
        Dd.append(cut_term(theta[yy, xx], theta[yy + dy, xx + dx], sigma))
    return np.concatenate(A), np.concatenate(B), np.concatenate(Dd)


def nearest_index_maps(pid):
    """per slice: flat index of the nearest skeleton pixel and its distance."""
    out = []
    for z in range(pid.shape[0]):
        dist, (iy, ix) = ndi.distance_transform_edt(pid[z] < 0, return_indices=True)
        out.append((dist.astype(np.float32), (iy * pid.shape[2] + ix).astype(np.int64)))
    return out


def z_edges(pid, nim, theta, sigma, max_d=2.5):
    A, B, Dd = [], [], []
    D, H, W = pid.shape
    for z in range(D - 1):
        ys, xs = np.nonzero(pid[z] >= 0)
        dist, idx = nim[z + 1]
        ok = dist[ys, xs] <= max_d
        ys, xs = ys[ok], xs[ok]
        j = idx[ys, xs]; yb, xb = j // W, j % W
        A.append(pid[z, ys, xs]); B.append(pid[z + 1, yb, xb])
        Dd.append(cut_term(theta[ys, xs], theta[yb, xb], sigma))
    return np.concatenate(A), np.concatenate(B), np.concatenate(Dd)


def across_edges(pid_t, nx, ny, theta, sigma, max_t=40, stride=2, return_src=False, min_t=2, min_par=0.0):
    """from every `stride`-th skeleton pixel march along +n to the first other piece."""
    dev = pid_t.device
    D, H, W = pid_t.shape
    z, y, x = torch.nonzero(pid_t >= 0, as_tuple=True)
    z, y, x = z[::stride], y[::stride], x[::stride]
    own = pid_t[z, y, x]
    nxs, nys = nx[z, y, x], ny[z, y, x]
    hit = torch.full_like(own, -1); hy = torch.zeros_like(y); hx = torch.zeros_like(x)
    done = torch.zeros_like(own, dtype=torch.bool)
    for t in range(min_t, max_t + 1):
        qy = (y.float() + t * nys).round().long(); qx = (x.float() + t * nxs).round().long()
        inb = (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
        p = torch.full_like(own, -1)
        p[inb] = pid_t[z[inb], qy[inb], qx[inb]]
        new = (~done) & (p >= 0) & (p != own)
        hit[new] = p[new]; hy[new] = qy[new]; hx[new] = qx[new]
        done |= new
    m = done
    if min_par > 0:
        par = (nxs * nx[z, hy, hx] + nys * ny[z, hy, hx]) >= min_par     # hit band must face the same way
        m = m & par
    a = own[m].cpu().numpy(); b = hit[m].cpu().numpy()
    th = torch.from_numpy(theta).to(dev)
    d = 1.0 + cut_term(th[y[m], x[m]].cpu().numpy(), th[hy[m], hx[m]].cpu().numpy(), sigma)
    if return_src:
        return a, b, d, (z[m].cpu().numpy(), y[m].cpu().numpy(), x[m].cpu().numpy())
    return a, b, d


def wi_edges(wi_root, zs, scale, y0, x0, pid, nim, theta, sigma, max_d=2.0, z_tol=None, return_pix=False):
    """winding_inference: consecutive crossings on a ray snapped to their nearest piece."""
    zs = np.asarray(zs); dz = zs[1] - zs[0] if len(zs) > 1 else 1
    z_tol = dz / 2 if z_tol is None else z_tol
    D, H, W = pid.shape
    A, B, Dd = [], [], []
    stats = dict(rays=0, crossings=0, snapped=0)
    PZ, PY, PX = [], [], []
    AN, AV = [], []
    zb0, zb1 = (zs[0] - z_tol) * scale, (zs[-1] + z_tol) * scale
    for sh in sorted(glob.glob(wi_root + '/shard_*')):
        o = np.load(sh + '/ray_origin_zyx.npy'); st = np.load(sh + '/ray_step_zyx.npy')
        off = np.load(sh + '/crossing_offsets.npy'); ct = np.load(sh + '/crossing_t.npy'); cl = np.load(sh + '/crossing_level.npy')
        anc = o + 192 * st
        sel = np.nonzero((anc[:, 0] >= zb0 - 200) & (anc[:, 0] < zb1 + 200))[0]
        if len(sel) == 0:
            continue
        cnt = off[sel + 1] - off[sel]
        rid = np.repeat(sel, cnt)
        ci = np.concatenate([np.arange(off[i], off[i + 1]) for i in sel])
        p = (o[rid] + ct[ci][:, None] * st[rid]) / scale            # working-level zyx
        k = np.rint((p[:, 0] - zs[0]) / dz).astype(int)
        ok = (k >= 0) & (k < D) & (np.abs(p[:, 0] - zs[np.clip(k, 0, D - 1)]) <= z_tol)
        iy = np.rint(p[:, 1]).astype(int) - y0; ix = np.rint(p[:, 2]).astype(int) - x0
        ok &= (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
        stats['rays'] += len(np.unique(rid[ok])); stats['crossings'] += int(ok.sum())
        node = np.full(len(ci), -1, np.int64); sy = np.zeros(len(ci), int); sx = np.zeros(len(ci), int)
        for kk in np.unique(k[ok]):
            mm = ok & (k == kk)
            dist, idx = nim[kk]
            dd = dist[iy[mm], ix[mm]]; j = idx[iy[mm], ix[mm]]
            good = dd <= max_d
            nn = np.where(good, pid[kk, j // W, j % W], -1)
            node[np.nonzero(mm)[0]] = nn; sy[mm] = j // W; sx[mm] = j % W
        stats['snapped'] += int((node >= 0).sum())
        swr = np.load(sh + '/seed_winding.npy')
        AN.append(node[node >= 0]); AV.append((swr[rid] + cl[ci])[node >= 0].astype(np.float64))
        PZ.append(k[node >= 0]); PY.append(sy[node >= 0]); PX.append(sx[node >= 0])
        # consecutive snapped crossings on the same ray
        keep = node >= 0
        r2, n2, l2, y2, x2 = rid[keep], node[keep], cl[ci][keep], sy[keep], sx[keep]
        same = r2[1:] == r2[:-1]
        a, b = n2[:-1][same], n2[1:][same]
        dl = (l2[1:] - l2[:-1])[same].astype(np.float64)
        dct = cut_term(theta[y2[:-1][same], x2[:-1][same]], theta[y2[1:][same], x2[1:][same]], sigma)
        diff = a != b
        A.append(a[diff]); B.append(b[diff]); Dd.append((dl + dct)[diff])
    cat = lambda L, dt: np.concatenate(L) if L else np.zeros(0, dt)     # slabs outside the rays' z range have none
    stats['anchors'] = (cat(AN, np.int64), cat(AV, np.float64))
    if return_pix:
        return cat(A, np.int64), cat(B, np.int64), cat(Dd, np.float64), stats, (cat(PZ, np.int64), cat(PY, np.int64), cat(PX, np.int64))
    return cat(A, np.int64), cat(B, np.int64), cat(Dd, np.float64), stats


def solve_l1(n, a, b, d, wt, prior, prior_w, dev='cuda', irls=15, cg_iters=300, eps=0.05, log=print, ineq=None, ineq_up=0.0):
    """min sum wt*|x_b - x_a - d| + prior_w*|x - prior|^2 via IRLS; CG on the GPU.
    Edges flagged in `ineq` are one-sided: cost wt*max(0, d - (x_b - x_a)) (step at least d)."""
    a = torch.from_numpy(a).long().to(dev); b = torch.from_numpy(b).long().to(dev)
    d = torch.from_numpy(d).float().to(dev); wt = torch.from_numpy(wt).float().to(dev)
    prior = torch.from_numpy(np.asarray(prior)).float().to(dev)
    pw = torch.from_numpy(np.broadcast_to(np.asarray(prior_w, dtype=np.float32), (n,)).copy()).to(dev)
    x = prior.clone()
    iq = torch.zeros_like(d, dtype=torch.bool) if ineq is None else torch.from_numpy(ineq).to(dev)
    for it in range(irls + 1):
        r = x[b] - x[a] - d
        c = wt if it == 0 else wt / torch.clamp(r.abs(), min=eps)
        if it > 0:
            c = torch.where(iq & (r >= 0), c * ineq_up, c)            # above the bound: small slope (fewest windings)
        def A(v):
            f = c * (v[b] - v[a])
            out = pw * v
            out.index_add_(0, b, f); out.index_add_(0, a, -f)
            return out
        rhs = pw * prior
        f = c * d
        rhs.index_add_(0, b, f); rhs.index_add_(0, a, -f)
        # Jacobi-preconditioned CG, warm start
        diag = pw.clone(); diag.index_add_(0, a, c); diag.index_add_(0, b, c)
        rr = rhs - A(x); zz = rr / diag; p = zz.clone(); rz = (rr * zz).sum()
        for _ in range(cg_iters):
            Ap = A(p); al = rz / (p * Ap).sum().clamp(min=1e-20)
            x = x + al * p; rr = rr - al * Ap; zz = rr / diag
            rzn = (rr * zz).sum(); p = zz + (rzn / rz) * p; rz = rzn
        res = (x[b] - x[a] - d)
        res = torch.where(iq, torch.relu(-res), res.abs())
        if it % 5 == 0 or it == irls:
            log(f'  IRLS {it}: weighted L1 {float((wt * res).sum()):.0f}, edges off by >0.5: {float((res > 0.5).float().mean()):.4f}')
    rr = x[b] - x[a] - d
    return x.cpu().numpy(), torch.where(iq, torch.relu(-rr), rr.abs()).cpu().numpy()


def integerize(n, a, b, d, wt, x, res, tol=0.25):
    """Exact integer labels consistent with the (approximately integral) L1 solution x.

    Integer edge differences are propagated along a maximum-weight spanning forest of the
    edges x already satisfies (|residual| < tol). A virtual root joins all trees; each tree
    is then shifted by the integer closest to median(x - k) over its nodes.
    Returns (k, fraction of edges used as candidates).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, connected_components
    ok = (res < tol) & (a != b)
    a, b, d, wt = a[ok], b[ok], np.rint(d[ok]), wt[ok]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    dd = np.where(a == lo, d, -d)                             # k_hi - k_lo
    # strongest edge per node pair
    order = np.argsort(-wt, kind='stable')
    _, first = np.unique(lo[order] * np.int64(n) + hi[order], return_index=True)
    sel = order[first]
    lo, hi, dd, wt = lo[sel], hi[sel], dd[sel], wt[sel]
    key = lo * np.int64(n) + hi
    srt = np.argsort(key); key, lo, hi, dd, wt = key[srt], lo[srt], hi[srt], dd[srt], wt[srt]
    # max-weight forest == MST on 1/(1+w)
    T = minimum_spanning_tree(coo_matrix((1.0 / (1.0 + wt), (lo, hi)), shape=(n + 1, n + 1)).tocsr())
    ti, tj = T.nonzero()
    # virtual root n linked to one node per component of the forest
    ncomp, lab = connected_components(T, directed=False)
    reps = np.unique(lab[:n], return_index=True)[1]
    R = coo_matrix((np.ones(len(ti) + len(reps)), (np.r_[ti, np.full(len(reps), n)], np.r_[tj, reps])), shape=(n + 1, n + 1)).tocsr()
    bfs, pred = breadth_first_order(R, n, directed=False, return_predecessors=True)
    # step(v) = k_v - k_pred(v); 0 for children of the virtual root
    v = np.arange(n); pv = pred[:n]
    real = pv >= 0
    real &= pv < n
    step = np.zeros(n)
    plo, phi = np.minimum(pv[real], v[real]), np.maximum(pv[real], v[real])
    pos = np.searchsorted(key, plo * np.int64(n) + phi)
    found = (pos < len(key)) & (key[np.minimum(pos, len(key) - 1)] == plo * np.int64(n) + phi)
    assert found.all(), 'tree edge missing from edge table'
    sgn = np.where(v[real] == phi, 1.0, -1.0)
    step[real] = sgn * dd[pos]
    # pointer jumping (list ranking): k_v = sum of steps on the path up to the virtual root
    val = step.copy(); anc = np.where(real, pv, -1)
    while (anc >= 0).any():
        m = anc >= 0
        nv = val.copy(); nv[m] = val[m] + val[anc[m]]
        na = np.full_like(anc, -1); na[m] = anc[anc[m]]
        val, anc = nv, na
    k = val
    # per tree: integer shift so labels stay closest to x
    tree = lab[:n]
    diff = x - k
    o = np.argsort(tree, kind='stable')
    bounds = np.searchsorted(tree[o], np.arange(ncomp + 1))
    med = np.array([np.median(diff[o[bounds[c]:bounds[c + 1]]]) if bounds[c + 1] > bounds[c] else 0.0 for c in range(ncomp)])
    shift = np.rint(med)
    k = k + shift[tree]
    return k, ok.mean()


def back_targets(pid_t, labels_t, nx, ny, sel, max_t=20):
    """For voxels in `sel` (e.g. back-layer V fibers): label of the nearest front centreline
    inward (-n) and outward (+n). Returns (z, y, x, k_in) for voxels whose outward neighbour
    is exactly k_in + 1, i.e. that lie between two consecutive fronts."""
    D, H, W = pid_t.shape
    z, y, x = torch.nonzero(sel, as_tuple=True)
    nxs, nys = nx[z, y, x], ny[z, y, x]
    def march(sign):
        k = torch.full(z.shape, float('nan'), device=z.device)
        for t in range(1, max_t + 1):
            qy = (y.float() + sign * t * nys).round().long().clamp(0, H - 1)
            qx = (x.float() + sign * t * nxs).round().long().clamp(0, W - 1)
            p = pid_t[z, qy, qx]
            new = torch.isnan(k) & (p >= 0)
            k[new] = labels_t[p[new]]
        return k
    kin, kout = march(-1.0), march(1.0)
    ok = (kout - kin - 1.0).abs() < 1e-3
    return z[ok], y[ok], x[ok], kin[ok]
