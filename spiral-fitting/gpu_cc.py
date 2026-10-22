"""GPU version of extract_surface_tracks.prepare_cc_labels in PyTorch (CUDA or ROCm).

Labels are numbered like cc3d's: 1..N in the order of each component's first voxel.
"""
import numpy as np
import torch


def _components(fg):
    """(int32 labels, 0 = background; component count) of the 6-connected components of bool tensor ``fg``."""
    shape = fg.shape
    flat = fg.reshape(-1)
    nodes = torch.nonzero(flat, as_tuple=True)[0]  # sorted linear indices
    m = nodes.numel()
    out = torch.zeros(flat.numel(), dtype=torch.int32, device=fg.device)
    if m == 0:
        return out.reshape(shape), 0
    ea, eb = [], []
    strides = [shape[1] * shape[2], shape[2], 1]
    for ax in range(3):
        if shape[ax] < 2:
            continue
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        both = fg[tuple(sl_a)] & fg[tuple(sl_b)]
        idx = torch.nonzero(both)  # coords in the cropped grid == coords of the 'a' voxel
        if idx.numel() == 0:
            continue
        lin = (idx[:, 0] * strides[0] + idx[:, 1] * strides[1] + idx[:, 2]).contiguous()
        ea.append(lin)
        eb.append(lin + strides[ax])
    lab = torch.arange(m, device=fg.device, dtype=torch.int64)
    if ea:
        # Union-find: hook the larger root of every cross-root edge onto the smaller, then
        # pointer-jump; each root ends as its component's smallest voxel index.
        a = torch.searchsorted(nodes, torch.cat(ea))
        b = torch.searchsorted(nodes, torch.cat(eb))
        while True:
            ra, rb = lab[a], lab[b]
            ne = ra != rb
            if not bool(ne.any()):
                break
            ra, rb = ra[ne], rb[ne]
            a, b = a[ne], b[ne]  # edges already inside one root stay inside it
            lo, hi = torch.minimum(ra, rb), torch.maximum(ra, rb)
            lab.scatter_reduce_(0, hi, lo, reduce='amin')
            while True:
                nxt = lab[lab]
                if torch.equal(nxt, lab):
                    break
                lab = nxt
    roots = lab == torch.arange(m, device=fg.device)
    rank = torch.cumsum(roots.to(torch.int32), 0, dtype=torch.int32)  # 1-based, first-voxel order
    out[nodes] = rank[lab]
    return out.reshape(shape), int(rank[-1])


def _maxpool(x, f):
    if f == 1:
        return x
    c = tuple(s - s % f for s in x.shape)
    x = x[:c[0], :c[1], :c[2]]
    out = None
    for i in range(f):
        for j in range(f):
            for k in range(f):
                v = x[i::f, j::f, k::f]
                out = v.clone() if out is None else torch.maximum(out, v)
    return out


def prepare_cc_labels(predictions_binary, downsample_factor, lower, upper):
    """uint32 labels after max-pooling by ``downsample_factor``, keeping sizes in [lower, upper)."""
    fg = torch.from_numpy(np.ascontiguousarray(predictions_binary)).to('cuda') > 0
    lab, n = _components(fg)
    del fg
    pooled = _maxpool(lab, downsample_factor)
    del lab
    if n:
        counts = torch.bincount(pooled.reshape(-1).to(torch.int64), minlength=n + 1)
        keep = (counts >= lower) & (counts < upper)
        keep[0] = False
        pooled = torch.where(keep[pooled.to(torch.int64)], pooled, torch.zeros_like(pooled))
    return pooled.cpu().numpy().astype(np.uint32, copy=False)
