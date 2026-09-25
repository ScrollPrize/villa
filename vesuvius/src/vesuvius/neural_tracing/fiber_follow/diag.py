"""Diagnostic images written during training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.evaluate import score_trace, summarize
from vesuvius.neural_tracing.fiber_follow.geometry import frame_from_heading, interp_at


def _to_index(pts, crop):
    """Local metric (a=u, b=v, c=f) -> crop sample indices (col_u, col_v, row)."""
    mid = (crop.width - 1) / 2.0
    return np.stack([pts[..., 0] / crop.spacing + mid, pts[..., 1] / crop.spacing + mid,
                     pts[..., 2] / crop.spacing + crop.behind], -1)


def _curved_slab(vol, centers, axis, half=2):
    """vol (D, H=v, W=u); per depth row take max over +-half samples around
    ``centers[row]`` along ``axis`` (1 = v, 2 = u)."""
    D, H, W = vol.shape
    n = H if axis == 1 else W
    out = np.zeros((D, W if axis == 1 else H), vol.dtype)
    for d in range(D):
        c = int(round(np.clip(centers[d], 0, n - 1)))
        lo, hi = max(0, c - half), min(n, c + half + 1)
        out[d] = vol[d, lo:hi, :].max(0) if axis == 1 else vol[d, :, lo:hi].max(1)
    return out


def plot_batch(x, pred, fut, fmask, crop, path, observed, hmask, history_gt, history_mask,
               n=6, source=None, offtrack=None, confidence=None):
    """Crops exactly as the model sees them (sample-index space, forward = up).

    Each panel is a thin curved slab (+-2 samples) that follows the GT fiber
    in the hidden lateral axis, so the fiber being traced stays visible.
    Gray = first image channel (CT or presence), red = observed history,
    green = GT, orange = proposal. The dotted orange segment is the
    actual tracer's first step from the current point.
    Bounds stay fixed to the actual crop even when GT leaves it. The orange
    path is the complete denoised curve before the confidence-gated commit.
    """
    source = source[:n].detach().cpu().numpy() if source is not None else None
    offtrack = offtrack[:n].detach().cpu().numpy() if offtrack is not None else None
    confidence = confidence[:n].detach().float().cpu().numpy() if confidence is not None else None
    observed = observed[:n].detach().float().cpu().numpy()
    hmask = hmask[:n].detach().float().cpu().numpy()
    history_gt = history_gt[:n].detach().float().cpu().numpy()
    history_mask = history_mask[:n].detach().float().cpu().numpy()
    x = x[:n].float().cpu().numpy()
    pred = pred[:n].float().detach().cpu().numpy()
    fut = fut[:n].cpu().numpy()
    fmask = fmask[:n].cpu().numpy()
    n = len(x)
    D = crop.depth
    fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 8.0), squeeze=False)
    rows = np.arange(D)
    fc = crop.forward_coords
    yt = [r for r in range(D) if fc[r] % 10 == 0]
    for i in range(n):
        m = fmask[i] > 0
        gi = _to_index(fut[i][m], crop)
        pi = _to_index(pred[i], crop)
        hi = _to_index(observed[i], crop)
        hgi = _to_index(history_gt[i], crop)
        mid = (crop.width - 1) / 2.0
        # GT lateral position per row (current point at the centre), held beyond the last future point
        rr = np.concatenate([[crop.behind], gi[:, 2]])
        for r, (axis, comp) in enumerate(((1, 0), (2, 1))):
            other = 1 - comp
            cen = np.interp(rows, rr, np.concatenate([[mid], gi[:, other]]))
            pres = _curved_slab(x[i, 0], cen, axis)
            hist = _curved_slab(x[i, -1], cen, axis)
            alpha = .8*hist[..., None]
            rgb = ((1-alpha)*pres[..., None] + alpha*np.array([1., .1, .1])).clip(0, 1)
            a = ax[r, i]
            a.imshow(rgb, origin="lower", aspect="auto")
            a.plot(gi[:, comp], gi[:, 2], "o--", color="lime", ms=3, lw=1)
            line, = a.plot(pi[:, comp], pi[:, 2], "x-", color="orange", ms=4, lw=1.5)
            line.set_path_effects([pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()])
            a.plot([mid, pi[0, comp]], [crop.behind, pi[0, 2]], ':', color='orange', lw=1.2)
            supplied = hmask[i] > 0
            a.plot(hi[supplied, comp][::-1], hi[supplied, 2][::-1], 'o-', color='red', ms=2.5, lw=1)
            valid = history_mask[i] > 0
            a.plot(hgi[valid, comp], hgi[valid, 2], '.', color='lime', ms=3)
            a.axhline(crop.behind, color='cyan', ls=':', lw=.8)
            a.plot(mid, crop.behind, 'D', color='cyan', ms=4)
            outside = (gi[:, :2] < 0).any(-1) | (gi[:, :2] > crop.width-1).any(-1)
            a.plot(np.clip(gi[outside, comp], 0, crop.width-1), gi[outside, 2], 's', color='magenta', ms=4)
            a.set_xlim(-.5, crop.width-.5)
            a.set_ylim(-.5, crop.depth-.5)
            a.set_xticks([])
            a.set_yticks(yt)
            a.set_yticklabels([f"{fc[t]:.0f}" for t in yt] if i == 0 else [], fontsize=7)
            if i == 0:
                a.set_ylabel("forward (voxels)  " + ("u panel" if r == 0 else "v panel"))
        err = np.linalg.norm(pred[i] - fut[i], axis=1)[m]
        title = f"err {err.mean():.2f}" if len(err) else "no fut"
        if outside.any():
            title += f"; {outside.sum()} GT outside"
        if offtrack is not None and offtrack[i]:
            title = 'OFF TRACK: reject continuation'
        if source is not None:
            # Codes follow data.REPLAY_SOURCES; unknown codes still get a label.
            names = {0: 'fresh', 1: 'fixed recovery', 2: 'recent replay'}
            title = names.get(int(source[i]), f'source {int(source[i])}') + '\n' + title
        if confidence is not None:
            title += f'\nnext-step confidence {confidence[i, 0]:.2f}'
        ax[0, i].set_title(title, fontsize=8)
    fig.suptitle("red observed history | green GT | orange full proposal\n"
                 "cyan: actual current point | dotted orange: actual first step", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _gt_frames(points):
    t = np.gradient(points, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    frames = []
    u = None
    for ti in t:
        fr = frame_from_heading(ti, u)
        u = fr[:, 0]
        frames.append(fr)
    return np.stack(frames)


def rollout_diag(tracer, fibers, seeds, path, max_len=400.0, half=15, batch=8):
    """Trace held-out seeds and show each in a straightened view along its GT
    fiber: arc length up, lateral offset across (u and v panels), presence
    background, GT = centre line (green), trace = orange."""
    old = tracer.p.max_len
    tracer.p.max_len = max_len
    try:
        paths, reasons = [], []
        for start in range(0, len(seeds), batch):
            chunk = seeds[start:start+batch]
            p, r = tracer.trace(np.stack([s["pos"] for s in chunk]), np.stack([s["heading"] for s in chunk]))
            paths.extend(p)
            reasons.extend(r)
    finally:
        tracer.p.max_len = old
    rows = []
    n = len(seeds)
    per_row = min(n, 8)
    nr = int(np.ceil(n / per_row))
    fig, ax = plt.subplots(nr, 2 * per_row, figsize=(1.1 * 2 * per_row, 7.5 * nr), squeeze=False)
    lat = np.arange(-half, half + 1, dtype=np.float64)
    vol = tracer.vol
    for k, (s, p, r) in enumerate(zip(seeds, paths, reasons)):
        f = fibers[s["fiber"]]
        m = score_trace(p, f, s["t"], s["sign"], tree=cKDTree(f.points))
        m["avail"] = min(m["avail"], max_len)
        m["followed"] = min(m["followed"], max_len)
        m["coverage"] = m["followed"] / max(m["avail"], 1e-6)
        m["avail_nb"] = min(m["avail_nb"], max_len)
        m["coverage_nb"] = min(m["followed"], m["avail_nb"]) / max(m["avail_nb"], 1e-6)
        rows.append(m)
        # Mark prediction-support gaps for diagnosis; their annotated geometry is still GT.
        arc = s["t"] + s["sign"] * np.arange(0, max_len + 20)
        arc = arc[(arc >= 0) & (arc <= f.length)]
        g = interp_at(f.points, f.s, arc)
        fr = _gt_frames(g)
        seg_tree = cKDTree(g)
        d, j = seg_tree.query(p)
        off = np.einsum("ni,nij->nj", p - g[j], fr[j])
        for c, name in ((0, "u"), (1, "v")):
            q = g[:, None, :] + lat[None, :, None] * fr[:, None, :, c]
            img = vol.sample_image_nearest(q[..., ::-1]).astype(np.float32) / 255.0
            a = ax[k // per_row, 2 * (k % per_row) + c]
            a.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=1,
                     extent=(-half - 0.5, half + 0.5, -0.5, len(g) - 0.5))
            a.axvline(0, color="lime", lw=1.5, alpha=0.5)
            if f.brk is not None:
                bk = np.interp(arc, f.s, f.brk.astype(float)) > 0.5
                a.plot(np.where(bk, 0.0, np.nan), np.arange(len(arc)), color="red", lw=3)
            ok = d < half
            a.plot(np.where(ok, off[:, c], np.nan), j, "-", color="orange", lw=1)
            a.set_xticks([])
            if c == 0:
                a.set_title(f"{m['followed']:.0f}/{m['avail']:.0f}\n{r}", fontsize=7)
            else:
                a.set_yticks([])
            a.set_xlabel(name, fontsize=7)
            a.tick_params(labelsize=6)
    for k in range(n, nr * per_row):
        for c in (0, 1):
            ax[k // per_row, 2 * (k % per_row) + c].axis("off")
    summ = summarize(rows)
    fig.suptitle(f"straightened along GT (arc up, lateral across; green GT, orange trace)  "
                 f"coverage {summ['coverage_mean']:.3f}  diverged {summ['diverged']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return summ


def plot_curves(log_path, path):
    recs = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    training = [r for r in recs if 'flow' in r]
    rollout = [r for r in recs if 'roll_coverage' in r]
    fig,axes = plt.subplots(1,3,figsize=(13,3.2))
    steps = [r['step'] for r in training]
    axes[0].plot(steps,[r['flow'] for r in training],label='normalized flow loss')
    axes[1].plot(steps,[r['confidence_loss'] for r in training],label='prefix confidence BCE')
    for threshold in (.5,.85):
        selected = [r for r in rollout if r['threshold']==threshold]
        for name in ('coverage','precision','diverged'):
            axes[2].plot([r['step'] for r in selected],[r['roll_'+name] for r in selected],label=f'{name} @ {threshold}')
    for ax in axes:
        ax.legend(fontsize=7);ax.set_xlabel('optimizer updates')
    axes[2].set_ylim(0,1)
    fig.tight_layout();fig.savefig(path,dpi=100);plt.close(fig)


def plot_denoising(curves, history, hmask, path):
    """Fixed observed history plus zero initialization and every midpoint update."""
    curves=curves.detach().float().cpu().numpy()
    history=history.detach().float().cpu().numpy()
    mask=hmask.detach().cpu().numpy().astype(bool)
    fig,axes=plt.subplots(2,len(curves),figsize=(4*len(curves),9),squeeze=False)
    for b in range(len(curves)):
        for lateral in (0,1):
            ax=axes[lateral,b]
            ax.plot(history[b,mask[b],lateral],history[b,mask[b],2],'r.-',label='observed history')
            for step,curve in enumerate(curves[b]):
                ax.plot(curve[:,lateral],curve[:,2],'.-',label=f'update {step}',alpha=.4+.6*step/max(1,len(curves[b])-1))
            ax.set_xlabel('lateral voxels');ax.set_ylabel('forward voxels');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
