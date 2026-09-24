"""Diagnostics for the beam re-ranker: pool images and the span restart metric."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vesuvius.neural_tracing.fiber_follow.beam.native import fiber_input_from_traced


def plot_pool(batch, output, crop, path, k_back, n=6):
    """Per state: two max projections of the CT crop with every candidate.

    Green: on-fiber label, red: off, grey: unlabeled. Orange ring marks the
    hand tracer's choice (pool index 0), cyan ring the model's argmax.
    """
    B = min(n, len(batch['x']))
    fig, axes = plt.subplots(B, 2, figsize=(7, 3.2 * B), squeeze=False)
    x = batch['x'][:, 0].float().cpu().numpy()
    cands = batch['candidates'].cpu().numpy()
    pmask = batch['point_mask'].cpu().numpy()
    cmask = batch['cand_mask'].cpu().numpy()
    on = batch['onfiber'].cpu().numpy()
    labeled = batch['label_mask'].cpu().numpy()
    ranks = output['ranks'].detach().float().cpu().numpy()
    mid = (crop.width - 1) / 2
    for i in range(B):
        valid = cmask[i] > 0
        model_pick = int(np.argmax(np.where(valid, ranks[i], -np.inf)))
        for j, (axis, lat) in enumerate(((1, 0), (2, 1))):  # project over v -> show u; over u -> show v
            ax = axes[i, j]
            img = x[i].max(axis)  # (D, W)
            ax.imshow(img, cmap='gray', origin='lower', aspect='auto')
            for c in range(len(valid)):
                if not valid[c]:
                    continue
                pts = cands[i, c][pmask[i, c] > 0]
                col = 'lime' if on[i, c] > 0 else ('red' if labeled[i, c] > 0 else '0.6')
                ax.plot(pts[:, lat] / crop.spacing + mid, pts[:, 2] / crop.spacing + crop.behind, color=col, lw=.8, alpha=.8)
                if c == 0 or c == model_pick:
                    end = pts[-1]
                    ax.scatter([end[lat] / crop.spacing + mid], [end[2] / crop.spacing + crop.behind], s=60,
                               facecolors='none', edgecolors='orange' if c == 0 else 'cyan', lw=1.5)
            ax.set_title(f"state {i} {'u' if lat == 0 else 'v'}-f  off={int(batch['offtrack'][i])} "
                         f"hand_on={int(on[i, 0])} model_on={int(on[i, model_pick])}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def span_diag(beam, fibers, grid_scale, hook=None, error_threshold_base=20.0):
    """VC's restart metric over ``fibers`` with and without the learned hook."""
    rows = []
    for fiber in fibers:
        fiber_input = fiber_input_from_traced(fiber, grid_scale)
        hand = beam.whole_fiber_metric(fiber_input, hook=None, error_threshold_base=error_threshold_base)
        row = dict(fiber=fiber.name, length_grid=hand['reference_length_grid'], segments=hand['segment_count'],
                   hand_restarts=hand['restart_count'])
        if hook is not None:
            model = beam.whole_fiber_metric(fiber_input, hook=hook, error_threshold_base=error_threshold_base)
            row['model_restarts'] = model['restart_count']
        rows.append(row)
    length = sum(r['length_grid'] for r in rows)
    segments = sum(r['segments'] for r in rows)
    summary = dict(span_fibers=len(rows), span_segments=segments, span_length_grid=length,
                   span_restarts_kvx_hand=1000. * sum(r['hand_restarts'] for r in rows) / max(length, 1e-9),
                   span_success_hand=1 - sum(r['hand_restarts'] for r in rows) / max(segments, 1))
    if hook is not None:
        summary.update(span_restarts_kvx_model=1000. * sum(r['model_restarts'] for r in rows) / max(length, 1e-9),
                       span_success_model=1 - sum(r['model_restarts'] for r in rows) / max(segments, 1))
    return rows, summary
