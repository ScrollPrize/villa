"""Figures for ARCHITECTURE.md (the m7 model). Diagrams are schematic but to scale; the crop and trace figures use a
real checkpoint and volume.

  python -m vesuvius.neural_tracing.sheet_tracing.docs.make_architecture_figures --checkpoint CKPT [--path-map OLD=NEW]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent/'figures'
SURFACE, INK, MUTED = '#fcfcfb', '#1f1f1e', '#6b6a64'
POINT, TOKEN, CONTEXT, MEMORY, OUTPUT = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'  # categorical slots 1-5
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE})


def box(ax, x, y, w, h, title, body='', color=MUTED, fill=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.008', linewidth=1.6,
                                edgecolor=color, facecolor=matplotlib.colors.to_rgba(color, fill)))
    ax.text(x+w/2, y+h-0.012, title, ha='center', va='top', fontsize=10, fontweight='bold', color=INK)
    if body:
        ax.text(x+w/2, y+h-0.043, body, ha='center', va='top', fontsize=8.2, color=INK, linespacing=1.35)


def arrow(ax, a, b, color=MUTED, style='-|>', rad=0.):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, linewidth=1.4, color=color,
                                 connectionstyle=f'arc3,rad={rad}'))


def route(ax, pts, color=MUTED):
    """Polyline through ``pts`` with an arrowhead at the end (for arrows routed through the gaps between boxes)."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], '-', color=color, lw=1.4, solid_joinstyle='miter')
    arrow(ax, pts[-2], pts[-1], color)


def dataflow():
    fig, ax = plt.subplots(figsize=(17, 10.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.01, 0.985, 'm7 sheet tracer: one tracing step (B = batch, all positions in level-0 voxels)', fontsize=13,
            fontweight='bold', color=INK, va='top')
    box(ax, 0.01, 0.62, 0.17, 0.30, 'Volume block', 'around the current front\n(level 1, i.e. 2 vx voxels)\n\n'
        'channel 0: CT, z-scored\nper crop\nchannel 1: m7 surface\nprediction, / 255\n\ntrilinear sampling,\n'
        'zeros outside', MUTED)
    box(ax, 0.01, 0.23, 0.17, 0.34, 'Front state', 'pos (B,5,3): control points\nprev (B,5,3): previous row\n'
        'conf (B,5): last confidence\ncurve (B,25,3): Hermite curve,\n  respaced by arc length\n\n'
        'grid memory: last 3 rows,\neach (tokens (B,25,192),\ncurve, frames, conf)\n\n'
        'point frames (t, a, n):\nalong front, advance, normal', MEMORY)
    box(ax, 0.21, 0.72, 0.18, 0.20, 'Point crops (5 per front)', '2 x 16 x 32 x 20 samples\n(normal x tangent x advance)\n'
        'at 2 vx: 30 x 62 x 38 vx box\nin the point frame, centre\n10 vx ahead of the point', POINT)
    box(ax, 0.21, 0.47, 0.18, 0.20, 'Context crops (5)', '2 x 16 x 16 x 16 samples\nat 6 vx: 90 vx cube\n'
        'in the point frame,\ncentred on the point', CONTEXT)
    box(ax, 0.21, 0.23, 0.18, 0.20, 'Grid-column crops (25)', '2 x 8 x 8 x 8 samples\nat 2 vx: 14 vx cube\n'
        'at each curve column, in\nthe column frame (curve\ntangent, interpolated a, n)', TOKEN)
    box(ax, 0.42, 0.72, 0.17, 0.20, 'Point CNN', 'Conv3d 2-32 s1, 32-64 s2,\n64-96 s2 (3x3x3, pad 1),\n'
        'each + GroupNorm(4) + GELU\n-> 96 x 4 x 8 x 5, flatten\n-> Linear 15360 -> 192', POINT)
    box(ax, 0.42, 0.47, 0.17, 0.20, 'Context CNN', 'Conv3d 2-8 s2, 8-16 s2,\n16-32 s1 + GN + GELU\n'
        '-> 32 x 4 x 4 x 4 -> Linear\n2048 -> 192, then a Linear\n192 -> 192 initialised at 0', CONTEXT)
    box(ax, 0.42, 0.23, 0.17, 0.20, 'Token CNN', 'Conv3d 2-24 s1, 24-48 s2,\n48-64 s2 + GN + GELU\n'
        '-> 64 x 2 x 2 x 2 -> Linear\n512 -> 192', TOKEN)
    box(ax, 0.63, 0.72, 0.16, 0.20, 'Point tokens x', 'x = CNN(point crop)\n + MLP(16 geometry features)\n'
        ' + proj(CNN(context crop))\n(B, 5, 192)', POINT)
    box(ax, 0.63, 0.23, 0.16, 0.43, 'Surface-token grid', 'current row (B,25,192)\n = CNN(column crop)\n'
        ' + Linear(point CNN output)\n   added at columns\n   0, 6, 12, 18, 24\n\nstack with the 3 stored rows\n'
        '-> (B, 4, 25, 192)\n + MLP(17 geometry features)\n\n4 axial blocks, each:\nrow attention (over 25)\n'
        'column attention (over 4)\nMLP 192 -> 384 -> 192\n(pre-norm, residual)\n\n-> memory (B, 100, 192)', TOKEN)
    box(ax, 0.82, 0.53, 0.17, 0.39, 'Point transformer', '4 blocks, each pre-norm\nresidual:\n\n'
        'self-attention over the\n5 point tokens\n\ncross-attention from the\npoints to the 100 grid\n'
        'tokens\n\nMLP 192 -> 384 -> 192\n\n6 heads, dim 192', POINT)
    box(ax, 0.82, 0.23, 0.17, 0.26, 'Head', 'LayerNorm, Linear 192 -> 6\n(initialised at 0)\n\n'
        'displacement (t, a, n),\nin 20 vx steps, added to a\nstraight 20 vx advance\n\nconfidence logit\n'
        'tangent tilt (a, n)', OUTPUT)
    box(ax, 0.21, 0.02, 0.78, 0.17, 'Curve update (no learned parameters)',
        'pred = pos + frame . (dt, 1 + da, dn) x 20 vx;   confident = sigmoid(logit) > threshold.   '
        'Stopped points move by their confident neighbours\' interpolated\ndisplacement, then onto the Hermite '
        'curve through those neighbours.   Tangents: Catmull-Rom (reflected at the ends); confident points add '
        'their tilt.\nHermite curve through the 5 points, 6 samples per gap: 25 output columns (written as one row).   '
        'Next front: 5 points re-spaced evenly by arc length on\nthat curve, prev = old pos.   The current grid '
        'row\'s tokens (before axial attention) join the memory, which keeps the last 3 rows.', OUTPUT, fill=0.08)
    for a, b in [((0.18, 0.80), (0.21, 0.82)), ((0.18, 0.72), (0.21, 0.60)), ((0.18, 0.66), (0.21, 0.37)),
                 ((0.18, 0.36), (0.21, 0.36)), ((0.39, 0.82), (0.42, 0.82)), ((0.39, 0.57), (0.42, 0.57)),
                 ((0.39, 0.33), (0.42, 0.33)), ((0.59, 0.84), (0.63, 0.84)), ((0.59, 0.33), (0.63, 0.33)),
                 ((0.79, 0.84), (0.82, 0.84)), ((0.79, 0.50), (0.82, 0.62)), ((0.905, 0.53), (0.905, 0.49)),
                 ((0.905, 0.23), (0.905, 0.19))]:
        arrow(ax, a, b)
    route(ax, [(0.59, 0.56), (0.61, 0.56), (0.61, 0.78), (0.63, 0.78)], CONTEXT)  # context into the point tokens
    route(ax, [(0.59, 0.76), (0.605, 0.76), (0.605, 0.60), (0.63, 0.60)], POINT)  # point CNN into the grid
    route(ax, [(0.095, 0.23), (0.095, 0.205), (0.615, 0.205), (0.615, 0.27), (0.63, 0.27)], MEMORY)  # stored rows
    ax.text(0.40, 0.208, 'stored grid rows', fontsize=8, color=MUTED, va='bottom')
    route(ax, [(0.21, 0.09), (0.195, 0.09), (0.195, 0.27), (0.18, 0.27)], OUTPUT)  # next front
    ax.text(0.19, 0.12, 'next\nstep', fontsize=8, color=MUTED, ha='right')
    fig.savefig(OUT/'dataflow.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def layout_top():
    """The front seen along the sheet normal: (t, a) plane, to scale."""
    fig, ax = plt.subplots(figsize=(13, 8.5))
    cols = np.arange(25)*10.
    pts = np.arange(5)*60.
    for i, a in enumerate((-60, -40, -20)):
        ax.plot(cols, np.full_like(cols, a), '-', color=MEMORY, lw=1.5)
        for c in cols:
            ax.add_patch(Rectangle((c-7, a-7), 14, 14, facecolor=matplotlib.colors.to_rgba(MEMORY, .25),
                                   edgecolor=MEMORY, lw=.6))
        ax.text(-60, a, f'stored row, {3-i} back', va='center', ha='left', fontsize=9, color=INK)
    ax.plot(cols, np.zeros_like(cols), '-', color=TOKEN, lw=2.2)
    for c in cols:
        ax.add_patch(Rectangle((c-7, -7), 14, 14, facecolor=matplotlib.colors.to_rgba(TOKEN, .25), edgecolor=TOKEN,
                               lw=.8))
    for p in pts:
        ax.add_patch(Rectangle((p-31, 10-19), 62, 38, facecolor=matplotlib.colors.to_rgba(POINT, .10),
                               edgecolor=POINT, lw=1.6))
        ax.plot(p, 0, 'o', ms=10, color=POINT, mec=SURFACE, mew=2, zorder=5)
        ax.add_patch(FancyArrowPatch((p, 2), (p+[0, 4, -3, 2, 0][int(p//60)], 20), arrowstyle='-|>',
                                     mutation_scale=14, lw=1.6, color=OUTPUT, zorder=6))
    ax.add_patch(Rectangle((120-45, -45), 90, 90, facecolor='none', edgecolor=CONTEXT, lw=1.8, ls='--'))
    nxt = np.array([[0, 20], [64, 21], [117, 19], [182, 20], [240, 20]], float)
    ax.plot(nxt[:, 0], nxt[:, 1], '--', color=OUTPUT, lw=1.6)
    ax.text(245, 22, 'next row (predicted)', color=INK, fontsize=9, va='bottom')
    ax.text(-60, 0, 'current row', va='center', ha='left', fontsize=9, color=INK, fontweight='bold')
    notes = [(POINT, 'point crop footprint: 62 x 38 vx (t x a), 30 vx deep along n, centred 10 vx ahead'),
             (TOKEN, 'grid-column crop: 14 vx cube on each of the 25 curve columns (10 vx apart)'),
             (MEMORY, 'grid columns of the 3 stored rows (tokens kept, not re-sampled)'),
             (CONTEXT, 'context crop of the middle point: 90 vx cube at 6 vx (one per point)'),
             (OUTPUT, 'predicted displacements: about one 20 vx step forward each')]
    for i, (c, t) in enumerate(notes):
        ax.add_patch(Rectangle((-60, 96-i*9), 8, 5, color=c))
        ax.text(-48, 98.5-i*9, t, va='center', fontsize=9, color=INK)
    ax.set_xlim(-65, 330)
    ax.set_ylim(-75, 105)
    ax.set_aspect('equal')
    ax.set_xlabel('t: along the front (vx); control points every 60 vx')
    ax.set_ylabel('a: advance direction (vx)')
    ax.set_title('Front layout seen along the sheet normal (to scale)', color=INK, fontweight='bold')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT/'layout_top.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def layout_section():
    """Section along the normal at one point: what each crop sees of the neighbouring sheets."""
    fig, ax = plt.subplots(figsize=(10, 7))
    a = np.linspace(-55, 55, 200)
    for k in range(-4, 5):
        n = k*12+1.5*np.sin(a/25+k)
        ax.plot(a, n, '-', color=INK if k == 0 else MUTED, lw=2.4 if k == 0 else 1, alpha=1 if k == 0 else .6)
    ax.fill_between(a, -6, 6, color=matplotlib.colors.to_rgba(OUTPUT, .12), lw=0)
    ax.add_patch(Rectangle((10-19, -15), 38, 30, facecolor='none', edgecolor=POINT, lw=2))
    ax.add_patch(Rectangle((-7, -7), 14, 14, facecolor='none', edgecolor=TOKEN, lw=2))
    ax.add_patch(Rectangle((-45, -45), 90, 90, facecolor='none', edgecolor=CONTEXT, lw=2, ls='--'))
    ax.plot(0, 0, 'o', ms=9, color=POINT, mec=SURFACE, mew=2, zorder=5)
    labels = [(29.5, 13, POINT, 'point crop: 30 vx along n,\n38 vx along a, 2 vx samples'),
              (7.5, -10, TOKEN, 'grid-column crop: 14 vx'),
              (-44, 41, CONTEXT, 'context crop: 90 vx, 6 vx samples'),
              (30, -4, OUTPUT, 'same-sheet tolerance: within 6 vx')]
    for x, y, c, t in labels:
        ax.text(x, y, t, color=INK, fontsize=9, va='center', bbox=dict(facecolor=SURFACE, edgecolor=c, lw=1.2))
    ax.text(52, 30, 'neighbouring sheets (about 12 vx apart)', color=MUTED, fontsize=9, ha='right', va='center',
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1))
    ax.set_xlim(-55, 55)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_xlabel('a: advance direction (vx)')
    ax.set_ylabel('n: sheet normal (vx)')
    ax.set_title('Section through one control point along the sheet normal', color=INK, fontweight='bold')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT/'layout_section.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def real_data(checkpoint: str, path_map: list[str], device: str):
    """Crops the model sees at the seed of a benchmark trace, and that trace."""
    from vesuvius.neural_tracing.fiber_follow.data.volume import ChunkedArray
    from vesuvius.neural_tracing.sheet_tracing.data.episodes import extra_block, extra_readers
    from vesuvius.neural_tracing.sheet_tracing.evaluate import load_checkpoint
    from vesuvius.neural_tracing.sheet_tracing.geometry import point_frames
    from vesuvius.neural_tracing.sheet_tracing.infer import read_block, resample, trace
    from vesuvius.neural_tracing.sheet_tracing.rollout import Front, Tracer
    from vesuvius.neural_tracing.sheet_tracing.sampling import sample_crops, view_radius
    from vesuvius.neural_tracing.sheet_tracing.trace_benchmark import draw_seeds, path_mapper, seed_polyline
    model, cfg = load_checkpoint(checkpoint, device)[:2]
    path_mapper(path_map)(cfg)
    cfg.trace.confidence_threshold = .4
    seeds, spec, data = draw_seeds(cfg, 12, 30, 240., .95)
    w = seeds[3]
    points, direction = seed_polyline(w, spec, data)
    seed = resample(points, cfg.data.point_spacing)
    idx = np.linspace(0, len(points)-1, len(seed))
    dirs = np.stack([np.interp(idx, np.arange(len(points)), direction[:, k]) for k in range(3)], 1)
    d = cfg.data
    unit = dirs/np.linalg.norm(dirs, axis=1, keepdims=True)
    pos = torch.tensor(seed, dtype=torch.float32, device=device)[None]
    prev = torch.tensor(seed-d.step*unit, dtype=torch.float32, device=device)[None]
    front = Front(pos, prev, torch.ones(1, len(seed), device=device))
    tracer = Tracer(model, cfg, device)
    volume = ChunkedArray(f'{d.volume}/{d.volume_level}', 1 << 29)
    block, origin = read_block(volume, seed, view_radius(cfg.model)+d.step+4, d.volume_level)
    extras = []
    for array, level in extra_readers(d, 1 << 28):
        eb, eo = extra_block(array, level, origin, block.shape, d.volume_level)
        extras.append((torch.from_numpy(eb).to(device).float()[None, None], torch.from_numpy(eo).to(device)[None], level))
    block = torch.from_numpy(block).to(device).float()[None, None]
    origin = torch.from_numpy(origin).to(device)[None]
    frames = point_frames(front.pos, front.prev)
    m = cfg.model
    point = sample_crops(block, origin, d.volume_level, front.pos, frames, tracer.offsets, extras)[0, 2].cpu().numpy()
    context = sample_crops(block, origin, d.volume_level, front.pos, frames, tracer.context_offsets, extras)[0, 2].cpu().numpy()
    grid, _ = tracer._grid(front, frames, block, origin, extras)
    token = grid['crops'][0, 12].cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    panels = [(point[:, :, :, point.shape[3]//2], 'point crop, normal x tangent\n(middle of the advance)', (32*2, 16*2)),
              (point[:, :, point.shape[2]//2, :], 'point crop, normal x advance\n(middle of the front)', (20*2, 16*2)),
              (token[:, :, :, token.shape[3]//2], 'grid-column crop (column 12),\nnormal x tangent', (8*2, 8*2)),
              (context[:, :, :, context.shape[3]//2], 'context crop, normal x tangent\n(6 vx samples)', (16*6, 16*6))]
    for col, (arr, title, (wx, hy)) in enumerate(panels):
        for row, (ch, name) in enumerate(((0, 'CT (z-scored)'), (1, 'm7 (/255)'))):
            ax = axes[row, col]
            lo, hi = np.percentile(arr[ch], [1, 99]) if ch == 0 else (0., 1.)
            ax.imshow(arr[ch], cmap='gray', origin='lower', extent=(-wx/2, wx/2, -hy/2, hy/2), interpolation='nearest',
                      vmin=lo, vmax=hi)
            ax.axhline(0, color=OUTPUT, lw=.8, alpha=.8)
            ax.set_title(f'{title}\n{name}' if row == 0 else name, fontsize=9, color=INK)
            ax.set_xlabel('vx', fontsize=8)
            if col == 0:
                ax.set_ylabel('normal (vx)', fontsize=8)
    fig.suptitle('What the model sees at a real seed (middle control point; the line marks the point\'s level)',
                 color=INK, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'crops_real.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    rows, confs = trace(model, cfg, seed, dirs, 30, device, min_confident=0., patience=10**9)
    center = rows[0].mean(0)
    t = rows[0][-1]-rows[0][0]
    t /= np.linalg.norm(t)
    a = (rows[1]-rows[0]).mean(0)
    a -= (a@t)*t
    a /= np.linalg.norm(a)
    n = np.cross(t, a)
    loc = (rows-center)@np.stack([t, a, n], 1)
    kept = confs > cfg.trace.confidence_threshold
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for r in range(len(loc)):
        ax1.plot(loc[r, :, 0], loc[r, :, 1], '-', color=MUTED if r else INK, lw=.8 if r else 2)
        ax1.plot(loc[r, ::6, 0], loc[r, ::6, 1], 'o', ms=4, color=POINT)
        bad = ~kept[r]
        if r and bad.any():
            ax1.plot(loc[r, bad, 0], loc[r, bad, 1], 'x', ms=6, color=OUTPUT, mew=1.6)
        ax2.plot(loc[r, :, 1], loc[r, :, 2], '.', ms=2, color=MUTED)
    ax2.plot(loc[:, 12, 1], loc[:, 12, 2], '-', color=POINT, lw=1.6)
    ax1.set_aspect('equal')
    ax1.set_xlabel('t at the seed (vx)')
    ax1.set_ylabel('a at the seed (vx)')
    ax1.set_title(f'{len(loc)-1} traced rows (seed row black). Dots: control points; x: below the confidence threshold',
                  fontsize=10, color=INK)
    ax2.set_xlabel('a at the seed (vx)')
    ax2.set_ylabel('n at the seed (vx)')
    ax2.set_title('Side view: the trace follows the sheet\'s curvature\n(line: the middle column)', fontsize=10,
                  color=INK)
    for ax in (ax1, ax2):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.savefig(OUT/'trace_real.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def tilt():
    """Catmull-Rom vs tilted tangents through the same points, over a sheet with a bump between two of them."""
    import torch as _torch
    from vesuvius.neural_tracing.sheet_tracing.geometry import chord_tangents, hermite_dense, normalize
    x = np.linspace(-10, 190, 400)
    sheet = 3.*np.exp(-((x-90)/18)**2)  # a 3 vx bump between the points at 60 and 120
    pts = _torch.tensor([[[0., 0, 0], [60., 0, 0], [120., 0, 0], [180., 0, 0]]])
    tangents = chord_tangents(pts)
    n = _torch.tensor([0., 1, 0])
    tilted = tangents.clone()
    tilted[0, 1] = normalize(tangents[0, 1]+.21*n)
    tilted[0, 2] = normalize(tangents[0, 2]-.21*n)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(x, sheet-6, sheet+6, color=matplotlib.colors.to_rgba(OUTPUT, .10), lw=0)
    ax.plot(x, sheet, '-', color=INK, lw=2.2, label='sheet (with a 3 vx bump)')
    for t, color, name in ((tangents, MUTED, 'Catmull-Rom (no tilt): straight between the points'),
                           (tilted, POINT, 'tilt 0.21 up at 60, down at 120: follows the bump,\nbut the neighbouring gaps bend the other way')):
        dense = hermite_dense(pts, t, 6)[0].numpy()
        ax.plot(dense[:, 0], dense[:, 1], 'o-' if color == POINT else 's--', color=color, ms=4, lw=1.6, label=name)
    for k, (p, t) in enumerate(zip(pts[0].numpy(), tilted[0].numpy())):
        ax.add_patch(FancyArrowPatch((p[0], p[1]), (p[0]+t[0]*22, p[1]+t[1]*22), arrowstyle='-|>', mutation_scale=12,
                                     lw=1.4, color=OUTPUT, zorder=6))
    ax.plot(pts[0, :, 0], pts[0, :, 1], 'o', ms=11, color=POINT, mec=SURFACE, mew=2, zorder=7)
    ax.text(150, -5, 'same-sheet tolerance (6 vx)', color=MUTED, fontsize=9)
    ax.set_xlabel('along the front (vx)')
    ax.set_ylabel('sheet normal (vx)')
    ax.set_ylim(-7, 8)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    ax.set_title('Tangent tilt: same control points, different curve between them (arrows: tilted tangents)',
                 color=INK, fontweight='bold')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT/'tilt.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def crops_xy(checkpoint: str, path_map: list[str], device: str, display_level: int | None = None,
             name: str = 'crops_xy.png'):
    """An xy slice (z = const) of the CT through a real seed whose advance is close to z, with the actual sample
    positions of the point, line and context crops that fall in the slice."""
    from scipy.spatial import ConvexHull
    from vesuvius.neural_tracing.fiber_follow.data.volume import ChunkedArray
    from vesuvius.neural_tracing.sheet_tracing.evaluate import load_checkpoint
    from vesuvius.neural_tracing.sheet_tracing.geometry import chord_tangents, normalize, point_frames
    from vesuvius.neural_tracing.sheet_tracing.infer import resample
    from vesuvius.neural_tracing.sheet_tracing.rollout import Front, Tracer
    from vesuvius.neural_tracing.sheet_tracing.trace_benchmark import draw_seeds, path_mapper, seed_polyline
    model, cfg = load_checkpoint(checkpoint, device)[:2]
    path_mapper(path_map)(cfg)
    seeds, spec, data = draw_seeds(cfg, 100, 30, 240., .95)
    best = None
    for w in seeds:  # advance along z and the front within the xy plane: the seed row lies in one slice
        pts, d = seed_polyline(w, spec, data)
        a = normalize(torch.tensor(d.mean(0)))
        t = normalize(torch.tensor(pts[-1]-pts[0]))
        score = abs(float(a[2]))-abs(float(t[2]))
        if best is None or score > best[0]:
            best = (score, w, pts, d)
    _, w, pts, d = best
    seed = resample(pts, cfg.data.point_spacing)
    idx = np.linspace(0, len(pts)-1, len(seed))
    dirs = np.stack([np.interp(idx, np.arange(len(pts)), d[:, k]) for k in range(3)], 1)
    unit = dirs/np.linalg.norm(dirs, axis=1, keepdims=True)
    pos = torch.tensor(seed, dtype=torch.float32)[None]
    prev = torch.tensor(seed-cfg.data.step*unit, dtype=torch.float32)[None]
    front = Front(pos, prev, torch.ones(1, len(seed)))
    tracer = Tracer(model, cfg, 'cpu')
    frames = point_frames(front.pos, front.prev)
    curve = tracer._curve(front)
    t = chord_tangents(curve)
    a = tracer._interp_front_points(frames[..., 1])
    a = normalize(a-(a*t).sum(-1, keepdim=True)*t)
    col_frames = torch.stack([t, a, normalize(torch.cross(t, a, dim=-1))], -1)
    world = lambda p, fr, off: (p[:, :, None, None, None]+torch.einsum('bnij,stuj->bnstui', fr, off.cpu()))[0].reshape(
        p.shape[1], -1, 3).numpy()
    point_s = world(front.pos, frames, tracer.offsets)
    line_s = world(curve, col_frames, tracer.token_offsets)
    ctx_s = world(front.pos, frames, tracer.context_offsets)
    z0 = float(seed[:, 2].mean())
    lo, hi = seed[:, :2].min(0)-70, seed[:, :2].max(0)+70
    level = cfg.data.volume_level if display_level is None else display_level
    vol = ChunkedArray(f'{cfg.data.volume}/{level}', 1 << 28)
    to_i = lambda v: int(np.floor((v+.5)/(1 << level)-.5))
    x0, x1, y0, y1, zi = to_i(lo[0]), to_i(hi[0])+2, to_i(lo[1]), to_i(hi[1])+2, int(round((z0+.5)/(1 << level)-.5))
    img = vol.read((zi, y0, x0), (1, y1-y0, x1-x0))[0].astype(np.float32)
    ext = [(x0+.5)*(1 << level)-.5, (x1+.5)*(1 << level)-.5, (y1+.5)*(1 << level)-.5, (y0+.5)*(1 << level)-.5]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax, zoom in zip(axes, (None, 1)):
        ax.imshow(img, cmap='gray', extent=ext, vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        for k, samples in enumerate(point_s):
            sl = samples[np.abs(samples[:, 2]-z0) <= 1.]
            if len(sl) > 2:
                ax.plot(sl[:, 0], sl[:, 1], '.', ms=1.6 if zoom is None else 3, color=POINT, alpha=.8)
                hull = ConvexHull(sl[:, :2])
                ax.fill(sl[hull.vertices, 0], sl[hull.vertices, 1], fill=False, ec=POINT, lw=1.8)
        for j, samples in enumerate(line_s):
            sl = samples[np.abs(samples[:, 2]-z0) <= 1.]
            if len(sl) > 2:
                if zoom is not None:
                    ax.plot(sl[:, 0], sl[:, 1], '.', ms=3, color=TOKEN)
                hull = ConvexHull(sl[:, :2])
                ax.fill(sl[hull.vertices, 0], sl[hull.vertices, 1], fill=False, ec=TOKEN, lw=1.2)
        mid = len(seed)//2
        sl = ctx_s[mid][np.abs(ctx_s[mid][:, 2]-z0) <= 3.]
        if len(sl) > 2:
            hull = ConvexHull(sl[:, :2])
            ax.fill(sl[hull.vertices, 0], sl[hull.vertices, 1], fill=False, ec=CONTEXT, lw=1.8, ls='--')
        c = curve[0].numpy()
        ax.plot(c[:, 0], c[:, 1], '-', color=OUTPUT, lw=1.2)
        ax.plot(seed[:, 0], seed[:, 1], 'o', ms=8 if zoom is None else 11, color=POINT, mec=SURFACE, mew=1.8, zorder=5)
        if zoom is not None:
            p = seed[mid]
            ax.set_xlim(p[0]-50, p[0]+50)
            ax.set_ylim(p[1]+50, p[1]-50)
            ax.set_title('Middle control point (dots: the crops\' sample positions in this slice)', fontsize=10,
                         color=INK)
        else:
            note = '' if level == cfg.data.volume_level else f' (the model reads level {cfg.data.volume_level})'
            ax.set_title(f'xy slice of the level-{level} CT at z = {z0:.0f}{note},\nthrough a real seed (advance close to +z)',
                         fontsize=10, color=INK)
        ax.set_xlabel('x (level-0 vx)')
        ax.set_ylabel('y (level-0 vx)')
    handles = [matplotlib.lines.Line2D([], [], color=POINT, lw=1.8, label='point crop (30 vx along the normal x 62 vx along the front)'),
               matplotlib.lines.Line2D([], [], color=TOKEN, lw=1.2, label='line crops (14 vx), one per curve column'),
               matplotlib.lines.Line2D([], [], color=CONTEXT, lw=1.8, ls='--', label='context crop of the middle point (90 vx)'),
               matplotlib.lines.Line2D([], [], color=OUTPUT, lw=1.2, label='seed curve (25 columns)'),
               matplotlib.lines.Line2D([], [], marker='o', ls='', color=POINT, label='control points')]
    axes[0].legend(handles=handles, loc='lower left', fontsize=8.5, framealpha=.9)
    fig.tight_layout()
    fig.savefig(OUT/name, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'seed {w["meta"]["patch"]}: advance z component {float(normalize(torch.tensor(dirs.mean(0)))[2]):.2f}, '
          f'front z spread {np.ptp(seed[:, 2]):.1f} vx, slice z {z0:.0f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', help='for the real-data figures (skipped without it)')
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    dataflow()
    layout_top()
    layout_section()
    tilt()
    if args.checkpoint:
        real_data(args.checkpoint, args.path_map, args.device)
        crops_xy(args.checkpoint, args.path_map, args.device)
        crops_xy(args.checkpoint, args.path_map, args.device, display_level=0, name='crops_xy_level0.png')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
