"""CPU-only plots of saved consistency analysis; no model or GPU imports."""
import argparse
import json
from pathlib import Path

import numpy as np

from .embedding_consistency import distribution, unit


def cosine_groups(cases):
    groups = {}
    for scroll in sorted({case['scroll'] for case in cases}):
        positive, negative, high = [], [], []
        for case in cases:
            if case['scroll'] != scroll:
                continue
            vectors = np.asarray(case['blended_embeddings'], dtype=float)
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError('invalid saved embedding vectors')
            if np.any(np.linalg.norm(vectors, axis=1) <= 1e-8):
                raise ValueError('zero GT embedding cannot define cosine')
            vectors = unit(vectors)
            pairs = np.triu_indices(len(vectors), 1)
            positive.extend((vectors @ vectors.T)[pairs])
            cosines = np.asarray(case['negative_cosines'], dtype=float)
            presence = np.asarray(case['negative_presences'], dtype=float)
            if cosines.shape != presence.shape or not np.isfinite(cosines).all() or not np.isfinite(presence).all():
                raise ValueError('invalid saved negative pairs')
            negative.extend(cosines)
            high.extend(cosines[presence >= .5])
        groups[scroll] = dict(positive=np.clip(positive, -1, 1),
                              negative=np.clip(negative, -1, 1),
                              high=np.clip(high, -1, 1))
    return groups


def histogram(values, edges):
    if not len(values):
        return None
    counts, _ = np.histogram(values, bins=edges)
    if counts.sum() != len(values):
        raise ValueError('histogram discarded values')
    return counts / counts.sum()


def render(settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(settings['output'])
    saved = json.loads((out / 'results.json').read_text())
    if 'seconds' not in saved:
        raise ValueError('consistency analysis is incomplete')
    groups = cosine_groups(saved['cases'])
    edges = np.linspace(-1, 1, 41)
    hist = {scroll: {key: histogram(values, edges) for key, values in row.items()}
            for scroll, row in groups.items()}
    aggregate = {}
    coverage = {}
    for key in ('positive', 'negative', 'high'):
        values = [row[key] for row in hist.values() if row[key] is not None]
        aggregate[key] = np.mean(values, axis=0) if values else None
        coverage[key] = len(values)
    rows = [('Equal-scroll mean', aggregate), *hist.items()]
    fig, axes = plt.subplots((len(rows)+1)//2, 2, figsize=(13, 10), sharex=True)
    colors = dict(positive='#2376b7', negative='#777777', high='#d95f02')
    labels = dict(positive='GT–GT (blended window)', negative='GT–off-GT (all)',
                  high='GT–off-GT (presence ≥ 0.5)')
    for axis, (name, row) in zip(axes.flat, rows):
        for key in ('negative', 'positive', 'high'):
            if row[key] is not None:
                axis.stairs(row[key]*100, edges, color=colors[key], label=labels[key],
                            linewidth=1.8, fill=key == 'positive', alpha=.25 if key == 'positive' else 1)
        axis.set_title(name, fontsize=12)
        axis.set_ylabel('Pairs per bin (%)')
        axis.set_xlabel('Cosine similarity')
        axis.set_xlim(-1, 1)
        axis.grid(alpha=.18)
        if name in groups:
            counts = groups[name]
            axis.text(.02, .96, f"N pos={len(counts['positive']):,}; neg={len(counts['negative']):,}; high={len(counts['high']):,}",
                      transform=axis.transAxes, va='top', fontsize=8)
    for axis in list(axes.flat)[len(rows):]:
        axis.set_visible(False)
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3)
    fig.suptitle(f"Embedding cosine distributions · raw update {saved['step']:,} · {saved['samples_seen']:,} samples", fontsize=15)
    fig.text(.5, .015, 'Positive: all distinct GT pairs in each blended window. Negative: saved same-tile controls >80 base voxels from GT.\n'
             'Each curve normalized separately; bin width 0.05. These are not matched-anchor gaps or the training pair distribution.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .91))
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'cosine_histograms.{suffix}', dpi=160)
    plt.close(fig)
    summary = dict(step=saved['step'], samples_seen=saved['samples_seen'],
                   checkpoint_sha256=saved['checkpoint_sha256'], edges=edges.tolist(),
                   aggregate={k: None if v is None else v.tolist() for k, v in aggregate.items()},
                   aggregate_scroll_coverage=coverage,
                   per_scroll={s: {k: distribution(v) for k, v in row.items()} for s, row in groups.items()})
    if aggregate['positive'] is not None and aggregate['high'] is not None:
        summary['binned_positive_high_overlap'] = float(np.minimum(aggregate['positive'], aggregate['high']).sum())
    (out / 'cosine_histograms.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"plot={out / 'cosine_histograms.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('control')
    args = parser.parse_args()
    render(json.loads(Path(args.control).read_text())['embedding_consistency'])
