"""Self-contained results page (HTML, example images embedded) and every metric (long-format CSV) from tube_report.

Usage: python -m vesuvius.neural_tracing.fiber_follow.evaluation.tube_tables REPORT.json --traces TRACES.csv \
    --html RESULTS.html --csv results.csv [--title TITLE]
REPORT.json and TRACES.csv are tube_report's --out and --traces. Rerun after adding models; nothing is edited by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SOURCE_NOTES = {'paris4': 'manual Paris4 fibers', '0175A_5mm_v1': 'AFV PHerc0175A', '1447_5mm_v1': 'AFV PHerc1447',
                'paris4_afv_central': 'AFV Paris4 central'}


def pct(v):
    return '-' if v is None else f'{100*v:.1f}%'


def ci_pct(m):
    return f'{100*m[0]:.1f}% ({100*m[1]:.1f}-{100*m[2]:.1f})' if len(m) == 3 else pct(m[0])


def mm_every(rate):
    return '-' if not rate else f'every {10/rate:.0f} mm' if 10/rate >= 20 else f'every {10/rate:.1f} mm'


def matched(entry, labels):
    """Each run's per-seed mistake share and loss rate at the highest early-stop share among the runs' own thresholds.

    Replay only raises a run's stop rate, so the common level is the largest base value; runs that cannot reach it
    (no curve) are reported as '-'.
    """
    curves = {}
    for label in labels:
        base = {k: v[0] for k, v in entry['runs'][label]['metrics'].items()}
        points = [base]+[m for _, m in entry.get('curves', {}).get(label, [])]
        curves[label] = points
    target = max(c[0]['seeds_stopped_early'] for c in curves.values())
    out = {}
    for label, points in curves.items():
        x = np.array([p['seeds_stopped_early'] for p in points])
        if x.max() < target-1e-9:
            out[label] = None
            continue
        order = np.argsort(x)
        out[label] = {k: float(np.interp(target, x[order], np.array([p[k] for p in points])[order]))
                      for k in ('seeds_mistake', 'losses_rate', 'fiber_coverage')}
    return target, out


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e3e1da;--card:#ffffff;--accent:#2f5d8a;--bad:#a8402f;--good:#2f7a4f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;
  --card:#1f1f1d;--accent:#8db4dc;--bad:#e08a78;--good:#7fc79d}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#1f1f1d;--accent:#8db4dc;--bad:#e08a78;--good:#7fc79d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:16px;margin:20px 0 6px}p,li{color:var(--fg)}.muted{color:var(--muted)}
.wrap{overflow-x:auto;margin:6px 0 14px}table{border-collapse:collapse;min-width:100%;background:var(--card);font-size:14px}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:color-mix(in srgb,var(--accent) 10%,var(--card))}.star{color:var(--bad);font-weight:600}
figure{margin:10px 0 22px;background:var(--card);border:1px solid var(--line);padding:8px}figure img{width:100%;height:auto;display:block}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;
margin-right:6px;background:color-mix(in srgb,var(--accent) 15%,var(--card))}.tag.bad{background:color-mix(in srgb,var(--bad) 20%,var(--card))}
.tag.good{background:color-mix(in srgb,var(--good) 20%,var(--card))}nav a{color:var(--accent);margin-right:12px}
"""


def esc(text):
    import html
    return html.escape(str(text))


def table(header, rows):
    return ('<div class="wrap"><table><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in header) + '</tr>'
            + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows) + '</table></div>')


def embed(path, width=1100):
    import base64, io
    from PIL import Image
    image = Image.open(path).convert('RGB')
    if image.width > width:
        image = image.resize((width, round(image.height*width/image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=82, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def examples(traces, source, run):
    """Longest correct trace, up to two mistakes (a switch first, then the longest run after the mistake) and the
    early stop with the most annotation left, among traces with a render."""
    rows = [t for t in traces if t['source'] == source and t['run'] == run and t['image']]
    num = lambda t, k: float(t[k]) if t[k] not in ('', 'None') else 0.
    picks = []
    if rows:
        picks.append(('longest correct trace', max(rows, key=lambda t: num(t, 'verified_length'))))
    mistakes = sorted([t for t in rows if t['outcome'] == 'mistake'], key=lambda t: -num(t, 'continued_after_loss'))
    switch = next((t for t in mistakes if t['loss_kind'] == 'switch'), None)
    chosen = ([switch] if switch else []) + [t for t in mistakes if t is not switch][:2-bool(switch)]
    picks += [('mistake: switched onto another fiber' if t['loss_kind'] == 'switch' else 'mistake: left the fiber', t)
              for t in chosen]
    early = [t for t in rows if t['outcome'] == 'stopped_early']
    if early:
        picks.append(('stopped early (most annotation left)', max(early, key=lambda t: num(t, 'remaining'))))
    return picks


def caption(kind, t, um):
    mm = lambda k: f"{float(t[k])*um/1e3:.1f} mm" if t[k] not in ('', 'None') else '-'
    parts = [f"fiber {esc(t['fiber_name'])}, direction {'+' if float(t['sign']) > 0 else '-'}, seed #{t['seed_index']}",
             f"on-fiber {mm('verified_length')} of {mm('available')} annotated ahead; traced {mm('total_length')} in total",
             f"stop: {esc(t['reason'])}"]
    if t['outcome'] == 'mistake':
        parts.append(f"mistake at {mm('loss_at')}, kept going {mm('continued_after_loss')}")
    if t['outcome'] == 'stopped_early':
        parts.append(f"{mm('remaining')} of annotation left")
    tone = 'bad' if t['outcome'] == 'mistake' else 'good' if kind.startswith('longest') else ''
    return f'<span class="tag {tone}">{esc(kind)}</span>' + ' | '.join(parts)


def html(report, title, traces=None):
    labels = list(report['runs'])
    s = report['settings']
    out = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Fiber Tracing Results</title><style>{CSS}</style></head><body><main><h1>{esc(title)}</h1>',
           '<nav><a href="#totals">Totals</a><a href="#runs">Runs</a><a href="#outcomes">Outcomes</a><a href="#distance">Distance to mistake</a>'
           '<a href="#stops">Stopping</a><a href="#quality">Quality</a><a href="#matched">Same stop rate</a>'
           '<a href="#paired">Paired</a><a href="#images">Images</a></nav>']
    out.append('<h2 id="totals">Total length traced</h2><p class="muted">Correct = on the fiber (in the tube) before the annotation '
               'end. Incorrect = off the fiber before the annotation end. Length traced past annotation ends is not scored and '
               'not included.</p>')
    rows, totals = [], {l: [0., 0.] for l in labels}
    for source, entry in report['sources'].items():
        cells = [esc(source)]
        for label in labels:
            m = entry['runs'].get(label, {}).get('metrics')
            if m is None:
                cells += ['-', '-', '-']
                continue
            c, w = m['correct_mm'][0], m['incorrect_mm'][0]
            totals[label][0] += c
            totals[label][1] += w
            cells += [f'{c/1e3:.2f} m', f'{w/1e3:.2f} m', pct(w/max(c+w, 1e-9))]
        rows.append(cells)
    rows.append(['<b>all volumes</b>'] + [x for l in labels for x in (f'<b>{totals[l][0]/1e3:.2f} m</b>', f'<b>{totals[l][1]/1e3:.2f} m</b>',
                                                                       f'<b>{pct(totals[l][1]/max(sum(totals[l]), 1e-9))}</b>')])
    out.append(table(['volume'] + [f'{l}: {k}' for l in labels for k in ('correct', 'incorrect', 'incorrect share')], rows))
    out.append('<h2 id="runs">Runs</h2>')
    rows = []
    for label, run in report['runs'].items():
        stop = run.get('annotation_end_stop')
        rows.append([esc(label), esc(f"{Path(str(run.get('checkpoint'))).name} (step {run.get('step')})"), run.get('confidence'),
                     run.get('n_commit'), esc(run.get('gate')), 'on' if run.get('refit_retry') else 'off', run.get('batch'),
                     f"+{stop['margin']:g} vox" if stop else 'no'])
    out.append(table(['run', 'checkpoint', 'threshold', 'commit', 'gate', 'refit retry', 'trace batch', 'traces end at annotation end'], rows))
    out.append(f"""<p>Every run uses the same frozen seeds: each held-out fiber traced in both directions from one point.
A trace is <b>on the fiber</b> while it stays in a tube of {s['tube_normal']} trace voxels along the sheet normal and
{s['tube_width']} across the strip width (1 trace voxel = 19.2 um on Paris4, 17.3 um on the AFV volumes).
A <b>mistake</b> means the trace left the tube for at least {s['loss_length']:g} voxels and never came back. A <b>switch</b> is a
mistake that runs along another annotated fiber. <b>Stopped early</b> means the model stopped on the fiber with at least
{s['premature_remaining']:g} voxels of annotation still ahead. Nothing past an annotation end is scored. Ranges are 95%
fiber-bootstrap intervals.</p>""")
    blocks = (
        ('outcomes', 'From a seed: how does the trace end?', (
            ('mistake, kept going', lambda m, e: ci_pct(m['seeds_mistake'])),
            ('stopped early (on the fiber)', lambda m, e: pct(m['seeds_stopped_early'][0])),
            ('reached the annotation end', lambda m, e: pct(m['seeds_reached_end'][0])),
            ('other (ended slightly off)', lambda m, e: pct(m['seeds_other'][0])),
            ('annotation available per seed', lambda m, e: f"{m['annotation_per_seed'][0]:.1f} mm"))),
        ('distance', 'How far before a mistake?', (
            ('median distance to the mistake', lambda m, e: '-' if e['mistake_at_median'] is None else f"{e['mistake_at_median']:.1f} mm"),
            *[(f'no mistake by {d}', (lambda d: lambda m, e: '-' if e['mistake_free'][d] is None else
                                     f"{100*e['mistake_free'][d][0]:.1f}% (n={e['mistake_free'][d][1]})")(d))
              for d in ('1mm', '2mm', '5mm', '10mm', '20mm')],
            ('on-fiber mm per mistake', lambda m, e: f"{m['distance_per_mistake'][0]:.0f} mm"),
            ('mm traced after a mistake (mean)', lambda m, e: f"{m['continued_after_mistake'][0]:.1f} mm"),
            ('mistakes / of which switches', lambda m, e: f"{m['losses'][0]:.0f} / {m['switches'][0]:.0f}"))),
        ('stops', 'Stopping and coverage', (
            ('stops early', lambda m, e: mm_every(m['premature_stops_rate'][0])),
            ('fiber covered (both directions)', lambda m, e: pct(m['fiber_coverage'][0])),
            ('seeds reaching the annotation end', lambda m, e: pct(m['reached_end'][0])))),
        ('quality', 'Tracing quality', (
            ('precision (on-fiber share of scored length)', lambda m, e: f"{m['precision'][0]:.3f}"),
            ('offset along sheet normal, p50 / p90', lambda m, e: f"{m['normal_p50'][0]:.1f} / {m['normal_p90'][0]:.1f} um"),
            ('offset across strip width, p50 / p90', lambda m, e: f"{m['width_p50'][0]:.1f} / {m['width_p90'][0]:.1f} um"),
            ('brief excursions (came back)', lambda m, e: mm_every(m['excursions_rate'][0])))),
    )
    for anchor, heading, block in blocks:
        out.append(f'<h2 id="{anchor}">{esc(heading)}</h2>')
        if anchor == 'distance':
            out.append('<p class="muted">"No mistake by X" counts traces that stopped or reached the annotation end as mistake-free up '
                       'to where they ended (Kaplan-Meier). "-" means fewer than 20 traces got that far. AFV seeds have only about '
                       '3.7 mm of annotation ahead, so only Paris4 reaches 10-20 mm.</p>')
        for source, entry in report['sources'].items():
            present = [l for l in labels if l in entry['runs']]
            out.append(f"<h3>{esc(source)} <span class='muted'>({esc(SOURCE_NOTES.get(source, source))}, "
                       f"{next(iter(entry['runs'].values()))['fibers']} fibers)</span></h3>")
            out.append(table([''] + present, [[esc(name)] + [esc(fn(entry['runs'][l]['metrics'], entry['runs'][l])) for l in present]
                                              for name, fn in block]))
    out.append('<h2 id="matched">Same stop rate</h2>')
    if any('curves' in e for e in report['sources'].values()):
        out.append('<p>A run that stops more makes fewer mistakes per seed. Each run is replayed at higher confidence thresholds '
                   'until it stops early as often as the most cautious run, then compared. This is approximate for flow models, '
                   'where replay overstates stops.</p>')
        rows = []
        for source, entry in report['sources'].items():
            target, values = matched(entry, [l for l in labels if l in entry['runs']])
            rows.append([esc(source), pct(target)] + ['-' if values.get(l) is None else pct(values[l]['seeds_mistake']) for l in labels])
        out.append(table(['source', 'early-stop share'] + [f'{l}: mistakes per seed' for l in labels], rows))
    if len(labels) > 1:
        out.append(f'<h2 id="paired">Paired differences against {esc(labels[0])}</h2><p class="muted">Same traces in both runs; '
                   '<span class="star">*</span> = the 95% interval excludes 0.</p>')
        keys = (('seeds_mistake', 'mistakes per seed', 100, '%'), ('losses_rate', 'mistakes per 10 mm', 1, ''),
                ('premature_stops_rate', 'early stops per 10 mm', 1, ''), ('precision', 'precision', 1, ''),
                ('fiber_coverage', 'fiber coverage', 100, '%'), ('normal_p90', 'normal offset p90 (um)', 1, ''))
        rows = []
        for source, entry in report['sources'].items():
            for label, diff in entry.get('paired', {}).items():
                cells = []
                for key, _, scale, unit in keys:
                    d = diff['metrics'][key]
                    star = '<span class="star">*</span>' if d[1] > 0 or d[2] < 0 else ''
                    cells.append(f"{scale*d[0]:+.{1 if scale == 100 else 3}f}{unit}{star}")
                rows.append([esc(source), esc(label)] + cells)
        out.append(table(['source', 'run'] + [k[1] for k in keys], rows))
    if traces:
        out.append('<h2 id="images">Images</h2><p>Flattened strip along the predicted trace (its centreline) with the annotated '
                   'fiber overlaid: green within 3 voxels of the strip plane, orange farther. The title inside each image shows '
                   'the old (legacy) scores; the caption underneath uses the metrics above.</p>')
        for source, entry in report['sources'].items():
            um = next(iter(entry['runs'].values()))['um_per_voxel'] or 1.
            out.append(f'<h3>{esc(source)}</h3>')
            for label in labels:
                for kind, t in examples(traces, source, label):
                    out.append(f'<figure><img loading="lazy" alt="{esc(kind)}" src="{embed(t["image"])}">'
                               f'<figcaption><b>{esc(label)}</b> | {caption(kind, t, um)}</figcaption></figure>')
    out.append('<p class="muted">Generated by evaluation/tube_tables.py from tube_report output. Every metric with its interval '
               'is in the accompanying CSV.</p></main></body></html>')
    return '\n'.join(out)


def metric_unit(key, rate_unit):
    if key.endswith('_rate'):
        return rate_unit
    if key.startswith('seeds_') or key in ('precision', 'fiber_coverage', 'reached_end', 'unscored_share'):
        return 'fraction'
    if key in ('distance_per_mistake', 'continued_after_mistake', 'annotation_per_seed', 'correct_mm', 'incorrect_mm'):
        return 'mm'
    if key.endswith(('_p50', '_p90')):
        return 'um'
    if key in ('verified_length', 'wrong_length', 'endpoint_overrun', 'wrong_per_verified'):
        return 'trace voxels' if key != 'wrong_per_verified' else 'ratio'
    return 'count'


def csv_rows(report):
    rows = []
    for source, entry in report['sources'].items():
        for label, run in entry['runs'].items():
            info = report['runs'][label]
            for key, value in run['metrics'].items():
                rows.append(dict(section='run', source=source, run=label, checkpoint_step=info.get('step'),
                                 threshold=info.get('confidence'), metric=key, value=value[0],
                                 low=value[1] if len(value) > 1 else '', high=value[2] if len(value) > 1 else '',
                                 unit=metric_unit(key, run['unit'])))
            for distance, value in run['mistake_free'].items():
                rows.append(dict(section='mistake_free', source=source, run=label, checkpoint_step=info.get('step'),
                                 threshold=info.get('confidence'), metric=f'no_mistake_by_{distance}',
                                 value='' if value is None else value[0], low='', high='',
                                 unit='' if value is None else f'n_at_risk={value[1]}'))
            rows.append(dict(section='run', source=source, run=label, checkpoint_step=info.get('step'),
                             threshold=info.get('confidence'), metric='mistake_at_median_mm', value=run['mistake_at_median'],
                             low='', high='', unit='mm'))
        for label, diff in entry.get('paired', {}).items():
            for key, value in diff['metrics'].items():
                rows.append(dict(section=f'paired_vs_{next(iter(entry["runs"]))}', source=source, run=label,
                                 checkpoint_step=report['runs'][label].get('step'), threshold=report['runs'][label].get('confidence'),
                                 metric=key, value=value[0], low=value[1], high=value[2], unit=metric_unit(key, 'per 10 mm')))
        for label, curve in entry.get('curves', {}).items():
            for threshold, metrics in curve:
                for key, value in metrics.items():
                    rows.append(dict(section='replayed_threshold', source=source, run=label,
                                     checkpoint_step=report['runs'][label].get('step'), threshold=threshold,
                                     metric=key, value=value, low='', high='', unit=metric_unit(key, 'per 10 mm')))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report')
    ap.add_argument('--html', help='self-contained results page (images embedded)')
    ap.add_argument('--traces', help="tube_report --traces CSV: selects the example images")
    ap.add_argument('--csv', help='every metric, long format')
    ap.add_argument('--title', default='Fiber tracing evaluation: held-out fibers')
    args = ap.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    if args.html:
        traces = list(csv.DictReader(open(args.traces))) if args.traces else None
        Path(args.html).write_text(html(report, args.title, traces))
    if args.csv:
        rows = csv_rows(report)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
