#!/usr/bin/env python3
"""Render an ink_metric output folder produced by get_ink_metrics.py into a single
self-contained report.html, written alongside metrics.json.

The report shows the run's headline numbers, a per-strip score table, and one
figure per strip: the ink-prediction overlay annotated with the detected column
runs (colored by width conformity against the column-width prior) and gap
cleanliness, above the pooled ink-density profile the column detector worked on.
Everything is embedded as data URIs so the one file can be shared as-is.

It needs only what get_ink_metrics.py persists -- metrics.json (including the
per-strip `columns` detection detail) and the predictions/<stem>_overlay.jpg
images -- so it can rebuild the report at any time without the probability maps:

    make_ink_report.py /path/to/ink_metric

get_ink_metrics.py also calls build_report() itself at the end of a run.
"""

import os
import io
import json
import base64
import argparse

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Score coloring: figure annotations grade at 0.8/0.4, the coarser table
# chips/bars at 0.7/0.4 (same grading the score table has always used).
FIG_COLORS = ('#1a9641', '#ff8c00', '#d7191c')


def fig_color(v):
    return FIG_COLORS[0] if v >= 0.8 else FIG_COLORS[1] if v >= 0.4 else FIG_COLORS[2]


def chip_class(v):
    return 'good' if v >= 0.7 else 'mid' if v >= 0.4 else 'bad'


def chip(v):
    return f'<span class="chip {chip_class(v)}">{v:.2f}</span>'


def bar(v):
    return (f'<div class="bar"><div class="bar-fill {chip_class(v)}" '
            f'style="width:{max(2, v * 100):.0f}%"></div></div>')


def strip_figure(row, pred_dir, max_w=1680, quality=82):
    """One strip's figure (as a jpeg data URI): annotated overlay + density
    profile. Returns None if the strip's overlay image is missing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    overlay_path = os.path.join(pred_dir, f"{row['strip']}_overlay.jpg")
    if not os.path.exists(overlay_path):
        return None
    img = Image.open(overlay_path)
    w0, h0 = img.size
    ds = max(1, round(w0 / max_w))
    img = img.resize((w0 // ds, h0 // ds), Image.BILINEAR)

    det = row.get('columns') or {}
    runs, gaps = det.get('runs', []), det.get('gaps', [])

    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(16, 7.2), dpi=100, sharex=True,
        gridspec_kw={'height_ratios': [3.2, 1.2], 'hspace': 0.06})
    ax.imshow(np.asarray(img), extent=[0, w0, h0, 0], aspect='auto')
    for r in runs:
        color = fig_color(r['width_score']) if r['interior'] else '#888888'
        for x in (r['start'], r['end']):
            ax.axvline(x, color=color, lw=2)
        label = (f"{r['width']}px\n{r['width_score']:.2f}" if r['interior']
                 else f"{r['width']}px\n(edge)")
        ax.text((r['start'] + r['end']) / 2, -0.02 * h0, label, ha='center',
                va='bottom', fontsize=8, color=color, fontweight='bold')
    for g in gaps:
        color = fig_color(g['cleanliness'])
        ax.add_patch(Rectangle((g['start'], 0), g['end'] - g['start'], h0,
                               color=color, alpha=0.12, lw=0))
        ax.text((g['start'] + g['end']) / 2, 0.985 * h0,
                f"gap\n{g['cleanliness'] * 100:.0f}%",
                ha='center', va='bottom', fontsize=7, color=color)
    ax.set_ylim(h0, -0.12 * h0)
    ax.set_yticks([])
    ax.set_title(
        f"{row['strip']}   col_score={row['col_score']:.2f} "
        f"(width-conformity {row['col_width_conformity']:.2f} x "
        f"gap-contrast {row['col_gap_contrast']:.2f}, {row['col_count']} cols, "
        f"median {row['col_median_width_px']}px)   "
        f"line_score={row['line_score']:.2f} @ {row['line_median_pitch_px']}px pitch "
        f"({row['line_gap_count']} gaps)", fontsize=10)

    profile = np.asarray(det.get('profile_ds', []), np.float32)
    if profile.size:
        xs = np.arange(profile.size) * det.get('profile_ds_factor', 1)
        axp.fill_between(xs, profile, color='#bbbbbb', lw=0, label='ink density p(x)')
        if det.get('threshold'):
            axp.axhline(det['threshold'], color='#2b83ba', lw=1, ls='--',
                        label='column threshold')
        for r in runs:
            axp.axvspan(r['start'], r['end'], color='#2b83ba', alpha=0.10, lw=0)
        axp.set_ylim(0, max(1e-3, float(profile.max()) * 1.05))
        axp.legend(loc='upper right', fontsize=7, framealpha=0.9)
    axp.set_xlim(0, w0)
    axp.set_xlabel('x (strip px)')

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', bbox_inches='tight', pil_kwargs={'quality': quality})
    plt.close(fig)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


CSS = '''
:root {
  --bg: #faf8f4; --panel: #ffffff; --text: #1c1917; --muted: #6f675e;
  --line: #e4ddd2; --accent: #a83226;
  --good: #1a9641; --mid: #c97a10; --bad: #c0392b;
  --chip-good-bg: #e3f2e5; --chip-mid-bg: #f8ecd9; --chip-bad-bg: #f9e2de;
}
@media (prefers-color-scheme: dark) { :root {
  --bg: #171412; --panel: #201c19; --text: #ece6dd; --muted: #a49a8d;
  --line: #38322c; --accent: #e0654f;
  --good: #52c06d; --mid: #e0a04a; --bad: #e26a5a;
  --chip-good-bg: #1e3324; --chip-mid-bg: #382b18; --chip-bad-bg: #3b201c;
} }
body { background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 2rem clamp(1rem, 4vw, 3rem) 4rem; }
h1, h2, h3 { font-family: "Iowan Old Style", Palatino, Georgia, serif; text-wrap: balance; }
h1 { font-size: 1.9rem; margin: 0 0 .25rem; }
h1 em { color: var(--accent); font-style: normal; }
h2 { font-size: 1.3rem; margin: 2.5rem 0 .75rem; }
h3 { font-size: 1.05rem; margin: 0 0 .2rem; display: flex; gap: 1rem;
  align-items: baseline; flex-wrap: wrap; }
h3 .scores { font-family: system-ui, sans-serif; font-size: .85rem; color: var(--muted); }
.sub { color: var(--muted); max-width: 80ch; margin: 0 0 1.5rem; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .92em; }
.stats { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.25rem 0 2rem; }
.stat { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
  padding: .7rem 1.1rem; min-width: 10rem; }
.stat .v { font-size: 1.5rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.stat .k { font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px;
  background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;
  font-size: .88rem; }
th, td { padding: .45rem .7rem; text-align: right; white-space: nowrap; }
th { font-size: .7rem; letter-spacing: .05em; text-transform: uppercase; color: var(--muted);
  border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--panel); }
td:first-child, th:first-child { text-align: left; }
tbody tr { border-bottom: 1px solid var(--line); cursor: pointer; }
tbody tr:hover { background: rgba(168, 50, 38, 0.06); }
td a { color: var(--accent); text-decoration: none; font-weight: 600; }
td a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.chip { display: inline-block; min-width: 2.6em; text-align: center; padding: .05rem .4rem;
  border-radius: 999px; font-weight: 650; font-size: .82rem; }
.chip.good { color: var(--good); background: var(--chip-good-bg); }
.chip.mid  { color: var(--mid);  background: var(--chip-mid-bg); }
.chip.bad  { color: var(--bad);  background: var(--chip-bad-bg); }
.bar { height: 4px; background: var(--line); border-radius: 2px; margin-top: .3rem; min-width: 5.5rem; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-fill.good { background: var(--good); }
.bar-fill.mid { background: var(--mid); }
.bar-fill.bad { background: var(--bad); }
section { margin: 2.2rem 0; scroll-margin-top: 1rem; }
.strip-note { color: var(--muted); font-size: .85rem; margin: 0 0 .5rem; max-width: 100ch; }
.figwrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
.figwrap img { display: block; min-width: 1100px; width: 100%; height: auto; }
'''


def build_report(out_dir):
    """Build report.html inside out_dir (a get_ink_metrics.py output folder
    containing metrics.json + predictions/). Returns the report path."""
    out_dir = os.path.abspath(out_dir)
    with open(os.path.join(out_dir, 'metrics.json')) as f:
        data = json.load(f)
    summary, rows = data['summary'], data['strips']
    pred_dir = os.path.join(out_dir, 'predictions')

    n = len(rows)
    med_pitch = sorted(r['line_median_pitch_px'] for r in rows)[n // 2] if n else 0
    col_lo = summary['col_width_px'] * (1 - summary['col_width_tol'])
    col_hi = summary['col_width_px'] * (1 + summary['col_width_tol'])
    band = summary['line_pitch_px']

    trs, sections = [], []
    for r in rows:
        trs.append(f'''<tr onclick="location.hash='{r['strip']}'">
<td class="mono"><a href="#{r['strip']}">{r['strip']}</a></td>
<td>{r['fg_fraction'] * 100:.2f}%</td>
<td>{chip(r['col_score'])}{bar(r['col_score'])}</td>
<td>{r['col_width_conformity']:.2f}</td>
<td>{r['col_gap_contrast']:.2f}</td>
<td>{r['col_count']}</td>
<td>{r['col_median_width_px']}</td>
<td>{chip(r['line_score'])}{bar(r['line_score'])}</td>
<td>{r['line_median_pitch_px']}</td>
<td>{r['line_gap_count']}</td>
</tr>''')
        uri = strip_figure(r, pred_dir)
        figure = (f'<div class="figwrap"><img src="{uri}" '
                  f'alt="{r["strip"]} column detection figure" loading="lazy"></div>'
                  if uri else '<p class="strip-note">(overlay image missing)</p>')
        sections.append(f'''<section id="{r['strip']}">
<h3><span class="mono">{r['strip']}</span>
<span class="scores">column-ness {chip(r['col_score'])} &nbsp; line-ness {chip(r['line_score'])}</span></h3>
<p class="strip-note">{r['col_count']} column runs, median width {r['col_median_width_px']} px
(conformity {r['col_width_conformity']:.2f}, gap contrast {r['col_gap_contrast']:.2f}) &middot;
line pitch {r['line_median_pitch_px']} px over {r['line_gap_count']} gaps &middot;
ink {r['fg_fraction'] * 100:.2f}%</p>
{figure}
</section>''')

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ink metrics — {os.path.basename(os.path.dirname(summary['ink_dir'])) or summary['ink_dir']}</title>
<style>{CSS}</style>
</head>
<body>
<h1>Ink layout metrics <em>— column-ness &amp; line-ness per strip</em></h1>
<p class="sub"><span class="mono">{summary['ink_dir']}</span><br>
model <span class="mono">{summary['model']}</span>, folds {summary['folds']},
fg threshold {summary['fg_threshold']:g}. Each figure shows the strip with predicted ink in red,
detected column spans and transitions as vertical lines (colored by width conformity against the
{summary['col_width_px']:g}&thinsp;px&nbsp;&plusmn;&nbsp;{summary['col_width_tol'] * 100:g}% prior,
i.e. {col_lo:g}&ndash;{col_hi:g}&thinsp;px), per-gap cleanliness, and the vertically pooled
ink-density profile the detector works on. Line-ness scores the text-line pitch against the
{band[0]:g}&ndash;{band[1]:g}&thinsp;px band.</p>

<div class="stats">
<div class="stat"><div class="v">{summary['overall_fg_fraction'] * 100:.2f}%</div>
<div class="k">overall ink fraction</div></div>
<div class="stat"><div class="v">{summary['overall_column_score']:.2f}</div>
<div class="k">overall column-ness</div></div>
<div class="stat"><div class="v">{summary['overall_line_score']:.2f}</div>
<div class="k">overall line-ness</div></div>
<div class="stat"><div class="v">{med_pitch} px</div><div class="k">median line pitch</div></div>
<div class="stat"><div class="v">{n}</div><div class="k">strips scored</div></div>
</div>

<h2>Score table</h2>
<div class="tablewrap"><table>
<thead><tr><th>strip</th><th>ink</th><th>column-ness</th><th>width conf.</th>
<th>gap contrast</th><th>cols</th><th>med. width px</th><th>line-ness</th>
<th>pitch px</th><th>gaps</th></tr></thead>
<tbody>{''.join(trs)}</tbody>
</table></div>

<h2>Per-strip detection</h2>
{''.join(sections)}
</body>
</html>
'''

    report_path = os.path.join(out_dir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)
    return report_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir', help='get_ink_metrics.py output folder (contains metrics.json)')
    args = ap.parse_args()
    print(build_report(args.out_dir))


if __name__ == '__main__':
    main()
