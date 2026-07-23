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
import glob
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


# The whole-scroll strip is far too wide to read in one figure, so each strip is
# displayed as a vertical stack of bands ROW_PX strip-pixels wide, each rendered at
# up to ROW_MAX_W px. MAX_ROWS caps the stack (widening the bands) for huge strips.
ROW_PX = 8192
ROW_MAX_W = 2200
MAX_ROWS = 48


def overlay_tiles(pred_dir, stem):
    """Return the strip's overlay as an ordered list of (x_offset, PIL image):
    a single <stem>_overlay.jpg at x=0, or the get_ink_metrics-chopped
    <stem>_overlay.NNN.jpg tiles with their cumulative x offsets. Empty if none."""
    single = os.path.join(pred_dir, f'{stem}_overlay.jpg')
    if os.path.exists(single):
        return [(0, Image.open(single))]
    out, x = [], 0
    for p in sorted(glob.glob(os.path.join(pred_dir, f'{stem}_overlay.[0-9]*.jpg'))):
        im = Image.open(p)
        out.append((x, im))
        x += im.width
    return out


def overlay_window(tiles, xa, xb, max_w):
    """Crop the overlay (from overlay_tiles) to strip x-range [xa, xb) -- across
    tile boundaries as needed -- onto a fixed [xb-xa]-wide canvas (any part past
    the strip's real width stays black), downsampled to <= max_w px wide. So every
    row is the same size regardless of how much real strip it covers."""
    if not tiles:
        return None
    h = max(t.height for _, t in tiles)
    win = Image.new('RGB', (xb - xa, h))
    for x0, im in tiles:
        a, b = max(xa, x0), min(xb, x0 + im.width)
        if a < b:
            win.paste(im.crop((a - x0, 0, b - x0, im.height)), (a - xa, 0))
    ds = max(1, round(win.width / max_w))
    return win.resize((win.width // ds, win.height // ds), Image.BILINEAR) if ds > 1 else win


def strip_ranges(row):
    """The x-ranges to chop the strip into for display, one per row. Uses the
    per-slice ranges get_ink_metrics stored (so the display rows line up with the
    score table's slices); falls back to ROW_PX-wide bands for older metrics.json
    without a `slices` list, widening the bands past MAX_ROWS if needed."""
    slices = row.get('slices')
    if slices:
        return [(int(s['x0']), int(s['x1'])) for s in slices]
    full_w = row.get('width') or 0
    row_px = max(ROW_PX, -(-full_w // MAX_ROWS)) if full_w else ROW_PX
    n_rows = max(1, -(-full_w // row_px)) if full_w else 1
    return [(i * row_px, min(full_w, (i + 1) * row_px)) for i in range(n_rows)]


def strip_row_figures(row, pred_dir, ranges, max_w=ROW_MAX_W, quality=82):
    """Render the strip as a top-to-bottom stack of display rows, one per (x0, x1)
    in `ranges`, each drawn at readable scale with the column annotations and
    density profile falling in its x-window. All rows share one band width (the
    widest range, padded with black) so they display at a uniform size. Returns a
    list of (x0, x1, jpeg data URI) aligned 1:1 with `ranges`; empty if the overlay
    is missing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tiles = overlay_tiles(pred_dir, row['strip'])
    if not tiles or not ranges:
        return []
    full_w = row.get('width') or sum(t.width for _, t in tiles)
    full_h = row.get('height') or max(t.height for _, t in tiles)
    det = row.get('columns') or {}
    runs, gaps = det.get('runs', []), det.get('gaps', [])
    profile = np.asarray(det.get('profile_ds', []), np.float32)
    pf = det.get('profile_ds_factor', 1)
    prof_ymax = max(1e-3, float(profile.max()) * 1.05) if profile.size else 1e-3

    band_w = max(x1 - x0 for x0, x1 in ranges)
    n_rows = len(ranges)
    aspect = full_h / band_w
    figw = 18.0

    figures = []
    for i, (x0, x1) in enumerate(ranges):
        xa, xb = x0, x0 + band_w  # uniform-width window (padded past the slice/strip)
        img = overlay_window(tiles, xa, xb, max_w)
        if img is None:
            continue
        fig, (ax, axp) = plt.subplots(
            2, 1, figsize=(figw, figw * aspect * 1.45 + 0.9), dpi=100, sharex=True,
            gridspec_kw={'height_ratios': [3.2, 1.2], 'hspace': 0.08})
        ax.imshow(np.asarray(img), extent=[xa, xb, full_h, 0], aspect='auto')
        for r in runs:
            if r['end'] < xa or r['start'] > xb:
                continue
            color = fig_color(r['width_score']) if r['interior'] else '#888888'
            for x in (r['start'], r['end']):
                if xa <= x <= xb:
                    ax.axvline(x, color=color, lw=2)
            cx = (r['start'] + r['end']) / 2
            if xa <= cx <= xb:
                label = (f"{r['width']}px\n{r['width_score']:.2f}" if r['interior']
                         else f"{r['width']}px\n(edge)")
                ax.text(cx, -0.02 * full_h, label, ha='center', va='bottom',
                        fontsize=8, color=color, fontweight='bold')
        for g in gaps:
            if g['end'] < xa or g['start'] > xb:
                continue
            color = fig_color(g['cleanliness'])
            ax.add_patch(Rectangle((g['start'], 0), g['end'] - g['start'], full_h,
                                   color=color, alpha=0.12, lw=0))
            cx = (g['start'] + g['end']) / 2
            if xa <= cx <= xb:
                ax.text(cx, 0.985 * full_h, f"gap\n{g['cleanliness'] * 100:.0f}%",
                        ha='center', va='bottom', fontsize=7, color=color)
        ax.set_ylim(full_h, -0.12 * full_h)
        ax.set_yticks([])
        ax.set_title(f"{row['strip']}  x {x0:,}-{x1:,} px  "
                     f"(slice {i + 1}/{n_rows})", fontsize=9, loc='left')

        if profile.size:
            xs = np.arange(profile.size) * pf
            axp.fill_between(xs, profile, color='#bbbbbb', lw=0, label='ink density p(x)')
            if det.get('threshold'):
                axp.axhline(det['threshold'], color='#2b83ba', lw=1, ls='--',
                            label='column threshold')
            for r in runs:
                axp.axvspan(r['start'], r['end'], color='#2b83ba', alpha=0.10, lw=0)
            axp.set_ylim(0, prof_ymax)
            if i == 0:
                axp.legend(loc='upper right', fontsize=7, framealpha=0.9)
        axp.set_xlim(xa, xb)
        axp.set_xlabel('x (strip px)')

        buf = io.BytesIO()
        fig.savefig(buf, format='jpg', bbox_inches='tight', pil_kwargs={'quality': quality})
        plt.close(fig)
        figures.append((x0, x1,
                        'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()))
    return figures


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
.xr { color: var(--muted); font-size: .8em; font-weight: 400; }
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
    col_lo = summary['col_width_px'] * (1 - summary['col_width_tol'])
    col_hi = summary['col_width_px'] * (1 + summary['col_width_tol'])
    band = summary['line_pitch_px']
    multi = n > 1  # more than one render: prefix slice labels with the render name

    def table_row(label, anchor, m):
        """One score-table row for a slice's metrics dict m."""
        return f'''<tr onclick="location.hash='{anchor}'">
<td class="mono"><a href="#{anchor}">{label}</a></td>
<td>{m['fg_fraction'] * 100:.2f}%</td>
<td>{chip(m['col_score'])}{bar(m['col_score'])}</td>
<td>{m['col_width_conformity']:.2f}</td>
<td>{m['col_gap_contrast']:.2f}</td>
<td>{m['col_count']}</td>
<td>{m['col_median_width_px']}</td>
<td>{chip(m['line_score'])}{bar(m['line_score'])}</td>
<td>{m['line_median_pitch_px']}</td>
<td>{m['line_gap_count']}</td>
</tr>'''

    # Flat list of slices: the render is just chopped into fixed-width slices, each
    # scored independently -- one table row and one figure per slice, no render level.
    trs, sections = [], []
    total_slices = 0
    for r in rows:
        strip = r['strip']
        ranges = strip_ranges(r)
        slices = r.get('slices') or [{**r, 'x0': a, 'x1': b} for a, b in ranges]
        total_slices += len(slices)
        uri_by_x0 = {x0: uri for x0, _x1, uri in strip_row_figures(r, pred_dir, ranges)}
        for i, s in enumerate(slices):
            anchor = f'{strip}__s{i}'
            lbl = (f'{strip} &middot; ' if multi else '') + f'slice {i + 1}/{len(slices)}'
            lbl += (f' <span class="xr">= {s["x0"] // 1000}&ndash;{s["x1"] // 1000}k px</span>')
            trs.append(table_row(lbl, anchor, s))
            uri = uri_by_x0.get(s['x0'])
            fig_html = (f'<div class="figwrap"><img src="{uri}" '
                        f'alt="{strip} slice {i + 1}" loading="lazy"></div>'
                        if uri else '<p class="strip-note">(overlay image missing)</p>')
            cap = f'{strip} &middot; ' if multi else ''
            sections.append(f'''<section id="{anchor}">
<h3><span class="mono">{cap}slice {i + 1}/{len(slices)}</span>
<span class="scores">= {s['x0']:,}&ndash;{s['x1']:,} px &nbsp; ink {s['fg_fraction'] * 100:.1f}%
&nbsp; column-ness {chip(s['col_score'])} &nbsp; line-ness {chip(s['line_score'])}</span></h3>
{fig_html}
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
<h1>Ink layout metrics <em>— column-ness &amp; line-ness</em></h1>
<p class="sub"><span class="mono">{summary['ink_dir']}</span><br>
model <span class="mono">{summary['model']}</span>, folds {summary['folds']},
fg threshold {summary['fg_threshold']:g}. The render is chopped into fixed-width slices, each scored
independently (below and in the table); every slice shows predicted ink in red,
detected column spans and transitions as vertical lines (colored by width conformity against the
{summary['col_width_px']:g}&thinsp;px&nbsp;&plusmn;&nbsp;{summary['col_width_tol'] * 100:g}% prior,
i.e. {col_lo:g}&ndash;{col_hi:g}&thinsp;px), per-gap cleanliness, and the vertically pooled
ink-density profile the detector works on. Line-ness scores the text-line pitch against the
{band[0]:g}&ndash;{band[1]:g}&thinsp;px band.</p>

<div class="stats">
<div class="stat"><div class="v">{summary['total_fg_pixels']:,}</div>
<div class="k">ink coverage (px)</div></div>
<div class="stat"><div class="v">{summary['overall_fg_fraction'] * 100:.2f}%</div>
<div class="k">overall ink fraction</div></div>
<div class="stat"><div class="v">{summary['overall_column_score']:.2f}</div>
<div class="k">overall column-ness</div></div>
<div class="stat"><div class="v">{summary['overall_line_score']:.2f}</div>
<div class="k">overall line-ness</div></div>
</div>

<h2>Score table</h2>
<div class="tablewrap"><table>
<thead><tr><th>slice</th><th>ink</th><th>column-ness</th><th>width conf.</th>
<th>gap contrast</th><th>cols</th><th>med. width px</th><th>line-ness</th>
<th>pitch px</th><th>gaps</th></tr></thead>
<tbody>{''.join(trs)}</tbody>
</table></div>

<h2>Detection detail</h2>
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
