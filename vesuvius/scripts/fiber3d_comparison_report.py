"""Read-only equal-scroll reporting for the approved tracing experiment runner."""
import json
import math
from pathlib import Path

from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import complete_report

SCROLLS = ('PHerc0139', 'PHerc0332', 'PHerc1299', 'PHerc1451', 'PHercParis4')
METRICS = [('failures', 'Restarts ↓', 0, False),
           ('mean_segment_length_mm', 'Mean ↑', 3, True),
           ('median_segment_length_mm', 'Median ↑', 3, True),
           ('p05_segment_length_mm', 'P05 ↑', 4, True),
           ('p95_segment_length_mm', 'P95 ↑', 3, True),
           ('zero_length_segments', 'Zero ↓', 0, False)]


def equal_scroll_score(metrics):
    if set(metrics['scrolls']) != set(SCROLLS):
        raise ValueError('comparison requires exactly the same five scrolls')
    values = [metrics['scrolls'][s]['mean_segment_length_mm'] for s in SCROLLS]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('invalid per-scroll mean')
    return math.fsum(values) / len(SCROLLS)


def comparison_markdown(rows):
    if not rows:
        raise ValueError('empty comparison')
    scores = [equal_scroll_score(row['metrics']) for row in rows]
    for row in rows:
        for scroll in SCROLLS:
            value = row['metrics']['scrolls'][scroll]
            ref = rows[0]['metrics']['scrolls'][scroll]
            if value['directions'] != ref['directions'] or not math.isclose(
                    value['directed_reference_mm'], ref['directed_reference_mm'], rel_tol=1e-9):
                raise ValueError('evaluation coverage differs between models')
            expected = value['directed_reference_mm'] / (value['failures'] + value['directions'])
            if not math.isclose(value['mean_segment_length_mm'], expected, rel_tol=1e-9):
                raise ValueError('inconsistent per-scroll mean')

    def cell(value, best, digits):
        text = f'{value:.{digits}f}'
        return f'**{text}**' if math.isclose(value, best, rel_tol=0, abs_tol=1e-10) else text

    lines = ['# Equal-scroll tracing comparison', '',
             'Overall score: arithmetic mean of the five per-scroll means, **20% each**.',
             'Lengths are mm. Best/tied values are bold; old pooled overall scores are omitted.',
             'Classic is longer-trained; NML/JSON/Combined raw/EMA rows use the600k cutoff.',
             'Prior EMA series and new EMA rows use their separately recorded later-run checkpoints.',
             'New EMA rows are frozen from the later training run; checkpoint ages are listed below.', '',
             '## Overall', '', '| Model | Equal-scroll score ↑ |', '|---|---:|']
    for row, score in zip(rows, scores):
        lines.append(f"| {row['name']} | {cell(score, max(scores), 3)} |")
    for scroll in SCROLLS:
        ref = rows[0]['metrics']['scrolls'][scroll]
        lines += ['', f"## {scroll} — {ref['directions'] // 2} fibers", '',
                  f"{ref['directions']} directions; {ref['directed_reference_mm']:.3f}mm directed reference.",
                  'Total segments = restarts + directions.', '',
                  '| Model | ' + ' | '.join(m[1] for m in METRICS) + ' |',
                  '|---|' + '|'.join('---:' for _ in METRICS) + '|']
        for row in rows:
            values = []
            for key, _, digits, higher in METRICS:
                best = (max if higher else min)(r['metrics']['scrolls'][scroll][key] for r in rows)
                values.append(cell(row['metrics']['scrolls'][scroll][key], best, digits))
            lines.append(f"| {row['name']} | " + ' | '.join(values) + ' |')
    lines += ['', '## New checkpoint provenance', '',
              '| Label | Step | Samples | Decay | Test loss |', '|---|---:|---:|---:|---:|']
    for row in rows:
        if 'selection' in row:
            s = row['selection']['selected']
            lines.append(f"| {row['name']} | {s['step']} | {s['samples']} | {s['decay']} | {s['metric']:.6f} |")
    lines += ['', '## Evaluation sources', '']
    for row in rows:
        lines.append(f"- {row['name']}: `{row['output']}`")
        if 'selection' in row:
            lines.append(f"  SHA256 `{row['selection']['sha256']}`.")
    lines += ['', 'Small-scroll support is limited (0332:3fibers;1451:2fibers).',
              'No statistical significance claim. Training is not stopped or restarted by evaluation.', '']
    return '\n'.join(lines)


def series_report(state):
    rows = []
    for baseline in state['settings']['baselines']:
        summary = json.loads((Path(baseline['output']) / 'full/summary.json').read_text())
        rows.append(dict(baseline, metrics=complete_report(summary)))
    for entry in state['entries']:
        if entry['status'] == 'done':
            rows.append(dict(name='New ' + entry['name'], output=entry['output'],
                             metrics=entry['metrics'], selection=entry['selection']))
    return comparison_markdown(rows)
