#!/usr/bin/env python3
"""Quality checks for a directory of VC3D fiber annotations.

Reads vc3d_fiber JSON files (as written by VC3D's line annotation and synced
by vc_sync.py) and reports, without changing any file:

- load: what an unmodified VC3D would refuse or rewrite on load (control
  points off the line, broken or one-sided links), via fiber_loader_check.
- fold_back: the saved line reverses direction (a hairpin), usually where a
  trace overshot a control point and came back.
- retrace: the saved line runs back over its own path for some length, so
  the fiber's length and any training target built from it count that
  stretch twice.
- jump: one step of the saved line is far longer than its usual spacing.
- duplicate: two fibers run along the same path for most of their length;
  vc_sync's hfsync skips fibers tagged `possible_duplicate` by default, and
  this lists the pairs that do not carry the tag yet.
- ends: fiber ends with no `kollesis_termination` or `break` tag (summary
  only; fiber evaluation scores length past such an end as unknown).

Distances are in the files' own coordinate units (voxels of the volume the
fibers were drawn on). Run `python fiber_qa.py --help` for options; see
volume-cartographer/docs/fiber_qa.md for what each check means.
"""

import argparse
import json
import os
import sys

import numpy as np

from fiber_loader_check import loader_issues

FIBER_TYPE = 'vc3d_fiber'
DUPLICATE_TAG = 'possible_duplicate'
END_TAGS = ('kollesis_termination', 'break')

DEFAULTS = {
    'fold_window': 40.0,      # chord length each side of a point, in units
    'fold_cos': -0.5,         # reversal when the two chords' cosine is below
    'retrace_radius': 4.0,    # another part of the line within this distance
    'retrace_cos': -0.8,      # ... running the opposite way
    'retrace_min_gap': 80.0,  # ... at least this far away along the line
    'retrace_min_length': 40.0,  # report retraced stretches this long
    'jump_factor': 8.0,       # step longer than this times the median step
    'jump_min': 20.0,         # ... and longer than this many units
    'duplicate_radius': 4.0,  # points of one fiber this close to another's
    'duplicate_fraction': 0.5,  # ... over this share of the shorter fiber
}

SEVERITY_ORDER = {'error': 0, 'warning': 1, 'info': 2}


def control_point_position(value):
    return value.get('position') if isinstance(value, dict) else value


def load_fibers(paths):
    """Return ({basename: doc}, [problems]) for every *.json under paths.

    A file that does not parse, or is not a vc3d_fiber document, becomes a
    problem instead of a fiber. Two files with the same basename in
    different directories are reported, since links name fibers by basename.
    """
    docs, where, problems = {}, {}, []
    for root in paths:
        names = ([os.path.basename(root)] if os.path.isfile(root)
                 else sorted(os.listdir(root)))
        base = os.path.dirname(root) if os.path.isfile(root) else root
        for name in names:
            if not name.endswith('.json'):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, encoding='utf-8') as handle:
                    doc = json.load(handle)
            except (OSError, ValueError) as exc:
                problems.append(finding(name, 'load', 'error',
                                        f'cannot read JSON: {exc}'))
                continue
            if not isinstance(doc, dict) or doc.get('type') != FIBER_TYPE:
                continue
            if (not isinstance(doc.get('line_points'), list) or
                    not isinstance(doc.get('control_points'), list)):
                problems.append(finding(name, 'load', 'error',
                                        'missing line_points or control_points'))
                continue
            if name in docs:
                problems.append(finding(
                    name, 'load', 'error',
                    f'same file name in {where[name]} and {base}; links '
                    'cannot tell them apart'))
                continue
            docs[name], where[name] = doc, base
    return docs, problems


def finding(name, check, severity, detail, **where):
    out = {'file': name, 'check': check, 'severity': severity,
           'detail': detail}
    out.update(where)
    return out


def line_array(doc):
    try:
        line = np.asarray(doc['line_points'], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if line.ndim != 2 or line.shape[1] != 3 or not np.isfinite(line).all():
        return None
    return line


def arclength(line):
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    return steps, np.concatenate([[0.0], np.cumsum(steps)])


def control_line_indices(doc, line):
    """Nearest line index of every control point (they lie on the line)."""
    out = []
    for value in doc['control_points']:
        point = control_point_position(value)
        try:
            point = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if point.shape != (3,) or not np.isfinite(point).all():
            out.append(None)
            continue
        out.append(int(np.argmin(np.sum((line - point) ** 2, axis=1))))
    return out


def nearest_control_point(cp_indices, index):
    best, best_gap = None, None
    for k, cp_index in enumerate(cp_indices):
        if cp_index is None:
            continue
        gap = abs(cp_index - index)
        if best_gap is None or gap < best_gap:
            best, best_gap = k, gap
    return best


def runs(mask):
    """[(start, stop)) index ranges where mask is True."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2]))


def chord_cosines(line, s, window):
    """Cosine between the chord `window` back and the chord `window` ahead
    of every line point; 1 where either chord is shorter than half the
    window (line ends, or a stretch shorter than the window)."""
    ahead = np.searchsorted(s, s + window).clip(0, len(line) - 1)
    behind = (np.searchsorted(s, s - window, side='right') - 1).clip(
        0, len(line) - 1)
    back = line - line[behind]
    fwd = line[ahead] - line
    nb = np.linalg.norm(back, axis=1)
    nf = np.linalg.norm(fwd, axis=1)
    ok = (nb >= 0.5 * window) & (nf >= 0.5 * window)
    cos = np.ones(len(line))
    cos[ok] = np.sum(back[ok] * fwd[ok], axis=1) / (nb[ok] * nf[ok])
    return cos


def unit_tangents(line):
    tangent = np.gradient(line, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.maximum(norm, 1e-12)


# Half of the 27 neighbouring grid cells (the cell itself plus 13), so each
# pair of cells is visited once.
_HALF_NEIGHBOURS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    for dz in (-1, 0, 1) if (dx, dy, dz) >= (0, 0, 0)]


def close_pairs(points, radius):
    """All index pairs (i, j), i != j, of `points` within `radius`, each
    pair once, as two arrays. Bins the points into cubes of side `radius` and
    compares each cube with itself and 13 of its neighbours."""
    empty = np.zeros(0, np.int64)
    if len(points) < 2 or radius <= 0:
        return empty, empty
    cells = np.floor(points / radius).astype(np.int64)
    cells -= cells.min(axis=0) - 1          # every neighbour index stays >= 0
    dims = cells.max(axis=0) + 2
    def code(c):
        return (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    keys = code(cells)
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    first, second = [], []
    for offset in _HALF_NEIGHBOURS:
        target = code(cells + np.asarray(offset))
        lo = np.searchsorted(sorted_keys, target, side='left')
        hi = np.searchsorted(sorted_keys, target, side='right')
        counts = hi - lo
        a = np.repeat(np.arange(len(points)), counts)
        starts = np.repeat(lo - np.cumsum(counts) + counts, counts)
        b = order[np.arange(len(a)) + starts]
        keep = a < b if offset == (0, 0, 0) else a != b
        a, b = a[keep], b[keep]
        d2 = np.sum((points[a] - points[b]) ** 2, axis=1)
        near = d2 <= radius * radius
        first.append(a[near])
        second.append(b[near])
    return np.concatenate(first), np.concatenate(second)


def shape_findings(name, doc, opts):
    """fold_back, retrace and jump findings for one fiber."""
    line = line_array(doc)
    if line is None:
        return [finding(name, 'shape', 'error',
                        'line_points are not finite 3D points')]
    if len(line) < 3:
        return []
    steps, s = arclength(line)
    cp_indices = control_line_indices(doc, line)
    out = []

    def where(index):
        index = int(index)
        return {'line_index': index, 'arclength': round(float(s[index]), 1),
                'position': [round(float(v), 2) for v in line[index]],
                'control_point': nearest_control_point(cp_indices, index)}

    cos = chord_cosines(line, s, opts['fold_window'])
    for start, stop in runs(cos < opts['fold_cos']):
        worst = start + int(np.argmin(cos[start:stop]))
        out.append(finding(
            name, 'fold_back', 'warning',
            f'line reverses direction (chord cosine {cos[worst]:.2f})',
            **where(worst)))

    median = float(np.median(steps))
    # Saved points sit about one step apart, so a path that runs over
    # another can pass up to half a step from its nearest saved point.
    a, b = close_pairs(line, opts['retrace_radius'] + 0.5 * median)
    if len(a):
        tangent = unit_tangents(line)
        keep = ((np.abs(s[a] - s[b]) >= opts['retrace_min_gap']) &
                (np.sum(tangent[a] * tangent[b], axis=1) <=
                 opts['retrace_cos']))
        mask = np.zeros(len(line), bool)
        mask[a[keep]] = True
        mask[b[keep]] = True
        stretches = []
        for start, stop in runs(mask):
            # One stretch, even where single points dip out of the radius.
            if (stretches and s[start] - s[stretches[-1][1] - 1] <=
                    2 * opts['fold_window']):
                stretches[-1][1] = stop
            else:
                stretches.append([start, stop])
        stretches = [(a0, b0) for a0, b0 in stretches
                     if s[b0 - 1] - s[a0] >= opts['retrace_min_length']]
        if stretches:
            # Both legs of a doubled path are marked, so the path traced
            # twice is half the marked length.
            marked = sum(float(s[b0 - 1] - s[a0]) for a0, b0 in stretches)
            out.append(finding(
                name, 'retrace', 'warning',
                f'about {marked / 2:.0f} units of path are traced twice '
                f'(line runs back over itself in {len(stretches)} '
                f'stretch{"es" if len(stretches) != 1 else ""})',
                length=round(marked / 2, 1),
                stretches=[[round(float(s[a0]), 1), round(float(s[b0 - 1]), 1)]
                           for a0, b0 in stretches],
                **where(stretches[0][0])))

    limit = max(opts['jump_factor'] * median, opts['jump_min'])
    for index in np.flatnonzero(steps > limit):
        out.append(finding(
            name, 'jump', 'warning',
            f'step of {steps[index]:.1f} units (median step {median:.2f})',
            **where(index)))
    return out


def end_findings(name, doc):
    cps = doc['control_points']
    out = []
    if not cps:
        return out
    for label, value in (('start', cps[0]), ('end', cps[-1])):
        tags = value.get('tags', []) if isinstance(value, dict) else []
        if not any(tag in END_TAGS for tag in tags):
            out.append(finding(name, 'ends', 'info',
                               f'{label} has no kollesis or break tag',
                               end=label))
        if len(cps) == 1:
            break
    return out


def duplicate_findings(docs, opts):
    """Pairs of fibers sharing most of their path, one finding per pair."""
    names, lines = [], []
    for name, doc in docs.items():
        line = line_array(doc)
        if line is not None and len(line) >= 2:
            names.append(name)
            lines.append(line)
    if len(names) < 2:
        return []
    # Half a typical step of slack, as for retrace.
    step = float(np.median(np.concatenate(
        [np.linalg.norm(np.diff(l, axis=0), axis=1) for l in lines])))
    radius = opts['duplicate_radius'] + 0.5 * step
    owner = np.concatenate([np.full(len(l), i) for i, l in enumerate(lines)])
    points = np.concatenate(lines)
    a, b = close_pairs(points, radius)
    cross = owner[a] != owner[b]
    a, b = a[cross], b[cross]
    # For each ordered fiber pair, how many points of the first lie near the
    # second (count each point once).
    keys = np.unique(np.stack([np.concatenate([a, b]),
                               owner[np.concatenate([b, a])]], axis=1), axis=0)
    near = {}
    for point, other in keys:
        pair = (int(owner[point]), int(other))
        near[pair] = near.get(pair, 0) + 1
    out, seen = [], set()
    for (i, j), count in sorted(near.items()):
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        frac_i = count / len(lines[i])
        frac_j = near.get((j, i), 0) / len(lines[j])
        if max(frac_i, frac_j) < opts['duplicate_fraction']:
            continue
        tagged = [n for n in (names[i], names[j])
                  if DUPLICATE_TAG in (docs[n].get('tags') or [])]
        shorter = names[i] if len(lines[i]) <= len(lines[j]) else names[j]
        out.append(finding(
            shorter, 'duplicate', 'info' if tagged else 'warning',
            f'{max(frac_i, frac_j):.0%} of the shorter fiber lies within '
            f'{radius:.1f} units of the other'
            + (f' (already tagged: {", ".join(tagged)})' if tagged else ''),
            other=names[j] if shorter == names[i] else names[i],
            shared=[round(frac_i, 3), round(frac_j, 3)],
            files=[names[i], names[j]]))
    return out


def run_checks(docs, checks, opts):
    out = []
    if 'load' in checks:
        for issue in loader_issues(docs):
            name, _, detail = issue.partition(': ')
            out.append(finding(name, 'load', 'error', detail))
    for name, doc in docs.items():
        if 'shape' in checks:
            out.extend(shape_findings(name, doc, opts))
        if 'ends' in checks:
            out.extend(end_findings(name, doc))
    if 'duplicate' in checks:
        out.extend(duplicate_findings(docs, opts))
    out.sort(key=lambda f: (SEVERITY_ORDER[f['severity']], f['check'],
                            f['file'], f.get('line_index', -1)))
    return out


def summarize(docs, findings):
    by_check = {}
    for f in findings:
        counts = by_check.setdefault(f['check'], {'findings': 0, 'files': set()})
        counts['findings'] += 1
        counts['files'].add(f['file'])
    return {
        'fibers': len(docs),
        'checks': {check: {'findings': c['findings'], 'fibers': len(c['files'])}
                   for check, c in sorted(by_check.items())},
    }


def print_report(summary, findings, show_info, out=None):
    out = out or sys.stdout
    print(f"{summary['fibers']} fibers", file=out)
    for check, counts in summary['checks'].items():
        print(f"  {check}: {counts['findings']} findings in "
              f"{counts['fibers']} fibers", file=out)
    for f in findings:
        if f['severity'] == 'info' and not show_info:
            continue
        place = ''
        if 'line_index' in f:
            place = (f" at line point {f['line_index']} (arclength "
                     f"{f['arclength']}, near control point "
                     f"{f['control_point']}) {f['position']}")
        if f['check'] == 'duplicate':
            place = f" with {f['other']}"
        print(f"{f['severity'].upper():7} {f['check']:9} {f['file']}: "
              f"{f['detail']}{place}", file=out)


ALL_CHECKS = ('load', 'shape', 'duplicate', 'ends')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Check VC3D fiber annotation files for load errors, '
                    'fold-backs, retraced stretches, jumps and duplicates. '
                    'Reads only; never changes a file.')
    parser.add_argument('paths', nargs='+',
                        help='fiber directories (or single fiber JSON files)')
    parser.add_argument('--checks', default=','.join(ALL_CHECKS),
                        help='comma-separated subset of: '
                             + ', '.join(ALL_CHECKS))
    parser.add_argument('--json', dest='json_out',
                        help='write the summary and every finding as JSON '
                             'here ("-" for stdout)')
    parser.add_argument('--show-info', action='store_true',
                        help='also list info findings (untagged ends, '
                             'already tagged duplicates)')
    parser.add_argument('--fail-on', choices=('error', 'warning', 'never'),
                        default='error',
                        help='exit 1 when a finding at this severity or '
                             'worse exists (default: error)')
    for key, value in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), type=float,
                            default=value, dest=key,
                            help=f'(default {value:g})')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checks = [c.strip() for c in args.checks.split(',') if c.strip()]
    unknown = sorted(set(checks) - set(ALL_CHECKS))
    if unknown:
        print(f"unknown check(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    opts = {key: getattr(args, key) for key in DEFAULTS}
    docs, problems = load_fibers(args.paths)
    findings = problems + run_checks(docs, checks, opts)
    summary = summarize(docs, findings)
    if args.json_out:
        payload = {'summary': summary, 'options': opts, 'checks': checks,
                   'findings': findings}
        if args.json_out == '-':
            json.dump(payload, sys.stdout, indent=1)
            print()
        else:
            with open(args.json_out, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=1)
    if args.json_out != '-':
        print_report(summary, findings, args.show_info)
    if args.fail_on == 'never':
        return 0
    limit = SEVERITY_ORDER[args.fail_on]
    return 1 if any(SEVERITY_ORDER[f['severity']] <= limit
                    for f in findings) else 0


if __name__ == '__main__':
    sys.exit(main())
