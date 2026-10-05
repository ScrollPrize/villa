#!/usr/bin/env python3
"""Create same-sheet fiber links from horizontals that complete a revolution.

Papyrus sheets carry their horizontal fibers on one face and their vertical
fibers on the other. When a horizontal fiber winds around the umbilicus more
than once, the radial gap between two consecutive revolutions is one winding of
the sheet, and anything crossing that gap is by construction the back of the
inner revolution: the same sheet, one layer behind the horizontal. This tool
finds vertical fibers that sit between consecutive revolutions of a horizontal
fiber and links each one to the horizontal at the crossing in front of it.

Detection, in fitter coordinates (fiber JSON coordinates times
``--coordinate-scale``), using the dataset umbilicus for theta and radius:

1. Every horizontal fiber whose unwrapped theta spans at least one revolution
   is binned by unwrapped theta (``--theta-bin`` radians); each bin keeps the
   mean z and radius of the fiber there.
2. For every vertical fiber and every revolution ``k`` of such a horizontal,
   the vertical point whose theta (plus ``2 pi k``) lands in an occupied bin and
   whose z is within ``--z-tolerance`` of the bin is a crossing; it yields the
   vertical's radius and the horizontal's radius at that place.
3. For consecutive revolutions ``k`` and ``k + 1`` whose radial gap is between
   ``--min-gap`` and ``--max-gap`` voxels, the vertical is bracketed when it
   lies outside the inner revolution and inside the outer one at the
   respective crossings.
4. The link joins the vertical's crossing point with the inner revolution to
   the nearest horizontal line point of that revolution.

Links are stored the way VC3D stores them: the two endpoints must be control
points, so each chosen line point is promoted to a control point (inheriting
the ``segment_to_next`` block of the segment it splits), and a reciprocal
``branches`` entry is written into both files. Promoting a control point
renumbers the later ones, so this tool also renumbers the affected indices in
the file's own branches and in every other file that links into it.

By default this is a dry run that only reports the planned links. Pass
``--apply`` to write; every touched file is copied to ``--backup-dir`` first.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import shutil
import sys
from dataclasses import dataclass, field

import click
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spiral_helpers import classify_fiber_hv  # noqa: E402
from umbilicus import json_umbilicus_z_to_yx  # noqa: E402

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Fiber:
    path: str
    data: dict
    # Full polyline in file coordinates (x, y, z) and fitter coordinates
    # (z, y, x); indices below refer to this full polyline.
    line_xyz: np.ndarray
    zyx: np.ndarray
    control_line_indices: np.ndarray  # control point ordinal -> line index
    tag: str | None
    span: tuple[int, int]  # inclusive line-index span between first/last control point
    z: np.ndarray = field(default=None)
    r: np.ndarray = field(default=None)
    theta_u: np.ndarray = field(default=None)  # unwrapped theta over the span
    duplicate_of: str | None = None  # basename of the canonical copy, if this is a duplicate

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)

    def branch_pairs(self) -> set[tuple[str, str]]:
        return {
            tuple(sorted((self.basename, os.path.basename(br['branch_file']))))
            for br in (self.data.get('branches') or [])
            if br.get('branch_file')
        }


def _control_positions(data: dict) -> np.ndarray:
    controls = data.get('control_points') or []
    if not controls:
        return np.zeros((0, 3), dtype=np.float64)
    if isinstance(controls[0], dict):
        return np.asarray([c['position'] for c in controls], dtype=np.float64)
    return np.asarray(controls, dtype=np.float64)


def load_fiber(path: str, umbilicus, *, coordinate_scale: float,
               min_z_fraction: float, min_auto_certainty: float) -> Fiber | None:
    with open(path, 'rt') as fp:
        data = json.load(fp)
    if data.get('type') != 'vc3d_fiber':
        return None
    line_xyz = np.asarray(data.get('line_points') or [], dtype=np.float64)
    controls = _control_positions(data)
    if line_xyz.ndim != 2 or len(line_xyz) < 3 or len(controls) < 2:
        return None
    squared = ((line_xyz[None, :, :] - controls[:, None, :]) ** 2).sum(axis=-1)
    control_line_indices = squared.argmin(axis=1)
    if not np.all(np.diff(control_line_indices) > 0):
        print(f'WARNING: {os.path.basename(path)}: control points are not an '
              'ordered subset of line_points; skipping')
        return None
    span = (int(control_line_indices[0]), int(control_line_indices[-1]))
    zyx = line_xyz[:, ::-1] * coordinate_scale
    tag = classify_fiber_hv(
        data.get('hv_classification'), zyx[span[0]:span[1] + 1],
        min_z_fraction=min_z_fraction, min_auto_certainty=min_auto_certainty)
    fiber = Fiber(path=path, data=data, line_xyz=line_xyz, zyx=zyx,
                  control_line_indices=control_line_indices, tag=tag, span=span)
    seg = zyx[span[0]:span[1] + 1]
    fiber.z = seg[:, 0]
    yx = seg[:, 1:] - umbilicus(seg[:, 0])
    fiber.r = np.linalg.norm(yx, axis=1)
    fiber.theta_u = np.unwrap(np.arctan2(yx[:, 0], yx[:, 1]))
    return fiber


def load_fibers(fibers_dir: str, umbilicus, **kwargs) -> dict[str, Fiber]:
    fibers = {}
    for entry in sorted(os.listdir(fibers_dir)):
        if not entry.endswith('.json'):
            continue
        fiber = load_fiber(os.path.join(fibers_dir, entry), umbilicus, **kwargs)
        if fiber is not None:
            fibers[fiber.basename] = fiber
    return fibers


# ---------------------------------------------------------------------------
# Bracket detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedLink:
    vertical: str
    horizontal: str
    vertical_line_index: int
    horizontal_line_index: int
    revolution: int          # inner revolution index the vertical sits behind
    gap: float               # radial gap between the two revolutions, voxels
    behind_by: float         # vertical radius minus inner-revolution radius
    in_front_of_outer_by: float
    z: float


class _BinnedHorizontal:
    def __init__(self, fiber: Fiber, theta_bin: float):
        self.fiber = fiber
        self.theta_bin = theta_bin
        bins = np.floor(fiber.theta_u / theta_bin).astype(np.int64)
        self.bin_ids, inverse = np.unique(bins, return_inverse=True)
        counts = np.bincount(inverse)
        self.bin_z = np.bincount(inverse, weights=fiber.z) / counts
        self.bin_r = np.bincount(inverse, weights=fiber.r) / counts
        self.theta_min = float(fiber.theta_u.min())
        self.theta_max = float(fiber.theta_u.max())

    def lookup(self, bins: np.ndarray):
        pos = np.searchsorted(self.bin_ids, bins)
        ok = pos < len(self.bin_ids)
        ok[ok] &= self.bin_ids[pos[ok]] == bins[ok]
        return ok, pos

    def nearest_line_index(self, theta_target: float, point_zyx: np.ndarray,
                           window_bins: int = 3) -> int | None:
        """Full-polyline index of the line point of this revolution nearest to a point."""
        window = window_bins * self.theta_bin
        candidates = np.flatnonzero(np.abs(self.fiber.theta_u - theta_target) <= window)
        if not len(candidates):
            return None
        seg = self.fiber.zyx[self.fiber.span[0]:self.fiber.span[1] + 1]
        distances = np.linalg.norm(seg[candidates] - point_zyx, axis=1)
        return int(candidates[np.argmin(distances)] + self.fiber.span[0])


def find_bracket_links(fibers: dict[str, Fiber], *, theta_bin: float,
                       z_tolerance: float, min_gap: float, max_gap: float,
                       margin: float) -> list[PlannedLink]:
    horizontals = [
        _BinnedHorizontal(f, theta_bin) for f in fibers.values()
        if f.tag == 'H' and not f.duplicate_of
        and f.theta_u.max() - f.theta_u.min() >= TWO_PI
    ]
    verticals = [f for f in fibers.values() if f.tag == 'V' and not f.duplicate_of]
    links = []
    for v in verticals:
        v_theta = np.mod(v.theta_u, TWO_PI)
        for hb in horizontals:
            h = hb.fiber
            if v.z.max() < h.z.min() - z_tolerance or v.z.min() > h.z.max() + z_tolerance:
                continue
            k_min = int(np.floor(hb.theta_min / TWO_PI)) - 1
            k_max = int(np.ceil(hb.theta_max / TWO_PI)) + 1
            crossings = {}
            for k in range(k_min, k_max + 1):
                target = v_theta + TWO_PI * k
                ok, pos = hb.lookup(np.floor(target / theta_bin).astype(np.int64))
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                dz = np.abs(hb.bin_z[pos[idx]] - v.z[idx])
                j = int(np.argmin(dz))
                if dz[j] <= z_tolerance:
                    crossings[k] = dict(
                        v_local=int(idx[j]), rv=float(v.r[idx[j]]),
                        rh=float(hb.bin_r[pos[idx[j]]]), z=float(v.z[idx[j]]),
                        theta=float(target[idx[j]]))
            ks = sorted(crossings)
            for ka, kb in zip(ks, ks[1:]):
                if kb != ka + 1:
                    continue
                a, b = crossings[ka], crossings[kb]
                inner, outer = (a, b) if a['rh'] < b['rh'] else (b, a)
                k_inner = ka if inner is a else kb
                gap = outer['rh'] - inner['rh']
                if not (min_gap <= gap <= max_gap):
                    continue
                behind = inner['rv'] - inner['rh']
                in_front = outer['rh'] - outer['rv']
                if behind <= margin or in_front <= margin:
                    continue
                v_index = inner['v_local'] + v.span[0]
                h_index = hb.nearest_line_index(inner['theta'], v.zyx[v_index])
                if h_index is None:
                    continue
                links.append(PlannedLink(
                    vertical=v.basename, horizontal=h.basename,
                    vertical_line_index=v_index, horizontal_line_index=h_index,
                    revolution=k_inner, gap=gap, behind_by=behind,
                    in_front_of_outer_by=in_front, z=inner['z']))
    return links


def canonical_basename(data: dict) -> str | None:
    """VC3D's canonical <username>_<startedAt>_<sequence>.json for a fiber document."""
    try:
        return f"{data['username']}_{data['started_at']}_{int(data['sequence']):06d}.json"
    except (KeyError, TypeError, ValueError):
        return None


def drop_duplicate_copies(fibers: dict[str, Fiber]) -> dict[str, str]:
    """Mark exact-geometry duplicates, keeping the canonical copy for linking.

    Service exports can leave a fiber both under its canonical name and under a
    runtime-id name (e.g. ``443.json``); VC3D merges those by geometry and
    keeps the canonical file. Mirror that so a link is written once, into the
    surviving file. Duplicates stay loaded (their existing branches into edited
    files still need renumbering) but never take part in detection. Returns
    {duplicate basename: surviving basename}.
    """
    groups: dict[bytes, list[str]] = {}
    for name, fiber in fibers.items():
        key = np.ascontiguousarray(np.round(fiber.line_xyz, 3)).tobytes()
        groups.setdefault(key, []).append(name)
    dropped = {}
    for names in groups.values():
        if len(names) < 2:
            continue
        canonical = [n for n in names if canonical_basename(fibers[n].data) == n]
        survivor = canonical[0] if canonical else sorted(names)[0]
        for n in names:
            if n != survivor:
                dropped[n] = survivor
                fibers[n].duplicate_of = survivor
    return dropped


def drop_already_linked(links: list[PlannedLink], fibers: dict[str, Fiber]) -> list[PlannedLink]:
    existing = set()
    for fiber in fibers.values():
        existing |= fiber.branch_pairs()
    return [l for l in links if tuple(sorted((l.vertical, l.horizontal))) not in existing]


def line_index_inside_gap_span(fiber: Fiber, line_index: int) -> bool:
    """True when the line index lies strictly inside a span whose descriptor
    carries VC3D's `gap` span tag (format version 4: the papyrus is missing
    there and VC3D refuses to place a control point inside it). Promoting a
    point there would split the gap into two spans that are gaps no more."""
    controls = fiber.data.get('control_points') or []
    if fiber.data.get('version', 1) < 4 or not controls or not isinstance(controls[0], dict):
        return False
    indices = fiber.control_line_indices.tolist()
    for k in range(min(len(indices), len(controls)) - 1):
        if not (indices[k] < line_index < indices[k + 1]):
            continue
        segment = controls[k].get('segment_to_next') or {}
        return 'gap' in (segment.get('tags') or [])
    return False


def drop_links_inside_gap_spans(links: list[PlannedLink],
                                fibers: dict[str, Fiber]) -> list[PlannedLink]:
    return [l for l in links
            if not line_index_inside_gap_span(fibers[l.vertical], l.vertical_line_index) and
            not line_index_inside_gap_span(fibers[l.horizontal], l.horizontal_line_index)]


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def promote_control_points(data: dict, control_line_indices: np.ndarray,
                           new_line_indices: set[int]):
    """Insert control points at the given line indices.

    Returns (new_control_points, index_map, line_to_control) where index_map
    maps each old control point ordinal to its new ordinal and line_to_control
    maps every control point's line index (old and new) to its new ordinal.
    Each inserted point inherits a deep copy of the ``segment_to_next`` block
    of the control point that precedes it (the segment it splits). Inserting
    outside the first-to-last control point span is not supported.
    """
    controls = data.get('control_points') or []
    line_points = data['line_points']
    old = list(zip(control_line_indices.tolist(), controls))
    span = (old[0][0], old[-1][0])
    inserts = sorted(i for i in new_line_indices if i not in set(control_line_indices.tolist()))
    for i in inserts:
        if not (span[0] < i < span[1]):
            raise ValueError(f'control point insertion at line index {i} is outside '
                             f'the control point span {span}')
    merged = []
    index_map = {}
    pending = list(inserts)
    for ordinal, (line_index, control) in enumerate(old):
        index_map[ordinal] = len(merged)
        merged.append((line_index, control, False))
        while pending and ordinal + 1 < len(old) and pending[0] < old[ordinal + 1][0]:
            merged.append((pending.pop(0), None, True))
    assert not pending
    new_controls = []
    for pos, (line_index, control, is_new) in enumerate(merged):
        if not is_new:
            new_controls.append(control)
            continue
        position = [float(c) for c in line_points[line_index]]
        prev = merged[pos - 1][1] if not merged[pos - 1][2] else new_controls[-1]
        if isinstance(prev, dict):
            entry = {'position': position}
            if 'segment_to_next' in prev:
                segment = copy.deepcopy(prev['segment_to_next'])
                # A span tag `gap` belongs to a span between two break points;
                # the split halves around an untagged new point are not that
                # (links inside gap spans are dropped before this anyway).
                if isinstance(segment, dict) and 'tags' in segment:
                    tags = [t for t in segment['tags'] if t != 'gap']
                    if tags:
                        segment['tags'] = tags
                    else:
                        del segment['tags']
                entry['segment_to_next'] = segment
            new_controls.append(entry)
        else:
            new_controls.append(position)
    line_to_control = {line_index: pos for pos, (line_index, _, _) in enumerate(merged)}
    return new_controls, index_map, line_to_control


def _unit_tangent(line_xyz: np.ndarray, index: int) -> list[float]:
    """VC3D's endpoint tangent: forward difference to the next line point
    (backward at the last point), normalized. VC3D re-derives this on load and
    rejects a branch whose stored direction differs beyond 1e-5, so the formula
    must match exactly."""
    lower = int(index)
    upper = min(lower + 1, len(line_xyz) - 1)
    if lower == upper and lower > 0:
        lower -= 1
    d = line_xyz[upper] - line_xyz[lower]
    n = float(np.linalg.norm(d))
    return [float(x) for x in (d / n if n > 0 else np.array([1.0, 0.0, 0.0]))]


def fix_branch_directions(fibers: dict[str, Fiber]) -> dict[str, dict]:
    """Return rewritten documents for files whose stored branch directions do
    not match VC3D's tangent formula at the referenced control points."""
    docs = {}
    for name, fiber in fibers.items():
        changed = False
        d = copy.deepcopy(fiber.data)
        for br in d.get('branches') or []:
            target = fibers.get(os.path.basename(br.get('branch_file') or ''))
            local_index = int(fiber.control_line_indices[int(br['control_point_index'])])
            expected = [(_unit_tangent(fiber.line_xyz, local_index), 'control_point_direction')]
            if target is not None and 0 <= int(br['branch_control_point_index']) < len(target.control_line_indices):
                t_index = int(target.control_line_indices[int(br['branch_control_point_index'])])
                expected.append((_unit_tangent(target.line_xyz, t_index), 'branch_control_point_direction'))
            for vec, key in expected:
                stored = np.asarray(br.get(key) or [0, 0, 0], dtype=np.float64)
                sn = np.linalg.norm(stored)
                ok = sn > 0 and abs(abs(float(np.dot(stored / sn, vec))) - 1.0) <= 1e-5
                if not ok:
                    br[key] = vec
                    changed = True
        if changed:
            d['generation'] = int(d.get('generation', 1)) + 1
            docs[name] = d
    return docs


def build_branch(local: Fiber, local_index: int, target: Fiber, target_index: int,
                 local_control: int, target_control: int, pending: bool) -> dict:
    entry = {
        'control_point_index': int(local_control),
        # VC3D rebinds this from branch_file at load time.
        'branch_fiber_id': 0,
        'branch_control_point_index': int(target_control),
        'control_point_direction': _unit_tangent(local.line_xyz, local_index),
        'branch_control_point_direction': _unit_tangent(target.line_xyz, target_index),
        'control_point_position': [float(x) for x in local.line_xyz[local_index]],
        'branch_control_point_position': [float(x) for x in target.line_xyz[target_index]],
        'branch_file': target.basename,
    }
    if pending:
        entry['pending'] = True
    return entry


def plan_edits(fibers: dict[str, Fiber], links: list[PlannedLink], *, pending: bool):
    """Return {basename: new json document} for every file that must change."""
    new_points: dict[str, set[int]] = {}
    for link in links:
        new_points.setdefault(link.vertical, set()).add(link.vertical_line_index)
        new_points.setdefault(link.horizontal, set()).add(link.horizontal_line_index)

    promoted = {}
    for name, indices in new_points.items():
        fiber = fibers[name]
        controls, index_map, line_to_control = promote_control_points(
            fiber.data, fiber.control_line_indices, indices)
        promoted[name] = (controls, index_map, line_to_control)

    docs: dict[str, dict] = {}

    def doc(name):
        if name not in docs:
            docs[name] = copy.deepcopy(fibers[name].data)
        return docs[name]

    # Files with promoted control points, and files whose branches point into them.
    for name, (controls, index_map, _) in promoted.items():
        d = doc(name)
        d['control_points'] = controls
        for br in d.get('branches') or []:
            br['control_point_index'] = index_map[int(br['control_point_index'])]
    for name, fiber in fibers.items():
        targets = {os.path.basename(br.get('branch_file') or '')
                   for br in fiber.data.get('branches') or []}
        if not targets & set(promoted):
            continue
        for entry in doc(name)['branches']:
            target = os.path.basename(entry.get('branch_file') or '')
            if target in promoted:
                index_map = promoted[target][1]
                entry['branch_control_point_index'] = index_map[int(entry['branch_control_point_index'])]

    for link in links:
        v, h = fibers[link.vertical], fibers[link.horizontal]
        v_control = promoted[link.vertical][2][link.vertical_line_index]
        h_control = promoted[link.horizontal][2][link.horizontal_line_index]
        doc(link.vertical).setdefault('branches', []).append(
            build_branch(v, link.vertical_line_index, h, link.horizontal_line_index,
                         v_control, h_control, pending))
        doc(link.horizontal).setdefault('branches', []).append(
            build_branch(h, link.horizontal_line_index, v, link.vertical_line_index,
                         h_control, v_control, pending))

    for name, d in docs.items():
        d['generation'] = int(d.get('generation', 1)) + 1
    return docs


def verify_documents(docs: dict[str, dict], fibers: dict[str, Fiber]):
    """Structural checks mirroring VC3D's and the fitter's loaders."""
    from vc3d_fiber_format_adapter import parse_vc3d_fiber_format
    all_docs = {name: fibers[name].data for name in fibers}
    all_docs.update(docs)
    problems = []
    for name, d in docs.items():
        try:
            parsed = parse_vc3d_fiber_format(d, path=name)
        except ValueError as exc:
            problems.append(f'{name}: fiber parser rejects the edited document: {exc}')
            continue
        line = np.asarray(d['line_points'], dtype=np.float64)
        controls = np.asarray(parsed.control_points_xyz, dtype=np.float64)
        idx = ((line[None] - controls[:, None]) ** 2).sum(-1).argmin(1)
        if np.any(((line[idx] - controls) ** 2).sum(-1) > 1e-12):
            problems.append(f'{name}: a control point is not a line point')
        if not np.all(np.diff(idx) > 0):
            problems.append(f'{name}: control points are not an ordered subset of line_points')
        for br in d.get('branches') or []:
            target = all_docs.get(os.path.basename(br['branch_file']))
            if target is None:
                problems.append(f'{name}: branch target {br["branch_file"]} not loaded')
                continue
            if not (0 <= br['control_point_index'] < len(d['control_points'])):
                problems.append(f'{name}: control_point_index out of range')
            if not (0 <= br['branch_control_point_index'] < len(target['control_points'])):
                problems.append(f'{name}: branch_control_point_index out of range for {br["branch_file"]}')
    return problems


def _write_documents(docs, fibers, fibers_dir, apply, backup_dir):
    if not apply:
        print('dry run: nothing written (pass --apply to write)')
        return
    if not docs:
        print('nothing to write')
        return
    if backup_dir is None:
        stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_dir = os.path.join(os.path.dirname(fibers_dir), f'fibers_backup_{stamp}')
    os.makedirs(backup_dir, exist_ok=True)
    for name in docs:
        shutil.copy2(fibers[name].path, os.path.join(backup_dir, name))
    print(f'backed up {len(docs)} files to {backup_dir}')
    for name, d in docs.items():
        tmp = fibers[name].path + '.tmp'
        with open(tmp, 'wt') as fp:
            json.dump(d, fp, indent=2)
        os.replace(tmp, fibers[name].path)
    print(f'wrote {len(docs)} files')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(help=__doc__)
@click.argument('fibers_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--umbilicus', 'umbilicus_path', type=click.Path(exists=True, dir_okay=False),
              default=None, help='umbilicus.json; defaults to <fibers_dir>/../umbilicus.json')
@click.option('--coordinate-scale', type=float, default=0.25, show_default=True,
              help='fitter voxels per fiber-file coordinate unit')
@click.option('--theta-bin', type=float, default=0.005, show_default=True, help='radians')
@click.option('--z-tolerance', type=float, default=15.0, show_default=True,
              help='max |z| mismatch between a vertical point and the horizontal at that theta')
@click.option('--min-gap', type=float, default=8.0, show_default=True,
              help='min radial gap between consecutive revolutions, voxels')
@click.option('--max-gap', type=float, default=60.0, show_default=True,
              help='max radial gap between consecutive revolutions, voxels')
@click.option('--margin', type=float, default=0.0, show_default=True,
              help='vertical must clear both revolutions by this many voxels')
@click.option('--min-z-fraction', type=float, default=0.8, show_default=True)
@click.option('--min-auto-certainty', type=float, default=0.5, show_default=True)
@click.option('--dedupe/--no-dedupe', default=True, show_default=True,
              help='link only the canonical copy of exact-geometry duplicate fibers')
@click.option('--skip-existing/--no-skip-existing', default=True, show_default=True,
              help='skip pairs that already share a link in either file')
@click.option('--pending', is_flag=True, help='write links as pending (fit ignores them until approved)')
@click.option('--fix-directions', is_flag=True,
              help='instead of linking, recompute every stored branch direction with VC3D\'s '
                   'tangent formula and rewrite the files where it differs (honours --apply)')
@click.option('--apply', is_flag=True, help='write the files; default is a dry run')
@click.option('--backup-dir', type=click.Path(file_okay=False), default=None,
              help='where touched files are copied before writing; '
                   'defaults to <fibers_dir>/../fibers_backup_<timestamp>')
@click.option('--report', 'report_path', type=click.Path(dir_okay=False), default=None,
              help='write the planned links and per-file changes as JSON')
def main(fibers_dir, umbilicus_path, coordinate_scale, theta_bin, z_tolerance, min_gap,
         max_gap, margin, min_z_fraction, min_auto_certainty, dedupe, skip_existing,
         pending, fix_directions, apply, backup_dir, report_path):
    fibers_dir = os.path.abspath(fibers_dir)
    if umbilicus_path is None:
        umbilicus_path = os.path.join(os.path.dirname(fibers_dir), 'umbilicus.json')
    umbilicus = json_umbilicus_z_to_yx(umbilicus_path)
    fibers = load_fibers(fibers_dir, umbilicus, coordinate_scale=coordinate_scale,
                         min_z_fraction=min_z_fraction, min_auto_certainty=min_auto_certainty)
    tags = {t: sum(1 for f in fibers.values() if f.tag == t) for t in ('H', 'V', None)}
    multi = sum(1 for f in fibers.values()
                if f.tag == 'H' and f.theta_u.max() - f.theta_u.min() >= TWO_PI)
    print(f'loaded {len(fibers)} fibers: {tags["H"]} horizontal ({multi} with >= 1 revolution), '
          f'{tags["V"]} vertical, {tags[None]} untagged')

    if fix_directions:
        docs = fix_branch_directions(fibers)
        print(f'files with branch directions that VC3D would reject: {len(docs)}')
        for name in sorted(docs):
            print(f'  {name}')
        _write_documents(docs, fibers, fibers_dir, apply, backup_dir)
        return

    if dedupe:
        duplicates = drop_duplicate_copies(fibers)
        print(f'exact-geometry duplicate copies excluded from linking: {len(duplicates)}')
        for dup, survivor in sorted(duplicates.items()):
            print(f'  {dup} duplicates {survivor}')

    links = find_bracket_links(fibers, theta_bin=theta_bin, z_tolerance=z_tolerance,
                               min_gap=min_gap, max_gap=max_gap, margin=margin)
    print(f'bracketed vertical-horizontal pairs: {len(links)}')
    if skip_existing:
        kept = drop_already_linked(links, fibers)
        print(f'  already linked (skipped): {len(links) - len(kept)}')
        links = kept
    kept = drop_links_inside_gap_spans(links, fibers)
    if len(kept) != len(links):
        print(f'  inside a gap span (skipped): {len(links) - len(kept)}')
    links = kept
    # One link per (vertical, horizontal) pair: keep the one with the widest clearance.
    best = {}
    for link in links:
        key = (link.vertical, link.horizontal)
        if key not in best or min(link.behind_by, link.in_front_of_outer_by) > \
                min(best[key].behind_by, best[key].in_front_of_outer_by):
            best[key] = link
    links = sorted(best.values(), key=lambda l: (l.vertical, l.horizontal))
    print(f'planned new links: {len(links)} '
          f'({len({l.vertical for l in links})} verticals, {len({l.horizontal for l in links})} horizontals)')
    for link in links:
        print(f'  {link.vertical} [line {link.vertical_line_index}] -> '
              f'{link.horizontal} [line {link.horizontal_line_index}] '
              f'rev {link.revolution}, z={link.z:.0f}, gap={link.gap:.1f}, '
              f'behind by {link.behind_by:.1f}, in front of outer by {link.in_front_of_outer_by:.1f}')

    docs = plan_edits(fibers, links, pending=pending) if links else {}
    print(f'files to modify: {len(docs)}')
    for name in sorted(docs):
        before = fibers[name].data
        after = docs[name]
        print(f'  {name}: control points {len(before["control_points"])} -> {len(after["control_points"])}, '
              f'branches {len(before.get("branches") or [])} -> {len(after.get("branches") or [])}')
    problems = verify_documents(docs, fibers) if docs else []
    for p in problems:
        print('PROBLEM:', p)

    if report_path:
        with open(report_path, 'wt') as fp:
            json.dump({
                'fibers_dir': fibers_dir,
                'parameters': dict(coordinate_scale=coordinate_scale, theta_bin=theta_bin,
                                   z_tolerance=z_tolerance, min_gap=min_gap, max_gap=max_gap,
                                   margin=margin, pending=pending),
                'links': [l.__dict__ for l in links],
                'files': {name: {'control_points': len(d['control_points']),
                                 'branches': len(d.get('branches') or [])}
                          for name, d in docs.items()},
                'problems': problems,
            }, fp, indent=2)
        print(f'report written to {report_path}')

    if not apply:
        print('dry run: nothing written (pass --apply to write)')
        return
    if problems:
        raise click.ClickException('refusing to write: verification problems above')
    _write_documents(docs, fibers, fibers_dir, apply, backup_dir)
    if not docs:
        return

    # Final check through the fitter's own loader and link resolution.
    from spiral_helpers import load_fiber_point_collections, resolve_fiber_links
    collections, _ = load_fiber_point_collections(fibers_dir, 0, min_point_spacing=40.0)
    for pcl in collections.values():
        pcl['file_basename'] = os.path.basename(pcl['source_file'])
    resolved = resolve_fiber_links(collections, include_pending=True, assume_unannotated=True)
    print(f'fitter link resolution after write: {len(resolved)} links '
          f'({sum(1 for l in resolved if l["pending"])} pending)')


if __name__ == '__main__':
    main()
