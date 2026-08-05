"""Unit tests for fiber_merge — pure JSON in/out, no I/O.

The high-value assertions here are loader-safety checks: `loader_issues`
below is a mini-port of VC3D's load-time validation (the C1/C2/C3
invariants from fiber_merge's module docstring). Merge output that fails
those checks gets destroyed by an unmodified VC3D, so every clean-merge
test runs its output through them.
"""
import copy
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fiber_merge
from fiber_merge import merge_fibers, refresh_pair_links, REOPTIMIZE_TAG


def cp(i, dz=0.0):
    """A control point on a simple line, spaced 10 apart."""
    return [100.0 + 10.0 * i, 200.0, 300.0 + dz]


def line_for(cps, samples_per_span=4):
    """A dense polyline passing through the control points exactly."""
    line = []
    for a, b in zip(cps[:-1], cps[1:]):
        line.append(list(a))
        for s in range(1, samples_per_span):
            t = s / samples_per_span
            line.append([a[k] + t * (b[k] - a[k]) for k in range(3)])
    line.append(list(cps[-1]))
    return line


def make_fiber(cps, branches=None, tags=None, generation=1,
               filename='dj_x_000001.json'):
    return {
        'type': 'vc3d_fiber',
        'version': 1,
        'filename': filename,
        'username': 'dj',
        'generation': generation,
        'control_points': [list(p) for p in cps],
        'line_points': line_for(cps),
        'branches': copy.deepcopy(branches or []),
        'tags': list(tags or []),
    }


def v3_segment(goal='global', mode='trace', metric=2.0, msg='trace'):
    trace = mode == 'trace'
    return {
        'optimizer': 'native_fiber_trace3d',
        'metadata_version': 3,
        'tracer_version': 2,
        'interp_goal': goal,
        'interp_mode': mode,
        'metric': metric,
        'msg': msg,
        'normal_manifest': '/data/normals.lasagna.json' if mode != 'cspline' else '',
        'fiber_manifest': 's3://bucket/fibers.lasagna.json' if trace else '',
        'trace_to_base_scale': 4.0,
        'meeting_error_base_voxels': 2.0 if trace else None,
        'meeting_error_ratio': 0.02 if trace else None,
        'meeting_source': 'forward_moving_plane' if trace else '',
        'failure_code': '',
        'failure_detail': '',
        'lasagna_failure_code': '',
        'lasagna_failure_detail': '',
        'config': {
            'step_voxels': 4.0,
            'cone_angle_degrees': 25.0,
            'cone_angle_step_degrees': 5.0,
            'cone_grid_size': 25,
            'beam_width': 8,
            'beam_prune_distance_voxels': 1.0,
            'beam_lookahead_steps': 2,
            'smoothness_weight': 2.0,
            'smoothness_normal_weight': 0.1,
            'smoothness_tangent_weight': 10.0,
            'smoothness_free_angle_degrees': 0.0,
            'cumulative_smoothness_steps': 4,
            'cumulative_smoothness_tangent_weight': 2.0,
            'initial_free_angle_degrees': 0.0,
            'max_step_factor': 3.0,
            'meeting_accept_max_error_ratio': 0.1,
            'endpoint_accept_threshold_base_voxels': 20.0,
        },
    }


def make_v3_fiber(cps, generation=1, filename='dj_x_000001.json',
                  optimization_mode='native_fiber_trace3d'):
    doc = make_fiber(cps, generation=generation, filename=filename)
    doc['version'] = 3
    doc['optimization_mode'] = optimization_mode
    doc['control_points'] = [
        {'position': list(point), 'segment_to_next': v3_segment()}
        if index + 1 < len(cps) else {'position': list(point)}
        for index, point in enumerate(cps)
    ]
    return doc


def set_v3_span(doc, index, *, goal, bend):
    mode = 'trace' if goal == 'global' else goal
    metric = None if mode == 'cspline' else 2.0
    doc['control_points'][index]['segment_to_next'] = v3_segment(
        goal=goal, mode=mode, metric=metric, msg=mode)
    start = index * 4
    for offset in range(1, 4):
        doc['line_points'][start + offset][1] += bend


def link(target, local_pos, remote_pos, local_index, remote_index=0,
         pending=True):
    return {
        'control_point_index': local_index,
        'branch_fiber_id': 7,
        'branch_control_point_index': remote_index,
        'branch_file': target,
        'control_point_direction': [1.0, 0.0, 0.0],
        'branch_control_point_direction': [0.0, 1.0, 0.0],
        'control_point_position': list(local_pos),
        'branch_control_point_position': list(remote_pos),
        'pending': pending,
    }


BASE_CPS = [cp(i) for i in range(8)]
OTHER = [999.0, 999.0, 999.0]


# --- independent mini-port of VC3D's load-time validation ----------------
# Deliberately does NOT reuse fiber_merge's predicates: these are separate
# implementations of the C++ (LineAnnotationController.cpp,
# FiberSliceGeometry.cpp, Atlas.cpp) so a fiber_merge primitive that
# drifts from the loader makes tests FAIL instead of drifting with it.

_REQUIRED_BRANCH_KEYS = (
    'control_point_index', 'branch_control_point_index',
    'control_point_direction', 'branch_control_point_direction',
    'control_point_position', 'branch_control_point_position', 'branch_file')


def _l_finite(p):
    return (isinstance(p, (list, tuple)) and len(p) == 3 and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and
                math.isfinite(x) for x in p))


def _l_cp_position(value):
    return value.get('position') if isinstance(value, dict) else value


def _l_pos_eq(a, b, tol=1.0e-6):
    """pointsApproximatelyEqual: EUCLIDEAN ball."""
    if not _l_finite(a) or not _l_finite(b):
        return False
    return sum((x - y) ** 2 for x, y in zip(a, b)) <= tol * tol


def _l_finite_direction(v):
    """finiteDirection: finite and norm > 1e-12."""
    return _l_finite(v) and math.sqrt(sum(x * x for x in v)) > 1.0e-12


def _l_dirs_compatible(a, b, tol=1.0e-5):
    """branchDirectionsCompatible: sign-agnostic, 1e-5 on |cos|."""
    if not _l_finite_direction(a) or not _l_finite_direction(b):
        return False
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    dot = sum((x / na) * (y / nb) for x, y in zip(a, b))
    return abs(abs(dot) - 1.0) <= tol


def _l_tangent(line, point):
    """nearestLinePointIndex + tangentAtLinePosition."""
    best, best_d2 = 0, float('inf')
    for i, p in enumerate(line):
        if not _l_finite(p):
            continue
        d2 = sum((x - y) ** 2 for x, y in zip(p, point))
        if d2 < best_d2:
            best_d2, best = d2, i
    n = len(line)
    lower = max(0, min(best, n - 1))
    upper = min(lower + 1, n - 1)
    if lower == upper and lower > 0:
        lower -= 1
    delta = [line[upper][k] - line[lower][k] for k in range(3)]
    norm = math.sqrt(sum(x * x for x in delta))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return [1.0, 0.0, 0.0]
    return [x / norm for x in delta]


def _l_basename(name):
    return name.rsplit('/', 1)[-1] if isinstance(name, str) else ''


def loader_issues(docs_by_name):
    """Independent mini-port of VC3D's load-time fiber validation. Returns
    a list of issue strings; [] means an unmodified VC3D loads the set
    cleanly and rewrites nothing destructive."""
    issues = []
    for name, doc in docs_by_name.items():
        cps = [_l_cp_position(cp) for cp in doc['control_points']]
        line = doc['line_points']
        for label, points in (('control point', cps), ('line point', line)):
            for k, point in enumerate(points):
                if not _l_finite(point):
                    issues.append(f"{name}: non-finite {label} {k} (fatal)")
        # C1: control points must be an ordered subset of line_points
        # within Euclidean 1e-8 (validateFiberInputControlPoints) — fatal.
        li = 0
        for k, point in enumerate(cps):
            while li < len(line) and not _l_pos_eq(line[li], point, tol=1.0e-8):
                li += 1
            if li >= len(line):
                issues.append(f"{name}: control point {k} not on line (fatal)")
                break
            li += 1
        for entry in doc.get('branches', []):
            # Parse-time stripping: the loader deletes (and rewrites the
            # file without) entries it cannot parse — surfaced as issues so
            # merge output never relies on that destruction.
            if (not isinstance(entry, dict) or
                    'link_direction' in entry or
                    any(key not in entry for key in _REQUIRED_BRANCH_KEYS) or
                    not _l_basename(entry['branch_file'])):
                issues.append(f"{name}: branch entry stripped at parse time "
                              "(file rewritten)")
                continue
            if (not _l_finite_direction(entry['control_point_direction']) or
                    not _l_finite_direction(
                        entry['branch_control_point_direction'])):
                issues.append(f"{name}: non-finite branch direction")
                continue
            i = entry['control_point_index']
            if not (isinstance(i, int) and 0 <= i < len(cps)):
                issues.append(f"{name}: local CP index out of range")
                continue
            if not _l_pos_eq(cps[i], entry['control_point_position']):
                issues.append(f"{name}: local CP position mismatch")
                continue
            if len(line) >= 2:
                tangent = _l_tangent(line, entry['control_point_position'])
                if not _l_dirs_compatible(entry['control_point_direction'],
                                          tangent):
                    issues.append(f"{name}: branch endpoint direction mismatch")
                    continue
            target = docs_by_name.get(_l_basename(entry['branch_file']))
            if target is None:
                issues.append(f"{name}: missing linked fiber "
                              f"{entry['branch_file']}")
                continue
            tcps = [_l_cp_position(cp) for cp in target['control_points']]
            tline = target['line_points']
            j = entry['branch_control_point_index']
            if not (isinstance(j, int) and 0 <= j < len(tcps)):
                issues.append(f"{name}: linked CP index out of range")
                continue
            if not _l_pos_eq(tcps[j], entry['branch_control_point_position']):
                issues.append(f"{name}: linked CP position mismatch")
                continue
            if len(tline) >= 2:
                tangent = _l_tangent(tline,
                                     entry['branch_control_point_position'])
                if not _l_dirs_compatible(
                        entry['branch_control_point_direction'], tangent):
                    issues.append(f"{name}: linked endpoint direction mismatch")
                    continue
            # C2: index-exact reciprocity, plus positions and directions
            # compared STORED against STORED.
            reciprocal = any(
                _l_basename(c.get('branch_file')) == name and
                c.get('control_point_index') == j and
                c.get('branch_control_point_index') == i and
                _l_pos_eq(c.get('control_point_position'),
                          entry['branch_control_point_position']) and
                _l_pos_eq(c.get('branch_control_point_position'),
                          entry['control_point_position']) and
                _l_dirs_compatible(c.get('control_point_direction'),
                                   entry['branch_control_point_direction']) and
                _l_dirs_compatible(c.get('branch_control_point_direction'),
                                   entry['control_point_direction'])
                for c in target.get('branches', []) if isinstance(c, dict))
            if not reciprocal:
                issues.append(f"{name}: missing reciprocal branch in "
                              f"{entry['branch_file']}")
    return issues


def make_pair(a_name, b_name, a_cps, b_cps, a_index, b_index, pending=True):
    """A consistent linked fiber pair, as VC3D would write it."""
    a = make_fiber(a_cps, filename=a_name)
    b = make_fiber(b_cps, filename=b_name)
    pa, pb = a_cps[a_index], b_cps[b_index]
    da = fiber_merge.endpoint_tangent(a['line_points'], pa)
    db = fiber_merge.endpoint_tangent(b['line_points'], pb)
    a['branches'] = [{
        'control_point_index': a_index, 'branch_fiber_id': 2,
        'branch_control_point_index': b_index, 'branch_file': b_name,
        'control_point_direction': da, 'branch_control_point_direction': db,
        'control_point_position': list(pa),
        'branch_control_point_position': list(pb), 'pending': pending,
    }]
    b['branches'] = [{
        'control_point_index': b_index, 'branch_fiber_id': 1,
        'branch_control_point_index': a_index, 'branch_file': a_name,
        'control_point_direction': db, 'branch_control_point_direction': da,
        'control_point_position': list(pb),
        'branch_control_point_position': list(pa), 'pending': pending,
    }]
    return a, b


def test_loader_issue_helper_accepts_consistent_pair():
    a, b = make_pair('a.json', 'b.json', BASE_CPS,
                     [cp(i, dz=50.0) for i in range(4)], 2, 1)
    assert loader_issues({'a.json': a, 'b.json': b}) == []


def test_link_union_incident_fixture():
    """The real incident: one side adds L2, the other adds L3; base has L1.
    Every link must survive with correct indices."""
    l1 = link('kb_a.json', BASE_CPS[2], OTHER, 2)
    base = make_fiber(BASE_CPS, branches=[l1])
    local = make_fiber(BASE_CPS, branches=[l1, link('kb_b.json', BASE_CPS[4], OTHER, 4)],
                       generation=3)
    remote = make_fiber(BASE_CPS, branches=[l1, link('kb_c.json', BASE_CPS[6], OTHER, 6)],
                        generation=2)

    result = merge_fibers(base, local, remote)
    assert result['ok']
    targets = sorted(b['branch_file'] for b in result['merged']['branches'])
    assert targets == ['kb_a.json', 'kb_b.json', 'kb_c.json']
    by_target = {b['branch_file']: b for b in result['merged']['branches']}
    assert by_target['kb_b.json']['control_point_index'] == 4
    assert by_target['kb_c.json']['control_point_index'] == 6
    assert result['merged']['generation'] == 4
    # No geometry change -> no reoptimization needed
    assert REOPTIMIZE_TAG not in result['merged']['tags']
    # The caller must mirror reciprocals into all three peers
    assert result['peer_files'] == ['kb_a.json', 'kb_b.json', 'kb_c.json']


def test_disjoint_cp_edits_merge_with_synthetic_line():
    base = make_fiber(BASE_CPS)
    local_cps = copy.deepcopy(BASE_CPS)
    local_cps[1] = cp(1, dz=5.0)          # local moves CP 1
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[6] = cp(6, dz=-5.0)        # remote moves CP 6
    local = make_fiber(local_cps, generation=2)
    remote = make_fiber(remote_cps, generation=2)

    result = merge_fibers(base, local, remote)
    assert result['ok']
    merged = result['merged']
    assert merged['control_points'][1] == cp(1, dz=5.0)
    assert merged['control_points'][6] == cp(6, dz=-5.0)
    assert len(merged['control_points']) == 8
    # Two-sided geometry: the line is the control-point polyline (trivially
    # loader-safe) pending reoptimization in VC3D
    assert merged['line_points'] == merged['control_points']
    assert result['stats']['geometry_merged']
    assert result['stats']['reoptimize']
    assert REOPTIMIZE_TAG in merged['tags']
    assert loader_issues({'dj_x_000001.json': merged}) == []


def test_one_sided_geometry_takes_that_sides_line():
    base = make_fiber(BASE_CPS)
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[6] = cp(6, dz=-5.0)
    local = make_fiber(BASE_CPS, tags=['meta-only'], generation=2)
    remote = make_fiber(remote_cps, generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['line_points'] == remote['line_points']
    assert result['merged']['control_points'] == remote['control_points']
    # A consistent written pair needs no reoptimization
    assert REOPTIMIZE_TAG not in result['merged']['tags']
    assert loader_issues({'dj_x_000001.json': result['merged']}) == []


def test_line_only_reoptimization_is_carried():
    """A side that re-optimized line_points without moving control points
    owns the geometry: its line is taken wholesale (the old splice code
    kept the stale local line instead)."""
    base = make_fiber(BASE_CPS)
    local = make_fiber(BASE_CPS, tags=['meta-edit'], generation=2)
    remote = make_fiber(BASE_CPS, generation=2)
    remote['line_points'] = [[p[0], p[1] + 0.5, p[2]]
                             for p in remote['line_points']]
    # C1 within remote's own file: its control points sit on its line
    remote['control_points'] = [[p[0], p[1] + 0.5, p[2]]
                                for p in remote['control_points']]
    result = merge_fibers(base, local, remote)
    assert result['ok']
    if result['stats'].get('reoptimize'):
        # Two-sided outcome is acceptable as long as it is loader-safe
        assert REOPTIMIZE_TAG in result['merged']['tags']
    else:
        assert result['merged']['line_points'] == remote['line_points']
    assert loader_issues({'dj_x_000001.json': result['merged']}) == []


def test_cp_insertion_shifts_indices_and_links_follow():
    base = make_fiber(BASE_CPS, branches=[link('kb_a.json', BASE_CPS[6], OTHER, 6)])
    # local inserts a point between 1 and 2 -> link anchor shifts to index 7
    local_cps = BASE_CPS[:2] + [[105.0, 205.0, 300.0]] + BASE_CPS[2:]
    local = make_fiber(local_cps,
                       branches=[link('kb_a.json', BASE_CPS[6], OTHER, 7)],
                       generation=2)
    remote = make_fiber(BASE_CPS,
                        branches=[link('kb_a.json', BASE_CPS[6], OTHER, 6),
                                  link('kb_d.json', BASE_CPS[3], OTHER, 3)],
                        generation=2)

    result = merge_fibers(base, local, remote)
    assert result['ok']
    merged = result['merged']
    assert len(merged['control_points']) == 9
    by_target = {b['branch_file']: b for b in merged['branches']}
    assert by_target['kb_a.json']['control_point_index'] == 7
    assert by_target['kb_d.json']['control_point_index'] == 4
    assert sorted(result['peer_files']) == ['kb_a.json', 'kb_d.json']


def test_overlapping_cp_edits_conflict():
    base = make_fiber(BASE_CPS)
    local_cps = copy.deepcopy(BASE_CPS)
    local_cps[3] = cp(3, dz=5.0)
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[3] = cp(3, dz=-5.0)
    result = merge_fibers(base, make_fiber(local_cps), make_fiber(remote_cps))
    assert not result['ok']
    assert any('control_points' in c for c in result['conflicts'])


def test_adjacent_cp_edits_without_anchor_conflict():
    base = make_fiber(BASE_CPS)
    local_cps = copy.deepcopy(BASE_CPS)
    local_cps[3] = cp(3, dz=5.0)
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[4] = cp(4, dz=-5.0)
    # An anchor must be unchanged on BOTH sides, so the region between the
    # anchors CP2 and CP5 contains both edits -> conflict.
    result = merge_fibers(base, make_fiber(local_cps), make_fiber(remote_cps))
    assert not result['ok']


def test_delete_vs_modify_conflicts():
    base = make_fiber(BASE_CPS)
    local_cps = BASE_CPS[:3] + BASE_CPS[4:]          # local deletes CP 3
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[3] = cp(3, dz=5.0)                    # remote moves CP 3
    result = merge_fibers(base, make_fiber(local_cps), make_fiber(remote_cps))
    assert not result['ok']


def test_deletion_on_one_side_merges():
    base = make_fiber(BASE_CPS)
    local_cps = BASE_CPS[:3] + BASE_CPS[4:]          # local deletes CP 3
    result = merge_fibers(base, make_fiber(local_cps, generation=2),
                          make_fiber(BASE_CPS, generation=2))
    assert result['ok']
    assert len(result['merged']['control_points']) == 7
    assert loader_issues({'dj_x_000001.json': result['merged']}) == []


def test_both_extend_same_end_conflicts():
    base = make_fiber(BASE_CPS)
    local = make_fiber(BASE_CPS + [cp(8)])
    remote = make_fiber(BASE_CPS + [cp(9)])
    result = merge_fibers(base, local, remote)
    assert not result['ok']


def test_extend_opposite_ends_merges():
    base = make_fiber(BASE_CPS)
    local = make_fiber([cp(-1)] + BASE_CPS, generation=2)
    remote = make_fiber(BASE_CPS + [cp(8)], generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    merged = result['merged']['control_points']
    assert merged[0] == cp(-1)
    assert merged[-1] == cp(8)
    assert len(merged) == 10
    assert REOPTIMIZE_TAG in result['merged']['tags']
    assert loader_issues({'dj_x_000001.json': result['merged']}) == []


def test_link_anchor_removed_by_merged_geometry_is_a_conflict():
    """#1223 dropped such links with a note; links are never silently
    dropped anymore — the whole merge falls back to manual resolution."""
    base = make_fiber(BASE_CPS)
    local_cps = BASE_CPS[:3] + BASE_CPS[4:]          # local deletes CP 3
    local = make_fiber(local_cps, generation=2)
    remote = make_fiber(BASE_CPS,
                        branches=[link('kb_a.json', BASE_CPS[3], OTHER, 3)],
                        generation=2)
    result = merge_fibers(base, local, remote)
    assert not result['ok']
    assert any('anchors at a control point absent' in c
               for c in result['conflicts'])


def test_pending_false_wins():
    pending_link = link('kb_a.json', BASE_CPS[2], OTHER, 2, pending=True)
    approved_link = link('kb_a.json', BASE_CPS[2], OTHER, 2, pending=False)
    base = make_fiber(BASE_CPS, branches=[pending_link])
    local = make_fiber(BASE_CPS, branches=[pending_link], generation=5)
    remote = make_fiber(BASE_CPS, branches=[approved_link], generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['branches'][0]['pending'] is False
    assert result['stats']['links_approved'] == 1


def make_fiber_with_marker(fiber):
    """Make a doc differ from base somewhere harmless so the local==base
    short-circuit doesn't bypass the branch merge under test."""
    fiber = copy.deepcopy(fiber)
    fiber['tags'] = fiber.get('tags', []) + ['marker']
    return fiber


def test_base_aware_deletion():
    l1 = link('kb_a.json', BASE_CPS[2], OTHER, 2)
    base = make_fiber(BASE_CPS, branches=[l1])
    local = make_fiber(BASE_CPS, branches=[l1])          # untouched
    remote = make_fiber(BASE_CPS, branches=[])           # deleted it
    result = merge_fibers(base, make_fiber_with_marker(local), remote)
    assert result['ok']
    assert result['merged']['branches'] == []
    assert result['stats']['links_deleted'] == 1
    # The deleted pair's peer still needs its reciprocal removed
    assert result['peer_files'] == ['kb_a.json']


def test_modify_beats_delete():
    pending_link = link('kb_a.json', BASE_CPS[2], OTHER, 2, pending=True)
    approved_link = link('kb_a.json', BASE_CPS[2], OTHER, 2, pending=False)
    base = make_fiber(BASE_CPS, branches=[pending_link])
    local = make_fiber(BASE_CPS, branches=[approved_link])   # approved it
    remote = make_fiber(BASE_CPS, branches=[])               # deleted it
    result = merge_fibers(base, local, make_fiber_with_marker(remote))
    assert result['ok']
    assert len(result['merged']['branches']) == 1
    assert result['merged']['branches'][0]['pending'] is False


def test_tags_base_aware_merge():
    base = make_fiber(BASE_CPS, tags=['old', 'shared'])
    local = make_fiber(BASE_CPS, tags=['shared', 'from-local'])      # dropped 'old'
    remote = make_fiber(BASE_CPS, tags=['old', 'shared', 'from-remote'])
    result = merge_fibers(base, local, remote)
    assert result['ok']
    tags = result['merged']['tags']
    assert 'shared' in tags
    assert 'from-local' in tags
    assert 'from-remote' in tags
    assert 'old' not in tags  # local removed it, remote left it untouched


def test_scalars_follow_newer_generation():
    base = make_fiber(BASE_CPS)
    local = make_fiber(BASE_CPS, tags=['a'], generation=2)
    remote = make_fiber(BASE_CPS, tags=['b'], generation=6)
    remote['username'] = 'kb'
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['username'] == 'kb'
    assert result['merged']['generation'] == 7


def test_short_circuit_local_unchanged():
    base = make_fiber(BASE_CPS)
    remote = make_fiber(BASE_CPS, tags=['new'], generation=4)
    result = merge_fibers(base, copy.deepcopy(base), remote)
    assert result['ok']
    assert result['merged'] == remote
    # A wholesale-adopted side is already consistent with its peers
    assert result['peer_files'] == []


def test_noop_stability():
    base = make_fiber(BASE_CPS)
    result = merge_fibers(base, copy.deepcopy(base), copy.deepcopy(base))
    assert result['ok']
    assert result['merged'] == base


def test_tolerance_bounds():
    jittered = [[p[0] + 1e-9, p[1], p[2]] for p in BASE_CPS]
    moved = [[p[0] + 1e-3, p[1], p[2]] for p in BASE_CPS]
    assert fiber_merge.pos_eq(BASE_CPS[0], jittered[0])
    assert not fiber_merge.pos_eq(BASE_CPS[0], moved[0])
    # jitter-only side counts as unchanged geometry
    base = make_fiber(BASE_CPS)
    local = make_fiber(jittered, tags=['touched'])
    remote = make_fiber(BASE_CPS, tags=['other'])
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert not result['stats'].get('geometry_merged', False)
    assert loader_issues({'dj_x_000001.json': result['merged']}) == []


def test_filename_mismatch_is_conflict():
    base = make_fiber(BASE_CPS)
    local = make_fiber(BASE_CPS, tags=['a'])
    remote = make_fiber(BASE_CPS, tags=['b'], filename='other.json')
    result = merge_fibers(base, local, remote)
    assert not result['ok']


def test_opaque_branch_entries_survive_merge():
    """Structurally unparseable entries must never vanish from a clean
    merge."""
    opaque = 'unparseable-link'
    base = make_fiber(BASE_CPS, branches=[opaque])
    local = make_fiber(BASE_CPS, branches=[opaque], tags=['from-local'])
    remote = make_fiber(BASE_CPS, branches=[opaque], tags=['from-remote'])
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['branches'] == [opaque]


def test_divergent_opaque_entries_conflict():
    base = make_fiber(BASE_CPS, branches=['legacy-a'])
    local = make_fiber(BASE_CPS, branches=['legacy-b'], tags=['x'])
    remote = make_fiber(BASE_CPS, branches=['legacy-c'], tags=['y'])
    result = merge_fibers(base, local, remote)
    assert not result['ok']
    assert any('unparseable' in c for c in result['conflicts'])


def test_opaque_entry_deleted_on_one_side_merges():
    base = make_fiber(BASE_CPS, branches=['legacy-a'])
    local = make_fiber(BASE_CPS, branches=[], tags=['x'])       # deleted it
    remote = make_fiber(BASE_CPS, branches=['legacy-a'], tags=['y'])
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['branches'] == []


def test_malformed_dict_branch_is_opaque_not_a_crash():
    broken = {'branch_file': 'kb_a.json', 'control_point_position': 'oops'}
    base = make_fiber(BASE_CPS, branches=[broken])
    local = make_fiber(BASE_CPS, branches=[broken], tags=['from-local'])
    remote = make_fiber(BASE_CPS, branches=[broken], tags=['from-remote'])
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['branches'] == [broken]


def test_link_identity_is_tolerance_based_not_rounding_based():
    """Positions within POS_TOL but on opposite sides of a rounding bucket
    must still be the SAME link."""
    pos_a = [100.0000004, 200.0, 300.0]
    pos_b = [100.0000013, 200.0, 300.0]  # within 1e-6 of pos_a
    cps = [pos_a] + [cp(i) for i in range(1, 4)]
    approved = link('kb_a.json', pos_a, OTHER, 0, pending=False)
    jittered_pending = link('kb_a.json', pos_b, OTHER, 0, pending=True)
    base = make_fiber(cps)
    local = make_fiber(cps, branches=[approved], generation=2)
    remote = make_fiber(cps, branches=[jittered_pending], generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    branches = result['merged']['branches']
    assert len(branches) == 1
    assert branches[0]['pending'] is False  # approval won


def test_pos_eq_metric_matches_loader():
    """pos_eq must be the loader's Euclidean ball, not a per-axis box
    (the PR #1246 review's M1: a box accepts positions up to sqrt(3)*tol
    apart that the loader rejects)."""
    a = [0.0, 0.0, 0.0]
    box_corner = [8e-7, 8e-7, 8e-7]        # per-axis ok, Euclid 1.386e-6
    inside = [5e-7, 5e-7, 5e-7]            # Euclid 8.66e-7
    assert not fiber_merge.pos_eq(a, box_corner)
    assert fiber_merge.pos_eq(a, inside)
    assert fiber_merge.pos_eq(a, box_corner) == _l_pos_eq(a, box_corner)
    assert fiber_merge.pos_eq(a, inside) == _l_pos_eq(a, inside)


def test_jitter_beyond_euclidean_tolerance_conflicts_not_drops():
    """Geometry jittered past the loader's Euclidean tolerance on one side
    plus a link added on the other: the anchor cannot be resolved, so the
    whole merge is a manual conflict — the link is neither dropped nor
    written loader-broken (the #1246 review's M1 repro)."""
    base = make_fiber(BASE_CPS)
    jittered = [[p[0] + 8e-7, p[1] + 8e-7, p[2] + 8e-7] for p in BASE_CPS]
    local = make_fiber(jittered, generation=2)
    local['tags'] = ['touched']
    remote = make_fiber(BASE_CPS,
                        branches=[link('kb_a.json', BASE_CPS[2], OTHER, 2)],
                        generation=2)
    result = merge_fibers(base, local, remote)
    assert not result['ok']
    assert any('absent from the merged geometry' in c
               for c in result['conflicts'])


def test_jitter_within_euclidean_tolerance_merges_and_snaps():
    """The incident shape with sub-tolerance jitter on one side: the merge
    succeeds, and the consistency pass snaps the loser-side link positions
    BYTE-EXACT to the merged control points so the pair clears the real
    loader (regression for M1's geometry_same repro)."""
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    base_a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1)

    def jitter(p):
        return [p[0] + 5e-7, p[1] + 5e-7, p[2] + 5e-7]  # Euclid 8.66e-7

    local = copy.deepcopy(base_a)
    local['control_points'] = [jitter(p) for p in local['control_points']]
    local['line_points'] = [jitter(p) for p in local['line_points']]
    local['branches'][0]['control_point_position'] = \
        list(local['control_points'][2])                # VC3D snaps on save
    local['generation'] = 3
    remote = copy.deepcopy(base_a)
    remote['branches'].append(
        make_pair('a.json', 'b.json', BASE_CPS, b_cps, 4, 2)[0]['branches'][0])
    remote['generation'] = 2

    result = merge_fibers(base_a, local, remote)
    assert result['ok']
    assert result['peer_files'] == ['b.json']
    merged = result['merged']
    # Carrier is the newer (jittered) side; both links survived
    assert len(merged['branches']) == 2

    out = refresh_pair_links(merged, b, 'a.json', 'b.json', base_doc=base_a)
    assert out['ok']
    fixed_a = out['a_doc']
    for entry in fixed_a['branches']:
        i = entry['control_point_index']
        assert entry['control_point_position'] == fixed_a['control_points'][i]
    assert loader_issues({'a.json': fixed_a, 'b.json': out['b_doc']}) == []


def test_pairwise_direction_ratchet_is_closed():
    """Two stored directions each within tolerance of the true tangent can
    be 2x tolerance apart from EACH OTHER — which the loader's
    stored-vs-stored reciprocity check rejects. The refresh must leave the
    pair byte-identical up to sign, not merely each-within-tolerance
    (the #1246 review's M2)."""
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1)

    def rotated(direction, degrees):
        axis = [0.0, 0.0, 1.0]  # orthogonal to the +x tangents used here
        u = [axis[1] * direction[2] - axis[2] * direction[1],
             axis[2] * direction[0] - axis[0] * direction[2],
             axis[0] * direction[1] - axis[1] * direction[0]]
        radians = math.radians(degrees)
        return [math.cos(radians) * d + math.sin(radians) * x
                for d, x in zip(direction, u)]

    true_db = b['branches'][0]['control_point_direction']
    plus = rotated(true_db, 0.2)
    minus = rotated(true_db, -0.2)
    # Each passes the loader's per-field check individually...
    assert _l_dirs_compatible(plus, true_db)
    assert _l_dirs_compatible(minus, true_db)
    # ...but not against each other.
    assert not _l_dirs_compatible(plus, minus)

    a['branches'][0]['branch_control_point_direction'] = plus
    b['branches'][0]['control_point_direction'] = minus
    assert loader_issues({'a.json': a, 'b.json': b}) != []  # broken pair

    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok'] and out['a_changed'] and out['b_changed']
    assert loader_issues({'a.json': out['a_doc'], 'b.json': out['b_doc']}) == []


def test_refresh_preserves_direction_sign():
    """A stored direction that is exactly the NEGATED tangent is loader-
    compatible and semantically meaningful; the refresh must not flip it."""
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1)
    negated = [-x for x in a['branches'][0]['control_point_direction']]
    a['branches'][0]['control_point_direction'] = list(negated)
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok'] and not out['a_changed']
    assert out['a_doc']['branches'][0]['control_point_direction'] == negated


def test_refresh_syncs_pending_on_existing_reciprocal():
    """An approval applied by the merge must reach the peer's existing
    reciprocal too — VC3D keeps the review state in lockstep on both
    refs."""
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1, pending=True)
    a['branches'][0].pop('pending')          # merged A carries the approval
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok'] and out['b_changed']
    assert not out['b_doc']['branches'][0].get('pending', False)
    assert loader_issues({'a.json': out['a_doc'], 'b.json': out['b_doc']}) == []


def test_branch_file_compared_as_basename():
    """The loader normalizes branch_file to its basename; a legacy
    'fibers/x.json' entry must be treated as the same peer."""
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1)
    a['branches'][0]['branch_file'] = 'fibers/b.json'
    assert fiber_merge.links_to(a, 'b.json') == a['branches']
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok'] and not out['b_changed']
    base = make_fiber(BASE_CPS, branches=[copy.deepcopy(a['branches'][0])])
    local = make_fiber_with_marker(base)
    remote = copy.deepcopy(base)
    remote['tags'] = ['other']
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['peer_files'] == ['b.json']


def test_reflag_beats_untouched_approval():
    """Base-aware review-state merge: deliberately re-flagging an approved
    link on one side beats the untouched approval on the other."""
    approved = link('kb_a.json', BASE_CPS[2], OTHER, 2, pending=False)
    reflagged = link('kb_a.json', BASE_CPS[2], OTHER, 2, pending=True)
    base = make_fiber(BASE_CPS, branches=[approved])
    local = make_fiber(BASE_CPS, branches=[reflagged], generation=2)
    remote = make_fiber(BASE_CPS, branches=[approved], tags=['x'], generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['branches'][0]['pending'] is True
    assert result['stats']['links_approved'] == 0


@pytest.mark.parametrize('mutation', [
    {'generation': 'x'},
    {'generation': [1]},
    {'control_points': [['a', 0, 0]]},
    {'control_points': [[float('nan'), 0.0, 0.0]]},
    {'tags': [{'k': 1}]},
    {'tags': 'foo'},
])
def test_malformed_docs_are_rejected_not_crashes(mutation):
    """Remote-controlled malformed input must degrade to a conflict (so
    vc_sync falls back to manual resolution), never raise (the #1246
    review's M6 crash inputs)."""
    doc = make_fiber(BASE_CPS)
    doc.update(mutation)
    assert not fiber_merge.is_fiber_doc(doc)
    good = make_fiber(BASE_CPS, tags=['x'])
    other = make_fiber(BASE_CPS, tags=['y'])
    result = merge_fibers(doc, good, other)
    assert not result['ok']
    out = refresh_pair_links(doc, good, 'a.json', 'b.json')
    assert not out['ok']


# --- refresh_pair_links: the cross-file consistency pass ------------------


def test_refresh_consistent_pair_changes_nothing():
    a, b = make_pair('a.json', 'b.json', BASE_CPS,
                     [cp(i, dz=50.0) for i in range(4)], 2, 1)
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok']
    assert not out['a_changed'] and not out['b_changed']
    assert out['a_doc'] == a and out['b_doc'] == b


def test_refresh_restores_missing_reciprocal():
    """The approval-beats-deletion case: the peer (synced from the deleting
    machine) lost its entry; the pass must restore an exact reciprocal."""
    a, b = make_pair('a.json', 'b.json', BASE_CPS,
                     [cp(i, dz=50.0) for i in range(4)], 2, 1)
    b['branches'] = []
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok'] and out['b_changed'] and not out['a_changed']
    assert loader_issues({'a.json': out['a_doc'], 'b.json': out['b_doc']}) == []


def test_refresh_fixes_stale_peer_index():
    """An index shift in merged A must be mirrored into B's
    branch_control_point_index (the loader's reciprocity is index-exact)."""
    a, b = make_pair('a.json', 'b.json', BASE_CPS,
                     [cp(i, dz=50.0) for i in range(4)], 2, 1)
    # Simulate A having gained a control point before the anchor: the merge
    # re-anchored A's own entry, but B still points at the old index.
    a['control_points'] = [cp(-1)] + a['control_points']
    a['line_points'] = line_for(a['control_points'])
    a['branches'][0]['control_point_index'] = 3
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok'] and out['b_changed']
    assert out['b_doc']['branches'][0]['branch_control_point_index'] == 3
    assert loader_issues({'a.json': out['a_doc'], 'b.json': out['b_doc']}) == []


def test_refresh_mirrors_deletion():
    a, b = make_pair('a.json', 'b.json', BASE_CPS,
                     [cp(i, dz=50.0) for i in range(4)], 2, 1)
    base_a = copy.deepcopy(a)
    a['branches'] = []          # the merge deleted the link
    out = refresh_pair_links(a, b, 'a.json', 'b.json', base_doc=base_a)
    assert out['ok'] and out['b_changed']
    assert out['b_doc']['branches'] == []


def test_refresh_leaves_unrelated_entries_alone():
    a, b = make_pair('a.json', 'b.json', BASE_CPS,
                     [cp(i, dz=50.0) for i in range(4)], 2, 1)
    orphan = link('a.json', b['control_points'][3], [1.0, 2.0, 3.0], 3)
    b['branches'].append(orphan)
    base_a = copy.deepcopy(a)
    out = refresh_pair_links(a, b, 'a.json', 'b.json', base_doc=base_a)
    assert out['ok']
    # The orphan half-link (a pair A never tracked) is preserved verbatim
    assert orphan in out['b_doc']['branches']


def test_refresh_unresolvable_anchor_is_a_conflict():
    a, b = make_pair('a.json', 'b.json', BASE_CPS,
                     [cp(i, dz=50.0) for i in range(4)], 2, 1)
    a['branches'][0]['branch_control_point_position'] = [5.0, 5.0, 5.0]
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert not out['ok']
    assert out['conflicts']


def test_merged_pair_end_to_end_is_loader_safe():
    """Full pipeline: two-sided geometry merge of a linked fiber, then the
    consistency pass — the pair must clear every loader check."""
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    base_a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1)
    local = copy.deepcopy(base_a)
    local['control_points'][6] = cp(6, dz=5.0)     # local moves CP 6
    local['line_points'] = line_for(local['control_points'])
    local['generation'] = 2
    remote = copy.deepcopy(base_a)
    remote['control_points'] = ([cp(-2), cp(-1)] +
                                remote['control_points'])  # remote prepends
    remote['line_points'] = line_for(remote['control_points'])
    remote['branches'][0]['control_point_index'] = 4
    remote['generation'] = 2

    result = merge_fibers(base_a, local, remote)
    assert result['ok']
    assert REOPTIMIZE_TAG in result['merged']['tags']
    assert result['peer_files'] == ['b.json']

    out = refresh_pair_links(result['merged'], b, 'a.json', 'b.json',
                             base_doc=base_a)
    assert out['ok']
    docs = {'a.json': out['a_doc'], 'b.json': out['b_doc']}
    assert loader_issues(docs) == []


# --- round-2 review findings ----------------------------------------------


def test_manual_hv_tag_merges_base_aware():
    """hv_classification.manual_tag is a USER decision: the side that
    changed it relative to the base wins, regardless of which side carries
    the geometry/generation."""
    base = make_fiber(BASE_CPS)
    base['hv_classification'] = {'manual_tag': 'unknown', 'score': 0.5}
    local = copy.deepcopy(base)
    local['hv_classification'] = {'manual_tag': 'horizontal', 'score': 0.5}
    local['generation'] = 2
    remote = copy.deepcopy(base)
    remote['tags'] = ['unrelated']
    remote['generation'] = 5          # remote is the carrier
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['hv_classification']['manual_tag'] == 'horizontal'
    assert any('manual hv tag' in n for n in result['notes'])


def test_manual_hv_tag_both_changed_conflicts():
    base = make_fiber(BASE_CPS)
    base['hv_classification'] = {'manual_tag': 'unknown'}
    local = copy.deepcopy(base)
    local['hv_classification'] = {'manual_tag': 'horizontal'}
    remote = copy.deepcopy(base)
    remote['hv_classification'] = {'manual_tag': 'vertical'}
    result = merge_fibers(base, local, remote)
    assert not result['ok']
    assert any('manual_tag' in c for c in result['conflicts'])


def test_legacy_link_direction_entries_are_opaque():
    """The loader strips entries carrying the obsolete link_direction key
    at parse time; the merge must not treat one as decided truth (its
    reciprocal would be cemented into the peer, then torn down by the
    loader's destructive repair)."""
    legacy = link('kb_a.json', BASE_CPS[2], OTHER, 2)
    legacy['link_direction'] = [1.0, 0.0, 0.0]
    base = make_fiber(BASE_CPS, branches=[legacy])
    local = make_fiber(BASE_CPS, branches=[legacy], tags=['x'])
    remote = make_fiber(BASE_CPS, branches=[legacy], tags=['y'])
    result = merge_fibers(base, local, remote)
    assert result['ok']
    # Carried as an opaque value, never mirrored into a peer
    assert result['merged']['branches'] == [legacy]
    assert result['peer_files'] == []


def test_is_fiber_doc_rejects_unloadable_variants():
    doc = make_fiber(BASE_CPS)
    doc['version'] = 2                # loader: "Unsupported ... version"
    assert not fiber_merge.is_fiber_doc(doc)
    doc = make_fiber(BASE_CPS)
    doc['tags'] = None                # loader: "tags must be an array"
    assert not fiber_merge.is_fiber_doc(doc)


def test_short_circuit_merges_still_report_peers():
    """One-side-unchanged merges take the other side wholesale; peer_files
    is still reported so the (no-op for consistent pairs) consistency pass
    can verify — a crash mid-VC3D-save breaks the lockstep-write invariant
    the wholesale adoption relies on."""
    entry = link('kb_a.json', BASE_CPS[2], OTHER, 2)
    base = make_fiber(BASE_CPS, branches=[entry])
    remote = make_fiber(BASE_CPS, branches=[entry], tags=['new'], generation=4)
    result = merge_fibers(base, copy.deepcopy(base), remote)
    assert result['ok']
    assert result['merged'] == remote
    assert result['peer_files'] == ['kb_a.json']


def test_snapped_direction_tolerates_ulp_noise():
    """A direction differing from the recomputed tangent by an ulp (e.g. a
    VC3D build with contracted float math) must be accepted as-is, or
    every refresh would rewrite (and re-upload) every direction."""
    tangent = [1.0, 0.0, 0.0]
    ulp_off = [1.0 + 1e-14, 1e-14, 0.0]
    assert fiber_merge._snapped_direction(ulp_off, tangent) == ulp_off
    assert fiber_merge._snapped_direction([-1.0 - 1e-14, 0.0, 0.0],
                                          tangent) == [-1.0 - 1e-14, 0.0, 0.0]
    rotated = [math.cos(0.001), math.sin(0.001), 0.0]  # well past ulp noise
    assert fiber_merge._snapped_direction(rotated, tangent) == tangent


def test_refresh_bumps_peer_generation_only_when_changed():
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1)
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['b_doc']['generation'] == b['generation']  # untouched
    b_missing = copy.deepcopy(b)
    b_missing['branches'] = []
    out = refresh_pair_links(a, b_missing, 'a.json', 'b.json')
    assert out['b_changed']
    assert out['b_doc']['generation'] == b['generation'] + 1


def test_duplicate_a_entries_claim_distinct_reciprocals():
    """Two pos_eq-identical A entries must not both snap the same B
    reciprocal, leaving B one entry short for the loader (GIGO input, but
    cheap to handle)."""
    b_cps = [cp(i, dz=50.0) for i in range(4)]
    a, b = make_pair('a.json', 'b.json', BASE_CPS, b_cps, 2, 1)
    a['branches'].append(copy.deepcopy(a['branches'][0]))
    b['branches'].append(copy.deepcopy(b['branches'][0]))
    out = refresh_pair_links(a, b, 'a.json', 'b.json')
    assert out['ok']
    assert len(out['b_doc']['branches']) == 2  # no third entry restored


def test_v3_separated_span_results_merge_atomically_without_generation_loss():
    base = make_v3_fiber(BASE_CPS)
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    local['generation'] = 2
    remote['generation'] = 9
    set_v3_span(local, 1, goal='cspline', bend=1.5)
    set_v3_span(remote, 3, goal='lasagna', bend=-2.0)

    result = merge_fibers(base, local, remote)

    assert result['ok'], result['conflicts']
    merged = result['merged']
    assert merged['control_points'][1]['segment_to_next']['interp_goal'] == 'cspline'
    assert merged['control_points'][3]['segment_to_next']['interp_goal'] == 'lasagna'
    assert merged['line_points'][5:8] == local['line_points'][5:8]
    assert merged['line_points'][13:16] == remote['line_points'][13:16]
    assert REOPTIMIZE_TAG not in merged['tags']
    assert loader_issues({'dj_x_000001.json': merged}) == []


def test_v3_adjacent_changes_are_a_manual_conflict():
    base = make_v3_fiber(BASE_CPS)
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    set_v3_span(local, 1, goal='cspline', bend=1.0)
    set_v3_span(remote, 2, goal='lasagna', bend=-1.0)

    result = merge_fibers(base, local, remote)

    assert not result['ok']
    assert any('unchanged base span' in conflict for conflict in result['conflicts'])


def test_v3_same_span_different_results_are_a_manual_conflict():
    base = make_v3_fiber(BASE_CPS)
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    set_v3_span(local, 2, goal='cspline', bend=1.0)
    set_v3_span(remote, 2, goal='lasagna', bend=-1.0)

    result = merge_fibers(base, local, remote)

    assert not result['ok']
    assert any('changed differently' in conflict for conflict in result['conflicts'])


def test_v3_identical_two_sided_span_result_is_accepted():
    base = make_v3_fiber(BASE_CPS)
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    set_v3_span(local, 2, goal='cspline', bend=1.0)
    set_v3_span(remote, 2, goal='cspline', bend=1.0)
    local['tags'] = ['local']
    remote['tags'] = ['remote']

    result = merge_fibers(base, local, remote)

    assert result['ok'], result['conflicts']
    assert result['merged']['line_points'] == local['line_points']
    assert result['merged']['control_points'] == local['control_points']
    assert sorted(result['merged']['tags']) == ['local', 'remote']


def test_v3_separated_topology_changes_preserve_dense_results_and_descriptors():
    base = make_v3_fiber(BASE_CPS)
    inserted = [105.0, 200.0, 300.0]
    local = make_v3_fiber(BASE_CPS[:1] + [inserted] + BASE_CPS[1:], generation=2)
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[6] = cp(6, dz=-5.0)
    remote = make_v3_fiber(remote_cps, generation=3)

    result = merge_fibers(base, local, remote)

    assert result['ok'], result['conflicts']
    merged = result['merged']
    positions = [control['position'] for control in merged['control_points']]
    assert inserted in positions
    assert cp(6, dz=-5.0) in positions
    assert all('segment_to_next' in control
               for control in merged['control_points'][:-1])
    assert 'segment_to_next' not in merged['control_points'][-1]
    assert REOPTIMIZE_TAG not in merged['tags']
    assert loader_issues({'dj_x_000001.json': merged}) == []


def test_v3_overlapping_topology_changes_are_a_manual_conflict():
    base = make_v3_fiber(BASE_CPS)
    inserted = [115.0, 200.0, 300.0]
    local = make_v3_fiber(BASE_CPS[:2] + [inserted] + BASE_CPS[2:])
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[2] = cp(2, dz=-5.0)
    remote = make_v3_fiber(remote_cps)

    result = merge_fibers(base, local, remote)

    assert not result['ok']
    assert any('changed differently' in conflict for conflict in result['conflicts'])


def test_v3_optimization_mode_uses_base_aware_merge_not_newer_generation():
    base = make_v3_fiber(BASE_CPS, optimization_mode='lasagna')
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    local['optimization_mode'] = 'native_fiber_trace3d'
    local['generation'] = 2
    remote['tags'] = ['reviewed']
    remote['generation'] = 20

    result = merge_fibers(base, local, remote)

    assert result['ok'], result['conflicts']
    assert result['merged']['optimization_mode'] == 'native_fiber_trace3d'
    assert result['merged']['tags'] == ['reviewed']


def test_v3_unordered_control_point_line_mapping_is_a_manual_conflict():
    base = make_v3_fiber(BASE_CPS)
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    del local['line_points'][8]  # remove CP 2 from its ordered dense line
    remote['tags'] = ['remote-edit']

    assert fiber_merge.is_fiber_doc(local)
    result = merge_fibers(base, local, remote)

    assert not result['ok']
    assert any('not an ordered dense-line point' in conflict
               for conflict in result['conflicts'])


def test_fiber_doc_rejects_unknown_optimization_mode():
    doc = make_v3_fiber(BASE_CPS)
    doc['optimization_mode'] = 'guess'
    assert not fiber_merge.is_fiber_doc(doc)
    doc['optimization_mode'] = []
    assert not fiber_merge.is_fiber_doc(doc)


def test_v3_fiber_rejects_obsolete_segment_metadata_versions():
    doc = make_v3_fiber([cp(0), cp(1)])
    doc['control_points'][0]['segment_to_next']['metadata_version'] = 2
    assert not fiber_merge.is_fiber_doc(doc)


@pytest.mark.parametrize('mutation', ['missing_mode', 'missing_segment',
                                      'final_segment', 'nonpositive_scale'])
def test_v3_fiber_requires_complete_non_repaired_schema(mutation):
    doc = make_v3_fiber([cp(0), cp(1), cp(2)])
    if mutation == 'missing_mode':
        del doc['optimization_mode']
    elif mutation == 'missing_segment':
        del doc['control_points'][0]['segment_to_next']
    elif mutation == 'final_segment':
        doc['control_points'][-1]['segment_to_next'] = v3_segment()
    else:
        doc['control_points'][0]['segment_to_next']['trace_to_base_scale'] = 0

    assert not fiber_merge.is_fiber_doc(doc)
    good = make_v3_fiber([cp(0), cp(1), cp(2)])
    result = merge_fibers(doc, good, copy.deepcopy(good))
    assert not result['ok']
    assert result['merged'] is None
    assert result['conflicts']


def test_review_invalidated_when_merged_geometry_was_not_reviewed():
    """hendrik P1: one machine retraces, the other verifies the OLD
    geometry; the merged (new) geometry must come back needing review."""
    base = make_v3_fiber(BASE_CPS)
    base['tags'] = [fiber_merge.TRACE_NEEDS_REVIEW_TAG]
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    set_v3_span(local, 1, goal='global', bend=1.5)  # retrace, tag kept
    remote['tags'] = []                             # reviewed old geometry

    result = merge_fibers(base, local, remote)

    assert result['ok'], result['conflicts']
    merged = result['merged']
    assert merged['line_points'][5:8] == local['line_points'][5:8]
    assert fiber_merge.TRACE_NEEDS_REVIEW_TAG in merged['tags']
    assert any('re-added' in note for note in result['notes'])


def test_review_survives_when_merged_geometry_is_the_reviewed_geometry():
    base = make_v3_fiber(BASE_CPS)
    base['tags'] = [fiber_merge.TRACE_NEEDS_REVIEW_TAG]
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    local['tags'] = [fiber_merge.TRACE_NEEDS_REVIEW_TAG, 'zebra']
    remote['tags'] = []  # reviewed; geometry unchanged everywhere

    result = merge_fibers(base, local, remote)

    assert result['ok'], result['conflicts']
    merged = result['merged']
    assert fiber_merge.TRACE_NEEDS_REVIEW_TAG not in merged['tags']
    assert 'zebra' in merged['tags']


def test_span_mix_of_two_reviewed_sides_needs_a_fresh_review():
    """Both sides were reviewed, but the span-atomic merge combines
    geometry no single reviewer ever saw."""
    base = make_v3_fiber(BASE_CPS)
    local = copy.deepcopy(base)
    remote = copy.deepcopy(base)
    set_v3_span(local, 1, goal='cspline', bend=1.5)
    set_v3_span(remote, 3, goal='lasagna', bend=-2.0)

    result = merge_fibers(base, local, remote)

    assert result['ok'], result['conflicts']
    merged = result['merged']
    assert fiber_merge.TRACE_NEEDS_REVIEW_TAG in merged['tags']
