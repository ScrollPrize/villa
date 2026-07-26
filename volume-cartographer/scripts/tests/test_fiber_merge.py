"""Unit tests for fiber_merge — pure JSON in/out, no I/O.

The high-value assertions here are loader-safety checks: `loader_issues`
below is a mini-port of VC3D's load-time validation (the C1/C2/C3
invariants from fiber_merge's module docstring). Merge output that fails
those checks gets destroyed by an unmodified VC3D, so every clean-merge
test runs its output through them.
"""
import copy
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


def loader_issues(docs_by_name):
    """Mini-port of VC3D's load-time fiber validation. Returns a list of
    issue strings; [] means an unmodified VC3D loads the set cleanly."""
    issues = []
    for name, doc in docs_by_name.items():
        cps = doc['control_points']
        line = doc['line_points']
        # C1: control points must be an ordered subset of line_points
        # within 1e-8 (validateFiberInputControlPoints) — fatal.
        li = 0
        for k, point in enumerate(cps):
            while li < len(line) and any(abs(line[li][j] - point[j]) > 1e-8
                                         for j in range(3)):
                li += 1
            if li >= len(line):
                issues.append(f"{name}: control point {k} not on line (fatal)")
                break
            li += 1
        for entry in doc.get('branches', []):
            if not isinstance(entry, dict) or 'control_point_index' not in entry:
                continue
            i = entry['control_point_index']
            if not (0 <= i < len(cps)):
                issues.append(f"{name}: local CP index out of range")
                continue
            if not fiber_merge.pos_eq(cps[i], entry['control_point_position']):
                issues.append(f"{name}: local CP position mismatch")
                continue
            if len(line) >= 2:
                tangent = fiber_merge.endpoint_tangent(
                    line, entry['control_point_position'])
                if not fiber_merge.directions_compatible(
                        entry['control_point_direction'], tangent):
                    issues.append(f"{name}: branch endpoint direction mismatch")
                    continue
            target = docs_by_name.get(entry['branch_file'])
            if target is None:
                issues.append(f"{name}: missing linked fiber "
                              f"{entry['branch_file']}")
                continue
            tcps = target['control_points']
            tline = target['line_points']
            j = entry['branch_control_point_index']
            if not (0 <= j < len(tcps)):
                issues.append(f"{name}: linked CP index out of range")
                continue
            if not fiber_merge.pos_eq(tcps[j],
                                      entry['branch_control_point_position']):
                issues.append(f"{name}: linked CP position mismatch")
                continue
            if len(tline) >= 2:
                tangent = fiber_merge.endpoint_tangent(
                    tline, entry['branch_control_point_position'])
                if not fiber_merge.directions_compatible(
                        entry['branch_control_point_direction'], tangent):
                    issues.append(f"{name}: linked endpoint direction mismatch")
                    continue
            # C2: index-exact reciprocity, plus positions and directions
            reciprocal = any(
                c.get('branch_file') == name and
                c.get('control_point_index') == j and
                c.get('branch_control_point_index') == i and
                fiber_merge.pos_eq(c.get('control_point_position'),
                                   entry['branch_control_point_position']) and
                fiber_merge.pos_eq(c.get('branch_control_point_position'),
                                   entry['control_point_position']) and
                fiber_merge.directions_compatible(
                    c.get('control_point_direction'),
                    entry['branch_control_point_direction']) and
                fiber_merge.directions_compatible(
                    c.get('branch_control_point_direction'),
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


def test_rebound_link_survives_pos_eq_but_not_euclidean_ball():
    """Regression for the #1223 review bug: a link whose anchor matches a
    merged control point per pos_eq (per-axis 1e-6) but not within a
    Euclidean 1e-6 ball must be re-anchored, not dropped."""
    base = make_fiber(BASE_CPS)
    jittered = [[p[0] + 8e-7, p[1] + 8e-7, p[2] + 8e-7] for p in BASE_CPS]
    local = make_fiber(jittered, generation=2)  # "unchanged" per pos_eq
    local['tags'] = ['touched']
    remote = make_fiber(BASE_CPS,
                        branches=[link('kb_a.json', BASE_CPS[2], OTHER, 2)],
                        generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert len(result['merged']['branches']) == 1


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
