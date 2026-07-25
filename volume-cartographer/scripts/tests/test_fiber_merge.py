"""Unit tests for fiber_merge — pure JSON in/out, no I/O."""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fiber_merge
from fiber_merge import merge_fibers, REOPTIMIZE_TAG


def cp(i, dz=0.0):
    """A control point on a simple line, spaced 10 apart."""
    return [100.0 + 10.0 * i, 200.0, 300.0 + dz]


def line_for(cps, samples_per_span=4):
    """A dense polyline passing through the control points."""
    line = []
    for a, b in zip(cps[:-1], cps[1:]):
        for s in range(samples_per_span):
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


def test_disjoint_cp_edits_merge():
    base = make_fiber(BASE_CPS)
    local_cps = copy.deepcopy(BASE_CPS)
    local_cps[1] = cp(1, dz=5.0)          # local moves CP 1
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[6] = cp(6, dz=-5.0)        # remote moves CP 6
    local = make_fiber(local_cps, generation=2)
    remote = make_fiber(remote_cps, generation=2)

    result = merge_fibers(base, local, remote)
    assert result['ok']
    merged = result['merged']['control_points']
    assert merged[1] == cp(1, dz=5.0)
    assert merged[6] == cp(6, dz=-5.0)
    assert len(merged) == 8
    assert result['stats']['geometry_merged']
    assert REOPTIMIZE_TAG in result['merged']['tags']


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


def test_base_aware_deletion():
    l1 = link('kb_a.json', BASE_CPS[2], OTHER, 2)
    base = make_fiber(BASE_CPS, branches=[l1])
    local = make_fiber(BASE_CPS, branches=[l1])          # untouched
    remote = make_fiber(BASE_CPS, branches=[])           # deleted it
    result = merge_fibers(base, make_fiber_with_marker(local), remote)
    assert result['ok']
    assert result['merged']['branches'] == []
    assert result['stats']['links_deleted'] == 1


def make_fiber_with_marker(fiber):
    """Make a doc differ from base somewhere harmless so the local==base
    short-circuit doesn't bypass the branch merge under test."""
    fiber = copy.deepcopy(fiber)
    fiber['tags'] = fiber.get('tags', []) + ['marker']
    return fiber


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


def test_line_points_taken_from_geometry_owner():
    base = make_fiber(BASE_CPS)
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[6] = cp(6, dz=-5.0)
    local = make_fiber(BASE_CPS, tags=['meta-only'], generation=2)
    remote = make_fiber(remote_cps, generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['line_points'] == remote['line_points']


def test_line_splice_between_owners():
    base = make_fiber(BASE_CPS)
    local_cps = copy.deepcopy(BASE_CPS)
    local_cps[1] = cp(1, dz=5.0)
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[6] = cp(6, dz=-5.0)
    local = make_fiber(local_cps, generation=2)
    remote = make_fiber(remote_cps, generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    line = result['merged']['line_points']
    assert len(line) > 0
    # Early line follows local's moved CP1, late line follows remote's CP6
    def nearest_d2(line, p):
        return min(sum((a - b) ** 2 for a, b in zip(q, p)) for q in line)
    assert nearest_d2(line, cp(1, dz=5.0)) < 1.0
    assert nearest_d2(line, cp(6, dz=-5.0)) < 1.0
    assert REOPTIMIZE_TAG in result['merged']['tags']


def test_filename_mismatch_is_conflict():
    base = make_fiber(BASE_CPS)
    local = make_fiber(BASE_CPS, tags=['a'])
    remote = make_fiber(BASE_CPS, tags=['b'], filename='other.json')
    result = merge_fibers(base, local, remote)
    assert not result['ok']


def test_link_anchor_removed_by_merged_geometry_dropped_with_note():
    base = make_fiber(BASE_CPS)
    # local deletes CP 3; remote links at CP 3 (positions identical to base)
    local_cps = BASE_CPS[:3] + BASE_CPS[4:]
    local = make_fiber(local_cps, generation=2)
    remote = make_fiber(BASE_CPS,
                        branches=[link('kb_a.json', BASE_CPS[3], OTHER, 3)],
                        generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['branches'] == []
    assert any('removed by the merged geometry' in n for n in result['notes'])


# --- review findings (PR #1223) ---


def test_opaque_branch_entries_survive_merge():
    """Structurally unparseable entries must never vanish from a clean
    merge (they'd be data loss the C++ loader deliberately preserves)."""
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
    """A dict entry with missing/non-vector positions must not reach the
    geometric code paths (previously an uncaught TypeError)."""
    broken = {'branch_file': 'kb_a.json', 'control_point_position': 'oops'}
    base = make_fiber(BASE_CPS, branches=[broken])
    local = make_fiber(BASE_CPS, branches=[broken], tags=['from-local'])
    remote = make_fiber(BASE_CPS, branches=[broken], tags=['from-remote'])
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['branches'] == [broken]


def test_link_identity_is_tolerance_based_not_rounding_based():
    """Positions within POS_TOL but on opposite sides of a rounding bucket
    must still be the SAME link (previously produced a duplicate)."""
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


def test_link_survives_jittered_geometry_within_tolerance():
    """Re-resolving link anchors must use the same per-axis pos_eq predicate
    as everything else. A pure Euclidean^2 <= POS_TOL^2 bound was up to 3x
    stricter and silently dropped a valid link when one side's geometry
    carried sub-tolerance jitter on every axis."""
    jitter = [[p[0] + 8e-7, p[1] + 8e-7, p[2] + 8e-7] for p in BASE_CPS]
    base = make_fiber(BASE_CPS)
    local = make_fiber(jitter, tags=['touched'], generation=2)  # "unchanged"
    remote = make_fiber(BASE_CPS,
                        branches=[link('kb_a.json', BASE_CPS[2], OTHER, 2)],
                        generation=2)
    result = merge_fibers(base, local, remote)
    assert result['ok']
    branches = result['merged']['branches']
    assert len(branches) == 1, result['notes']
    assert fiber_merge.pos_eq(
        result['merged']['control_points'][branches[0]['control_point_index']],
        branches[0]['control_point_position'])


def test_line_only_reoptimization_is_never_dropped_silently():
    """A side can re-optimize line_points without moving any control point;
    region ownership cannot see that, but it must not vanish untagged."""
    base = make_fiber(BASE_CPS)
    local = make_fiber(BASE_CPS, tags=['meta-edit'], generation=2)
    remote = make_fiber(BASE_CPS, generation=2)
    remote['line_points'] = [[p[0], p[1] + 0.5, p[2]]
                             for p in remote['line_points']]
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['line_points'] == local['line_points']
    assert any('re-optimized the line' in n for n in result['notes'])
    assert REOPTIMIZE_TAG in result['merged']['tags']


def test_splice_rejected_when_anchors_not_on_both_lines():
    """Anchors must actually lie on both polylines; unrelated lines fall
    back to the local line + reoptimization tag instead of a fake splice."""
    base = make_fiber(BASE_CPS)
    local_cps = copy.deepcopy(BASE_CPS)
    local_cps[1] = cp(1, dz=5.0)          # local owns a region
    remote_cps = copy.deepcopy(BASE_CPS)
    remote_cps[6] = cp(6, dz=-5.0)        # remote owns a region
    local = make_fiber(local_cps, generation=2)
    remote = make_fiber(remote_cps, generation=2)
    # Remote's line is unrelated garbage far from every anchor
    remote['line_points'] = [[-900.0 - i, -900.0, -900.0] for i in range(40)]
    result = merge_fibers(base, local, remote)
    assert result['ok']
    assert result['merged']['line_points'] == local['line_points']
    assert any('splice failed' in n for n in result['notes'])
    assert REOPTIMIZE_TAG in result['merged']['tags']
