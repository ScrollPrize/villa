"""Synthetic end-to-end test for bracket_fiber_links.

Builds a two-and-a-half revolution horizontal spiral fiber and one vertical
fiber sitting 10 voxels behind its first revolution, then checks that the tool
finds exactly that link, promotes the two crossing points to control points,
renumbers existing branch indices, writes reciprocal branches that VC3D's and
the fitter's loaders accept, and is idempotent.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import bracket_fiber_links as bfl  # noqa: E402
from umbilicus import json_umbilicus_z_to_yx  # noqa: E402

SCALE = 0.25          # fitter voxels per file unit
UMB_YX = (4000.0, 4000.0)
PITCH = 25.0          # fitter voxels per revolution


def _to_file_xyz(zyx):
    return (np.asarray(zyx, dtype=np.float64)[:, ::-1] / SCALE).tolist()


def _segment():
    # A valid version-3 lasagna-fallback segment, as vc3d_fiber_format requires.
    return {
        'optimizer': 'native_fiber_trace3d', 'metadata_version': 3, 'tracer_version': 2,
        'interp_goal': 'global', 'interp_mode': 'lasagna', 'metric': None, 'msg': 'lasagna',
        'normal_manifest': '', 'fiber_manifest': '', 'trace_to_base_scale': 2.0,
        'meeting_error_base_voxels': None, 'meeting_error_ratio': None, 'meeting_source': '',
        'failure_code': '', 'failure_detail': '', 'lasagna_failure_code': '',
        'lasagna_failure_detail': '',
        'config': {
            'step_voxels': 4.0, 'cone_angle_degrees': 25.0, 'cone_angle_step_degrees': 5.0,
            'cone_grid_size': 25, 'beam_width': 8, 'beam_prune_distance_voxels': 1.0,
            'beam_lookahead_steps': 2, 'smoothness_weight': 2.0, 'smoothness_normal_weight': 0.1,
            'smoothness_tangent_weight': 10.0, 'smoothness_free_angle_degrees': 0.0,
            'cumulative_smoothness_steps': 4, 'cumulative_smoothness_tangent_weight': 2.0,
            'initial_free_angle_degrees': 0.0, 'max_step_factor': 3.0,
            'meeting_accept_max_error_ratio': 0.1, 'endpoint_accept_threshold_base_voxels': 20.0,
        },
    }


def _fiber_doc(zyx, control_every, tag, branches=None, version=3):
    line_xyz = _to_file_xyz(zyx)
    control_indices = list(range(0, len(line_xyz), control_every))
    if control_indices[-1] != len(line_xyz) - 1:
        control_indices.append(len(line_xyz) - 1)
    if version == 3:
        controls = []
        for k, i in enumerate(control_indices):
            entry = {'position': line_xyz[i]}
            if k + 1 < len(control_indices):
                entry['segment_to_next'] = _segment()
            controls.append(entry)
    else:
        controls = [line_xyz[i] for i in control_indices]
    return {
        'type': 'vc3d_fiber', 'version': version, 'generation': 3,
        'optimization_mode': 'native_fiber_trace3d',
        'line_points': line_xyz, 'control_points': controls,
        'branches': branches or [],
        'hv_classification': {'automatic_tag': tag, 'automatic_certainty': 1.0, 'manual_tag': ''},
        'sequence': 1, 'started_at': '20260101T000000000', 'tags': [], 'username': 'test',
        'filename': 'x.json',
    }, control_indices


def _spiral(theta0, theta1, r0, z, n):
    theta = np.linspace(theta0, theta1, n)
    r = r0 + PITCH * theta / (2 * np.pi)
    y = UMB_YX[0] + r * np.sin(theta)
    x = UMB_YX[1] + r * np.cos(theta)
    return np.stack([np.full(n, z), y, x], axis=1)


@pytest.fixture
def dataset(tmp_path):
    fibers = tmp_path / 'fibers'
    fibers.mkdir()
    with open(tmp_path / 'umbilicus.json', 'wt') as fp:
        json.dump({'control_points': [{'x': UMB_YX[1], 'y': UMB_YX[0], 'z': z, 'score': 100}
                                      for z in (0.0, 5000.0)]}, fp)
    # Horizontal: 2.5 revolutions at z=1000, every 2 fitter voxels along the path.
    h_zyx = _spiral(0.0, 2.5 * 2 * np.pi, 1000.0, 1000.0, 9000)
    h_doc, h_controls = _fiber_doc(h_zyx, 400, 'H')
    # Vertical at theta = pi, radius 10 voxels outside revolution 0 there
    # (r_h(pi) = 1000 + 12.5), spanning z 700..1300.
    theta_v = np.pi
    r_v = 1000.0 + PITCH * theta_v / (2 * np.pi) + 10.0
    n_v = 300
    zs = np.linspace(700.0, 1300.0, n_v)
    v_zyx = np.stack([zs, np.full(n_v, UMB_YX[0] + r_v * np.sin(theta_v)),
                      np.full(n_v, UMB_YX[1] + r_v * np.cos(theta_v))], axis=1)
    v_doc, v_controls = _fiber_doc(v_zyx, 50, 'V')
    # A third, unrelated horizontal that already links into the vertical's last
    # control point, to exercise cross-file renumbering.
    o_zyx = _spiral(0.0, 0.3, 3000.0, 1250.0, 200)
    o_doc, o_controls = _fiber_doc(o_zyx, 50, 'H')
    last_v = len(v_controls) - 1
    o_doc['branches'].append({
        'control_point_index': 0, 'branch_fiber_id': 0, 'branch_control_point_index': last_v,
        'control_point_direction': [1.0, 0.0, 0.0], 'branch_control_point_direction': [0.0, 0.0, 1.0],
        'control_point_position': o_doc['line_points'][0],
        'branch_control_point_position': v_doc['line_points'][v_controls[-1]],
        'branch_file': 'test_20260101T000000000_000007.json'})
    v_doc['branches'].append({
        'control_point_index': last_v, 'branch_fiber_id': 0, 'branch_control_point_index': 0,
        'control_point_direction': [0.0, 0.0, 1.0], 'branch_control_point_direction': [1.0, 0.0, 0.0],
        'control_point_position': v_doc['line_points'][v_controls[-1]],
        'branch_control_point_position': o_doc['line_points'][0],
        'branch_file': 'other.json'})
    v_doc['username'] = 'test'; v_doc['started_at'] = '20260101T000000000'; v_doc['sequence'] = 7
    v_doc['filename'] = 'test_20260101T000000000_000007.json'
    copy_doc = json.loads(json.dumps(v_doc))
    copy_doc['branches'] = []
    for name, doc in (('horizontal.json', h_doc), ('test_20260101T000000000_000007.json', v_doc),
                      ('other.json', o_doc), ('12.json', copy_doc)):
        with open(fibers / name, 'wt') as fp:
            json.dump(doc, fp)
    return tmp_path


def _run(dataset, *args):
    cmd = [sys.executable, os.path.join(ROOT, 'bracket_fiber_links.py'), str(dataset / 'fibers'), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_detection_finds_the_bracketed_vertical(dataset):
    umb = json_umbilicus_z_to_yx(str(dataset / 'umbilicus.json'))
    fibers = bfl.load_fibers(str(dataset / 'fibers'), umb, coordinate_scale=SCALE,
                             min_z_fraction=0.8, min_auto_certainty=0.5)
    assert {f.tag for f in fibers.values()} == {'H', 'V'}
    dropped = bfl.drop_duplicate_copies(fibers)
    assert dropped == {'12.json': 'test_20260101T000000000_000007.json'}
    links = bfl.find_bracket_links(fibers, theta_bin=0.005, z_tolerance=15.0,
                                   min_gap=8.0, max_gap=60.0, margin=0.0)
    assert {(l.vertical, l.horizontal) for l in links} == {('test_20260101T000000000_000007.json', 'horizontal.json')}
    link = links[0]
    assert link.revolution == 0
    assert abs(link.gap - PITCH) < 1.0
    assert 9.0 < link.behind_by < 11.0
    assert abs(link.z - 1000.0) < 3.0
    # The horizontal endpoint is on revolution 0 near theta = pi, a few voxels
    # from the vertical point.
    h = fibers['horizontal.json']
    v = fibers['test_20260101T000000000_000007.json']
    d = np.linalg.norm(h.zyx[link.horizontal_line_index] - v.zyx[link.vertical_line_index])
    assert 9.0 < d < 12.0


def test_dry_run_writes_nothing_and_apply_writes_valid_reciprocal_links(dataset):
    before = {n: open(dataset / 'fibers' / n).read() for n in os.listdir(dataset / 'fibers')}
    out = _run(dataset, '--report', str(dataset / 'report.json'))
    assert 'planned new links: 1' in out and 'dry run' in out
    assert {n: open(dataset / 'fibers' / n).read() for n in os.listdir(dataset / 'fibers')} == before
    report = json.load(open(dataset / 'report.json'))
    assert len(report['links']) == 1 and not report['problems']

    out = _run(dataset, '--apply', '--backup-dir', str(dataset / 'bak'))
    assert 'wrote 3 files' in out
    assert 'fitter link resolution after write: 2 links' in out
    assert sorted(os.listdir(dataset / 'bak')) == ['horizontal.json', 'other.json', 'test_20260101T000000000_000007.json']
    # The numeric duplicate copy of the vertical is left untouched.
    assert json.load(open(dataset / 'fibers' / '12.json'))['branches'] == []

    h = json.load(open(dataset / 'fibers' / 'horizontal.json'))
    v = json.load(open(dataset / 'fibers' / 'test_20260101T000000000_000007.json'))
    o = json.load(open(dataset / 'fibers' / 'other.json'))
    # One promoted control point each, generation bumped, segment inherited.
    assert len(h['control_points']) == len(json.loads(before['horizontal.json'])['control_points']) + 1
    assert len(v['control_points']) == len(json.loads(before['test_20260101T000000000_000007.json'])['control_points']) + 1
    assert h['generation'] == 4 and v['generation'] == 4 and o['generation'] == 4
    for doc in (h, v):
        positions = [c['position'] for c in doc['control_points']]
        idx = [doc['line_points'].index(p) for p in positions]
        assert idx == sorted(idx) and len(set(idx)) == len(idx)
        assert all('segment_to_next' in c for c in doc['control_points'][:-1])
        assert 'segment_to_next' not in doc['control_points'][-1]
    # Reciprocal new branches referencing each other's promoted control points.
    hv = [b for b in h['branches'] if b['branch_file'] == 'test_20260101T000000000_000007.json']
    vh = [b for b in v['branches'] if b['branch_file'] == 'horizontal.json']
    assert len(hv) == 1 and len(vh) == 1
    assert hv[0]['control_point_index'] == vh[0]['branch_control_point_index']
    assert vh[0]['control_point_index'] == hv[0]['branch_control_point_index']
    assert h['control_points'][hv[0]['control_point_index']]['position'] == hv[0]['control_point_position']
    assert v['control_points'][vh[0]['control_point_index']]['position'] == vh[0]['control_point_position']
    assert 'pending' not in hv[0]
    # The pre-existing link into the vertical's last control point was renumbered
    # in both the vertical's own branch and the untouched other fiber's branch...
    last_v = len(v['control_points']) - 1
    vo = [b for b in v['branches'] if b['branch_file'] == 'other.json'][0]
    assert vo['control_point_index'] == last_v
    # ...which means other.json had to be rewritten too, even without new links.
    ov = [b for b in o['branches'] if b['branch_file'] == 'test_20260101T000000000_000007.json'][0]
    assert ov['branch_control_point_index'] == last_v

    # Idempotent: the pair is now linked, so nothing more is planned.
    out = _run(dataset)
    assert 'planned new links: 0' in out


def test_pending_flag(dataset):
    _run(dataset, '--apply', '--pending', '--backup-dir', str(dataset / 'bak'))
    v = json.load(open(dataset / 'fibers' / 'test_20260101T000000000_000007.json'))
    assert [b for b in v['branches'] if b['branch_file'] == 'horizontal.json'][0]['pending'] is True
