"""Unit tests for fiber_qa on small synthetic fibers."""
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fiber_merge
import fiber_qa


STEP = 8.0


def straight(n, start=(1000.0, 2000.0, 3000.0), direction=(1.0, 0.0, 0.0),
             step=STEP):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    return [list(np.asarray(start) + i * step * d) for i in range(n)]


def wavy(n, start=(1000.0, 2000.0, 3000.0), amplitude=30.0, period=400.0):
    """A gently curving line, like a traced fiber, about STEP apart."""
    return [[start[0] + i * STEP,
             start[1] + amplitude * math.sin(2 * math.pi * i * STEP / period),
             start[2] + 0.2 * i] for i in range(n)]


def fiber(line, cp_every=10, name='dj_20260101T000000000_000001.json',
          tags=None, cp_tags=None):
    """A v3-style fiber whose control points are every cp_every-th point."""
    indices = list(range(0, len(line), cp_every))
    if indices[-1] != len(line) - 1:
        indices.append(len(line) - 1)
    cps = []
    for k, i in enumerate(indices):
        cp = {'position': list(line[i])}
        if k + 1 < len(indices):
            cp['segment_to_next'] = {'interp_mode': 'trace', 'metric': 1.0}
        if cp_tags and k in cp_tags:
            cp['tags'] = list(cp_tags[k])
        cps.append(cp)
    return {
        'type': 'vc3d_fiber', 'version': 3, 'filename': name,
        'username': 'dj', 'generation': 1,
        'control_points': cps,
        'line_points': [list(p) for p in line],
        'branches': [], 'adjacent_branches': [],
        'tags': list(tags or []),
    }


def opts(**over):
    out = dict(fiber_qa.DEFAULTS)
    out.update(over)
    return out


def checks_of(findings, check):
    return [f for f in findings if f['check'] == check]


def test_close_pairs_matches_brute_force():
    rng = np.random.default_rng(3)
    points = rng.uniform(0.0, 60.0, (400, 3))
    for radius in (2.0, 5.0, 17.0):
        a, b = fiber_qa.close_pairs(points, radius)
        got = {tuple(sorted(p)) for p in zip(a.tolist(), b.tolist())}
        want = {(i, j) for i in range(len(points))
                for j in range(i + 1, len(points))
                if np.sum((points[i] - points[j]) ** 2) <= radius * radius}
        assert got == want
        assert len(a) == len(got)  # each pair once


def test_clean_fiber_has_no_shape_findings():
    doc = fiber(wavy(400))
    assert fiber_qa.shape_findings(doc['filename'], doc, opts()) == []


def test_hairpin_is_a_fold_back_near_its_control_point():
    out_leg = straight(60)
    back_leg = straight(30, start=out_leg[-1], direction=(-1.0, 0.3, 0.0))[1:]
    doc = fiber(out_leg + back_leg)
    found = checks_of(fiber_qa.shape_findings('f.json', doc, opts()),
                      'fold_back')
    assert len(found) == 1
    assert abs(found[0]['line_index'] - 59) <= 2
    assert found[0]['control_point'] == 6  # the control point at index 60


def test_gentle_bend_is_not_a_fold_back():
    out_leg = straight(60)
    turn = straight(60, start=out_leg[-1], direction=(1.0, 1.0, 0.0))[1:]
    doc = fiber(out_leg + turn)
    assert checks_of(fiber_qa.shape_findings('f.json', doc, opts()),
                     'fold_back') == []


def test_line_running_back_over_itself_is_a_retrace():
    out_leg = straight(100)
    # Comes back along the same path, offset by less than the radius and by
    # half a step along it, then carries on elsewhere.
    back = [[p[0] - 4.0, p[1] + 1.5, p[2]] for p in reversed(out_leg[20:99])]
    away = straight(40, start=back[-1], direction=(0.0, 1.0, 0.0))[1:]
    doc = fiber(out_leg + back + away)
    found = checks_of(fiber_qa.shape_findings('f.json', doc, opts()),
                      'retrace')
    assert len(found) == 1
    # 79 points of 8 units come back: about 600 units traced twice.
    assert 500 <= found[0]['length'] <= 700


def test_adjacent_parallel_line_is_not_a_retrace():
    # A fiber that comes back near itself in the SAME direction (like the
    # next turn of a spiral, squeezed close) is not running back.
    first = straight(80)
    hop = straight(4, start=first[-1], direction=(0.0, 0.0, 1.0))[1:]
    second = straight(80, start=[first[0][0] - 200.0, first[0][1] + 5.0,
                                 first[0][2] + 24.0])
    line = first + hop + second
    doc = fiber(line)
    assert checks_of(fiber_qa.shape_findings('f.json', doc, opts()),
                     'retrace') == []


def test_long_step_is_a_jump():
    line = straight(50)
    line = line[:25] + [[p[0] + 300.0, p[1], p[2]] for p in line[25:]]
    doc = fiber(line)
    found = checks_of(fiber_qa.shape_findings('f.json', doc, opts()), 'jump')
    assert [f['line_index'] for f in found] == [24]


def test_duplicate_pair_found_once_and_tag_respected():
    a = fiber(wavy(300), name='a.json')
    b = fiber([[x + 2.0, y - 1.0, z] for x, y, z in wavy(200)], name='b.json')
    far = fiber(wavy(300, start=(1000.0, 2400.0, 3000.0)), name='c.json')
    docs = {'a.json': a, 'b.json': b, 'c.json': far}
    found = fiber_qa.duplicate_findings(docs, opts())
    assert len(found) == 1
    assert found[0]['file'] == 'b.json'      # the shorter one
    assert found[0]['other'] == 'a.json'
    assert found[0]['severity'] == 'warning'
    assert found[0]['shared'][1] >= 0.95     # b lies along a
    b['tags'] = [fiber_qa.DUPLICATE_TAG]
    found = fiber_qa.duplicate_findings(docs, opts())
    assert found[0]['severity'] == 'info'


def test_crossing_fibers_are_not_duplicates():
    a = fiber(straight(200), name='a.json')
    b = fiber(straight(200, start=(1800.0, 1200.0, 3000.0),
                       direction=(0.0, 1.0, 0.0)), name='b.json')
    assert fiber_qa.duplicate_findings({'a.json': a, 'b.json': b}, opts()) == []


def test_untagged_ends_reported_as_info():
    doc = fiber(wavy(100), cp_tags={0: ['kollesis_termination']})
    found = fiber_qa.end_findings('f.json', doc)
    assert [(f['end'], f['severity']) for f in found] == [('end', 'info')]


def test_load_check_reports_a_one_sided_link(tmp_path):
    a = fiber(straight(40), name='a.json')
    b = fiber(straight(40, start=(1160.0, 1900.0, 3000.0),
                       direction=(0.0, 1.0, 0.0)), name='b.json')
    pa = a['control_points'][2]['position']
    pb = b['control_points'][1]['position']
    entry = {
        'control_point_index': 2, 'branch_fiber_id': 2,
        'branch_control_point_index': 1, 'branch_file': 'b.json',
        'control_point_direction': fiber_merge.endpoint_tangent(
            a['line_points'], pa),
        'branch_control_point_direction': fiber_merge.endpoint_tangent(
            b['line_points'], pb),
        'control_point_position': list(pa),
        'branch_control_point_position': list(pb), 'pending': True,
    }
    a['branches'] = [entry]
    for doc in (a, b):
        (tmp_path / doc['filename']).write_text(json.dumps(doc))
    docs, problems = fiber_qa.load_fibers([str(tmp_path)])
    assert problems == []
    found = fiber_qa.run_checks(docs, ['load'], opts())
    assert len(found) == 1
    assert found[0]['severity'] == 'error'
    assert 'missing reciprocal branch' in found[0]['detail']
    # With the reciprocal entry written, the pair loads cleanly.
    back = copy.deepcopy(entry)
    back.update({
        'control_point_index': 1, 'branch_control_point_index': 2,
        'branch_file': 'a.json',
        'control_point_direction': entry['branch_control_point_direction'],
        'branch_control_point_direction': entry['control_point_direction'],
        'control_point_position': entry['branch_control_point_position'],
        'branch_control_point_position': entry['control_point_position'],
    })
    b['branches'] = [back]
    assert fiber_qa.run_checks({'a.json': a, 'b.json': b}, ['load'],
                               opts()) == []


def test_load_fibers_reports_bad_json_and_skips_other_types(tmp_path):
    (tmp_path / 'broken.json').write_text('{"type": ')
    (tmp_path / 'notes.json').write_text('{"type": "something_else"}')
    (tmp_path / 'readme.txt').write_text('hello')
    good = fiber(straight(20), name='good.json')
    (tmp_path / 'good.json').write_text(json.dumps(good))
    docs, problems = fiber_qa.load_fibers([str(tmp_path)])
    assert list(docs) == ['good.json']
    assert [p['file'] for p in problems] == ['broken.json']


def test_cli_writes_json_and_exit_code(tmp_path, capsys):
    out_leg = straight(60)
    back_leg = straight(30, start=out_leg[-1], direction=(-1.0, 0.3, 0.0))[1:]
    (tmp_path / 'hairpin.json').write_text(
        json.dumps(fiber(out_leg + back_leg, name='hairpin.json')))
    report = tmp_path / 'report.json'
    # Warnings alone do not fail the default run...
    assert fiber_qa.main([str(tmp_path), '--json', str(report)]) == 0
    payload = json.loads(report.read_text())
    assert payload['summary']['checks']['fold_back'] == {'findings': 1,
                                                        'fibers': 1}
    assert 'fold_back' in capsys.readouterr().out
    # ...unless asked to.
    assert fiber_qa.main([str(tmp_path), '--fail-on', 'warning',
                          '--checks', 'shape']) == 1
    assert fiber_qa.main([str(tmp_path), '--checks', 'nope']) == 2


def test_files_are_never_modified(tmp_path):
    doc = fiber(straight(60) + straight(30, start=straight(60)[-1],
                                        direction=(-1.0, 0.0, 0.0))[1:],
                name='x.json')
    path = tmp_path / 'x.json'
    path.write_text(json.dumps(doc))
    before = path.read_bytes()
    fiber_qa.main([str(tmp_path), '--fail-on', 'never'])
    assert path.read_bytes() == before
