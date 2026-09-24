"""Per-control-point tags are validated for shape only: any string loads, so
VC3D's `kollesis_termination` and `break` tags reach every reader of the format
without a loader change (and a fiber without tags is written exactly as before)."""
from __future__ import annotations

import copy

import pytest

from vc3d_fiber_format import parse_vc3d_fiber_format


def _cspline_segment():
    return {
        'optimizer': 'native_fiber_trace3d',
        'metadata_version': 3,
        'tracer_version': 2,
        'interp_goal': 'cspline',
        'interp_mode': 'cspline',
        'metric': None,
        'msg': 'cspline',
        'meeting_error_base_voxels': None,
        'meeting_error_ratio': None,
        'normal_manifest': '',
        'fiber_manifest': '',
        'trace_to_base_scale': 4.0,
        'meeting_source': '',
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


def _v3_fiber():
    points = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
    controls = [{'position': list(p), 'segment_to_next': _cspline_segment()} for p in points[:-1]]
    controls.append({'position': list(points[-1])})
    return {
        'type': 'vc3d_fiber',
        'version': 3,
        'optimization_mode': 'native_fiber_trace3d',
        'generation': 1,
        'line_points': [list(p) for p in points],
        'control_points': controls,
        'tags': [],
    }


def test_break_and_kollesis_tags_load_on_any_control_point():
    doc = _v3_fiber()
    doc['control_points'][1]['tags'] = ['break']
    doc['control_points'][2]['tags'] = ['break']
    doc['control_points'][3]['tags'] = ['kollesis_termination']
    parsed = parse_vc3d_fiber_format(doc)
    assert len(parsed.control_points_xyz) == 4
    assert parsed.control_point_segments[-1] is None


def test_control_point_tags_must_be_a_list_of_strings():
    doc = _v3_fiber()
    doc['control_points'][1]['tags'] = 'break'
    with pytest.raises(ValueError):
        parse_vc3d_fiber_format(doc)
    doc = _v3_fiber()
    doc['control_points'][1]['tags'] = [1]
    with pytest.raises(ValueError):
        parse_vc3d_fiber_format(doc)
    # Any other per-point field is still rejected: the tags array is the one
    # open slot for new per-point information.
    doc = _v3_fiber()
    doc['control_points'][1]['break'] = True
    with pytest.raises(ValueError):
        parse_vc3d_fiber_format(doc)
