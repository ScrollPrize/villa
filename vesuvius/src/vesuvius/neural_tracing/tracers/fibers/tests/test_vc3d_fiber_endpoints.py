"""A kollesis_termination tag on an end control point becomes a known endpoint in every
vc3d_fiber version that carries control point tags (VC3D saves version 4)."""
import json

import pytest

from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers


def _segment():
    return {
        'optimizer': 'native_fiber_trace3d', 'metadata_version': 3, 'tracer_version': 2,
        'interp_goal': 'cspline', 'interp_mode': 'cspline', 'metric': None, 'msg': 'cspline',
        'meeting_error_base_voxels': None, 'meeting_error_ratio': None,
        'normal_manifest': '', 'fiber_manifest': '', 'trace_to_base_scale': 4.0,
        'meeting_source': '', 'failure_code': '', 'failure_detail': '',
        'lasagna_failure_code': '', 'lasagna_failure_detail': '',
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


def _fiber(version, first_tags, last_tags):
    points = [[float(x), 0.0, 0.0] for x in (0, 100, 200, 300)]
    controls = [{'position': p, 'segment_to_next': _segment()} for p in points[:-1]]
    controls.append({'position': points[-1]})
    if first_tags:
        controls[0]['tags'] = first_tags
    if last_tags:
        controls[-1]['tags'] = last_tags
    return {'type': 'vc3d_fiber', 'version': version, 'optimization_mode': 'native_fiber_trace3d',
            'generation': 1, 'line_points': points, 'control_points': controls, 'tags': []}


@pytest.mark.parametrize('version', [3, 4])
@pytest.mark.parametrize('first_tags, last_tags, expected', [
    ([], ['kollesis_termination'], (False, True)),
    (['kollesis_termination'], [], (True, False)),
    (['kollesis_termination'], ['kollesis_termination'], (True, True)),
    (['break'], [], (False, False)),
    ([], [], (False, False)),
])
def test_kollesis_termination_is_a_known_endpoint(tmp_path, version, first_tags, last_tags, expected):
    (tmp_path / 'fiber.json').write_text(json.dumps(_fiber(version, first_tags, last_tags)))
    (fiber,) = load_fibers(str(tmp_path))
    assert fiber.endpoint_stop == expected


def test_version_4_spans_without_tracing_metadata(tmp_path):
    """Exported version 4 fibers may have spans with no segment_to_next."""
    doc = _fiber(4, [], ['kollesis_termination'])
    for control in doc['control_points']:
        control.pop('segment_to_next', None)
    (tmp_path / 'fiber.json').write_text(json.dumps(doc))
    (fiber,) = load_fibers(str(tmp_path))
    assert fiber.endpoint_stop == (False, True)
