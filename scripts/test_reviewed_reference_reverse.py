import numpy as np
from reviewed_reference_reverse import convergence, errors_for, slice_line


def test_convergence_excludes_shared_correct_path():
    points = [[0,0,0],[40,0,0],[80,0,0],[120,0,0]]
    assert convergence(points, [50]*4, points, [50]*4)['converged']
    assert not convergence(points, [0]*4, points, [50]*4)['converged']
    assert not convergence(points, [50]*4, points, [0]*4)['converged']
    assert not convergence(points, [50]*4, np.asarray(points)+[0,11,0], [50]*4)['converged']


def test_slice_and_absolute_errors():
    assert slice_line([[0,0,0],[100,0,0]],20,80)[::-1] == [[80,0,0],[20,0,0]]
    errors = errors_for(dict(trace_points_base_xyz=[[0,0,0],[1,0,0]], matches=[dict(
        trace_point_index=1, threshold_error_base_voxels=25, threshold_error_ratio=2.5e-8)]))
    assert errors.tolist() == [0,25]


def test_midpoint_only_overlap_does_not_credit_whole_edge():
    result = convergence([[-50,0,0],[50,0,0]], [50,50], [[-1,0,0],[1,0,0]], [50,50])
    assert result['longest_convergence_base'] == 0
