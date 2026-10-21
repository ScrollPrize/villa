from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from test_fiber_annotations import annotated_loader
from vesuvius.neural_tracing.fiber_trace_3d.annotations import sample_baked_normals, resample_baked_metadata
from vesuvius.neural_tracing.fiber_trace_3d.refinement import (
    RefinementConfig, empty_refinement_statistics, merge_refinement_statistics,
    upgrade_refinement_statistics, log_refinement_scalars, arclength,
    displacement_maxima, normal_projectors,
)


def test_maximum_not_sum_or_mean_across_test_waves_and_resume():
    cfg = RefinementConfig()
    a = empty_refinement_statistics(); a[0], a[-5:] = 3., [4.,4.,0.,2.,20.]
    b = empty_refinement_statistics(); b[0], b[-5:] = 7., [6.,0.,6.,3.,10.]
    merged = merge_refinement_statistics(a,b,cfg)
    assert merged[0] == 10 and merged[-5:] == [6.,4.,6.,3.,20.]
    assert merge_refinement_statistics(merged,a,cfg)[-5:] == [6.,4.,6.,3.,20.]
    old = upgrade_refinement_statistics(a[:12],cfg)
    assert old[-5] == 4 and np.isnan(old[-4:]).all()
    assert merge_refinement_statistics(old,b,cfg)[-5:] == [6.,0.,6.,3.,10.]
    calls = {}
    log_refinement_scalars(SimpleNamespace(add_scalar=lambda k,v,s:calls.update({k:v})), 'test', merged, 128,cfg)
    assert calls['refinement_test/max_displacement_base_voxels'] == 6
    assert calls['refinement_test/max_normal_displacement_base_voxels'] == 4
    assert calls['refinement_test/max_tangent_displacement_base_voxels'] == 6
    assert calls['refinement_test/max_width_deviation_base_voxels'] == 3
    assert calls['refinement_test/max_width_deviation_percent'] == 20


def test_directional_components_and_sign_invariant_interpolated_normals():
    delta = np.array([[3.,4.,0.], [0.,0.,6.], [7.,0.,0.]])
    normals = np.tile([1.,0.,0.],(3,1))
    assert displacement_maxima(delta,normal_projectors(normals)) == (7.,7.,6.)
    assert displacement_maxima(delta,normal_projectors(-normals)) == (7.,7.,6.)
    # Mixed reference projectors use the same quadratic split as the solver;
    # at each node normal²+tangent² equals total² even between normal samples.
    p = np.array([np.diag([.5,.5,0.])])
    total, normal, tangent = displacement_maxima(delta[:1],p)
    assert normal**2 + tangent**2 == pytest.approx(total**2)
    assert normal == pytest.approx(np.sqrt(12.5))


def test_baked_normal_material_and_segment_identity_at_close_return():
    fiber = annotated_loader().records[0].fiber
    xyz = np.array([[0,0,0],[10,0,0],[10,1,0],[0,1,0.]])
    axes = [[0,0,1],[0,0,1],[0,1,0],[0,1,0]]
    fiber = replace(fiber,line_points_xyz=xyz, metadata={'width':8.,'baked_annotations':{
        'version':1,'normals_xyz':axes}})
    # Spatial query sits on the wrong nearby return; material/segment identities
    # still select the intended first segment's original normal.
    query = np.array([[0,1,5.]])
    np.testing.assert_allclose(sample_baked_normals(fiber,query,material=[5.]),[[1,0,0]])
    np.testing.assert_allclose(sample_baked_normals(fiber,query,segment_indices=[0]),[[1,0,0]])
    metadata = resample_baked_metadata(fiber,arclength(fiber.line_points_zyx),[5.,16.])
    np.testing.assert_allclose(metadata['baked_annotations']['normals_xyz'],[[0,0,1],[0,1,0]])
