"""Bulk sweep coverage, cache equivalence, and partition boundaries."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bulk import anchor_positions, pack_paths, training_eligible
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import MiningConfig, PolylineIndex, exact_nearest
from vesuvius.neural_tracing.fiber_follow.shared.data import ZBand


def test_sweep_is_dense_deterministic_and_preserves_endpoint_margins():
    positions = anchor_positions(183.,8.,MiningConfig())
    assert positions[0] == 40.
    assert positions[-1] <= 143. and positions[-1]+8 > 143.
    np.testing.assert_array_equal(np.diff(positions),8.)
    assert not len(anchor_positions(50.,8.,MiningConfig()))


def test_evaluation_annotations_and_crops_never_become_training_negatives():
    cfg = MiningConfig()
    band = ZBand(100,200)
    assert training_eligible(True,np.array([0,0,0]),cfg,band)
    assert not training_eligible(False,np.array([0,0,0]),cfg,band)
    assert not training_eligible(True,np.array([0,0,20]),cfg,band)
    assert not training_eligible(True,np.array([0,0,200]),cfg,band)
    assert training_eligible(True,np.array([0,0,202]),cfg,band)


def test_packed_paths_preserve_geometry_and_path_partition():
    paths = [np.arange(9.).reshape(3,3),np.arange(12.).reshape(4,3)]
    data = pack_paths(paths,[[1,2],[4,8]],[True,False],[1.5,6.])
    np.testing.assert_array_equal(data['offsets'],[0,3,7])
    for i,path in enumerate(paths):
        np.testing.assert_array_equal(data['points'][data['offsets'][i]:data['offsets'][i+1]],path)
    assert data['train_eligible'].tolist() == [True,False]
    assert pack_paths([],[],[],[])['points'].shape == (0,3)


def test_reused_polyline_index_is_identical_to_unprepared_queries():
    rng = np.random.default_rng(91)
    target = np.cumsum(rng.normal(size=(42,3)),axis=0)
    points = rng.normal(size=(100,3))*4
    index = PolylineIndex(target)
    for old,new in zip(exact_nearest(points,target),exact_nearest(points,target,index)):
        np.testing.assert_array_equal(old,new)
