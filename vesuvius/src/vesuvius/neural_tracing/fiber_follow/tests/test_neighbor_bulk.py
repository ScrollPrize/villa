"""Bulk sweep coverage, cache equivalence, and partition boundaries."""
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bulk import anchor_positions, pack_paths, training_eligible, build_parser, main
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


def test_long_bank_options_require_room_for_the_complete_path(tmp_path):
    args = ['--fibers','unused','--fiber-zarrs','unused','--ct','unused',
            '--native-build-python','unused','--output',str(tmp_path/'bank'),
            '--seed-spacing','20','--extrapolation','70','--block-size','192']
    parsed = build_parser().parse_args(args)
    cfg = MiningConfig(seed_spacing=parsed.seed_spacing,extrapolation=parsed.extrapolation,block_size=parsed.block_size)
    assert cfg.seed_spacing+2*cfg.extrapolation == 160.
    assert anchor_positions(400.,8.,cfg)[0] == 96.
    assert not training_eligible(True,np.array([0,0,0]),cfg,ZBand(150.,200.))
    with pytest.raises(ValueError,match='Block must contain'):
        main(args[:-1]+['80'])
    assert not (tmp_path/'bank').exists()


def test_path_range_options_validate_before_loading_inputs(tmp_path):
    args = ['--fibers','unused','--fiber-zarrs','unused','--ct','unused',
            '--native-build-python','unused','--output',str(tmp_path/'bank'),
            '--min-path-length','80','--max-path-length','160','--block-size','192']
    parsed = build_parser().parse_args(args)
    cfg = MiningConfig(min_path_length=parsed.min_path_length,max_path_length=parsed.max_path_length,
                       block_size=parsed.block_size)
    assert cfg.path_length_limit == 160.
    with pytest.raises(ValueError,match='Block must contain'):
        main(args[:-1]+['160'])
    with pytest.raises(ValueError,match='Supply both'):
        main(args[:10]+['--min-path-length','80'])
    assert not (tmp_path/'bank').exists()


def test_outer_band_options_validate_before_loading_inputs(tmp_path):
    args = ['--fibers','unused','--fiber-zarrs','unused','--ct','unused',
            '--native-build-python','unused','--output',str(tmp_path/'bank'),
            '--min-distance','12','--max-distance','32']
    parsed = build_parser().parse_args(args)
    assert (parsed.min_distance,parsed.max_distance)==(12.,32.)
    with pytest.raises(ValueError,match='min_distance'):
        main(args[:-1]+['12'])
    assert not (tmp_path/'bank').exists()
