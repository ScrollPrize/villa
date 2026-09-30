"""Live negative refresh, immutable publication, geometry safety and integration."""
import copy
from dataclasses import asdict
import hashlib
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig, sample_features
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bulk import digest, pack_paths, write_json
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import MiningConfig
from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule, crop_indices
from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber, ZBand, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, crop_local_grid


def make_bank(root, *, refresh_seconds=0., with_path=False, training=True):
    root.mkdir(exist_ok=True)
    points = np.c_[np.zeros(201),np.zeros(201),np.arange(201.)]
    fiber = TracedFiber('target.json',points,arclength(points),'H',source_hash='source-hash')
    band = ZBand(1000.,1100.)
    record = dict(fiber_manifest([fiber])[0],training_fiber=training,length=fiber.length,tag='H')
    run = dict(version=1,mining=asdict(MiningConfig()),excluded_z=[band.lo,band.hi],fibers=[record],
               prediction_manifest={'groups':{c:dict(zarr=f'/data/preds/{c}.ome.zarr/3')
                                               for c in ('presence','nx','ny')}},ct='/data/ct.zarr/1')
    run['digest'] = digest(run)
    write_json(root/'run.json',run)
    publish(root,[])
    if with_path:
        publish(root,[add_shard(root,0,eligible=training)])
    return NeighborBank(root,[fiber],band,refresh_seconds=refresh_seconds,training=training),fiber


def publish(root,shards):
    run = json.loads((root/'run.json').read_text())
    value = dict(version=run['version'],run_digest=run['digest'],complete=False,mining=run['mining'],
                 excluded_z=run['excluded_z'],fibers=run['fibers'],shards=shards)
    value['sha256'] = digest(value)
    write_json(root/'bank.json',value)


def add_shard(root,number,*,x=6.,eligible=True,z_range=(72.,112.)):
    directory = root/'shards'/f'{number:04d}'
    directory.mkdir(parents=True,exist_ok=True)
    lo,hi = z_range
    points = np.array([[x,0.,lo],[x,0.,hi]])
    with (directory/'bank.npz').open('wb') as stream:
        np.savez(stream,**pack_paths([points],[[lo,hi]],[eligible],[(lo+hi)/2]))
    return dict(path=str(directory.relative_to(root)),fiber=0,begin=number,end=number+1,
                anchor_range=[(lo+hi)/2]*2,candidates=1,training_candidates=int(eligible),
                bank_sha256=hashlib.sha256((directory/'bank.npz').read_bytes()).hexdigest())


def item(cfg, reverse=False):
    sign = -1 if reverse else 1
    pos = np.array([.13,-.17,80.13])
    frame = np.diag([sign,1,sign])
    along = np.arange(-24.,40.01,.25)
    world = np.c_[along*0,along*0,80.+sign*along]
    return dict(pos=pos,frame=frame,fiber_ref=(0,120. if reverse else 80.,reverse),
                identity_curve=(world-pos) @ frame,identity_seed=19,
                reference_on_fiber=np.ones((cfg.n_history+1),np.float32),source=0,offtrack=False)


def test_live_refresh_obeys_interval_and_sees_new_shards_after_empty_lookup(tmp_path,monkeypatch):
    clock = [100.]
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank.time.monotonic',lambda:clock[0])
    bank,_ = make_bank(tmp_path,refresh_seconds=30.)
    assert not bank.paths(0,80.)
    first = add_shard(tmp_path,0)
    publish(tmp_path,[first])
    clock[0] = 129.
    assert not bank.paths(0,80.)
    clock[0] = 131.
    assert len(bank.paths(0,80.)) == 1
    cached = bank._cache[first['path']]
    second = add_shard(tmp_path,1,x=8.)
    publish(tmp_path,[first,second])
    clock[0] = 162.
    assert len(bank.paths(0,80.)) == 2
    assert bank._cache[first['path']] is cached


def test_unpublished_and_heldout_shards_are_not_training_negatives(tmp_path):
    bank,_ = make_bank(tmp_path)
    first = add_shard(tmp_path,0)
    assert not bank.paths(0,80.)  # file exists, but producer has not committed it
    heldout = add_shard(tmp_path,1,eligible=False)
    publish(tmp_path,[heldout])
    assert not bank.paths(0,80.)
    publish(tmp_path,[heldout,first])
    assert len(bank.paths(0,80.)) == 1


def test_evaluation_paths_are_available_only_in_explicit_evaluation_mode(tmp_path):
    bank,fiber = make_bank(tmp_path,with_path=True,training=False)
    assert len(bank.paths(0,80.)) == 1
    assert bank.draw_path(np.random.default_rng(0)) is None
    with pytest.raises(ValueError,match='no training annotation'):
        NeighborBank(tmp_path,[fiber],bank.band)


def test_identity_supervision_and_training_cli_require_a_bank():
    builder = IdentityObservationBuilder(DirectConfig())
    with pytest.raises(ValueError,match='requires a negative bank'):
        builder.identity_targets([])
    from vesuvius.neural_tracing.fiber_follow.regression.train import main
    with pytest.raises(ValueError,match='requires --negative-bank'):
        main(['--name','unused','--fiber-zarrs','unused','--fibers','unused','--ct','unused',
              '--manifest','unused','--device','cpu','--threads','1'])


def test_changed_published_shards_fail_closed_and_resume_allows_growth(tmp_path):
    bank,_ = make_bank(tmp_path,with_path=True)
    first = list(bank._known.values())[0]
    saved = bank.provenance()
    second = add_shard(tmp_path,1,x=8.)
    publish(tmp_path,[first,second])
    bank.validate_resume(saved)
    publish(tmp_path,[second])
    with pytest.raises(ValueError,match='removed or modified'):
        bank.refresh(force=True)


def test_corrupt_shard_and_wrong_annotation_or_band_are_rejected(tmp_path):
    bank,fiber = make_bank(tmp_path,with_path=True)
    entry = list(bank._known.values())[0]
    with (tmp_path/entry['path']/'bank.npz').open('ab') as stream:
        stream.write(b'damaged')
    with pytest.raises(ValueError,match='checksum'):
        bank.paths(0,80.)
    changed = copy.deepcopy(fiber)
    changed.points[3,0] = 1
    with pytest.raises(ValueError,match='annotation changed'):
        NeighborBank(tmp_path,[changed],bank.band)
    with pytest.raises(ValueError,match='holdout'):
        NeighborBank(tmp_path,[fiber],ZBand(999.,1100.))


def test_pickle_drops_worker_caches_and_retains_live_refresh(tmp_path):
    bank,_ = make_bank(tmp_path,with_path=True)
    assert bank.paths(0,80.)
    copied = pickle.loads(pickle.dumps(bank))
    assert not copied._cache and copied._pid is None
    first = list(bank._known.values())[0]
    second = add_shard(tmp_path,1,x=8.)
    publish(tmp_path,[first,second])
    assert len(copied.paths(0,80.)) == 2


def test_prediction_and_ct_sources_must_match_training(tmp_path):
    bank,_ = make_bank(tmp_path)
    spec = SimpleNamespace(fiber_zarr_dir='/data/preds',fiber_level=3,ct_zarr='/data/ct.zarr')
    bank.validate_volume(spec)
    spec.ct_zarr = '/data/other.zarr'
    with pytest.raises(ValueError,match='CT source'):
        bank.validate_volume(spec)
    spec.ct_zarr = '/data/ct.zarr'
    spec.fiber_level = 4
    with pytest.raises(ValueError,match='prediction volume'):
        bank.validate_volume(spec)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('angle',[0.,.37])
def test_training_targets_stay_exactly_on_both_centerlines(tmp_path,reverse,angle):
    bank,fiber = make_bank(tmp_path)
    cfg = DirectConfig()
    builder = IdentityObservationBuilder(cfg,[fiber],negative_bank=bank)
    state = item(cfg,reverse)
    rotation = np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    state['frame'] = state['frame'] @ rotation
    state['identity_curve'] = state['identity_curve'] @ rotation
    # No negatives before a validated path arrives.
    empty = builder.identity_targets([state])
    assert not empty['negative_mask'].any() and not empty['foreign'].any()
    publish(tmp_path,[add_shard(tmp_path,0,x=6.13)])
    result = builder.identity_targets([state])
    assert result['negative_mask'].any() and result['foreign'].any()
    k,m = builder.sampling.positives,builder.sampling.negatives
    positive = result['identity_points'][0,:k].numpy()
    negative = result['identity_points'][0,k:].reshape(k,m,3).numpy()
    valid = result['negative_mask'][0].numpy().astype(bool)
    world = positive @ state['frame'].T+state['pos']
    np.testing.assert_allclose(world[:,:2],0.,atol=1e-6)
    for p,points,keep in zip(positive,negative,valid):
        world = points[keep] @ state['frame'].T+state['pos']
        # Off-grid centerline must remain off-grid: neither dilation nor phase
        # matching may displace queries from the stored native polyline.
        np.testing.assert_allclose(world[:,0],6.13,atol=1e-6)
        np.testing.assert_allclose(world[:,1],0.,atol=1e-6)
        assert bank.clear_of_target(0,world).all()
    assert result['negative_bank_shards'].item() == 1


def test_bank_rasterization_marks_only_cells_containing_line_samples(tmp_path):
    bank,_ = make_bank(tmp_path,with_path=True)
    cfg = DirectConfig()
    state = item(cfg)
    found = bank.candidates(state,cfg.fine,ComponentRule())
    indices = np.rint(crop_indices(cfg.fine,found['local'])).astype(int)
    expected = np.zeros_like(found['foreign'])
    expected[tuple(indices.T)] = True
    np.testing.assert_array_equal(found['foreign'],expected)




def test_worker_rejects_a_replaced_run_instead_of_silently_relabeling(tmp_path):
    bank,_ = make_bank(tmp_path)
    value = json.loads((tmp_path/'bank.json').read_text())
    value['run_digest'] = 'different-run'
    value['sha256'] = digest({k:v for k,v in value.items() if k != 'sha256'})
    write_json(tmp_path/'bank.json',value)
    with pytest.raises(ValueError,match='run changed'):
        bank.paths(0,80.)


def test_foreign_cell_extent_cannot_reach_target_exclusion_tube(tmp_path):
    bank,_ = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=3.)])
    cfg = DirectConfig()
    state = item(cfg)
    found = bank.candidates(state,cfg.fine,ComponentRule())
    grid = crop_local_grid(cfg.fine)[found['foreign']]
    world = grid @ state['frame'].T+state['pos']
    # The straight target is the z axis; evaluate the most adverse cell corner.
    half = cfg.fine.spacing/2
    lower_distance = np.linalg.norm(np.maximum(abs(world[:,:2])-half,0),axis=1)
    assert (lower_distance > bank.exclusion).all()
    # Even with a forged close path, the annotation itself must stay unknown.
    assert not found['foreign'][cfg.fine.behind,cfg.fine.width//2,cfg.fine.width//2]


class BankProbe(torch.utils.data.Dataset):
    def __init__(self,bank,continuations=False):
        self.bank,self.continuations = bank,continuations
    def __len__(self):
        return 1
    def __getitem__(self,index):
        if self.continuations:
            from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
            from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
            state = wrong_continuation(self.bank,SampleConfig(),np.random.default_rng(8),tail_length_range=(4.,12.))
            return os.getpid(),int(state is not None)
        return os.getpid(),len(self.bank.paths(0,80.))


@pytest.mark.parametrize('continuations',[False,True])
def test_persistent_loader_worker_discovers_appended_paths_without_restart(tmp_path,continuations):
    bank,_ = make_bank(tmp_path)
    loader = torch.utils.data.DataLoader(BankProbe(bank,continuations),batch_size=None,num_workers=1,
        persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
    try:
        pid,before = next(iter(loader))
        assert before == 0
        publish(tmp_path,[add_shard(tmp_path,0)])
        same_pid,after = next(iter(loader))
        assert same_pid == pid and after == 1
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
