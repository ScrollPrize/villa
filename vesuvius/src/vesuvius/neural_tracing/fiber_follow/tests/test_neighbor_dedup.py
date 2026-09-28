"""Spatial duplicates across workers, annotations, shards and interrupted runs."""
import hashlib
import json

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.regression.neighbor_dedup import PathCoverage
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bulk import (
    commit_shard, pack_paths, restore_coverage, write_json,
)


def line(start=0.,end=100.,y=0.):
    return np.array([[start,y,0.],[end,y,0.]])


@pytest.mark.parametrize('flush',[False,True])
def test_reversal_resampling_and_shifted_rediscoveries_are_duplicates(flush):
    index = PathCoverage()
    index.add(line())
    if flush: index.flush()
    assert index.duplicate(line()[::-1])
    assert index.duplicate(np.c_[np.linspace(0,100,301),np.zeros((301,2))])
    assert index.duplicate(line(10,110))
    assert index.duplicate(line(0,100,1.))
    assert not index.duplicate(line(0,100,3.))
    assert not index.duplicate(line(80,180))  # mostly new extension
    assert not index.duplicate(np.array([[50.,-40,0],[50.,40,0]]))  # crossing


def test_coverage_combines_multiple_shards_and_ignores_orthogonal_nearest():
    index = PathCoverage()
    index.add(line(0,45))
    index.flush()
    index.add(line(55,100))
    assert index.duplicate(line())
    vertical = np.array([[50.,-.5,0],[50.,.5,0]])
    assert index.coverage(vertical) == 0
    index.add(vertical+np.array([.1,0,0]))
    assert index.duplicate(vertical)


def proposal(root,fi,begin,paths,*,eligible=True):
    folder = root/'pending'/f'{fi:04d}'/f'{begin:06d}'
    folder.mkdir(parents=True,exist_ok=True)
    rows = []
    for i,path in enumerate(paths):
        row = dict(id=f'candidate_{i}',negative_xyz=path.tolist(),training_eligible=eligible,
                   metrics={'target_arc_range':[0.,100.]})
        rows.append(row)
        write_json(folder/(row['id']+'.json'),row)
        (folder/(row['id']+'.png')).write_bytes(b'review fixture')
    with (folder/'bank.npz').open('wb') as stream:
        np.savez(stream,**pack_paths(paths,[[0,100]]*len(paths),[eligible]*len(paths),[50]*len(paths)))
    write_json(folder/'candidates.json',rows)
    row = dict(run_digest='run',path=f'shards/{fi:04d}/{begin:06d}',fiber=fi,begin=begin,end=begin+64,
               candidates=len(paths),training_candidates=len(paths)*int(eligible),rejected={},anchor_range=[50.,50.],
               bank_sha256=hashlib.sha256((folder/'bank.npz').read_bytes()).hexdigest())
    write_json(folder/'proposal.json',row)
    return row


def test_coordinator_deduplicates_across_shards_annotations_and_resume(tmp_path):
    run = {'digest':'run'}
    index = PathCoverage()
    # Simulate later workers finishing first. Publication still follows the
    # coordinator's stable order, not these proposal completion times.
    last = proposal(tmp_path,1,0,[line()[::-1],line(80,180)])
    second = proposal(tmp_path,0,64,[line(10,110),line(y=5)])
    first = proposal(tmp_path,0,0,[line(),line()[::-1]])
    a = commit_shard(tmp_path,run,first,index)
    assert a['candidates'] == 2 and a['draw_candidates'] == 1 and a['suppressed_draws'] == 1
    b = commit_shard(tmp_path,run,second,index)
    assert b['candidates'] == 2 and b['draw_candidates'] == 1 and b['suppressed_draws'] == 1
    original_bytes = (tmp_path/a['path']/'bank.npz').read_bytes()
    restored = restore_coverage(tmp_path,[a,b],{})
    c = commit_shard(tmp_path,run,last,restored)
    assert c['candidates'] == 2 and c['draw_candidates'] == 1 and c['suppressed_draws'] == 1
    assert (tmp_path/a['path']/'bank.npz').read_bytes() == original_bytes
    with np.load(tmp_path/c['path']/'bank.npz') as data:
        np.testing.assert_array_equal(data['points'],np.concatenate([line()[::-1],line(80,180)]))
        assert data['train_eligible'].tolist() == [True,True]
        assert data['draw_eligible'].tolist() == [False,True]
    assert not (tmp_path/'pending/0001/000000').exists()


def test_interrupted_commit_restores_only_published_geometry(tmp_path,monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.regression.neighbor_bulk as bulk
    run = {'digest':'run'}
    first = proposal(tmp_path,0,0,[line()])
    original_write = bulk.write_json
    def fail_commit(path,value):
        if path.name == 'done.json': raise RuntimeError('interrupted')
        original_write(path,value)
    with monkeypatch.context() as m:
        m.setattr(bulk,'write_json',fail_commit)
        with pytest.raises(RuntimeError,match='interrupted'):
            commit_shard(tmp_path,run,first,PathCoverage())
    assert not (tmp_path/first['path']/'done.json').exists()
    assert (tmp_path/'pending/0000/000000/proposal.json').exists()
    completed = commit_shard(tmp_path,run,first,restore_coverage(tmp_path,[],{}))
    assert completed['candidates'] == 1
    restored = restore_coverage(tmp_path,[completed],{})
    assert restored.duplicate(line())


def test_damaged_geometry_cannot_seed_resume_index(tmp_path):
    row = proposal(tmp_path,0,0,[line()])
    done = commit_shard(tmp_path,{'digest':'run'},row,PathCoverage())
    (tmp_path/done['path']/'bank.npz').write_bytes(b'damaged')
    with pytest.raises(ValueError,match='Damaged shard'):
        restore_coverage(tmp_path,[done],{})


def test_evaluation_geometry_never_suppresses_training_draws(tmp_path):
    coverage = PathCoverage()
    a = commit_shard(tmp_path,{'digest':'run'},proposal(tmp_path,0,0,[line()],eligible=False),coverage)
    coverage = restore_coverage(tmp_path,[a],{})
    b = commit_shard(tmp_path,{'digest':'run'},proposal(tmp_path,1,0,[line()]),coverage)
    assert a['draw_candidates'] == 0 and b['draw_candidates'] == 1


def test_loader_retains_each_parent_relationship_but_draws_one_geometry(tmp_path):
    from test_neighbor_bank import make_bank,publish
    from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bulk import digest
    from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
    from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber,fiber_manifest
    bank,first = make_bank(tmp_path)
    second = TracedFiber('second.json',first.points+[12,0,0],first.s,'H',source_hash='second')
    run = json.loads((tmp_path/'run.json').read_text())
    run['version'] = 2
    run['fibers'].append(dict(fiber_manifest([second])[0],training_fiber=True,length=second.length,tag='H'))
    run['digest'] = digest({k:v for k,v in run.items() if k != 'digest'})
    write_json(tmp_path/'run.json',run)
    path = np.array([[6.,0.,20.],[6.,0.,120.]])
    coverage,rows = PathCoverage(),[]
    for fi in (0,1):
        p = proposal(tmp_path,fi,0,[path])
        p['run_digest'] = run['digest']
        rows.append(commit_shard(tmp_path,run,p,coverage))
    publish(tmp_path,rows)
    bank = NeighborBank(tmp_path,[first,second],bank.band)
    assert len(bank.paths(0,50.)) == len(bank.paths(1,50.)) == 1
    for seed in range(10):
        assert bank.draw_path(np.random.default_rng(seed))[0] == 0
    assert {bank.draw_path(np.random.default_rng(seed),unique=False)[0] for seed in range(20)} == {0,1}
