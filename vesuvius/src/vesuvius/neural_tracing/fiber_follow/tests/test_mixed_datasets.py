"""AFV coordinates/holdouts, CT-only inputs, cache reads and source weighting."""
import json
import pickle
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from model_fixtures import coordinate_batch, coordinate_config, config
from model_fixtures import array_at
from vesuvius.neural_tracing.fiber_follow.data.afv import AFVFibers
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec, RemoteChunkedArray
from vesuvius.neural_tracing.fiber_follow.data.data import make_sample, SampleConfig
from vesuvius.neural_tracing.fiber_follow.data.datasets import AFVBank, WeightedDatasets, read_dataset_config
from vesuvius.neural_tracing.fiber_follow.data.observations import image_crop, IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms


def afv_fixture(path, length=80., neighbor_x=40., offset=0.):
    with sqlite3.connect(path) as c:
        c.executescript('''PRAGMA application_id=1447249475; PRAGMA user_version=1;
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT,family TEXT,length REAL,min_z REAL,max_z REAL);
            CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER,first_segment INTEGER,points BLOB);
            CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);''')
        # Set the exact format magic independently of SQL's decimal rendering.
        c.execute(f'PRAGMA application_id={0x56434643}')
        for k,v in dict(complete=True, uuid='fixture', fiber_count=3,
                        frame={'vc_open_data_coordinate_space':'fixture/scan'}).items():
            c.execute('INSERT INTO metadata VALUES(?,?)',(k,json.dumps(v)))
        for fid,x,z0,z1 in [(1,32.,0.,length),(2,neighbor_x,0.,length),(3,32.,120.,140.)]:
            x, y, z0, z1 = x+offset, 32.+offset, z0+offset, z1+offset
            pts=np.array([[x,y,z0],[x,y,(z0+z1)/2],[x,y,z1]],dtype='<f8')
            c.execute('INSERT INTO fibers VALUES(?,?,?,?,?,?)',(fid,str(fid),'H',z1-z0,z0,z1))
            # Two blocks share exactly one endpoint.
            for j in range(2):
                bid=2*fid+j
                c.execute('INSERT INTO blocks VALUES(?,?,?,?)',(bid,fid,j,pts[j:j+2].tobytes()))
                c.execute('INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)',(bid,x,x,y,y,pts[j,2],pts[j+1,2]))


def test_afv_native_coordinates_block_overlap_holdout_and_pickle(tmp_path):
    p=tmp_path/'test.afv';afv_fixture(p)
    fibers=AFVFibers(p,2.,validation_z=[100,150])
    assert len(fibers)==2 and len(fibers._cache)==0
    f=fibers[0]
    assert f.length==40 and f.endpoint_stop==(False,False)
    np.testing.assert_allclose(f.points[[0,-1]],[[16,16,0],[16,16,40]])
    assert f.s[-1]==40 and np.linalg.norm(np.diff(f.points,axis=0),axis=1).max()<=1
    restored=pickle.loads(pickle.dumps(fibers))
    np.testing.assert_array_equal(restored[0].points,f.points)
    assert len(AFVFibers(p,2.,validation_z=[100,150],split='validation'))==1
    lines=list(fibers.nearby_blocks(np.array([[14,14,10],[24,18,20]]),1))
    assert lines and all(np.all(line[:,0]==20) for line in lines)


def test_afv_foreign_masks_exclude_own_target(tmp_path):
    p=tmp_path/'test.afv';afv_fixture(p);fibers=AFVFibers(p,1.)
    cfg=config()
    # The neighbor sits at the crop edge; geometry here excludes simulated tracing error.
    s=SampleConfig(crop=cfg.fine,n_future=4,n_history=32,startup_shares=(0.,0.,0.,1.),excursion_probability=0.,
                   trace_noise_sigma=(0.,0.))
    rng=np.random.default_rng(17);item=make_sample(fibers[0],40.,False,s,rng)
    item['fiber_ref']=(0,40.,False)
    builder=IdentityObservationBuilder(cfg,fibers,IdentitySampling(),negative_bank=AFVBank(fibers))
    item=builder.prepare(item,fibers[0],rng)
    labels=builder.bank_targets([item])
    assert labels['foreign'].any()
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import crop_local_grid
    local=crop_local_grid(cfg.fine)[labels['foreign'][0].bool().numpy()]
    world=local@item['frame'].T+item['pos']
    assert np.all(np.linalg.norm(world[:,:2]-[32.,32.],axis=1)>1.5)


def test_ct_only_reads_no_auxiliary_and_requires_auxiliary_when_enabled(tmp_path):
    array_at(tmp_path/'ct'/'0',np.full((48,48,48),127,np.uint8))
    spec=FiberVolumeSpec('does-not-exist',ct_zarr=str(tmp_path/'ct'),ct_level=0,
        grid_scale=1.,ct_grid_scale=1.,inputs='ct',load_presence=False)
    from test_ct_normalization import record
    spec.ct_normalization = record(spec)
    vol=FiberVolume(spec);cfg=config()
    items=[dict(pos=np.array([24.,24.,24.]),frame=np.eye(3))]
    x=image_crop(items,vol,cfg.fine)
    assert x.shape[1]==1
    torch.testing.assert_close(x,torch.zeros_like(x))
    with pytest.raises(TypeError, match='direction_inputs'):
        CoordinateRegressionConfig(direction_inputs=True)


def test_ct_only_model_backward_and_config_roundtrip():
    torch.set_num_threads(2)
    cfg=coordinate_config()
    m=build_model(cfg);b=coordinate_batch(cfg,1)
    assert b['x']['fine'].shape[1]==1
    out=m(b['x'],b['hist'],b['hmask']);terms=loss_terms(out,b,cfg)
    loss=terms['geometry_per_state'].sum()+terms['confidence_per_state'].sum()
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    assert CoordinateRegressionConfig(**cfg.to_dict()).input_channels==1


class FakeDataset:
    chunk=2
    def __init__(self,source):self.source=source
    def __iter__(self):
        while True:yield {'hist':torch.zeros(2,3,3),'source_fixture':self.source}


def test_weighted_source_sampling_preserves_pairs_and_reproducibility():
    d=WeightedDatasets([FakeDataset(i) for i in range(3)],['p','a','b'],[.1,.45,.45],seed=42)
    def draw():
        it=iter(d);counts=np.zeros(3,int);sequence=[]
        for _ in range(3000):
            b=next(it);i=b['source_fixture'];assert (b['dataset_id']==i).all()
            counts[i]+=1;sequence.append(i)
        return counts,sequence
    counts,sequence=draw()
    assert np.all(abs(counts/3000-[.1,.45,.45])<.035)
    assert sequence==draw()[1]


def test_real_config_has_requested_sources_and_stable_paths():
    p=Path(__file__).resolve().parents[1]/'configs/mixed_ct_datasets_paris50.json'
    d,digest=read_dataset_config(p)
    assert [s['weight'] for s in d['sources']]==[.5,.25,.25]
    assert [s['validation']['count'] for s in d['sources'] if s['kind']=='afv']==[406,406]
    assert 'SMALLMORE' not in json.dumps(d)
    assert d['cache_dir']=='/mnt/raid_nvme/volume_cache'
    assert digest==read_dataset_config(p)[1]


def test_resume_allows_cache_relocation_but_not_dataset_changes():
    from copy import deepcopy
    from vesuvius.neural_tracing.fiber_follow.data.datasets import validate_dataset_resume
    p=Path(__file__).resolve().parents[1]/'configs/mixed_ct_datasets_paris50.json'
    document,digest=read_dataset_config(p)
    old=deepcopy(document);old['cache_dir']='/old/cache'
    checkpoint=dict(dataset_config=old,dataset_config_sha256='old-digest')
    validate_dataset_resume(checkpoint,document,digest)
    validate_dataset_resume(dict(dataset_config_sha256=digest),document,digest)
    validate_dataset_resume({},None,None)  # Legacy single-source checkpoints.
    relocated=deepcopy(document);relocated['sources'][1]['ct']='s3://different/volume'
    validate_dataset_resume(checkpoint,relocated,'changed-digest')  # CT location is not part of the data identity
    for key,value in [('weight',.2),
                      ('validation',dict(strategy='fiber_hash',count=405,seed=7349))]:
        changed=deepcopy(document);changed['sources'][1][key]=value
        with pytest.raises(ValueError,match='dataset configuration changed'):
            validate_dataset_resume(checkpoint,changed,'changed-digest')
    for missing in ({},dict(dataset_config_sha256='old-digest')):
        with pytest.raises(ValueError,match='dataset configuration changed'):
            validate_dataset_resume(missing,document,digest)
    assert checkpoint['dataset_config']['cache_dir']=='/old/cache'


@pytest.mark.parametrize('separator',['.','/'])
def test_remote_array_chunk_edges_and_pickle(tmp_path,monkeypatch,separator):
    values=np.arange(9*10*11,dtype=np.uint16).reshape(9,10,11)
    class Array:
        shape=values.shape;chunks=(4,4,4);dtype=values.dtype;fill_value=0
        dimension_separator=separator
        def __getitem__(self,key):return values[key]
    monkeypatch.setattr(RemoteChunkedArray,'_open',lambda self:Array())
    a=RemoteChunkedArray('s3://fixture/ct',0,tmp_path,1024)
    assert Path(a.path)==tmp_path/'s3/fixture/ct/0'
    a=pickle.loads(pickle.dumps(a))
    np.testing.assert_array_equal(a.read([7,8,9],[2,2,2]),values[7:9,8:10,9:11])
    block=a.read([-1,-1,-1],[3,3,3])
    assert not block[0].any()
    np.testing.assert_array_equal(block[1:,1:,1:],values[:2,:2,:2])
    # Persist raw full-size chunks, then reopen with network access forbidden.
    metadata=json.loads((Path(a.path)/'.zarray').read_text())
    assert metadata['compressor'] is None and metadata['filters'] is None
    assert metadata['dimension_separator']==separator
    assert (Path(a.path)/separator.join(['1','2','2'])).stat().st_size==4**3*values.dtype.itemsize
    assert isinstance(a.chunk((1,2,2)),np.memmap)
    def forbidden(self):raise AssertionError('Cached reads must not open the remote store')
    monkeypatch.setattr(RemoteChunkedArray,'_open',forbidden)
    reopened=RemoteChunkedArray('s3://fixture/ct/',0,tmp_path,1024)
    np.testing.assert_array_equal(reopened.read([7,8,9],[2,2,2]),values[7:9,8:10,9:11])
    with pytest.raises(AssertionError,match='remote store'):
        reopened.read([4,4,4],[1,1,1])
    # A different volume cannot reuse the first volume's metadata or chunks.
    with pytest.raises(AssertionError,match='remote store'):
        RemoteChunkedArray('s3://fixture/other',0,tmp_path,1024)


def test_whole_fiber_holdout_excludes_neighbor_queries_and_manifest(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data.data import fiber_manifest
    p=tmp_path/'test.afv';afv_fixture(p)
    validation=dict(strategy='fiber_hash',fraction=.34,seed=19)
    tr=AFVFibers(p,validation=validation);va=AFVFibers(p,validation=validation,split='validation')
    assert not tr.ids & va.ids and tr.ids|va.ids=={1,2,3}
    assert fiber_manifest(tr)!=fiber_manifest(va)
    assert not tr._cache  # Manifest doesn't decode every polyline.
    for f in tr:
        assert not (set(tr.nearby_fiber_ids([36,32,40],100,-1)) & va.ids)
    assert AFVFibers(p,validation=validation).ids==tr.ids


def test_fixed_count_holdout_is_exact_stable_and_rejects_invalid_sizes():
    from vesuvius.neural_tracing.fiber_follow.data.dataset_split import heldout_ids
    policy=dict(strategy='fiber_hash',count=406,seed=7349)
    ids=list(range(2000))
    reserved=heldout_ids(ids,policy)
    assert len(reserved)==406
    assert reserved==heldout_ids(ids[::-1],policy)
    assert reserved < heldout_ids(ids,dict(policy,count=500))
    for changes in (dict(count=0),dict(count=-1),dict(count=2.5),dict(count=True),
                    dict(count=len(ids)),dict(fraction=.1)):
        with pytest.raises(ValueError):
            heldout_ids(ids,dict(policy,**changes))


def test_afv_supports_certified_synthetic_failures_and_switch_detection(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data.neighbor_continuations import wrong_continuation
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import TERMINAL
    from sampling_fixtures import clean_sample
    p=tmp_path/'test.afv';afv_fixture(p,length=800,neighbor_x=36)
    fibers=AFVFibers(p);bank=AFVBank(fibers)
    cfg=config(fine=replace(config().fine,depth=48,behind=24))
    sample=clean_sample(cfg);rng=np.random.default_rng(13)
    def find(fn):
        for _ in range(40):
            item=fn()
            if item is not None:return item
        pytest.fail('Could not construct synthetic AFV task')
    switch=find(lambda:wrong_continuation(bank,sample,rng,tail_length_range=(16.,32.),prefix_length=128))
    assert switch['supervision']==TERMINAL and switch['terminal'] and not switch['geometry_valid']
    from vesuvius.neural_tracing.fiber_follow.data.bank_geometry import BankSwitchDetector
    assert BankSwitchDetector([bank]).first_contact(0,40.,np.array([[32.,32.,40.],[36.,32.,44.]])) is not None


def test_replay_scheduler_cycles_all_sources_without_parallel_collectors():
    from vesuvius.neural_tracing.fiber_follow.train.online import MultiSourceCollector
    class Collector:
        def __init__(self):self.calls=[];self.busy_skips=0
        def due(self,step):return step%10==0
        def launch(self,step,save):self.calls.append(step);return step%10==0
        def poll(self,step=None):return {'dagger_states':2}
        def close(self):return None
    collectors=[(n,Collector()) for n in ['p','a','b']]
    scheduler=MultiSourceCollector(collectors)
    for step,name in zip([10,20,30,40],['p','a','b','p']):
        assert scheduler.launch(step,None)
        assert not scheduler.launch(step,None)  # busy launch skipped and counted
        assert dict(collectors)[name].busy_skips >= 1
        assert scheduler.poll()['dataset']==name
    assert scheduler.close() is None


@pytest.mark.parametrize('kind', ['coordinate_regression', 'flow_matching'])
def test_real_collector_roundtrip_on_afv_with_ct_only_inputs(tmp_path, kind):
    import hashlib
    from vesuvius.neural_tracing.fiber_follow.tracing.collect import main as collect
    from vesuvius.neural_tracing.fiber_follow.train.train import save_checkpoint
    from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates
    # Leave room for the 65-voxel CT context throughout this short trace, even
    # when the randomly initialized model makes a small lateral correction.
    p=tmp_path/'test.afv';afv_fixture(p, offset=16.)
    # These fibers run along z inside a CT sheet with normal y. Collection now
    # estimates seed headings from CT plus the family, so uniform CT is invalid.
    with sqlite3.connect(p) as c:
        c.execute("UPDATE fibers SET family='V'")
    sheet = 50+150*np.exp(-.5*((np.arange(128)-48)/2.)**2)
    ct = np.broadcast_to(sheet[None,:,None], (192,128,128)).astype(np.uint8)
    array_at(tmp_path/'ct'/'0',ct)
    validation=dict(strategy='fiber_hash',fraction=.1,seed=7349)
    source=dict(name='fixture',kind='afv',path=str(p),ct=str(tmp_path/'ct'),grid_scale=1.,
                ct_grid_scale=1.,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),validation=validation)
    document=dict(sources=[source],cache_dir=str(tmp_path/'cache'))
    from test_flow_model import config as flow_config
    cfg=config() if kind == 'coordinate_regression' else flow_config()
    model=build_model(cfg)
    spec=FiberVolumeSpec('',ct_zarr=str(tmp_path/'ct'),ct_level=0,grid_scale=1.,ct_grid_scale=1.,inputs='ct',load_presence=False)
    sample=SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future)
    ck=tmp_path/'ck.pt';out=tmp_path/'replay.npz'
    from test_ct_normalization import record
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import ZSCORE_METHOD, volume_key
    spec.ct_normalization = record(spec)
    normalization = dict(method=ZSCORE_METHOD, volumes={volume_key(spec): spec.ct_normalization})
    save_checkpoint(ck,model,model,spec,sample,dict(step=1,dataset_config=document, ct_normalization=normalization))
    collect(['--checkpoint',str(ck),'--fibers',str(p),'--dataset-name','fixture',
        '--device','cpu','--threads','2','--fibers-per-collection','2','--batch','2','--trace-len','8',
        '--after','4','--out',str(out)])
    states=OnPolicyStates.load(out)
    # One directed episode per distinct fiber; the cursor for the next collection is published.
    assert len(set(states.episode.tolist())) == len(set(states.fiber_idx.tolist()))
    assert out.with_suffix('.coverage.json').exists()
    assert states.provenance['operating_policy']['confidence'] == .5
    assert states.provenance['label_contract']['departure_patience'] == 3
    training=AFVFibers(p,validation=validation)
    states.validate_fibers(training)
    assert len(states)>0 and states.provenance['volume']['ct_zarr']==str(tmp_path/'ct')
    with pytest.raises(ValueError,match='incompatible fibers'):
        states.validate_fibers(AFVFibers(p,validation=validation,split='validation'))


def test_initialization_sources_match_after_explicit_relocation_only(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.data.datasets import same_sources
    source = lambda root, weight=.5: dict(name='paris4', kind='paris4', weight=weight, fibers=f'{root}/fibers',
        manifest=f'{root}/seeds.json', ct='s3://bucket/ct.zarr', validation=dict(strategy='fiber_hash', seed=1))
    recorded, here = dict(sources=[source('/mnt/a')]), dict(sources=[source('/copy', weight=1.)])
    assert same_sources(recorded, dict(sources=[source('/mnt/a', weight=1.)]))  # weights may change
    assert not same_sources(recorded, here)
    monkeypatch.setenv('FIBER_FOLLOW_PATH_MAP', '/mnt/a=/copy')
    assert same_sources(recorded, here)
    here['sources'][0]['validation'] = dict(strategy='fiber_hash', seed=2)
    assert not same_sources(recorded, here)  # holdouts never relocate
    # A warm start may add a source, but every recorded one must stay unchanged.
    extra = dict(source('/new'), name='new_afv')
    assert same_sources(recorded, dict(sources=[source('/copy', weight=.1), extra]), allow_added=True)
    assert not same_sources(recorded, dict(sources=[source('/copy', weight=.1), extra]))
    assert not same_sources(recorded, dict(sources=[extra]), allow_added=True)


def test_checkpoint_volume_paths_and_normalization_keys_relocate_together(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.shared.paths import relocate_checkpoint
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import volume_key
    remote = 's3://bucket/ct.zarr::0'
    make = lambda: dict(vol_spec=dict(fiber_zarr_dir='/mnt/a/preds', ct_zarr='/mnt/a/ct.zarr', ct_level=0,
                                      ct_normalization=dict(volume='/mnt/a/ct.zarr::0')),
                        ct_normalization=dict(method='m', volumes={'/mnt/a/ct.zarr::0': dict(volume='/mnt/a/ct.zarr::0'),
                                                                    remote: dict(volume=remote)}))
    assert relocate_checkpoint(make()) == make()  # no map: unchanged
    monkeypatch.setenv('FIBER_FOLLOW_PATH_MAP', '/mnt/a=/copy')
    ck = relocate_checkpoint(make())
    assert ck['vol_spec']['fiber_zarr_dir'] == '/copy/preds' and ck['vol_spec']['ct_zarr'] == '/copy/ct.zarr'
    spec = type('Spec', (), dict(ct_zarr=ck['vol_spec']['ct_zarr'], ct_level=0))
    # The relocated spec finds its normalization under its own key; remote stores keep theirs.
    assert ck['ct_normalization']['volumes'][volume_key(spec)]['volume'] == volume_key(spec)
    assert ck['vol_spec']['ct_normalization']['volume'] == volume_key(spec)
    assert ck['ct_normalization']['volumes'][remote]['volume'] == remote
