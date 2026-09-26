import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_follow.events import DepartureEvents, label_path, truncate_path
from vesuvius.neural_tracing.fiber_follow.geometry import arclength, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.volume import ChunkedArray, sample_supported
from vesuvius.neural_tracing.fiber_follow.direct.judge_slices import SliceConfig, SliceStream, sample_planes, footprint_allowed
from vesuvius.neural_tracing.fiber_follow.direct.judge_model import CTJudge, JudgeConfig, FeatureCache, sequence_tensors
from vesuvius.neural_tracing.fiber_follow.direct.judge_policy import JudgePolicy, JudgePolicyConfig
from vesuvius.neural_tracing.fiber_follow.direct.judge_supervision import masked_bce
from vesuvius.neural_tracing.fiber_follow.direct.judge_evaluation import metrics, paired_row, bootstrap


def line(length=200):
    return np.array([[0., 0., 0.], [length, 0., 0.]])


def test_continuous_correspondence_and_tie():
    e = DepartureEvents(line(), 0)
    assert e.correspondence(np.array([2.3, 1., 0.]), 2)[0] == pytest.approx(2.3)
    e = DepartureEvents(np.array([[0,0,0],[10,0,0],[0,0,0]]), 0)
    assert e.correspondence(np.array([3,0,0]), 0)[0] == 3


def test_seven_grid_samples_confirm_six_do_not_and_return_latches():
    # Ramp finishes at arc 4, then six off-fiber samples through 6.5.
    p = np.array([[0,0,0],[0,4,0],[2.5,4,0]],float)
    six = label_path(p, line(), 0)
    assert six.onset is None and six.run == 4
    assert not six.labels([4,6])[1].any()
    seven = label_path(np.r_[p,[[3,4,0]]], line(), 0)
    assert seven.onset == 4 and seven.confirmation == 7
    seven.append([[8,0,0]])
    assert seven.labels([seven.arc])[0][0] == 0


def test_drift_return_resolves_pending_and_invalid_seed_unknown():
    e = label_path(np.array([[0,0,0],[0,4,0],[0,0,0]],float), line(), 0)
    assert e.onset is None and e.run is None and e.labels([4])[1][0]
    e = label_path(np.array([[0,5,0],[5,5,0]],float), line(), 0)
    assert e.invalid_seed and not e.labels([0,4])[1].any()


@pytest.mark.parametrize('physical', [False, True])
def test_endpoint_crossing_and_stream_partition_invariance(physical):
    path = np.array([[0,0,0],[4.2,0,0],[8.7,0,0],[12,0,0]])
    offline = label_path(path, line(10), 0, physical)
    stream = DepartureEvents(line(10), 0, physical)
    for p in path:
        stream.append([p])
    assert offline.to_dict() == stream.to_dict()
    assert offline.labels([9,10,11])[1].tolist() == ([True]*3 if physical else [True,False,False])
    assert (offline.onset if physical else offline.censored) == pytest.approx(10)


def test_pending_before_censor_remains_unknown():
    e = label_path(np.array([[0,0,0],[8,4,0],[11,0,0]]), line(10), 0)
    if e.censored is not None and e.run is not None:
        assert not e.labels([e.run])[1][0]


class ArrayReader:
    def __init__(self, array, chunks=(8,8,8), missing=()):
        self.array, self.shape, self.chunks = array, array.shape, chunks
        self.missing, self.keys = set(missing), set()
        self.identity = 'test'
    def chunk(self, key):
        self.keys.add(tuple(key))
        if tuple(key) in self.missing:
            return None
        lo = np.array(key)*self.chunks
        return self.array[tuple(slice(a,a+n) for a,n in zip(lo,self.chunks))]


def test_native_interpolation_support_and_zeros():
    z,y,x = np.mgrid[:24,:24,:24]
    reader = ArrayReader((x+2*y+3*z).astype('uint8'))
    points = np.array([[4.2,5.3,6.4],[7.9,7.8,7.7],[-.2,2,2]])
    values, supported = sample_supported(reader, points)
    np.testing.assert_allclose(values[:2], (points[:2] @ [1,2,3])/255, atol=1e-7)
    assert supported.tolist() == [True,True,False]
    zero = ArrayReader(np.zeros((24,24,24),np.uint8), missing={(1,1,1)})
    values, supported = sample_supported(zero, [[1,1,1],[8,8,8],[7,7,7]])
    assert not values.any() and supported.tolist() == [True,False,True]


@pytest.mark.parametrize('missing', [(), ((0, 0, 0), (1, 2, 1))])
def test_native_corner_buckets_match_scalar_reference(missing):
    rng = np.random.default_rng(29)
    array = rng.integers(0, 256, (17, 23, 19), dtype=np.uint8)
    chunks = np.array([6, 7, 5])
    points = np.concatenate((rng.uniform(-2, 25, (500, 3)),
                             rng.integers(-1, 25, (100, 3)),
                             [[0, 0, 0], [18, 22, 16], [5, 7, 6], [4.999999999, 7, 6]]))
    expected = np.zeros(len(points), np.float32)
    support = np.ones(len(points), bool)
    keys = set()
    for i, point in enumerate(points):
        low = np.floor(point[::-1]).astype(np.int64)
        fz, fy, fx = point[::-1]-low
        value = 0.
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    weight = (fz if dz else 1-fz)*(fy if dy else 1-fy)*(fx if dx else 1-fx)
                    if weight <= 0:
                        continue
                    index = low+[dz, dy, dx]
                    if ((index < 0) | (index >= array.shape)).any():
                        support[i] = False
                        continue
                    key = tuple(index//chunks)
                    keys.add(key)
                    if key in missing:
                        support[i] = False
                    else:
                        value += weight*float(array[tuple(index)])
        expected[i] = value
    reader = ArrayReader(array, tuple(chunks), missing)
    actual, actual_support = sample_supported(reader, points)
    np.testing.assert_array_equal(actual, expected/255.)
    np.testing.assert_array_equal(actual_support, support)
    assert reader.keys == keys
    empty, empty_support = sample_supported(reader, np.empty((0, 3)))
    assert empty.shape == empty_support.shape == (0,)


def test_slice_transform_planes_and_thin_reads():
    z,y,x = np.mgrid[:64,:64,:64]
    reader = ArrayReader((x+y+z).astype('uint8'))
    cfg = SliceConfig(pixels=5, spacing=.125, grid_scale=1.)
    frame = frame_from_heading([0,0,1])
    views = sample_planes(reader, np.array([4.,4.,4.]), frame, cfg)
    assert views.shape == (3,3,5,5) and views[:,2].all()
    np.testing.assert_allclose(views[:,0,2,2], 96/255)
    assert len(reader.keys) <= 8
    assert np.all(views[:,1,2,2] == 1)


def test_frames_grid_phase_and_temporary_endpoint_are_prefix_invariant():
    reader = ArrayReader(np.zeros((64,64,64),np.uint8))
    cfg = SliceConfig(pixels=5, spacing=1, trace_scale=1, grid_scale=1.)
    path = np.array([[20,20,20],[24,20,20],[27,22,20],[35,24,20]],float)
    a, b = [SliceStream(reader,cfg,frame_from_heading([1,0,0])) for _ in range(2)]
    for i in range(1,len(path)+1):
        ra = a.update(path[:i])
    rb = b.update(path)
    for x,y in zip(ra,rb):
        assert x['arc'] == y['arc']
        np.testing.assert_allclose(x['frame'], y['frame'], atol=1e-12)
    assert [r['arc'] for r in rb if r['regular']] == list(np.arange(0,arclength(path)[-1],4))
    assert not rb[-1]['regular']


def records(end, start=0, unsupported=()):
    recent = list(np.arange(start,end+1,4.))
    arcs = sorted(set([0.,4.,8.,12.]+recent))
    return [dict(arc=s, query=s in recent, regular=True, reference=s < 16, support=s not in unsupported) for s in arcs]


def test_policy_delay_alarm_revision_and_latching():
    p = JudgePolicy()
    r = records(24)
    assert p.decide(r, np.ones(len(r))) is None and p.accepted == 16
    scores = np.ones(len(r)); scores[3] = .1
    assert p.decide(r,scores) == 'judge_alarm' and p.accepted == 8
    p.decide(r,np.ones(len(r)))
    assert p.accepted == 8 and p.alarm is not None


def test_policy_handoff_seed_reference_gap_and_unresolved_support():
    p = JudgePolicy()
    for end in range(16,201,8):
        r = records(end, max(0,end-128))
        assert p.decide(r,np.ones(len(r))) is None
    assert p.accepted == 192
    p = JudgePolicy()
    r = records(40,unsupported=(8,))
    assert p.decide(r,np.ones(len(r))) == 'judge_unverified'
    assert p.accepted == 4
    p = JudgePolicy()
    r = records(160,32)
    assert p.decide(r,np.ones(len(r))) == 'judge_unverified' and p.accepted == 0


def test_final_audit_partial_endpoint_and_short_start():
    p = JudgePolicy()
    r = records(16)+[dict(arc=18.5,query=True,regular=False,reference=False,support=True)]
    p.decide(r,np.ones(len(r)),final=True)
    assert p.accepted == 18.5
    p = JudgePolicy()
    r = records(8)[:3]
    p.decide(r,np.ones(len(r)),final=True)
    path, reason = p.export(np.array([[0,0,0],[8,0,0]]),'confidence')
    assert len(path) == 1 and reason == 'judge_unverified'


def small_judge():
    return CTJudge(JudgeConfig(pixels=17,spacing=.5,widths=(4,8,16),width=16,heads=2,feedforward=32,view_batch=3))


def test_model_shapes_lattices_unknown_safe_and_gradient():
    model = small_judge()
    images = torch.rand(1,4,3,3,17,17)
    tokens = model.encode(images)
    assert tokens.shape == (1,4,3,89,16)
    torch.testing.assert_close(model.center_coordinates[12],torch.zeros(2))
    assert model.full_coordinates.abs().max() <= 4
    metadata = torch.zeros(1,4,17)
    valid = torch.zeros(1,4,dtype=torch.bool)
    logits = model.decode(tokens,metadata,valid,valid)
    assert torch.isfinite(logits).all()
    loss = masked_bce(logits,torch.full_like(logits,float('nan')),valid,valid)
    assert loss.item() == 0
    mask = torch.ones_like(valid)
    logits = model.decode(tokens,metadata,mask,mask)
    masked_bce(logits,torch.ones_like(logits),mask,mask).sum().backward()
    assert model.local[0].weight.grad.abs().sum() > 0


def test_cache_reuses_views_and_invalidates_weight_change():
    model = small_judge().eval()
    rs = [dict(key=str(i), images=np.zeros((3,3,17,17),np.float32)) for i in range(2)]
    cache = FeatureCache()
    cache.tokens(model,rs,'cpu'); cache.tokens(model,rs,'cpu')
    assert cache.encoded_views == 6
    with torch.no_grad():
        next(model.parameters()).add_(1)
    cache.tokens(model,rs,'cpu')
    assert cache.encoded_views == 12


def test_metric_partition_does_not_relabel_retracted_confirmation():
    base = np.array([[0,0,0],[0,4,0],[5,4,0]],float)
    event = label_path(base,line(),0)
    retained = truncate_path(base,5)
    row = paired_row(base,retained,event,8,'a')
    assert row['Wj'] == 1 and row['F'] == 0
    report = metrics([row])
    assert report['wrong_reduction'] == pytest.approx(.8)
    zero = dict(row,W0=0,Wj=0)
    assert metrics([zero])['wrong_reduction'] is None and not metrics([zero])['qualifies']
    assert bootstrap([zero],10)['wrong_reduction']['undefined'] == 10


def test_replay_archive_preserves_cutoff_and_complete_labels(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.direct.judge_archive import save_archive,PathArchive
    path = np.array([[0,0,0],[0,4,0],[5,4,0]],float)
    f = SimpleNamespace(points=line(),endpoint_stop=(False,False))
    save_archive(tmp_path/'paths.npz',[dict(path=path,seed_frame=np.eye(3),q0=0,reverse=False,ledger=[])])
    context = PathArchive(tmp_path/'paths.npz').context(0,5,f)
    assert arclength(context['path'])[-1] == 5 and arclength(context['complete_path'])[-1] == 9


def test_joint_microbatch_equivalence_and_independent_gradient_routes():
    # Both branches normalize over their own effective state counts.
    from test_direct import config,batch
    from vesuvius.neural_tracing.fiber_follow.direct.model import DirectFollower
    from vesuvius.neural_tracing.fiber_follow.direct.train import optimizer_update
    torch.manual_seed(4)
    follower = DirectFollower(config())
    judge = small_judge()
    state = batch(follower.cfg,2)
    sequence = dict(images=torch.rand(1,4,3,3,17,17),metadata=torch.zeros(1,4,17),
                    valid=torch.ones(1,4,dtype=torch.bool),queries=torch.ones(1,4,dtype=torch.bool),
                    target=torch.tensor([[1.,1.,0.,0.]]),known=torch.ones(1,4,dtype=torch.bool),
                    eligible=torch.ones(1,4,dtype=torch.bool))
    logits = judge(**{k:sequence[k] for k in ('images','metadata','valid','queries')})
    masked_bce(logits,sequence['target'],sequence['known'],sequence['eligible']).sum().backward()
    assert all(p.grad is None for p in follower.parameters())
    judge.zero_grad(set_to_none=True)
    state['judge']=[sequence,copy.deepcopy(sequence)]
    def sliced(b,i):
        return {k:{kk:vv[i:i+1] for kk,vv in v.items()} if isinstance(v,dict)
                else [v[i]] if isinstance(v,list) else v[i:i+1] for k,v in b.items()}
    models = [(copy.deepcopy(follower),copy.deepcopy(judge)) for _ in range(2)]
    results=[]
    for (m,j),batches in zip(models,([state],[sliced(state,0),sliced(state,1)])):
        e,je=copy.deepcopy(m),copy.deepcopy(j)
        opt=torch.optim.SGD(list(m.parameters())+list(j.parameters()),lr=.001)
        results.append(optimizer_update(m,e,opt,batches,1,.001,judge=j,judge_ema=je,compute_metrics=False))
    assert results[0]['judge_loss'] == pytest.approx(results[1]['judge_loss'],rel=1e-6)
    for a,b in zip(models[0][1].parameters(),models[1][1].parameters()):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


def test_selection_deterministic_ties_failure_and_locked_protocol(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.direct.judge_evaluation import freeze_protocol,select,load_selection
    ck=tmp_path/'checkpoint.pt';ck.write_bytes(b'fixture')
    protocol=freeze_protocol(tmp_path,[ck],'manifest')
    row=dict(fiber='a',C0=100000.,Cj=100000.,W0=100.,Wj=10.,U0=0.,Uj=0.,F=0,unknown_stops=0,missing_source=0)
    rows=[row,dict(row,fiber='b')]
    common=dict(checkpoint=str(ck),checkpoint_sha256=protocol['checkpoints'][0][1],rows=rows,metrics=metrics(rows),
                n_commit=4,sampling_seed=0,max_len=6000,judge_source_sha256='source')
    reports=[dict(common,threshold=t,judge_policy=JudgePolicyConfig(accept=.9,alarm=.5).to_dict()) for t in (.9,.5)]
    selected=select(reports,protocol,tmp_path)
    assert selected['threshold']==.5 and not selected['opt_in']
    assert load_selection(tmp_path/'selection.json')['threshold']==.5
    ck.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        load_selection(tmp_path/'selection.json')
    other=tmp_path/'failed';other.mkdir()
    assert select([],protocol,other)['default']=='forecast_only'
    assert not (other/'selection.json').exists()


def test_source_metadata_and_xyz_origin(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.volume import NativeCT
    root=tmp_path/'0';root.mkdir()
    meta=dict(zarr_format=2,shape=[8,8,8],chunks=[8,8,8],dtype='|u1',compressor=None,fill_value=0,order='C',filters=None)
    (root/'.zarray').write_text(json.dumps(meta))
    (root/'0.0.0').write_bytes(np.zeros((8,8,8),np.uint8).tobytes())
    reader=NativeCT(tmp_path)
    cfg=SliceConfig(pixels=5,spacing=.125,grid_scale=1.,origin=(80,80,80))
    images=sample_planes(reader,np.array([10.5,10.5,10.5]),np.eye(3),cfg)
    assert images[:,2].all() and not images[:,0].any()
    bad=replace(cfg,origin=(0,0,0))
    assert not sample_planes(reader,np.array([10.5,10.5,10.5]),np.eye(3),bad)[:,2].any()


def test_footprints_include_old_seed_and_interpolation_halo():
    cfg=SliceConfig(pixels=5,spacing=1,trace_scale=1,grid_scale=1)
    band=SimpleNamespace(lo=30.,hi=40.)
    records=[dict(center=np.array([0.,0.,27.5]),frame=np.eye(3))]
    assert not footprint_allowed(records,cfg,band)
    records[0]['center'][2]=26.
    assert footprint_allowed(records,cfg,band)


def test_missing_regular_record_cannot_bridge_or_manufacture_ledger():
    p=JudgePolicy()
    p.accepted=32  # no supporting ledger: cannot hand off
    assert p.anchor(records(160,32)) is None
    p=JudgePolicy()
    r=records(24)
    p.decide(r,np.ones(len(r)))
    r=[r for r in records(40) if r['arc']!=20]
    p.decide(r,np.ones(len(r)))
    assert p.accepted==16 and p.evidence_gaps
    assert p.anchor(records(160,32)) is None


def test_shared_loop_final_audit_and_disabled_exact_geometry():
    from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer,TraceParams
    from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
    class Follower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg=SimpleNamespace(n_future=4,max_recovery_distance=6)
        def forward(self,x,hist,hmask):
            points=torch.zeros(len(hist),4,3);points[:,:,2]=torch.arange(1,5)
            return dict(points=points,confidence=torch.ones(len(hist),4))
    class Tracer(ModelTracer):
        def build_inputs(self,*args):return None
    class Audited(Tracer):
        def begin_observed(self,frames):return [JudgePolicy() for _ in frames]
        def inspect_observed(self,state,path,final=False):
            length=arclength(path)[-1]
            rr=[dict(arc=float(s),query=True,regular=True,reference=s<16,support=True)
                for s in np.arange(0,length+1e-7,4)]
            if length>rr[-1]['arc']+1e-7:
                rr.append(dict(arc=length,query=True,regular=False,reference=False,support=True))
            return state.decide(rr,np.ones(len(rr)),final)
        def export_observed(self,state,path,reason):return state.export(path,reason)
    vol=SimpleNamespace(shape=(1000,1000,1000))
    kwargs=dict(vol=vol,crop=__import__('vesuvius.neural_tracing.fiber_follow.geometry',fromlist=['CropSpec']).CropSpec(),
                n_history=8,params=TraceParams(max_len=18.5,n_commit=4),device='cpu')
    a,b=Tracer(Follower(),**kwargs),Audited(Follower(),**kwargs)
    try:
        baseline,br=a.trace([[100,100,100]],[[1,0,0]])
        judged,jr=b.trace([[100,100,100]],[[1,0,0]])
        np.testing.assert_array_equal(judged[0],baseline[0])
        assert br==jr==['max_len']
        assert b.observed_states[0].audit[-1]['final']
        b.p.max_len=7
        short,reasons=b.trace([[100,100,100]],[[1,0,0]])
        assert len(short[0])==1 and reasons==['judge_unverified']
    finally:
        a.close();b.close()


def test_unknown_nonfinite_logits_have_zero_finite_gradients():
    logits=torch.tensor([[float('nan'),1.]],requires_grad=True)
    loss=masked_bce(logits,torch.tensor([[float('nan'),1.]]),torch.tensor([[False,True]]),torch.ones(1,2,dtype=torch.bool))
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0,0]==0


def test_replay_later_handoff_uses_actual_ledger_not_labels():
    from vesuvius.neural_tracing.fiber_follow.direct.judge_supervision import make_sequence
    reader=ArrayReader(np.zeros((256,32,32),np.uint8))
    # Small views keep this geometry-only eligibility test cheap.
    cfg=SliceConfig(pixels=5,spacing=1,grid_scale=1.,trace_scale=1,history_length=128)
    path=np.array([[16,16,16],[16,16,176]],float)
    context=dict(path=path,annotation=path,q0=0,physical_end=False,seed_frame=np.eye(3))
    sequence=make_sequence(dict(judge_context=context),reader,cfg)
    assert not sequence['eligible'].any()
    context['policy_audit']=dict(previous_accepted=144.,accepted=152.,endpoint=160.,
                                 arcs=list(np.arange(0,161,4)),support=[True]*41)
    sequence=make_sequence(dict(judge_context=context),reader,cfg)
    assert sequence['eligible'].any()
    assert not sequence['eligible'][0,:4].any()  # seed-only references are context


def test_judge_defaults_to_follower_level_zero_without_upsampling():
    import argparse
    from vesuvius.neural_tracing.fiber_follow.direct.judge_options import add_judge_options, configs
    from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec
    parser=argparse.ArgumentParser();add_judge_options(parser)
    volume=FiberVolumeSpec('unused',ct_zarr='/local/follower.zarr',ct_level=0,ct_grid_scale=4.,grid_scale=8.)
    slices,model,_=configs(parser.parse_args(['--judge']),volume)
    assert slices.source==volume.ct_zarr and slices.level==0 and slices.cache is None
    assert slices.grid_scale==4 and slices.spacing==model.spacing==.5
    assert slices.spacing*slices.trace_scale/slices.grid_scale==1
    assert (slices.pixels-1)*slices.spacing==64
    with pytest.raises(ValueError,match='upsampling'):
        configs(parser.parse_args(['--judge','--judge-pixel-spacing','.125']),volume)
    with pytest.raises(ValueError,match='grid-scale'):
        configs(parser.parse_args(['--judge','--judge-ct','/different.zarr']),volume)
    with pytest.raises(ValueError,match='source'):
        SliceConfig().open()
    external,_,_=configs(parser.parse_args(['--judge','--judge-ct','/explicit.zarr','--judge-ct-grid-scale','1']),volume)
    assert external.spacing==.125 and external.spacing*external.trace_scale/external.grid_scale==1


def test_level_zero_slice_pixels_follow_source_voxels():
    z,y,x=np.mgrid[:32,:32,:32]
    reader=ArrayReader((x+2*y+3*z).astype(np.uint8))
    cfg=SliceConfig(pixels=5)
    images=sample_planes(reader,np.array([8.,8.,8.]),np.eye(3),cfg)
    expected=(16+np.arange(-2,3)+2*(16+np.arange(-2,3)[:,None])+3*16)/255
    np.testing.assert_allclose(images[0,0],expected,atol=1e-7)
    assert images[:,2].all()
    assert images[0,1,2,3]==pytest.approx(np.exp(-1/8))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compiled_judge_updates_variable_lengths_ema_and_resume(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable in this environment')
    from vesuvius.neural_tracing.fiber_follow.direct.train import compile_training_judge
    from vesuvius.neural_tracing.fiber_follow.runloop import update_ema
    # CPU exercises Dynamo/AOT forward and backward without native compilation;
    # CUDA exercises the production Inductor backend and BF16 autocast.
    if device == 'cpu':
        real_compile = torch.compile
        monkeypatch.setattr(torch, 'compile', lambda fn, **kw: real_compile(fn, backend='aot_eager', fullgraph=True, **kw))
    torch.manual_seed(27)
    model = CTJudge(replace(small_judge().cfg, view_batch=5)).to(device)
    eager = copy.deepcopy(model)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    parameters, keys = list(model.parameters()), list(model.state_dict())
    compile_training_judge(model)
    assert list(model.state_dict()) == keys
    assert all(a is b for a,b in zip(parameters, model.parameters()))
    assert 'encode_views' not in ema.__dict__ and 'decode' not in ema.__dict__
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step, count in enumerate((1, 4, 7), 1):
        images = torch.rand(1,count,3,3,17,17,device=device)
        metadata = torch.rand(1,count,17,device=device)
        valid = torch.ones(1,count,dtype=torch.bool,device=device)
        queries = valid.clone()
        if count > 1:
            valid[:, -1] = False
        optimizer.zero_grad(set_to_none=True)
        eager.load_state_dict(model.state_dict())
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = model(images, metadata, valid, queries)
            with torch.no_grad():
                expected = eager(images, metadata, valid, queries)
            loss = masked_bce(logits, torch.ones_like(logits), valid, queries).mean()
        torch.testing.assert_close(logits, expected, rtol=.04 if device == 'cuda' else 1e-5,
                                   atol=.03 if device == 'cuda' else 1e-6)
        loss.backward()
        assert model.local[0].weight.grad.abs().sum() > 0
        assert model.head.weight.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        update_ema(ema, model, step, .999)
    path = tmp_path/'compiled_judge.pt'
    torch.save(dict(judge=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict()), path)
    restored = CTJudge(model.cfg).to(device)
    restored_ema = copy.deepcopy(restored).requires_grad_(False)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    saved = torch.load(path, map_location=device, weights_only=True)
    restored.load_state_dict(saved['judge']); restored_ema.load_state_dict(saved['ema'])
    restored_opt.load_state_dict(saved['optimizer'])
    assert saved['judge'].keys() == restored.state_dict().keys()
    for a,b in zip(ema.parameters(), restored_ema.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    compile_training_judge(restored)
    restored_opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        loss = restored(images,metadata,valid,queries).square().mean()
    loss.backward(); restored_opt.step()
    assert torch.isfinite(loss)


def test_synthetic_contact_cache_and_bounds_pruning_are_exact():
    from vesuvius.neural_tracing.fiber_follow.direct.judge_supervision import synthetic_contact
    t = np.arange(0, 200, 1.)
    def fiber(name, offset, bend=0.):
        points = np.stack((t, offset[1]+bend*np.sin(t/20), offset[2]+0*t), 1)+[offset[0], 0, 0]
        return SimpleNamespace(name=name, source_hash=name, points=points, s=arclength(points),
                               length=float(arclength(points)[-1]), endpoint_stop=(False, False))
    fibers = [fiber('a', (0, 0, 0)), fiber('b', (0, 3, 0), 2.), fiber('c', (0, 5.9, 0)), fiber('d', (0, 50, 0)),
              fiber('e', (0, 0, 6.))]
    def run(cache_factory):
        rng, cache = np.random.default_rng(3), cache_factory()
        return [synthetic_contact(fibers, rng, cache=cache) for _ in range(12)], rng.bit_generator.state
    # Unbounded boxes disable pruning, reproducing the exhaustive search.
    unpruned = lambda: {('bounds', i): (np.full(3, -np.inf), np.full(3, np.inf)) for i in range(len(fibers))}
    expected, state = run(unpruned)
    assert any(c is not None for c in expected)
    for factory in (dict, lambda: None):
        actual, actual_state = run(factory)
        assert actual_state == state
        np.testing.assert_equal(actual, expected)
