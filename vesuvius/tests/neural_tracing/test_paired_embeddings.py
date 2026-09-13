"""Pair-local supervision, shared value augmentation and atomic scheduling."""
from dataclasses import replace
import os
import numpy as np
import pytest
import torch

from test_fiber_embeddings import training_batch, tiny_config, pair_prepass_counts
from test_fiber_trace_3d import _loader_with_mapping
from vesuvius.neural_tracing.fiber_trace_3d.embedding import EmbeddingConfig, embedding_statistics
from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DSharedTensorSlots
from vesuvius.neural_tracing.fiber_trace_3d.arc_sampling import ScrollArcSampler
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_resources import EffectiveBatchSampleIndex
from vesuvius.neural_tracing.fiber_trace_3d.train import _slice_batch, _forward_loss_statistics
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model


def paired_batch(pair_id=0):
    b = training_batch()
    return replace(b, embedding_geometry=tuple(replace(g, objective='local_contrastive',
        pair_id=pair_id, pair_member=i, fiber_id='same', pairs=512,
        eligible=torch.ones_like(g.eligible)) for i,g in enumerate(b.embedding_geometry)))


def test_pair_joint_gradients_and_detached_presence():
    b = paired_batch()
    torch.manual_seed(4)
    e = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    p = torch.rand(2, 1, 8, 8, 8, requires_grad=True)
    # Only the second patch has eligible negatives; the first must use those.
    b = replace(b, embedding_geometry=(replace(b.embedding_geometry[0], eligible=torch.zeros(8,8,8,dtype=torch.bool)), b.embedding_geometry[1]))
    diagnostics=[]
    result = embedding_statistics(e,p,b,diagnostics)
    assert result[-1] == 32 and len(diagnostics) == 2
    assert all('negative_mean' in d for d in diagnostics)
    (result[-2]/result[-1]).backward()
    assert p.grad is None
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in e.grad)
    with pytest.raises(ValueError, match='remain together'):
        embedding_statistics(e[:1],p[:1],_slice_batch(b,0,1))
    bad=replace(b,embedding_geometry=(b.embedding_geometry[0],replace(b.embedding_geometry[1],fiber_id='wrong')))
    with pytest.raises(ValueError, match='identity'):
        embedding_statistics(e,p,bad)


@pytest.mark.parametrize('sizes', [(2,), (2,4,6), (2,)*8, (2,)*70])
def test_partition_keeps_whole_pairs(sizes):
    seen=[]
    for rank in range(len(sizes)):
        index=EffectiveBatchSampleIndex(128,sizes,rank,7,2)
        assert all(v%2==0 for v in index.bounds+index.batch_sizes)
        seen.extend(index(i) for i in range(index.local_samples))
    assert seen==list(range(7*128,8*128))
    with pytest.raises(ValueError,match='whole'):
        EffectiveBatchSampleIndex(128,(3,),0,group_size=2)


def test_loader_pair_rng_retry_and_slots(monkeypatch):
    import vc.volume
    from vesuvius.neural_tracing.fiber_trace_3d.embedding import InvalidNormalSampleError
    class Sampler:
        def __init__(self,*args): pass
        def sample(self,xyz,threads):
            return np.tile([0.,0.,1.],(len(xyz),1)),np.ones(len(xyz),bool)
    monkeypatch.setattr(vc.volume,'NormalSampler',Sampler)
    loader=_loader_with_mapping(dict(augment_brightness=.2,augment_noise_std=.1,augment_blur_sigma=.5,
        augment_smooth_displacement_probability=1.,augment_smooth_displacement_mode='3d',
        augment_smooth_displacement_amplitude_zyx=[1,1,1]))
    for r in loader.records:
        r.dataset_config.update(scroll_id='s',dense_supervision=True,lasagna_manifest_path='synthetic.json')
    loader.config=replace(loader.config,embedding=EmbeddingConfig(enabled=True,objective='local_contrastive',
        paired_patches=True,output_stride=4),sampling_strategy='scroll_balanced_arclength')
    loader._arc_sampler=ScrollArcSampler(loader.records)
    a=loader.descriptor_for_sample_index(10); b=loader.descriptor_for_sample_index(11)
    assert a[1]==b[1] and a[2]!=b[2]
    original=loader._build_embedding_geometry
    calls=0
    def retry(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1: raise InvalidNormalSampleError('injected')
        return original(*args,**kwargs)
    monkeypatch.setattr(loader,'_build_embedding_geometry',retry)
    x,y=loader.load_raw_sample(10),loader.load_raw_sample(11)
    for field in ('brightness','contrast','gamma','noise_std','blur_sigma',
                  'anisotropic_blur_sigma_along','anisotropic_blur_sigma_across','value_seed_index'):
        assert getattr(x.augmentation,field)==getattr(y.augmentation,field)
    assert not np.array_equal(x.augmentation.source_to_output_zyx,y.augmentation.source_to_output_zyx)
    assert not np.array_equal(x.augmentation.smooth.controls[0],y.augmentation.smooth.controls[0])
    assert x.sample.embedding_geometry.pair_id==y.sample.embedding_geometry.pair_id==5
    assert x.sample.embedding_geometry.fiber_id==y.sample.embedding_geometry.fiber_id
    slots=FiberTrace3DSharedTensorSlots(start_index=10,slot_count=2,patch_shape_zyx=(16,)*3,embedding_stride=4)
    restored=slots.decode(10,slots.encode(10,x,final=False),final=False)
    assert restored.augmentation.value_seed_index==5
    assert restored.sample.embedding_geometry.pair_member==0


def test_pair_preserving_loss_split():
    b=paired_batch()
    e=torch.randn(4,4,8,8,8,requires_grad=True); p=torch.rand(4,1,8,8,8)
    # Four patches, two independent pairs, using duplicated target geometry.
    from types import SimpleNamespace
    large=SimpleNamespace(embedding_geometry=b.embedding_geometry+tuple(replace(g,pair_id=1,seed=g.seed+10) for g in b.embedding_geometry),
        target_segment_offsets=torch.cat((b.target_segment_offsets,b.target_segment_offsets)),
        target_segment_counts=torch.cat((b.target_segment_counts,b.target_segment_counts)),
        target_segment_starts_zyx=b.target_segment_starts_zyx,target_segment_ends_zyx=b.target_segment_ends_zyx)
    full=embedding_statistics(e,p,large)
    expected=torch.autograd.grad(full[-2]/full[-1],e)[0]
    parts=[]
    for start in (0,2):
        small=replace(b,embedding_geometry=large.embedding_geometry[start:start+2])
        parts.append(embedding_statistics(e[start:start+2],p[start:start+2],small)[-2])
    actual=torch.autograd.grad(sum(parts)/full[-1],e)[0]
    torch.testing.assert_close(actual,expected)


def test_full_model_single_backward_pair_accumulation(monkeypatch):
    import copy
    from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import NormalizedGradientAccumulator
    from vesuvius.neural_tracing.fiber_trace_3d.train import _finalize_loss_statistics
    cfg=tiny_config();cfg['embedding'].update(objective='local_contrastive',paired_patches=True)
    torch.manual_seed(8)
    model=build_fiber_trace_3d_model(cfg);oracle=copy.deepcopy(model)
    batches=[paired_batch(0),paired_batch(1)]
    total=_forward_loss_statistics(oracle,batches[0])+_forward_loss_statistics(oracle,batches[1])
    _finalize_loss_statistics(total,direction_weight=1.,presence_weight=1.)['total'].backward()
    counts=sum(pair_prepass_counts(b) for b in batches)
    accumulator=NormalizedGradientAccumulator(model,counts,direction_weight=1.,presence_weight=1.)
    calls=[];original=torch.Tensor.backward
    def backward(t,*args,**kwargs):
        calls.append(1);return original(t,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'backward',backward)
    for b in batches:accumulator.add(_forward_loss_statistics(model,b))
    accumulator.reduce_and_assign()
    assert len(calls)==2
    for a,b in zip(model.parameters(),oracle.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=3e-5,rtol=3e-4)


@pytest.mark.skipif(os.environ.get('FIBER_COOPERATIVE_CUDA_TEST')!='1',reason='manager-only GPU test')
def test_cuda_embedding_paired_forward_backward():
    import time
    torch.set_num_threads(1)
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=tiny_config(); cfg['embedding'].update(objective='local_contrastive',paired_patches=True)
    model=build_fiber_trace_3d_model(cfg).to(device)
    batch=paired_batch().to(device)
    start=time.perf_counter()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            stats=_forward_loss_statistics(model,batch)
            loss=stats.contrastive_embedding_sum/stats.contrastive_embedding_count
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    torch.cuda.synchronize()
    print('PAIRED_GPU_SMOKE',dict(gpus=1,patches_per_step=2,steps=3,seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def _paired_distributed_worker(rank, directory):
    import torch.distributed as dist
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(rank).total_memory,rank)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(f'cuda:{rank}')
    dist.init_process_group('nccl',init_method=f'file://{directory}/pair-init',rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(71)
        cfg=tiny_config(); cfg['embedding'].update(objective='local_contrastive',paired_patches=True)
        model=build_fiber_trace_3d_model(cfg).to(device)
        import copy
        oracle=copy.deepcopy(model)
        def run(m,which):
            batch=paired_batch(which)
            batch=replace(batch,embedding_geometry=tuple(replace(g,seed=g.seed+which*100) for g in batch.embedding_geometry)).to(device)
            stats=_forward_loss_statistics(m,batch)
            return stats.contrastive_embedding_sum/stats.contrastive_embedding_count
        run(model,rank).backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad);param.grad.div_(2)
        (sum(run(oracle,k) for k in range(2))/2).backward()
        for a,b in zip(model.parameters(),oracle.parameters()):
            if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,atol=3e-5,rtol=3e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('FIBER_COOPERATIVE_CUDA_TEST')!='1',reason='manager-only GPU test')
def test_cuda_embedding_paired_two_gpu(tmp_path):
    torch.multiprocessing.spawn(_paired_distributed_worker,args=(str(tmp_path),),nprocs=2)


@pytest.mark.skipif(os.environ.get('FIBER_EMBEDDING_DATA_TEST')!='1',reason='approved host-data test only')
def test_paired_real_data_preflight():
    import json
    from pathlib import Path
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader,config_from_mapping
    path=Path(__file__).parents[2]/'src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_combined_5scroll_s1a_paired_contrastive_ema_16x8.json'
    raw=json.loads(path.read_text());torch.set_num_threads(1)
    for kind in ('datasets','test_datasets'):
        for entry in raw[kind]:
            cfg=config_from_mapping({**raw,'datasets':[entry],'batch_size':2,
                'augment_enabled':kind=='datasets','volume_io_threads':1},config_dir=path.parent)
            loader=FiberTrace3DLoader(cfg)
            samples=[loader.load_raw_sample(i) for i in range(2)]
            a,b=[s.sample.embedding_geometry for s in samples]
            assert a.fiber_id==b.fiber_id and (a.pair_member,b.pair_member)==(0,1)
            print('PAIRED_DATA_OK',kind,entry['scroll_id'],a.fiber_id,flush=True)
