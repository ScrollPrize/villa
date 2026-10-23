import os
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_fiber_annotations import annotated_loader, model
from vesuvius.neural_tracing.fiber_trace_3d.annotations import (
    AnnotationConfig, annotation_counts, annotation_statistics, normal_tube_targets,
    width_tube_targets, width_similarity, segment_endpoint_widths,
)
from vesuvius.neural_tracing.fiber_trace_3d.direction import encode_lasagna_direction_3x2, projection_magnitude_weights_3x2
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.loader import _TARGET_MODE_DENSE_LINE
from vesuvius.neural_tracing.fiber_trace_3d.train import _slice_batch, _forward_loss_statistics, _finalize_loss_statistics
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import MacroBatchTargetSource
from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import NormalizedGradientAccumulator


def tube_batch(device='cpu', stride=1, sideband=False):
    loader = annotated_loader()
    cfg = replace(loader.config, presence_target_mode='anisotropic_gaussian',
                  presence_radius_voxels=.75, presence_gaussian_width_ratio=4,
                  presence_sideband_weight=1. if sideband else 0.,
                  annotations=AnnotationConfig(10., .01, 1.))
    batch = loader.load_batch(0)
    starts = torch.tensor([[8.,8.,2.], [3.,3.,3.], [6.,6.,3.]])
    ends = torch.tensor([[8.,8.,13.], [3.,10.,3.], [6.,6.,9.]])
    normals = torch.tensor([[1.,0.,0.], [0.,0.,1.], [1.,1.,0.]])
    normals /= normals.norm(dim=1, keepdim=True)
    batch = replace(batch, presence_target_mode='anisotropic_gaussian',
        presence_sideband_weight=cfg.presence_sideband_weight,
        target_modes=torch.full((2,), _TARGET_MODE_DENSE_LINE),
        target_segment_offsets=torch.tensor([0,2]), target_segment_counts=torch.tensor([2,1]),
        target_segment_starts_zyx=starts, target_segment_ends_zyx=ends,
        target_segment_normals_zyx=normals,
        target_segment_widths_input=torch.tensor([[4.,8.],[10.,14.],[6.,6.]]),
        target_segment_bbox_lo_zyx=torch.zeros(3,3,dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.full((3,3),16,dtype=torch.long))
    return materialize_targets(batch.to(device), cfg, output_stride=stride), cfg


@pytest.mark.parametrize('stride', [1, 4])
def test_width_tube_weighting_and_sideband_exclusion(stride):
    batch, cfg = tube_batch(stride=stride, sideband=True)
    indices, widths, falloff = width_tube_targets(batch)
    assert len(indices) > 0 and (widths > 0).all()
    assert (falloff > 0).all()
    assert not batch.presence_sideband_mask[indices[:,0], :, indices[:,1], indices[:,2], indices[:,3]].any()
    torch.testing.assert_close(annotation_counts(batch, cfg.annotations)[1], falloff.sum())
    out = torch.randn((2, 8, *batch.presence_target.shape[-3:]), requires_grad=True)
    stats = annotation_statistics(out, batch, cfg.annotations)
    pred = torch.nn.functional.softplus(out[:,6])[tuple(indices.T)]
    expected = (torch.nn.functional.smooth_l1_loss(pred, widths, reduction='none') * falloff).sum() * cfg.annotations.width_weight
    torch.testing.assert_close(stats['width_sum'], expected)
    for i in range(2):
        part = _slice_batch(batch, i, i+1).to('cpu')
        torch.testing.assert_close(part.width_target_input, batch.width_target_input[i:i+1])
    missing = replace(batch, width_target_input=torch.zeros_like(batch.width_target_input))
    assert annotation_counts(missing, cfg.annotations)[1] == 0


def test_segment_width_endpoints_and_missing_labels():
    points = np.array([[0,0,-10.],[0,0,10.],[0,1,10.],[0,1,-10.]])
    metadata = {'width':10., 'baked_annotations':{'version':1,
        'normals_xyz':np.tile([0.,0.,1.],(4,1)).tolist(), 'widths_base':[10.,30.,70.,90.]}}
    fiber = SimpleNamespace(line_points_xyz=points[:,::-1], line_points_zyx=points, metadata=metadata)
    record = SimpleNamespace(fiber=fiber, volume_spacing_base=1.)
    loader = SimpleNamespace(_source_to_output_jacobian=lambda p,_:np.broadcast_to(np.eye(3),(len(p),3,3)))
    # Segment0 enters the crop at x=0 (fraction.5):20, not25 from nearest in-crop row.
    # Geometrically close return segment2 retains its distinct annotation identity.
    result = segment_endpoint_widths(loader,record,None,[0,2],[[.5,1.],[0.,.5]])
    np.testing.assert_allclose(result, [[20.,30.],[70.,80.]])
    metadata['baked_annotations']['widths_base'][1] = 0
    assert not segment_endpoint_widths(loader,record,None,[0],[[.5,1.]]).any()


def test_width_raster_gaussian_winner_and_sideband_nearest_tie():
    batch, cfg = tube_batch()
    batch = _slice_batch(batch, 0, 1)
    cfg = replace(cfg, presence_radius_voxels=.25, presence_sideband_weight=1.)
    batch = replace(batch, direction_indices_bzyx=None, presence_sideband_weight=1.,
        target_segment_starts_zyx=torch.tensor([[8.,2.,2.],[8.,12.,2.]]),
        target_segment_ends_zyx=torch.tensor([[8.,2.,14.],[8.,12.,14.]]),
        target_segment_normals_zyx=torch.tensor([[1.,0.,0.],[1.,0.,0.]]),
        target_segment_widths_input=torch.tensor([[2.,4.],[10.,14.]]))
    result = materialize_targets(batch, cfg)
    assert result.width_target_input[0,0,8,2,8] == 3  # Gaussian first segment
    assert result.width_target_input[0,0,8,12,8] == 12  # Gaussian second segment
    assert result.presence_target[0,0,8,7,8] == 0
    assert result.presence_sideband_mask[0,:,8,7,8].all()
    assert result.width_target_input[0,0,8,7,8] == 3  # Equal-distance first winner
    assert result.width_target_input[0,0,8,8,8] == 12  # Nearest second segment


def check_sideband_gate(device='cpu', cosine=False):
    from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses
    batch, cfg = tube_batch(device=device, sideband=True)
    if cosine:
        cfg = replace(cfg, annotations=replace(cfg.annotations,
            sideband_width_fraction=.4, sideband_width_gate='cosine'))
    # Force an explicit background-sideband region and a uniform reference.
    mask = (batch.presence_target[:,0] == 0)[:,None].expand(-1,2,-1,-1,-1)
    batch = replace(batch, presence_sideband_mask=mask,
                    width_target_input=torch.full_like(batch.presence_target, 10.))
    output = torch.full((2,7,*batch.presence_target.shape[-3:]), .5, device=device, requires_grad=True)
    results = []
    cases = ([(10.,1.), (11.,.853553390593), (12.,.5), (13.,.146446609407),
              (14.,0.), (8.,.5), (6.,0.), (15.,0.)] if cosine else
             [(10.,1.), (11.,.5), (12.,0.), (8.,0.), (15.,0.)])
    for width, factor in cases:
        aux = torch.zeros((2,8,*output.shape[-3:]), device=device)
        aux[:,6] = np.log(np.expm1(width))
        aux.requires_grad_()
        stats = compute_losses(output, batch, direction_weight=10, presence_weight=1,
            annotation_output=aux, annotation_config=cfg.annotations, _return_statistics=True)
        results.append(stats)
        side = stats.sideband_left_presence_sum + stats.sideband_right_presence_sum
        grad_out, grad_aux = torch.autograd.grad(side, (output, aux), allow_unused=True, retain_graph=True)
        assert grad_aux is None  # Can't evade sideband penalty by changing width.
        if factor == 0:
            assert not grad_out.any()
        torch.testing.assert_close(stats.sideband_left_presence_sum,
                                   results[0].sideband_left_presence_sum * factor)
        for key in ('positive_presence_sum', 'negative_presence_sum', 'distribution_presence_sum',
                    'sideband_left_presence_count', 'sideband_right_presence_count'):
            torch.testing.assert_close(getattr(stats,key), getattr(results[0],key))
    ref = torch.tensor([10.,10.,10.,10.,0.,float('nan')], device=device)
    w = torch.tensor([10.,9.,8.,12.,10.,10.], device=device)
    torch.testing.assert_close(width_similarity(w,ref), torch.tensor([1.,.5,0.,0.,0.,0.], device=device))


def test_sideband_width_gate_detached_and_regular_presence_unchanged():
    check_sideband_gate()
    check_sideband_gate(cosine=True)


def test_cosine_width_gate_numpy_and_config():
    reference = np.array([10.,10.,10.,10.,10.,10.,0.,np.nan])
    width = np.array([10.,11.,12.,13.,14.,8.,10.,10.])
    np.testing.assert_allclose(width_similarity(width, reference, fraction=.4, falloff='cosine'),
        [1.,.853553390593,.5,.146446609407,0.,.5,0.,0.], atol=1e-10)
    cfg = AnnotationConfig.from_mapping({'annotations': {'width_weight': .01,
        'sideband_width_fraction': .4, 'sideband_width_gate': 'cosine'}})
    assert cfg.names == ('width',) and cfg.channels == 1
    for bad in (0, -1, float('nan')):
        with pytest.raises(ValueError):
            AnnotationConfig.from_mapping({'annotations': {'sideband_width_fraction': bad}})
    with pytest.raises(ValueError):
        AnnotationConfig.from_mapping({'annotations': {'sideband_width_gate': 'bad'}})


@pytest.fixture(autouse=True)
def single_core():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('stride', [1, 4])
def test_normal_tube_matches_direction_support_falloff_and_winner(stride):
    batch, cfg = tube_batch(stride=stride)
    indices, axes, falloff = normal_tube_targets(batch)
    assert torch.equal(indices, batch.direction_indices_bzyx)
    assert len(indices) > sum(len(a) for a in batch.annotation_targets)
    torch.testing.assert_close(annotation_counts(batch, cfg.annotations)[0], falloff.sum()*6)
    # Each tangent's winning segment carries that segment's normal, even overlaps.
    first = indices[:,0] == 0
    tangent = batch.direction_tangent_sparse_zyx[first]
    expect = torch.where((tangent[:,2] > .9)[:,None],
                         torch.tensor([0.,0.,1.]), torch.tensor([1.,0.,0.]))
    torch.testing.assert_close(axes[first], expect)
    for i in range(2):
        part = _slice_batch(batch, i, i+1).to('cpu')
        torch.testing.assert_close(part.normal_axis_sparse_zyx, batch.normal_axis_sparse_zyx[indices[:,0] == i])
    missing = replace(batch, annotation_targets=(None, None))
    assert annotation_counts(missing, cfg.annotations)[0] == 0


def check_loss_and_gradients(device, sideband=False, distribution_gate=False):
    import copy
    torch.manual_seed(7)
    batch, cfg = tube_batch(device, sideband=sideband)
    if distribution_gate:
        cfg = replace(cfg, annotations=replace(cfg.annotations, distribution_agreement_gate=True,
            sideband_width_fraction=.4, sideband_width_gate='cosine'))
    if sideband:
        batch = replace(batch, width_target_input=torch.full_like(batch.width_target_input, .7))
    indices, axes, falloff = normal_tube_targets(batch)
    output = torch.randn(2,8,16,16,16,device=device,requires_grad=True)
    stats = annotation_statistics(output, batch, cfg.annotations)
    pred = output[:,:6].sigmoid().permute(0,2,3,4,1)[tuple(indices.T)]
    expected = (((pred - encode_lasagna_direction_3x2(axes))**2)
                * projection_magnitude_weights_3x2(axes) * falloff[:,None]).sum()
    torch.testing.assert_close(stats['normal_sum'], expected*10)
    torch.testing.assert_close(stats['normal_unweighted_sum'], expected)
    assert bool((falloff < .5).any())
    actual = model().to(device)
    actual.annotations = cfg.annotations
    reference = copy.deepcopy(actual)
    source = MacroBatchTargetSource(None, config=cfg, device=torch.device(device), output_stride=1, batch_sizes=[2])
    # Production prepares counts per microbatch, in the same reduction order
    # as the forward passes (Gaussian masses are fractional, not integers).
    counts = sum(source._counts(_slice_batch(batch,i,i+1)) for i in range(2)).to(device)
    full = _forward_loss_statistics(reference, batch)
    losses = _finalize_loss_statistics(full, direction_weight=10., presence_weight=1., sideband_weight=cfg.presence_sideband_weight)
    torch.testing.assert_close(losses['normal'], full.normal_sum/full.normal_count/10)
    torch.testing.assert_close(losses['total'], 10*losses['direction']+losses['presence']+
                              10*losses['normal']+.01*losses['width']+losses['hv'])
    losses['total'].backward()
    accumulator = NormalizedGradientAccumulator(actual, counts, direction_weight=10., presence_weight=1., gaussian_presence=True,
                                               sideband_presence=sideband,
                                               sideband_weight=cfg.presence_sideband_weight)
    for i in range(2):
        accumulator.add(_forward_loss_statistics(actual, _slice_batch(batch,i,i+1)))
    accumulator.reduce_and_assign()
    for a,b in zip(actual.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=3e-4)


def test_normal_tube_loss_logging_and_accumulation():
    check_loss_and_gradients('cpu')


def test_width_gated_sideband_accumulation():
    check_loss_and_gradients('cpu', sideband=True)


def check_distribution_gate(device='cpu'):
    from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses
    batch, cfg = tube_batch(device=device, sideband=True)
    batch = replace(batch, width_target_input=torch.full_like(batch.presence_target, 10.))
    cfg = replace(cfg.annotations, distribution_agreement_gate=True,
                  sideband_width_fraction=.4, sideband_width_gate='cosine')
    idx = tuple(batch.direction_indices_bzyx.T)
    covered = torch.zeros_like(batch.presence_target[:,0], dtype=torch.bool)
    covered[idx] = True
    assert torch.equal(covered, batch.presence_target[:,0] > 0)
    axis = torch.tensor([1.,0.,0.],device=device).expand(len(idx[0]),-1)
    batch = replace(batch, direction_target_sparse=encode_lasagna_direction_3x2(axis))
    baseline = None
    for angle, width, factor in [(0.,10.,1.), (180.,10.,1.), (60.,10.,.5),
                                  (0.,12.,.5), (60.,12.,.25), (90.,10.,0.), (0.,14.,0.)]:
        pred = torch.tensor([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.],
                            dtype=torch.float32,device=device).expand(len(idx[0]),-1)
        output = torch.full((2,7,*batch.presence_target.shape[-3:]), .5,device=device)
        output[:,:6].permute(0,2,3,4,1)[idx] = encode_lasagna_direction_3x2(pred)
        output.requires_grad_()
        aux = torch.zeros((2,8,*output.shape[-3:]),device=device)
        aux[:,6] = np.log(np.expm1(width))
        aux.requires_grad_()
        stats = compute_losses(output,batch,direction_weight=10,presence_weight=1,
            annotation_output=aux,annotation_config=cfg,_return_statistics=True)
        ungated = compute_losses(output,batch,direction_weight=10,presence_weight=1,
            annotation_output=aux,annotation_config=replace(cfg,distribution_agreement_gate=False),
            _return_statistics=True)
        if baseline is None:
            baseline = ungated.distribution_presence_sum.detach()
        torch.testing.assert_close(stats.distribution_presence_sum,baseline*factor,atol=1e-6,rtol=1e-5)
        for key in ('distribution_presence_count','positive_presence_sum','negative_presence_sum',
                    'sideband_left_presence_sum','sideband_right_presence_sum'):
            torch.testing.assert_close(getattr(stats,key),getattr(ungated,key))
        grad, aux_grad = torch.autograd.grad(stats.distribution_presence_sum,(output,aux),allow_unused=True)
        assert aux_grad is None
        assert not grad[:,:6].any()
        if factor > 0:
            assert grad[:,6].abs().sum() > 0
    with pytest.raises(ValueError,match='width prediction'):
        compute_losses(output,batch,direction_weight=10,presence_weight=1,annotation_config=cfg)
    for value in ('true',1,None):
        with pytest.raises(ValueError,match='boolean'):
            AnnotationConfig.from_mapping({'annotations':{'distribution_agreement_gate':value}})
    # Production runs BF16: indexed gate writes must preserve the field dtype.
    bf = output.detach().to(torch.bfloat16).requires_grad_()
    bf_aux = aux.detach().to(torch.bfloat16).requires_grad_()
    bf_stats = compute_losses(bf,batch,direction_weight=10,presence_weight=1,
        annotation_output=bf_aux,annotation_config=cfg,_return_statistics=True)
    assert torch.isfinite(bf_stats.distribution_presence_sum)
    bf_stats.distribution_presence_sum.backward()
    assert not bf.grad[:,:6].any()
    assert bf_aux.grad is None


def test_distribution_agreement_gate():
    check_distribution_gate()
    check_loss_and_gradients('cpu', sideband=True, distribution_gate=True)


@pytest.mark.parametrize('corrupt_local', [False, True])
def test_counts_checked_exactly_before_reporting_collective(monkeypatch, corrupt_local):
    from vesuvius.neural_tracing.fiber_trace_3d.loss_statistics import LossStatistics
    batch, cfg = tube_batch()
    net = model()
    net.annotations = cfg.annotations
    source = MacroBatchTargetSource(None, config=cfg, device=torch.device('cpu'), output_stride=1, batch_sizes=[2])
    counts = source._counts(batch)
    acc = NormalizedGradientAccumulator(net, counts, direction_weight=10, presence_weight=1, gaussian_presence=True)
    acc.add(_forward_loss_statistics(net, batch))
    index = list(LossStatistics.__dataclass_fields__).index('normal_count')
    if corrupt_local:
        acc.statistics[index] += .5
    def reduce(packed, **kwargs):
        if packed.numel() == len(LossStatistics.__dataclass_fields__) + 1:
            packed[index] += .5  # Reporting reduction may round differently.
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    acc.distributed = True
    if corrupt_local:
        with pytest.raises(RuntimeError, match='target normalization counts changed'):
            acc.reduce_and_assign()
    else:
        acc.reduce_and_assign()


@pytest.mark.skipif(not os.environ.get('FIBER_COOPERATIVE_TEST_GPUS'), reason='manager allocation required')
def test_normal_tube_cuda():
    # Compare FP32 gradients across different batch convolution shapes without
    # TF32 algorithm-dependent rounding. Production BF16 policy is unchanged.
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        check_loss_and_gradients('cuda:0')
        check_sideband_gate('cuda:0')
        check_sideband_gate('cuda:0', cosine=True)
        check_loss_and_gradients('cuda:0', sideband=True)
        check_distribution_gate('cuda:0')
        check_loss_and_gradients('cuda:0', sideband=True, distribution_gate=True)
    finally:
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
