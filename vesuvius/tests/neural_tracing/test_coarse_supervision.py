import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.coarse_supervision import (
    CoarseGeometry, ellipse_offsets, normal_plane, projected_direction_error,
    background_mask, coarse_sample_statistics, center_rotation_frame, jittered_coarse_sites,
)
from vesuvius.neural_tracing.fiber_trace_3d.direction import encode_lasagna_direction_3x2


def test_coarse_jitter_reproducible_bounded_and_dense_along_line():
    sites, offsets = jittered_coarse_sites(128., 123)
    again = jittered_coarse_sites(128., 123)
    np.testing.assert_array_equal(sites, again[0])
    np.testing.assert_array_equal(offsets, again[1])
    assert len(sites) == 32 and 0 <= sites[0] < 4 and sites[-1] < 128
    np.testing.assert_allclose(np.diff(sites), 4)
    assert np.all(np.sum((offsets / [40., 20.]) ** 2, axis=1) <= 1.)
    np.testing.assert_array_equal(offsets[-1], [0, 0])
    assert 140 < len(offsets) < 180
    phases = np.remainder(offsets[:-1], 4.)
    np.testing.assert_allclose(phases, np.broadcast_to(phases[0], phases.shape), atol=4e-6)
    other = jittered_coarse_sites(128., 124)
    assert not np.array_equal(sites, other[0])
    assert not np.array_equal(offsets, other[1])
    short, _ = jittered_coarse_sites(.001, 123)
    assert len(short) == 1 and 0 <= short[0] < .001
    with pytest.raises(ValueError):
        jittered_coarse_sites(0., 123)


def test_center_rotation_discards_stretch_and_preserves_flips():
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, -1]])
    stretch = np.array([[2., .3, 0], [.3, 1., 0], [0, 0, .5]])
    frame = center_rotation_frame((rotation @ stretch)[None], 2.)[0]
    np.testing.assert_allclose(frame, rotation / 2, atol=1e-12)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3)/4, atol=1e-12)
    offsets = ellipse_offsets()
    points = offsets[:, :1] * frame[:, 0] + offsets[:, 1:] * frame[:, 1]
    assert np.linalg.norm(points, axis=1).max() == pytest.approx(20.)
    with pytest.raises(ValueError, match='singular'):
        center_rotation_frame(np.zeros((1, 3, 3)), 2.)


def test_full_diameters_and_plane_projection():
    offsets = ellipse_offsets()
    np.testing.assert_array_equal(np.ptp(offsets, axis=0), [80, 40])
    t, n, w = normal_plane(torch.tensor([[0., 0, 1]]), torch.tensor([[1., 0, 0]]))
    assert (t*n).sum() == 0
    for tilt in (0., 1., 100.):
        direction = t + tilt * w
        error, angle = projected_direction_error(direction, t, n)
        assert error.item() == 0 and angle.item() == 0
        error, angle = projected_direction_error(direction + n, t, n)
        assert error.item() == pytest.approx(.5)
        assert angle.item() == pytest.approx(45)
    error, angle = projected_direction_error(w, t, n)
    assert error.item() == 1 and angle.item() == 90
    torch.testing.assert_close(projected_direction_error(-t-n, t, n)[0], torch.tensor([.5]))


def test_parallel_normal_fallback_is_finite_and_deterministic():
    t = torch.tensor([[0., 0, 1.], [1., 0, 0]])
    a = normal_plane(t, t)
    b = normal_plane(t, -t)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
        assert torch.isfinite(x).all()
    torch.testing.assert_close((a[0]*a[1]).sum(-1), torch.zeros(2))


@pytest.mark.parametrize('label, expected', [({}, -1),
    ({'automatic_tag': 'V'}, 1), ({'manual_tag': 'H', 'automatic_tag': 'V'}, 0)])
def test_coarse_hv_targets_ignore_other_annotations(label, expected):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import AnnotationConfig, line_targets
    loader = SimpleNamespace(config=SimpleNamespace(
        annotations=AnnotationConfig(normal_weight=10, width_weight=.01, hv_weight=.01),
        patch_shape_zyx=(16, 16, 16)),
        _source_points_to_output_np=lambda p, *a, **kw: (p + 1, np.ones(len(p), bool)))
    fiber = SimpleNamespace(line_points_zyx=np.array([[2., 3, 4], [8, 3, 4]]),
        metadata={'width': 80, 'baked_annotations': {'invalid': True}, 'hv_classification': label})
    rows = line_targets(loader, SimpleNamespace(fiber=fiber, volume_spacing_base=1), None, hv_only=True)
    assert len(rows) == 6
    assert np.all(rows[:, 3:7] == 0) and np.all(rows[:, 7] == expected)
    np.testing.assert_array_equal(rows[:, 0], np.arange(3, 9))


@pytest.mark.parametrize('stride', [1, 2])
def test_unlabelled_coarse_peak_selection(stride):
    center = np.array([[4., 4., 4.]], np.float32) * stride
    geometry = CoarseGeometry(center, np.array([[0., 0., 1.]], np.float32),
        np.eye(3, dtype=np.float32)[None] * stride, center, center,
        np.array([1.], np.float32), offsets=np.array([[0., 0.], [0., 1.]], np.float32))
    def field(axis):
        return encode_lasagna_direction_3x2(torch.tensor(axis))[:, None, None, None].expand(6, 8, 8, 8).clone().requires_grad_()
    direction, normal = field([1., 0., 0.]), field([0., 0., 1.])
    logits = torch.zeros(8, 8, 8)
    logits[5, 4, 4] = 4.
    logits[3, 4, 4] = 4.  # Normal axis is sign ambiguous.
    logits.requires_grad_()
    background = torch.zeros_like(logits, dtype=torch.bool)
    plain = coarse_sample_statistics(logits, direction, normal, geometry, stride=stride, background=background)
    assert plain[0].item() < .02
    plain[0].backward()
    assert logits.grad[4, 4, 4] == 0
    assert logits.grad[5, 4, 4] + logits.grad[3, 4, 4] < 0
    assert direction.grad is None and normal.grad is None


def test_width_supervision_requires_labels_and_covers_gaussian_tube():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import annotation_statistics
    cfg = SimpleNamespace(names=['width'], width_weight=.01)
    # First sample labelled, second sample coarse/unlabelled. Both have tube sites.
    target = torch.zeros(2, 1, 1, 1, 2)
    target[0] = 10.
    batch = SimpleNamespace(presence_target_mode='anisotropic_gaussian',
        width_target_input=target,
        direction_indices_bzyx=torch.tensor([[0,0,0,0], [0,0,0,1], [1,0,0,0], [1,0,0,1]]),
        presence_target=torch.tensor([[[[[1., .25]]]], [[[[1., .25]]]]]))
    output = torch.zeros(2, 1, 1, 1, 2, requires_grad=True)
    stats = annotation_statistics(output, batch, cfg)
    assert stats['width_count'].item() == pytest.approx(1.25)
    stats['width_sum'].backward()
    assert output.grad[0,0,0,0,0] < 0
    assert output.grad[0,0,0,0,1] == output.grad[0,0,0,0,0] * .25
    assert output.grad[1].count_nonzero() == 0
    batch.width_target_input.zero_()
    empty = annotation_statistics(output, batch, cfg)
    assert empty['width_count'].item() == 0 and empty['width_sum'].item() == 0


def test_peak_gradient_background_exclusion_and_detached_normal():
    # 4base voxels per output site: entire ellipse fits 32^3 output.
    center = np.array([[16., 16., 16.]], np.float32)
    geometry = CoarseGeometry(center, np.array([[0, 0, 1]], np.float32),
        np.eye(3, dtype=np.float32)[None] / 4, center, center,
        np.array([10.], np.float32))
    presence = torch.full((32, 32, 32), .2, requires_grad=True)
    with torch.no_grad():
        presence[17, 16, 16] = .9
    def field(axis):
        encoded = encode_lasagna_direction_3x2(torch.tensor(axis))
        return encoded[:, None, None, None].expand(6, 32, 32, 32).clone().requires_grad_()
    direction = field([1., 0, .1])
    normal = field([0., 0, 1.])
    mask = background_mask(geometry, presence.shape, 1, presence.device)
    assert not mask[17, 16, 16] and mask[0, 0, 0]
    pos, neg, d, angle = coarse_sample_statistics(torch.logit(presence), direction, normal, geometry,
                                                stride=1, background=mask)
    assert pos.item() == pytest.approx(-np.log(.9 / np.sqrt(1.01)), abs=1e-6)
    assert neg.item() == pytest.approx(-np.log(.8), abs=1e-6)
    (pos+neg+d).backward()
    assert normal.grad is None
    assert torch.isfinite(direction.grad).all()
    assert presence.grad[17, 16, 16] < 0
    assert presence.grad[0, 0, 0] > 0
    assert torch.isfinite(angle)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_coarse_crossing_rejected_and_gate_detached(device, sign, monkeypatch):
    import os
    if device == 'cuda' and not os.environ.get('FIBER_COOPERATIVE_TEST_GPUS'):
        pytest.skip('CUDA requires manager allocation')
    center = np.array([[4.,4.,4.]],np.float32)
    geometry = CoarseGeometry(center,np.array([[0.,0.,1.]],np.float32),
        np.eye(3,dtype=np.float32)[None],center,center,np.array([1.],np.float32),
        offsets=np.array([[0.,0.],[0.,1.]],np.float32))
    def field(axis):
        return encode_lasagna_direction_3x2(torch.tensor(axis,device=device))[:,None,None,None].expand(6,8,8,8).clone()
    # tangent is X, normal Z: second candidate is one Z voxel away.
    directions = field([0.,sign,0.])
    directions[:,5,4,4] = encode_lasagna_direction_3x2(torch.tensor([sign,0.,0.],device=device))
    directions[:,3,4,4] = directions[:,5,4,4]
    directions.requires_grad_()
    normals = field([0.,0.,1.]).requires_grad_()
    logits = torch.zeros((8,8,8),device=device)
    logits[4,4,4] = torch.logit(torch.tensor(.99))
    logits[5,4,4] = torch.logit(torch.tensor(.6))
    logits[3,4,4] = logits[5,4,4]
    logits.requires_grad_()
    mask = torch.zeros_like(logits,dtype=torch.bool)
    pos,_,_,_ = coarse_sample_statistics(logits,directions,normals,geometry,stride=1,background=mask)
    torch.testing.assert_close(pos,torch.tensor(-np.log(.6),device=device,dtype=torch.float32))
    grad, dgrad, ngrad = torch.autograd.grad(pos,(logits,directions,normals),allow_unused=True)
    assert grad[4,4,4].abs() < 1e-6 and (grad[5,4,4] + grad[3,4,4]) < 0
    assert dgrad is None and ngrad is None
    # No aligned candidate: finite zero numerator/gradient, no crossing fallback.
    crossed = field([0.,sign,0.]).requires_grad_()
    # Isolate exactly-zero agreement from float32 encoding/decoding roundoff.
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_trace_3d.coarse_supervision.decode_lasagna_direction_3x2_analytic',
        lambda values: values[...,:3]*0 + values.new_tensor([0.,1.,0.]))
    pos,_,direction,angle = coarse_sample_statistics(logits,crossed,normals,geometry,stride=1,background=mask)
    assert pos == 0 and direction == 0 and angle == 90
    grad, dgrad = torch.autograd.grad(pos+direction,(logits,crossed),allow_unused=True)
    assert not grad.any() and not dgrad.any()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('value', [-100., 100.])
def test_coarse_saturated_bf16_logits_have_finite_gradients(device, value):
    import os
    if device == 'cuda' and not os.environ.get('FIBER_COOPERATIVE_TEST_GPUS'):
        pytest.skip('CUDA requires the manager-gated test launcher')
    center = np.array([[16.25, 16.25, 16.25]], np.float32)
    geometry = CoarseGeometry(center, np.array([[0, 0, 1]], np.float32),
        np.eye(3, dtype=np.float32)[None] / 4, center, center, np.array([10.], np.float32),
        offsets=jittered_coarse_sites(128., 123)[1])
    logits = torch.full((32, 32, 32), value, device=device, dtype=torch.bfloat16, requires_grad=True)
    def field(axis):
        return encode_lasagna_direction_3x2(torch.tensor(axis, device=device))[:, None, None, None].expand(6,32,32,32)
    mask = background_mask(geometry, logits.shape, 1, device)
    with torch.autocast(device, dtype=torch.bfloat16):
        pos, neg, _, _ = coarse_sample_statistics(logits, field([1.,0,.1]), field([0.,0,1.]),
                                                 geometry, stride=1, background=mask)
        loss = pos + neg
    assert loss.dtype == torch.float32 and torch.isfinite(loss) and loss > 90
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('upsample_mode', ['pixelshuffle', 'trilinear'])
@pytest.mark.parametrize('tube_weight', [0., 1.])
def test_loader_mixed_batch_counts_and_backward_without_remote_normals(device, upsample_mode, tube_weight):
    import os
    if device == 'cuda' and not os.environ.get('FIBER_COOPERATIVE_TEST_GPUS'):
        pytest.skip('CUDA requires the manager-gated test launcher')
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader, config_from_mapping
    from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
    from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import MacroBatchTargetSource
    from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses
    from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import loss_components

    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        points = np.array([[80, 96, 96], [112, 96, 96]], np.float32)
        baked = {'baked_annotations': {'version': 1, 'normals_xyz': [[0, 0, 1], [0, 0, 1]]}}
        records = [dict(volume=np.zeros((192,)*3, np.uint8),
                        fiber=Vc3dFiber(None, 1, points, points[:1], (None,), 0, metadata),
                        scroll_id='A', dense_supervision=True, supervision_mode=mode)
                   for mode, metadata in [('precise', baked), ('coarse_area',
                       {'width': 80, 'hv_classification': {'manual_tag': 'H', 'automatic_tag': 'V'}})]]
        from vesuvius.neural_tracing.fiber_trace_3d.coarse_supervision import CoarseTubeConfig
        tube_config = CoarseTubeConfig(weight=tube_weight)
        config = config_from_mapping(dict(_array_records=records, mixed_supervision=True,
            coarse_tube={'weight': tube_weight},
            datasets=[dict(array_records=True, dense_supervision=True)],
            training={'optimizer_global_batch_size': 2}, batch_size=2, patch_shape_zyx=[128]*3,
            sampling_strategy='scroll_balanced_arclength', presence_target_mode='anisotropic_gaussian',
            presence_radius_voxels=1, presence_gaussian_width_ratio=2,
            annotations={'normal_weight': 10, 'hv_weight': .01}, augment_enabled=False, image_normalization='none',
            geometry_backend='eager', value_augmentation_backend='torch',
            round_source_to_chunk_boundaries=False))
        loader = FiberTrace3DLoader(config)
        raw = loader.load_batch(0)
        assert sum(g is not None for g in raw.coarse_geometry) == 1
        coarse_index = next(i for i, g in enumerate(raw.coarse_geometry) if g is not None)
        rows = raw.annotation_targets[coarse_index]
        assert len(rows) > 0 and np.all(rows[:, 7] == 0)
        assert np.all(rows[:, 3:7] == 0)  # No normal/width labels or remote reads.
        assert all(g is None for g in raw.refinement_geometry)
        target = materialize_targets(raw.to(device), config, output_stride=4)
        output = torch.full((2, 7, 32, 32, 32), .51, device=device, requires_grad=True)
        normal = torch.full((2, 7, 32, 32, 32), .1, device=device, requires_grad=True)
        stats = compute_losses(output, target, direction_weight=10, presence_weight=1,
            coarse_tube_config=tube_config,
            presence_logits=torch.logit(output[:, 6]),
            annotation_output=normal, annotation_config=config.annotations, _return_statistics=True)
        counts = MacroBatchTargetSource(None, config=config, device=torch.device(device),
                                       output_stride=4, batch_sizes=[2])._counts(raw)
        names = loss_components(SimpleNamespace(annotations=config.annotations, coarse_supervision=True, coarse_tube=tube_config),
                                gaussian_presence=True)
        actual = torch.stack([getattr(stats, name+'_count') for name in names])
        torch.testing.assert_close(counts.to(device), actual, rtol=0, atol=0)
        assert stats.coarse_positive_count == 1
        assert stats.hv_count == len(rows)
        assert stats.hv_unweighted_sum > 0
        assert stats.positive_presence_count == 1
        assert stats.negative_presence_count == 1
        losses = compute_losses(output, target, direction_weight=10, presence_weight=1,
            coarse_tube_config=tube_config,
            presence_logits=torch.logit(output[:, 6]),
            annotation_output=normal, annotation_config=config.annotations)
        losses['total'].backward()
        assert torch.isfinite(output.grad).all()
        assert torch.isfinite(normal.grad).all()
        # Exact same objective when rank/microbatch partition changes. Slicing
        # must retain coarse metadata, and all counts precede every forward.
        from vesuvius.neural_tracing.fiber_trace_3d.train import _slice_batch
        from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import NormalizedGradientAccumulator
        model = torch.nn.Module()
        model.p = torch.nn.Parameter(output.detach())
        model.n = torch.nn.Parameter(normal.detach())
        model.annotations = config.annotations
        model.coarse_supervision = True
        model.coarse_tube = tube_config
        accumulator = NormalizedGradientAccumulator(model, counts, direction_weight=10,
            presence_weight=1, gaussian_presence=True)
        for i in range(2):
            part = _slice_batch(target, i, i+1)
            values = compute_losses(model.p[i:i+1], part, direction_weight=10, presence_weight=1,
                coarse_tube_config=tube_config,
                presence_logits=torch.logit(model.p[i:i+1, 6]),
                annotation_output=model.n[i:i+1], annotation_config=config.annotations,
                _return_statistics=True)
            accumulator.add(values)
        accumulator.reduce_and_assign()
        torch.testing.assert_close(model.p.grad, output.grad, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(model.n.grad, normal.grad, rtol=2e-5, atol=1e-7)
        if device == 'cuda':
            import time
            from dataclasses import replace
            from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
            from vesuvius.neural_tracing.fiber_trace_3d.train import _forward_loss_statistics, _finalize_loss_statistics
            torch.manual_seed(0)
            native_stride = 2 if tube_weight else 1
            network = build_fiber_trace_3d_model(dict(mixed_supervision=True,
                coarse_tube={'weight': tube_weight},
                annotations={'normal_weight': 10, 'hv_weight': .01}, patch_shape_zyx=[128]*3,
                model_3d=dict(unet_base_channels=16, unet_depth=5, output_stride=native_stride,
                              normalization='group', decoder_upsample_mode=upsample_mode))).to(device)
            target = materialize_targets(raw.to(device), config, output_stride=native_stride)
            counts = MacroBatchTargetSource(None, config=config, device=torch.device(device),
                                           output_stride=native_stride, batch_sizes=[2])._counts(raw)
            optimizer = torch.optim.AdamW(network.parameters(), lr=.001)
            times = []
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                accumulation = NormalizedGradientAccumulator(network, counts, direction_weight=10,
                    presence_weight=1, gaussian_presence=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    stats = _forward_loss_statistics(network, target)
                    accumulation.add(stats)
                accumulation.reduce_and_assign()
                assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
                optimizer.step()
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                evaluated = _finalize_loss_statistics(_forward_loss_statistics(network, target, training_loss=False),
                    direction_weight=10, presence_weight=1)
                assert torch.isfinite(evaluated['total'])
            # Isolate the added loss on the same cached predictions/targets.
            with torch.no_grad():
                prediction = network(target.volume, return_embeddings=True)
                loss_ms = {}
                for label, batch in [('without_coarse', replace(target, coarse_geometry=None)), ('with_coarse', target)]:
                    durations = []
                    for _ in range(10):
                        torch.cuda.synchronize(); start = time.perf_counter()
                        compute_losses(prediction['prediction'], batch, direction_weight=10, presence_weight=1,
                            presence_logits=prediction['presence_logits'],
                            annotation_output=prediction['annotations'], annotation_config=config.annotations)
                        torch.cuda.synchronize(); durations.append((time.perf_counter()-start)*1000)
                    loss_ms[label] = dict(median=np.median(durations), p95=np.percentile(durations,95))
            print('mixed smoke: 1 GPU, batch2,128^3,fullres UNet16/depth5,bf16; '
                  f'4 updates seconds={times}; isolated loss milliseconds={loss_ms}; '
                  f'peak_allocated_GiB={torch.cuda.max_memory_allocated()/1024**3:.3f}', flush=True)
    finally:
        torch.set_num_threads(before_threads)


def test_real_mixed_data_and_refinement_boundary(tmp_path, monkeypatch):
    import os
    if os.environ.get('FIBER_MIXED_DATA_TEST') != '1':
        pytest.skip('real-data preflight requires the approved test launcher')
    from pathlib import Path
    from collections import Counter
    from dataclasses import replace
    from vesuvius.neural_tracing.fiber_trace_3d.loader import load_config, FiberTrace3DLoader
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import baked_normals
    monkeypatch.setenv('VILLA_FIBER_REFINEMENT_STATE', str(tmp_path / 'geometry'))
    config_path = Path(__file__).resolve().parents[2] / 'src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_mixed_reviewed_coarse_16x8.json'
    config = load_config(config_path)
    loader = FiberTrace3DLoader(config)
    counts = Counter((r.dataset_config['supervision_mode'], r.dataset_config['scroll_id']) for r in loader.records)
    for record, key in zip(loader.records, loader._refinement_keys, strict=True):
        coarse = record.dataset_config['supervision_mode'] == 'coarse_area'
        assert (key is None) == coarse
        if not coarse:
            assert baked_normals(record.fiber) is not None
        assert not record.dataset_config.get('lasagna_manifest_path')
    selected = [loader.descriptor_for_sample_index(i)[1] for i in range(128)]
    assert sum(loader.records[i].dataset_config['supervision_mode'] == 'coarse_area' for i in selected) == 64
    # Exercise the actual vc3d reader, geometry transforms and sparse metadata.
    modes = set()
    for index in range(128):
        mode = loader.records[selected[index]].dataset_config['supervision_mode']
        if mode in modes:
            continue
        sample = loader.load_raw_sample(index).sample
        assert (sample.coarse_geometry is not None) == (mode == 'coarse_area')
        if mode == 'coarse_area':
            assert sample.refinement_geometry is None
        modes.add(mode)
        if len(modes) == 2:
            break
    assert len(modes) == 2
    print('real mixed eligible records by (mode,scroll):', dict(counts), flush=True)
    # Exercise actual locked test bindings; any remote normal resolution here
    # is a regression, not a permitted fallback.
    import json
    from vesuvius.neural_tracing.fiber_trace_3d import evaluation_workload
    from vesuvius.neural_tracing.fiber_trace_3d.training_trace_benchmark import build_workload
    checkpoint = Path('/ephemeral/hendrik/fiber3d_train2/runs/reviewed_normal_width_hv_fullres_128_sd1_bs128_ema_20261007_225939/snapshots/current.pt')
    template = json.loads((config_path.parent / 'eval_reviewed_failure_fiber_normals_fw512.json').read_text())
    workload = build_workload(template, checkpoint=checkpoint, output_root=tmp_path/'prepared',
        split_root=Path('/ephemeral/hendrik/fiber3d_train/2026-10-05_reviewed_failure_union_baked_v1'),
        devices='0', native_binary=Path('/home/hendrik/villa/volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric'))
    def forbid(*args, **kwargs):
        raise AssertionError('no runtime Lasagna lookup allowed')
    monkeypatch.setattr(evaluation_workload, 'resolve_normal_binding', forbid)
    evaluation_workload.prepare(workload)
    prepared, audit = evaluation_workload.audit_prepared(workload)
    assert sum(len(s['fibers']) for s in prepared['sources']) == 32
    assert all(s['reference_normals'] == 'baked' and 'normal_manifest' not in s for s in prepared['sources'])
