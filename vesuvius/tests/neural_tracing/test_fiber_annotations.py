from dataclasses import replace
import copy
import numpy as np
import pytest
import torch

from test_fiber_trace_3d import _loader
from vesuvius.neural_tracing.fiber_trace_3d.annotations import AnnotationConfig, baked_normals, annotation_counts
from vesuvius.neural_tracing.fiber_trace_3d.model import FiberTrace3DNet, FiberTrace3DModelConfig
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import MacroBatchTargetSource
from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import NormalizedGradientAccumulator
from vesuvius.neural_tracing.fiber_trace_3d.train import _forward_loss_statistics, _finalize_loss_statistics, _slice_batch


@pytest.fixture(autouse=True)
def single_core():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def annotated_loader(augment=False):
    loader = _loader(augment_enabled=augment)
    loader.config = replace(loader.config, annotations=AnnotationConfig(1., .1, 1.))
    record = loader.records[0]
    fiber = record.fiber
    normals = np.tile([0., 0., 1.], (len(fiber.line_points_xyz), 1))
    normals[1::2] *= -1  # sign-ambiguous labels must still interpolate correctly.
    fiber = replace(fiber, metadata={**fiber.metadata, "width": 8.,
        "hv_classification": {"manual_tag": "V"},
        "baked_annotations": {"version": 1, "normals_xyz": normals.tolist()}})
    loader.records[0] = replace(record, fiber=fiber)
    return loader


def model():
    return FiberTrace3DNet(FiberTrace3DModelConfig(features_per_stage=(4, 8),
        normalization="group", annotations=AnnotationConfig(1., .1, 1.)))


def test_baked_source_free_loading_and_geometry():
    loader = annotated_loader()
    batch = loader.load_batch(0)
    points = sum(len(rows) for rows in batch.annotation_targets)
    assert annotation_counts(batch, loader.config.annotations) == [6 * points, points, points]
    for rows in batch.annotation_targets:
        np.testing.assert_allclose(np.linalg.norm(rows[:, 3:6], axis=1), 1.)
        np.testing.assert_allclose(rows[:, 6], 8., atol=1e-5)
        assert (rows[:, 7] == 1).all()
    augmented = annotated_loader(True).load_batch(0)
    for rows in augmented.annotation_targets:
        np.testing.assert_allclose(np.linalg.norm(rows[:, 3:6], axis=1), 1., atol=1e-5)
        assert np.all((rows[:, 6] >= 7.6) & (rows[:, 6] <= 8.4))


def test_head_layout_and_global_single_backward():
    loader = annotated_loader()
    batch = loader.load_batch(0)
    cfg = loader.config
    batch = materialize_targets(batch, cfg)
    # Unequal label counts must not turn into equal patch weights.
    batch = replace(batch, annotation_targets=(batch.annotation_targets[0][:1], batch.annotation_targets[1]))
    torch.manual_seed(7)
    actual = model()
    expected = copy.deepcopy(actual)
    assert actual(batch.volume).shape[1] == 7
    output = actual(batch.volume, return_annotations=True)
    assert output['annotations'].shape[1] == 8
    assert output['embeddings'].shape[1] == 0
    source = MacroBatchTargetSource(None, config=cfg, device=torch.device('cpu'), output_stride=1, batch_sizes=[2])
    counts = source._counts(batch)
    points = sum(len(rows) for rows in batch.annotation_targets)
    assert counts[-3:].tolist() == [6 * points, points, points]
    stats = _forward_loss_statistics(expected, batch)
    losses = _finalize_loss_statistics(stats, direction_weight=1., presence_weight=1.)
    losses['total'].backward()
    accumulator = NormalizedGradientAccumulator(actual, counts, direction_weight=1., presence_weight=1.)
    for i in range(2):
        accumulator.add(_forward_loss_statistics(actual, _slice_batch(batch, i, i+1)))
    accumulator.reduce_and_assign()
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=2e-4)
    for name in ('normal', 'width', 'hv'):
        assert float(losses[name].detach()) > 0


def test_missing_annotations_zero_not_background():
    loader = _loader()
    loader.config = replace(loader.config, annotations=AnnotationConfig(1, 1, 1))
    batch = materialize_targets(loader.load_batch(0), loader.config)
    assert annotation_counts(batch, loader.config.annotations) == [0, 0, 0]
    stats = _forward_loss_statistics(model(), batch)
    for name in ('normal', 'width', 'hv'):
        assert getattr(stats, name + '_sum') == 0
        assert getattr(stats, name + '_count') == 0


def test_baked_alignment_fails_closed():
    fiber = annotated_loader().records[0].fiber
    broken = replace(fiber, metadata={**fiber.metadata,
        'baked_annotations': {'version': 1, 'normals_xyz': [[0., 0., 1.]]}})
    with pytest.raises(ValueError, match='misaligned'):
        baked_normals(broken)


def test_nonlinear_geometry_and_dataset_affine():
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import line_targets
    from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import apply_fiber_transform
    loader = annotated_loader()
    original = loader.records[0].fiber
    transformed = apply_fiber_transform(original, np.diag([2., 3., 4., 1.]))
    np.testing.assert_allclose(np.abs(baked_normals(transformed)), [[1., 0., 0.]] * len(original.line_points_xyz))
    # Synthetic fiber follows X, transverse Y: width scales by 3, not 2 or 4.
    np.testing.assert_allclose(transformed.metadata['baked_annotations']['widths_base'], 24.)
    params = loader._sample_augment_params(loader.records[0], 0, 5)
    map_original = loader._source_points_to_output_np
    def nonlinear(points, params, **kwargs):
        out, valid = map_original(points, params, **kwargs)
        out[:, 2] += .01 * out[:, 1] ** 2
        return out, valid
    loader._source_points_to_output_np = nonlinear
    rows = line_targets(loader, loader.records[0], params)
    assert len(rows)
    np.testing.assert_allclose(np.linalg.norm(rows[:, 3:6], axis=1), 1., atol=1e-5)
    assert np.all(rows[:, 6] >= 8.)


def test_invalid_baked_regions_are_not_supervised():
    loader = annotated_loader()
    record = loader.records[0]
    baked = record.fiber.metadata['baked_annotations']
    n = len(record.fiber.line_points_xyz)
    baked.update(normals_xyz=np.zeros((n, 3)).tolist(), valid=[False] * n)
    batch = loader.load_batch(0)
    assert annotation_counts(batch, loader.config.annotations) == [0, 0, sum(len(r) for r in batch.annotation_targets)]


def test_auxiliary_losses_weight_points_not_patches():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import annotation_statistics
    # Three H points and one V point: mean over four points, not two patches.
    a = np.array([[1, 1, 1, 0, 0, 1, 4, 0]] * 3, dtype=np.float32)
    b = np.array([[1, 1, 1, 0, 0, 1, 4, 1]], dtype=np.float32)
    batch = SimpleNamespace(volume=torch.zeros(2, 1, 4, 4, 4), annotation_targets=(a, b), presence_target_mode='binary')
    cfg = AnnotationConfig(1., .1, 1.)
    logits = torch.ones(2, 8, 4, 4, 4, requires_grad=True)
    stats = annotation_statistics(logits, batch, cfg)
    assert annotation_counts(batch, cfg) == [24, 4, 4]
    assert [stats[k + '_count'].item() for k in cfg.names] == [24, 4, 4]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.ones(4), torch.tensor([0., 0., 0., 1.]))
    torch.testing.assert_close(stats['hv_sum'] / stats['hv_count'], expected)


def test_native_baker_corrections_and_validity(tmp_path):
    import json
    import subprocess
    from pathlib import Path
    from tests.zarr_utils import create_v2_group_array
    import os
    import shutil
    binary = Path(os.environ.get('VC_FIBER_BAKE_BINARY') or shutil.which('vc_fiber_bake_annotations') or
                  Path(__file__).resolve().parents[3] / 'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_bake_annotations')
    if not binary.exists():
        pytest.skip('native annotation baker not built')
    groups = {}
    for name, value in (('grad_mag', 255), ('nx', 128), ('ny', 128)):
        a = create_v2_group_array(tmp_path / (name + '.zarr'), '0', shape=(8,8,8), chunks=(4,4,4), dtype='uint8')
        a[:] = value
        if name == 'grad_mag':
            a[2,2,3] = 0
        groups[name] = dict(zarr=name+'.zarr/0', scaledown=0, channels=[name])
    manifest = tmp_path / 'normal.json'
    manifest.write_text(json.dumps(dict(version=2, source_to_base=1, grad_mag_encode_scale=1000,
                                        grad_mag_factor=.25, groups=groups, base_shape_zyx=[8,8,8])))
    # v1 controls have no correction; native library returns decoded +Z.
    fiber = dict(type='vc3d_fiber', version=1, width=10., vc_open_data_coordinate_space='test/scan@L0',
                 line_points=[[1,2,2],[2,2,2],[3,2,2],[4,2,2]], control_points=[[1,2,2],[4,2,2]])
    result = subprocess.run([str(binary)], input=json.dumps(dict(fiber=fiber, manifest=str(manifest)))+'\n',
                            text=True, capture_output=True, check=True, timeout=30)
    output = json.loads(result.stdout)
    assert output['valid'] == [True, True, False, True]
    np.testing.assert_allclose(np.abs(output['normals_xyz'][0]), [0,0,1], atol=.01)
    assert output['normals_xyz'][2] == [0,0,0]
    from tests.test_vc3d_fiber_format_tags import _cspline_segment
    fiber.update(version=3, optimization_mode='native_fiber_trace3d', control_points=[
        dict(position=[1,2,2], display_normal=[0,1,0], display_normal_source='manual',
             segment_to_next=_cspline_segment()),
        dict(position=[4,2,2], display_normal=[0,1,0], display_normal_source='manual')])
    corrected = subprocess.run([str(binary)], input=json.dumps(dict(fiber=fiber, manifest=str(manifest)))+'\n',
                               text=True, capture_output=True, check=True, timeout=30)
    result = json.loads(corrected.stdout)
    np.testing.assert_allclose(np.abs(result['normals_xyz'][0]), [0,1,0], atol=1e-6)
    fiber['control_points'][0]['display_normal'] = [1,0,0]
    invalid = subprocess.run([str(binary)], input=json.dumps(dict(fiber=fiber, manifest=str(manifest)))+'\n',
                             text=True, capture_output=True, timeout=30)
    assert invalid.returncode != 0 and 'refusing implicit reset' in invalid.stderr


def test_baked_eligible_span_alignment_and_resampling():
    from test_training_spans import example
    from vesuvius.neural_tracing.fiber_trace_3d.training_spans import training_fiber_spans
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import resample_baked_metadata
    fiber = example()
    count = len(fiber.line_points_xyz)
    fiber = replace(fiber, metadata={'width': 8., 'baked_annotations': {
        'version': 1, 'normals_xyz': [[0,0,1]] * count,
        'valid': [True] * count, 'widths_base': list(range(count))}})
    first, second = training_fiber_spans(fiber)
    assert first.metadata['baked_annotations']['widths_base'] == list(range(10,31))
    assert len(baked_normals(second)) == len(second.line_points_xyz)
    resampled = resample_baked_metadata(first, np.arange(21), np.linspace(0,20,11))
    assert len(resampled['baked_annotations']['normals_xyz']) == 11
    assert resampled['baked_annotations']['widths_base'] == list(range(10,31,2))
