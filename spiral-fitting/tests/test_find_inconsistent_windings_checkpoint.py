from types import SimpleNamespace

import numpy as np
import pytest
import torch

from config import Config
from find_inconsistent_windings import build_fit_inputs, build_transform
from transforms import SpiralAndTransform
from fit_spiral import FitContext


@pytest.mark.parametrize('use_fiber_directions', [False, True])
def test_analysis_reaches_patch_loading_without_training_samples(monkeypatch, use_fiber_directions):
    saved_cfg = Config().as_dict()
    saved_cfg.update(
        input_use_fiber_directions=use_fiber_directions,
        loss_weight_fiber_directions=1.0,
        input_use_outer_shell=True,
        dense_spacing_mode='winding_model',
        loss_weight_shell_outer=1.0,
        loss_weight_shell_patch_radius=1.0,
    )
    checkpoint = dict(cfg=saved_cfg, z_begin=0, z_end=64)
    cfg, scroll, paths, _, _ = build_fit_inputs(
        checkpoint, '/unused/patches', [], 8, 56, '/unused/umbilicus.json',
        fibers_path='/unused/fibers')
    context = FitContext(cfg, scroll=scroll, paths=paths)
    assert context.fiber_directions_path is None
    assert context.fibers_path == '/unused/fibers'
    assert saved_cfg['loss_weight_fiber_directions'] == 1.0
    monkeypatch.setattr(context, 'umbilicus_z_to_yx', lambda: lambda z: np.zeros((len(z), 2)))

    class ReachedPatchLoading(Exception):
        pass

    def stop_at_patches(*args, **kwargs):
        raise ReachedPatchLoading

    # Exercise the real host loader through its training-input validation,
    # stopping before patch I/O or any device setup is needed.
    monkeypatch.setattr(context, '_load_patches_from_dir', stop_at_patches)
    with pytest.raises(ReachedPatchLoading):
        context.load_host_inputs()


def test_legacy_checkpoint_preserves_transform_and_strict_loading(capsys):
    cfg = Config().as_dict()
    cfg.update(
        model_flow_field_type='cartesian',
        model_flow_bounds_radius=32,
        model_flow_bounds_z_margin=0,
        model_flow_voxel_resolution=4,
        model_linear_z_resolution=16,
        model_gap_expander_logit_resolution=16,
        model_gap_expander_capacity_windings=4,
        model_gap_expander_num_windings=4,
    )
    model = SpiralAndTransform(
        cfg['model_num_flow_integration_steps'],
        cfg['model_flow_integration_solver'],
        torch.tensor([0, -32, -32]), torch.tensor([64, 32, 32]),
        torch.tensor([[0., 0., 0.], [63., 0., 0.]]), cfg,
    )
    state = model.state_dict()
    state['shell_outer_winding_param'] = torch.tensor(130.)
    saved_cfg = dict(cfg, optimizer_reset_interval=0)
    saved_cfg.pop('loss_weight_patch_radius')
    checkpoint = dict(cfg=saved_cfg, spiral_and_transform=state, z_begin=0, z_end=64)
    fit_cfg, scroll, _, lo, hi = build_fit_inputs(
        checkpoint, '/unused/patches', [], 8, 56, '/unused/umbilicus.json')
    assert 'optimizer_reset_interval' not in fit_cfg
    assert fit_cfg['loss_weight_patch_radius'] == cfg['loss_weight_patch_radius']
    assert 'optimizer_reset_interval' in capsys.readouterr().out
    assert (lo, hi) == (0, 64)
    context = SimpleNamespace(
        umbilicus_z_to_yx=lambda: lambda z: np.zeros((len(z), 2)),
        spiral_outward_sense=scroll.spiral_outward_sense,
    )
    points = torch.tensor([[32., 10., 15.]])
    with torch.no_grad():
        transform, dr = build_transform(checkpoint, fit_cfg, context, lo, hi, device='cpu')
        torch.testing.assert_close(transform(points), model.get_slice_to_spiral_transform()(points))
        torch.testing.assert_close(dr, model.get_dr_per_winding())
    assert 'shell_outer_winding_param' in state
    assert 'optimizer_reset_interval' in saved_cfg

    del state['linear_logits']
    with pytest.raises(RuntimeError, match='linear_logits'):
        build_transform(checkpoint, fit_cfg, context, lo, hi, device='cpu')


def test_analysis_scroll_carries_dataset_base_shape(tmp_path):
    import json
    from find_inconsistent_windings import resolve_base_shape_zyx

    saved_cfg = Config().as_dict()
    patches_dir = tmp_path / 'verified_patches'
    patches_dir.mkdir()

    # Checkpoints record the domain they were fit against; that wins.
    checkpoint = dict(cfg=saved_cfg, z_begin=0, z_end=64, base_shape_zyx=[100, 50, 50])
    _, scroll, _, _, _ = build_fit_inputs(
        checkpoint, str(patches_dir), [], 8, 56, '/unused/umbilicus.json')
    assert scroll.base_shape_zyx == (100, 50, 50)

    # Legacy checkpoints fall back to the dataset's spiral-scroll.json.
    legacy = dict(cfg=saved_cfg, z_begin=0, z_end=64)
    assert resolve_base_shape_zyx(legacy, str(patches_dir)) is None
    (tmp_path / 'spiral-scroll.json').write_text(json.dumps(dict(
        schema_version=1, name='x', voxel_size_um=9.6, spiral_outward_sense='CW',
        base_shape_zyx=[18946, 8174, 8174])))
    _, scroll, _, _, _ = build_fit_inputs(
        legacy, str(patches_dir), [], 8, 56, '/unused/umbilicus.json')
    assert scroll.base_shape_zyx == (18946, 8174, 8174)
