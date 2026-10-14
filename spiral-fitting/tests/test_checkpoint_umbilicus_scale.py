"""Checkpoints preserve the coordinate domain of the fitted scroll axis."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from config import Config, FitConfig
from fit_session import ScrollSpec, SpiralInputPaths
from fit_spiral import FitContext
from find_inconsistent_windings import build_fit_inputs
from flatten_spiral_checkpoint import _build_model, _checkpoint_config
from test_winding_seed_checkpoint import inference
from transforms import SpiralAndTransform


@pytest.fixture(params=[0.5, 1.0])
def fitted(request, tmp_path):
    scale = request.param
    z_begin, z_end = int(3100 * scale), int(3100 * scale) + 260
    config = Config().as_dict()
    config.update(
        z_begin=z_begin, z_end=z_end,
        model_initial_dr_per_winding=10., model_flow_field_type="cartesian",
        model_flow_voxel_resolution=16, model_linear_z_resolution=32,
        model_gap_expander_num_windings=14, model_gap_expander_capacity_windings=14,
        model_gap_expander_logit_resolution=24, model_gap_expander_lr_scale=100.,
        model_flow_bounds_z_margin=0, model_flow_bounds_radius=160,
        model_num_flow_integration_steps=24, model_num_flow_stages=1,
        model_flow_integration_solver="rk4", output_first_winding=10,
        output_step_size=8, shell_outer_winding_idx=12,
    )
    umbilicus = Path(__file__).parent / "data/umbilicus.json"
    context = FitContext(
        FitConfig(config),
        scroll=ScrollSpec(name="PHerc0139", voxel_size_um=9.596 * scale,
                          spiral_outward_sense="CW", umbilicus_coordinate_scale=scale),
        paths=SpiralInputPaths(umbilicus=str(umbilicus)),
    )
    zs = np.arange(z_begin, z_end)
    centre = context.umbilicus_z_to_yx()
    model = SpiralAndTransform(
        24, "rk4", torch.tensor([z_begin, -160, -160]),
        torch.tensor([z_end, 160, 160]),
        torch.from_numpy(np.c_[zs, centre(zs)].astype(np.float32)), config,
    ).eval()
    with torch.no_grad():
        model.linear_logits[:, 0, 0] = .001
    context.spiral_and_transform = model
    context.optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    context.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(context.optimiser, gamma=.99)
    context.model_z_begin, context.model_z_end = z_begin, z_end
    context.winding_inference = None
    context.winding_model_mode = False
    path = tmp_path / "fit.ckpt"
    context.save_checkpoint(str(path), 0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return context, checkpoint, path, umbilicus


def test_scaled_checkpoint_restores_export_and_analysis_geometry(fitted):
    context, checkpoint, _, umbilicus = fitted
    assert checkpoint["umbilicus_coordinate_scale"] == context.scroll.umbilicus_coordinate_scale
    restored = _build_model(checkpoint, _checkpoint_config(checkpoint), umbilicus, torch.device("cpu"))
    points = torch.tensor([[context.model_z_begin + 40., 120., 80.],
                           [context.model_z_begin + 80., -90., 150.]])
    with torch.inference_mode():
        expected = context.spiral_and_transform.get_slice_to_spiral_transform().inv(points)
        torch.testing.assert_close(restored.get_slice_to_spiral_transform().inv(points), expected, rtol=0, atol=0)
    config, scroll, paths, z_begin, z_end = build_fit_inputs(
        checkpoint, ".", [], context.model_z_begin, context.model_z_end, str(umbilicus))
    analysis = FitContext(config, scroll=scroll, paths=paths)
    np.testing.assert_array_equal(analysis.umbilicus_z_to_yx()(points[:, 0].numpy()),
                                  context.umbilicus_z_to_yx()(points[:, 0].numpy()))


def test_scaled_checkpoint_restores_winding_seed_coordinates(fitted):
    context, _, path, umbilicus = fitted
    args = inference.argparse.Namespace(
        fit_checkpoint=str(path), umbilicus=str(umbilicus), z_range=None,
        winding_range=(10, 12), winding_step=1, seed_spacing=16,
    )
    rays = inference.build_seed_rays(args, device="cpu")
    from sample_spiral import get_spiral_yxs
    with torch.inference_mode():
        model = context.spiral_and_transform
        z_rows = torch.arange(context.model_z_begin, context.model_z_end, 16, dtype=torch.float32)
        yxs = get_spiral_yxs(13, model.get_dr_per_winding(), 16, True)
        transform = model.get_slice_to_spiral_transform()
        for winding in (10, 11, 12):
            grid = torch.cat([
                z_rows[:, None, None].expand(-1, len(yxs[winding]), 1),
                yxs[winding][None].expand(len(z_rows), -1, 2),
            ], dim=-1)
            expected = transform.inv(grid.reshape(-1, 3)).numpy()[:, ::-1]
            np.testing.assert_array_equal(
                rays["seed_xyz"][rays["seed_winding"] == winding], expected,
            )


def test_scale_mismatch_refuses_checkpoint_without_changing_live_model(fitted):
    context, checkpoint, _, _ = fitted
    assert context.inspect_checkpoint(checkpoint).accepted
    before = {key: value.clone() for key, value in context.spiral_and_transform.state_dict().items()}
    context.scroll = replace(context.scroll, umbilicus_coordinate_scale=2.)
    verdict = context.inspect_checkpoint(checkpoint)
    assert not verdict.accepted
    assert any("umbilicus_coordinate_scale" in reason for reason in verdict.reasons)
    for key, value in context.spiral_and_transform.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    # Legacy checkpoints have no recoverable scale; retain the existing load rule.
    legacy = dict(checkpoint)
    legacy.pop("umbilicus_coordinate_scale")
    assert context.inspect_checkpoint(legacy).accepted
