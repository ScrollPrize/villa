"""Shared small model fixtures for numerical reference tests."""

import torch
from config import Config
from transforms import SpiralAndTransform


def _make_small_spiral_model(seed, flow_field_type, device='cpu'):
    cfg = Config().as_dict()
    cfg['model_flow_field_type'] = flow_field_type
    cfg['model_gap_expander_num_windings'] = 10
    cfg['model_gap_expander_capacity_windings'] = 10
    z_span = 16 * 12  # 12 flow lattice voxels per axis at the default resolution
    flow_min = torch.tensor([0, -96, -96], dtype=torch.int64, device=device)
    flow_max = torch.tensor([z_span, 96, 96], dtype=torch.int64, device=device)
    zs = torch.arange(0, z_span + 1, dtype=torch.float32, device=device)
    umbilicus_zyx = torch.stack(
        [zs, torch.full_like(zs, 3.), torch.full_like(zs, -2.)], dim=-1)
    torch.manual_seed(seed)
    model = SpiralAndTransform(
        flow_integration_steps=3,
        flow_integration_solver='rk4',
        flow_min_corner_zyx=flow_min,
        flow_max_corner_zyx=flow_max,
        umbilicus_zyx=umbilicus_zyx,
        config=cfg,
        spiral_outward_sense='CW',
    ).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() > 1:
                parameter.normal_(std=0.01)
    return model


def _sample_scroll_points(num_points, seed):
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(num_points, generator=generator) * 150 + 20
    theta = torch.rand(num_points, generator=generator) * 2 * torch.pi
    radius = torch.rand(num_points, generator=generator) * 60 + 20
    y = 3. + torch.sin(theta) * radius
    x = -2. + torch.cos(theta) * radius
    return torch.stack([z, y, x], dim=-1)
