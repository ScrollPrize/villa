"""Synthetic sheet volumes and differentiable transforms for SDT tests."""

import math
import numpy as np
import torch


DR_PER_WINDING = 10.0
TWO_PI = 2.0 * math.pi


def spiral_shifted_radius(points):
    """Shared spiral -> shifted-radius prologue for the mock transforms: the
    spiral has radius 0 at winding angle 0 and grows at DR_PER_WINDING, so a
    perfect fit maps winding k to shifted radius k * DR_PER_WINDING."""
    y, x = points[:, 1], points[:, 2]
    radius = torch.sqrt(y * y + x * x + 1e-12)
    theta = torch.arctan2(y, x) % TWO_PI
    return radius - theta / TWO_PI * DR_PER_WINDING


def make_volume(array_u8, *, scale=1.0, z_origin=0, unit=1.0, offset=128,
                cap=127.0, kind='sdt'):
    volume = torch.as_tensor(np.asarray(array_u8), dtype=torch.uint8)
    return {
        'backend': 'dense',
        'kind': kind,
        'volume': volume,
        'z_origin': z_origin,
        'scale_zyx': (scale,) * 3,
        'unit': unit,
        'offset': offset,
        'cap': cap,
        'shape': tuple(volume.shape),
        'fingerprint': {},
    }


def sheet_volume(x_size, sheet_centers, half_thickness=2.0, zy=(4, 4)):
    """Planar sheets normal to x: sd(x) = min |x - c| - half_thickness, exact
    under the encoding since the field varies along one axis only."""
    x = np.arange(x_size, dtype=np.float32)
    sd = np.min([np.abs(x - c) for c in sheet_centers], axis=0) - half_thickness
    encoded = (np.clip(np.rint(sd), -127, 127) + 128).astype(np.uint8)
    return make_volume(np.broadcast_to(encoded, (*zy, x_size)).copy())


class PerfectSpiralToX:
    """A 'fit' whose windings sit exactly on sheets at x = k * dr: maps a
    spiral point to (z0, y0, shifted_radius + x_offset), optionally scaled.

    ``radial_scale`` < 1 collapses windings together; ``x_offset`` shifts the
    whole fit off the sheets. Differentiable through both parameters when they
    are tensors, which the gradient-recovery tests rely on.
    """

    def __init__(self, z0=1.5, y0=1.5, x_offset=0.0, radial_scale=1.0):
        self.z0, self.y0 = z0, y0
        self.x_offset = x_offset
        self.radial_scale = radial_scale

    def inv(self, points):
        mapped_x = self.x_offset + self.radial_scale * spiral_shifted_radius(points)
        return torch.stack([
            torch.full_like(mapped_x, self.z0),
            torch.full_like(mapped_x, self.y0),
            mapped_x,
        ], dim=-1)


def spacing_cfg(**overrides):
    cfg = {
        'sample_count_dense_spacing_pairs': 256,
        'dense_spacing_pair_m_short': (1, 1),
        'dense_spacing_pair_m_long': (1, 1),
        'dense_spacing_pair_long_fraction': 0.0,
        'dense_spacing_count_temperature_wv': 0.5,
        'sample_count_dense_spacing_count_extra_pairs': 0,
        'dense_spacing_target_step_wv': 1.0,
        'dense_spacing_max_step_wv': 2.0,
        'dense_spacing_max_steps': 64,
        'dense_spacing_step_oversample': 1.25,
        'dense_spacing_use_support_gate': True,
        'dense_spacing_support_sigma': 4.0,
        'dense_spacing_support_floor_alpha': 0.05,
        'dense_spacing_support_policy': 'product',
        'dense_spacing_phase_huber_delta': 0.5,
        'dense_spacing_phase_extension_windings': 1.0,
        'dense_spacing_phase_min_center_gap_wv': 4.0,
        'dense_spacing_phase_graze_dot': 0.4,
        'dense_spacing_phase_graze_depth_wv': 1.0,
        'dense_spacing_phase_window_windings': 1.0,
        'dense_spacing_phase_end_free_margin_windings': 0.5,
        'dense_spacing_phase_missing_cost': 0.7,
        'dense_spacing_phase_missing_extend_cost': 0.7,
        'dense_spacing_phase_extra_cost': 0.9,
        'dense_spacing_phase_extra_extend_cost': 0.9,
        'dense_spacing_phase_temperature': 0.2,
        'dense_spacing_phase_band_confidence_cost': 0.25,
        'dense_spacing_phase_top2_margin': 0.2,
        'dense_spacing_phase_min_matched_windings': 2,
        'dense_spacing_phase_min_matched_mass': 1.0,
        'loss_weight_dense_spacing': 12.0,
        'loss_weight_dense_spacing_count': 8.0,
        'loss_weight_min_spacing': 0.0,
        'loss_weight_dense_attachment': 0.0,
        'dense_min_spacing_d_min_wv': 6.0,
        'sample_count_minimum_spacing_independent_samples': 64,
        'sample_count_dense_attachment_points': 512,
        'dense_attachment_scale': 8.0,
    }
    cfg.update(overrides)
    return cfg
