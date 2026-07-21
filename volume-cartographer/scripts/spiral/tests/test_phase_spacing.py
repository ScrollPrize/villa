"""Synthetic tests for the phase dense-spacing bundle: complete-band
detection, soft-sequence registration, shared phase/count rays, native
anti-collapse, and the two-mode session contract."""

import math
import os

import numpy as np
import pytest
import torch

from sdt_losses import (
    _build_phase_padding,
    _pair_counts_from_samples,
    detect_complete_sdt_bands,
    get_min_spacing_loss,
    iter_phase_bundle_losses,
    sample_lasagna_normals_nearest,
    sample_pair_polylines,
    sample_sdt_trilinear,
)
from transforms import GapExpanderParams, GapExpandingTransform
from fit_session import (
    SpiralInputPaths, SpiralRunConfig, apply_optional_input_selection,
    validate_session_request,
)
from test_sdt_losses import (
    DR_PER_WINDING,
    PerfectSpiralToX,
    make_volume,
    sheet_volume,
    spacing_cfg,
)


def normal_volume(shape, *, nx=255, ny=128, scale=1.0, z_origin=0):
    volume = torch.zeros([3, *shape], dtype=torch.uint8)
    volume[0].fill_(nx)
    volume[1].fill_(ny)
    return {
        'backend': 'dense', 'volume': volume, 'shape': shape,
        'lasagna_scale': scale, 'z_origin': z_origin,
    }


def fractional_sheet_volume(x_size, centers, half_thickness, unit=0.25):
    x = np.arange(x_size, dtype=np.float32)
    sd = np.min([np.abs(x - center) for center in centers], axis=0) - half_thickness
    encoded = (np.clip(np.rint(sd / unit), -127, 127) + 128).astype(np.uint8)
    return make_volume(
        np.broadcast_to(encoded, (4, 4, x_size)).copy(), unit=unit,
        cap=127 * unit)


def fixed_ray(transform, cfg, start=1.0, end=6.0):
    return sample_pair_polylines(
        transform, torch.tensor(DR_PER_WINDING),
        torch.tensor([start]), torch.tensor([end]), torch.tensor([0.0]),
        torch.tensor([1.5]), cfg, no_grad=True)


def detect(ray, volume, normals, cfg):
    """Detection-view wrapper: sample the SDT and detect complete bands."""
    field, valid, _ = sample_sdt_trilinear(volume, ray['scroll_poly'])
    view = {
        'scroll_poly': ray['scroll_poly'],
        'sample_phase': ray['sample_phase'],
        'pair_id': ray['pair_id'],
        'has_successor': ray['has_successor'],
        'num_pairs': ray['too_long'].numel(),
    }
    return detect_complete_sdt_bands(view, field, valid, normals, cfg)


def phase_cfg(**overrides):
    values = {
        'dense_spacing_num_pairs': 192,
        'dense_spacing_pair_m_short': (2, 3),
        'dense_spacing_pair_m_long': (2, 3),
        'dense_spacing_max_steps': 160,
    }
    values.update(overrides)
    return spacing_cfg(**values)


def run_bundle(volume, normals, transform, outer_winding_idx, cfg,
               generator=None):
    """Run the phase bundle and return {component: (loss, metrics)}."""
    return {
        name: (loss, metrics)
        for name, loss, metrics in iter_phase_bundle_losses(
            None, transform, torch.tensor(DR_PER_WINDING), volume, normals,
            outer_winding_idx, cfg, 1, 2, generator=generator)
    }


def run_phase(volume, normals, transform, outer_winding_idx, cfg,
              generator=None):
    return run_bundle(
        volume, normals, transform, outer_winding_idx, cfg, generator,
    )['dense_spacing_phase']


class TestCompleteBandDetection:
    def test_complete_bands_ignore_partial_ray_end_intervals(self):
        volume = sheet_volume(80, [10, 20, 30, 40, 50, 60])
        cfg = phase_cfg()
        bands = detect(
            fixed_ray(PerfectSpiralToX(), cfg, 1.0, 6.0), volume,
            normal_volume(volume['shape']), cfg)
        # The ray begins/ends at the centers of sheets 1 and 6. Those partial
        # intervals are ignored; only sheets 2..5 are complete.
        torch.testing.assert_close(
            bands['phase'], torch.tensor([2.0, 3.0, 4.0, 5.0]), atol=1e-5,
            rtol=0)
        torch.testing.assert_close(bands['width'], torch.full([4], 4.0))

    def test_missing_band_leaves_a_two_winding_phase_gap(self):
        volume = sheet_volume(80, [10, 20, 30, 50, 60])
        cfg = phase_cfg()
        bands = detect(
            fixed_ray(PerfectSpiralToX(), cfg), volume,
            normal_volume(volume['shape']), cfg)
        torch.testing.assert_close(
            bands['phase'], torch.tensor([2.0, 3.0, 5.0]), atol=1e-5,
            rtol=0)
        assert float(bands['phase'][2] - bands['phase'][1]) >= 2.0

    def test_close_complete_fragments_merge_using_projected_spacing(self):
        volume = fractional_sheet_volume(
            96, [40.0, 45.0, 60.0], half_thickness=0.75)
        cfg = phase_cfg(
            dense_spacing_target_step_wv=0.25,
            dense_spacing_max_step_wv=0.5,
            dense_spacing_max_steps=400,
        )
        bands = detect(
            fixed_ray(PerfectSpiralToX(radial_scale=2.0), cfg, 1.0, 4.0),
            volume,
            # Along-ray separation is 5 wv, but |dot| ~= 0.5 makes the
            # normal-projected separation 2.5 wv and therefore mergeable.
            normal_volume(volume['shape'], nx=192, ny=128), cfg)
        assert bands['merged_count'] == 1
        assert len(bands['phase']) == 2

    def test_graze_guard_distinguishes_shallow_from_deep_oblique_bands(self):
        # nx=ny=128 reconstructs +nz, perpendicular to this x-directed ray.
        cfg = phase_cfg()
        normals = normal_volume((4, 4, 64), nx=128, ny=128)
        shallow = fractional_sheet_volume(64, [20, 30], half_thickness=0.5)
        deep = sheet_volume(64, [20, 30], half_thickness=2.0)
        shallow_bands = detect(
            fixed_ray(PerfectSpiralToX(), cfg, 1.0, 4.0), shallow,
            normals, cfg)
        deep_bands = detect(
            fixed_ray(PerfectSpiralToX(), cfg, 1.0, 4.0), deep,
            normals, cfg)
        assert bool(shallow_bands['graze'].all())
        assert not bool(deep_bands['graze'].any())

    def test_subvoxel_band_translation_moves_localized_center(self):
        cfg = phase_cfg(
            dense_spacing_target_step_wv=0.125,
            dense_spacing_max_step_wv=0.25,
            dense_spacing_max_steps=320)
        centers = []
        for sheet_center in (20.0, 20.25):
            volume = fractional_sheet_volume(
                48, [sheet_center], half_thickness=1.0)
            bands = detect(
                fixed_ray(PerfectSpiralToX(), cfg, 1.0, 3.0), volume,
                normal_volume(volume['shape']), cfg)
            centers.append(float(bands['center'][0, 2]))
        assert centers[1] - centers[0] == pytest.approx(0.25, abs=0.03)

    def test_normal_decode_scale_origin_order_and_invalid_convention(self):
        normals = normal_volume((3, 4, 4), nx=255, ny=128, scale=2.0,
                                z_origin=3)
        # working z=8 -> stored z=4 -> local z=1
        vector, valid = sample_lasagna_normals_nearest(
            normals, torch.tensor([[8.0, 2.0, 2.0]]))
        assert bool(valid[0])
        torch.testing.assert_close(vector[0], torch.tensor([0.0, 0.0, 1.0]))
        normals['volume'][0, 1, 1, 1] = 0
        normals['volume'][1, 1, 1, 1] = 0
        _, valid = sample_lasagna_normals_nearest(
            normals, torch.tensor([[8.0, 2.0, 2.0]]))
        assert not bool(valid[0])

    def test_dense_and_mmap_normal_gathers_agree(self):
        dense = normal_volume((3, 4, 5), nx=200, ny=90, scale=2.0,
                              z_origin=2)

        class Store:
            def __init__(self, volume):
                self.volume = volume

            def gather_pair(self, normal_indices, grad_indices, device):
                z, y, x = normal_indices.unbind(dim=-1)
                values = torch.stack([
                    self.volume[0, z, y, x], self.volume[1, z, y, x]], -1)
                return values.to(device), torch.zeros(
                    [len(grad_indices)], dtype=torch.uint8, device=device)

        mmap = {key: value for key, value in dense.items() if key != 'volume'}
        mmap.update({'backend': 'mmap', 'store': Store(dense['volume'])})
        points = torch.tensor([[4.0, 2.0, 4.0], [8.0, 6.0, 8.0]])
        dense_result = sample_lasagna_normals_nearest(dense, points)
        mmap_result = sample_lasagna_normals_nearest(mmap, points)
        torch.testing.assert_close(dense_result[0], mmap_result[0])
        torch.testing.assert_close(dense_result[1], mmap_result[1])

    def test_band_spanning_invalid_interior_is_marked_ambiguous(self):
        volume = sheet_volume(80, [10, 20, 30, 40, 50, 60])
        # Poison the interior of sheet 3 (x = 30) with no-data voxels: its
        # entry/exit crossings stay valid but the interior is unobserved and
        # could hide an exit/re-entry (two sheets read as one band).
        volume['volume'][:, :, 30] = 0
        cfg = phase_cfg()
        bands = detect(
            fixed_ray(PerfectSpiralToX(), cfg, 1.0, 6.0), volume,
            normal_volume(volume['shape']), cfg)
        poisoned = (bands['phase'] - 3.0).abs().argmin()
        assert bool(bands['interior_invalid'][poisoned])
        assert bool(bands['ambiguous'][poisoned])
        clean = (bands['phase'] - 2.0).abs().argmin()
        assert not bool(bands['interior_invalid'][clean])
        assert not bool(bands['ambiguous'][clean])

    def test_non_axis_normal_uses_positive_nz_not_an_nz_only_flip(self):
        normals = normal_volume((2, 2, 2), nx=192, ny=166)
        vector, valid = sample_lasagna_normals_nearest(
            normals, torch.tensor([[0.0, 0.0, 0.0]]))
        assert bool(valid[0])
        assert float(vector[0, 0]) > 0.0
        ray = torch.nn.functional.normalize(
            torch.tensor([0.8, 0.3, 0.5]), dim=0)
        positive_dot = abs(float(torch.dot(vector[0], ray)))
        nz_flipped = vector[0].clone()
        nz_flipped[0] *= -1
        flipped_dot = abs(float(torch.dot(nz_flipped, ray)))
        assert positive_dot > flipped_dot + 0.5


class TestPhaseLoss:
    def test_perfect_stack_is_zero_and_shift_has_local_pullback_gradient(self):
        torch.manual_seed(3)
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        normals = normal_volume(volume['shape'])
        cfg = phase_cfg()
        loss, metrics = run_phase(volume, normals, PerfectSpiralToX(), 8, cfg)
        assert float(loss) < 1e-5
        assert metrics['dense_spacing_phase_valid_fraction'] > 0.9
        # Rays clamped at the fitted-domain edge legitimately miss their
        # boundary band (the ray ends at that sheet's center), so a perfect
        # stack still reports some missing mass - but no extra bands.
        assert metrics['dense_spacing_phase_missing_per_ray'] < 1.0
        assert metrics['dense_spacing_phase_extra_per_ray'] < 0.1

        offset = torch.tensor(2.0, requires_grad=True)
        torch.manual_seed(3)
        loss, metrics = run_phase(
            volume, normals, PerfectSpiralToX(x_offset=offset), 8, cfg)
        assert metrics['dense_spacing_phase_valid_fraction'] > 0.75
        loss.backward()
        # Gradient descent decreases the positive x offset toward band centers.
        assert float(offset.grad) > 0

    def test_half_winding_ambiguity_has_no_midpoint_attraction(self):
        torch.manual_seed(4)
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        offset = torch.tensor(5.0, requires_grad=True)
        loss, metrics = run_phase(
            volume, normal_volume(volume['shape']),
            PerfectSpiralToX(x_offset=offset), 8, phase_cfg())
        # Every winding sits exactly between two bands. The truly symmetric
        # (50/50) windings are suppressed by the top-2 margin; windings the
        # sequence context does break toward a band score that band - never
        # the empty midpoint - so accepted residuals sit at exactly 0.5.
        assert metrics['dense_spacing_phase_ambiguous_match_mass_fraction'] > 0.1
        assert metrics['dense_spacing_phase_residual_abs_p50'] == pytest.approx(
            0.5, abs=0.01)
        assert math.isfinite(float(loss))
        # No net midpoint-attraction gradient: the symmetric band pulls
        # cancel, unlike the decisive pullback of a genuinely shifted stack.
        loss.backward()
        symmetric_gradient = abs(float(offset.grad))

        torch.manual_seed(4)
        shifted = torch.tensor(2.0, requires_grad=True)
        shifted_loss, _ = run_phase(
            volume, normal_volume(volume['shape']),
            PerfectSpiralToX(x_offset=shifted), 8, phase_cfg())
        shifted_loss.backward()
        assert float(shifted.grad) > 0
        assert symmetric_gradient < 0.05 * abs(float(shifted.grad))

    def test_missing_band_uses_skip_state_and_resumes_matching(self):
        torch.manual_seed(7)
        # Removing sheet 40 leaves a two-winding phase gap between sheets 3/5.
        # The soft aligner explains it with a missing-observation state and
        # keeps matching beyond the hole without a one-winding suffix shift.
        volume = sheet_volume(100, [10, 20, 30, 50, 60, 70, 80])
        loss, metrics = run_phase(
            volume, normal_volume(volume['shape']), PerfectSpiralToX(), 8,
            phase_cfg(dense_spacing_pair_m_short=(3, 4),
                      dense_spacing_pair_m_long=(3, 4)))
        assert metrics['dense_spacing_phase_missing_per_ray'] > 0.1
        assert metrics['dense_spacing_phase_valid_fraction'] > 0.5
        # A one-winding suffix shift would put ~1.0-winding residuals in the
        # accepted set; recovery keeps them near zero.
        assert metrics['dense_spacing_phase_residual_abs_p95'] < 0.1
        assert float(loss) < 1e-3

    def test_extra_inserted_band_is_skipped_without_biasing_residuals(self):
        torch.manual_seed(6)
        # A spurious extra sheet at x = 45 sits mid-gap between windings 4 and
        # 5. Hard unit enumeration would hand winding 5 to the extra band and
        # shift every later target a full winding outward - a systematic
        # positive-rho bias. The extra-observation state absorbs it instead.
        volume = sheet_volume(100, [10, 20, 30, 40, 45, 50, 60, 70, 80])
        loss, metrics = run_phase(
            volume, normal_volume(volume['shape']), PerfectSpiralToX(), 8,
            phase_cfg())
        assert metrics['dense_spacing_phase_extra_per_ray'] > 0.05
        assert metrics['dense_spacing_phase_valid_fraction'] > 0.25
        assert metrics['dense_spacing_phase_residual_abs_p95'] < 0.15
        assert float(loss) < 5e-3

    def test_single_band_rays_have_zero_weight(self):
        torch.manual_seed(12)
        volume = sheet_volume(80, [30])
        loss, metrics = run_phase(
            volume, normal_volume(volume['shape']), PerfectSpiralToX(), 6,
            phase_cfg())
        assert float(loss) == 0.0
        assert metrics['dense_spacing_phase_valid_fraction'] == 0.0

    def test_over_budget_central_rays_are_wholly_rejected(self):
        torch.manual_seed(13)
        volume = sheet_volume(200, [30, 60, 90, 120, 150])
        loss, metrics = run_phase(
            volume, normal_volume(volume['shape']),
            PerfectSpiralToX(radial_scale=3.0), 6,
            phase_cfg(dense_spacing_max_steps=16))
        assert float(loss) == 0.0
        assert metrics['dense_spacing_phase_too_long_fraction'] == 1.0
        assert metrics['dense_spacing_phase_valid_fraction'] == 0.0

    def test_private_generator_does_not_advance_global_rng(self):
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        normals = normal_volume(volume['shape'])
        torch.manual_seed(99)
        expected = torch.rand(5)
        torch.manual_seed(99)
        generator = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            run_bundle(volume, normals, PerfectSpiralToX(), 8, phase_cfg(),
                       generator=generator)
        torch.testing.assert_close(torch.rand(5), expected)

    def test_detached_observations_carry_no_gradient(self):
        # The phase gradient flows only through the modeled targets: band
        # centers, directions, and reference gaps are detached observations.
        torch.manual_seed(3)
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        offset = torch.tensor(1.0, requires_grad=True)
        cfg = phase_cfg(loss_weight_dense_spacing_count=0.0)
        components = run_bundle(
            volume, normal_volume(volume['shape']),
            PerfectSpiralToX(x_offset=offset), 8, cfg)
        assert set(components) == {'dense_spacing_phase'}
        loss, _ = components['dense_spacing_phase']
        loss.backward()
        assert torch.isfinite(offset.grad)


class TestSharedRayValidity:
    def test_bad_padding_rejects_phase_but_keeps_central_count(self):
        # The mapping is regular over the central interval but pathological
        # over the outward extension, so the pad goes over budget while the
        # central ray stays valid: phase must reject, count must not.
        class StretchedPad:
            def inv(self, points):
                y, x = points[:, 1], points[:, 2]
                radius = torch.sqrt(y * y + x * x + 1e-12)
                theta = torch.atan2(y, x) % (2 * torch.pi)
                phase = (radius - theta / (2 * torch.pi) * DR_PER_WINDING) / DR_PER_WINDING
                mapped = torch.where(
                    phase <= 3.0, phase * 10.0,
                    30.0 + (phase - 3.0) * 4000.0)
                return torch.stack([
                    torch.full_like(mapped, 1.5),
                    torch.full_like(mapped, 1.5), mapped], dim=-1)

        volume = sheet_volume(96, [10, 20, 30, 40])
        cfg = phase_cfg(dense_spacing_max_steps=200)
        k = torch.tensor([2.0])
        pair_m = torch.tensor([1], dtype=torch.long)
        theta = torch.tensor([0.0])
        z = torch.tensor([1.5])
        transform = StretchedPad()
        dr = torch.tensor(DR_PER_WINDING)
        central = sample_pair_polylines(
            transform, dr, k, k + 1.0, theta, z, cfg)
        field, valid, _ = sample_sdt_trilinear(volume, central['scroll_poly'])
        pair = _pair_counts_from_samples(central, field, valid, volume, cfg)
        assert bool(pair['seg_valid'][0])
        pads, pad_rejected, _ = _build_phase_padding(
            transform, dr, volume, k, pair_m, theta, z, 1, 5, cfg)
        assert bool(pad_rejected[0])

    def test_count_survives_when_phase_finds_no_bands(self):
        # No complete bands (one huge sheet): phase scores nothing, but the
        # crossing count stays live on the shared central samples.
        torch.manual_seed(2)
        volume = sheet_volume(96, [25], half_thickness=20.0)
        components = run_bundle(
            volume, normal_volume(volume['shape']), PerfectSpiralToX(), 4,
            phase_cfg(dense_spacing_pair_m_short=(1, 1),
                      dense_spacing_pair_m_long=(1, 1)))
        phase_loss, phase_metrics = components['dense_spacing_phase']
        count_loss, count_metrics = components['dense_spacing_count']
        assert phase_metrics['dense_spacing_phase_valid_fraction'] == 0.0
        assert count_metrics['dense_spacing_count_valid_fraction'] > 0.99
        assert math.isfinite(float(count_loss))


class TestBundleComposition:
    def test_all_four_components_execute_with_nonzero_weights(self):
        torch.manual_seed(5)
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        normals = normal_volume(volume['shape'])
        params = GapExpanderParams(
            resolution=24, min_z=0.0, max_z=96.0, num_windings=10,
            dr_per_winding=DR_PER_WINDING)
        transform = GapExpandingTransform(
            params, torch.tensor(DR_PER_WINDING), 0.0, 96.0,
            gap_expander_lr_scale=0.3)

        class Model:
            device = torch.device('cpu')

            def get_native_log_gaps(self, winding, theta, z):
                return transform.get_native_log_gaps(winding, theta, z)

        cfg = phase_cfg(loss_weight_min_spacing=2.0,
                        loss_weight_dense_attachment=1.0)
        names = [
            name for name, _loss, _metrics in iter_phase_bundle_losses(
                Model(), PerfectSpiralToX(), torch.tensor(DR_PER_WINDING),
                volume, normals, 8, cfg, 1, 2)
        ]
        assert names == ['dense_spacing_count', 'dense_spacing_phase',
                         'min_spacing', 'dense_attachment']

    def test_zero_sub_weights_disable_components_without_new_modes(self):
        torch.manual_seed(5)
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        normals = normal_volume(volume['shape'])
        cfg = phase_cfg(loss_weight_dense_spacing=0.0)
        components = run_bundle(volume, normals, PerfectSpiralToX(), 8, cfg)
        assert set(components) == {'dense_spacing_count'}

    def test_bundle_requires_a_signed_distance_store(self):
        surf = make_volume(np.full([4, 4, 8], 200, np.uint8), kind='surf',
                           unit=None, cap=None)
        with pytest.raises(ValueError):
            list(iter_phase_bundle_losses(
                None, PerfectSpiralToX(), torch.tensor(DR_PER_WINDING), surf,
                normal_volume(surf['shape']), 6, phase_cfg(), 1, 2))


class TestModeContract:
    def base_request(self, tmp_path, config):
        umbilicus = tmp_path / 'umbilicus.json'
        umbilicus.write_text('{}')
        output = tmp_path / 'output'
        cache = tmp_path / 'cache'
        output.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        paths = SpiralInputPaths(
            umbilicus=str(umbilicus), output_directory=str(output),
            cache_directory=str(cache))
        base = {
            'disable_patches': True,
            'loss_weight_shell_outer': 0.0,
            'loss_weight_shell_patch_radius': 0.0,
            'loss_weight_dense_normals': 0.0,
        }
        base.update(config)
        return paths, SpiralRunConfig(z_begin=1, z_end=2, config=base)

    def test_exactly_the_two_modes_are_accepted(self, tmp_path):
        # 'phase' and 'grad_mag' pass mode validation; anything else -
        # including retired values like the old 'crossing_count' - is a
        # plain error (no migration handling).
        for mode in ('phase', 'grad_mag'):
            paths, run = self.base_request(tmp_path, {
                'dense_spacing_mode': mode,
                'loss_weight_dense_spacing': 0.0,
            })
            fields = {error['field']
                      for error in validate_session_request(paths, run)}
            assert 'dense_spacing_mode' not in fields
        for mode in ('crossing_count', 'anything_else'):
            paths, run = self.base_request(tmp_path, {
                'dense_spacing_mode': mode,
            })
            fields = {error['field']
                      for error in validate_session_request(paths, run)}
            assert 'dense_spacing_mode' in fields

    def test_missing_mode_defaults_to_phase_and_requires_bundle_assets(
        self, tmp_path,
    ):
        # No dense_spacing_mode in the request: the default is 'phase', so
        # the bundle's core assets are required even with zero sub-weights.
        paths, run = self.base_request(tmp_path, {})
        assert run.config.get('dense_spacing_mode') is None
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert {'normal_x', 'normal_y', 'surf_sdt'} <= fields

    def test_phase_mode_requires_normals_and_sdt(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'loss_weight_dense_spacing': 0.0,
            'loss_weight_dense_attachment': 0.0,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert {'normal_x', 'normal_y', 'surf_sdt'} <= fields

    def test_disabled_phase_inputs_are_not_required(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'use_normals': False,
            'use_surf_sdt': False,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert 'normal_x' not in fields
        assert 'normal_y' not in fields
        assert 'surf_sdt' not in fields

    def test_grad_mag_mode_requires_grad_mag_not_sdt(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'grad_mag',
            'loss_weight_dense_spacing': 12.0,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert 'gradient_magnitude' in fields
        assert 'surf_sdt' not in fields
        assert 'normal_x' not in fields

    def test_disabled_grad_mag_is_not_required(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'grad_mag',
            'loss_weight_dense_spacing': 12.0,
            'use_gradient_magnitude': False,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert 'gradient_magnitude' not in fields

    def test_invalid_mode_is_rejected_before_asset_errors(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'crossing_count',
        })
        errors = validate_session_request(paths, run)
        by_field = {error['field']: error['message'] for error in errors}
        assert 'dense_spacing_mode' in by_field
        # The invalid-mode error must appear instead of misleading
        # mode-derived asset errors.
        assert 'surf_sdt' not in by_field
        assert 'gradient_magnitude' not in by_field


class TestOptionalInputSelection:
    def test_disabled_inputs_zero_their_weights_and_sampling(self):
        config = {
            'dense_spacing_mode': 'phase',
            'use_verified_patches': False,
            'use_unverified_patches': False,
            'use_normals': False,
            'use_surf_sdt': False,
            'use_tracks': False,
            'use_fibers': False,
            'loss_weight_patch_radius': 8.0,
            'loss_weight_patch_dt': 4.0,
            'num_patches_per_step': 360,
            'num_patches_per_step_for_dt': 240,
            'num_points_per_patch': 800,
            'loss_weight_unverified_patch_radius': 2.0,
            'loss_weight_unverified_patch_dt': 1.0,
            'unverified_num_patches_per_step': 120,
            'unverified_num_patches_per_step_for_dt': 80,
            'unverified_num_points_per_patch': 800,
            'loss_weight_dense_normals': 100.0,
            'dense_normals_num_points': 60_000,
            'loss_weight_dense_spacing': 12.0,
            'loss_weight_dense_spacing_count': 2.0,
            'loss_weight_dense_spacing_density': 3.0,
            'loss_weight_min_spacing': 4.0,
            'loss_weight_dense_attachment': 5.0,
            'dense_spacing_num_pairs': 12_000,
            'dense_spacing_density_extra_pairs': 24_000,
            'dense_attachment_num_points': 20_000,
            'min_spacing_independent_samples': 2_000,
            'loss_weight_track_radius': 50.0,
            'loss_weight_track_dt': 10.0,
            'track_num_per_step': 48_000,
            'track_num_points_per_step': 24,
            'loss_weight_unattached_pcl_radius': 2.0,
            'loss_weight_unattached_pcl_dt': 4.0,
            'unattached_pcl_num_per_step': 84,
            'unattached_pcl_num_points_per_step': 32,
        }
        apply_optional_input_selection(config)
        expected_zero = {
            'loss_weight_patch_radius', 'loss_weight_patch_dt',
            'num_patches_per_step', 'num_patches_per_step_for_dt',
            'num_points_per_patch', 'loss_weight_unverified_patch_radius',
            'loss_weight_unverified_patch_dt', 'unverified_num_patches_per_step',
            'unverified_num_patches_per_step_for_dt',
            'unverified_num_points_per_patch', 'loss_weight_dense_normals',
            'dense_normals_num_points', 'loss_weight_dense_spacing',
            'loss_weight_dense_spacing_count',
            'loss_weight_dense_spacing_density', 'loss_weight_min_spacing',
            'loss_weight_dense_attachment', 'dense_spacing_num_pairs',
            'dense_spacing_density_extra_pairs', 'dense_attachment_num_points',
            'min_spacing_independent_samples', 'loss_weight_track_radius',
            'loss_weight_track_dt', 'track_num_per_step',
            'track_num_points_per_step', 'loss_weight_unattached_pcl_radius',
            'loss_weight_unattached_pcl_dt', 'unattached_pcl_num_per_step',
            'unattached_pcl_num_points_per_step',
        }
        assert all(config[key] == 0 for key in expected_zero)

    def test_disabled_grad_mag_zeroes_legacy_spacing(self):
        config = {
            'dense_spacing_mode': 'grad_mag',
            'use_gradient_magnitude': False,
            'loss_weight_dense_spacing': 12.0,
            'dense_spacing_num_pairs': 12_000,
        }
        apply_optional_input_selection(config)
        assert config['loss_weight_dense_spacing'] == 0
        assert config['dense_spacing_num_pairs'] == 0


class TestNativeMinimumGap:
    def make_transform(self, dr=None):
        dr = torch.tensor(10.0, requires_grad=True) if dr is None else dr
        params = GapExpanderParams(
            resolution=24, min_z=0.0, max_z=96.0, num_windings=8,
            dr_per_winding=10.0)
        transform = GapExpandingTransform(
            params, dr, 0.0, 96.0, gap_expander_lr_scale=0.3)
        return params, transform, dr

    def test_log_gap_matches_forward_distance_and_detaches_global_dr(self):
        params, transform, dr = self.make_transform()
        with torch.no_grad():
            params.logits.normal_(mean=0.0, std=0.002)
        theta = torch.tensor([0.4, 1.7])
        z = torch.tensor([25.0, 70.0])
        winding = torch.tensor([2, 4])
        ell = transform.get_native_log_gaps(winding, theta, z)
        radii = transform.get_transformed_winding_radii(theta, z)
        distance = torch.gather(
            radii[..., 1:] - radii[..., :-1], 1,
            winding[:, None]).squeeze(-1)
        torch.testing.assert_close(torch.exp(ell), distance)
        ell.sum().backward()
        assert dr.grad is None or float(dr.grad.abs()) == 0.0
        assert float(params.logits.grad.abs().sum()) > 0.0

    def test_underflowed_distance_keeps_nonzero_barrier_gradient(self):
        params, transform, _ = self.make_transform(torch.tensor(10.0))
        with torch.no_grad():
            params.logits.fill_(-100.0)

        class Wrapper:
            device = torch.device('cpu')

            def get_native_log_gaps(self, winding, theta, z):
                return transform.get_native_log_gaps(winding, theta, z)

        cfg = {
            'min_spacing_independent_samples': 64,
            'min_spacing_d_min_wv': 6.0,
        }
        loss, metrics = get_min_spacing_loss(
            Wrapper(), 7, cfg, 1, 95,
            generator=torch.Generator().manual_seed(8))
        assert metrics['min_spacing_active_fraction'] > 0.9
        loss.backward()
        assert torch.isfinite(params.logits.grad).all()
        assert float(params.logits.grad.abs().sum()) > 0.0


@pytest.mark.skipif(
    not os.environ.get('SPIRAL_PHASE_INTEGRATION'),
    reason='machine-local production probe; set SPIRAL_PHASE_INTEGRATION=1')
class TestProductionNormalConvention:
    def test_positive_nz_hemisphere_matches_sdt_gradient_lines(self):
        """Probe the shipped store's convention on non-axis-aligned normals.

        ``abs(dot)`` is evaluated for both +nz and an nz-only sign flip. The
        positive hemisphere must align better with the independently generated
        SDT gradient lines; otherwise phase incidence classification is unsafe.
        """
        import zarr

        base = os.environ.get(
            'SPIRAL_PHASE_INPUT_DIR',
            '/home/sean/Desktop/spiral_dataset/to_hf/lasagna_inputs')
        nx_root = zarr.open(f'{base}/las_008_nx.ome.zarr', mode='r')
        ny_root = zarr.open(f'{base}/las_008_ny.ome.zarr', mode='r')
        sdt_root = zarr.open(f'{base}/las_008_surf_sdt.ome.zarr', mode='r')
        nx, ny, sdt = nx_root['4'], ny_root['4'], sdt_root['1']
        assert nx.dtype == ny.dtype == np.dtype('uint8')
        assert nx.shape == ny.shape
        # Normal group 4 is 16 source voxels = 4 working voxels after the
        # lasagna scale conversion. SDT group 1 is 2 working voxels, hence the
        # 2:1 index mapping below.
        assert nx_root.attrs['multiscales'][0]['datasets'][0][
            'coordinateTransformations'][0]['scale'] == [16.0, 16.0, 16.0]
        assert sdt_root.attrs['multiscales'][0]['datasets'][0][
            'coordinateTransformations'][0]['scale'] == [2.0, 2.0, 2.0]
        assert sdt_root.attrs['nodata_value'] == 0
        assert sdt_root.attrs['sign'] == 'positive_outside'

        plus_scores, flipped_scores = [], []
        for zn, y0, x0 in ((2000, 769, 961), (2200, 1249, 1201)):
            size = 96
            nx_u8 = np.asarray(nx[zn, y0:y0 + size, x0:x0 + size])
            ny_u8 = np.asarray(ny[zn, y0:y0 + size, x0:x0 + size])
            zs, ys, xs = zn * 2, y0 * 2, x0 * 2
            slab = np.asarray(
                sdt[zs - 1:zs + 2,
                    ys - 1:(y0 + size) * 2 + 1,
                    xs - 1:(x0 + size) * 2 + 1], dtype=np.float32) - 128
            center = slab[1]
            gz = (slab[2] - slab[0]) * 0.5
            gy = (center[2:, 1:-1] - center[:-2, 1:-1]) * 0.5
            gx = (center[1:-1, 2:] - center[1:-1, :-2]) * 0.5
            gradient = np.stack([
                gz[1:-1:2, 1:-1:2], gy[::2, ::2], gx[::2, ::2]], axis=-1)
            gradient_norm = np.linalg.norm(gradient, axis=-1)
            gradient /= gradient_norm[..., None] + 1e-9
            nx_f = (nx_u8.astype(np.float32) - 128.0) / 127.0
            ny_f = (ny_u8.astype(np.float32) - 128.0) / 127.0
            nz_f = np.sqrt(np.maximum(0.0, 1.0 - nx_f ** 2 - ny_f ** 2))
            valid = (
                ((nx_u8 != 0) | (ny_u8 != 0)) & (gradient_norm > 0.2)
                & (np.abs(center[1:-1:2, 1:-1:2]) < 20))
            plus = np.abs(
                gradient[..., 0] * nz_f + gradient[..., 1] * ny_f
                + gradient[..., 2] * nx_f)
            flipped = np.abs(
                -gradient[..., 0] * nz_f + gradient[..., 1] * ny_f
                + gradient[..., 2] * nx_f)
            plus_scores.extend(plus[valid].tolist())
            flipped_scores.extend(flipped[valid].tolist())
        assert len(plus_scores) >= 100
        nodata_nx = np.asarray(nx[2000, :96, :96])
        nodata_ny = np.asarray(ny[2000, :96, :96])
        assert ((nodata_nx == 0) & (nodata_ny == 0)).any()
        assert np.mean(plus_scores) > np.mean(flipped_scores) + 0.05
