"""Synthetic tests for complete-band phase spacing and native anti-collapse."""

import math
import os

import numpy as np
import pytest
import torch

from sdt_losses import (
    detect_complete_sdt_bands,
    get_min_spacing_loss,
    get_phase_spacing_loss,
    sample_lasagna_normals_nearest,
    sample_pair_polylines,
)
from transforms import GapExpanderParams, GapExpandingTransform
from fit_session import (
    SpiralInputPaths, SpiralRunConfig, validate_session_request,
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


def phase_cfg(**overrides):
    values = {
        'dense_spacing_num_pairs': 192,
        'dense_spacing_pair_m_short': (2, 3),
        'dense_spacing_pair_m_long': (2, 3),
        'dense_spacing_max_steps': 160,
    }
    values.update(overrides)
    return spacing_cfg(**values)


class TestCompleteBandDetection:
    def test_complete_bands_ignore_partial_ray_end_intervals(self):
        volume = sheet_volume(80, [10, 20, 30, 40, 50, 60])
        cfg = phase_cfg()
        bands = detect_complete_sdt_bands(
            fixed_ray(PerfectSpiralToX(), cfg, 1.0, 6.0), volume,
            normal_volume(volume['shape']), cfg)
        # The ray begins/ends at the centers of sheets 1 and 6. Those partial
        # intervals are ignored; only sheets 2..5 are complete.
        torch.testing.assert_close(
            bands['phase'], torch.tensor([2.0, 3.0, 4.0, 5.0]), atol=1e-5,
            rtol=0)
        torch.testing.assert_close(bands['width'], torch.full([4], 4.0))

    def test_missing_band_leaves_a_two_winding_censor_gap(self):
        volume = sheet_volume(80, [10, 20, 30, 50, 60])
        cfg = phase_cfg()
        bands = detect_complete_sdt_bands(
            fixed_ray(PerfectSpiralToX(), cfg), volume,
            normal_volume(volume['shape']), cfg)
        torch.testing.assert_close(
            bands['phase'], torch.tensor([2.0, 3.0, 5.0]), atol=1e-5,
            rtol=0)
        assert float(bands['phase'][2] - bands['phase'][1]) >= 2.0

    def test_physically_huge_single_modeled_gap_is_not_a_censor(self):
        class HugeSingleGap:
            def inv(self, points):
                y, x = points[:, 1], points[:, 2]
                radius = torch.sqrt(y * y + x * x + 1e-12)
                theta = torch.atan2(y, x) % (2 * torch.pi)
                phase = (radius - theta / (2 * torch.pi) * DR_PER_WINDING) / DR_PER_WINDING
                mapped_x = torch.where(
                    phase <= 2.0, phase * 10.0,
                    torch.where(phase < 3.0, 20.0 + (phase - 2.0) * 30.0,
                                50.0 + (phase - 3.0) * 10.0))
                return torch.stack([
                    torch.full_like(mapped_x, 1.5),
                    torch.full_like(mapped_x, 1.5), mapped_x], dim=-1)

        volume = sheet_volume(96, [20, 50, 60])
        cfg = phase_cfg(dense_spacing_max_steps=320)
        bands = detect_complete_sdt_bands(
            fixed_ray(HugeSingleGap(), cfg, 1.0, 4.0), volume,
            normal_volume(volume['shape']), cfg)
        assert float(bands['center_arc'][1] - bands['center_arc'][0]) > 25.0
        assert float(bands['phase'][1] - bands['phase'][0]) < 2.0

    def test_close_complete_fragments_merge_using_projected_spacing(self):
        volume = fractional_sheet_volume(
            96, [40.0, 45.0, 60.0], half_thickness=0.75)
        cfg = phase_cfg(
            dense_spacing_target_step_wv=0.25,
            dense_spacing_max_step_wv=0.5,
            dense_spacing_max_steps=400,
        )
        bands = detect_complete_sdt_bands(
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
        shallow_bands = detect_complete_sdt_bands(
            fixed_ray(PerfectSpiralToX(), cfg, 1.0, 4.0), shallow,
            normals, cfg)
        deep_bands = detect_complete_sdt_bands(
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
            bands = detect_complete_sdt_bands(
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
        bands = detect_complete_sdt_bands(
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
        loss, metrics = get_phase_spacing_loss(
            PerfectSpiralToX(), torch.tensor(DR_PER_WINDING), volume,
            normals, 8, cfg, 1, 2)
        assert float(loss) < 1e-8
        assert metrics['dense_spacing_phase_span_minus_m_mean'] == 0.0

        offset = torch.tensor(2.0, requires_grad=True)
        torch.manual_seed(3)
        loss, metrics = get_phase_spacing_loss(
            PerfectSpiralToX(x_offset=offset),
            torch.tensor(DR_PER_WINDING), volume, normals, 8, cfg, 1, 2)
        assert metrics['dense_spacing_phase_valid_fraction'] > 0.75
        loss.backward()
        # Gradient descent decreases the positive x offset toward band centers.
        assert float(offset.grad) > 0

    def test_half_winding_anchor_is_rejected_as_ambiguous(self):
        torch.manual_seed(4)
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        loss, metrics = get_phase_spacing_loss(
            PerfectSpiralToX(x_offset=5.0),
            torch.tensor(DR_PER_WINDING), volume,
            normal_volume(volume['shape']), 8, phase_cfg(), 1, 2)
        assert metrics['dense_spacing_phase_anchor_ambiguity_fraction'] > 0.75
        assert metrics['dense_spacing_phase_valid_fraction'] < 0.1
        assert math.isfinite(float(loss))

    def test_missing_band_censors_span_but_keeps_scoreable_prefixes(self):
        torch.manual_seed(7)
        # Removing sheet 40 leaves a two-winding phase gap between sheets 3/5.
        volume = sheet_volume(100, [10, 20, 30, 50, 60, 70, 80])
        _, metrics = get_phase_spacing_loss(
            PerfectSpiralToX(), torch.tensor(DR_PER_WINDING), volume,
            normal_volume(volume['shape']), 8,
            phase_cfg(dense_spacing_pair_m_short=(3, 4),
                      dense_spacing_pair_m_long=(3, 4)), 1, 2)
        assert metrics['dense_spacing_phase_censor_wide_gap_fraction'] > 0
        assert metrics['dense_spacing_phase_valid_fraction'] > 0
        assert metrics['dense_spacing_phase_span_valid_fraction'] < 0.75

    def test_extra_inserted_band_censors_instead_of_biasing_residuals(self):
        torch.manual_seed(6)
        # A spurious extra sheet at x = 45 sits mid-gap between windings 4 and
        # 5, so detection sees two bands half a modeled winding apart. Unit
        # enumeration would hand winding 5 to the extra band and shift every
        # later target a full winding outward - a systematic positive-rho
        # bias. The narrow-gap censor must stop enumeration at the close pair
        # instead, keeping all accepted residuals near zero.
        volume = sheet_volume(100, [10, 20, 30, 40, 45, 50, 60, 70, 80])
        loss, metrics = get_phase_spacing_loss(
            PerfectSpiralToX(), torch.tensor(DR_PER_WINDING), volume,
            normal_volume(volume['shape']), 8, phase_cfg(), 1, 2)
        assert metrics['dense_spacing_phase_censor_narrow_gap_fraction'] > 0
        # Rays anchored right at the insertion correctly go quiet; the rest of
        # the batch must keep scoring.
        assert metrics['dense_spacing_phase_valid_fraction'] > 0.25
        assert metrics['dense_spacing_phase_residual_abs_p95'] < 0.1
        # An un-censored insertion would put whole +1-winding residuals in the
        # mean (huber(1.0) ~ 0.375 per hit); only encoding-quantization noise
        # from the tight fixture may remain.
        assert float(loss) < 1e-3

    def test_anchor_only_prefix_has_zero_weight(self):
        torch.manual_seed(12)
        volume = sheet_volume(80, [30])
        loss, metrics = get_phase_spacing_loss(
            PerfectSpiralToX(), torch.tensor(DR_PER_WINDING), volume,
            normal_volume(volume['shape']), 6, phase_cfg(), 1, 2)
        assert float(loss) == 0.0
        assert metrics['dense_spacing_phase_valid_fraction'] == 0.0

    def test_over_budget_extended_rays_are_wholly_rejected(self):
        torch.manual_seed(13)
        volume = sheet_volume(200, [30, 60, 90, 120, 150])
        loss, metrics = get_phase_spacing_loss(
            PerfectSpiralToX(radial_scale=3.0),
            torch.tensor(DR_PER_WINDING), volume,
            normal_volume(volume['shape']), 6,
            phase_cfg(dense_spacing_max_steps=16), 1, 2)
        assert float(loss) == 0.0
        assert metrics['dense_spacing_phase_too_long_fraction'] == 1.0

    def test_shadow_generator_does_not_advance_global_rng(self):
        volume = sheet_volume(100, [10, 20, 30, 40, 50, 60, 70, 80])
        normals = normal_volume(volume['shape'])
        torch.manual_seed(99)
        expected = torch.rand(5)
        torch.manual_seed(99)
        generator = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            get_phase_spacing_loss(
                PerfectSpiralToX(), torch.tensor(DR_PER_WINDING), volume,
                normals, 8, phase_cfg(), 1, 2, generator=generator)
        torch.testing.assert_close(torch.rand(5), expected)


class TestPhaseSessionActivation:
    def test_phase_shadow_requires_normals_and_sdt_with_zero_loss_weights(
        self, tmp_path,
    ):
        umbilicus = tmp_path / 'umbilicus.json'
        umbilicus.write_text('{}')
        output = tmp_path / 'output'
        cache = tmp_path / 'cache'
        output.mkdir()
        cache.mkdir()
        paths = SpiralInputPaths(
            umbilicus=str(umbilicus), output_directory=str(output),
            cache_directory=str(cache))
        run = SpiralRunConfig(
            z_begin=1, z_end=2,
            config={
                'disable_patches': True,
                'loss_weight_shell_outer': 0.0,
                'loss_weight_shell_patch_radius': 0.0,
                'loss_weight_dense_normals': 0.0,
                'loss_weight_dense_spacing': 0.0,
                'loss_weight_dense_attachment': 0.0,
                'dense_spacing_mode': 'crossing_count',
                'dense_spacing_phase_shadow': True,
            })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert {'normal_x', 'normal_y', 'surf_sdt'} <= fields


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
