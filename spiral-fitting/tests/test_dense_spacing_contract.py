"""The dense-spacing mode contract (which stores each mode requires) and the
asset-independent native minimum-spacing barrier."""

from dataclasses import replace
import json

import torch

from fit_session import (
    SpiralInputPaths, SpiralRunConfig, validate_session_request,
)
from losses import get_min_spacing_loss
from transforms import GapExpanderParams, GapExpandingTransform


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
            'input_disable_patches': True,
            'loss_weight_shell_outer': 0.0,
            'loss_weight_dense_normals': 0.0,
        }
        base.update(config)
        return paths, SpiralRunConfig(z_begin=1, z_end=2, config=base)

    def test_exactly_the_supported_modes_are_accepted(self, tmp_path):
        # The two current modes pass mode validation; anything else -
        # including retired values like 'phase' and 'crossing_count' - is a
        # plain error (no migration handling).
        for mode in ('grad_mag', 'winding_model'):
            paths, run = self.base_request(tmp_path, {
                'dense_spacing_mode': mode,
                'loss_weight_dense_spacing': 0.0,
            })
            fields = {error['field']
                      for error in validate_session_request(paths, run)}
            assert 'dense_spacing_mode' not in fields
        for mode in ('phase', 'crossing_count', 'anything_else'):
            paths, run = self.base_request(tmp_path, {
                'dense_spacing_mode': mode,
            })
            fields = {error['field']
                      for error in validate_session_request(paths, run)}
            assert 'dense_spacing_mode' in fields

    def test_winding_model_mode_requires_a_valid_crossing_manifest(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'winding_model',
        })
        fields = {error['field']
                  for error in validate_session_request(paths, run)}
        assert {'winding_inference', 'outer_shell'} <= fields

        store = tmp_path / 'winding_inference'
        store.mkdir()
        shell = tmp_path / 'outer_shell'
        shell.mkdir()
        paths = replace(
            paths, winding_inference=str(store), outer_shell=str(shell))
        messages = {error['field']: error['message']
                    for error in validate_session_request(paths, run)}
        assert messages['winding_inference'] == 'manifest.json is missing'

        (store / 'manifest.json').write_text(json.dumps({
            'artifact_type': 'winding_inference_crossings',
            'format_version': 1,
        }))
        fields = {error['field']
                  for error in validate_session_request(paths, run)}
        assert 'winding_inference' not in fields

    def test_missing_mode_defaults_to_winding_model_and_requires_its_assets(
        self, tmp_path,
    ):
        # Omitted modes use the fitter's winding-model default in preflight.
        paths, run = self.base_request(tmp_path, {})
        assert run.config.get('dense_spacing_mode') is None
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert {'winding_inference', 'outer_shell'} <= fields

    def test_grad_mag_mode_requires_grad_mag_only(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'grad_mag',
            'loss_weight_dense_spacing': 12.0,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert 'gradient_magnitude' in fields
        assert 'normal_x' not in fields

    def test_zero_weight_grad_mag_is_not_required(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'grad_mag',
            'loss_weight_dense_spacing': 0.0,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert 'gradient_magnitude' not in fields

    def test_disabled_grad_mag_source_is_not_required(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'grad_mag',
            'loss_weight_dense_spacing': 12.0,
            'input_use_gradient_magnitude': False,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert 'gradient_magnitude' not in fields

    def test_disabled_outer_shell_cascades_winding_inference_off(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'winding_model',
            'input_use_outer_shell': False,
        })
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert 'outer_shell' not in fields
        assert 'winding_inference' not in fields

    def test_invalid_mode_is_rejected_before_asset_errors(self, tmp_path):
        paths, run = self.base_request(tmp_path, {
            'dense_spacing_mode': 'crossing_count',
        })
        errors = validate_session_request(paths, run)
        by_field = {error['field']: error['message'] for error in errors}
        assert 'dense_spacing_mode' in by_field
        # The invalid-mode error must appear instead of misleading
        # mode-derived asset errors.
        assert 'gradient_magnitude' not in by_field
        assert 'winding_inference' not in by_field


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

    def _wrapper(self, transform):
        class Wrapper:
            device = torch.device('cpu')

            def get_native_log_gaps(self, winding, theta, z):
                return transform.get_native_log_gaps(winding, theta, z)

        return Wrapper()

    def test_near_floor_distance_keeps_nonzero_barrier_gradient(self):
        params, transform, _ = self.make_transform(torch.tensor(10.0))
        with torch.no_grad():
            # Close to the numerical floor but not in softplus's asymptotic
            # tail: the geological 6-voxel preference can still push it up.
            params.logits.fill_(-0.04)
        cfg = {
            'sample_count_minimum_spacing_independent_samples': 64,
            'dense_min_spacing_d_min_wv': 6.0,
        }
        loss, metrics = get_min_spacing_loss(
            self._wrapper(transform), 7, cfg, 1, 95,
            generator=torch.Generator().manual_seed(8))
        assert metrics['min_spacing_active_fraction'] > 0.9
        loss.backward()
        assert torch.isfinite(params.logits.grad).all()
        assert float(params.logits.grad.abs().sum()) > 0.0

    def test_metrics_can_be_skipped_without_changing_the_loss(self):
        params, transform, _ = self.make_transform(torch.tensor(10.0))
        with torch.no_grad():
            params.logits.fill_(-0.04)
        cfg = {
            'sample_count_minimum_spacing_independent_samples': 64,
            'dense_min_spacing_d_min_wv': 6.0,
        }
        with_metrics, metrics = get_min_spacing_loss(
            self._wrapper(transform), 7, cfg, 1, 95,
            generator=torch.Generator().manual_seed(8))
        without, none = get_min_spacing_loss(
            self._wrapper(transform), 7, cfg, 1, 95,
            generator=torch.Generator().manual_seed(8), with_metrics=False)
        assert metrics and none == {}
        torch.testing.assert_close(with_metrics, without)

    def test_unresolved_outer_winding_or_zero_budget_is_a_defined_zero(self):
        _, transform, _ = self.make_transform(torch.tensor(10.0))
        cfg = {
            'sample_count_minimum_spacing_independent_samples': 0,
            'dense_min_spacing_d_min_wv': 6.0,
        }
        loss, metrics = get_min_spacing_loss(
            self._wrapper(transform), None, cfg, 1, 95)
        assert float(loss) == 0.0 and metrics == {}
        loss, metrics = get_min_spacing_loss(
            self._wrapper(transform), 7, cfg, 1, 95)
        assert float(loss) == 0.0 and metrics == {}
