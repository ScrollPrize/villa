import numpy as np
import torch
import pytest
import os
from dataclasses import replace

from vesuvius.neural_tracing.fiber_trace_3d.cross_section import (
    CrossSectionAccumulator,
    CrossSectionConfig,
    merge_shards,
    write_cross_section_artifacts,
)
from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DBatch


def _batch() -> FiberTrace3DBatch:
    return FiberTrace3DBatch(
        volume=torch.zeros((1, 1, 128, 128, 128)),
        valid_mask=None,
        cp_local_zyx=torch.tensor([[64.0, 64.0, 64.0]]),
        crop_origin_zyx=torch.zeros((1, 3)),
        stream_indices=torch.zeros(1, dtype=torch.long),
        data_indices=torch.zeros(1, dtype=torch.long),
        record_indices=torch.zeros(1, dtype=torch.long),
        control_point_indices=torch.zeros(1, dtype=torch.long),
        fiber_paths=("fiber.json",),
        target_modes=torch.ones(1, dtype=torch.long),
        target_segment_offsets=torch.zeros(1, dtype=torch.long),
        target_segment_counts=torch.ones(1, dtype=torch.long),
        target_segment_starts_zyx=torch.tensor([[48.0, 64.0, 64.0]]),
        target_segment_ends_zyx=torch.tensor([[80.0, 64.0, 64.0]]),
        target_segment_bbox_lo_zyx=torch.zeros((1, 3), dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.full((1, 3), 128, dtype=torch.long),
        target_tangent_zyx=torch.tensor([[1.0, 0.0, 0.0]]),
        target_segment_normals_zyx=torch.tensor([[0.0, 1.0, 0.0]]),
    )


def test_cross_section_samples_normal_and_in_sheet_axes() -> None:
    z, y, x = torch.meshgrid(*(torch.arange(32, dtype=torch.float32) for _ in range(3)), indexing="ij")
    presence = torch.exp(-0.5 * (((y - 16.0) / 1.0) ** 2 + ((x - 16.0) / 3.0) ** 2))
    output = torch.zeros((1, 7, 32, 32, 32))
    output[0, 6] = presence
    accumulator = CrossSectionAccumulator(CrossSectionConfig(histogram_bins=32))
    accumulator(_batch(), output)

    mean = accumulator.mean()
    assert mean.shape == (81, 161)
    assert accumulator.sections == 1
    assert mean[40, 80] > 0.99
    # Tangent=Z, normal=Y and width=cross(Z,Y)=-X. The wider synthetic X
    # profile must therefore be wider along image columns than rows.
    assert mean[40, 88] > mean[48, 80]
    assert int(accumulator.count[40, 80]) == 1


def test_cross_section_histogram_merge_quantiles_and_float_tiffs(tmp_path) -> None:
    config = CrossSectionConfig(width_base=2, height_base=2, spacing_base=1.0, histogram_bins=10)
    left = CrossSectionAccumulator(config)
    right = CrossSectionAccumulator(config)
    left.histogram[..., 1] = 1
    left.histogram[..., 5] = 1
    left.count[...] = 2
    left.sum[...] = 0.6
    left.sections = 2
    right.histogram[..., 9] = 2
    right.count[...] = 2
    right.sum[...] = 1.8
    right.sections = 2
    left_path, right_path = tmp_path / "left.npz", tmp_path / "right.npz"
    left.save_npz(left_path)
    right.save_npz(right_path)

    merged = merge_shards([left_path, right_path])
    assert merged.sections == 4
    assert np.allclose(merged.mean(), 0.6)
    assert np.allclose(merged.quantile(0.10), 0.15)
    assert np.allclose(merged.quantile(0.50), 0.55)
    assert np.allclose(merged.quantile(0.90), 0.95)
    assert np.allclose(merged.exceedance_threshold(0.10), 0.95)
    assert np.allclose(merged.exceedance_threshold(0.50), 0.55)
    paths = write_cross_section_artifacts(merged, tmp_path / "artifacts")
    import tifffile
    for key in ("mean", "mean_gaussian_fit", "q0.001", "q0.01", "q0.1", "q0.2", "q0.5"):
        image = tifffile.imread(paths[key])
        assert image.dtype == np.float32
        assert image.shape == config.shape


def test_fit_mean_gaussian_recovers_spatial_parameters() -> None:
    from vesuvius.neural_tracing.fiber_trace_3d.cross_section import fit_mean_gaussian

    config = CrossSectionConfig(width_base=60, height_base=40)
    yy, xx = np.meshgrid(config.normal_offsets_base, config.width_offsets_base, indexing="ij")
    mean = 0.04 + 0.8 * np.exp(-0.5 * (((yy - 1.5) / 3.0) ** 2 + ((xx + 2.0) / 7.0) ** 2))
    fitted, params = fit_mean_gaussian(mean.astype(np.float32), config)
    assert fitted.dtype == np.float32
    assert np.isclose(params["background"], 0.04, atol=1.0e-4)
    assert np.isclose(params["amplitude"], 0.8, atol=1.0e-4)
    assert np.isclose(params["center_normal_base"], 1.5, atol=1.0e-4)
    assert np.isclose(params["center_width_base"], -2.0, atol=1.0e-4)
    assert np.isclose(params["sigma_normal_base"], 3.0, atol=1.0e-4)
    assert np.isclose(params["sigma_width_base"], 7.0, atol=1.0e-4)


def test_cross_section_handoff_preserves_paired_arrays_and_metadata():
    import pickle
    from vesuvius.neural_tracing.fiber_trace_3d.training_state import cpu_clone, replica_signature
    config = CrossSectionConfig(width_base=6, height_base=4, histogram_bins=16)
    total = CrossSectionAccumulator(config, width_filtered=True)
    expected = CrossSectionAccumulator(config, width_filtered=True)
    for wave in range(3):
        delta = CrossSectionAccumulator(config, width_filtered=True)
        delta.filter_metadata = {'reference': 'refined_target_segment_width_at_section_center'}
        for section in (delta, delta.width_filtered):
            section.sections = wave + 1
            section.skipped = wave
            section.count[:] = wave + 1
            section.sum[:] = (wave + 1) * .25
            section.histogram[..., 4] = wave + 1
        total.merge(delta)
        expected.merge(delta)
        state = cpu_clone(total.state_dict())
        signature = replica_signature({'sections': state})
        total = CrossSectionAccumulator.from_state_dict(pickle.loads(pickle.dumps(state)))
        assert replica_signature({'sections': total.state_dict()}) == signature
        delta.clear()
        assert delta.sections == delta.width_filtered.sections == 0
        assert not delta.histogram.any()
    for actual, reference in ((total, expected), (total.width_filtered, expected.width_filtered)):
        for name in ('sum', 'count', 'histogram'):
            np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))
        assert actual.sections == reference.sections == 6
        assert actual.skipped == reference.skipped == 3
    assert total.filter_metadata == expected.filter_metadata


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_paired_cross_section_raw_unchanged_center_width_and_persistence(tmp_path, device):
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import AnnotationConfig
    if device == 'cuda' and not os.environ.get('FIBER_COOPERATIVE_TEST_GPUS'):
        pytest.skip('GPU requires manager grant')
    config = CrossSectionConfig(width_base=8, height_base=8, histogram_bins=128)
    batch = replace(_batch(), target_segment_widths_input=torch.tensor([[8., 12.]])).to(device)
    output = torch.zeros(1, 7, 32, 32, 32, device=device)
    output[:, 6] = .8
    annotations = AnnotationConfig(width_weight=.01, sideband_width_fraction=.4, sideband_width_gate='cosine')
    width_output = torch.full((1,1,32,32,32), float(np.log(np.expm1(12.))), device=device)
    paired = CrossSectionAccumulator(config, width_filtered=True)
    raw = CrossSectionAccumulator(config)
    raw(batch, output)
    paired.observe_annotations(batch, output, width_output, annotations)
    np.testing.assert_array_equal(raw.sum, paired.sum)
    np.testing.assert_array_equal(raw.histogram, paired.histogram)
    np.testing.assert_allclose(paired.width_filtered.mean(), .4, atol=2e-6)
    np.testing.assert_array_equal(paired.count, paired.width_filtered.count)
    width_output.fill_(float(np.log(np.expm1(14.))))
    zero = CrossSectionAccumulator(config, width_filtered=True)
    zero.observe_annotations(batch, output, width_output, annotations)
    np.testing.assert_allclose(zero.width_filtered.mean(), 0, atol=1e-7)
    np.testing.assert_array_equal(zero.count, zero.width_filtered.count)
    path = tmp_path / 'paired.npz'
    paired.save_npz(path)
    loaded = CrossSectionAccumulator.load_npz(path)
    loaded.merge(zero)
    np.testing.assert_allclose(loaded.width_filtered.mean(), .2, atol=2e-6)
    assert loaded.width_filtered.sections == 2 and loaded.filter_metadata == paired.filter_metadata
    for name, section in [('raw', loaded), ('width_filtered', loaded.width_filtered)]:
        paths = write_cross_section_artifacts(section, tmp_path/name)
        assert os.path.isfile(paths['mean']) and os.path.isfile(paths['q0.1'])
    invalid = CrossSectionAccumulator(config, width_filtered=True)
    invalid.observe_annotations(replace(batch, target_segment_widths_input=torch.zeros(1,2,device=device)),
                                output, width_output, annotations)
    assert invalid.sections == 1 and invalid.width_filtered.sections == 0
    assert invalid.width_filtered.skipped == 1
