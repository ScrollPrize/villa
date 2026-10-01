import numpy as np
import torch

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
