from __future__ import annotations

import numpy as np
import torch
import sys
from types import SimpleNamespace

from vesuvius.neural_tracing.winding_models import infer_winding_volume as infer
from vesuvius.neural_tracing.winding_models.volume_slab_extractor import SlabFrame


def test_phase_consensus_cli_options(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "infer_winding_volume.py", "fit.ckpt", "out.zarr",
        "--model-ckpt", "model.pth", "--reference-zarr", "reference.zarr",
        "--prob-volume", "--prob-combine", "phase",
        "--prob-phase-level-half-life", "3",
        "--prob-phase-max-level", "5",
        "--prob-phase-edge-taper", "6",
        "--prob-phase-agreement-power", "2",
        "--prob-phase-min-observations", "4",
        "--prob-phase-band-sigma", "3.75",
        "--slab-center-width", "64",
        "--phase-cache", "cache.zarr",
        "--phase-cache-winding-stride", "2",
    ])
    args = infer.parse_args()
    assert args.prob_combine == "phase"
    assert args.prob_phase_level_half_life == 3
    assert args.prob_phase_max_level == 5
    assert args.prob_phase_edge_taper == 6
    assert args.prob_phase_agreement_power == 2
    assert args.prob_phase_min_observations == 4
    assert args.prob_phase_band_sigma == 3.75
    assert args.slab_center_width == 64
    assert args.phase_cache_winding_stride == 2


def test_center_width_selects_same_physical_tile_across_column_grids():
    args = SimpleNamespace(slab_center_width=48)
    native = infer._selected_columns(32, 4, 1, args)
    upsampled = infer._selected_columns(125, 1, 1, args)
    np.testing.assert_array_equal(native, np.arange(10, 22))
    np.testing.assert_array_equal(upsampled, np.arange(38, 87))


def test_cached_winding_subsample_preserves_indices_and_both_bounds():
    windings = np.repeat(np.array([10, 13, 16, 19]), 2)
    rays = {
        "seed_windings": np.array([10, 13, 16, 19]),
        "seed_winding": windings,
        "seed_xyz": np.arange(24).reshape(8, 3),
        "direction_xyz": np.arange(24, 48).reshape(8, 3),
        "global_index": np.arange(8),
    }
    selected = infer._subsample_cached_winding_sheets(rays, 2)
    np.testing.assert_array_equal(selected["seed_windings"], [10, 16, 19])
    np.testing.assert_array_equal(selected["seed_winding"], [10, 10, 16, 16,
                                                              19, 19])
    np.testing.assert_array_equal(selected["global_index"], [0, 1, 4, 5, 6, 7])


def test_phase_consensus_renders_physical_distance_kernel():
    # One crossing at x=8 and a phase density of one winding per 16 voxels.
    phase = (np.arange(16) - 8) / 16
    angle = phase * (2 * np.pi)
    weight = np.full(16, 2.0)
    evidence = infer._render_phase_consensus(
        np.cos(angle) * weight,
        np.sin(angle) * weight,
        np.full(16, 2 / 16),
        weight,
        np.full(16, 2),
        sigma_voxels=1,
        min_observations=2,
    )
    assert evidence[8] == 255
    assert evidence[7] == evidence[9] == 155
    assert evidence[6] == evidence[10] == 35
    assert evidence[5] <= 3 and evidence[11] <= 3


def test_phase_consensus_attenuates_disagreement_and_requires_support():
    # Two equal observations at phases -1/8 and +1/8 average to a crossing,
    # but their circular concentration is sqrt(1/2), not false certainty.
    angle = np.pi / 4
    cosine = np.array([2 * np.cos(angle)])
    sine = np.array([0.0])
    density = np.array([2 / 16])
    weight = np.array([2.0])
    count = np.array([2])
    evidence = infer._render_phase_consensus(
        cosine, sine, density, weight, count,
        sigma_voxels=1, agreement_power=1, min_observations=2)
    assert evidence[0] == 180

    unsupported = infer._render_phase_consensus(
        cosine, sine, density, weight, count,
        sigma_voxels=1, agreement_power=1, min_observations=3)
    assert unsupported[0] == 0


def test_phase_consensus_is_invariant_to_integer_phase_offsets():
    phase = np.array([0.0, 1.0, -3.0, 12.0])
    angle = phase * (2 * np.pi)
    evidence = infer._render_phase_consensus(
        np.array([np.cos(angle).sum()]),
        np.array([np.sin(angle).sum()]),
        np.array([4 / 16]), np.array([4.0]), np.array([4]),
        sigma_voxels=1, min_observations=2)
    assert evidence[0] == 255


def test_phase_spill_roundtrip_preserves_sufficient_statistics(tmp_path):
    writer = infer._PhaseSpillWriter(
        tmp_path / "phase_spill_0", (32, 32, 32), 16)
    keys = np.array([
        (1 << 42) + (2 << 21) + 3,
        (20 << 42) + (21 << 21) + 22,
    ], dtype=np.uint64)
    values = [
        np.array([1.25, -2.5], dtype=np.float32),
        np.array([0.5, 3.0], dtype=np.float32),
        np.array([0.2, 0.4], dtype=np.float32),
        np.array([2.0, 4.0], dtype=np.float32),
        np.array([3, 5], dtype=np.uint32),
    ]
    writer.add_sorted_aggregates(
        keys, *values, np.array([0, 7]), np.array([0, 1]))
    records = [
        block for path in sorted(writer.directory.glob("*.rec"))
        for block in infer._iter_phase_aggregates(path)
    ]
    merged = np.concatenate(records)
    np.testing.assert_array_equal(merged["key"], keys)
    for name, expected in zip(
            ("cosine", "sine", "density", "weight", "count"), values):
        np.testing.assert_array_equal(merged[name], expected)


def test_phase_projection_registers_anchor_and_tracks_density_on_cpu():
    length = 16
    phase_line = (torch.arange(length, dtype=torch.float32) - 8) / 4
    phase = phase_line.expand(5, 5, -1).clone()
    valid = torch.ones_like(phase, dtype=torch.bool)
    frame = SlabFrame(
        origin=np.zeros(3), axis_a=np.array([1.0, 0, 0]),
        axis_b=np.array([0, 1.0, 0]), direction=np.array([0, 0, 1.0]),
        spacing=1.0)
    args = SimpleNamespace(
        prob_column_margin=1, prob_ray_margin=2, prob_column_step=1,
        column_step=1, prob_phase_max_level=2.5, max_level=2,
        prob_phase_level_half_life=2.0, prob_phase_edge_taper=0,
        prob_phase_band_sigma=4.0, passage_sigma_samples=1.0,
        model_spacing=1.0,
        output_downsample=1,
    )
    records = infer._phase_volume_records_cuda(
        phase, valid, frame, length, 1, args, (16, 16, 16))
    assert records is not None
    keys, cosine, sine, density, weight, count = records
    assert len(keys) == 3 * 3 * 12
    np.testing.assert_allclose(
        (density / weight).numpy(), 0.25, rtol=1e-6)
    np.testing.assert_array_equal(count.numpy(), 1)

    z = (keys >> 42).numpy()
    crossing = np.isin(z, [4, 8, 12])
    evidence = infer._render_phase_consensus(
        cosine.numpy(), sine.numpy(), density.numpy(), weight.numpy(),
        count.numpy(), sigma_voxels=1, min_observations=1)
    np.testing.assert_array_equal(evidence[crossing], 255)


def test_phase_projection_supports_zero_ray_margin():
    length = 8
    phase = (torch.arange(length, dtype=torch.float32) / 4) \
        .expand(3, 3, -1).clone()
    valid = torch.ones_like(phase, dtype=torch.bool)
    frame = SlabFrame(
        origin=np.zeros(3), axis_a=np.array([1.0, 0, 0]),
        axis_b=np.array([0, 1.0, 0]), direction=np.array([0, 0, 1.0]),
        spacing=1.0)
    args = SimpleNamespace(
        prob_column_margin=0, prob_ray_margin=0, prob_column_step=1,
        column_step=1, prob_phase_max_level=3, max_level=3,
        prob_phase_level_half_life=2.0, prob_phase_edge_taper=0,
        prob_phase_band_sigma=4.0, passage_sigma_samples=1.0,
        model_spacing=1.0,
        output_downsample=1)
    records = infer._phase_volume_records_cuda(
        phase, valid, frame, length, 1, args, (8, 8, 8))
    assert records is not None
    np.testing.assert_allclose(
        (records[3] / records[4]).numpy(), 0.25, rtol=1e-6)


def test_phase_accumulator_reduces_cross_slab_records_on_cpu(tmp_path):
    writer = infer._PhaseSpillWriter(
        tmp_path / "phase_spill_0", (16, 16, 16), 16)
    observation_a = (
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0]),
        torch.tensor([0.1, 0.2]), torch.tensor([0.75, 1.0]),
        torch.tensor([1, 1], dtype=torch.int64),
    )
    observation_b = (
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([3.0]), torch.tensor([-0.5]),
        torch.tensor([0.3]), torch.tensor([0.25]),
        torch.tensor([1], dtype=torch.int64),
    )
    accumulator = infer._GpuPhaseAccumulator(writer, flush_records=100)
    accumulator.add_batch([observation_a, observation_b])
    accumulator.close()
    merged = np.concatenate([
        block for path in writer.directory.glob("*.rec")
        for block in infer._iter_phase_aggregates(path)
    ])
    np.testing.assert_array_equal(merged["key"], [1, 2])
    np.testing.assert_allclose(merged["cosine"], [4, 2])
    np.testing.assert_allclose(merged["sine"], [0, 1])
    np.testing.assert_allclose(merged["density"], [0.4, 0.2])
    np.testing.assert_allclose(merged["weight"], [1, 1])
    np.testing.assert_array_equal(merged["count"], [2, 1])


def test_phase_bucket_folds_spills_and_writes_evidence(tmp_path, monkeypatch):
    shape = (4, 4, 16)
    x = np.arange(16, dtype=np.uint64)
    keys = x.copy()
    phase = (np.arange(16) - 8) / 16
    angle = phase * (2 * np.pi)
    weight = np.full(16, 2.0, dtype=np.float32)
    writer = infer._PhaseSpillWriter(
        tmp_path / "phase_spill_0", shape, 16)
    writer.add_sorted_aggregates(
        keys, np.cos(angle).astype(np.float32) * weight,
        np.sin(angle).astype(np.float32) * weight,
        np.full(16, 2 / 16, dtype=np.float32), weight,
        np.full(16, 2, dtype=np.uint32), np.array([0]), np.array([0]))

    output = np.zeros(shape, dtype=np.uint8)
    fake_group = {"crossing_prob": output}
    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(
        open_group=lambda *_args, **_kwargs: fake_group))
    written = infer._write_phase_bucket((
        "unused.zarr", "crossing_prob", shape, 16, 0, 0, 0,
        [str(next(writer.directory.glob("*.rec")))], 0, 1.0, 1.0, 2,
    ))
    assert written > 0
    assert output[0, 0, 8] == 255
    assert output[0, 0, 7] == output[0, 0, 9] == 155
