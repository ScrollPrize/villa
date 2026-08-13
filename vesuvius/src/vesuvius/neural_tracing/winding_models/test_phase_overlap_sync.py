from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.winding_models import infer_winding_volume as infer
from vesuvius.neural_tracing.winding_models.phase_overlap_sync import (
    _phase_sync_neighbor_table,
    _sample_phase_trilinear,
    _solve_phase_overlap_graph,
)
from vesuvius.neural_tracing.winding_models.volume_slab_extractor import (
    SlabFrame,
)


def test_overlap_registration_cli_is_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "infer_winding_volume.py", "fit.ckpt", "out.zarr",
        "--model-ckpt", "model.pth",
        "--reference-zarr", "reference.zarr",
        "--phase-cache", "phase.zarr",
        "--phase-cache-allow-relocated-inputs",
        "--phase-registration", "overlap",
        "--prob-volume", "--prob-combine", "phase-label",
        "--phase-sync-radius", "144",
        "--phase-sync-neighbors", "18",
        "--prob-phase-min-effective-observations", "1.75",
        "--prob-phase-min-weight", "0.6",
    ])
    args = infer.parse_args()
    assert args.phase_registration == "overlap"
    assert args.phase_cache_allow_relocated_inputs
    assert args.prob_combine == "phase-label"
    assert args.phase_sync_radius == 144
    assert args.phase_sync_neighbors == 18
    assert args.prob_phase_min_effective_observations == 1.75
    assert args.prob_phase_min_weight == 0.6


def test_phase_cache_supplies_model_geometry_without_checkpoint(tmp_path):
    attributes = {
        "artifact_type": "winding_native_phase_cache",
        "ray_length": 384,
        "transverse_size": 128,
        "column_stride": 4,
        "spacing": 1.0,
        "sampling": "trilinear",
        "crossing_sigma_wv": 1.25,
    }
    (tmp_path / "zarr.json").write_text(json.dumps({
        "attributes": attributes, "zarr_format": 3, "node_type": "group"}))
    config = infer._native_phase_cache_model_cfg(tmp_path)
    assert config["model"]["use_crossing_head"] is False
    assert config["ray_length"] == 384
    assert config["column_stride"] == 4
    assert config["crossing_sigma_wv"] == 1.25


def test_phase_trilinear_sampling_matches_affine_field():
    aa, bb, kk = np.meshgrid(
        np.arange(4), np.arange(5), np.arange(6), indexing="ij")
    phase = 2.0 * aa - 3.0 * bb + 0.25 * kk + 7.0
    point = np.array([1.25, 2.5, 3.75])
    expected = 2.0 * point[0] - 3.0 * point[1] + 0.25 * point[2] + 7.0
    assert _sample_phase_trilinear(phase, point) == expected
    assert np.isnan(_sample_phase_trilinear(phase, [-0.1, 2, 3]))


def test_neighbor_table_uses_world_distance_not_winding_adjacency():
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
    ])
    # Deliberately nonsensical fitted winding labels: spatial neighbors must
    # still connect, while the topologically adjacent but distant seed cannot.
    winding = np.array([90, 3, 71, 4])
    neighbors = _phase_sync_neighbor_table(
        xyz, winding, radius=5, max_neighbors=3)
    assert 1 in neighbors[0]
    assert 2 in neighbors[0]
    assert 3 not in neighbors[0]


def test_robust_overlap_graph_recovers_offsets_with_bad_seed_and_cycle():
    nodes = 8
    truth = np.linspace(-0.35, 0.35, nodes)
    prior = truth.copy()
    prior[6] += 2.0  # locally wrong fitted winding/anchor
    edge_u, edge_v, delta, weight = [], [], [], []
    for left in range(nodes):
        for right in range(left + 1, nodes):
            edge_u.append(left)
            edge_v.append(right)
            delta.append(truth[right] - truth[left])
            weight.append(1.0)
    # One cycle-inconsistent match is strongly downweighted by IRLS.
    edge_u.append(0)
    edge_v.append(nodes - 1)
    delta.append(4.0)
    weight.append(0.2)

    solved, stats = _solve_phase_overlap_graph(
        prior, edge_u, edge_v, delta, weight,
        iterations=6, huber=0.15, prior_weight=0.02,
        prior_huber=0.4, max_correction=4.0)
    solved -= np.median(solved - truth)
    np.testing.assert_allclose(solved, truth, atol=0.01)
    assert stats["edges"] == len(edge_u)
    assert stats["supported_nodes"] == nodes
    assert stats["edge_residual_p95_abs"] < 0.01


def test_phase_offset_moves_passages_without_changing_default():
    length = 32
    line = np.arange(length, dtype=np.float32) / 4.0
    phase = np.broadcast_to(line, (3, 3, length)).copy()
    valid = np.ones_like(phase, dtype=bool)
    frame = SlabFrame(
        origin=np.zeros(3), axis_a=np.array([1.0, 0, 0]),
        axis_b=np.array([0, 1.0, 0]), direction=np.array([0, 0, 1.0]),
        spacing=1.0)
    args = SimpleNamespace(
        column_step=1, slab_center_width=None, min_prob_keep=0,
        max_level=8, passage_sigma_samples=1.0)
    legacy = infer._decode_slab_phase(
        None, phase, valid, frame, length, 1, 20, args)
    explicit_zero = infer._decode_slab_phase(
        None, phase, valid, frame, length, 1, 20, args,
        phase_offset=0.0)
    for actual, expected in zip(explicit_zero, legacy):
        np.testing.assert_array_equal(actual, expected)

    shifted = infer._decode_slab_phase(
        None, phase, valid, frame, length, 1, 20, args,
        phase_offset=0.5)
    # The finite ray edge gate changes which outer passage is retained. Match
    # equal winding labels and verify that +0.5 moves those passages two
    # samples toward the ray origin.
    common = np.intersect1d(legacy[1], shifted[1])
    legacy_positions = np.concatenate([
        legacy[0][legacy[1] == level, 2] for level in common])
    shifted_positions = np.concatenate([
        shifted[0][shifted[1] == level, 2] for level in common])
    np.testing.assert_allclose(
        shifted_positions, legacy_positions - 2.0, atol=1e-6)


def test_labeled_phase_requires_effective_weighted_support():
    evidence = infer._render_phase_label_consensus(
        np.array([2.0, 1.01]), np.zeros(2), np.array([0.2, 0.101]),
        np.array([2.0, 1.01]), np.array([2.0, 1.0001]),
        np.array([2, 2]), sigma_voxels=1.0,
        min_observations=2, min_effective_observations=1.5,
        min_weight=0.5)
    assert evidence[0] == 255  # two equal, meaningful slab votes
    assert evidence[1] == 0    # one full vote plus an almost-zero unlock vote

    # Different synchronized winding labels are reduced as separate rows;
    # neither can satisfy the raw distinct-slab gate by itself.
    separated = infer._render_phase_label_consensus(
        [1.0, 1.0], [0.0, 0.0], [0.1, 0.1],
        [1.0, 1.0], [1.0, 1.0], [1, 1], sigma_voxels=1.0,
        min_observations=2, min_effective_observations=1.0,
        min_weight=0.0)
    np.testing.assert_array_equal(separated, [0, 0])


def test_labeled_phase_projection_carries_absolute_winding():
    length = 16
    line = (np.arange(length, dtype=np.float32) - 8.0) / 4.0
    phase = np.broadcast_to(line, (5, 5, length)).copy()
    valid = np.ones_like(phase, dtype=bool)
    frame = SlabFrame(
        origin=np.zeros(3), axis_a=np.array([1.0, 0, 0]),
        axis_b=np.array([0, 1.0, 0]), direction=np.array([0, 0, 1.0]),
        spacing=1.0)
    args = SimpleNamespace(
        prob_column_margin=1, prob_ray_margin=2, prob_column_step=1,
        column_step=1, slab_center_width=None,
        prob_phase_max_level=3.0, max_level=3,
        prob_phase_level_half_life=2.0, prob_phase_edge_taper=0,
        prob_phase_band_sigma=4.0, passage_sigma_samples=1.0,
        model_spacing=1.0, output_downsample=1)
    import torch

    records = infer._phase_volume_records_cuda(
        torch.from_numpy(phase), torch.from_numpy(valid), frame, length, 1,
        args, (16, 16, 16), phase_offset=0.5, seed_winding=20,
        label_aware=True)
    assert records is not None and len(records) == 7
    keys, _cos, _sin, _density, weight, weight_sq, count = records
    winding = ((keys & 0xFFFF).numpy().astype(np.int64) - 32768)
    assert set(np.unique(winding)).issubset(set(range(18, 24)))
    np.testing.assert_allclose(weight_sq.numpy(), weight.numpy() ** 2)
    np.testing.assert_array_equal(count.numpy(), 1)
