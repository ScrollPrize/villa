from unittest.mock import patch

import pytest
import torch

from winding_supervision import get_winding_inference_losses


RELATIVE = "dense_spacing_winding_model_relative"
DENSITY = "dense_spacing_winding_model_density"


class _IdentityTransform:
    def __call__(self, points):
        return points


class _Store:
    def __init__(self, relative, density):
        self.relative = relative
        self.density = density

    def sample_relative(self, count, min_delta, max_delta, *, generator=None):
        return self.relative

    def sample_adjacent(self, count, *, generator=None):
        return self.density


class _ShellMap:
    def __init__(self, shell_radii, confidence, min_confidence=0.25):
        self.shell_radii = torch.tensor(shell_radii, dtype=torch.float32)
        self.confidence = torch.tensor(confidence, dtype=torch.float32)
        self.min_confidence = min_confidence

    def lookup(self, scan_zyx):
        assert self.shell_radii.shape == scan_zyx.shape[:-1]
        radius = torch.linalg.norm(scan_zyx[..., 1:], dim=-1)
        valid = self.confidence >= self.min_confidence
        return self.shell_radii, radius, self.confidence, valid


def _samples(present):
    if not present:
        return {
            "points": torch.empty((0, 2, 3), dtype=torch.float32),
            "target": torch.empty((0,), dtype=torch.float32),
        }
    # The identity transform predicts one winding between radii 10 and 20.
    # A target of two therefore gives a non-zero Huber loss for valid pairs.
    return {
        "points": torch.tensor(
            [[[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]]],
            dtype=torch.float32,
        ),
        "target": torch.tensor([2.0]),
    }


def _run(relative_shell=None, density_shell=None, *,
         relative_confidence=(1.0, 1.0), density_confidence=(1.0, 1.0)):
    relative_present = relative_shell is not None
    density_present = density_shell is not None
    store = _Store(_samples(relative_present), _samples(density_present))
    shell_radii = []
    confidence = []
    if relative_present:
        shell_radii.append(relative_shell)
        confidence.append(relative_confidence)
    if density_present:
        shell_radii.append(density_shell)
        confidence.append(density_confidence)
    shell_map = _ShellMap(shell_radii, confidence)
    cfg = {
        "sample_count_winding_model_relative_pairs": int(relative_present),
        "sample_count_winding_model_density_pairs": int(density_present),
        "loss_weight_dense_spacing": 1.0,
        "loss_weight_dense_spacing_density": 1.0,
        "winding_model_relative_pair_delta": (1, 2),
        "winding_model_huber_delta": 1.0,
    }
    with patch("winding_supervision.record_loss_samples") as record:
        losses, metrics = get_winding_inference_losses(
            _IdentityTransform(), torch.tensor(10.0), store, shell_map, cfg,
            z_begin=-1, z_end=1,
        )
    recorded_masks = {
        call.args[0]: call.args[3] for call in record.call_args_list
    }
    return losses, metrics, recorded_masks


def test_pair_with_both_endpoints_inside_shell_remains_valid():
    losses, metrics, masks = _run(relative_shell=(11.0, 21.0))

    torch.testing.assert_close(losses[RELATIVE], torch.tensor(0.5))
    assert metrics[f"{RELATIVE}_valid_fraction"] == 1.0
    torch.testing.assert_close(masks[RELATIVE], torch.tensor([True]))


@pytest.mark.parametrize("shell_radii", [(9.0, 21.0), (11.0, 19.0)])
def test_pair_is_masked_when_either_endpoint_is_outside(shell_radii):
    losses, metrics, masks = _run(relative_shell=shell_radii)

    torch.testing.assert_close(losses[RELATIVE], torch.tensor(0.0))
    assert metrics[f"{RELATIVE}_valid_fraction"] == 0.0
    torch.testing.assert_close(masks[RELATIVE], torch.tensor([False]))


def test_points_exactly_on_shell_boundary_remain_valid():
    losses, metrics, masks = _run(relative_shell=(10.0, 20.0))

    torch.testing.assert_close(losses[RELATIVE], torch.tensor(0.5))
    assert metrics[f"{RELATIVE}_valid_fraction"] == 1.0
    torch.testing.assert_close(masks[RELATIVE], torch.tensor([True]))


def test_low_confidence_at_either_endpoint_masks_pair():
    losses, metrics, masks = _run(
        relative_shell=(11.0, 21.0),
        relative_confidence=(1.0, 0.24),
    )

    torch.testing.assert_close(losses[RELATIVE], torch.tensor(0.0))
    assert metrics[f"{RELATIVE}_valid_fraction"] == 0.0
    torch.testing.assert_close(masks[RELATIVE], torch.tensor([False]))


def test_entirely_shell_masked_batch_has_finite_zero_losses_and_metrics():
    losses, metrics, masks = _run(
        relative_shell=(9.0, 21.0),
        density_shell=(11.0, 21.0),
        density_confidence=(0.24, 1.0),
    )

    for name in (RELATIVE, DENSITY):
        assert torch.isfinite(losses[name])
        torch.testing.assert_close(losses[name], torch.tensor(0.0))
        assert metrics[f"{name}_valid_fraction"] == 0.0
        torch.testing.assert_close(masks[name], torch.tensor([False]))


@pytest.mark.parametrize(
    ("relative_shell", "density_shell", "valid_name", "masked_name"),
    [
        ((11.0, 21.0), (11.0, 19.0), RELATIVE, DENSITY),
        ((11.0, 19.0), (11.0, 21.0), DENSITY, RELATIVE),
    ],
)
def test_relative_and_adjacent_density_components_both_use_shell_mask(
    relative_shell, density_shell, valid_name, masked_name,
):
    losses, metrics, masks = _run(
        relative_shell=relative_shell, density_shell=density_shell)

    torch.testing.assert_close(losses[valid_name], torch.tensor(0.5))
    torch.testing.assert_close(losses[masked_name], torch.tensor(0.0))
    assert metrics[f"{valid_name}_valid_fraction"] == 1.0
    assert metrics[f"{masked_name}_valid_fraction"] == 0.0
    torch.testing.assert_close(masks[valid_name], torch.tensor([True]))
    torch.testing.assert_close(masks[masked_name], torch.tensor([False]))


def test_mixed_valid_and_invalid_pairs_reduce_only_valid_entries():
    points = torch.tensor([
        [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]],
        [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]],
    ])
    empty = _samples(False)
    store = _Store({"points": points, "target": torch.tensor([2.0, 2.0])},
                   empty)
    shell_map = _ShellMap(
        [[11.0, 21.0], [9.0, 21.0]],
        [[1.0, 1.0], [1.0, 1.0]])
    cfg = {
        "sample_count_winding_model_relative_pairs": 2,
        "sample_count_winding_model_density_pairs": 0,
        "loss_weight_dense_spacing": 1.0,
        "loss_weight_dense_spacing_density": 1.0,
        "winding_model_relative_pair_delta": (1, 2),
        "winding_model_huber_delta": 1.0,
    }
    with patch("winding_supervision.record_loss_samples") as record:
        losses, metrics = get_winding_inference_losses(
            _IdentityTransform(), torch.tensor(10.0), store, shell_map, cfg,
            z_begin=-1, z_end=1)

    torch.testing.assert_close(losses[RELATIVE], torch.tensor(0.5))
    assert metrics[f"{RELATIVE}_valid_fraction"] == 0.5
    torch.testing.assert_close(
        record.call_args_list[0].args[3], torch.tensor([True, False]))


# --- ray orientation: crossing levels count along +t, not outward ---------

import json

import numpy as np

from winding_supervision import (
    ARTIFACT_TYPE, FORMAT_VERSION, WindingInferenceStore, _canonical_digest,
    _sha256,
)


def _write_store(root, origins, steps, crossing_t, crossing_level, offsets):
    """A one-shard crossing store laid out the way the exporter writes it."""
    shard = root / "shard_0"
    shard.mkdir(parents=True)
    values = {
        "ray_origin_zyx": np.asarray(origins, np.float32),
        "ray_step_zyx": np.asarray(steps, np.float32),
        "seed_winding": np.zeros(len(origins), np.int16),
        "crossing_t": np.asarray(crossing_t, np.float32),
        "crossing_level": np.asarray(crossing_level, np.int16),
        "crossing_offsets": np.asarray(offsets, np.int64),
    }
    arrays = {}
    for name, value in values.items():
        path = shard / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        arrays[name] = {"file": path.name, "shape": list(value.shape),
                        "dtype": value.dtype.str, "sha256": _sha256(path)}
    manifest = {
        "artifact_type": ARTIFACT_TYPE, "format_version": FORMAT_VERSION,
        "coordinate_order": "zyx", "num_rays": len(origins),
        "num_crossings": len(crossing_t),
        "shards": [{"name": "shard_0", "arrays": arrays}],
    }
    manifest["fingerprint"] = _canonical_digest(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))


class _OpenShell:
    def lookup(self, scan_zyx):
        radius = torch.linalg.norm(scan_zyx[..., 1:], dim=-1)
        ones = torch.ones_like(radius)
        return ones * 1e6, radius, ones, torch.ones_like(radius, dtype=torch.bool)


def _two_ray_store(tmp_path, **kwargs):
    # Umbilicus on the z axis. Sheets every 10 voxels along +y (one winding
    # per 10 voxels of radius). Ray 0 points outward from y=100, ray 1 points
    # at the umbilicus from y=300. Both carry levels 0, 1, 2 along +t, as the
    # exporter writes them (the model's phase always increases along the ray).
    _write_store(
        tmp_path,
        origins=[[0.0, 100.0, 0.0], [0.0, 300.0, 0.0]],
        steps=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        crossing_t=[0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
        crossing_level=[0, 1, 2, 0, 1, 2],
        offsets=[0, 3, 6],
    )
    return WindingInferenceStore(tmp_path, "cpu", **kwargs)


def _umbilicus_on_z_axis(z):
    return np.zeros((len(z), 2))


def _perfect_spiral_loss(store):
    # The identity transform is the true spiral here, so a correct target
    # gives zero residual on every pair.
    cfg = {
        "sample_count_winding_model_relative_pairs": 256,
        "sample_count_winding_model_density_pairs": 256,
        "loss_weight_dense_spacing": 1.0,
        "loss_weight_dense_spacing_density": 1.0,
        "winding_model_relative_pair_delta": (1, 2),
        "winding_model_huber_delta": 1.0,
    }
    with patch("winding_supervision.record_loss_samples"):
        losses, metrics = get_winding_inference_losses(
            _IdentityTransform(), torch.tensor(10.0), store, _OpenShell(), cfg,
            z_begin=-1, z_end=1, generator=torch.Generator().manual_seed(0))
    return losses, metrics


def test_ray_pointing_at_the_umbilicus_is_flipped(tmp_path):
    store = _two_ray_store(tmp_path, z_to_umbilicus_yx=_umbilicus_on_z_axis)

    torch.testing.assert_close(store.orientation, torch.tensor([1.0, -1.0]))
    assert store.num_inward_rays == 1
    inward = store._materialize(
        torch.tensor([1]), torch.tensor([0]), torch.tensor([2]))
    torch.testing.assert_close(inward["target"], torch.tensor([-2.0]))


def test_perfect_spiral_scores_zero_on_inward_rays(tmp_path):
    store = _two_ray_store(tmp_path, z_to_umbilicus_yx=_umbilicus_on_z_axis)

    losses, metrics = _perfect_spiral_loss(store)

    for name in (RELATIVE, DENSITY):
        assert metrics[f"{name}_valid_fraction"] == 1.0
        torch.testing.assert_close(losses[name], torch.tensor(0.0))


def test_without_an_umbilicus_rays_keep_their_exported_orientation(tmp_path):
    store = _two_ray_store(tmp_path)

    torch.testing.assert_close(store.orientation, torch.tensor([1.0, 1.0]))
    assert store.num_inward_rays == 0
