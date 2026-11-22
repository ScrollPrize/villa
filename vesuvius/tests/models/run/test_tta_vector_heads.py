"""TTA has to transform the components of vector and tensor heads, not only their positions.

Mirroring the input over an axis negates the component of a direction along that axis;
transposing two axes swaps their components. The auxiliary heads (surface_normals,
nearest_component, inplane_direction: x y z components; structure_tensor: Jzz Jzy Jzx Jyy
Jyx Jxx) therefore need more than the spatial un-flip, or the TTA variants cancel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from vc_reader.reader import ZarrWriter, open_zarr
import vesuvius.models
from vesuvius.models.run import inference, tta
from vesuvius.models.run.inference import Inferer

TARGETS = {
    "surface": {"out_channels": 2},
    "normals": {"out_channels": 3, "auxiliary_task": True, "task_type": "surface_normals"},
    "st": {"out_channels": 6, "auxiliary_task": True, "task_type": "structure_tensor"},
    "direction": {"out_channels": 3, "auxiliary_task": True, "task_type": "inplane_direction"},
    "nearest": {"out_channels": 4, "auxiliary_task": True, "task_type": "nearest_component"},
}
TARGETS_2D = {
    "surface": {"out_channels": 2},
    "normals": {"out_channels": 2, "auxiliary_task": True, "task_type": "surface_normals"},
    "st": {"out_channels": 3, "auxiliary_task": True, "task_type": "structure_tensor"},
    "direction": {"out_channels": 2, "auxiliary_task": True, "task_type": "inplane_direction"},
}


def _target_info(targets):
    info, start = {}, 0
    for name, cfg in targets.items():
        info[name] = {"out_channels": cfg["out_channels"], "start_channel": start,
                      "end_channel": start + cfg["out_channels"]}
        start += cfg["out_channels"]
    return info


class Oracle(torch.nn.Module):
    """Heads computed from the input gradient: equivariant to every flip and axis swap."""

    def forward(self, x):
        s = x[:, :1].float()
        nd = s.ndim - 2
        grads = torch.gradient(s, dim=tuple(range(2, 2 + nd)))  # (z,) y, x
        comps = list(reversed(grads))  # x, y (, z)
        g = torch.cat([-c for c in comps], dim=1)
        n = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
        if nd == 3:
            gz, gy, gx = grads
            st = torch.cat([gz * gz, gz * gy, gz * gx, gy * gy, gy * gx, gx * gx], dim=1)
        else:
            gy, gx = grads
            st = torch.cat([gy * gy, gy * gx, gx * gx], dim=1)
        # an eigenvector: the direction is defined, its sign is not (here: the half-space
        # 4x + 2y + z >= 0 in which vesuvius.predict reports it)
        side = 4 * n[:, :1] + 2 * n[:, 1:2] + (n[:, 2:3] if nd == 3 else 0)
        direction = n * torch.where(side < 0, -1.0, 1.0)
        out = {"surface": torch.cat([-s, s], dim=1), "normals": n, "st": st, "direction": direction}
        if nd == 3:
            out["nearest"] = torch.cat([-n, s.abs()], dim=1)
        return out


def _volume(shape):
    grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    centre = [0.45 * n + 0.3 for n in shape]
    r = np.sqrt(sum((g - c) ** 2 for g, c in zip(grids, centre)))
    return torch.from_numpy((1.0 / (1.0 + np.exp((r - 0.3 * min(shape)) / 1.5))).astype(np.float32))


def _concat(targets):
    return lambda out: torch.cat([out[name] for name in targets if name in out], dim=1)


def _defined(x):
    """Voxels where the gradient direction is defined (an undefined normal has no sign)."""
    n = Oracle()(x)["normals"]
    return (n.norm(dim=1) > 0.5)[0]


@pytest.mark.parametrize("tta_type", ["mirroring", "rotation"])
@pytest.mark.parametrize("nd", [3, 2])
def test_tta_keeps_vector_and_tensor_heads(nd, tta_type):
    targets = TARGETS if nd == 3 else TARGETS_2D
    x = _volume((20, 20, 20) if nd == 3 else (24, 24))[None, None]
    model, concat = Oracle(), _concat(targets)
    reference = concat(model(x))
    groups = tta.tta_channel_groups(targets, _target_info(targets), nd)
    assert [g["kind"] for g in groups] == (["vector", "tensor", "axial", "vector"] if nd == 3
                                           else ["vector", "tensor", "axial"])

    out = tta.infer_with_tta(model, x, tta_type, is_multi_task=True,
                             concat_multi_task_outputs=concat, channel_groups=groups)

    mask = _defined(x)
    diff = (out - reference).abs()[0][:, mask]
    assert diff.max() < 1e-5, f"max |TTA - no TTA| over defined voxels = {diff.max():.4f}"


def test_batched_mirroring_matches_sequential():
    x = _volume((16, 16, 16))[None, None]
    model, concat = Oracle(), _concat(TARGETS)
    groups = tta.tta_channel_groups(TARGETS, _target_info(TARGETS), 3)
    batched = tta.infer_with_tta_batched_3d(model, x, is_multi_task=True,
                                            concat_multi_task_outputs=concat, channel_groups=groups)
    sequential = tta.infer_with_tta_sequential(model, x, "mirroring", is_multi_task=True,
                                               concat_multi_task_outputs=concat, channel_groups=groups)
    assert torch.allclose(batched, sequential, atol=1e-6)
    assert torch.allclose(batched[:, :2], concat(model(x))[:, :2], atol=1e-6)


def test_segmentation_only_models_average_as_before():
    x = torch.rand(1, 1, 8, 8, 8)
    model = torch.nn.Conv3d(1, 2, 3, padding=1)
    flips = [[], [-1], [-2], [-3], [-1, -2], [-1, -3], [-2, -3], [-1, -2, -3]]
    with torch.no_grad():
        expected = torch.stack([torch.flip(model(torch.flip(x, f)), f) if f else model(x) for f in flips]).mean(0)
        assert torch.equal(tta.infer_with_tta(model, x, "mirroring"), expected)
    assert tta.tta_channel_groups({"surface": {"out_channels": 2}}, _target_info({"surface": {"out_channels": 2}}), 3) == []


def test_predict_keeps_normals_with_default_tta(tmp_path, monkeypatch):
    """vesuvius.predict on a multi-task checkpoint (default mirroring TTA) writes the normals it predicts."""
    shape, patch = (32, 32, 32), (16, 16, 16)
    volume = _volume(shape).numpy()
    writer = ZarrWriter.create(str(tmp_path / "volume.zarr"), shape=shape, chunks=patch, dtype=np.float32)
    writer[:] = volume
    (tmp_path / "model.pth").write_bytes(b"")
    targets = {name: TARGETS[name] for name in ("surface", "normals")}

    def fake_load(self, checkpoint_path):
        return {"network": Oracle(), "patch_size": patch, "num_input_channels": 1, "num_seg_heads": 5,
                "targets": targets, "normalization_scheme": "none"}

    monkeypatch.setattr(inference.Inferer, "_load_train_py_model", fake_load)

    def predict(name, **kwargs):
        out = tmp_path / name
        Inferer(model_path=str(tmp_path / "model.pth"), input_dir=str(tmp_path / "volume.zarr"),
                output_dir=str(tmp_path / "parts"), model_type="train_py", device="cpu",
                num_dataloader_workers=0, in_memory_blend={"output_path": str(out)}, **kwargs).infer()
        return np.asarray(open_zarr(str(out))[:], dtype=np.float32)

    with_tta = predict("tta.zarr")
    without = predict("no_tta.zarr", do_tta=False)
    mask = _defined(torch.from_numpy(volume)[None, None]).numpy()
    mask[[0, -1]] = mask[:, [0, -1]] = mask[:, :, [0, -1]] = False
    length = np.sqrt((with_tta[2:5] ** 2).sum(0))[mask]
    assert length.mean() > 0.99, f"TTA normals have mean length {length.mean():.3f}"
    assert np.abs(with_tta[2:5] - without[2:5])[:, mask].max() < 1e-2  # float16 blend


def _run_helper(helper, model, x, targets, groups):
    kwargs = dict(is_multi_task=True, concat_multi_task_outputs=_concat(targets), channel_groups=groups)
    with torch.inference_mode():
        if helper == "batched":
            return tta.infer_with_tta_batched_3d(model, x, **kwargs)
        return tta.infer_with_tta_sequential(model, x, "mirroring", **kwargs)


@pytest.mark.parametrize("helper", ["batched", "sequential"])
def test_half_precision_sign_alignment_keeps_small_directions(helper):
    """CUDA inference runs under autocast: the heads come out in float16. Opposite in-plane
    vectors of length 1e-4 have a float16 product that rounds to zero."""
    targets = {name: TARGETS[name] for name in ("surface", "direction")}
    groups = tta.tta_channel_groups(targets, _target_info(targets), 3)

    def model(x):  # the x axis in every mirrored frame, at length 1e-4
        direction = torch.zeros(x.shape[0], 3, *x.shape[2:], dtype=torch.float16)
        direction[:, 0] = 1e-4
        return {"surface": x.repeat(1, 2, 1, 1, 1).half(), "direction": direction}

    out = _run_helper(helper, model, torch.rand(1, 1, 4, 4, 4), targets, groups)
    assert out.dtype == torch.float16
    assert torch.allclose(out[:, 2:5].float(), torch.tensor([1e-4, 0.0, 0.0]).view(1, 3, 1, 1, 1),
                          rtol=0, atol=1e-6), f"x component {out[0, 2].float().mean():.6f}"


def _peak_bytes(fn):
    """Peak bytes of the CPU tensors fn() allocates, from torch.profiler's allocation events."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    live = peak = 0
    for event in sorted(prof.events(), key=lambda e: e.time_range.start):
        live += event.self_cpu_memory_usage
        peak = max(peak, live)
    return peak


@pytest.mark.parametrize("helper", ["batched", "sequential"])
@pytest.mark.parametrize("head", ["normals", "direction"])
def test_component_transform_keeps_peak_memory(helper, head):
    """Undoing the component change works on the variants in place: no extra full-size copies
    of the output beyond what the plain mirroring average already holds."""
    targets = {name: TARGETS[name] for name in ("surface", head)}
    groups = tta.tta_channel_groups(targets, _target_info(targets), 3)

    def model(x):
        y = x.half()
        return {"surface": torch.cat([y, -y], dim=1), head: torch.cat([y, -y, y], dim=1)}

    x = torch.rand(1, 1, 32, 32, 32)
    variant = 5 * 32 ** 3 * 2  # one float16 output of 5 channels
    plain = _peak_bytes(lambda: _run_helper(helper, model, x, targets, None))
    transformed = _peak_bytes(lambda: _run_helper(helper, model, x, targets, groups))
    assert transformed - plain < variant, f"{(transformed - plain) / variant:.1f} extra output copies"


def _predict(tmp_path, monkeypatch, volume, network, targets, patch, name, **kwargs):
    """vesuvius.predict with an in-memory blend of `volume`; returns the written output."""
    if not (tmp_path / "model.pth").exists():
        (tmp_path / "model.pth").write_bytes(b"")
    writer = ZarrWriter.create(str(tmp_path / f"{name}_volume.zarr"), shape=volume.shape, chunks=patch,
                               dtype=np.float32)
    writer[:] = volume

    def fake_load(self, checkpoint_path):
        return {"network": network, "patch_size": patch, "num_input_channels": 1,
                "num_seg_heads": sum(t["out_channels"] for t in targets.values()),
                "targets": targets, "normalization_scheme": "none"}

    monkeypatch.setattr(inference.Inferer, "_load_train_py_model", fake_load)
    out = tmp_path / f"{name}.zarr"
    blend = {"output_path": str(out), **kwargs.pop("in_memory_blend", {})}
    Inferer(model_path=str(tmp_path / "model.pth"), input_dir=str(tmp_path / f"{name}_volume.zarr"),
            output_dir=str(tmp_path / f"{name}_parts"), model_type="train_py", device="cpu",
            num_dataloader_workers=0, in_memory_blend=blend, **kwargs).infer()
    return np.asarray(open_zarr(str(out))[:])


class AxisWithPatchSign(torch.nn.Module):
    """The x axis everywhere, with a sign picked by the patch's first voxel. In-plane directions
    are trained with a sign-invariant loss, so neighbouring patches may report one axis either way."""

    def forward(self, x):
        s = x[:, :1].float()
        sign = torch.where(s[:, :, :1, :1, :1] > 0.5, 1.0, -1.0).expand_as(s)
        return {"surface": torch.cat([-s, s], dim=1),
                "direction": torch.cat([sign, torch.zeros_like(s), torch.zeros_like(s)], dim=1)}


def test_overlapping_patches_keep_a_sign_ambiguous_axis(tmp_path, monkeypatch):
    """Two patches (50% overlap) predict the x axis with opposite signs; blending must keep it.

    The second patch's first voxel is dark, so it reports -x unmirrored and +x in the mirrored
    variants. A sign chosen per patch made the two cancel where they overlap (0 at x = 12, where
    both weigh the same)."""
    volume = np.full((16, 16, 24), 0.9, dtype=np.float32) + np.linspace(0, 0.01, 24, dtype=np.float32)
    volume[0, 0, 8] = 0.0
    targets = {name: TARGETS[name] for name in ("surface", "direction")}
    out = _predict(tmp_path, monkeypatch, volume, AxisWithPatchSign(), targets, (16, 16, 16), "seam")
    direction = out[2:5].astype(np.float32)
    assert direction[0, 8, 8, 12] > 0.99, f"x component at the seam {direction[0, 8, 8, 12]:.3f}"
    inner = direction[:, 4:12, 4:12, 4:20]  # away from the volume faces, where the weights vanish
    assert inner[0].min() > 0.99 and np.abs(inner[1:]).max() == 0


class ConstantScores(Oracle):
    """Oracle heads, with class 1 scoring 0.5 and class 0 scoring 0 everywhere."""

    def forward(self, x):
        out = super().forward(x)
        s = x[:, :1].float()
        out["surface"] = torch.cat([torch.zeros_like(s), torch.full_like(s, 0.5)], dim=1)
        return out


def test_predict_multiclass_classes_come_from_segmentation_heads(tmp_path, monkeypatch):
    """--blend_in_memory --mode multiclass with default TTA: the restored normal components
    (up to 1.0) are not class scores; class 1 (score 0.5) wins everywhere."""
    from vesuvius.models.run.finalize_outputs import FinalizeConfig

    targets = {name: TARGETS[name] for name in ("surface", "normals")}
    out = _predict(tmp_path, monkeypatch, _volume((32, 32, 32)).numpy(), ConstantScores(), targets,
                   (16, 16, 16), "multiclass",
                   in_memory_blend={"finalize_config": FinalizeConfig(mode="multiclass")})
    assert out.shape == (3, 32, 32, 32)  # softmax class 0, class 1, argmax
    assert np.unique(out[-1]).tolist() == [1]


def _seg_and_normals_info(seg_channels):
    info = _target_info({"surface": {"out_channels": seg_channels}, "normals": {"out_channels": 3}})
    info["normals"]["auxiliary_task"] = True
    return info


def test_multiclass_finalization_skips_auxiliary_heads():
    """Three class scores and a normal, no TTA: the class is the best score, not the normal's x."""
    from vesuvius.models.run.finalize_outputs import (FinalizeConfig, apply_finalization,
                                                      compute_finalized_shape)

    config = FinalizeConfig(mode="multiclass", is_multi_task=True, target_info=_seg_and_normals_info(3))
    logits = np.zeros((6, 1, 1, 2), dtype=np.float32)
    logits[:, 0, 0, 0] = [0.0, 0.5, 0.2, 1.0, 0.0, 0.0]
    logits[:, 0, 0, 1] = [0.3, 0.0, 0.1, 0.0, 0.0, -1.0]
    out, is_empty = apply_finalization(logits, 6, config)
    assert not is_empty
    assert out.shape == compute_finalized_shape((1, 1, 2), 6, config) == (4, 1, 1, 2)
    assert out[-1].ravel().tolist() == [1, 0]


def test_finalize_logits_multiclass_skips_auxiliary_heads(tmp_path):
    """The staged path (blend_logits, then finalize_outputs) reads the heads from the logits attrs."""
    from vesuvius.models.run.finalize_outputs import finalize_logits

    logits = np.zeros((5, 4, 4, 4), dtype=np.float32)
    logits[1], logits[2] = 0.5, 1.0  # class 1 beats class 0; the normal's x is larger still
    writer = ZarrWriter.create(str(tmp_path / "logits.zarr"), shape=logits.shape, chunks=(5, 4, 4, 4),
                               dtype=np.float32)
    writer[:] = logits
    writer.attrs["is_multi_task"] = True
    writer.attrs["target_info"] = _seg_and_normals_info(2)
    finalize_logits(str(tmp_path / "logits.zarr"), str(tmp_path / "final.zarr"), mode="multiclass",
                    num_workers=1, verbose=False)
    out = np.asarray(open_zarr(str(tmp_path / "final.zarr"))[:])
    assert out.shape == (3, 4, 4, 4)
    assert np.unique(out[-1]).tolist() == [1]


# --- The checkpoint's losses decide whether a direction is an axis (no sign) ---------------------

def _shipped_nearest_component():
    """The nearest_component head as the shipped nearest_component_3d.yaml trains it: the direction
    with SignInvariantCosineLoss, the trailing distance with SmoothL1Loss."""
    path = Path(vesuvius.models.__file__).parent / "configuration" / "aux_tasks" / "nearest_component_3d.yaml"
    losses = yaml.safe_load(path.read_text())["auxiliary_tasks"]["nearest_component"]["losses"]
    return {"out_channels": 4, "auxiliary_task": True, "task_type": "nearest_component", "losses": losses}


def _nearest_component(*direction_losses):
    """The shipped head with other losses on the direction channels (the distance keeps SmoothL1Loss)."""
    head = _shipped_nearest_component()
    distance = [loss for loss in head["losses"] if loss["kwargs"]["channels"] == "last"]
    head["losses"] = [{"name": "ChannelSelectLoss", "weight": 1.0,
                       "kwargs": {"channels": "all_but_last", "base": {"name": name, "kwargs": {}}}}
                      for name in direction_losses] + distance
    return head


@pytest.mark.parametrize(("head", "kind"), [
    (_shipped_nearest_component(), "axial"),  # SignInvariantCosineLoss on the direction
    (_nearest_component("CosineSimilarityLoss"), "vector"),
    # orientation supervised by any loss: signed, also next to a sign-invariant loss
    (_nearest_component("SignInvariantCosineLoss", "CosineSimilarityLoss"), "vector"),
    (_nearest_component("SignInvariantCosineLoss", "MSELoss"), "vector"),
    ({**TARGETS["nearest"]}, "vector"),  # no losses configured: the task's default
    ({**TARGETS["direction"], "losses": [{"name": "SignInvariantCosineLoss"}]}, "axial"),
    ({**TARGETS["direction"], "losses": [{"name": "CosineSimilarityLoss"}]}, "vector"),
    ({**TARGETS["direction"]}, "axial"),
])
def test_direction_kind_comes_from_the_training_losses(head, kind):
    targets = {"surface": TARGETS["surface"], "h": head}
    groups = tta.tta_channel_groups(targets, _target_info(targets), 3)
    assert [g["kind"] for g in groups] == [kind]
    assert sorted(groups[0]["channels"].values()) == [2, 3, 4]  # a nearest_component distance (5) stays scalar


class SameSignXAxis(torch.nn.Module):
    """Reports the x axis with the same sign in every mirrored frame, at distance 2. Under the
    shipped SignInvariantCosineLoss that is a perfect direction wherever x is the true axis."""

    def forward(self, x):
        s = x[:, :1].float()
        one = torch.ones_like(s)
        return {"surface": torch.cat([-s, s], dim=1),
                "nearest": torch.cat([one, 0 * one, 0 * one, 2 * one], dim=1)}


def test_predict_keeps_a_sign_invariant_nearest_component_axis(tmp_path, monkeypatch):
    """Default mirroring with the shipped nearest_component losses keeps the axis and the
    distance: (1, 0, 0, 2). As a signed vector the mirrored variants cancelled to (0, 0, 0, 2)."""
    targets = {"surface": TARGETS["surface"], "nearest": _shipped_nearest_component()}
    out = _predict(tmp_path, monkeypatch, _volume((32, 32, 32)).numpy(), SameSignXAxis(), targets,
                   (16, 16, 16), "nearest").astype(np.float32)
    nearest = out[2:6]
    assert nearest[:, 16, 16, 16].tolist() == [1.0, 0.0, 0.0, 2.0], nearest[:, 16, 16, 16]
    inner = nearest[:, 4:28, 4:28, 4:28]  # away from the volume faces, where the weights vanish
    assert np.abs(inner - np.array([1.0, 0, 0, 2.0]).reshape(4, 1, 1, 1)).max() < 1e-3


@pytest.mark.parametrize("losses", [("CosineSimilarityLoss",), ("SignInvariantCosineLoss", "CosineSimilarityLoss")])
def test_predict_never_reverses_a_signed_nearest_component(tmp_path, monkeypatch, losses):
    """With a signed loss on the direction (alone or next to a sign-invariant one), TTA and blending
    keep the predicted orientation, also where it points out of the half-space axes are written in."""
    targets = {"surface": TARGETS["surface"], "nearest": _nearest_component(*losses)}
    volume = _volume((32, 32, 32)).numpy()
    with_tta = _predict(tmp_path, monkeypatch, volume, Oracle(), targets, (16, 16, 16), "tta").astype(np.float32)
    without = _predict(tmp_path, monkeypatch, volume, Oracle(), targets, (16, 16, 16), "no_tta",
                       do_tta=False).astype(np.float32)
    mask = _defined(torch.from_numpy(volume)[None, None]).numpy()
    mask[[0, -1]] = mask[:, [0, -1]] = mask[:, :, [0, -1]] = False
    assert np.abs(with_tta[2:6] - without[2:6])[:, mask].max() < 1e-2
    # the Oracle's direction points inwards: a good part of it lies outside the half-space
    side = 4 * with_tta[2] + 2 * with_tta[3] + with_tta[4]
    assert (side[mask] < -0.5).mean() > 0.3
