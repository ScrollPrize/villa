"""TTA has to transform the components of vector and tensor heads, not only their positions.

Mirroring the input over an axis negates the component of a direction along that axis;
transposing two axes swaps their components. The auxiliary heads (surface_normals,
nearest_component, inplane_direction: x y z components; structure_tensor: Jzz Jzy Jzx Jyy
Jyx Jxx) therefore need more than the spatial un-flip, or the TTA variants cancel.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vc_reader.reader import ZarrWriter, open_zarr
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
        # an eigenvector: the direction is defined, its sign is not (here: x component >= 0)
        direction = n * torch.where(n[:, :1] < 0, -1.0, 1.0)
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
