"""Real annotated Paris4 crops through the queued trainer, heads and checkpoints."""
import json
import os
from pathlib import Path

import pytest
import torch

from tests.neural_tracing.test_resumable_training import _bounded_config, _finish


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1", reason="manager grant required"))])
@pytest.mark.parametrize("paired", [False, True])
def test_real_annotated_training_updates_and_saves(tmp_path, monkeypatch, device, paired):
    raw, path = _bounded_config(tmp_path, device=device, depth=2 if device.startswith("cuda") else 0)
    for key in ("datasets", "test_datasets"):
        for dataset in raw[key]:
            dataset.update(dense_supervision=True, scroll_id="PHercParis4")
    # Refinement enforces parent-disjoint splits. Use a second real piece,
    # rather than relabeling the training parent as a test parent.
    source = Path(raw["datasets"][0]["fiber_paths"][0])
    parent = source.name.split("_run")[0]
    other = next(p for p in sorted(source.parent.glob("*.json"))
                 if p.name.split("_run")[0] != parent)
    raw["test_datasets"][0]["fiber_paths"] = [str(other)]
    raw.update(sampling_strategy="scroll_balanced_arclength",
               presence_target_mode="anisotropic_gaussian",
               presence_radius_voxels=1., presence_gaussian_width_ratio=4.,
               annotations={"normal_weight": 10., "width_weight": .01, "hv_weight": .01})
    raw["model"].update(features_per_stage=[2, 4, 8], normalization="group", output_stride=2)
    raw["training"].update(max_steps=2, optimizer_global_batch_size=4, test_control_points=2, test_start_sample_index=0,
        test_interval=1, checkpoint_interval=1, sample_vis_interval=0, test_sample_vis_interval=0,
        mixed_precision="bf16" if device.startswith("cuda") else "off", ema_decays=[.99, .999])
    if paired:
        raw["embedding"] = dict(enabled=True, channels=16, pairs_at_128=128,
            objective="local_contrastive", paired_patches=True, short_range_channels=8)
    else:
        raw["refinement"].update(enabled=True, iterations=2, direction_modulation=True,
                                 width_modulation=True, width_enabled=True)
    path.write_text(json.dumps(raw))
    monkeypatch.setenv("VILLA_FIBER_REFINEMENT_STATE", str(tmp_path / "refinement"))
    state, writer, boundaries = _finish(path, physical=(2,))
    assert state["progress"]["sample_frontier"] == 8
    assert boundaries[-1] == (2, "update")
    assert state["optimizer"]["state"]
    assert state["ema"]["tracks"].keys() == {"ema_0p99", "ema_0p999"}
    for tag in ("train/loss_total", "test/loss_total", "test/loss_normal", "test/loss_width", "test/loss_hv"):
        rows = [value for (name, _), value in writer.scalars.items() if name == tag]
        assert rows and all(torch.isfinite(torch.tensor(rows))), tag
    snapshots = next((tmp_path / "runs").glob("*/snapshots"))
    assert (snapshots / "current.pt").is_file()
    assert (snapshots / "current_ema_0p999.pt").is_file()
    print(f"REAL_ANNOTATED_TRAINING device={device} paired={paired} samples=8; "
          "queued loading, Gaussian/auxiliary losses, backward, EMA/checkpoint")
