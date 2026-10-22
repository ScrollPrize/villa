import pytest
import torch
import os

from vesuvius.neural_tracing.fiber_trace_3d.ema import ModelEMA
from vesuvius.neural_tracing.fiber_trace_3d.training_state import cpu_clone


def test_ema_reference_and_roundtrip():
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, [0.99, 0.999], start_step=100)
    initial = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
    ema.update(model)
    for track in ema.tracks.values():
        torch.testing.assert_close(track["model"]["weight"], initial + 2 * (1 - track["decay"]))
        assert track["updates"] == 1 and track["start_step"] == 100
    restored = ModelEMA(model, [0.99, 0.999])
    restored.load_state_dict(cpu_clone(ema.state_dict()))
    for name in ema.tracks:
        torch.testing.assert_close(restored.tracks[name]["model"]["weight"], ema.tracks[name]["model"]["weight"])


def test_ema_restores_identity_weights_buffers_modes_on_error():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    model.train()
    model[1].eval()
    ema = ModelEMA(model, [0.99])
    with torch.no_grad():
        model[1].weight.add_(1)
        model[0].num_batches_tracked.add_(7)
    before = cpu_clone(model.state_dict())
    parameters = list(model.parameters())
    with pytest.raises(RuntimeError, match="injected"):
        with ema.apply(model, "ema_0p99"):
            assert not model.training
            raise RuntimeError("injected")
    assert model.training and not model[1].training
    assert all(old is new for old, new in zip(parameters, model.parameters()))
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("decays", [[0], [1], [float("nan")], [True], [0.99, 0.99]])
def test_ema_invalid_decays(decays):
    with pytest.raises(ValueError):
        ModelEMA(torch.nn.Linear(1, 1), decays)


def test_ema_handoff_and_export_resume_guard():
    from vesuvius.neural_tracing.fiber_trace_3d.training_state import (
        capture_handoff, restore_handoff, restore_model_optimizer, replica_signature)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = ModelEMA(model, [0.99])
    ema.update(model)
    state = capture_handoff(model=model, optimizer=optimizer, progress={"step": 1}, ema=ema)
    signature = replica_signature(state)
    ema.update(model)
    restore_handoff(state, model=model, optimizer=optimizer)
    ema.load_state_dict(state["ema"])
    assert ema.tracks["ema_0p99"]["updates"] == 1
    assert replica_signature(state) == signature
    with pytest.raises(ValueError, match="inference-only"):
        restore_model_optimizer({"model": model.state_dict(), "snapshot_kind": "ema"},
                                model=model, optimizer=optimizer)


def test_historical_source_does_not_change_training_and_mask_is_central():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d.historical_evaluation import (
        historical_test_config, historical_presence_mask)
    raw = {"batch_size": 16, "sampling_strategy": "scroll_balanced_arclength",
           "augment_enabled": True, "historical_test_datasets": [{"fiber_glob": "old.json"}]}
    old = historical_test_config(raw)
    assert old["batch_size"] == 8 and old["sampling_strategy"] == "control_points"
    assert raw["batch_size"] == 16 and raw["augment_enabled"]
    batch = SimpleNamespace(presence_target=torch.zeros(2, 1, 8, 8, 8),
                            volume=torch.zeros(2, 1, 8, 8, 8), valid_mask=None)
    config = SimpleNamespace(presence_negative_edge_margin_voxels=None, augment_shift_zyx=(2, 2, 2))
    mask = historical_presence_mask(batch, config)
    assert mask.sum() == 2 * 4**3
    assert mask[:, :, 2:6, 2:6, 2:6].all()


def test_evaluation_variants_preserve_raw_metrics_rng_and_parameters():
    from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
    from vesuvius.neural_tracing.fiber_trace_3d import train
    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8], "batch_size": 2,
                                  "augment_enabled": False})
    model = torch.nn.Sequential(torch.nn.Conv3d(1, 7, 1), torch.nn.Sigmoid())
    ema = ModelEMA(model, [.99, .999])
    with torch.no_grad():
        model[0].weight.add_(.1)
    options = dict(device=torch.device("cpu"), start_sample_index=0, sample_count=4,
                   direction_weight=10, presence_weight=1)
    expected = train._evaluate_dense_loss_rows(model, loader, **options)
    state = cpu_clone(model.state_dict())
    rng = torch.get_rng_state().clone()
    actual = train._evaluate_dense_loss_rows(model, loader, ema=ema, microbatch_size=1, **options)
    assert torch.equal(rng, torch.get_rng_state())
    for (_, row), (_, reference) in zip(actual, expected):
        for key, value in reference.items():
            assert row[key] == pytest.approx(value, abs=1e-5)
        assert "ema_0p99/total" in row and "ema_0p999/angle_mean_deg" in row
    for key, tensor in state.items():
        torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)


def test_raw_and_ema_snapshots_are_distinct_and_loadable(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import train
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = ModelEMA(model, [.99])
    with torch.no_grad():
        model.weight.add_(1)
    ema.update(model)
    options = dict(model=model, optimizer=optimizer, step=1, config={}, metric=.5, ema=ema)
    raw_path, ema_path = tmp_path / "current.pt", tmp_path / "current_ema_0p99.pt"
    train._save_snapshot(raw_path, **options)
    train._save_snapshot(ema_path, **options, ema_name="ema_0p99")
    raw = torch.load(raw_path, weights_only=True)
    averaged = torch.load(ema_path, weights_only=True)
    assert "optimizer" in raw and "ema" in raw
    assert "optimizer" not in averaged and averaged["snapshot_kind"] == "ema"
    torch.testing.assert_close(averaged["model"]["weight"], ema.tracks["ema_0p99"]["model"]["weight"])
    restored = torch.nn.Linear(2, 1)
    train._load_snapshot(ema_path, model=restored)
    with pytest.raises(ValueError, match="inference-only"):
        train._load_snapshot(ema_path, model=model, optimizer=optimizer)


@pytest.mark.skipif(os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1",
                    reason="requires manager-authorized launcher")
def test_cuda_ema_overhead_and_bf16():
    import json
    import statistics
    import time
    from pathlib import Path
    from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
    torch.cuda.set_per_process_memory_fraction(16 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    config = json.loads((Path(__file__).parents[2] /
        "src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_16x8.json").read_text())
    torch.manual_seed(11)
    model = build_fiber_trace_3d_model(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    ema = ModelEMA(model, [.99, .999])
    inputs = torch.randn(4, 1, 128, 128, 128, device="cuda")
    times = {"raw": [], "two_ema": []}
    update_times = []
    for iteration in range(26):
        enabled = bool(iteration % 2)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inputs)
            loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        if enabled:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            ema.update(model)
            end.record()
        torch.cuda.synchronize()
        if iteration >= 6:
            times["two_ema" if enabled else "raw"].append(time.perf_counter() - start)
            if enabled:
                update_times.append(begin.elapsed_time(end))
    summary = {name: {"mean_ms": statistics.mean(values) * 1000,
        "p50_ms": statistics.median(values) * 1000,
        "min_ms": min(values) * 1000, "max_ms": max(values) * 1000,
        "samples_per_second_1gpu": 4 / statistics.mean(values)} for name, values in times.items()}
    assert all(t.dtype == torch.float32 for t in ema.tracks["ema_0p99"]["model"].values() if t.is_floating_point())
    print("EMA_PROFILE", json.dumps({"gpus": 1, "samples_per_gpu": 4, "patch": 128,
        "precision": "bf16", "measured_steps_per_case": 10, "results": summary,
        "two_ema_update_mean_cuda_ms": statistics.mean(update_times),
        "shadow_bytes": sum(t.numel() * t.element_size() for track in ema.tracks.values()
                            for t in track["model"].values())}), flush=True)
