"""Real Paris4 regression for the existing resumable, bounded trainer."""

import json
import os
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.neural_tracing.test_training_queue_integration import (
    _config, _Writer, _observe_queue, _assert_workers_exited, _serial_test_rows,
)
from vesuvius.neural_tracing.fiber_trace_3d import train
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_resources import EffectiveBatchSampleIndex
from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader, config_from_mapping
from vesuvius.neural_tracing.fiber_trace_3d.training_state import cpu_clone, replica_signature


def _bounded_config(tmp_path, *, device="cpu", depth=0, mode=None):
    raw, path = _config(tmp_path, device=device, depth=depth, mode=mode)
    raw["training"].update(bounded_gradient_accumulation=True,
        optimizer_global_batch_size=5, initialization_seed=177,
        ddp_broadcast_buffers=False, test_control_points=5)
    # Three logical test groups, with a partial final group. Their normalization
    # and canonical IDs must not change when the physical microbatch changes.
    path.write_text(json.dumps(raw))
    return raw, path


def _assert_state_close(actual, expected, *, rtol=3e-5, atol=3e-6):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_close(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            _assert_state_close(a, e, rtol=rtol, atol=atol)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def _finish(path, *, physical, handoff=None):
    writer = _Writer()
    units = train.training_units(path, physical_microbatch_sizes=physical, handoff=handoff,
        summary_writer_factory=lambda *a, **k: writer)
    progress = []
    result = None
    try:
        for boundary in units:
            progress.append((boundary.step, boundary.phase))
            if boundary.done:
                result = boundary.capture()
    finally:
        units.close()
    assert writer.closed
    assert result is not None
    return result, writer, progress


@pytest.mark.parametrize("interrupt", ["initial_test", "update", "periodic_test", "complete"])
def test_real_paris4_resume_matches_full_effective_batch(tmp_path, monkeypatch, interrupt):
    # Fix the CPU convolution implementation as well as the CUDA algorithm;
    # oneDNN can select different reductions for different physical batches.
    with torch.backends.mkldnn.flags(enabled=False):
        _resume_equivalence(tmp_path, monkeypatch, interrupt=interrupt)


def _resume_equivalence(tmp_path, monkeypatch, *, interrupt, device="cpu", depth=0, mode=None):
    raw, path = _bounded_config(tmp_path, device=device, depth=depth, mode=mode)
    queues = _observe_queue(monkeypatch)
    # Keep this regression focused on state and scheduling; the queue integration
    # suite separately runs the actual contact-sheet renderer on these crops.
    sheets = []
    monkeypatch.setattr(train, "_write_3d_sample_sheet",
        lambda writer, tag, model, batch, step, **kw: sheets.append((tag, step)))
    random.seed(31)
    np.random.seed(32)
    reference, reference_writer, _ = _finish(path, physical=(5,))
    reference_rng = (random.random(), np.random.rand(), torch.rand(4))
    random.seed(31)
    np.random.seed(32)
    writer = _Writer()
    units = train.training_units(path, physical_microbatch_sizes=(2,),
        summary_writer_factory=lambda *a, **k: writer)
    captured = None
    try:
        for boundary in units:
            selected = ((interrupt == "initial_test" and boundary.phase == "test" and boundary.step == 0)
                or (interrupt == "update" and boundary.phase == "update" and boundary.step == 1)
                or (interrupt == "periodic_test" and boundary.phase == "test" and boundary.step == 2)
                or (interrupt == "complete" and boundary.done))
            if selected:
                captured = boundary.capture()
                break
    finally:
        units.close()
    assert captured is not None and writer.closed
    immutable = cpu_clone(captured)
    before_resume = len(queues)
    # Reconstruct with a different physical microbatch, but identical effective
    # batch/config. No disk snapshots or dataset edits are used for the handoff.
    resumed, resumed_writer, progress = _finish(path, physical=(1,), handoff=captured)
    _assert_state_close(captured, immutable, rtol=0, atol=0)
    _assert_state_close(resumed["model"], reference["model"])
    _assert_state_close(resumed["optimizer"], reference["optimizer"])
    actual_progress, expected_progress = dict(resumed["progress"]), dict(reference["progress"])
    assert actual_progress.pop("best_metric") == pytest.approx(expected_progress.pop("best_metric"),
                                                              rel=3e-5, abs=3e-6)
    assert actual_progress == expected_progress
    actual_rng = (random.random(), np.random.rand(), torch.rand(4))
    _assert_state_close(actual_rng, reference_rng, rtol=0, atol=0)
    assert resumed["progress"]["sample_frontier"] == 20
    assert progress[-1] == (4, "update")
    if interrupt in ("initial_test", "periodic_test"):
        assert len(captured["progress"]["pending_test"]["rows"]) == 1
        assert [phase for _, phase in progress[:2]] == ["test", "test"]
        assert ("test/loss_total", captured["progress"]["step"]) in resumed_writer.scalars
    else:
        assert captured["progress"]["pending_test"] is None
        assert not any(step == captured["progress"]["step"] and phase == "test"
                       for step, phase in progress)
    # Canonical train sample IDs are neither duplicated nor skipped on resume.
    admissions = [indices for q in queues[1:] for source, indices in q.admissions if source == "train"]
    consumed_after = [i for q in queues[before_resume:] for source, ids in q.admissions
                      if source == "train" for i in ids]
    assert consumed_after == list(range(captured["progress"]["step"] * 5, 20))
    assert admissions
    for queue in queues:
        _assert_workers_exited(queue)
        assert not queue._owned and not queue._ready
    for key, value in reference_writer.scalars.items():
        if key[0].startswith("test/") and key in resumed_writer.scalars:
            assert resumed_writer.scalars[key] == pytest.approx(value, rel=3e-5, abs=3e-6)


@pytest.mark.parametrize("physical", [1, 2])
@pytest.mark.parametrize("conditioned", [False, True])
def test_real_paris4_test_groups_keep_logical_normalization(tmp_path, physical, conditioned):
    raw, _ = _bounded_config(tmp_path)
    if conditioned:
        raw["model"]["conditioned_decoder_enabled"] = True
    model = train.build_fiber_trace_3d_model(raw)
    loader = FiberTrace3DLoader(config_from_mapping(
        train._make_test_loader_raw_config(raw, raw["training"])))
    options = dict(device=torch.device("cpu"), start_sample_index=5, sample_count=5,
        direction_weight=10., presence_weight=1.)
    expected = _serial_test_rows(model, loader, options)
    actual = _serial_test_rows(model, loader, {**options, "microbatch_size": physical})
    for (index, row), (ref_index, ref) in zip(actual, expected, strict=True):
        assert index == ref_index
        assert row == pytest.approx(ref, rel=3e-5, abs=3e-6)


def test_effective_sample_index_is_source_partition_with_empty_ranks():
    maps = [EffectiveBatchSampleIndex(2, (1, 1, 1), rank, 3) for rank in range(3)]
    assert [m.batch_sizes for m in maps] == [(), (1,), (1,)]
    assert [m(0) for m in maps if m.local_samples] == [6, 7]
    with pytest.raises(ValueError, match="zero-work"):
        maps[0](0)


def test_real_paris4_one_backward_per_microbatch_and_capture_can_continue(tmp_path, monkeypatch):
    _one_backward_and_continue(tmp_path, monkeypatch)


def _one_backward_and_continue(tmp_path, monkeypatch, *, device="cpu", depth=0):
    _, path = _bounded_config(tmp_path, device=device, depth=depth)
    backward_calls, sample_ids = [], []
    backward, forward = torch.autograd.backward, train._forward_loss_statistics

    def record_backward(*args, **kwargs):
        backward_calls.append(1)
        return backward(*args, **kwargs)

    def record_forward(model, batch, **kwargs):
        if kwargs.get("training_loss", True):
            sample_ids.extend(batch.stream_indices.tolist())
        return forward(model, batch, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", record_backward)
    monkeypatch.setattr(train, "_forward_loss_statistics", record_forward)
    monkeypatch.setattr(train, "_write_3d_sample_sheet", lambda *a, **kw: None)
    units = train.training_units(path, physical_microbatch_sizes=(2,))
    try:
        for boundary in units:
            if boundary.phase == "update":
                captured = boundary.capture()
                assert captured["progress"]["sample_frontier"] == boundary.step * 5
    finally:
        units.close()
    assert sample_ids == list(range(20))
    assert len(backward_calls) == 4 * 3


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager allocation")
def test_cuda_real_paris4_capture_resumes_geometry_prepass_on_same_iterator(tmp_path, monkeypatch):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    _one_backward_and_continue(tmp_path, monkeypatch, device="cuda:0", depth=2)


@pytest.mark.parametrize("change,match", [
    ({"bounded_gradient_accumulation": 1}, "must be a boolean"),
    ({"evaluation_microbatch_size": 0}, "positive integer"),
    ({"ddp_broadcast_buffers": True}, "broadcast_buffers"),
    ({"optimizer_global_batch_size": 0}, "positive integer"),
])
def test_invalid_bounded_settings_do_not_start_queue(tmp_path, monkeypatch, change, match):
    raw, path = _bounded_config(tmp_path)
    raw["training"].update(change)
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(train, "_make_training_sample_queue", lambda *a, **k: pytest.fail("started queue"))
    with pytest.raises(ValueError, match=match):
        list(train.training_units(path))


def test_real_paris4_open_ended_bounded_stream_can_be_closed(tmp_path, monkeypatch):
    raw, path = _bounded_config(tmp_path)
    raw.pop("test_datasets")
    raw["training"].update(max_steps=0, tensorboard_enabled=False, sample_vis_interval=0)
    path.write_text(json.dumps(raw))
    queues = _observe_queue(monkeypatch)
    units = train.training_units(path, physical_microbatch_sizes=(1,))
    try:
        assert next(units).step == 0
        boundary = next(units)
        assert boundary.step == 1 and not boundary.done
        assert boundary.capture()["progress"]["sample_frontier"] == 5
    finally:
        units.close()
    _assert_workers_exited(queues[0])


@pytest.mark.parametrize("interrupt,mode", [("update", None), ("periodic_test", "all")])
@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager allocation")
def test_cuda_real_paris4_async_resume_changed_microbatch(tmp_path, monkeypatch, interrupt, mode):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    # Use the same batch-shape-stable convolution backend on both sides of a
    # numerical equivalence test. Production optimized kernels stay unchanged.
    with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
        _resume_equivalence(tmp_path, monkeypatch, interrupt=interrupt,
                            device="cuda:0", depth=2, mode=mode)


def _bounded_distributed_worker(rank, path, rendezvous, out, interrupt_phase):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        # Spawn does not inherit the parent's cuDNN context-manager state.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
    # Advance one update, capture completed state, then reconstruct with a
    # different unequal physical partition. This uses the public loop entrypoint
    # and its actual process-group creation/cleanup, not a replacement trainer.
    units = train.training_units(path, physical_microbatch_sizes=(1, 3),
        distributed_init_method=rendezvous)
    handoff = None
    try:
        for boundary in units:
            if boundary.phase == interrupt_phase and boundary.step == (2 if interrupt_phase == "test" else 1):
                handoff = boundary.capture()
                signatures = [None, None]
                dist.all_gather_object(signatures, replica_signature(handoff))
                assert signatures[0] == signatures[1]
                torch.save(handoff, f"{out}.interrupted.{rank}.pt")
                break
    finally:
        units.close()
    assert handoff is not None
    units = train.training_units(path, physical_microbatch_sizes=(2, 1), handoff=handoff,
        distributed_init_method=rendezvous + "_resume")
    result = None
    try:
        for boundary in units:
            if boundary.done:
                result = boundary.capture()
    finally:
        units.close()
    assert result is not None
    torch.save(result, f"{out}.{rank}.pt")


def _distributed_bounded_equivalence(tmp_path, *, device, effective, interrupt_phase):
    raw, path = _bounded_config(tmp_path, device=device, depth=2 if device == "cuda" else 0)
    raw["training"].update(optimizer_global_batch_size=effective, tensorboard_enabled=False,
        test_control_points=5 if interrupt_phase == "test" else 1, sample_pipeline_workers=2)
    path.write_text(json.dumps(raw))
    # Actual full-batch production loop on one device, not a hand-coded loss.
    reference, _, _ = _finish(path, physical=(effective,))
    mp.spawn(_bounded_distributed_worker, args=(str(path),
        (tmp_path / "bounded_rendezvous").as_uri(), str(tmp_path / "result"), interrupt_phase), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path / f"result.{rank}.pt", map_location="cpu", weights_only=False)
        _assert_state_close(actual["model"], reference["model"], rtol=8e-5, atol=8e-6)
        _assert_state_close(actual["optimizer"], reference["optimizer"], rtol=8e-5, atol=8e-6)
        assert actual["progress"]["sample_frontier"] == effective * 4
    # Also shrink from two ranks to one using exactly that same captured state.
    interrupted = torch.load(tmp_path / "result.interrupted.0.pt", map_location="cpu", weights_only=False)
    single, _, _ = _finish(path, physical=(1,), handoff=interrupted)
    _assert_state_close(single["model"], reference["model"], rtol=8e-5, atol=8e-6)
    _assert_state_close(single["optimizer"], reference["optimizer"], rtol=8e-5, atol=8e-6)


@pytest.mark.parametrize("effective,interrupt_phase", [(1, "update"), (5, "update"), (5, "test")])
def test_real_paris4_bounded_distributed_resume_unequal_and_empty_ranks(tmp_path, effective, interrupt_phase):
    with torch.backends.mkldnn.flags(enabled=False):
        _distributed_bounded_equivalence(tmp_path, device="cpu", effective=effective,
                                         interrupt_phase=interrupt_phase)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager allocation")
@pytest.mark.parametrize("effective,interrupt_phase", [(1, "update"), (5, "update"), (5, "test")])
def test_cuda_real_paris4_bounded_distributed_resume_unequal_and_empty_ranks(tmp_path, effective, interrupt_phase):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    assert torch.cuda.device_count() >= 2
    with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
        _distributed_bounded_equivalence(tmp_path, device="cuda", effective=effective,
                                         interrupt_phase=interrupt_phase)
