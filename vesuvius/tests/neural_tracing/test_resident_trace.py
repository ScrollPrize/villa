"""Resident inference must not reload weights or mutate training state."""
import copy
import json
import multiprocessing as mp
import os
import queue
import threading
import time

import numpy as np
import pytest
import torch

from lasagna import tiled_predict3d as tiled
from vesuvius.neural_tracing.fiber_trace_3d.resident_trace import ResidentWorkers, serve_resident


class Adapter:
    def __init__(self):
        # Deliberately larger than pipe capacity to exercise startup ordering.
        self.config = {"inference": {"allow_tf32": False}, "padding": "x" * 524288}

    def load_model(self, **_kwargs):
        raise AssertionError("resident evaluation must not load a model")

    def run_tile_inference(self, model, tile, *, device):
        return model(tile)

    def product_tensors_from_output(self, output):
        return {"presence": output}


def _exercise(device):
    torch.manual_seed(37)
    model = torch.nn.Sequential(torch.nn.Conv3d(1, 1, 1), torch.nn.Dropout(0.3)).to(device)
    model.train()
    model[1].eval()  # Mixed module modes must also survive.
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 1, 4, 4, 4, device=device)).sum().backward()
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    parameters = [p.detach().clone() for p in model.parameters()]
    gradients = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    layouts, nbytes = tiled._packed_layouts((("input", (4, 4, 4), np.uint8),))
    outlayouts, outbytes = tiled._packed_layouts((("product:presence", (1, 4, 4, 4), np.float32),))
    inputs = tiled._create_shared_slots(1, nbytes)
    outputs = tiled._create_shared_slots(1, outbytes)
    parent, child = mp.get_context("spawn").Pipe()
    errors = []
    workers = None
    try:
        raw = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
        tiled._slot_array(inputs[0], layouts[0])[:] = raw
        ins = tuple(tiled._SharedSlotSpec(s.name, nbytes, layouts) for s in inputs)
        outs = tuple(tiled._SharedSlotSpec(s.name, outbytes, outlayouts) for s in outputs)
        work, events = queue.Queue(), queue.Queue()
        work.put((0, (0, 0, 0), 0, 0, ("presence",), np.dtype(np.uint8).str))
        def serve():
            try:
                assert child.recv() == "trace"
                args = child.recv()
                serve_resident(child, model, 0, args)
            except BaseException as error:
                errors.append(error)
            finally:
                child.close()
        thread = threading.Thread(target=serve)
        args = (Adapter(), None, 4, np.ones(4, np.float32), {"presence": 1}, 1,
                ins, outs, work, events, False, 2, True)
        workers = ResidentWorkers([parent], args, thread.start, threading.Event())
        deadline = time.monotonic() + 20
        while True:
            workers.check()
            event = events.get(timeout=max(.1, deadline-time.monotonic()))
            assert event[0] != "error", event
            if event[0] == "result":
                break
        workers.close()  # No work sentinel: shutdown must wake an idle worker.
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not errors
        assert model.training and not model[1].training
        assert torch.equal(rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        for before, grad, after in zip(parameters, gradients, model.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            torch.testing.assert_close(grad, after.grad, rtol=0, atol=0)
        for key, state in saved["state"].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
        with torch.no_grad():
            expected = model[0](torch.from_numpy(raw).float().to(device)[None, None] / 255)[0].cpu().numpy()
        np.testing.assert_allclose(tiled._slot_array(outputs[0], outlayouts[0]), expected, rtol=1e-6)
        # A training step still works on the same optimizer and module objects.
        optimizer.zero_grad()
        model(torch.ones(1, 1, 4, 4, 4, device=device)).sum().backward()
        optimizer.step()
    finally:
        if workers is not None:
            workers.close()
        parent.close()
        for slot in inputs + outputs:
            slot.close()
            slot.unlink()


def test_resident_preserves_model_optimizer_rng_and_modes():
    _exercise(torch.device("cpu"))


def test_resident_oom_reduces_batch_not_tile_geometry():
    class SmallBudget(Adapter):
        def run_tile_inference(self, model, tile, *, device):
            if tile.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("test budget")
            return tile * 2
    tile = torch.arange(4 * 64).reshape(4, 1, 4, 4, 4)
    actual = tiled._resident_tile_inference(SmallBudget(), None, tile, torch.device("cpu"))
    torch.testing.assert_close(actual, tile * 2, rtol=0, atol=0)


def test_resident_single_tile_oom_is_explicit():
    class NoBudget(Adapter):
        def run_tile_inference(self, *args, **kwargs):
            raise torch.cuda.OutOfMemoryError("test budget")
    with pytest.raises(RuntimeError, match="cannot fit one inference tile"):
        tiled._resident_tile_inference(NoBudget(), None, torch.ones(1, 1, 4, 4, 4), torch.device("cpu"))


def test_training_trace_calls_shared_functions_without_python_subprocess(monkeypatch, tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import training_trace_benchmark as module
    calls = []
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"snapshot")
    template = tmp_path / "template.json"
    template.write_text("{}")
    job = {"inherited_manager_grant": True, "gpu_grant": {"GPU-resident": 100},
           "devices": "0", "checkpoint": str(checkpoint),
           "checkpoint_sha256": module.sha256_file(checkpoint),
           "template": str(template), "output_root": str(tmp_path),
           "split_root": str(tmp_path), "native_binary": "native", "samples_seen": 128}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    factory, cancel = object(), threading.Event()
    monkeypatch.setattr(module, "build_workload", lambda *a, **k: {})
    monkeypatch.setattr(module, "write_benchmark_identity", lambda *a: {})
    monkeypatch.setattr(module, "prepare", lambda workload: calls.append("prepare"))
    def run(workload, **kwargs):
        assert kwargs["worker_factory"] is factory
        assert kwargs["cancel_event"] is cancel
        calls.append("run")
        (tmp_path / "full").mkdir()
        (tmp_path / "full/summary.json").write_text("{}")
    monkeypatch.setattr(module, "run_prepared", run)
    monkeypatch.setattr(module, "directional_restart_metrics", lambda *a, **k: {"restarts_combined": 3})
    monkeypatch.setattr(module, "cleanup_prediction_cache", lambda *a: calls.append("cleanup"))
    monkeypatch.setattr(module, "write_completion", lambda *a, **k: calls.append("complete"))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned Python evaluator"))
    result = module.run_job(path, worker_factory=factory, cancel_event=cancel)
    assert result["restarts_combined"] == 3
    assert calls == ["prepare", "run", "cleanup", "complete"]


@pytest.mark.skipif(not os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), reason="manager GPU grant required")
def test_cuda_resident_preserves_training_state():
    # The approved test launcher supplies UUIDs only inside its manager lease.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["FIBER_COOPERATIVE_TEST_GPUS"]
    torch.cuda.set_per_process_memory_fraction(0.10, 0)
    _exercise(torch.device("cuda:0"))


def test_resident_cancel_does_not_consume_more_work():
    from vesuvius.neural_tracing.fiber_trace_3d.resident_trace import _RemoteWork
    parent, child = mp.get_context("spawn").Pipe()
    work, events = queue.Queue(), queue.Queue()
    cancel = threading.Event()
    cancel.set()
    consumed = []
    def serve():
        assert child.recv() == "trace"
        child.recv()
        consumed.append(_RemoteWork(child).get())
        child.send(("done", None))
    thread = threading.Thread(target=serve)
    work.put("must remain queued")
    args = (None, None, None, None, None, None, None, None, work, events, False, 1, False)
    workers = ResidentWorkers([parent], args, thread.start, cancel)
    workers.close()
    thread.join(2)
    assert consumed == [None]
    assert work.get_nowait() == "must remain queued"
    parent.close()
    child.close()


def exercise_cohort(cohort, wait_until):
    """Use actual training processes, then allow the caller to resume/compare."""
    from vesuvius.neural_tracing.fiber_trace_3d.cooperative_maintenance import MaintenanceMailbox
    cohort.maintenance_mailbox = MaintenanceMailbox()
    request = {"kind": "reviewed_trace", "step": 1, "samples_seen": 5}
    cohort.maintenance_mailbox.publish(request)
    assert cohort.begin_maintenance()
    pids = list(cohort.gpu.processes)
    layout, size = tiled._packed_layouts((("input", (8, 8, 8), np.uint8),))
    outlayout, outsize = tiled._packed_layouts((("product:presence", (7, 8, 8, 8), np.float32),))
    inputs, outputs = tiled._create_shared_slots(1, size), tiled._create_shared_slots(1, outsize)
    workers = None
    try:
        tiled._slot_array(inputs[0], layout[0])[:] = 123
        ins = tuple(tiled._SharedSlotSpec(s.name, size, layout) for s in inputs)
        outs = tuple(tiled._SharedSlotSpec(s.name, outsize, outlayout) for s in outputs)
        work, events = queue.Queue(), queue.Queue()
        work.put((0, (0, 0, 0), 0, 0, ("presence",), np.dtype(np.uint8).str))
        args = (Adapter(), None, 8, np.ones(8, np.float32), {"presence": 1}, 1,
                ins, outs, work, events, False, 1, True)
        workers = cohort.resident_worker_factory(threading.Event())(args, len(pids))
        received = []
        def has_result():
            workers.check()
            while not events.empty():
                received.append(events.get_nowait())
            assert not any(row[0] == "error" for row in received), received
            return any(row[0] == "result" for row in received)
        wait_until(has_result)
        workers.close()
        wait_until(lambda: cohort.gpu.idle)
        assert np.isfinite(tiled._slot_array(outputs[0], outlayout[0])).all()
        assert cohort.gpu.processes == pids
        cohort.maintenance_mailbox.complete(request)
        cohort.finish_maintenance()
        assert cohort.idle
    finally:
        if workers is not None:
            workers.close()
        for slot in inputs + outputs:
            slot.close()
            slot.unlink()
