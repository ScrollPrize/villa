"""Worker transport and real-data training through a persistent CPU owner."""

import json
import os
import queue
import threading
import time

import pytest
import torch
import torch.multiprocessing as mp

from tests.neural_tracing.test_resumable_training import _bounded_config, _finish, _assert_state_close
from vesuvius.neural_tracing.fiber_trace_3d import train
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_resources import EffectiveBatchSampleIndex
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_worker import QueueBatchSource
from vesuvius.neural_tracing.fiber_trace_3d.loader import config_from_mapping
from vesuvius.neural_tracing.fiber_trace_3d.sample_broker import CanonicalSampleBroker


def _consumer(path, microbatch, start, state, batches, received, test_connection, control, owner_pid):
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.allow_tf32 = False
    source = QueueBatchSource(batches, received, owner_pid,
        EffectiveBatchSampleIndex(5, (microbatch,), 0, start), test_connection)

    def forbidden(*a, **k):
        raise AssertionError("external consumer constructed a replacement loading pool")

    train._make_training_sample_queue = forbidden
    units = train.training_units(path, handoff=state, batch_source=source,
                                 physical_microbatch_sizes=(microbatch,))
    try:
        for boundary in units:
            control.send(("boundary", boundary.step, boundary.phase, boundary.done))
            command = control.recv()
            if command == "capture":
                control.send(("state", boundary.capture()))
                # Tensor FD transfer must complete before this process exits.
                assert control.recv() == "state-received"
                break
            assert command == "next"
    finally:
        source.close()
        units.close()
        test_connection.close()
        control.close()


def _drive_consumer(broker, path, *, microbatch, state=None, stop_phase=None):
    """Small owner test harness for the existing internal transport messages.

    This is not a manager/cohort implementation. It gives the actual production
    trainer batches from the real broker and acknowledges its source protocol.
    """
    context = mp.get_context("spawn")
    batches = context.Queue(maxsize=2)
    received = context.Value("q", 0)
    test, child_test = context.Pipe()
    control, child_control = context.Pipe()
    start = broker.frontier // 5
    mapping = EffectiveBatchSampleIndex(5, (microbatch,), 0, start)
    process = context.Process(target=_consumer, args=(str(path), microbatch, start,
        state, batches, received, child_test, child_control, os.getpid()))
    sent = ordinal = offset = 0
    pending = None
    captured = None
    stopping = False
    drained = False
    deadline = time.monotonic() + 60
    process.start()
    child_test.close()
    child_control.close()
    try:
        while captured is None:
            assert time.monotonic() < deadline, "broker-backed consumer stalled"
            broker.poll()
            if test.poll():
                kind, step, ids = test.recv()
                if kind == "test":
                    assert pending is None
                    pending = (step, ids)
                elif kind == "release-test":
                    broker.release_test(step, ids)
                else:
                    assert kind in ("begin-test", "end-test")
                    getattr(broker, "begin_test" if kind == "begin-test" else "end_test")(step)
                    test.send((kind, step))
            if pending is not None:
                step, ids = pending
                batch = broker.test_batch(step, ids)
                if batch is not None:
                    test.send(batch)
                    pending = None
            if control.poll():
                message = control.recv()
                if message[0] == "state":
                    captured = message[1]
                    control.send("state-received")
                    continue
                _, step, phase, done = message
                if step * 5 > broker.frontier:
                    broker.commit(step)
                selected = ((stop_phase == "update" and step == 1 and phase == "update")
                    or (stop_phase == "test" and step == 2 and phase == "test"))
                stopping = done or selected
                control.send("capture" if stopping else "next")
            if not stopping:
                take = mapping.batch_sizes[ordinal % len(mapping.batch_sizes)]
                ids = tuple(mapping(offset + i) for i in range(take))
                if ids[-1] < broker.end_index:
                    batch = broker.batch(ids)
                    if batch is not None:
                        try:
                            batches.put_nowait(batch)
                        except queue.Full:
                            pass
                        else:
                            ordinal += 1
                            offset += take
                            sent += 1
            if not process.is_alive() and not control.poll():
                raise AssertionError(f"consumer exited without state: {process.exitcode}")
            time.sleep(.002)
        process.join(15)
        assert process.exitcode == 0
        # Join the old publisher before another consumer can reuse its retained
        # CPU slots. Items not consumed by the old worker are speculative only.
        for _ in range(sent - received.value):
            batches.get(timeout=2)
        drained = True
        return captured, process.pid
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        test.close()
        control.close()
        if not drained:
            batches.cancel_join_thread()
        batches.close()
        if drained:
            batches.join_thread()
        process.close()


def _persistent_training_equivalence(tmp_path, *, phase, device="cpu", depth=0, mode=None):
    raw, path = _bounded_config(tmp_path, device=device, depth=depth, mode=mode)
    raw["training"].update(tensorboard_enabled=False, sample_vis_interval=0, test_sample_vis_interval=0)
    path.write_text(json.dumps(raw))
    reference, _, _ = _finish(path, physical=(5,))
    broker = CanonicalSampleBroker.create(config_path=path, loader_config=config_from_mapping(raw),
        start_step=0, max_steps=4, effective_batch_size=5, workers=2,
        slot_count=5, initial_in_flight=2, defer_cuda_value_augmentation=mode)
    pids = tuple(p.pid for p in broker.pipeline.processes)
    try:
        broker.set_worker_limit(2)
        handoff, first = _drive_consumer(broker, path, microbatch=2, stop_phase=phase)
        assert handoff["progress"]["sample_frontier"] == broker.frontier
        broker.set_worker_limit(0)
        while not broker.workers_within_limit():
            broker.poll()
            time.sleep(.002)
        assert tuple(p.pid for p in broker.pipeline.processes) == pids
        broker.set_worker_limit(1)
        actual, second = _drive_consumer(broker, path, microbatch=1, state=handoff)
        assert first != second
        assert tuple(p.pid for p in broker.pipeline.processes) == pids
        _assert_state_close(actual["model"], reference["model"])
        _assert_state_close(actual["optimizer"], reference["optimizer"])
        assert actual["progress"]["sample_frontier"] == broker.frontier == 20
        assert broker.test_step is None
        assert not broker.queue._owned and not broker.queue._ready
    finally:
        broker.close()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("phase", ["update", "test"])
def test_real_paris4_persistent_workers_survive_consumer_replacement(tmp_path, phase):
    with torch.backends.mkldnn.flags(enabled=False):
        _persistent_training_equivalence(tmp_path, phase=phase)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager reservation")
@pytest.mark.parametrize("phase,mode", [("update", None), ("test", "all")])
def test_cuda_real_persistent_cpu_owner_survives_async_gpu_replacement(tmp_path, phase, mode):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    with torch.backends.cudnn.flags(enabled=False, allow_tf32=False):
        _persistent_training_equivalence(tmp_path, phase=phase, device="cuda:0", depth=2, mode=mode)


def test_external_source_requires_target_only_accumulation(tmp_path, monkeypatch):
    raw = {"training": {"bounded_gradient_accumulation": False}}
    monkeypatch.setattr(train, "_load_raw_config", lambda _: raw)
    monkeypatch.setattr(train, "load_config", lambda _: None)
    with pytest.raises(ValueError, match="external canonical"):
        next(train.training_units(tmp_path / "unused.json", batch_source=object()))


def test_real_worker_transport_rejects_noncanonical_sample_ids(tmp_path):
    from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader
    loader = _real_paris4_loader(tmp_path)
    batches = queue.Queue()
    batches.put(loader.load_batch(5))
    received = mp.get_context("spawn").Value("q", 0)
    source = QueueBatchSource(batches, received, os.getppid(), EffectiveBatchSampleIndex(5, (2,), 0))
    try:
        with pytest.raises(RuntimeError, match="canonical training"):
            next(source)
        assert received.value == 1
    finally:
        source.close()


def test_worker_detects_exited_supervisor_without_waiting():
    source = QueueBatchSource(queue.Queue(), None, -1, EffectiveBatchSampleIndex(5, (1,), 0))
    with pytest.raises(RuntimeError, match="supervisor exited"):
        next(source)
    source.close()


@pytest.mark.parametrize("kind", ["train", "test", "boundary"])
def test_waiting_worker_source_is_cancellable(kind):
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    source = QueueBatchSource(queue.Queue(), context.Value("q", 0), os.getppid(),
        EffectiveBatchSampleIndex(5, (1,), 0), child)
    errors = []

    def wait():
        try:
            if kind == "train":
                next(source)
            elif kind == "test":
                next(source.test_batches(step=0, global_batch_indices=(0,), batch_size=1, sample_count=1))
            else:
                source.begin_test(0)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        if kind != "train":
            assert parent.poll(2)
            parent.recv()
        source.close()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], (StopIteration, RuntimeError))
    finally:
        source.close()
        thread.join(2)
        parent.close()
        child.close()
