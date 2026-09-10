"""No CUDA imports or driver probes: fake inventory and tiny CPU processes."""

import os
import socket
import time
import uuid

import pytest

from villa_gpu_manager.cohort import ProcessCohort, _Channel
from villa_gpu_manager.engine import Engine
from villa_gpu_manager.policy import empty
from villa_gpu_manager.supervisor import Supervisor


GRANT = {"cpu_cores": 2, "gpus": {"GPU-fake-a": 100, "GPU-fake-b": 100}}
REQUEST = dict(max_cpu_cores=2, max_gpus=2, min_vram_bytes_per_gpu=50,
               max_vram_bytes_per_gpu=100, allowed_gpu_uuids=[], priority=0, exclusive=False)


@pytest.fixture(autouse=True)
def cpu_only_native_limits(monkeypatch):
    # Include the nested Torch-CPU pickle fixture when this suite is invoked
    # directly, without the training project's environment wrapper.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")


class LocalClient:
    def __init__(self):
        self.engine = Engine(GRANT, controller_uids=[os.getuid()], inspect=lambda p, uid: "alive", discover=lambda p, uid: [])
        self.epoch = None
        self.task = None
        self.key = str(uuid.uuid4())
        self.task_id = self.key
        self.offline = False
        self.lose_report = False
        self.lose_finish = False
        self.reports = []

    def connect(self):
        if self.offline:
            raise ConnectionError("offline")
        self.epoch = self.engine.epoch

    def close(self):
        pass

    def _call(self, op, body):
        if self.offline:
            raise ConnectionError("offline")
        response = self.engine.dispatch(dict(version=1, id=str(uuid.uuid4()), op=op,
            server_epoch=self.epoch, body=body), os.getuid())
        self.task = response["task"]
        return self.task

    def register(self, label, request):
        return self._call("register", dict(task_id=self.key, token="b" * 64, label=label,
            root=dict(pid=123, start_us=1, boot_us=1), request=request, reconfigure_mode="full_release"))

    def _task_call(self, op, **body):
        return self._call(op, dict(task_id=self.key, token="b" * 64, **body))

    def poll(self):
        return self._task_call("poll", last_generation=self.task["generation"])

    def report(self, phase, held, *, processes=(), detail=""):
        result = self._task_call("report", generation=self.task["generation"], phase=phase,
                                 held=held, processes=list(processes), detail=detail)
        self.reports.append((phase, held))
        if self.lose_report:
            self.lose_report = False
            raise ConnectionError("response lost after accepted report")
        return result

    def pause(self, paused):
        return self._task_call("set_paused", paused=paused, scope="self")

    def finish(self, outcome):
        result = self._task_call("finish", outcome=outcome)
        if self.lose_finish:
            self.lose_finish = False
            raise ConnectionError("finish response lost")
        return result

    def status(self):
        return {"tasks": [self.engine._public(row) for row in self.engine.tasks.values()]}


class FakeCohort:
    def __init__(self, grant, state):
        self.ready = self.idle = True
        self.authorized = False
        self.closed = self.done = self.draining = False
        self.state = state or 0
        self.processes = []
        self.admitted = 0

    def poll(self):
        if self.draining and self.idle:
            self.closed = True

    def authorize(self):
        self.authorized = True

    def start_unit(self):
        assert not self.draining and self.idle
        self.admitted += 1
        self.idle = False

    def complete(self):
        assert not self.idle
        self.state += 1
        self.idle = True

    def drain(self):
        self.draining = True

    def abort(self):
        self.closed = True


def setup():
    client = LocalClient()
    now = [0.0]
    supervisor = Supervisor(client, label="test", request=REQUEST,
                            cohort_factory=FakeCohort, clock=lambda: now[0])
    def tick():
        now[0] += 1
        supervisor.tick()
    tick()
    tick()
    assert supervisor.cohort.admitted == 1
    return client, supervisor, tick


def test_release_waits_for_safe_boundary_and_join():
    client, supervisor, tick = setup()
    old = supervisor.cohort
    client.pause(True)
    tick()
    assert old.draining and not old.closed
    assert client.task["granted"] == GRANT
    assert not any(phase == "released" for phase, _ in client.reports)
    old.complete()
    tick()
    assert supervisor.cohort is None and supervisor.state == 1
    assert client.task["granted"] == empty()
    client.pause(False)
    tick()
    tick()
    assert supervisor.cohort is not old and supervisor.cohort.state == 1
    assert supervisor.cohort.admitted == 1


@pytest.mark.parametrize("restart", [False, True])
def test_disconnect_never_admits_more_work_and_restores_state(restart):
    client, supervisor, tick = setup()
    old = supervisor.cohort
    client.offline = True
    tick()
    old.complete()
    for _ in range(3):
        tick()
    assert old.closed and old.admitted == 1
    assert supervisor.state == 1 and supervisor.cohort is None
    if restart:
        client.engine = Engine(GRANT, state=client.engine.state, controller_uids=[os.getuid()],
                               inspect=lambda p, uid: "alive", discover=lambda p, uid: [])
    client.offline = False
    for _ in range(4):
        tick()
    assert supervisor.cohort.state == 1
    assert supervisor.cohort.admitted == 1


def test_lost_applied_reply_drains_and_releases_old_grant():
    client = LocalClient()
    supervisor = Supervisor(client, label="test", request=REQUEST, cohort_factory=FakeCohort)
    supervisor.tick()
    client.lose_report = True
    supervisor.tick()
    old = supervisor.cohort
    assert old.draining and old.admitted == 0
    for _ in range(5):
        supervisor.next_poll = 0
        supervisor.tick()
    assert old.closed
    assert supervisor.cohort is not old and supervisor.cohort.admitted == 1


def test_finish_pauses_before_release_to_prevent_immediate_regrant():
    client, supervisor, tick = setup()
    supervisor.stop()
    tick()
    assert not supervisor.finished
    supervisor.cohort.complete()
    tick()
    assert supervisor.finished and client.task["terminal"]
    assert client.task["granted"] == empty()


def test_work_boundary_poll_does_not_throttle_to_one_hertz():
    client, supervisor, tick = setup()
    supervisor.cohort.complete()
    supervisor.tick()  # clock unchanged, next scheduled time is in the future
    assert supervisor.cohort.admitted == 2


def test_lost_finish_reply_reconciles_terminal_state():
    client, supervisor, tick = setup()
    supervisor.stop()
    supervisor.cohort.complete()
    client.lose_finish = True
    tick()
    assert client.task["terminal"] and not supervisor.finished
    tick()
    assert supervisor.finished


class CpuAdapter:
    def __init__(self, rank, grant, state):
        self.rank = rank
        self.state = state or 0
        assert os.environ["CUDA_VISIBLE_DEVICES"] == sorted(grant["gpus"])[rank]

    def run_unit(self):
        self.state += 1
        return self.state

    def capture(self):
        return str(self.state), self.state

    def close(self):
        pass


def until(cohort, predicate):
    deadline = time.monotonic() + 20
    while not predicate():
        cohort.poll()
        if time.monotonic() > deadline:
            pytest.fail("CPU cohort test timed out")
        time.sleep(0.01)


def test_real_spawn_join_handoff_and_resume():
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    completed = []
    cohort = ProcessCohort(CpuAdapter, GRANT, on_unit=lambda rows: completed.append(rows))
    try:
        cohort.authorize()
        until(cohort, lambda: cohort.ready)
        cohort.start_unit()
        cohort.drain()  # may arrive before the worker starts; still one unit
        until(cohort, lambda: cohort.closed)
        assert cohort.state.load() == 1 and completed == [[1, 1]]
        assert not cohort.processes
    finally:
        cohort.abort()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original
    resumed = ProcessCohort(CpuAdapter, {"cpu_cores": 1, "gpus": {"GPU-fake-b": 50}}, cohort.state)
    try:
        resumed.authorize()
        until(resumed, lambda: resumed.ready)
        resumed.start_unit()
        until(resumed, lambda: resumed.idle)
        resumed.drain()
        until(resumed, lambda: resumed.closed)
        assert resumed.state.load() == 2
    finally:
        resumed.abort()
        cohort.state.close()
        if resumed.state is not cohort.state:
            resumed.state.close()


class CpuTensorAdapter(CpuAdapter):
    def __init__(self, rank, grant, state):
        import torch
        assert not torch.cuda.is_initialized()
        self.rank = rank
        self.state = state if state is not None else {"optimizer": {0: {"exp_avg": torch.arange(1024)}}}

    def run_unit(self):
        self.state["optimizer"][0]["exp_avg"] += 1
        return 1

    def capture(self):
        return "same", self.state


def test_tensor_state_survives_sender_exit_without_resource_sharer():
    import torch
    cohort = ProcessCohort(CpuTensorAdapter, GRANT)
    try:
        cohort.authorize()
        until(cohort, lambda: cohort.ready)
        cohort.start_unit()
        cohort.drain()
        until(cohort, lambda: cohort.closed)
        assert not cohort.processes
        state = cohort.state.load()
        torch.testing.assert_close(state["optimizer"][0]["exp_avg"], torch.arange(1024) + 1)
        assert not torch.cuda.is_initialized()
    finally:
        cohort.abort()
        if cohort.state is not None:
            cohort.state.close()


class MustNotInitialize:
    def __init__(self, *args):
        raise AssertionError("unregistered child initialized")


def test_cancel_before_registration_never_calls_application_factory():
    cohort = ProcessCohort(MustNotInitialize, GRANT)
    try:
        cohort.drain()
        until(cohort, lambda: cohort.closed)
        assert cohort.state is None and not cohort.processes
    finally:
        cohort.abort()


def test_partial_control_frame_does_not_block_polling():
    sender, receiver = socket.socketpair()
    channel = _Channel(receiver)
    try:
        sender.sendall(b'{"kind":"unit",')
        started = time.monotonic()
        for _ in range(100):
            assert channel.poll() is None
        assert time.monotonic() - started < 1
        sender.sendall(b'"payload":null}\n')
        assert channel.poll() == ("unit", None)
    finally:
        sender.close()
        channel.close()
