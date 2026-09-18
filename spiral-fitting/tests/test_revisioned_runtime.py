"""Exercise revision commands through the actual fitter boundary queue."""

from runtime_fixtures import resident

import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from spiral_runtime import _apply_input_changes_at_boundary


def test_idle_batch_applies_without_run_and_replays_once(resident):
    session, active, _, _ = resident
    result = session.apply_input_changes("batch", [{"revision": 2}], timeout=2)
    assert result["applied"]
    assert active["revision"] == 2
    assert session._completed == session._target == 7
    assert session.apply_input_changes("batch", [{"revision": 2}]) == result
    assert session._context.install_input_changes.call_count == 1
    with pytest.raises(ValueError, match="different content"):
        session.apply_input_changes("batch", [{"revision": 3}])


def test_preparation_failure_leaves_active_state_and_allows_next_batch(resident):
    session, active, _, _ = resident
    result = session.apply_input_changes("invalid", [{"invalid": True}], timeout=2)
    assert not result["applied"]
    assert active == {"revision": 1}
    session._context.install_input_changes.assert_not_called()
    assert session.apply_input_changes("repaired", [{"revision": 3}], timeout=2)["applied"]
    assert active["revision"] == 3


def test_device_failure_during_preparation_remains_fail_stop():
    from spiral_runtime import InputBatchCommand
    from runtime_fixtures import RuntimeFixture

    session = RuntimeFixture()._idle_session(completed=7)
    failure = RuntimeError('CUDA device failure')
    session._context = SimpleNamespace(
        prepare_input_changes=Mock(side_effect=failure))
    command = InputBatchCommand(batch_id='device-failure')
    with pytest.raises(RuntimeError, match='CUDA device failure') as raised:
        session._run_input_batch(command)
    assert raised.value is failure
    assert command.error == 'RuntimeError: CUDA device failure'


def test_timeout_preserves_captured_batch_and_boundary_for_retry(resident):
    session, active, entered, release = resident
    release.clear()
    records = [{"revision": 2}]
    with pytest.raises(TimeoutError, match="unknown"):
        session.apply_input_changes("batch", records, timeout=.02)
    assert entered.wait(2)
    records[0]["revision"] = 3
    assert active["revision"] == 1
    with pytest.raises(ValueError, match="outstanding"):
        session.apply_input_changes("newer", records, timeout=.02)
    release.set()
    assert session.apply_input_changes("batch", [{"revision": 2}], timeout=2)["applied"]
    assert active["revision"] == 2
    assert session._context.prepare_input_changes.call_count == 1
    assert session.apply_input_changes("newer", records, timeout=2)["applied"]
    assert active["revision"] == 3


@pytest.mark.parametrize("failure", ["validation", "membership", None])
def test_distributed_decision_requires_all_ranks_prepared(failure):
    decisions = []
    session = SimpleNamespace(_condition=threading.Condition(), _live_reservation_epoch=0,
                              status=lambda: {"current_iteration": 7})

    def call(name, arguments, **kwargs):
        assert not kwargs["collective"]
        if name == "reserve_input_boundary":
            assert arguments["include_idle"]
            return {0: {"reserved": True}, 1: {"reserved": True}}
        if name == "prepare_input_batch":
            return {0: {"prepared": True, "membership": {"revision": 2}},
                    1: {"prepared": failure != "validation",
                        "membership": {"revision": 3 if failure == "membership" else 2},
                        "error": "invalid draft"}}
        assert name == "finish_input_batch"
        decisions.append(arguments["install"])
        return {0: {}, 1: {}}

    session._call = call
    result = _apply_input_changes_at_boundary(
        session, "batch", [{"revision": 2}], {}, timeout=2, distributed=True)
    assert decisions == [failure is None]
    assert result["applied"] is (failure is None)


@pytest.mark.parametrize("faster_rank", [0, 1])
def test_distributed_boundary_retry_handles_mixed_reservations(faster_rank):
    session = SimpleNamespace(_condition=threading.Condition(), _live_reservation_epoch=0,
                              status=lambda: {"current_iteration": 7})
    calls = []

    def call(name, arguments, **kwargs):
        calls.append((name, arguments.copy()))
        if name == "reserve_input_boundary":
            target = arguments["target_iteration"]
            return {rank: ({"reserved": False, "next_iteration": 8}
                           if rank == faster_rank and target == 7 else
                           {"reserved": True, "target_iteration": target})
                    for rank in range(2)}
        if name == "prepare_input_batch":
            assert arguments["target_iteration"] == 8
            return {rank: {"prepared": True, "membership": {"revision": 2}}
                    for rank in range(2)}
        return {0: {}, 1: {}}

    session._call = call
    result = _apply_input_changes_at_boundary(
        session, "batch", [{"revision": 2}], {}, timeout=2, distributed=True)
    assert result["applied"] and result["iteration"] == 8
    assert [name for name, _ in calls] == [
        "reserve_input_boundary", "cancel_input_boundary", "reserve_input_boundary",
        "prepare_input_batch", "finish_input_batch"]
    assert calls[0][1]["reservation_epoch"] == calls[1][1]["reservation_epoch"]
    assert calls[2][1]["reservation_epoch"] == calls[1][1]["reservation_epoch"]
    assert calls[-1][1]["install"]


def test_distributed_close_retry_rejects_surviving_file_users():
    from spiral_runtime import DistributedInteractiveFitSession
    session = DistributedInteractiveFitSession.__new__(DistributedInteractiveFitSession)
    session._closed = True
    worker = Mock()
    worker.is_alive.return_value = True
    session._processes = [worker]
    session._listener = session._watchdog_thread = None
    with pytest.raises(TimeoutError):
        session.close()
    worker.is_alive.return_value = False
    listener = Mock()
    listener.is_alive.return_value = True
    session._listener = listener
    with pytest.raises(TimeoutError):
        session.close()
    listener.is_alive.return_value = False
    session.close()
