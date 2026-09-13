"""Exercise revision commands through the actual fitter boundary queue."""
import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fit_session import SessionState
from spiral_runtime import _SessionShutdown, _apply_input_changes_at_boundary


@pytest.fixture
def resident():
    from test_spiral_headless import ProtocolTests
    session = ProtocolTests()._idle_session(completed=7)
    session._live_reservation_iteration = None
    session._live_reservation_epoch = 0
    session._iteration_in_progress = None
    session._input_batches = {}
    session.status = lambda: {"current_iteration": session._completed, "state": session._state}
    entered, release = threading.Event(), threading.Event()
    release.set()
    active = {"revision": 1}

    def prepare(records, config, **kwargs):
        entered.set()
        assert release.wait(5)
        if records[0].get("invalid"):
            raise ValueError("invalid selected draft")
        return SimpleNamespace(
            _workspace_membership=copy.deepcopy(records[0]),
            verified_patches={}, unverified_patches={})

    def install(candidate):
        active.update(candidate._workspace_membership)
        return []

    session._context = SimpleNamespace(
        prepare_input_changes=Mock(side_effect=prepare),
        install_input_changes=Mock(side_effect=install))
    errors = []

    def worker():
        try:
            session.wait_for_iteration(7)
        except _SessionShutdown:
            pass
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    yield session, active, entered, release
    release.set()
    with session._condition:
        session._shutdown = True
        session._condition.notify_all()
    thread.join(5)
    assert not thread.is_alive()
    assert not errors


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


def test_final_running_boundary_can_be_reserved(resident):
    session, _, _, _ = resident
    with session._condition:
        session._state = SessionState.Running
        session._iteration_in_progress = 6
    result = session.reserve_input_boundary(7, 1, include_idle=True)
    assert result["reserved"]
    session.cancel_input_boundary(1)


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


def test_worker_failure_is_reported_instead_of_waiting_for_timeout(resident):
    from spiral_runtime import InputBatchCommand
    session, _, _, _ = resident
    with session._condition:
        session._state = SessionState.Error
        session._error = 'device failure'
    with pytest.raises(RuntimeError, match='device failure'):
        session._wait_input_batch(InputBatchCommand(), 5)


def test_failed_run_configuration_completes_its_command(resident):
    from spiral_runtime import ConfigureCommand
    from unittest.mock import Mock
    resident, *_ = resident
    resident._context.apply_config = Mock(side_effect=ValueError('invalid setting'))
    resident._pending = 2
    resident._warnings = []
    command = ConfigureCommand(config={'optimizer_learning_rate': -1})
    resident._run_configuration(command)
    assert command.done.is_set() and 'invalid setting' in command.error
    assert resident._pending == 0
    assert resident.status()['state'] == 'Idle'
