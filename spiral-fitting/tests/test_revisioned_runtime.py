"""Commands through the actual fitter boundary queue."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from spiral_runtime import _SessionShutdown


@pytest.fixture
def resident():
    from test_spiral_headless import ProtocolTests
    session = ProtocolTests()._idle_session(completed=7)
    session.status = lambda: {"current_iteration": session._completed, "state": session._state}
    session._context = SimpleNamespace()
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
    yield session, None, None, None
    with session._condition:
        session._shutdown = True
        session._condition.notify_all()
    thread.join(5)
    assert not thread.is_alive()
    assert not errors


def test_failed_run_configuration_completes_its_command(resident):
    from spiral_runtime import ConfigureCommand
    resident, *_ = resident
    resident._context.apply_config = Mock(side_effect=ValueError('invalid setting'))
    resident._pending = 2
    resident._warnings = []
    command = ConfigureCommand(config={'optimizer_learning_rate': -1})
    resident._run_configuration(command)
    assert command.done.is_set() and 'invalid setting' in command.error
    assert resident._pending == 0
    assert resident.status()['state'] == 'Idle'


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
