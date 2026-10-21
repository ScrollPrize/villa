"""Driver stdout may finish between a nonblocking read and waitpid."""
import io
import pytest

from villa_gpu_manager.processes import NvidiaMonitor


@pytest.mark.parametrize('poll_delay', [0, 25])
def test_final_stdout_is_drained_after_process_exit(monkeypatch, poll_delay):
    import villa_gpu_manager.processes as module

    xml = (b'<nvidia_smi_log><gpu><uuid>GPU-a</uuid><fb_memory_usage>'
           b'<total>100 MiB</total><used>0 MiB</used></fb_memory_usage>'
           b'<processes/></gpu></nvidia_smi_log>')

    class Pipe(io.BytesIO):
        def fileno(self):
            return 999

    class Process:
        stdout = Pipe()
        returncode = 0
        exited = False

        def poll(self):
            self.exited = True
            return 0

    process = Process()
    pending = [xml, b'']

    def read(fd, size):
        assert fd == 999
        if not process.exited:
            raise BlockingIOError()
        return pending.pop(0)

    monkeypatch.setattr(module.os, 'read', read)
    monitor = NvidiaMonitor({'cpu_cores': 1, 'gpus': {'GPU-a': 90 * 1024**2}}, 0)
    monitor.process = process
    monitor.started = module.time.monotonic() - poll_delay
    assert monitor.poll(set()) == set()
    assert monitor.last_snapshot_at is not None
    assert not pending
