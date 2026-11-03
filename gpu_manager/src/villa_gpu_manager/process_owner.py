"""Independent ownership root for sibling worker cohorts in a coordinator.

No resource request or CUDA here. The existing protocol supports explicitly
reported workers; this root must not own unrelated coordinator children.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .processes import process_identity
from .exclusive import OwnedProcessGroup


def watch(fd, ack_fd):
    groups = {}
    try:
        with os.fdopen(fd) as stream:
            for line in stream:
                message = json.loads(line)
                identity = message['identity']
                key = (identity['pid'], identity['start_us'], identity['boot_us'])
                if message['op'] == 'track':
                    groups[key] = OwnedProcessGroup(identity)
                elif message['op'] == 'retire':
                    group = groups[key]
                    if group.members():
                        raise RuntimeError('cannot retire live workers')
                    del groups[key]
                else:
                    raise ValueError('unknown owner operation')
                os.write(ack_fd, b'1')
    finally:
        # EOF means normal shutdown or coordinator death. On normal shutdown
        # the cohort has already been joined, so these are identity-safe no-ops.
        for group in groups.values():
            while True:
                try:
                    group.close()
                    break
                except Exception:
                    time.sleep(1)  # Keep the root alive until cleanup is proven.
        os.close(ack_fd)


class ProcessOwner:
    def __init__(self):
        read_fd, write_fd = os.pipe()
        self.ack, ack_write = os.pipe()
        try:
            code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                    "from villa_gpu_manager.process_owner import watch;watch(int(sys.argv[2]),int(sys.argv[3]))")
            self.process = subprocess.Popen([sys.executable, "-c", code,
                str(Path(__file__).resolve().parents[1]), str(read_fd), str(ack_write)],
                pass_fds=(read_fd, ack_write), start_new_session=True)
        except BaseException:
            os.close(write_fd)
            os.close(self.ack)
            raise
        finally:
            os.close(read_fd)
            os.close(ack_write)
        self.stream = os.fdopen(write_fd, "w", buffering=1)
        self.identity = process_identity(self.process.pid)

    def track(self, identities):
        self._send('track', identities)

    def retire(self, identities):
        self._send('retire', identities)

    def _send(self, op, identities):
        if self.process.poll() is not None:
            raise RuntimeError("worker ownership watchdog exited")
        for identity in identities:
            self.stream.write(json.dumps(dict(op=op,identity=identity)) + "\n")
            if os.read(self.ack, 1) != b'1':
                raise RuntimeError("worker ownership handoff failed")

    def close(self):
        self.stream.close()
        os.close(self.ack)
        if self.process.wait(timeout=30) != 0:
            raise RuntimeError("worker ownership watchdog failed cleanup")
