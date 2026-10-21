"""CPU-only real subprocess server for bootstrap race regression tests."""
import signal
import sys

from villa_gpu_manager.server import Server

running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
with Server(sys.argv[1], {"cpu_cores": 4, "gpus": {"GPU-fake": 100}}) as server:
    while running:
        server.step()
