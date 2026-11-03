"""Service-owner configuration, automatically reloaded without restarting clients."""

import json
import os
from pathlib import Path
import time


def config_path():
    return Path(os.environ.get("VILLA_GPU_MANAGER_CONSTRAINTS",
                str(Path.home() / ".config/villa-gpu-manager/constraints.json"))).expanduser()


class Constraints:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else config_path()
        self.last = None
        self.next_check = 0

    def poll(self, engine, *, force=False):
        if not force and time.monotonic() < self.next_check:
            return
        self.next_check = time.monotonic() + 1
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            # Deleting an established policy must not silently enable all GPUs.
            if self.last is not None:
                return
            data = json.dumps({"allowed_gpu_uuids": engine.state.get(
                "allowed_gpu_uuids", list(engine.inventory["gpus"]))}).encode()
        if data == self.last:
            return
        value = json.loads(data)
        if not isinstance(value, dict) or set(value) - {"allowed_gpu_uuids"}:
            raise ValueError("constraints must contain only allowed_gpu_uuids")
        allowed = value.get("allowed_gpu_uuids", list(engine.inventory["gpus"]))
        if (not isinstance(allowed, list) or any(not isinstance(g, str) for g in allowed)
                or len(set(allowed)) != len(allowed)
                or not set(allowed) <= engine.inventory["gpus"].keys()):
            raise ValueError("allowed_gpu_uuids must be unique inventory UUIDs (empty disables new grants)")
        engine.set_allowed_gpus(allowed)
        self.last = data
        print(f"gpu manager: constraints loaded {self.path}: {allowed}", flush=True)
