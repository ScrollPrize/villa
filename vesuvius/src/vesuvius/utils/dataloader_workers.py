"""DataLoader worker helpers shared by inference and training."""
from __future__ import annotations

import os

import torch


def keep_worker_descriptors_private(_worker_id: int) -> None:
    """``worker_init_fn`` that stops a spawned worker's helper processes inheriting its descriptors.

    With torch's ``file_system`` sharing strategy (the only one on macOS) each worker starts a ``torch_shm_manager``.
    It inherits the pipe the parent watches to see the worker exit and outlives the worker, so every join in the
    DataLoader shutdown waits its full 5 s timeout after the worker has gone: 40 s per loader with 8 workers. Marking
    the worker's descriptors close-on-exec keeps the helper off that pipe. Elsewhere this does nothing.
    """
    if torch.multiprocessing.get_sharing_strategy() != "file_system":
        return
    try:
        descriptors = [int(name) for name in os.listdir("/dev/fd")]
    except (OSError, ValueError):
        return
    for descriptor in descriptors:
        if descriptor > 2:
            try:
                os.set_inheritable(descriptor, False)
            except OSError:
                pass
