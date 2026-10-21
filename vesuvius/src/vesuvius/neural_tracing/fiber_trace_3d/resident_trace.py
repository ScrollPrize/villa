"""Shared-slot inference serviced by the existing training GPU replicas."""
import queue
import threading


class _RemoteWork:
    def __init__(self, connection):
        self.connection = connection

    def get(self):
        self.connection.send(("get", True))
        return self.connection.recv()

    def get_nowait(self):
        self.connection.send(("get", False))
        present, value = self.connection.recv()
        if not present:
            raise queue.Empty
        return value


class _RemoteEvents:
    def __init__(self, connection):
        self.connection = connection

    def put(self, event):
        self.connection.send(("event", event))


def serve_resident(connection, model, rank, arguments):
    """Run the shared tiled kernel on a raw module, preserving training state."""
    import torch
    from lasagna.tiled_predict3d import _multi_gpu_worker_main
    # No model loading, optimizer mutation, DDP wrapper or training RNG use.
    cuda = next(model.parameters()).device
    modes = [(part, part.training) for part in model.modules()]
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        with torch.random.fork_rng(devices=[cuda.index] if cuda.type == "cuda" else []):
            adapter, state, tile, ramp, scales, stride, inputs, results, profile, batch, report = arguments
            allow_tf32 = adapter.config.get("inference", {}).get("allow_tf32")
            if allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                torch.backends.cudnn.allow_tf32 = allow_tf32
            _multi_gpu_worker_main(
                rank, str(cuda), adapter, state, tile, ramp, scales, stride,
                inputs, results, _RemoteWork(connection), _RemoteEvents(connection),
                profile, batch, report, resident_model=model)
        if cuda.type == "cuda":
            torch.cuda.synchronize(cuda)
            torch.cuda.empty_cache()  # Only unused inference scratch; live training tensors stay.
        connection.send(("done", None))
    finally:
        for part, training in modes:
            part.training = training
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


class ResidentWorkers:
    """Bridge descriptor queues to borrowed replicas; never owns GPU processes."""
    def __init__(self, connections, arguments, start, cancel_event):
        self.connections = connections
        self.stop = threading.Event()
        self.cancel_event = cancel_event
        self.errors = queue.SimpleQueue()
        self.threads = []
        adapter, state, tile, ramp, scales, stride, inputs, results, work, events, profile, batch, report = arguments
        self.work, self.events = work, events
        descriptor = (adapter, state, tile, ramp, scales, stride, inputs, results, profile, batch, report)
        # Announce with a tiny message before starting the unit so run_unit's
        # poll cannot miss it. Send the potentially large descriptor only once
        # workers are running and can read: never fill an idle worker's pipe.
        for connection in connections:
            connection.send("trace")
        start()
        for index, connection in enumerate(connections):
            thread = threading.Thread(target=self._bridge, args=(connection, descriptor),
                                      name=f"resident-trace-{index}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def _bridge(self, connection, descriptor):
        try:
            connection.send(descriptor)
            while True:
                kind, value = connection.recv()
                if kind == "done":
                    return
                if kind == "event":
                    self.events.put(value)
                elif kind == "get":
                    item = None
                    if value:
                        while not self.stop.is_set() and not self.cancel_event.is_set():
                            try:
                                item = self.work.get(timeout=0.1)
                                break
                            except queue.Empty:
                                pass
                        connection.send(item)
                    else:
                        if self.stop.is_set() or self.cancel_event.is_set():
                            connection.send((True, None))
                            continue
                        try:
                            item = self.work.get_nowait()
                            connection.send((True, item))
                        except queue.Empty:
                            connection.send((False, None))
                else:
                    raise RuntimeError(f"invalid resident inference event {kind!r}")
        except BaseException as error:
            self.errors.put(error)

    def check(self):
        try:
            error = self.errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("resident inference worker failed") from error

    def close(self):
        self.stop.set()
        # This runs on the evaluation coordinator thread. Manager polling stays
        # responsive; slot ownership remains held until every GPU acknowledges.
        while any(thread.is_alive() for thread in self.threads):
            # Drain result events so a bridge cannot block on a full event queue
            # after cancellation, while the GPU still owns shared result memory.
            try:
                self.events.get(timeout=0.01)
            except queue.Empty:
                pass
            for thread in self.threads:
                thread.join(timeout=0)
        self.check()
