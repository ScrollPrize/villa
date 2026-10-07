"""Completion-driven batching of independent, persistent follower rollouts."""
from concurrent.futures import Future
from queue import Queue, Empty
from threading import Thread
import torch


def merge_rows(values):
    first = values[0]
    if isinstance(first, dict):
        if any(v.keys() != first.keys() for v in values):
            raise ValueError('incompatible model input keys')
        return {k: merge_rows([v[k] for v in values]) for k in first}
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if any(v != first for v in values[1:]):
        raise ValueError('incompatible model operating policies')
    return first


def slice_rows(value, start, stop):
    if isinstance(value, dict):
        return {k: slice_rows(v, start, stop) for k, v in value.items()}
    return value[start:stop] if torch.is_tensor(value) and value.ndim else value


class ForwardQueue:
    """No fixed trace batches: drain whatever forward requests are ready.

    Each caller keeps its complete upstream rollout (including memory and RNG
    counters) alive. This only combines independent model-forward rows.
    """
    def __init__(self, forward, max_batch=8, device='cpu'):
        if max_batch < 1:
            raise ValueError('max_batch must be positive')
        self.forward, self.max_batch, self.device = forward, max_batch, device
        self.queue = Queue()
        self.batch_sizes = []
        self.thread = Thread(target=self._run, name='neural-forward', daemon=True)
        self.thread.start()

    def __call__(self, x, hist, mask, sampling):
        future = Future()
        ready = None
        if str(self.device).startswith('cuda'):
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        self.queue.put((future, x, hist, mask, sampling, ready))
        return future.result()

    def _run(self):
        with torch.inference_mode():
            while True:
                first = self.queue.get()
                if first is None:
                    return
                batch = [first]
                while len(batch) < self.max_batch:
                    try:
                        value = self.queue.get_nowait()
                    except Empty:
                        break
                    if value is None:
                        self.queue.put(None)
                        break
                    batch.append(value)
                try:
                    if str(self.device).startswith('cuda'):
                        for row in batch:
                            torch.cuda.current_stream(self.device).wait_event(row[5])
                    counts = [len(row[2]) for row in batch]
                    self.batch_sizes.append(sum(counts))
                    output = self.forward(*(merge_rows([row[i] for row in batch]) for i in range(1, 5)))
                    # Requests are consumed on other CPU threads/streams. Do not
                    # publish CUDA tensors until this forward is complete.
                    if str(self.device).startswith('cuda'):
                        torch.cuda.synchronize(self.device)
                    offset = 0
                    for row, count in zip(batch, counts):
                        row[0].set_result(slice_rows(output, offset, offset+count))
                        offset += count
                except BaseException as error:
                    for row in batch:
                        row[0].set_exception(error)

    def close(self):
        self.queue.put(None)
        self.thread.join()
