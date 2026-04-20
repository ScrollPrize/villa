import threading
import time
from concurrent.futures import ThreadPoolExecutor


class BoundedPatchWriter:
    """Thread-pooled writer with a bounded in-flight queue.

    Wraps a ThreadPoolExecutor so the producer (inference loop) blocks once
    ``max_workers * 2`` write tasks are pending. This prevents the executor's
    internal queue from growing unbounded and pinning a float16 patch per
    pending task in host RAM when the sink (e.g. S3) is slower than the GPU.

    Also measures producer wait time and emits throttled stall reports via the
    provided tqdm bar, so an operator can see immediately when the GPU is idle
    waiting on writes.
    """

    def __init__(self, output_store, max_workers, *, pbar=None,
                 on_progress=None, stall_report_interval=5.0):
        self._store = output_store
        self._pbar = pbar
        self._on_progress = on_progress
        self._max_workers = max_workers
        self._max_inflight = max_workers * 2
        self._inflight = threading.BoundedSemaphore(self._max_inflight)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._error_lock = threading.Lock()
        self._error = None

        self._total_wait = 0.0
        self._last_stall_report = 0.0
        self._stall_report_interval = stall_report_interval
        self._wall_start = time.monotonic()

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def max_inflight(self):
        return self._max_inflight

    @property
    def total_wait_seconds(self):
        return self._total_wait

    @property
    def elapsed_seconds(self):
        return time.monotonic() - self._wall_start

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._executor.__exit__(exc_type, exc, tb)
        return False

    def submit(self, write_index, patch_data):
        self._raise_if_failed()

        if not self._inflight.acquire(blocking=False):
            wait_start = time.monotonic()
            self._inflight.acquire()
            waited = time.monotonic() - wait_start
            self._total_wait += waited
            self._maybe_report_stall(waited)

        try:
            self._raise_if_failed()
            self._executor.submit(self._write, write_index, patch_data)
        except BaseException:
            self._inflight.release()
            raise

    def _write(self, write_index, patch_data):
        try:
            self._store[write_index] = patch_data
            if self._on_progress is not None:
                self._on_progress()
        except Exception as e:
            with self._error_lock:
                if self._error is None:
                    self._error = e
            self._log(f"Error writing patch {write_index}: {e}")
        finally:
            self._inflight.release()

    def _raise_if_failed(self):
        with self._error_lock:
            err = self._error
        if err is not None:
            raise err

    def _maybe_report_stall(self, waited):
        now = time.monotonic()
        if now - self._last_stall_report < self._stall_report_interval:
            return
        elapsed = now - self._wall_start
        pct = (self._total_wait / elapsed * 100.0) if elapsed > 0 else 0.0
        self._log(
            f"[writer stall] waited {waited:.2f}s this patch, "
            f"{self._total_wait:.1f}s total ({pct:.1f}% of wall clock)"
        )
        self._last_stall_report = now

    def _log(self, msg):
        if self._pbar is not None:
            self._pbar.write(msg)
        else:
            print(msg)
