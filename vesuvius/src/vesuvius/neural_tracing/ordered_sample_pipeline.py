"""Bounded, ordered, two-process-stage sample pipeline.

Stage workers may complete out of order.  A single credit window bounds every
claimed-but-not-yet-delivered item, including items held by the coordinator's
reorder map.  The coordinator always drains completions while waiting for the
next index, avoiding result-queue head-of-line deadlock.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable


@dataclass(frozen=True)
class OrderedPipelineResult:
    index: int
    value: Any
    stage1_service_ms: float
    stage1_cpu_ms: float
    raw_queue_ms: float
    stage2_service_ms: float
    stage2_cpu_ms: float
    result_queue_ms: float
    reorder_wait_ms: float
    stage1_worker: int
    stage2_worker: int
    stage2_finished_ns: int
    delivered_ns: int


@dataclass(frozen=True)
class _RawItem:
    index: int
    value: Any
    stage1_service_ms: float
    stage1_cpu_ms: float
    stage1_worker: int
    stage1_finished_ns: int


@dataclass(frozen=True)
class _CompletedItem:
    index: int
    value: Any
    stage1_service_ms: float
    stage1_cpu_ms: float
    raw_queue_ms: float
    stage2_service_ms: float
    stage2_cpu_ms: float
    stage2_worker: int
    stage2_finished_ns: int
    stage1_worker: int


def _set_native_thread_limits() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _put_bounded(target: Any, value: Any, stop_event: Any) -> float:
    started = time.perf_counter()
    while not stop_event.is_set():
        try:
            target.put(value, timeout=0.1)
            return (time.perf_counter() - started) * 1000.0
        except queue.Full:
            continue
    raise RuntimeError("pipeline stopped while publishing an item")


def _send_control(control: Connection, *message: Any) -> None:
    try:
        control.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _stage1_main(
    worker_id: int,
    factory: Callable[[], Callable[[int], Any]],
    raw_queue: Any,
    control: Connection,
    start_event: Any,
    stop_event: Any,
    next_index: Any,
    claim_lock: Any,
    credits: Any,
    end_index: int,
) -> None:
    _set_native_thread_limits()
    try:
        operation = factory()
        _send_control(control, "ready", "load", worker_id, os.getpid())
        while not start_event.wait(timeout=0.1):
            if stop_event.is_set():
                return
        while not stop_event.is_set():
            if not credits.acquire(timeout=0.1):
                continue
            with claim_lock:
                index = int(next_index.value)
                if index < int(end_index):
                    next_index.value = index + 1
                else:
                    index = -1
            if index < 0:
                credits.release()
                break
            started = time.perf_counter()
            cpu_started = time.process_time()
            value = operation(index)
            cpu_ms = (time.process_time() - cpu_started) * 1000.0
            service_ms = (time.perf_counter() - started) * 1000.0
            finished_ns = time.perf_counter_ns()
            item = _RawItem(
                index=index,
                value=value,
                stage1_service_ms=service_ms,
                stage1_cpu_ms=cpu_ms,
                stage1_worker=int(worker_id),
                stage1_finished_ns=finished_ns,
            )
            _put_bounded(raw_queue, item, stop_event)
    except BaseException as exc:
        if not stop_event.is_set():
            _send_control(
                control,
                "error",
                "load",
                worker_id,
                os.getpid(),
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
            stop_event.set()
        return
    finally:
        _send_control(control, "done", "load", worker_id, os.getpid())
        # Torch queue payloads may own shared tensor storage created by this
        # process. Keep the producer alive until the coordinator has delivered
        # every sample and explicitly stops the pipeline.
        while not stop_event.wait(timeout=0.1):
            pass
        control.close()


def _stage2_main(
    worker_id: int,
    factory: Callable[[], Callable[[Any], Any]],
    raw_queue: Any,
    result_queue: Any,
    control: Connection,
    start_event: Any,
    stop_event: Any,
) -> None:
    _set_native_thread_limits()
    try:
        operation = factory()
        _send_control(control, "ready", "augment", worker_id, os.getpid())
        while not start_event.wait(timeout=0.1):
            if stop_event.is_set():
                return
        while not stop_event.is_set():
            try:
                raw = raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            received_ns = time.perf_counter_ns()
            started = time.perf_counter()
            cpu_started = time.process_time()
            value = operation(raw.value)
            cpu_ms = (time.process_time() - cpu_started) * 1000.0
            service_ms = (time.perf_counter() - started) * 1000.0
            finished_ns = time.perf_counter_ns()
            completed = _CompletedItem(
                index=int(raw.index),
                value=value,
                stage1_service_ms=float(raw.stage1_service_ms),
                stage1_cpu_ms=float(raw.stage1_cpu_ms),
                raw_queue_ms=max(0.0, (received_ns - int(raw.stage1_finished_ns)) / 1.0e6),
                stage2_service_ms=service_ms,
                stage2_cpu_ms=cpu_ms,
                stage2_worker=int(worker_id),
                stage2_finished_ns=finished_ns,
                stage1_worker=int(raw.stage1_worker),
            )
            _put_bounded(result_queue, completed, stop_event)
    except BaseException as exc:
        if not stop_event.is_set():
            _send_control(
                control,
                "error",
                "augment",
                worker_id,
                os.getpid(),
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
            stop_event.set()
        return
    finally:
        _send_control(control, "done", "augment", worker_id, os.getpid())
        while not stop_event.wait(timeout=0.1):
            pass
        control.close()


class OrderedTwoStageSamplePipeline:
    """Two process stages with bounded lookahead and ordered final delivery."""

    def __init__(
        self,
        *,
        stage1_factory: Callable[[], Callable[[int], Any]],
        stage2_factory: Callable[[], Callable[[Any], Any]],
        start_index: int,
        item_count: int,
        stage1_workers: int,
        stage2_workers: int,
        raw_queue_capacity: int,
        result_queue_capacity: int,
        max_in_flight: int,
        multiprocessing_context: str = "forkserver",
        startup_timeout_seconds: float = 120.0,
        no_progress_timeout_seconds: float = 300.0,
    ) -> None:
        for name, value in (
            ("item_count", item_count),
            ("stage1_workers", stage1_workers),
            ("stage2_workers", stage2_workers),
            ("raw_queue_capacity", raw_queue_capacity),
            ("result_queue_capacity", result_queue_capacity),
            ("max_in_flight", max_in_flight),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be > 0")
        if multiprocessing_context == "fork":
            raise ValueError("fork is not supported for the torch sample pipeline")
        self.start_index = int(start_index)
        self.end_index = self.start_index + int(item_count)
        self.next_expected = self.start_index
        self.item_count = int(item_count)
        self.no_progress_timeout_seconds = float(no_progress_timeout_seconds)
        self.context_name = str(multiprocessing_context)
        self.context = mp.get_context(self.context_name)
        self.raw_queue = self.context.Queue(maxsize=int(raw_queue_capacity))
        self.result_queue = self.context.Queue(maxsize=int(result_queue_capacity))
        self.start_event = self.context.Event()
        self.stop_event = self.context.Event()
        self.next_index = self.context.Value("q", self.start_index)
        self.claim_lock = self.context.Lock()
        self.credits = self.context.BoundedSemaphore(int(max_in_flight))
        self.processes: list[Any] = []
        self.controls: list[Connection] = []
        self.reorder: dict[int, _CompletedItem] = {}
        self.received_ns: dict[int, int] = {}
        self.reorder_highwater = 0
        self.received = 0
        self.delivered = 0
        self.closed = False
        self.started_ns = 0

        try:
            for worker_id in range(int(stage1_workers)):
                parent, child = self.context.Pipe(duplex=False)
                process = self.context.Process(
                    target=_stage1_main,
                    args=(
                        worker_id,
                        stage1_factory,
                        self.raw_queue,
                        child,
                        self.start_event,
                        self.stop_event,
                        self.next_index,
                        self.claim_lock,
                        self.credits,
                        self.end_index,
                    ),
                    name=f"ordered-sample-load-{worker_id}",
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.controls.append(parent)
            for worker_id in range(int(stage2_workers)):
                parent, child = self.context.Pipe(duplex=False)
                process = self.context.Process(
                    target=_stage2_main,
                    args=(
                        worker_id,
                        stage2_factory,
                        self.raw_queue,
                        self.result_queue,
                        child,
                        self.start_event,
                        self.stop_event,
                    ),
                    name=f"ordered-sample-augment-{worker_id}",
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.controls.append(parent)
            self._await_ready(float(startup_timeout_seconds))
            self.started_ns = time.perf_counter_ns()
            self.start_event.set()
        except BaseException:
            self.close()
            raise

    def _read_controls(self) -> None:
        for connection in self.controls:
            while connection.poll():
                try:
                    message = connection.recv()
                except EOFError:
                    break
                if message and message[0] == "error":
                    _kind, stage, worker, pid, exc_type, detail, trace = message
                    raise RuntimeError(
                        f"{stage} worker {worker} pid={pid} failed: "
                        f"{exc_type}: {detail}\n{trace}"
                    )

    def _await_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        ready: set[tuple[str, int]] = set()
        expected = len(self.processes)
        while len(ready) < expected:
            for connection in self.controls:
                while connection.poll():
                    try:
                        message = connection.recv()
                    except EOFError:
                        break
                    if message and message[0] == "ready":
                        ready.add((str(message[1]), int(message[2])))
                    elif message and message[0] == "error":
                        _kind, stage, worker, pid, exc_type, detail, trace = message
                        raise RuntimeError(
                            f"{stage} worker {worker} pid={pid} failed during startup: "
                            f"{exc_type}: {detail}\n{trace}"
                        )
            for process in self.processes:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"pipeline worker {process.name} exited during startup "
                        f"with code {process.exitcode}"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"pipeline workers did not become ready within {timeout_seconds}s")
            time.sleep(0.01)

    def _check_processes(self) -> None:
        self._read_controls()
        if self.delivered >= self.item_count:
            return
        for process in self.processes:
            if process.exitcode is not None:
                raise RuntimeError(
                    f"pipeline worker {process.name} exited unexpectedly with code {process.exitcode}"
                )

    def next_ordered(self) -> OrderedPipelineResult:
        if self.closed:
            raise RuntimeError("pipeline is closed")
        if self.delivered >= self.item_count:
            raise StopIteration
        wait_started = time.monotonic()
        while self.next_expected not in self.reorder:
            self._check_processes()
            remaining = self.no_progress_timeout_seconds - (time.monotonic() - wait_started)
            if remaining <= 0.0:
                raise TimeoutError(
                    f"pipeline made no ordered-delivery progress for "
                    f"{self.no_progress_timeout_seconds}s; next={self.next_expected} "
                    f"received={self.received} reorder={len(self.reorder)}"
                )
            try:
                completed = self.result_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            received_ns = time.perf_counter_ns()
            index = int(completed.index)
            if index < self.next_expected or index in self.reorder:
                raise RuntimeError(f"duplicate pipeline completion for sample {index}")
            if not (self.start_index <= index < self.end_index):
                raise RuntimeError(f"out-of-range pipeline completion for sample {index}")
            self.reorder[index] = completed
            self.received_ns[index] = received_ns
            self.received += 1
            self.reorder_highwater = max(self.reorder_highwater, len(self.reorder))
        completed = self.reorder.pop(self.next_expected)
        received_ns = self.received_ns.pop(self.next_expected)
        delivered_ns = time.perf_counter_ns()
        result = OrderedPipelineResult(
            index=int(completed.index),
            value=completed.value,
            stage1_service_ms=float(completed.stage1_service_ms),
            stage1_cpu_ms=float(completed.stage1_cpu_ms),
            raw_queue_ms=float(completed.raw_queue_ms),
            stage2_service_ms=float(completed.stage2_service_ms),
            stage2_cpu_ms=float(completed.stage2_cpu_ms),
            result_queue_ms=max(0.0, (received_ns - int(completed.stage2_finished_ns)) / 1.0e6),
            stage1_worker=int(completed.stage1_worker),
            stage2_worker=int(completed.stage2_worker),
            reorder_wait_ms=max(0.0, (delivered_ns - received_ns) / 1.0e6),
            stage2_finished_ns=int(completed.stage2_finished_ns),
            delivered_ns=int(delivered_ns),
        )
        self.next_expected += 1
        self.delivered += 1
        self.credits.release()
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.start_event.set()
        deadline = time.monotonic() + 5.0
        for process in self.processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2.0)
            process.close()
        for connection in self.controls:
            connection.close()
        self.reorder.clear()
        self.received_ns.clear()
        for data_queue in (self.raw_queue, self.result_queue):
            try:
                while True:
                    data_queue.get_nowait()
            except (queue.Empty, EOFError, OSError):
                pass
            data_queue.close()
            data_queue.join_thread()

    def __enter__(self) -> "OrderedTwoStageSamplePipeline":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        del exc_type, exc, traceback_value
        self.close()
