"""Bounded, ordered sample pipeline with fixed or shared process stages.

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
from dataclasses import dataclass, replace
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
    stage1_started_ns: int
    stage1_finished_ns: int
    stage2_finished_ns: int
    delivered_ns: int


@dataclass(frozen=True)
class _RawItem:
    index: int
    value: Any
    stage1_service_ms: float
    stage1_cpu_ms: float
    stage1_worker: int
    stage1_started_ns: int
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
    stage1_started_ns: int
    stage1_finished_ns: int


def native_thread_environment() -> dict[str, str]:
    """Use before spawning as well as inside native-compute workers."""
    return {name: "1" for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )}


def _set_native_thread_limits() -> None:
    os.environ.update(native_thread_environment())
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
    transport: Any,
    stage1_is_final: bool,
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
            started_ns = time.perf_counter_ns()
            cpu_started = time.process_time()
            value = operation(index)
            if transport is not None:
                value = transport.encode(
                    index, value, final=bool(stage1_is_final)
                )
            cpu_ms = (time.process_time() - cpu_started) * 1000.0
            service_ms = (time.perf_counter() - started) * 1000.0
            finished_ns = time.perf_counter_ns()
            item = _RawItem(
                index=index,
                value=value,
                stage1_service_ms=service_ms,
                stage1_cpu_ms=cpu_ms,
                stage1_worker=int(worker_id),
                stage1_started_ns=started_ns,
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
    transport: Any,
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
            raw_value = (
                transport.decode(raw.index, raw.value, final=False)
                if transport is not None
                else raw.value
            )
            started = time.perf_counter()
            cpu_started = time.process_time()
            value = operation(raw_value)
            if transport is not None:
                value = transport.encode(raw.index, value, final=True)
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
                stage1_started_ns=int(raw.stage1_started_ns),
                stage1_finished_ns=int(raw.stage1_finished_ns),
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


def _shared_worker_main(
    worker_id: int,
    stage1_factory: Callable[[], Callable[[int], Any]],
    stage2_factory: Callable[[], Callable[[Any], Any]],
    control: Connection,
    start_event: Any,
    stop_event: Any,
    transport: Any,
    raw_queue: Any,
    result_queue: Any,
    stage1_affinity_workers: int,
    next_index: Any,
    target_frontier: Any,
    claim_lock: Any,
    credits: Any,
    end_index: int,
    start_index: int,
    slot_count: int,
    slot_owners: Any,
    outstanding: Any,
    admission_highwater: Any,
    stage1_counts: Any,
    stage2_counts: Any,
    shared_worker_count: int,
    active_workers: Any,
    worker_busy: Any,
    elastic_workers: bool,
    admitted_frontier: Any,
) -> None:
    """Claim eligible work without a coordinator round trip."""

    _set_native_thread_limits()
    stage = "startup"
    index = -1
    try:
        stage1_operation = (
            stage1_factory() if not elastic_workers and worker_id < int(stage1_affinity_workers) else None
        )
        stage2_operation = (
            None if elastic_workers or worker_id < int(stage1_affinity_workers) else stage2_factory()
        )
        _send_control(control, "ready", "shared", worker_id, os.getpid())
        while not start_event.wait(timeout=0.1):
            if stop_event.is_set():
                return

        def claim_load() -> int | None:
            with claim_lock:
                candidate = int(next_index.value)
                if (candidate >= int(end_index) or candidate >= int(target_frontier.value)
                        or candidate >= int(admitted_frontier.value)):
                    return None
                slot = (
                    (candidate - int(start_index)) % int(slot_count)
                    if int(slot_count) > 0
                    else -1
                )
                if slot >= 0 and int(slot_owners[slot]) >= 0:
                    return None
                if not credits.acquire(block=False):
                    return None
                next_index.value = candidate + 1
                outstanding.value = int(outstanding.value) + 1
                admission_highwater.value = max(
                    int(admission_highwater.value), int(outstanding.value)
                )
                if slot >= 0:
                    slot_owners[slot] = candidate
                return candidate

        def run_augment(raw: _RawItem) -> None:
            nonlocal stage, index, stage2_operation
            stage = "augment"
            index = int(raw.index)
            if stage2_operation is None:
                stage2_operation = stage2_factory()
            received_ns = time.perf_counter_ns()
            raw_value = (
                transport.decode(index, raw.value, final=False)
                if transport is not None
                else raw.value
            )
            started = time.perf_counter()
            cpu_started = time.process_time()
            value = stage2_operation(raw_value)
            if transport is not None:
                value = transport.encode(index, value, final=True)
            finished_ns = time.perf_counter_ns()
            completed = _CompletedItem(
                index=index,
                value=value,
                stage1_service_ms=float(raw.stage1_service_ms),
                stage1_cpu_ms=float(raw.stage1_cpu_ms),
                raw_queue_ms=max(
                    0.0, (received_ns - int(raw.stage1_finished_ns)) / 1.0e6
                ),
                stage2_service_ms=(time.perf_counter() - started) * 1000.0,
                stage2_cpu_ms=(time.process_time() - cpu_started) * 1000.0,
                stage2_worker=int(worker_id),
                stage2_finished_ns=finished_ns,
                stage1_worker=int(raw.stage1_worker),
                stage1_started_ns=int(raw.stage1_started_ns),
                stage1_finished_ns=int(raw.stage1_finished_ns),
            )
            _put_bounded(result_queue, completed, stop_event)
            stage2_counts[worker_id] = int(stage2_counts[worker_id]) + 1

        home_load = worker_id < int(stage1_affinity_workers)
        while not stop_event.is_set():
            if elastic_workers:
                with claim_lock:
                    active = int(active_workers.value)
                    enabled = worker_id < active
                    worker_busy[worker_id] = int(enabled)
                if not enabled:
                    stop_event.wait(0.01)
                    continue
                # Even a one-core grant must be able to execute both stages.
                # Never park every augmenter behind occupied raw/credit slots.
                affinity = max(1, min(int(stage1_affinity_workers), active - 1))
                home_load = worker_id < affinity
            candidate = claim_load() if home_load else None
            if candidate is not None:
                stage = "load"
                index = candidate
                if stage1_operation is None:
                    stage1_operation = stage1_factory()
                started = time.perf_counter()
                started_ns = time.perf_counter_ns()
                cpu_started = time.process_time()
                value = stage1_operation(index)
                if transport is not None:
                    value = transport.encode(index, value, final=False)
                item = _RawItem(
                    index=index,
                    value=value,
                    stage1_service_ms=(time.perf_counter() - started) * 1000.0,
                    stage1_cpu_ms=(time.process_time() - cpu_started) * 1000.0,
                    stage1_worker=int(worker_id),
                    stage1_started_ns=started_ns,
                    stage1_finished_ns=time.perf_counter_ns(),
                )
                stage1_counts[worker_id] = int(stage1_counts[worker_id]) + 1
                with claim_lock:
                    target_frontier.value = min(
                        int(end_index), int(target_frontier.value) + 2
                    )
                if int(shared_worker_count) == 1 or elastic_workers and active == 1:
                    # Do not publish and then assume we can get our own item
                    # back: a pre-shrink consumer may still take it. Keeping
                    # ownership avoids a blocking get with no remaining loader.
                    run_augment(item)
                else:
                    try:
                        raw_queue.put(item, timeout=0.001)
                    except queue.Full:
                        # Consume ready work directly instead of blocking a
                        # shared worker on publication into a full raw queue.
                        run_augment(item)
            else:
                if home_load:
                    with claim_lock:
                        loads_fully_dispatched = int(next_index.value) >= int(end_index)
                    if not loads_fully_dispatched and not elastic_workers:
                        time.sleep(0.001)
                        continue
                try:
                    raw = raw_queue.get(timeout=0.01 if not home_load else 0.001)
                except queue.Empty:
                    continue
                run_augment(raw)
            stage = "idle"
            index = -1
    except BaseException as exc:
        if not stop_event.is_set():
            _send_control(
                control,
                "error",
                stage,
                worker_id,
                os.getpid(),
                index,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
    finally:
        control.close()


class OrderedTwoStageSamplePipeline:
    """One or two process stages with bounded lookahead and ordered delivery.

    Setting ``stage2_workers=0`` and ``stage2_factory=None`` exposes completed
    stage-one values directly.  This is used to measure the real raw-sample
    queue without adding an identity/augmentation process stage. Shared mode
    uses one worker entrypoint with soft stage affinity and worker-side claims.
    """

    def __init__(
        self,
        *,
        stage1_factory: Callable[[], Callable[[int], Any]],
        stage2_factory: Callable[[], Callable[[Any], Any]] | None,
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
        transport: Any = None,
        shared_workers: int = 0,
        initial_in_flight: int | None = None,
        shared_stage1_affinity_workers: int | None = None,
        elastic_workers: bool = False,
        initial_active_workers: int | None = None,
        manual_admission: bool = False,
    ) -> None:
        for name, value in (
            ("item_count", item_count),
            ("raw_queue_capacity", raw_queue_capacity),
            ("result_queue_capacity", result_queue_capacity),
            ("max_in_flight", max_in_flight),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name, value in (
            ("stage1_workers", stage1_workers),
            ("stage2_workers", stage2_workers),
            ("shared_workers", shared_workers),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be >= 0")
        self.shared_pool = int(shared_workers) > 0
        if elastic_workers and not self.shared_pool:
            raise ValueError("elastic workers require the shared pool")
        if manual_admission and not self.shared_pool:
            raise ValueError("manual admission requires the shared pool")
        if initial_active_workers is not None and not elastic_workers:
            raise ValueError("initial_active_workers requires elastic_workers")
        if self.shared_pool:
            if int(stage1_workers) != 0 or int(stage2_workers) != 0:
                raise ValueError(
                    "stage1_workers and stage2_workers must be 0 in shared-pool mode"
                )
            if stage2_factory is None:
                raise ValueError("shared-pool mode requires stage2_factory")
        else:
            if int(stage1_workers) <= 0:
                raise ValueError("stage1_workers must be > 0 outside shared-pool mode")
            if (int(stage2_workers) == 0) != (stage2_factory is None):
                raise ValueError(
                    "stage2_factory must be None exactly when stage2_workers is 0"
                )
        self.has_stage2 = self.shared_pool or int(stage2_workers) > 0
        initial = int(shared_workers) if initial_in_flight is None else int(initial_in_flight)
        if self.shared_pool and not (0 < initial <= int(max_in_flight)):
            raise ValueError("initial_in_flight must be in [1, max_in_flight]")
        if self.shared_pool and transport is not None:
            slot_count = getattr(transport, "slot_count", None)
            if slot_count is None or int(slot_count) < int(max_in_flight):
                raise ValueError(
                    "shared-pool transport slot_count must be >= max_in_flight"
                )
        affinity_workers = (
            int(round(int(shared_workers) * (2.0 / 3.0)))
            if shared_stage1_affinity_workers is None
            else int(shared_stage1_affinity_workers)
        )
        if self.shared_pool:
            valid_affinity = (
                affinity_workers == 1
                if int(shared_workers) == 1
                else 0 < affinity_workers < int(shared_workers)
            )
            if not valid_affinity:
                raise ValueError(
                    "shared_stage1_affinity_workers must leave at least one "
                    "worker for each stage"
                )
        if multiprocessing_context == "fork":
            raise ValueError("fork is not supported for the torch sample pipeline")
        self.start_index = int(start_index)
        self.end_index = self.start_index + int(item_count)
        self.next_expected = self.start_index
        self.item_count = int(item_count)
        self.max_in_flight = int(max_in_flight)
        self.raw_queue_capacity = int(raw_queue_capacity)
        self.no_progress_timeout_seconds = float(no_progress_timeout_seconds)
        self.context_name = str(multiprocessing_context)
        self.context = mp.get_context(self.context_name)
        self.raw_queue = self.context.Queue(maxsize=int(raw_queue_capacity))
        self.result_queue = self.context.Queue(maxsize=int(result_queue_capacity))
        self.start_event = self.context.Event()
        self.stop_event = self.context.Event()
        self.next_index = self.context.Value("q", self.start_index)
        self.manual_admission = bool(manual_admission)
        self._admitted_frontier = self.context.Value(
            "q", self.start_index if manual_admission else self.end_index
        )
        self.claim_lock = self.context.Lock()
        self.elastic_workers = bool(elastic_workers)
        initial_active = int(shared_workers) if initial_active_workers is None else initial_active_workers
        if type(initial_active) is not int or not 0 <= initial_active <= int(shared_workers):
            raise ValueError("initial active workers exceeds shared pool")
        self._active_workers = self.context.Value("q", initial_active)
        self._worker_busy = self.context.Array("b", [0] * max(1, int(shared_workers)), lock=False)
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
        self.transport = transport
        self.leased: set[int] = set()
        self.shared_workers = int(shared_workers)
        self.initial_in_flight = initial if self.shared_pool else 0
        self.shared_stage1_affinity_workers = affinity_workers if self.shared_pool else 0
        self.admission_highwater = 0
        self.shared_stage1_counts: dict[int, int] = {}
        self.shared_stage2_counts: dict[int, int] = {}
        self.shared_stage1_target_min = 0
        self.shared_stage1_target_max = 0
        self.shared_stage1_target_final = 0
        shared_slot_count = (
            int(getattr(transport, "slot_count"))
            if self.shared_pool and transport is not None
            else 0
        )
        self._shared_target_frontier = self.context.Value(
            "q", min(self.end_index, self.start_index + self.initial_in_flight)
        )
        self._shared_slot_owners = self.context.Array(
            "q", [-1] * max(1, shared_slot_count), lock=False
        )
        self._shared_outstanding = self.context.Value("q", 0)
        self._shared_admission_highwater = self.context.Value("q", 0)
        self._shared_stage1_count_array = self.context.Array(
            "q", [0] * max(1, self.shared_workers), lock=False
        )
        self._shared_stage2_count_array = self.context.Array(
            "q", [0] * max(1, self.shared_workers), lock=False
        )
        if self.shared_pool:
            self.shared_stage1_target_min = self.shared_stage1_affinity_workers
            self.shared_stage1_target_max = self.shared_stage1_affinity_workers
            self.shared_stage1_target_final = self.shared_stage1_affinity_workers

        try:
            if self.shared_pool:
                assert stage2_factory is not None
                for worker_id in range(self.shared_workers):
                    parent, child = self.context.Pipe(duplex=False)
                    process = self.context.Process(
                        target=_shared_worker_main,
                        args=(
                            worker_id,
                            stage1_factory,
                            stage2_factory,
                            child,
                            self.start_event,
                            self.stop_event,
                            self.transport,
                            self.raw_queue,
                            self.result_queue,
                            self.shared_stage1_affinity_workers,
                            self.next_index,
                            self._shared_target_frontier,
                            self.claim_lock,
                            self.credits,
                            self.end_index,
                            self.start_index,
                            shared_slot_count,
                            self._shared_slot_owners,
                            self._shared_outstanding,
                            self._shared_admission_highwater,
                            self._shared_stage1_count_array,
                            self._shared_stage2_count_array,
                            self.shared_workers,
                            self._active_workers,
                            self._worker_busy,
                            self.elastic_workers,
                            self._admitted_frontier,
                        ),
                        name=f"ordered-sample-shared-{worker_id}",
                    )
                    process.start()
                    child.close()
                    self.processes.append(process)
                    self.controls.append(parent)
            for worker_id in range(0 if self.shared_pool else int(stage1_workers)):
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
                        self.transport,
                        not self.has_stage2,
                    ),
                    name=f"ordered-sample-load-{worker_id}",
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.controls.append(parent)
            for worker_id in range(0 if self.shared_pool else int(stage2_workers)):
                assert stage2_factory is not None
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
                        self.transport,
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

    @staticmethod
    def _worker_error(message: tuple[Any, ...], *, startup: bool = False) -> RuntimeError:
        if len(message) == 8:
            _kind, stage, worker, pid, index, exc_type, detail, trace = message
            where = f" during startup" if startup else f" at sample {index}"
        else:
            _kind, stage, worker, pid, exc_type, detail, trace = message
            where = " during startup" if startup else ""
        return RuntimeError(
            f"{stage} worker {worker} pid={pid} failed{where}: "
            f"{exc_type}: {detail}\n{trace}"
        )

    def _read_controls(self) -> None:
        for connection in self.controls:
            while connection.poll():
                try:
                    message = connection.recv()
                except EOFError:
                    break
                if message and message[0] == "error":
                    raise self._worker_error(message)

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
                        raise self._worker_error(message, startup=True)
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

    def _release_shared_credit(self, index: int) -> None:
        with self.claim_lock:
            if self.transport is not None:
                slot = int(self.transport._location(int(index))[0])
                owner = int(self._shared_slot_owners[slot])
                if owner != int(index):
                    raise RuntimeError(
                        f"shared slot {slot} owner mismatch: {owner} != {index}"
                    )
                self._shared_slot_owners[slot] = -1
            if int(self._shared_outstanding.value) <= 0:
                raise RuntimeError("shared-pool outstanding credit underflow")
            self._shared_outstanding.value = int(self._shared_outstanding.value) - 1
            self.credits.release()

    def next_ordered(self, *, release_credit: bool = True) -> OrderedPipelineResult:
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
                completed = (
                    self.result_queue if self.has_stage2 else self.raw_queue
                ).get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            self._record_completed(completed)
        return self._deliver_next(release_credit=release_credit)

    def _record_completed(self, completed: Any) -> None:
        index = int(completed.index)
        if index < self.next_expected or index in self.reorder:
            raise RuntimeError(f"duplicate pipeline completion for sample {index}")
        if not (self.start_index <= index < self.end_index):
            raise RuntimeError(f"out-of-range pipeline completion for sample {index}")
        self.reorder[index] = completed
        self.received_ns[index] = time.perf_counter_ns()
        self.received += 1
        self.reorder_highwater = max(self.reorder_highwater, len(self.reorder))

    def poll_ready(self, *, limit: int = 64) -> int:
        """Drain completions without waiting for the next canonical sample.

        Also call while shrinking/paused: completed workers must not remain
        blocked publishing into a full result queue. Credits still bound RAM.
        """
        if self.closed:
            raise RuntimeError("pipeline is closed")
        self._check_processes()
        received = 0
        for _ in range(limit):
            try:
                completed = (self.result_queue if self.has_stage2 else self.raw_queue).get_nowait()
            except queue.Empty:
                break
            self._record_completed(completed)
            received += 1
        return received

    def try_next_ordered(self, *, release_credit: bool = True) -> OrderedPipelineResult | None:
        if self.delivered >= self.item_count:
            raise StopIteration
        self.poll_ready()
        if self.next_expected not in self.reorder:
            return None
        return self._deliver_next(release_credit=release_credit)

    def admit_until(self, end: int) -> None:
        """Publish initialized descriptors, never change already admitted work.

        This is separate from credit/ramp-up limits: workers still claim only
        runnable work through the ordinary shared-pool scheduler.
        """
        if not self.manual_admission or self.closed:
            raise RuntimeError("explicit admission requires an open manual stream")
        with self.claim_lock:
            if not int(self._admitted_frontier.value) <= end <= self.end_index:
                raise ValueError("admission frontier must advance within the stream")
            self._admitted_frontier.value = end

    def set_worker_limit(self, count: int) -> None:
        """Cooperative shrink: current jobs finish, surplus workers then park."""
        if not self.elastic_workers or self.closed:
            raise RuntimeError("worker limit requires an open elastic pool")
        if type(count) is not int or not 0 <= count <= self.shared_workers:
            raise ValueError("worker count exceeds the persistent shared pool")
        with self.claim_lock:
            self._active_workers.value = count

    def workers_within_limit(self) -> bool:
        """True only when every surplus worker has finished its current job."""
        if not self.elastic_workers:
            raise RuntimeError("worker limit requires an elastic pool")
        with self.claim_lock:
            return not any(self._worker_busy[int(self._active_workers.value):])

    def _deliver_next(self, *, release_credit: bool) -> OrderedPipelineResult:
        completed = self.reorder.pop(self.next_expected)
        received_ns = self.received_ns.pop(self.next_expected)
        delivered_ns = time.perf_counter_ns()
        if self.transport is not None:
            completed = replace(
                completed,
                value=self.transport.decode(
                    completed.index, completed.value, final=True
                ),
            )
        if self.has_stage2:
            result = OrderedPipelineResult(
                index=int(completed.index),
                value=completed.value,
                stage1_service_ms=float(completed.stage1_service_ms),
                stage1_cpu_ms=float(completed.stage1_cpu_ms),
                raw_queue_ms=float(completed.raw_queue_ms),
                stage2_service_ms=float(completed.stage2_service_ms),
                stage2_cpu_ms=float(completed.stage2_cpu_ms),
                result_queue_ms=max(
                    0.0, (received_ns - int(completed.stage2_finished_ns)) / 1.0e6
                ),
                stage1_worker=int(completed.stage1_worker),
                stage2_worker=int(completed.stage2_worker),
                stage1_started_ns=int(completed.stage1_started_ns),
                stage1_finished_ns=int(completed.stage1_finished_ns),
                reorder_wait_ms=max(0.0, (delivered_ns - received_ns) / 1.0e6),
                stage2_finished_ns=int(completed.stage2_finished_ns),
                delivered_ns=int(delivered_ns),
            )
        else:
            result = OrderedPipelineResult(
                index=int(completed.index),
                value=completed.value,
                stage1_service_ms=float(completed.stage1_service_ms),
                stage1_cpu_ms=float(completed.stage1_cpu_ms),
                raw_queue_ms=max(
                    0.0, (received_ns - int(completed.stage1_finished_ns)) / 1.0e6
                ),
                stage2_service_ms=0.0,
                stage2_cpu_ms=0.0,
                result_queue_ms=0.0,
                stage1_worker=int(completed.stage1_worker),
                stage2_worker=-1,
                stage1_started_ns=int(completed.stage1_started_ns),
                stage1_finished_ns=int(completed.stage1_finished_ns),
                reorder_wait_ms=max(0.0, (delivered_ns - received_ns) / 1.0e6),
                stage2_finished_ns=int(completed.stage1_finished_ns),
                delivered_ns=int(delivered_ns),
            )
        self.next_expected += 1
        self.delivered += 1
        if release_credit:
            if self.shared_pool:
                self._release_shared_credit(int(result.index))
            else:
                self.credits.release()
        else:
            self.leased.add(int(result.index))
        return result

    def release(self, index: int) -> None:
        index_i = int(index)
        if index_i not in self.leased:
            raise RuntimeError(f"pipeline sample {index_i} is not leased")
        self.leased.remove(index_i)
        if self.shared_pool:
            self._release_shared_credit(index_i)
        else:
            self.credits.release()

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
        if self.shared_pool:
            self.admission_highwater = int(self._shared_admission_highwater.value)
            self.shared_stage1_counts = {
                worker_id: int(self._shared_stage1_count_array[worker_id])
                for worker_id in range(self.shared_workers)
                if int(self._shared_stage1_count_array[worker_id]) > 0
            }
            self.shared_stage2_counts = {
                worker_id: int(self._shared_stage2_count_array[worker_id])
                for worker_id in range(self.shared_workers)
                if int(self._shared_stage2_count_array[worker_id]) > 0
            }
        for process in self.processes:
            process.close()
        for connection in self.controls:
            connection.close()
        self.reorder.clear()
        self.received_ns.clear()
        self.leased.clear()
        for data_queue in (self.raw_queue, self.result_queue):
            if data_queue is None:
                continue
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
