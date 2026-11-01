"""Opt-in stage attribution shared by inference readers and their coordinator."""
from contextlib import contextmanager
import threading
import time

try:
    import resource
except ImportError:
    resource = None


def _thread_usage():
    if resource is None or not hasattr(resource, "RUSAGE_THREAD"):
        return {}
    try:
        usage = resource.getrusage(resource.RUSAGE_THREAD)
    except OSError:
        return {}
    return {name: getattr(usage, name) for name in
            ("ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw")}


@contextmanager
def stage_timing(samples, stage):
    """Append one record on exit; None disables clocks/counters completely.

    CPU is for the calling thread, not the process/native worker pool. Elapsed
    minus CPU includes blocking AND descheduling/GIL contention; it cannot by
    itself identify the cause. Nested stage records must not be summed together.
    """
    if samples is None:
        yield
        return
    usage = _thread_usage()
    thread_id = threading.get_native_id()
    details = {}
    started = time.perf_counter()
    cpu = time.thread_time()
    try:
        yield details
    finally:
        cpu_seconds = time.thread_time() - cpu
        ended = time.perf_counter()
        after = _thread_usage()
        samples.append({"stage": stage, "thread_id": thread_id,
            "start_seconds": started, "end_seconds": ended,
            "wall_seconds": ended - started, "thread_cpu_seconds": cpu_seconds,
            "off_cpu_seconds_estimate": max(0., ended - started - cpu_seconds),
            **details,
            **{name: after[name] - value for name, value in usage.items() if name in after}})
