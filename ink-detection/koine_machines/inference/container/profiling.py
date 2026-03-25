"""
Low-overhead profiling helpers for the optimized ink-detection workflow.

The helpers in this module are intentionally best-effort:
- profiler setup failures never raise into the workload
- unavailable telemetry is recorded explicitly
- summary files are flushed from finally blocks where possible
"""

from __future__ import annotations

import atexit
import csv
import json
import logging
import math
import os
import shutil
import statistics
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

FLAG_EXACT = "exact"
FLAG_ESTIMATED = "estimated"
FLAG_APPROXIMATE = "approximate"
FLAG_UNAVAILABLE = "unavailable"

SEMANTICS_DURATION = "wall-clock duration"
SEMANTICS_COUNTER = "counter delta"
SEMANTICS_SAMPLE = "sampled statistic"
SEMANTICS_PEAK = "peak value"
SEMANTICS_METADATA = "metadata"

PROFILE_SUMMARY_NAME = "profiling-summary.json"
PROFILE_TIMESERIES_NAME = "profiling-timeseries.jsonl"
TORCH_TRACE_NAME = "torch-trace.json"
TORCH_OPS_SUMMARY_NAME = "torch-ops-summary.txt"
TORCH_MEMORY_SUMMARY_NAME = "torch-memory-summary.json"

WORKFLOW_SUMMARY_JSON = "workflow-profiling-summary.json"
WORKFLOW_SUMMARY_MD = "workflow-profiling-summary.md"
WORKFLOW_PARTITIONS_CSV = "workflow-profiling-partitions.csv"
WORKFLOW_PARTITIONS_JSONL = "workflow-profiling-partitions.jsonl"

DURATION_METRICS = (
    "total_wall_seconds",
    "s3_list_seconds",
    "download_seconds",
    "remote_read_seconds",
    "local_read_seconds",
    "decode_or_deserialize_seconds",
    "cache_fill_seconds",
    "preprocess_seconds",
    "model_load_seconds",
    "compile_warmup_seconds",
    "host_to_device_seconds",
    "forward_seconds",
    "device_to_host_seconds",
    "postprocess_seconds",
    "zarr_write_seconds",
    "local_write_seconds",
    "upload_seconds",
    "reduce_seconds",
)

COUNTER_METRICS = (
    "s3_download_bytes",
    "s3_upload_bytes",
    "remote_read_bytes",
    "local_read_bytes",
    "cache_fill_bytes",
    "local_write_bytes",
    "process_io_read_bytes",
    "process_io_write_bytes",
    "process_io_read_count",
    "process_io_write_count",
    "cache_hits",
    "cache_misses",
    "cache_negative_hits",
    "partition_tiles",
    "partition_batches",
)

STAT_METRICS = (
    "process_cpu_utilization_percent_avg",
    "process_cpu_utilization_percent_max",
    "process_cpu_utilization_percent_p95",
    "system_cpu_utilization_percent_avg",
    "system_cpu_utilization_percent_max",
    "system_cpu_utilization_percent_p95",
    "system_iowait_delta_seconds",
    "process_rss_bytes_avg",
    "process_rss_bytes_max",
    "process_rss_bytes_peak",
    "process_uss_bytes_avg",
    "process_uss_bytes_max",
    "process_pss_bytes_avg",
    "process_pss_bytes_max",
    "gpu_utilization_percent_avg",
    "gpu_utilization_percent_max",
    "gpu_utilization_percent_p95",
    "vram_used_bytes_avg",
    "vram_used_bytes_max",
    "vram_used_bytes_peak",
    "gpu_temperature_celsius_max",
    "gpu_power_watts_max",
    "torch_cuda_max_memory_allocated_bytes",
    "torch_cuda_max_memory_reserved_bytes",
    "partition_throughput_tiles_per_second",
    "steady_state_tiles_per_second",
)

_ACTIVE_PROFILER: Optional["WorkflowProfiler"] = None
_WORKER_PROFILER: Optional["WorkerProfiler"] = None
_WORKER_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def dir_size_bytes(path: str | Path) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    total = 0
    for file_path in target.rglob("*"):
        if file_path.is_file():
            with suppress(OSError):
                total += file_path.stat().st_size
    return total


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_detailed_partition_selector(selector: str) -> Dict[str, Any]:
    value = (selector or "first").strip().lower()
    if not value or value == "first":
        return {"mode": "first", "part_ids": []}
    if value == "all":
        return {"mode": "all", "part_ids": []}
    part_ids: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        part_ids.append(int(token))
    return {"mode": "explicit", "part_ids": sorted(set(part_ids))}


def should_profile_partition(selector: str, part_id: Optional[int]) -> bool:
    parsed = parse_detailed_partition_selector(selector)
    if parsed["mode"] == "all":
        return True
    if parsed["mode"] == "first":
        return part_id in (None, 0)
    return part_id is not None and int(part_id) in parsed["part_ids"]


def maybe_mkdir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False))
            handle.write("\n")


def copy_if_exists(src: str | Path, dst: str | Path) -> None:
    source = Path(src)
    if not source.exists():
        return
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def metric_entry(
    value: Any,
    flag: str,
    semantics: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "value": value,
        "flag": flag,
        "semantics": semantics,
    }
    if reason:
        entry["reason"] = reason
    return entry


@contextmanager
def noop_scope() -> Iterator[None]:
    yield


def scoped_timer(profiler: Optional["MetricSink"], metric_name: str, **kwargs: Any):
    if profiler is None:
        return noop_scope()
    return profiler.scope(metric_name, **kwargs)


class MetricSink:
    def scope(self, metric_name: str, **kwargs: Any):
        raise NotImplementedError

    def add_duration(self, metric_name: str, seconds: float, **kwargs: Any) -> None:
        raise NotImplementedError

    def increment_counter(self, metric_name: str, delta: int | float, **kwargs: Any) -> None:
        raise NotImplementedError

    def set_metric(self, metric_name: str, value: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def mark_unavailable(self, metric_name: str, reason: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def record_store_event(
        self,
        category: str,
        seconds: float,
        num_bytes: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        raise NotImplementedError


class _ScopeTimer:
    def __init__(
        self,
        sink: MetricSink,
        metric_name: str,
        flag: str = FLAG_EXACT,
        semantics: str = SEMANTICS_DURATION,
        cuda_sync: bool = False,
    ):
        self.sink = sink
        self.metric_name = metric_name
        self.flag = flag
        self.semantics = semantics
        self.cuda_sync = cuda_sync
        self.start = 0.0

    def __enter__(self) -> "_ScopeTimer":
        if self.cuda_sync and torch is not None and torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.synchronize()
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.cuda_sync and torch is not None and torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.synchronize()
        elapsed = time.monotonic() - self.start
        self.sink.add_duration(self.metric_name, elapsed, flag=self.flag, semantics=self.semantics)


class TransferTracker:
    def __init__(self):
        self.bytes_transferred = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount: int) -> None:
        with self._lock:
            self.bytes_transferred += int(bytes_amount)


class WorkerProfiler(MetricSink):
    def __init__(self, output_dir: str):
        self.output_dir = maybe_mkdir(output_dir)
        self.summary_path = self.output_dir / f"worker-summary-{os.getpid()}.json"
        self.metrics: Dict[str, Any] = {}
        self.flags: Dict[str, str] = {}
        self.semantics: Dict[str, str] = {}
        self.reasons: Dict[str, str] = {}
        self.start_utc = utc_now_iso()
        self.start_monotonic = time.monotonic()
        self.pid = os.getpid()
        self._lock = threading.Lock()
        atexit.register(self.flush)

    def scope(
        self,
        metric_name: str,
        flag: str = FLAG_APPROXIMATE,
        semantics: str = SEMANTICS_DURATION,
        cuda_sync: bool = False,
    ):
        return _ScopeTimer(self, metric_name, flag=flag, semantics=semantics, cuda_sync=cuda_sync)

    def add_duration(
        self,
        metric_name: str,
        seconds: float,
        flag: str = FLAG_APPROXIMATE,
        semantics: str = SEMANTICS_DURATION,
    ) -> None:
        with self._lock:
            self.metrics[metric_name] = float(self.metrics.get(metric_name, 0.0)) + float(seconds)
            self.flags[metric_name] = flag
            self.semantics[metric_name] = semantics

    def increment_counter(
        self,
        metric_name: str,
        delta: int | float,
        flag: str = FLAG_EXACT,
        semantics: str = SEMANTICS_COUNTER,
    ) -> None:
        with self._lock:
            current = self.metrics.get(metric_name, 0)
            if current is None:
                current = 0
            self.metrics[metric_name] = current + delta
            self.flags[metric_name] = flag
            self.semantics[metric_name] = semantics

    def set_metric(
        self,
        metric_name: str,
        value: Any,
        flag: str = FLAG_EXACT,
        semantics: str = SEMANTICS_SAMPLE,
    ) -> None:
        with self._lock:
            self.metrics[metric_name] = value
            self.flags[metric_name] = flag
            self.semantics[metric_name] = semantics

    def mark_unavailable(
        self,
        metric_name: str,
        reason: str,
        semantics: str = SEMANTICS_SAMPLE,
    ) -> None:
        with self._lock:
            self.metrics[metric_name] = None
            self.flags[metric_name] = FLAG_UNAVAILABLE
            self.semantics[metric_name] = semantics
            self.reasons[metric_name] = reason

    def record_store_event(
        self,
        category: str,
        seconds: float,
        num_bytes: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        if category == "remote_read":
            self.add_duration("remote_read_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("remote_read_bytes", num_bytes)
        elif category == "local_read":
            self.add_duration("local_read_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("local_read_bytes", num_bytes)
        elif category == "cache_fill":
            self.add_duration("cache_fill_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("cache_fill_bytes", num_bytes)
                self.increment_counter("local_write_bytes", num_bytes)
        elif category == "local_write":
            self.add_duration("local_write_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("local_write_bytes", num_bytes)
        elif category == "cache_hit":
            self.increment_counter("cache_hits", 1)
        elif category == "cache_miss":
            self.increment_counter("cache_misses", 1)
        elif category == "cache_negative_hit":
            self.increment_counter("cache_negative_hits", 1)
        if reason:
            self.mark_unavailable(f"{category}_note", reason, semantics=SEMANTICS_METADATA)

    def flush(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "pid": self.pid,
            "worker": True,
            "timestamps": {
                "start_utc": self.start_utc,
                "end_utc": utc_now_iso(),
            },
            "metrics": {
                name: metric_entry(
                    self.metrics.get(name),
                    self.flags.get(name, FLAG_EXACT),
                    self.semantics.get(name, SEMANTICS_SAMPLE),
                    self.reasons.get(name),
                )
                for name in sorted(self.metrics.keys())
            },
        }
        write_json(self.summary_path, payload)


def get_worker_profiler() -> Optional[WorkerProfiler]:
    global _WORKER_PROFILER
    if os.getenv("PROFILING_LEVEL", "basic").strip().lower() == "off":
        return None
    worker_dir = os.getenv("PROFILING_WORKER_SUMMARY_DIR", "").strip()
    if not worker_dir:
        return None
    with _WORKER_LOCK:
        if _WORKER_PROFILER is None:
            _WORKER_PROFILER = WorkerProfiler(worker_dir)
    return _WORKER_PROFILER


def get_active_profiler() -> Optional["WorkflowProfiler"]:
    return _ACTIVE_PROFILER


def record_store_event(
    category: str,
    seconds: float,
    num_bytes: Optional[int] = None,
    reason: Optional[str] = None,
) -> None:
    sink: Optional[MetricSink] = get_worker_profiler()
    if sink is None:
        sink = get_active_profiler()
    if sink is None:
        return
    with suppress(Exception):
        sink.record_store_event(category, seconds, num_bytes=num_bytes, reason=reason)


class WorkflowProfiler(MetricSink):
    def __init__(
        self,
        *,
        level: str,
        sample_interval_ms: int,
        raw_root: Optional[str],
        local_root: str,
        step_name: str,
        template_name: str,
        part_id: Optional[int],
        metadata: Dict[str, Any],
        runtime_parameters: Dict[str, Any],
        detailed_selector: str = "first",
    ):
        self.level = (level or "basic").strip().lower()
        self.enabled = self.level != "off"
        self.sample_interval_ms = max(100, int(sample_interval_ms))
        self.step_name = step_name
        self.template_name = template_name
        self.part_id = part_id
        self.metadata = dict(metadata)
        self.runtime_parameters = dict(runtime_parameters)
        self.detailed_selector = detailed_selector or "first"
        self.detailed_enabled = self.level == "detailed" and should_profile_partition(
            self.detailed_selector,
            self.part_id,
        )
        self.status = "running"
        self.profiler_failures: List[Dict[str, Any]] = []
        self.notes: List[str] = [
            "Phase durations may overlap; preprocessing includes worker-side read and decode time.",
            "Cache fill and local write metrics may overlap when remote data is persisted into the local cache.",
        ]
        self.metrics: Dict[str, Any] = {}
        self.flags: Dict[str, str] = {}
        self.semantics: Dict[str, str] = {}
        self.reasons: Dict[str, str] = {}
        self.samples: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._start_monotonic = time.monotonic()
        self._start_utc = utc_now_iso()
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._cpu_count = psutil.cpu_count() if psutil is not None else os.cpu_count() or 1
        self._last_cpu_total: Optional[float] = None
        self._last_sample_monotonic: Optional[float] = None
        self._baseline_stats = self._collect_process_tree_stats()
        self._end_stats: Optional[Dict[str, Any]] = None
        self._system_iowait_start = self._get_system_iowait()
        self._torch_profiler = None
        self._nvml_handle = None
        self._torch_trace_path: Optional[Path] = None
        self._torch_ops_path: Optional[Path] = None
        self._torch_memory_path: Optional[Path] = None

        pod_name = str(self.metadata.get("pod_name") or f"pid-{os.getpid()}")
        raw_base = None
        if raw_root:
            raw_base = Path(raw_root) / self.step_name
            if self.part_id is not None:
                raw_base = raw_base / f"part-{int(self.part_id):03d}"
            raw_base = raw_base / f"pod-{pod_name}"
            raw_base.mkdir(parents=True, exist_ok=True)
        self.raw_dir = raw_base
        self.local_dir = maybe_mkdir(local_root)
        self.worker_summary_dir = maybe_mkdir(self.local_dir / "workers")
        os.environ["PROFILING_WORKER_SUMMARY_DIR"] = str(self.worker_summary_dir)
        os.environ["PROFILING_LEVEL"] = self.level

        self.summary_path = self.local_dir / PROFILE_SUMMARY_NAME
        self.timeseries_path = self.local_dir / PROFILE_TIMESERIES_NAME
        self._seed_unavailable_metrics()
        self._init_optional_gpu()
        self.start()

    def _seed_unavailable_metrics(self) -> None:
        for name in DURATION_METRICS:
            self.mark_unavailable(name, "not recorded in this step", semantics=SEMANTICS_DURATION)
        for name in COUNTER_METRICS:
            self.mark_unavailable(name, "not recorded in this step", semantics=SEMANTICS_COUNTER)
        for name in STAT_METRICS:
            self.mark_unavailable(name, "not recorded in this step", semantics=SEMANTICS_SAMPLE)
        self.mark_unavailable("gpu_temperature_celsius_max", "GPU telemetry not initialized", semantics=SEMANTICS_PEAK)
        self.mark_unavailable("gpu_power_watts_max", "GPU telemetry not initialized", semantics=SEMANTICS_PEAK)

    def _record_profiler_failure(self, component: str, exc: BaseException) -> None:
        self.profiler_failures.append(
            {
                "component": component,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    def _init_optional_gpu(self) -> None:
        if self.level == "off":
            return
        if pynvml is None:
            self.mark_unavailable("gpu_utilization_percent_avg", "NVML unavailable")
            self.mark_unavailable("vram_used_bytes_avg", "NVML unavailable")
            return
        try:
            pynvml.nvmlInit()
            index = 0
            visible = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            if visible and visible != "NoDevFiles":
                with suppress(ValueError):
                    index = int(visible)
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            self._record_profiler_failure("nvml_init", exc)
            self.mark_unavailable("gpu_utilization_percent_avg", f"NVML unavailable: {exc}")
            self.mark_unavailable("vram_used_bytes_avg", f"NVML unavailable: {exc}")

    def start(self) -> None:
        global _ACTIVE_PROFILER
        _ACTIVE_PROFILER = self
        if self.enabled:
            self._start_sampler()

    def stop(self) -> None:
        global _ACTIVE_PROFILER
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=max(2.0, self.sample_interval_ms / 1000.0 * 2.0))
        self._end_stats = self._collect_process_tree_stats()
        _ACTIVE_PROFILER = None

    def scope(
        self,
        metric_name: str,
        flag: str = FLAG_EXACT,
        semantics: str = SEMANTICS_DURATION,
        cuda_sync: bool = False,
    ):
        return _ScopeTimer(self, metric_name, flag=flag, semantics=semantics, cuda_sync=cuda_sync)

    def add_duration(
        self,
        metric_name: str,
        seconds: float,
        flag: str = FLAG_EXACT,
        semantics: str = SEMANTICS_DURATION,
    ) -> None:
        with self._lock:
            self.metrics[metric_name] = float(self.metrics.get(metric_name, 0.0) or 0.0) + float(seconds)
            self.flags[metric_name] = flag
            self.semantics[metric_name] = semantics
            self.reasons.pop(metric_name, None)

    def increment_counter(
        self,
        metric_name: str,
        delta: int | float,
        flag: str = FLAG_EXACT,
        semantics: str = SEMANTICS_COUNTER,
    ) -> None:
        with self._lock:
            current = self.metrics.get(metric_name, 0)
            if current is None:
                current = 0
            self.metrics[metric_name] = current + delta
            self.flags[metric_name] = flag
            self.semantics[metric_name] = semantics
            self.reasons.pop(metric_name, None)

    def set_metric(
        self,
        metric_name: str,
        value: Any,
        flag: str = FLAG_EXACT,
        semantics: str = SEMANTICS_SAMPLE,
    ) -> None:
        with self._lock:
            self.metrics[metric_name] = value
            self.flags[metric_name] = flag
            self.semantics[metric_name] = semantics
            self.reasons.pop(metric_name, None)

    def set_peak(self, metric_name: str, value: Any) -> None:
        if value is None:
            return
        existing = self.metrics.get(metric_name)
        if existing is None or value > existing:
            self.set_metric(metric_name, value, flag=FLAG_EXACT, semantics=SEMANTICS_PEAK)

    def mark_unavailable(
        self,
        metric_name: str,
        reason: str,
        semantics: str = SEMANTICS_SAMPLE,
    ) -> None:
        with self._lock:
            if metric_name not in self.metrics or self.flags.get(metric_name) == FLAG_UNAVAILABLE:
                self.metrics[metric_name] = None
                self.flags[metric_name] = FLAG_UNAVAILABLE
                self.semantics[metric_name] = semantics
                self.reasons[metric_name] = reason

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def register_artifact(self, name: str, path: str | Path) -> None:
        self.artifacts[name] = str(path)

    def record_store_event(
        self,
        category: str,
        seconds: float,
        num_bytes: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        if category == "remote_read":
            self.add_duration("remote_read_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("remote_read_bytes", num_bytes)
        elif category == "local_read":
            self.add_duration("local_read_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("local_read_bytes", num_bytes)
        elif category == "cache_fill":
            self.add_duration("cache_fill_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("cache_fill_bytes", num_bytes)
                self.increment_counter("local_write_bytes", num_bytes)
        elif category == "local_write":
            self.add_duration("local_write_seconds", seconds, flag=FLAG_APPROXIMATE)
            if num_bytes is not None:
                self.increment_counter("local_write_bytes", num_bytes)
        elif category == "cache_hit":
            self.increment_counter("cache_hits", 1)
        elif category == "cache_miss":
            self.increment_counter("cache_misses", 1)
        elif category == "cache_negative_hit":
            self.increment_counter("cache_negative_hits", 1)
        if reason:
            self.add_note(reason)

    def _start_sampler(self) -> None:
        if psutil is None:
            self._record_profiler_failure("psutil_import", RuntimeError("psutil unavailable"))
            self.mark_unavailable("process_cpu_utilization_percent_avg", "psutil unavailable")
            self.mark_unavailable("process_rss_bytes_avg", "psutil unavailable")
            return
        with suppress(Exception):
            psutil.cpu_percent(interval=None)
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            name=f"profiling-sampler-{os.getpid()}",
            daemon=True,
        )
        self._sampler_thread.start()

    def _sampler_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_ms / 1000.0):
            try:
                sample = self._sample_once()
            except Exception as exc:
                self._record_profiler_failure("sampler", exc)
                continue
            if sample:
                self.samples.append(sample)

    def _collect_process_tree_stats(self) -> Optional[Dict[str, Any]]:
        if self._process is None:
            return None
        stats: Dict[str, Any] = {
            "cpu_user": 0.0,
            "cpu_system": 0.0,
            "rss": 0,
            "vms": 0,
            "uss": 0,
            "pss": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "read_count": 0,
            "write_count": 0,
            "uss_available": False,
            "pss_available": False,
        }
        try:
            processes = [self._process] + self._process.children(recursive=True)
        except Exception:
            processes = [self._process]
        for proc in processes:
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                cpu = proc.cpu_times()
                stats["cpu_user"] += safe_float(getattr(cpu, "user", 0.0)) or 0.0
                stats["cpu_system"] += safe_float(getattr(cpu, "system", 0.0)) or 0.0
                mem = proc.memory_info()
                stats["rss"] += safe_int(getattr(mem, "rss", 0)) or 0
                stats["vms"] += safe_int(getattr(mem, "vms", 0)) or 0
                full = None
                with suppress(Exception):
                    full = proc.memory_full_info()
                if full is not None and hasattr(full, "uss"):
                    stats["uss"] += safe_int(getattr(full, "uss", 0)) or 0
                    stats["uss_available"] = True
                if full is not None and hasattr(full, "pss"):
                    stats["pss"] += safe_int(getattr(full, "pss", 0)) or 0
                    stats["pss_available"] = True
                io = None
                with suppress(Exception):
                    io = proc.io_counters()
                if io is not None:
                    stats["read_bytes"] += safe_int(getattr(io, "read_bytes", 0)) or 0
                    stats["write_bytes"] += safe_int(getattr(io, "write_bytes", 0)) or 0
                    stats["read_count"] += safe_int(getattr(io, "read_count", 0)) or 0
                    stats["write_count"] += safe_int(getattr(io, "write_count", 0)) or 0
        return stats

    def _get_system_iowait(self) -> Optional[float]:
        if psutil is None:
            return None
        with suppress(Exception):
            cpu = psutil.cpu_times()
            if hasattr(cpu, "iowait"):
                return float(cpu.iowait)
        return None

    def _sample_once(self) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        row: Dict[str, Any] = {
            "timestamp_utc": utc_now_iso(),
            "monotonic_seconds": now - self._start_monotonic,
        }

        stats = self._collect_process_tree_stats()
        if stats is not None:
            cpu_total = float(stats["cpu_user"]) + float(stats["cpu_system"])
            util = None
            if self._last_cpu_total is not None and self._last_sample_monotonic is not None:
                elapsed = max(now - self._last_sample_monotonic, 1e-6)
                util = ((cpu_total - self._last_cpu_total) / elapsed) * 100.0 / max(self._cpu_count or 1, 1)
                util = max(0.0, util)
            self._last_cpu_total = cpu_total
            self._last_sample_monotonic = now
            row["process_cpu_percent"] = util
            row["process_rss_bytes"] = stats["rss"]
            row["process_vms_bytes"] = stats["vms"]
            row["process_uss_bytes"] = stats["uss"] if stats["uss_available"] else None
            row["process_pss_bytes"] = stats["pss"] if stats["pss_available"] else None
            row["process_io_read_bytes"] = stats["read_bytes"]
            row["process_io_write_bytes"] = stats["write_bytes"]
            row["process_io_read_count"] = stats["read_count"]
            row["process_io_write_count"] = stats["write_count"]
        with suppress(Exception):
            row["system_cpu_percent"] = psutil.cpu_percent(interval=None) if psutil is not None else None

        if self._nvml_handle is not None:
            with suppress(Exception):
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                row["gpu_utilization_percent"] = float(util.gpu)
            with suppress(Exception):
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                row["vram_used_bytes"] = int(mem.used)
            with suppress(Exception):
                row["gpu_temperature_celsius"] = int(
                    pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                )
            with suppress(Exception):
                row["gpu_power_watts"] = float(pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)) / 1000.0

        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            with suppress(Exception):
                row["torch_cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
            with suppress(Exception):
                row["torch_cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved())
        return row

    def merge_worker_summaries(self) -> None:
        if not self.worker_summary_dir.exists():
            return
        aggregate: Dict[str, float] = defaultdict(float)
        for summary_path in self.worker_summary_dir.glob("worker-summary-*.json"):
            with suppress(Exception):
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                for name, entry in payload.get("metrics", {}).items():
                    value = entry.get("value")
                    if isinstance(value, (int, float)):
                        aggregate[name] += float(value)
                        if entry.get("flag") == FLAG_APPROXIMATE:
                            self.flags[name] = FLAG_APPROXIMATE
                        self.semantics[name] = entry.get("semantics", SEMANTICS_SAMPLE)
                        self.reasons.pop(name, None)
        for name, value in aggregate.items():
            if self.semantics.get(name) == SEMANTICS_COUNTER:
                self.increment_counter(
                    name,
                    int(value) if value.is_integer() else value,
                    flag=self.flags.get(name, FLAG_APPROXIMATE),
                    semantics=self.semantics.get(name, SEMANTICS_COUNTER),
                )
            else:
                self.add_duration(
                    name,
                    value,
                    flag=self.flags.get(name, FLAG_APPROXIMATE),
                    semantics=self.semantics.get(name, SEMANTICS_DURATION),
                )

    def enable_torch_profiler(self) -> bool:
        return self.detailed_enabled and torch is not None and hasattr(torch, "profiler")

    def start_torch_profiler(self, local_dir: Optional[str | Path] = None, use_cuda: bool = False) -> None:
        if not self.enable_torch_profiler():
            return
        if self._torch_profiler is not None:
            return
        output_dir = Path(local_dir) if local_dir is not None else self.local_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._torch_trace_path = output_dir / TORCH_TRACE_NAME
        self._torch_ops_path = output_dir / TORCH_OPS_SUMMARY_NAME
        self._torch_memory_path = output_dir / TORCH_MEMORY_SUMMARY_NAME
        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            self._torch_profiler = torch.profiler.profile(
                activities=activities,
                profile_memory=True,
                record_shapes=True,
                with_stack=False,
            )
            self._torch_profiler.__enter__()
        except Exception as exc:
            self._record_profiler_failure("torch_profiler_start", exc)
            self._torch_profiler = None

    def stop_torch_profiler(self) -> None:
        if self._torch_profiler is None:
            return
        try:
            profiler = self._torch_profiler
            profiler.__exit__(None, None, None)
            if self._torch_trace_path is not None:
                profiler.export_chrome_trace(str(self._torch_trace_path))
                self.register_artifact("torch_trace", self._torch_trace_path)
            if self._torch_ops_path is not None:
                sort_key = "cuda_time_total" if torch is not None and torch.cuda.is_available() else "self_cpu_time_total"
                table = profiler.key_averages().table(sort_by=sort_key, row_limit=64)
                self._torch_ops_path.write_text(table, encoding="utf-8")
                self.register_artifact("torch_ops_summary", self._torch_ops_path)
            if self._torch_memory_path is not None:
                memory_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "torch_cuda_max_memory_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated()) if torch is not None and torch.cuda.is_available() else None
                    ),
                    "torch_cuda_max_memory_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved()) if torch is not None and torch.cuda.is_available() else None
                    ),
                }
                write_json(self._torch_memory_path, memory_payload)
                self.register_artifact("torch_memory_summary", self._torch_memory_path)
        except Exception as exc:
            self._record_profiler_failure("torch_profiler_stop", exc)
        finally:
            self._torch_profiler = None

    def _finalize_summary_metrics(self) -> None:
        total_wall = time.monotonic() - self._start_monotonic
        self.set_metric("total_wall_seconds", total_wall, flag=FLAG_EXACT, semantics=SEMANTICS_DURATION)

        if self._baseline_stats is not None and self._end_stats is not None:
            self.set_metric(
                "process_cpu_user_seconds",
                max(0.0, float(self._end_stats["cpu_user"]) - float(self._baseline_stats["cpu_user"])),
                flag=FLAG_EXACT,
                semantics=SEMANTICS_COUNTER,
            )
            self.set_metric(
                "process_cpu_system_seconds",
                max(0.0, float(self._end_stats["cpu_system"]) - float(self._baseline_stats["cpu_system"])),
                flag=FLAG_EXACT,
                semantics=SEMANTICS_COUNTER,
            )
            self.set_metric(
                "process_io_read_bytes",
                max(0, int(self._end_stats["read_bytes"]) - int(self._baseline_stats["read_bytes"])),
                flag=FLAG_EXACT,
                semantics=SEMANTICS_COUNTER,
            )
            self.set_metric(
                "process_io_write_bytes",
                max(0, int(self._end_stats["write_bytes"]) - int(self._baseline_stats["write_bytes"])),
                flag=FLAG_EXACT,
                semantics=SEMANTICS_COUNTER,
            )
            self.set_metric(
                "process_io_read_count",
                max(0, int(self._end_stats["read_count"]) - int(self._baseline_stats["read_count"])),
                flag=FLAG_EXACT,
                semantics=SEMANTICS_COUNTER,
            )
            self.set_metric(
                "process_io_write_count",
                max(0, int(self._end_stats["write_count"]) - int(self._baseline_stats["write_count"])),
                flag=FLAG_EXACT,
                semantics=SEMANTICS_COUNTER,
            )

        iowait_end = self._get_system_iowait()
        if self._system_iowait_start is not None and iowait_end is not None:
            self.set_metric(
                "system_iowait_delta_seconds",
                max(0.0, iowait_end - self._system_iowait_start),
                flag=FLAG_APPROXIMATE,
                semantics=SEMANTICS_COUNTER,
            )

        process_cpu_values = [row["process_cpu_percent"] for row in self.samples if row.get("process_cpu_percent") is not None]
        system_cpu_values = [row["system_cpu_percent"] for row in self.samples if row.get("system_cpu_percent") is not None]
        rss_values = [row["process_rss_bytes"] for row in self.samples if row.get("process_rss_bytes") is not None]
        uss_values = [row["process_uss_bytes"] for row in self.samples if row.get("process_uss_bytes") is not None]
        pss_values = [row["process_pss_bytes"] for row in self.samples if row.get("process_pss_bytes") is not None]
        gpu_util_values = [row["gpu_utilization_percent"] for row in self.samples if row.get("gpu_utilization_percent") is not None]
        vram_values = [row["vram_used_bytes"] for row in self.samples if row.get("vram_used_bytes") is not None]
        gpu_temp_values = [row["gpu_temperature_celsius"] for row in self.samples if row.get("gpu_temperature_celsius") is not None]
        gpu_power_values = [row["gpu_power_watts"] for row in self.samples if row.get("gpu_power_watts") is not None]

        self._set_stat_triplet("process_cpu_utilization_percent", process_cpu_values, pct_flag=FLAG_APPROXIMATE)
        self._set_stat_triplet("system_cpu_utilization_percent", system_cpu_values, pct_flag=FLAG_APPROXIMATE)
        self._set_stat_triplet("process_rss_bytes", rss_values)
        self._set_stat_triplet("process_uss_bytes", uss_values)
        self._set_stat_triplet("process_pss_bytes", pss_values)
        self._set_stat_triplet("gpu_utilization_percent", gpu_util_values, pct_flag=FLAG_APPROXIMATE)
        self._set_stat_triplet("vram_used_bytes", vram_values)

        if rss_values:
            self.set_metric("process_rss_bytes_peak", max(rss_values), flag=FLAG_EXACT, semantics=SEMANTICS_PEAK)
        if vram_values:
            self.set_metric("vram_used_bytes_peak", max(vram_values), flag=FLAG_EXACT, semantics=SEMANTICS_PEAK)
        if gpu_temp_values:
            self.set_metric("gpu_temperature_celsius_max", max(gpu_temp_values), flag=FLAG_APPROXIMATE, semantics=SEMANTICS_PEAK)
        if gpu_power_values:
            self.set_metric("gpu_power_watts_max", max(gpu_power_values), flag=FLAG_APPROXIMATE, semantics=SEMANTICS_PEAK)

        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            with suppress(Exception):
                self.set_metric(
                    "torch_cuda_max_memory_allocated_bytes",
                    int(torch.cuda.max_memory_allocated()),
                    flag=FLAG_EXACT,
                    semantics=SEMANTICS_PEAK,
                )
            with suppress(Exception):
                self.set_metric(
                    "torch_cuda_max_memory_reserved_bytes",
                    int(torch.cuda.max_memory_reserved()),
                    flag=FLAG_EXACT,
                    semantics=SEMANTICS_PEAK,
                )

    def _set_stat_triplet(self, prefix: str, values: Sequence[float], pct_flag: str = FLAG_EXACT) -> None:
        if not values:
            self.mark_unavailable(f"{prefix}_avg", f"{prefix} unavailable")
            self.mark_unavailable(f"{prefix}_max", f"{prefix} unavailable")
            self.mark_unavailable(f"{prefix}_p95", f"{prefix} unavailable")
            return
        self.set_metric(f"{prefix}_avg", statistics.fmean(values), flag=pct_flag, semantics=SEMANTICS_SAMPLE)
        self.set_metric(f"{prefix}_max", max(values), flag=pct_flag, semantics=SEMANTICS_PEAK)
        self.set_metric(f"{prefix}_p95", percentile(values, 95.0), flag=pct_flag, semantics=SEMANTICS_SAMPLE)

    def flush(self, status: str, error: Optional[BaseException] = None) -> None:
        self.status = status
        if error is not None:
            self.profiler_failures.append(
                {
                    "component": "workload",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                }
            )
        self.stop_torch_profiler()
        self.merge_worker_summaries()
        self.stop()
        self._finalize_summary_metrics()

        summary_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "profiling": {
                "level": self.level,
                "enabled": self.enabled,
                "detailed_enabled": self.detailed_enabled,
                "sample_interval_ms": self.sample_interval_ms,
                "detailed_selector": self.detailed_selector,
                "profiler_failures": self.profiler_failures,
                "notes": self.notes,
            },
            "identifiers": {
                **self.metadata,
                "template_name": self.template_name,
                "step_name": self.step_name,
                "part_id": self.part_id,
            },
            "runtime_parameters": self.runtime_parameters,
            "timestamps": {
                "start_utc": self._start_utc,
                "end_utc": utc_now_iso(),
            },
            "metrics": {
                name: metric_entry(
                    self.metrics.get(name),
                    self.flags.get(name, FLAG_EXACT),
                    self.semantics.get(name, SEMANTICS_SAMPLE),
                    self.reasons.get(name),
                )
                for name in sorted(self.metrics.keys())
            },
            "artifacts": self.artifacts,
        }

        write_json(self.summary_path, summary_payload)
        if self.samples:
            write_jsonl(self.timeseries_path, self.samples)
        elif self.level == "off":
            write_jsonl(
                self.timeseries_path,
                [
                    {
                        "timestamp_utc": utc_now_iso(),
                        "profiling_enabled": False,
                    }
                ],
            )

        if self.raw_dir is not None:
            copy_if_exists(self.summary_path, self.raw_dir / PROFILE_SUMMARY_NAME)
            copy_if_exists(self.timeseries_path, self.raw_dir / PROFILE_TIMESERIES_NAME)
            for name, path in self.artifacts.items():
                source = Path(path)
                if source.exists():
                    copy_if_exists(source, self.raw_dir / source.name)


def collect_env_metadata(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = {
        "workflow_name": os.getenv("WORKFLOW_NAME", ""),
        "workflow_uid": os.getenv("WORKFLOW_UID", ""),
        "namespace": os.getenv("WORKFLOW_NAMESPACE", os.getenv("POD_NAMESPACE", "")),
        "pod_name": os.getenv("POD_NAME", ""),
        "node_name": os.getenv("NODE_NAME", ""),
        "model": os.getenv("MODEL", ""),
        "model_type": os.getenv("MODEL_TYPE", ""),
        "cpu_request": os.getenv("CPU_REQUEST", ""),
        "cpu_limit": os.getenv("CPU_LIMIT", ""),
        "memory_request": os.getenv("MEMORY_REQUEST", ""),
        "memory_limit": os.getenv("MEMORY_LIMIT", ""),
        "gpu_request": os.getenv("GPU_REQUEST", ""),
        "gpu_limit": os.getenv("GPU_LIMIT", ""),
    }
    if extra:
        metadata.update(extra)
    return metadata


def build_runtime_parameters(**kwargs: Any) -> Dict[str, Any]:
    payload = {}
    for key, value in kwargs.items():
        if value not in (None, ""):
            payload[key] = value
    return payload


def select_latest_record(summaries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not summaries:
        return None
    def _sort_key(summary: Dict[str, Any]) -> tuple[int, str]:
        status = (summary.get("status") or "").lower()
        success = 1 if status in {"succeeded", "completed", "ok"} else 0
        end_utc = summary.get("timestamps", {}).get("end_utc") or ""
        return (success, end_utc)
    return sorted(summaries, key=_sort_key)[-1]


def metric_value(summary: Dict[str, Any], metric_name: str) -> Optional[float]:
    entry = summary.get("metrics", {}).get(metric_name)
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    return safe_float(value)


def metric_flag(summary: Dict[str, Any], metric_name: str) -> Optional[str]:
    entry = summary.get("metrics", {}).get(metric_name)
    if not isinstance(entry, dict):
        return None
    return entry.get("flag")


def dominant_phase(summary: Dict[str, Any]) -> Optional[str]:
    candidates = [
        "download_seconds",
        "remote_read_seconds",
        "local_read_seconds",
        "preprocess_seconds",
        "model_load_seconds",
        "compile_warmup_seconds",
        "forward_seconds",
        "reduce_seconds",
        "local_write_seconds",
        "upload_seconds",
    ]
    winner = None
    best = -1.0
    for name in candidates:
        value = metric_value(summary, name)
        if value is not None and value > best:
            best = value
            winner = name
    return winner


def classify_partition(summary: Dict[str, Any]) -> str:
    wall = metric_value(summary, "total_wall_seconds") or 0.0
    if wall <= 0.0:
        return "mixed"
    compile_seconds = metric_value(summary, "compile_warmup_seconds") or 0.0
    network = sum(
        metric_value(summary, name) or 0.0
        for name in ("s3_list_seconds", "download_seconds", "upload_seconds", "remote_read_seconds")
    )
    local_io = sum(
        metric_value(summary, name) or 0.0
        for name in ("local_read_seconds", "local_write_seconds", "zarr_write_seconds", "cache_fill_seconds")
    )
    forward = metric_value(summary, "forward_seconds") or 0.0
    gpu_avg = metric_value(summary, "gpu_utilization_percent_avg") or 0.0
    cpu_avg = metric_value(summary, "process_cpu_utilization_percent_avg") or 0.0
    rss_peak = metric_value(summary, "process_rss_bytes_peak") or 0.0
    vram_peak = metric_value(summary, "vram_used_bytes_peak") or 0.0
    memory_limit = parse_bytes_string(summary.get("identifiers", {}).get("memory_limit"))

    if compile_seconds / wall >= 0.20:
        return "compile-bound"
    if network >= max(local_io, forward, 0.0) and network / wall >= 0.25:
        return "s3/network-bound"
    if local_io >= max(network, forward, 0.0) and local_io / wall >= 0.25:
        return "efs/local-io-bound"
    if forward >= max(network, local_io, 0.0) and gpu_avg >= 70.0:
        return "gpu-bound"
    if memory_limit and rss_peak >= 0.85 * memory_limit:
        return "memory-bound"
    if vram_peak > 0:
        total_vram = summary.get("gpu_metadata", {}).get("total_vram_bytes")
        if isinstance(total_vram, (int, float)) and vram_peak >= 0.85 * float(total_vram):
            return "memory-bound"
    if cpu_avg >= 70.0:
        return "cpu-bound"
    return "mixed"


def parse_bytes_string(raw: Any) -> Optional[int]:
    if raw in (None, "", 0):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip()
    if not text:
        return None
    units = {
        "ki": 1024,
        "mi": 1024 ** 2,
        "gi": 1024 ** 3,
        "ti": 1024 ** 4,
        "k": 1000,
        "m": 1000 ** 2,
        "g": 1000 ** 3,
        "t": 1000 ** 4,
    }
    lower = text.lower()
    for suffix, factor in units.items():
        if lower.endswith(suffix):
            return int(float(lower[:-len(suffix)]) * factor)
    with suppress(ValueError):
        return int(float(lower))
    return None


def aggregate_workflow_profiling(
    raw_root: str | Path,
    output_dir: str | Path,
    output_prefix: str = "",
) -> Dict[str, Any]:
    raw_path = Path(raw_root)
    out_dir = maybe_mkdir(output_dir)
    summaries_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    workflow_metadata: Dict[str, Any] = {}

    for summary_path in raw_path.rglob(PROFILE_SUMMARY_NAME):
        with suppress(Exception):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            identifiers = payload.get("identifiers", {})
            template_name = identifiers.get("template_name") or identifiers.get("step_name") or "unknown"
            part_id = identifiers.get("part_id")
            logical_key = f"{template_name}:{part_id if part_id is not None else 'none'}"
            summaries_by_key[logical_key].append(payload)
            if not workflow_metadata and identifiers:
                workflow_metadata = identifiers

    selected = [record for record in (select_latest_record(items) for items in summaries_by_key.values()) if record is not None]
    partition_summaries = [
        summary for summary in selected
        if summary.get("identifiers", {}).get("part_id") is not None
    ]
    for summary in partition_summaries:
        throughput = metric_value(summary, "partition_throughput_tiles_per_second")
        if throughput is None:
            wall = metric_value(summary, "total_wall_seconds") or 0.0
            tiles = metric_value(summary, "partition_tiles") or 0.0
            if wall > 0 and tiles > 0:
                throughput = tiles / wall
                summary["metrics"]["partition_throughput_tiles_per_second"] = metric_entry(
                    throughput,
                    FLAG_ESTIMATED,
                    SEMANTICS_SAMPLE,
                )

    def _stats(metric_name: str) -> Dict[str, Optional[float]]:
        values = [metric_value(summary, metric_name) for summary in partition_summaries]
        numbers = [value for value in values if value is not None]
        if not numbers:
            return {"p50": None, "p95": None, "max": None}
        return {
            "p50": percentile(numbers, 50.0),
            "p95": percentile(numbers, 95.0),
            "max": max(numbers),
        }

    slowest = sorted(
        partition_summaries,
        key=lambda summary: metric_value(summary, "total_wall_seconds") or -1.0,
        reverse=True,
    )[:5]
    slowest_rows = []
    for summary in slowest:
        identifiers = summary.get("identifiers", {})
        slowest_rows.append(
            {
                "part_id": identifiers.get("part_id"),
                "pod_name": identifiers.get("pod_name"),
                "wall_seconds": metric_value(summary, "total_wall_seconds"),
                "dominant_phase": dominant_phase(summary),
                "classification": classify_partition(summary),
            }
        )

    partition_rows: List[Dict[str, Any]] = []
    for summary in sorted(partition_summaries, key=lambda item: int(item.get("identifiers", {}).get("part_id") or 0)):
        identifiers = summary.get("identifiers", {})
        row = {
            "part_id": identifiers.get("part_id"),
            "pod_name": identifiers.get("pod_name"),
            "node_name": identifiers.get("node_name"),
            "total_wall_seconds": metric_value(summary, "total_wall_seconds"),
            "download_seconds": metric_value(summary, "download_seconds"),
            "upload_seconds": metric_value(summary, "upload_seconds"),
            "remote_read_seconds": metric_value(summary, "remote_read_seconds"),
            "local_read_seconds": metric_value(summary, "local_read_seconds"),
            "local_write_seconds": metric_value(summary, "local_write_seconds"),
            "compile_warmup_seconds": metric_value(summary, "compile_warmup_seconds"),
            "forward_seconds": metric_value(summary, "forward_seconds"),
            "peak_rss_bytes": metric_value(summary, "process_rss_bytes_peak"),
            "peak_vram_bytes": metric_value(summary, "vram_used_bytes_peak"),
            "gpu_utilization_percent_avg": metric_value(summary, "gpu_utilization_percent_avg"),
            "gpu_utilization_percent_max": metric_value(summary, "gpu_utilization_percent_max"),
            "partition_throughput_tiles_per_second": metric_value(summary, "partition_throughput_tiles_per_second"),
            "dominant_phase": dominant_phase(summary),
            "classification": classify_partition(summary),
        }
        partition_rows.append(row)

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "raw_root": str(raw_path),
        "workflow_identifiers": workflow_metadata,
        "selected_records": len(selected),
        "partition_count": len(partition_rows),
        "partition_statistics": {
            "partition_wall_time_seconds": _stats("total_wall_seconds"),
            "download_seconds": _stats("download_seconds"),
            "upload_seconds": _stats("upload_seconds"),
            "local_read_seconds": _stats("local_read_seconds"),
            "local_write_seconds": _stats("local_write_seconds"),
            "peak_rss_bytes": _stats("process_rss_bytes_peak"),
            "peak_gpu_memory_bytes": _stats("vram_used_bytes_peak"),
            "avg_gpu_utilization_percent": _stats("gpu_utilization_percent_avg"),
            "max_gpu_utilization_percent": _stats("gpu_utilization_percent_max"),
            "compile_warmup_seconds": _stats("compile_warmup_seconds"),
            "steady_state_forward_seconds": _stats("forward_seconds"),
            "throughput_tiles_per_second": _stats("partition_throughput_tiles_per_second"),
        },
        "top_slowest_partitions": slowest_rows,
        "partitions": partition_rows,
    }

    markdown_lines = [
        "# Workflow Profiling Summary",
        "",
        f"- Records aggregated: {len(selected)}",
        f"- Partitions aggregated: {len(partition_rows)}",
        "",
        "## Top Slowest Partitions",
    ]
    if slowest_rows:
        for row in slowest_rows:
            markdown_lines.append(
                f"- part {row['part_id']}: {row['wall_seconds']:.2f}s, dominant={row['dominant_phase']}, bottleneck={row['classification']}"
            )
    else:
        markdown_lines.append("- none")
    markdown_lines.extend(
        [
            "",
            "## Partition Wall Time",
            f"- p50: {summary_payload['partition_statistics']['partition_wall_time_seconds']['p50']}",
            f"- p95: {summary_payload['partition_statistics']['partition_wall_time_seconds']['p95']}",
            f"- max: {summary_payload['partition_statistics']['partition_wall_time_seconds']['max']}",
        ]
    )

    summary_json_path = out_dir / WORKFLOW_SUMMARY_JSON
    summary_md_path = out_dir / WORKFLOW_SUMMARY_MD
    partitions_csv_path = out_dir / WORKFLOW_PARTITIONS_CSV
    partitions_jsonl_path = out_dir / WORKFLOW_PARTITIONS_JSONL

    write_json(summary_json_path, summary_payload)
    summary_md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    with partitions_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(partition_rows[0].keys()) if partition_rows else ["part_id"])
        writer.writeheader()
        if partition_rows:
            writer.writerows(partition_rows)
    write_jsonl(partitions_jsonl_path, partition_rows)

    if output_prefix:
        copy_bundle_to_prefix(
            output_prefix,
            {
                WORKFLOW_SUMMARY_JSON: summary_json_path,
                WORKFLOW_SUMMARY_MD: summary_md_path,
                WORKFLOW_PARTITIONS_CSV: partitions_csv_path,
                WORKFLOW_PARTITIONS_JSONL: partitions_jsonl_path,
            },
        )

    return {
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "partitions_csv": str(partitions_csv_path),
        "partitions_jsonl": str(partitions_jsonl_path),
    }


def copy_bundle_to_prefix(prefix: str, files: Dict[str, str | Path]) -> None:
    if not prefix:
        return
    if prefix.startswith("s3://"):
        import boto3

        bucket, key_prefix = parse_s3_uri(prefix)
        s3_client = boto3.client("s3")
        for name, file_path in files.items():
            source = Path(file_path)
            if not source.exists():
                continue
            key = f"{key_prefix.rstrip('/')}/{name}" if key_prefix else name
            s3_client.upload_file(str(source), bucket, key)
        return

    destination = Path(prefix)
    destination.mkdir(parents=True, exist_ok=True)
    for name, file_path in files.items():
        copy_if_exists(file_path, destination / name)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"invalid s3 uri: {s3_uri}")
    path = s3_uri[5:]
    bucket, _, key = path.partition("/")
    return bucket, key
