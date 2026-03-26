from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext
from contextlib import contextmanager
from dataclasses import dataclass

import torch


class NormalPoolingProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.section_totals = defaultdict(float)
        self.step_count = 0

    def start_step(self):
        if self.enabled:
            self.step_count += 1

    def add_duration(self, name: str, duration_seconds: float):
        if not self.enabled:
            return
        self.section_totals[name] += float(duration_seconds)

    def _synchronize(self, device: torch.device | None):
        if not self.enabled or device is None or device.type != 'cuda' or not torch.cuda.is_available():
            return
        torch.cuda.synchronize(device)

    def section(self, name: str, device: torch.device | None):
        if not self.enabled:
            return nullcontext()
        profiler = self

        class _TimedSection:
            def __enter__(self):
                profiler._synchronize(device)
                self._start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                profiler._synchronize(device)
                profiler.section_totals[name] += time.perf_counter() - self._start
                return False

        return _TimedSection()

    def summary_lines(self) -> list[str]:
        if not self.enabled or self.step_count == 0:
            return []

        lines = [f"[profile] profiled timings across {self.step_count} training steps"]
        for name, total_seconds in sorted(self.section_totals.items(), key=lambda item: item[1], reverse=True):
            avg_ms = (total_seconds / self.step_count) * 1000.0
            lines.append(
                f"[profile] {name}: total={total_seconds:.6f}s avg_step={avg_ms:.3f}ms"
            )
        return lines


class DatasetSampleProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.section_totals = defaultdict(float)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.section_totals[name] += time.perf_counter() - start

    def as_dict(self) -> dict[str, float]:
        return dict(self.section_totals)


@dataclass
class _BackwardProfileMarkers:
    profiler: NormalPoolingProfiler
    device: torch.device
    _start_time: float | None = None
    _loss_preds_ready_time: float | None = None
    _preds_ready_time: float | None = None
    _handles: tuple = ()
    _preds_is_loss_preds: bool = False

    @classmethod
    def create(cls, profiler, preds: torch.Tensor, loss_preds: torch.Tensor):
        if profiler is None or not profiler.enabled:
            return None
        if not isinstance(preds, torch.Tensor) or not preds.requires_grad:
            return None
        if not isinstance(loss_preds, torch.Tensor) or not loss_preds.requires_grad:
            return None

        markers = cls(
            profiler=profiler,
            device=preds.device,
            _preds_is_loss_preds=preds is loss_preds,
        )
        handles = []

        def _record_loss_preds_ready(grad):
            markers.profiler._synchronize(markers.device)
            if markers._start_time is not None and markers._loss_preds_ready_time is None:
                markers._loss_preds_ready_time = time.perf_counter()
                if markers._preds_is_loss_preds and markers._preds_ready_time is None:
                    markers._preds_ready_time = markers._loss_preds_ready_time
            return grad

        def _record_preds_ready(grad):
            markers.profiler._synchronize(markers.device)
            if markers._start_time is not None and markers._preds_ready_time is None:
                markers._preds_ready_time = time.perf_counter()
            return grad

        handles.append(loss_preds.register_hook(_record_loss_preds_ready))
        if preds is not loss_preds:
            handles.append(preds.register_hook(_record_preds_ready))

        markers._handles = tuple(handles)
        return markers

    def start(self):
        self.profiler._synchronize(self.device)
        self._start_time = time.perf_counter()

    def finish(self):
        if self._start_time is None:
            self.close()
            return

        self.profiler._synchronize(self.device)
        end_time = time.perf_counter()
        self.profiler.add_duration('train/backward_total', end_time - self._start_time)

        if self._loss_preds_ready_time is not None:
            self.profiler.add_duration(
                'loss/backward',
                self._loss_preds_ready_time - self._start_time,
            )

        if self._preds_ready_time is not None:
            self.profiler.add_duration(
                'model/backward',
                end_time - self._preds_ready_time,
            )
            if self._loss_preds_ready_time is not None:
                self.profiler.add_duration(
                    'normal_pooling/backward',
                    self._preds_ready_time - self._loss_preds_ready_time,
                )

        self.close()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = ()
