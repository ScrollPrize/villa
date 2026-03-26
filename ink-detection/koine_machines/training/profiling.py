from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext

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
