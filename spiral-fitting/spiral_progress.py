"""Shared, dependency-free progress reporting for Spiral operations."""

from __future__ import annotations

import copy
import sys
import threading
import time
from typing import Callable, Mapping, TextIO


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class ProgressReporter:
    """Own one active stage and publish rate-limited status snapshots.

    Updating the reporter never synchronizes CUDA or inspects worker state.
    ``snapshot`` derives elapsed time from the monotonic clock, so a polling
    client continues to see useful elapsed time while an opaque native call is
    in flight.
    """

    def __init__(
        self,
        publish: Callable[[Mapping[str, object] | None], None] | None = None,
        *,
        stream: TextIO | None = None,
        publish_interval: float = 1.0,
        console_interval: float = 5.0,
        heartbeat_interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._publish = publish
        self._stream = stream
        self._publish_interval = float(publish_interval)
        self._console_interval = float(console_interval)
        self._heartbeat_interval = float(heartbeat_interval)
        self._clock = clock
        self._lock = threading.Lock()
        self._active: dict[str, object] | None = None
        self._started = 0.0
        self._last_publish = float("-inf")
        self._last_console = float("-inf")
        self._closed = False
        self._tty = bool(stream is not None and getattr(stream, "isatty", lambda: False)())
        self._heartbeat_thread: threading.Thread | None = None
        if stream is not None and heartbeat_interval > 0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_main,
                name="spiral-progress-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def begin(
        self,
        operation: str,
        stage_name: str,
        *,
        step: int | None = None,
        total_steps: int | None = None,
        unit: str | None = None,
        detail: str | None = None,
    ) -> None:
        now = self._clock()
        with self._lock:
            self._started = now
            self._active = {
                "operation": str(operation),
                "stage_name": str(stage_name),
                "detail": None if detail is None else str(detail),
                "step": None if step is None else int(step),
                "total_steps": (
                    None if total_steps is None else max(0, int(total_steps))
                ),
                "unit": None if unit is None else str(unit),
            }
        self._emit(force=True)

    def update(
        self,
        step: int | None = None,
        *,
        total_steps: int | None = None,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            if self._active is None:
                return
            if step is not None:
                self._active["step"] = int(step)
            if total_steps is not None:
                self._active["total_steps"] = max(0, int(total_steps))
            if detail is not None:
                self._active["detail"] = str(detail)
        self._emit(force=False)

    def pulse(self, detail: str | None = None) -> None:
        self.update(detail=detail)

    def finish(self, detail: str | None = None) -> None:
        with self._lock:
            if self._active is None:
                return
            total = self._active.get("total_steps")
            if total is not None:
                self._active["step"] = int(total)
            if detail is not None:
                self._active["detail"] = str(detail)
        self._emit(force=True)

    def clear(self) -> None:
        with self._lock:
            had_active = self._active is not None
            self._active = None
        if had_active:
            if self._tty and self._stream is not None:
                self._stream.write("\n")
                self._stream.flush()
            if self._publish is not None:
                self._publish(None)

    def snapshot(self) -> dict[str, object] | None:
        now = self._clock()
        with self._lock:
            if self._active is None:
                return None
            result = copy.deepcopy(self._active)
            elapsed = max(0.0, now - self._started)
        result["elapsed_seconds"] = elapsed
        step = result.get("step")
        total = result.get("total_steps")
        eta = None
        if (
            isinstance(step, int)
            and isinstance(total, int)
            and step > 0
            and total > step
            and elapsed >= 2.0
        ):
            eta = elapsed * (total - step) / step
        elif (
            isinstance(step, int)
            and isinstance(total, int)
            and total > 0
            and step >= total
        ):
            eta = 0.0
        result["eta_seconds"] = eta
        return result

    def close(self) -> None:
        self._closed = True
        self.clear()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)

    def _emit(self, *, force: bool) -> None:
        now = self._clock()
        snapshot = self.snapshot()
        if snapshot is None:
            return
        publish = force or now - self._last_publish >= self._publish_interval
        console = force or now - self._last_console >= self._console_interval
        if publish and self._publish is not None:
            self._last_publish = now
            self._publish(snapshot)
        if console and self._stream is not None:
            self._last_console = now
            self._write_console(snapshot)

    def _write_console(self, snapshot: Mapping[str, object]) -> None:
        stage = str(snapshot["stage_name"])
        detail = snapshot.get("detail")
        step, total = snapshot.get("step"), snapshot.get("total_steps")
        unit = snapshot.get("unit")
        elapsed = float(snapshot["elapsed_seconds"])
        parts = [f"PROGRESS {stage}"]
        if isinstance(step, int) and isinstance(total, int) and total > 0:
            label = f"{step:,}/{total:,}"
            if unit:
                label += f" {unit}"
            label += f" ({100.0 * min(step, total) / total:.1f}%)"
            parts.append(label)
        elif detail:
            parts.append(str(detail))
        if unit == "iterations" and isinstance(step, int) and step > 0 and elapsed > 0:
            parts.append(f"{step / elapsed:.1f} it/s")
        parts.append(f"elapsed {_format_duration(elapsed)}")
        eta = snapshot.get("eta_seconds")
        if isinstance(eta, (int, float)):
            parts.append(f"ETA {_format_duration(float(eta))}")
        line = " — ".join(parts)
        if self._tty:
            self._stream.write("\r" + line)
        else:
            self._stream.write(line + "\n")
        self._stream.flush()

    def _heartbeat_main(self) -> None:
        while not self._closed:
            time.sleep(min(1.0, self._heartbeat_interval))
            if self._closed:
                return
            now = self._clock()
            with self._lock:
                active = self._active is not None
                due = now - self._last_console >= self._heartbeat_interval
            if active and due:
                snapshot = self.snapshot()
                if snapshot is not None and self._stream is not None:
                    self._last_console = now
                    self._write_console(snapshot)


class NullProgressReporter:
    """No-op reporter used by helper callers that do not request progress."""

    def begin(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def pulse(self, *args, **kwargs) -> None:
        pass

    def finish(self, *args, **kwargs) -> None:
        pass

    def clear(self) -> None:
        pass

    def snapshot(self):
        return None

    def close(self) -> None:
        pass


def progress_or_null(progress):
    return progress if progress is not None else NullProgressReporter()
