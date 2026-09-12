"""Memory observations from real training boundaries, without replayed updates.

This controller is CPU-only. The supervisor owns exclusive authorization, cohort
handoffs and persistence; GPU workers report peaks from the ordinary trainer.
"""

from copy import deepcopy

from .memory_calibration import calibration_from_observations


class StartupMemoryCalibration:
    def __init__(self, fingerprint, *, target_microbatch, reserve_bytes=4 * 1024**3,
                 updates_per_size=3):
        if type(target_microbatch) is not int or target_microbatch < 1:
            raise ValueError("startup target microbatch must be positive")
        if type(updates_per_size) is not int or updates_per_size < 2:
            raise ValueError("startup measurement needs at least two real optimizer updates")
        self.fingerprint = fingerprint
        self.target = target_microbatch
        self.reserve_bytes = reserve_bytes
        self.updates_per_size = updates_per_size
        self.microbatch_size = 1
        self.observations = []
        self.latest = None
        self.test_peaks = {}
        self.initial_test_completed = False
        self.ready = False

    def observe(self, rows, progress):
        if not rows or len({row["gpu_uuid"] for row in rows}) != len(rows):
            raise ValueError("startup needs exactly one observation per allocated GPU")
        if any(row["microbatch_size"] != self.microbatch_size or row["world_size"] != len(rows)
               or row["config_fingerprint"] != self.fingerprint for row in rows):
            raise ValueError("startup observation differs from the running cohort")
        if progress["phase"] == "test":
            for row in rows:
                self.test_peaks[row["gpu_uuid"]] = max(
                    self.test_peaks.get(row["gpu_uuid"], 0), row["peak_reserved_bytes"],
                    row["driver_peak_bytes"])
        self.initial_test_completed |= all(row["initial_test_completed"] for row in rows)
        self.latest = deepcopy(rows)
        for row in rows:
            if row.get("initial_test_peak_bytes", 0):
                self.test_peaks[row["gpu_uuid"]] = max(self.test_peaks.get(row["gpu_uuid"], 0),
                                                      row["initial_test_peak_bytes"])
        self.ready = (self.initial_test_completed and progress["phase"] == "update"
                      and min(row["optimizer_steps"] for row in rows) >= self.updates_per_size)

    def complete_size(self, grant):
        """Accept a completed size; return the next safely bounded candidate.

        The next candidate is *not* a measured grant estimate. It is attempted
        only under the unchanged full-device exclusive lease. No OOM retry or
        discarded optimizer update is permitted.
        """
        if not self.ready or self.latest is None:
            raise RuntimeError("startup memory observation is incomplete")
        rows = deepcopy(self.latest)
        if set(grant["gpus"]) != {row["gpu_uuid"] for row in rows}:
            raise ValueError("startup observations do not cover the exclusive grant")
        for row in rows:
            gpu = row["gpu_uuid"]
            if row["driver_peak_bytes"] <= 0 or gpu not in self.test_peaks:
                raise RuntimeError("startup lacks driver or complete initial evaluation measurements")
            # Later evaluation retains optimizer state. Conservatively add the
            # initial test peak to the observed training footprint rather than
            # assuming evaluation before optimizer initialization is sufficient.
            row["evaluation_overhead_bytes"] = self.test_peaks[gpu]
            row["evaluation_microbatch_size"] = 1
            row["initial_test_completed"] = True
        report = calibration_from_observations(self.fingerprint, rows, reserve_bytes=self.reserve_bytes)
        if any(max(row["driver_peak_bytes"], row["peak_reserved_bytes"]) + row["evaluation_overhead_bytes"] + self.reserve_bytes
               > grant["gpus"][row["gpu_uuid"]] for row in rows):
            raise RuntimeError("conservative train plus evaluation memory envelope exceeds GPU capacity")
        self.observations.extend(report["observations"])
        candidate = min(self.target, self.microbatch_size * 2)
        fits = all((max(row["driver_peak_bytes"], row["peak_reserved_bytes"]) + row["evaluation_overhead_bytes"])
                   * candidate / self.microbatch_size + self.reserve_bytes <= grant["gpus"][row["gpu_uuid"]]
                   for row in rows)
        if candidate <= self.microbatch_size or not fits:
            return None
        self.microbatch_size = candidate
        self.ready = False
        self.latest = None
        return candidate

    def report(self):
        if not self.observations or not self.initial_test_completed:
            raise RuntimeError("startup cannot publish an unmeasured calibration")
        return calibration_from_observations(self.fingerprint, deepcopy(self.observations),
                                            reserve_bytes=self.reserve_bytes)
