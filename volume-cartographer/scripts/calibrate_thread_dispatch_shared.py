#!/usr/bin/env python3
"""Collect a minimal single-CCD native thread-dispatch calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


CALIBRATION_CPUS = tuple(range(8))
MONITOR_CPU = 8
MAX_CALIBRATED_WORKERS = 7
SERIAL_FIT_WORK = (0, 400_000, 1_600_000, 6_400_000)
SERIAL_HOLDOUT_WORK = (150_000, 2_800_000)
DISPATCH_FIT_WORKERS = (2, 4, 6)
HOLDOUT_WORKERS = (1, 3, 5, 7)
WORK_HOLDOUT_ITERATIONS = (7_500, 350_000, 1_400_000)
DIAGNOSTIC_WORKERS = (3, 7)
DIAGNOSTIC_WORK_ITERATIONS = 400_000
DIAGNOSTIC_IDLE_NS = (0, 100_000, 1_000_000)
CPU_SYSFS = Path("/sys/devices/system/cpu")
FREQUENCY_RECOVERY_STATE = Path("/run/volume-cartographer-cpu-frequency.json")
FREQUENCY_TOLERANCE = 0.03
MIN_FREQUENCY_SAMPLES_PER_POLICY = 20
MAX_SERIAL_MEDIAN_ERROR_PERCENT = 5.0
MAX_SERIAL_INDIVIDUAL_ERROR_PERCENT = 10.0
MAX_DISPATCH_FIT_ERROR_PERCENT = 15.0
MAX_HOLDOUT_MEDIAN_ERROR_PERCENT = 20.0
MAX_HOLDOUT_INDIVIDUAL_ERROR_PERCENT = 20.0


@dataclass(frozen=True)
class Case:
    workers: int
    mode: str
    tasks: int
    work_iterations: int
    role: str
    idle_nanoseconds: int = 0

    @property
    def case_id(self) -> str:
        suffix = f"-idle{self.idle_nanoseconds}" if self.idle_nanoseconds else ""
        return (
            f"{self.role}-{self.mode}-w{self.workers}-t{self.tasks}"
            f"-i{self.work_iterations}{suffix}"
        )


def affinity(workers: int) -> str:
    if workers < 1 or workers > MAX_CALIBRATED_WORKERS:
        raise ValueError(
            f"calibrated workers must be from 1 to {MAX_CALIBRATED_WORKERS}"
        )
    return ",".join(str(cpu) for cpu in CALIBRATION_CPUS)


def wave_task_counts(workers: int) -> tuple[int, ...]:
    return tuple(sorted({max(1, workers // 2), workers}))


def make_serial_cases(work_values: tuple[int, ...], role: str) -> list[Case]:
    return [Case(1, "serial", 1, work, role) for work in work_values]


def make_dispatch_cases(workers_values: tuple[int, ...], role: str) -> list[Case]:
    return [
        Case(workers, "futures", tasks, 0, role)
        for workers in workers_values
        for tasks in range(1, workers + 1)
    ]


def make_work_holdout_cases() -> list[Case]:
    return [
        Case(workers, "futures", tasks, work, "work-holdout")
        for workers in HOLDOUT_WORKERS
        for tasks in range(1, workers + 1)
        for work in WORK_HOLDOUT_ITERATIONS
    ]


def make_gate_diagnostic_cases() -> list[Case]:
    return [
        Case(workers, mode, tasks, DIAGNOSTIC_WORK_ITERATIONS, "gate-diagnostic")
        for workers in DIAGNOSTIC_WORKERS
        for tasks in wave_task_counts(workers)
        for mode in ("futures", "futures-gate-open", "futures-gate-closed")
    ]


def make_idle_diagnostic_cases() -> list[Case]:
    return [
        Case(
            workers,
            "futures",
            tasks,
            DIAGNOSTIC_WORK_ITERATIONS,
            "idle-diagnostic",
            idle,
        )
        for workers in DIAGNOSTIC_WORKERS
        for tasks in wave_task_counts(workers)
        for idle in DIAGNOSTIC_IDLE_NS
    ]


def make_lifecycle_diagnostic_cases() -> list[Case]:
    return [
        Case(workers, "lifecycle", 1, 0, "lifecycle-diagnostic")
        for workers in HOLDOUT_WORKERS
    ]


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def _last_level_cache_id(cpu_root: Path, cpu: int) -> int:
    caches = []
    for index in (cpu_root / f"cpu{cpu}" / "cache").glob("index*"):
        try:
            caches.append((_read_int(index / "level"), _read_int(index / "id")))
        except (OSError, ValueError):
            continue
    if not caches:
        raise RuntimeError(f"cannot determine last-level cache for CPU {cpu}")
    return max(caches)[1]


def validate_topology(cpu_root: Path = CPU_SYSFS) -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("native affinity calibration requires Linux")
    records = {}
    for cpu in (*CALIBRATION_CPUS, MONITOR_CPU):
        topology = cpu_root / f"cpu{cpu}" / "topology"
        try:
            records[cpu] = {
                "package": _read_int(topology / "physical_package_id"),
                "core": _read_int(topology / "core_id"),
                "llc": _last_level_cache_id(cpu_root, cpu),
            }
        except (OSError, ValueError) as error:
            raise RuntimeError(f"cannot inspect topology for CPU {cpu}") from error
    identities = {
        (records[cpu]["package"], records[cpu]["core"])
        for cpu in CALIBRATION_CPUS
    }
    if len(identities) != len(CALIBRATION_CPUS):
        raise RuntimeError("calibration affinity includes SMT siblings")
    child_caches = {records[cpu]["llc"] for cpu in CALIBRATION_CPUS}
    if len(child_caches) != 1:
        raise RuntimeError("calibration CPUs do not share one last-level cache")
    monitor_identity = (
        records[MONITOR_CPU]["package"],
        records[MONITOR_CPU]["core"],
    )
    if monitor_identity in identities:
        raise RuntimeError("monitor CPU is an SMT sibling of a calibration CPU")
    if records[MONITOR_CPU]["llc"] in child_caches:
        raise RuntimeError("monitor CPU shares the calibration last-level cache")
    return {
        "calibration_cpus": list(CALIBRATION_CPUS),
        "monitor_cpu": MONITOR_CPU,
        "calibration_llc": next(iter(child_caches)),
        "monitor_llc": records[MONITOR_CPU]["llc"],
        "cpu_records": {str(cpu): value for cpu, value in records.items()},
    }


def validate_fixed_frequency(
    cpu_root: Path = CPU_SYSFS,
    recovery_state: Path = FREQUENCY_RECOVERY_STATE,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    policy_root = cpu_root / "cpufreq"
    policies = tuple(
        sorted(
            policy_root.glob("policy*"),
            key=lambda path: int(path.name.removeprefix("policy")),
        )
    )
    if not policies:
        raise RuntimeError("no CPU frequency policies found")
    targets = set()
    for policy in policies:
        try:
            minimum = _read_int(policy / "scaling_min_freq")
            maximum = _read_int(policy / "scaling_max_freq")
            governor = (policy / "scaling_governor").read_text().strip()
        except (OSError, ValueError) as error:
            raise RuntimeError(f"cannot inspect {policy.name}") from error
        if minimum != maximum:
            raise RuntimeError(f"{policy.name} frequency is not pinned")
        if governor != "performance":
            raise RuntimeError(f"{policy.name} governor is not performance")
        targets.add(minimum)
    if len(targets) != 1:
        raise RuntimeError("CPU frequency policies have different targets")
    if _read_int(policy_root / "boost") != 0:
        raise RuntimeError("CPU boost is enabled")
    if not recovery_state.is_file():
        raise RuntimeError(f"frequency recovery state is missing: {recovery_state}")
    names = {f"policy{cpu}" for cpu in (*CALIBRATION_CPUS, MONITOR_CPU)}
    monitored = tuple(policy for policy in policies if policy.name in names)
    if {policy.name for policy in monitored} != names:
        raise RuntimeError("a monitored CPU frequency policy is missing")
    return (
        {
            "target_khz": targets.pop(),
            "policy_count": len(policies),
            "monitored_policies": [policy.name for policy in monitored],
            "governor": "performance",
            "boost": 0,
            "tolerance_fraction": FREQUENCY_TOLERANCE,
            "recovery_state": str(recovery_state),
            "restore_deferred": True,
        },
        monitored,
    )


class FrequencyMonitor:
    def __init__(self, policies: tuple[Path, ...]) -> None:
        self._policies = policies
        self._samples = {policy.name: [] for policy in policies}
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "FrequencyMonitor":
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while not self._stopped.is_set():
            for policy in self._policies:
                try:
                    self._samples[policy.name].append(
                        _read_int(policy / "scaling_cur_freq")
                    )
                except (OSError, ValueError):
                    continue
            self._stopped.wait(0.05)

    def report(self, target_khz: int) -> dict[str, object]:
        lower = target_khz * (1.0 - FREQUENCY_TOLERANCE)
        upper = target_khz * (1.0 + FREQUENCY_TOLERANCE)
        per_policy = {}
        flat = []
        for name, values in self._samples.items():
            flat.extend(values)
            mean = statistics.fmean(values) if values else 0.0
            per_policy[name] = {
                "sample_count": len(values),
                "minimum_khz": min(values) if values else None,
                "maximum_khz": max(values) if values else None,
                "mean_khz": mean,
                "within_tolerance": (
                    len(values) >= MIN_FREQUENCY_SAMPLES_PER_POLICY
                    and lower <= mean <= upper
                ),
            }
        if not flat:
            raise RuntimeError("no CPU frequency readbacks were collected")
        return {
            "sample_count": len(flat),
            "minimum_khz": min(flat),
            "maximum_khz": max(flat),
            "mean_khz": statistics.fmean(flat),
            "minimum_samples_per_policy": MIN_FREQUENCY_SAMPLES_PER_POLICY,
            "per_policy": per_policy,
            "within_tolerance": all(
                bool(value["within_tolerance"]) for value in per_policy.values()
            ),
        }


def command_for(binary: Path, case: Case, rounds: int) -> list[str]:
    command = [
        "taskset",
        "-c",
        affinity(case.workers),
        str(binary),
        "--mode",
        case.mode,
        "--workers",
        str(case.workers),
        "--tasks",
        str(case.tasks),
        "--work-iterations",
        str(case.work_iterations),
        "--rounds",
        str(rounds),
    ]
    if case.idle_nanoseconds:
        command.extend(("--idle-nanoseconds", str(case.idle_nanoseconds)))
    return command


def read_schedstat(pid: int, latest: dict[int, tuple[int, int]]) -> None:
    for path in Path(f"/proc/{pid}/task").glob("*/schedstat"):
        try:
            fields = path.read_text().split()
            tid = int(path.parent.name)
            value = (int(fields[0]), int(fields[1]))
        except (OSError, IndexError, ValueError):
            continue
        previous = latest.get(tid, (0, 0))
        latest[tid] = (max(previous[0], value[0]), max(previous[1], value[1]))


def monitor_schedstat(
    pid: int,
    latest: dict[int, tuple[int, int]],
    stopped: threading.Event,
) -> None:
    while not stopped.is_set():
        read_schedstat(pid, latest)
        stopped.wait(0.005)


def run_case(binary: Path, case: Case, rounds: int) -> dict[str, object]:
    started = time.monotonic_ns()
    process = subprocess.Popen(
        command_for(binary, case, rounds),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    schedstat: dict[int, tuple[int, int]] = {}
    stopped = threading.Event()
    monitor = threading.Thread(
        target=monitor_schedstat, args=(process.pid, schedstat, stopped)
    )
    monitor.start()
    try:
        output, error = process.communicate()
    finally:
        stopped.set()
        monitor.join()
    external_ns = time.monotonic_ns() - started
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, command_for(binary, case, rounds), output, error
        )
    result = json.loads(output)
    running_ns = sum(value[0] for value in schedstat.values())
    waiting_ns = sum(value[1] for value in schedstat.values())
    result.update(
        external_ns=external_ns,
        case_id=case.case_id,
        schedstat_running_ns=running_ns,
        schedstat_waiting_ns=waiting_ns,
        schedstat_wait_fraction=(
            waiting_ns / (running_ns + waiting_ns)
            if running_ns + waiting_ns > 0
            else 0.0
        ),
    )
    return result


def choose_rounds(binary: Path, case: Case, target_seconds: float) -> int:
    pilot = run_case(binary, case, 32)
    measured_ns = float(pilot["nanoseconds_per_round"])
    elapsed_ns = measured_ns + float(pilot["actual_idle_nanoseconds_per_round"])
    if elapsed_ns <= 0:
        raise RuntimeError(f"nonpositive pilot time for {case.case_id}")
    maximum_rounds = 64 if case.mode == "lifecycle" else 262_144
    return max(8, min(maximum_rounds, math.ceil(target_seconds * 1e9 / elapsed_ns)))


def collect_cases(
    binary: Path,
    cases: list[Case],
    trials: int,
    target_seconds: float,
) -> list[dict[str, object]]:
    rounds = {case.case_id: choose_rounds(binary, case, target_seconds) for case in cases}
    samples = {case.case_id: [] for case in cases}
    total = len(cases) * trials
    completed = 0
    started = time.monotonic()
    for trial in range(trials):
        ordered = cases[trial % len(cases) :] + cases[: trial % len(cases)]
        if trial % 2:
            ordered.reverse()
        for case in ordered:
            samples[case.case_id].append(
                run_case(binary, case, rounds[case.case_id])
            )
            completed += 1
            elapsed = time.monotonic() - started
            eta = elapsed * (total - completed) / completed
            print(
                f"[dispatch-calibration] {completed}/{total} {case.case_id} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    return [
        {
            "case": asdict(case),
            "case_id": case.case_id,
            "rounds": rounds[case.case_id],
            "samples": samples[case.case_id],
        }
        for case in cases
    ]


def median_observation(record: dict[str, object]) -> dict[str, object]:
    samples = record["samples"]
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("observation record has no samples")
    ns = [float(sample["nanoseconds_per_round"]) for sample in samples]
    wait = [float(sample["schedstat_wait_fraction"]) for sample in samples]
    median = statistics.median(ns)
    return {
        **record["case"],
        "case_id": record["case_id"],
        "rounds": record["rounds"],
        "median_ns": median,
        "minimum_ns": min(ns),
        "maximum_ns": max(ns),
        "mad_ns": statistics.median(abs(value - median) for value in ns),
        "sample_ns": ns,
        "median_schedstat_wait_fraction": statistics.median(wait),
        "sample_actual_idle_ns": [
            float(sample["actual_idle_nanoseconds_per_round"]) for sample in samples
        ],
        "sample_raw_dispatch_ns": [
            float(sample["raw_dispatch_nanoseconds_per_round"]) for sample in samples
        ],
        "sample_clock_overhead_ns": [
            float(sample["clock_overhead_nanoseconds_per_round"])
            for sample in samples
        ],
    }


def measure_work_scale(
    records: list[dict[str, object]],
) -> tuple[float, dict[str, object]]:
    observations = [median_observation(record) for record in records]
    work = np.asarray([float(row["work_iterations"]) for row in observations])
    measured = np.asarray([float(row["median_ns"]) for row in observations])
    design = np.column_stack((np.ones(len(work)), work))
    intercept, scale = np.linalg.lstsq(design, measured, rcond=None)[0]
    if scale <= 0:
        raise RuntimeError("serial work calibration produced a nonpositive scale")
    trial_slopes = []
    for trial in range(len(records[0]["samples"])):
        values = np.asarray(
            [float(record["samples"][trial]["nanoseconds_per_round"]) for record in records]
        )
        trial_slopes.append(float(np.linalg.lstsq(design, values, rcond=None)[0][1]))
    predicted = design @ np.array([intercept, scale])
    nonzero = work > 0
    errors = 100.0 * (predicted[nonzero] / measured[nonzero] - 1.0)
    return float(scale), {
        "diagnostic_intercept_ns": float(intercept),
        "trial_slopes_ns_per_iteration": trial_slopes,
        "minimum_trial_slope_ns_per_iteration": min(trial_slopes),
        "maximum_trial_slope_ns_per_iteration": max(trial_slopes),
        "maximum_fit_error_percent": float(np.max(np.abs(errors))),
    }


def fit_dispatch(
    records: list[dict[str, object]],
) -> tuple[np.ndarray, dict[str, object]]:
    observations = [median_observation(record) for record in records]
    tasks = np.asarray([float(row["tasks"]) for row in observations])
    measured = np.asarray([float(row["median_ns"]) for row in observations])

    def residuals(value: np.ndarray) -> np.ndarray:
        return (value[0] + value[1] * tasks - measured) / measured

    result = least_squares(
        residuals,
        np.array([8_000.0, 200.0]),
        bounds=(np.zeros(2), np.array([1e6, 1e5])),
        loss="soft_l1",
        f_scale=0.02,
    )
    if not result.success:
        raise RuntimeError(f"dispatch fit failed: {result.message}")
    jacobian = np.asarray(result.jac)
    rank = int(np.linalg.matrix_rank(jacobian))
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(diagonal, diagonal)
    errors = 100.0 * ((result.x[0] + result.x[1] * tasks) / measured - 1.0)
    bounds_hit = any(math.isclose(value, 0.0, abs_tol=1e-5) for value in result.x)
    return result.x, {
        "jacobian_rank": rank,
        "parameter_count": 2,
        "absolute_parameter_correlation": float(abs(correlation[0, 1])),
        "parameters_hit_bounds": bounds_hit,
        "maximum_fit_error_percent": float(np.max(np.abs(errors))),
    }


def predict_round(
    parameters: dict[str, float], observation: dict[str, object]
) -> float:
    workers = int(observation["workers"])
    tasks = int(observation["tasks"])
    if workers < 1 or workers > MAX_CALIBRATED_WORKERS:
        raise ValueError(
            f"minimal model supports 1 to {MAX_CALIBRATED_WORKERS} workers"
        )
    if observation["mode"] == "serial":
        return parameters["work_ns_per_iteration"] * tasks * int(
            observation["work_iterations"]
        )
    if observation["mode"] != "futures":
        raise ValueError("minimal model predicts ordinary futures only")
    if tasks > workers:
        raise ValueError("minimal model requires tasks <= workers")
    dispatch = (
        parameters["fixed_dispatch_ns"]
        + parameters["per_future_dispatch_ns"] * tasks
    )
    total_work = tasks * int(observation["work_iterations"])
    work = parameters["work_ns_per_iteration"] * total_work / min(workers, tasks)
    return dispatch + work


def report_errors(
    parameters: dict[str, float], records: list[dict[str, object]]
) -> list[dict[str, object]]:
    reports = []
    for record in records:
        observation = median_observation(record)
        predicted = predict_round(parameters, observation)
        measured = float(observation["median_ns"])
        sample_errors = [
            100.0 * (predicted / float(sample) - 1.0)
            for sample in observation["sample_ns"]
        ]
        reports.append(
            {
                "case_id": observation["case_id"],
                "measured_ns": measured,
                "predicted_ns": predicted,
                "runtime_error_percent": 100.0 * (predicted / measured - 1.0),
                "maximum_individual_runtime_error_percent": max(
                    abs(value) for value in sample_errors
                ),
            }
        )
    return reports


def summarize_diagnostics(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [median_observation(record) for record in records]


def maximum_error(reports: list[dict[str, object]], key: str) -> float:
    return max(abs(float(report[key])) for report in reports)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def collect_frequency_block(
    binary: Path,
    groups: list[tuple[str, list[Case]]],
    trials: int,
    target_seconds: float,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    state, policies = validate_fixed_frequency()
    monitor = FrequencyMonitor(policies)
    with monitor:
        observations = {
            name: collect_cases(binary, cases, trials, target_seconds)
            for name, cases in groups
        }
    postflight, _ = validate_fixed_frequency()
    report = monitor.report(int(state["target_khz"]))
    report["postflight"] = postflight
    return observations, {**state, **report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--target-seconds", type=float, default=0.5)
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--reuse-diagnostics", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology = validate_topology()
    os.sched_setaffinity(0, {MONITOR_CPU})

    if args.reuse_calibration:
        fit_payload = json.loads(
            (args.output_dir / "fit-observations.json").read_text()
        )
        fit = {name: fit_payload[name] for name in ("serial", "dispatch")}
        fit_frequency = fit_payload["frequency"]
        topology = fit_payload["topology"]
    else:
        fit, fit_frequency = collect_frequency_block(
            args.benchmark,
            [
                ("serial", make_serial_cases(SERIAL_FIT_WORK, "serial-fit")),
                (
                    "dispatch",
                    make_dispatch_cases(DISPATCH_FIT_WORKERS, "dispatch-fit"),
                ),
            ],
            args.trials,
            args.target_seconds,
        )
    work_scale, work_report = measure_work_scale(fit["serial"])
    dispatch, dispatch_report = fit_dispatch(fit["dispatch"])
    parameters = {
        "work_ns_per_iteration": work_scale,
        "fixed_dispatch_ns": float(dispatch[0]),
        "per_future_dispatch_ns": float(dispatch[1]),
    }
    if not args.reuse_calibration:
        write_json(
            args.output_dir / "fit-observations.json",
            {**fit, "frequency": fit_frequency, "topology": topology},
        )
    write_json(
        args.output_dir / "model-before-holdout.json",
        {
            "schema_version": 3,
            "parameters": parameters,
            "work_fit": work_report,
            "dispatch_fit": dispatch_report,
            "synthetic_calibration_valid": False,
            "timing_claims_enabled": False,
        },
    )

    if args.reuse_calibration:
        holdout_payload = json.loads(
            (args.output_dir / "holdout-observations.json").read_text()
        )
        holdout = {
            name: holdout_payload[name] for name in ("serial", "dispatch", "work")
        }
        holdout_frequency = holdout_payload["frequency"]
    else:
        holdout, holdout_frequency = collect_frequency_block(
            args.benchmark,
            [
                (
                    "serial",
                    make_serial_cases(SERIAL_HOLDOUT_WORK, "serial-holdout"),
                ),
                (
                    "dispatch",
                    make_dispatch_cases(HOLDOUT_WORKERS, "dispatch-holdout"),
                ),
                ("work", make_work_holdout_cases()),
            ],
            args.trials,
            args.target_seconds,
        )
        write_json(
            args.output_dir / "holdout-observations.json",
            {**holdout, "frequency": holdout_frequency},
        )
    serial_errors = report_errors(parameters, holdout["serial"])
    dispatch_errors = report_errors(parameters, holdout["dispatch"])
    work_errors = report_errors(parameters, holdout["work"])
    holdout_errors = dispatch_errors + work_errors
    max_serial_median = maximum_error(serial_errors, "runtime_error_percent")
    max_serial_individual = maximum_error(
        serial_errors, "maximum_individual_runtime_error_percent"
    )
    max_holdout_median = maximum_error(holdout_errors, "runtime_error_percent")
    max_holdout_individual = maximum_error(
        holdout_errors, "maximum_individual_runtime_error_percent"
    )
    synthetic_valid = (
        max_serial_median <= MAX_SERIAL_MEDIAN_ERROR_PERCENT
        and max_serial_individual <= MAX_SERIAL_INDIVIDUAL_ERROR_PERCENT
        and float(dispatch_report["maximum_fit_error_percent"])
        <= MAX_DISPATCH_FIT_ERROR_PERCENT
        and int(dispatch_report["jacobian_rank"]) == 2
        and float(dispatch_report["absolute_parameter_correlation"]) < 0.98
        and not bool(dispatch_report["parameters_hit_bounds"])
        and max_holdout_median <= MAX_HOLDOUT_MEDIAN_ERROR_PERCENT
        and max_holdout_individual <= MAX_HOLDOUT_INDIVIDUAL_ERROR_PERCENT
        and bool(fit_frequency["within_tolerance"])
        and bool(holdout_frequency["within_tolerance"])
    )

    diagnostic_cases = {
        "gate": make_gate_diagnostic_cases(),
        "idle": make_idle_diagnostic_cases(),
        "lifecycle": make_lifecycle_diagnostic_cases(),
    }
    if args.reuse_diagnostics:
        diagnostic_payload = json.loads(
            (args.output_dir / "diagnostics.json").read_text()
        )
        diagnostic_records = diagnostic_payload["observations"]
        diagnostics = diagnostic_payload["summary"]
    else:
        diagnostic_records = {}
        diagnostics = {}
        for name, cases in diagnostic_cases.items():
            records = collect_cases(
                args.benchmark, cases, args.trials, args.target_seconds
            )
            diagnostic_records[name] = records
            diagnostics[name] = summarize_diagnostics(records)
            write_json(
                args.output_dir / "diagnostics.json",
                {"observations": diagnostic_records, "summary": diagnostics},
            )

    model = {
        "schema_version": 3,
        "domain": {
            "mode": "futures",
            "maximum_workers": MAX_CALIBRATED_WORKERS,
            "tasks_at_most_workers": True,
            "fixed_physical_affinity": affinity(MAX_CALIBRATED_WORKERS),
            "monitor_cpu": MONITOR_CPU,
            "actual_renderer_validation_complete": False,
        },
        "parameter_count": 3,
        "parameters": parameters,
        "work_fit": work_report,
        "dispatch_fit": dispatch_report,
        "serial_holdout_errors": serial_errors,
        "dispatch_holdout_errors": dispatch_errors,
        "work_holdout_errors": work_errors,
        "maximum_serial_holdout_error_percent": max_serial_median,
        "maximum_individual_serial_holdout_error_percent": max_serial_individual,
        "maximum_synthetic_holdout_error_percent": max_holdout_median,
        "maximum_individual_synthetic_holdout_error_percent": max_holdout_individual,
        "frequency": {"fit": fit_frequency, "holdout": holdout_frequency},
        "topology": topology,
        "diagnostics_file": "diagnostics.json",
        "diagnostics_affect_validity": False,
        "synthetic_calibration_valid": synthetic_valid,
        "timing_claims_enabled": False,
    }
    write_json(args.output_dir / "model.json", model)
    print(json.dumps(model, indent=2, sort_keys=True))
    return 0 if synthetic_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
