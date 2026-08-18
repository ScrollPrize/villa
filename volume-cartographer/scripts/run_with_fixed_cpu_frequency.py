#!/usr/bin/env python3
"""Pin CPU frequency for a calibration run and restore the prior state."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


POLICY_FIELDS = (
    "scaling_governor",
    "energy_performance_preference",
    "scaling_min_freq",
    "scaling_max_freq",
)
DEFAULT_RECOVERY_FILE = Path("/run/volume-cartographer-cpu-frequency.json")


def _read(path: Path) -> str:
    return path.read_text().strip()


def _write(path: Path, value: str) -> None:
    path.write_text(f"{value}\n")


def policy_paths(sysfs_root: Path) -> list[Path]:
    policies = list((sysfs_root / "cpufreq").glob("policy*"))
    return sorted(policies, key=lambda path: int(path.name.removeprefix("policy")))


def nominal_frequency_khz(sysfs_root: Path) -> int:
    nominal_paths = list(sysfs_root.glob("cpu[0-9]*/acpi_cppc/nominal_freq"))
    frequencies = {int(_read(path)) * 1000 for path in nominal_paths}
    if not frequencies:
        raise RuntimeError(
            "cannot discover the nominal CPU frequency; pass --target-khz"
        )
    if len(frequencies) != 1:
        values = ", ".join(str(value) for value in sorted(frequencies))
        raise RuntimeError(
            f"CPUs report different nominal frequencies ({values} kHz); "
            "pass --target-khz"
        )
    return frequencies.pop()


def resolve_target_khz(sysfs_root: Path, requested_khz: int | None) -> int:
    if requested_khz is not None:
        return requested_khz
    return nominal_frequency_khz(sysfs_root)


def snapshot_state(sysfs_root: Path) -> dict[str, object]:
    policies = {}
    for policy in policy_paths(sysfs_root):
        policies[policy.name] = {
            field: _read(policy / field)
            for field in POLICY_FIELDS
            if (policy / field).exists()
        }
    boost = sysfs_root / "cpufreq" / "boost"
    return {
        "schema_version": 1,
        "sysfs_root": str(sysfs_root),
        "boost": _read(boost) if boost.exists() else None,
        "policies": policies,
        "restored": False,
    }


def save_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def apply_request(sysfs_root: Path, target_khz: int) -> None:
    boost = sysfs_root / "cpufreq" / "boost"
    if boost.exists():
        _write(boost, "0")
    for policy in policy_paths(sysfs_root):
        governor = policy / "scaling_governor"
        epp = policy / "energy_performance_preference"
        minimum = policy / "scaling_min_freq"
        maximum = policy / "scaling_max_freq"
        if governor.exists():
            _write(governor, "performance")
        if epp.exists():
            _write(epp, "performance")
        current_min = int(_read(minimum))
        current_max = int(_read(maximum))
        if target_khz < current_min:
            _write(minimum, str(target_khz))
            _write(maximum, str(target_khz))
        elif target_khz > current_max:
            _write(maximum, str(target_khz))
            _write(minimum, str(target_khz))
        else:
            _write(maximum, str(target_khz))
            _write(minimum, str(target_khz))


def verify_request(sysfs_root: Path, target_khz: int) -> None:
    errors = []
    boost = sysfs_root / "cpufreq" / "boost"
    if boost.exists() and _read(boost) != "0":
        errors.append(f"{boost} did not disable boost")
    for policy in policy_paths(sysfs_root):
        for field in ("scaling_min_freq", "scaling_max_freq"):
            value = int(_read(policy / field))
            if value != target_khz:
                errors.append(f"{policy.name}/{field}={value}, expected {target_khz}")
    if errors:
        raise RuntimeError("frequency request verification failed: " + "; ".join(errors))


def require_root(sysfs_root: Path) -> None:
    if os.geteuid() != 0 and sysfs_root == Path("/sys/devices/system/cpu"):
        raise RuntimeError("CPU-frequency changes must run as root")


def set_frequency(args: argparse.Namespace) -> int:
    require_root(args.sysfs_root)
    if args.recovery_file.exists():
        previous = json.loads(args.recovery_file.read_text())
        if not previous.get("restored", False):
            raise RuntimeError(
                "an unrestored frequency snapshot already exists at "
                f"{args.recovery_file}; run the restore action first"
            )

    target_khz = resolve_target_khz(args.sysfs_root, args.target_khz)
    state = snapshot_state(args.sysfs_root)
    save_state(args.recovery_file, state)
    try:
        apply_request(args.sysfs_root, target_khz)
        verify_request(args.sysfs_root, target_khz)
    except BaseException:
        restore_state(state, args.recovery_file)
        raise

    print(
        f"CPU frequency pinned to {target_khz / 1000:g} MHz; "
        f"recovery state: {args.recovery_file}"
    )
    return 0


def restore_state(state: dict[str, object], recovery_file: Path | None = None) -> None:
    sysfs_root = Path(str(state["sysfs_root"]))
    policies = state["policies"]
    if not isinstance(policies, dict):
        raise RuntimeError("invalid frequency recovery state")
    errors = []
    for name, raw_fields in policies.items():
        if not isinstance(raw_fields, dict):
            errors.append(f"invalid fields for {name}")
            continue
        policy = sysfs_root / "cpufreq" / str(name)
        try:
            # The saved maximum is above the calibration request on this host.
            if "scaling_max_freq" in raw_fields:
                _write(policy / "scaling_max_freq", str(raw_fields["scaling_max_freq"]))
            if "scaling_min_freq" in raw_fields:
                _write(policy / "scaling_min_freq", str(raw_fields["scaling_min_freq"]))
            if "scaling_governor" in raw_fields:
                _write(policy / "scaling_governor", str(raw_fields["scaling_governor"]))
            if "energy_performance_preference" in raw_fields:
                _write(
                    policy / "energy_performance_preference",
                    str(raw_fields["energy_performance_preference"]),
                )
        except OSError as error:
            errors.append(f"{name}: {error}")
    boost = sysfs_root / "cpufreq" / "boost"
    try:
        if state.get("boost") is not None and boost.exists():
            _write(boost, str(state["boost"]))
    except OSError as error:
        errors.append(f"boost: {error}")
    if errors:
        raise RuntimeError("frequency state restoration failed: " + "; ".join(errors))
    state["restored"] = True
    if recovery_file is not None:
        save_state(recovery_file, state)


def _drop_privileges(uid: int, gid: int) -> None:
    user = pwd.getpwuid(uid)
    os.setgroups(os.getgrouplist(user.pw_name, gid))
    os.setgid(gid)
    os.setuid(uid)


def _child_environment(uid: int) -> dict[str, str]:
    user = pwd.getpwuid(uid)
    environment = os.environ.copy()
    environment.update(
        HOME=user.pw_dir,
        LOGNAME=user.pw_name,
        USER=user.pw_name,
    )
    return environment


def start_turbostat(output: Path) -> subprocess.Popen[str] | None:
    executable = Path("/usr/bin/turbostat")
    if not executable.exists():
        return None
    return subprocess.Popen(
        [
            str(executable),
            "--quiet",
            "--Summary",
            "--interval",
            "0.5",
            "--show",
            "Busy%,Bzy_MHz,TSC_MHz,PkgTmp",
            "--out",
            str(output),
        ],
        text=True,
    )


def stop_turbostat(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def parse_turbostat_mhz(path: Path) -> list[float]:
    if not path.exists():
        return []
    header: list[str] | None = None
    values = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if "Bzy_MHz" in fields:
            header = fields
            continue
        if header is None or len(fields) != len(header):
            continue
        try:
            values.append(float(fields[header.index("Bzy_MHz")]))
        except (ValueError, IndexError):
            continue
    return values


def run_guarded(args: argparse.Namespace) -> int:
    require_root(args.sysfs_root)
    target_khz = resolve_target_khz(args.sysfs_root, args.target_khz)
    state = snapshot_state(args.sysfs_root)
    save_state(args.recovery_file, state)
    child: subprocess.Popen[str] | None = None
    turbostat: subprocess.Popen[str] | None = None
    cur_samples: list[int] = []
    restored = False

    def interrupt(_signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        apply_request(args.sysfs_root, target_khz)
        verify_request(args.sysfs_root, target_khz)
        turbostat = start_turbostat(args.turbostat_output)
        child = subprocess.Popen(
            args.command,
            env=_child_environment(args.run_as_uid),
            preexec_fn=lambda: _drop_privileges(args.run_as_uid, args.run_as_gid),
            text=True,
        )
        current_paths = [
            policy / "scaling_cur_freq" for policy in policy_paths(args.sysfs_root)
        ]
        while child.poll() is None:
            for path in current_paths:
                try:
                    cur_samples.append(int(_read(path)))
                except (OSError, ValueError):
                    pass
            time.sleep(args.monitor_interval)
        return_code = child.wait()
        stop_turbostat(turbostat)
        turbostat = None
        bzy_mhz = parse_turbostat_mhz(args.turbostat_output)
        summary = {
            "target_khz": target_khz,
            "scaling_cur_freq": {
                "samples": len(cur_samples),
                "minimum_khz": min(cur_samples) if cur_samples else None,
                "median_khz": statistics.median(cur_samples) if cur_samples else None,
                "maximum_khz": max(cur_samples) if cur_samples else None,
            },
            "turbostat_bzy_mhz": {
                "samples": len(bzy_mhz),
                "minimum": min(bzy_mhz) if bzy_mhz else None,
                "median": statistics.median(bzy_mhz) if bzy_mhz else None,
                "maximum": max(bzy_mhz) if bzy_mhz else None,
            },
            "command_return_code": return_code,
        }
        if not bzy_mhz:
            raise RuntimeError("turbostat produced no Bzy_MHz samples")
        target_mhz = target_khz / 1000.0
        drift = abs(statistics.median(bzy_mhz) / target_mhz - 1.0)
        summary["median_effective_frequency_drift"] = drift
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if drift > args.drift_fraction:
            raise RuntimeError(
                f"effective frequency drift {drift:.2%} exceeds "
                f"{args.drift_fraction:.2%}"
            )
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, args.command)
        return 0
    finally:
        stop_turbostat(turbostat)
        try:
            restore_state(state, args.recovery_file)
            restored = True
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if not restored:
                print(
                    f"frequency restoration incomplete; recovery state: {args.recovery_file}",
                    file=sys.stderr,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    set_parser = subparsers.add_parser("set", help="pin frequency until restore")
    set_parser.add_argument(
        "--target-khz",
        type=int,
        help="fixed frequency; defaults to the CPUs' nominal frequency",
    )
    set_parser.add_argument(
        "--recovery-file", type=Path, default=DEFAULT_RECOVERY_FILE
    )
    set_parser.add_argument(
        "--sysfs-root", type=Path, default=Path("/sys/devices/system/cpu")
    )
    run = subparsers.add_parser("run")
    run.add_argument(
        "--target-khz",
        type=int,
        help="fixed frequency; defaults to the CPUs' nominal frequency",
    )
    run.add_argument("--drift-percent", type=float, default=3.0)
    run.add_argument("--monitor-interval", type=float, default=0.5)
    run.add_argument("--run-as-uid", type=int, required=True)
    run.add_argument("--run-as-gid", type=int, required=True)
    run.add_argument("--recovery-file", type=Path, required=True)
    run.add_argument("--turbostat-output", type=Path, required=True)
    run.add_argument("--summary", type=Path, required=True)
    run.add_argument(
        "--sysfs-root", type=Path, default=Path("/sys/devices/system/cpu")
    )
    run.add_argument("command", nargs=argparse.REMAINDER)
    restore = subparsers.add_parser("restore", help="restore state saved by set")
    restore.add_argument(
        "--recovery-file", type=Path, default=DEFAULT_RECOVERY_FILE
    )
    args = parser.parse_args()
    if args.action == "run":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            parser.error("run requires a command after --")
        args.drift_fraction = args.drift_percent / 100.0
    return args


def main() -> int:
    args = parse_args()
    if args.action == "set":
        return set_frequency(args)
    if args.action == "restore":
        state = json.loads(args.recovery_file.read_text())
        require_root(Path(str(state["sysfs_root"])))
        restore_state(state, args.recovery_file)
        print(f"CPU frequency state restored from {args.recovery_file}")
        return 0
    return run_guarded(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"fixed-frequency runner: {error}", file=sys.stderr)
        raise SystemExit(1)
