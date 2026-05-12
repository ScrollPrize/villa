"""Canary watchdog: SIGTERM the training process before the kernel wedges.

Why this exists: the prior autoreg_mesh full-config launch wedged the host
kernel during DataLoader worker spawn-IPC, requiring a hardware reboot.
This script polls system resources (mem, /dev/shm, FDs, S3 connections,
recent dmesg lines) every few seconds and sends SIGTERM to the training
process when any threshold is breached. SIGKILL after 60s if SIGTERM
didn't take effect.

Usage:
    python scripts/canary.py --ppid <torchrun_parent_pid> --log <path>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _read_mem_available_kb() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _read_shm_free_pct() -> float | None:
    try:
        st = os.statvfs("/dev/shm")
        if st.f_blocks == 0:
            return None
        return 100.0 * st.f_bavail / st.f_blocks
    except OSError:
        return None


def _descendant_pids(root_pid: int) -> list[int]:
    """All descendant PIDs of root_pid (incl. root)."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,ppid", "--no-headers"],
            capture_output=True, text=True, timeout=2.0,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return [root_pid]
    ppid_map: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        ppid_map.setdefault(ppid, []).append(pid)
    seen: set[int] = set()
    stack = [int(root_pid)]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for child in ppid_map.get(cur, []):
            stack.append(child)
    return sorted(seen)


def _count_fds(pids: list[int]) -> int:
    total = 0
    for pid in pids:
        try:
            total += len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            continue
    return total


def _count_s3_connections() -> int:
    """Established TCP connections to dport 443 (rough proxy for S3 sockets)."""
    count = 0
    try:
        with open("/proc/net/tcp", "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                rem = parts[2]
                state = parts[3]
                if state != "01":  # ESTABLISHED
                    continue
                if rem.endswith(":01BB"):  # 0x01BB = 443
                    count += 1
    except OSError:
        return -1
    return count


def _recent_dmesg(seconds: int = 10) -> list[str]:
    """Best-effort. Returns [] if unavailable (no CAP_SYSLOG)."""
    try:
        out = subprocess.run(
            ["dmesg", "--ctime", "--since", f"-{seconds}s"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode != 0:
            return []
        return out.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return []


def _scan_dmesg_for_patterns(lines: list[str], patterns: tuple[str, ...]) -> list[str]:
    matched = []
    for ln in lines:
        lower = ln.lower()
        for p in patterns:
            if p.lower() in lower:
                matched.append(ln)
                break
    return matched


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def trigger(pid: int, reason: str, snapshot: dict, log_fh):
    snapshot = {**snapshot, "event": "TRIGGER", "reason": reason,
                "ts": datetime.now(timezone.utc).isoformat()}
    log_fh.write(json.dumps(snapshot) + "\n")
    log_fh.flush()
    print(f"[canary] TRIGGER reason={reason} -> SIGTERM pid={pid}", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError) as e:
        print(f"[canary] SIGTERM failed: {e}", flush=True)
    # 60s grace, then SIGKILL.
    for _ in range(60):
        if not _pid_alive(pid):
            break
        time.sleep(1)
    if _pid_alive(pid):
        print(f"[canary] still alive after SIGTERM, SIGKILL pid={pid}", flush=True)
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--ppid", type=int, required=True,
                        help="PID of the training process to monitor + signal")
    parser.add_argument("--log", type=str, required=True,
                        help="JSONL log path for probes and events")
    parser.add_argument("--mem-floor-gb", type=float, default=200.0,
                        help="SIGTERM if MemAvailable falls below this")
    parser.add_argument("--shm-floor-pct", type=float, default=5.0,
                        help="SIGTERM if /dev/shm free falls below this pct")
    parser.add_argument("--fd-ceiling", type=int, default=500_000,
                        help="SIGTERM if total FD count exceeds this")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Probe interval in seconds")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")
    print(f"[canary] watching ppid={args.ppid} log={log_path}", flush=True)
    print(f"[canary] thresholds: mem<{args.mem_floor_gb}GB shm<{args.shm_floor_pct}% fd>{args.fd_ceiling}",
          flush=True)
    log_fh.write(json.dumps({
        "event": "START",
        "ppid": int(args.ppid),
        "mem_floor_gb": float(args.mem_floor_gb),
        "shm_floor_pct": float(args.shm_floor_pct),
        "fd_ceiling": int(args.fd_ceiling),
        "interval": float(args.interval),
        "ts": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    log_fh.flush()

    dmesg_patterns = ("oom-kill", "out of memory", "soft lockup",
                      "task hung", "blocked for more than",
                      "rcu_sched detected", "general protection fault")

    while True:
        if not _pid_alive(args.ppid):
            print(f"[canary] ppid={args.ppid} exited; canary stopping", flush=True)
            log_fh.write(json.dumps({"event": "STOP", "reason": "ppid_gone",
                                     "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
            log_fh.flush()
            log_fh.close()
            return 0

        snapshot: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "PROBE",
        }
        # Memory
        mem_kb = _read_mem_available_kb()
        if mem_kb is not None:
            snapshot["mem_avail_gb"] = round(mem_kb / 1024 / 1024, 2)

        # /dev/shm
        shm_pct = _read_shm_free_pct()
        if shm_pct is not None:
            snapshot["shm_free_pct"] = round(shm_pct, 2)

        # FDs over descendant tree
        pids = _descendant_pids(int(args.ppid))
        snapshot["proc_count"] = len(pids)
        snapshot["fd_count"] = _count_fds(pids)

        # S3-ish connections
        s3_count = _count_s3_connections()
        snapshot["s3_conn"] = s3_count

        # Dmesg scan
        dmesg_hits = _scan_dmesg_for_patterns(_recent_dmesg(int(args.interval) + 2),
                                              dmesg_patterns)
        if dmesg_hits:
            snapshot["dmesg_hits"] = dmesg_hits[:5]

        log_fh.write(json.dumps(snapshot) + "\n")
        log_fh.flush()

        # Triggers
        if mem_kb is not None and (mem_kb / 1024 / 1024) < float(args.mem_floor_gb):
            trigger(int(args.ppid), "low_mem", snapshot, log_fh)
            log_fh.close()
            return 0
        if shm_pct is not None and shm_pct < float(args.shm_floor_pct):
            trigger(int(args.ppid), "shm_full", snapshot, log_fh)
            log_fh.close()
            return 0
        if snapshot["fd_count"] > int(args.fd_ceiling):
            trigger(int(args.ppid), "fd_exhaustion", snapshot, log_fh)
            log_fh.close()
            return 0
        if dmesg_hits:
            trigger(int(args.ppid), f"dmesg:{dmesg_hits[0][:80]}", snapshot, log_fh)
            log_fh.close()
            return 0

        time.sleep(float(args.interval))


if __name__ == "__main__":
    sys.exit(main())
