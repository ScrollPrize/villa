"""Bounded read-only OS observations of an existing inference process tree.

No CUDA context, workload pause, ptrace, or changes to the running program.
Thread wait sites are samples, not attribution of critical-path stall time.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import time

import psutil


def tcp_counters(text, pids):
    """Linux ss -tinpH: pair connection headers with their detail lines."""
    result = {}
    key = None
    owner = None
    for line in text.splitlines():
        found = re.search(r"pid=(\d+)", line)
        if found:
            owner = int(found[1])
            fields = line.split()
            descriptor = re.search(r"fd=(\d+)", line)
            # Queue occupancy changes between samples; it must not change a
            # connection's identity or make cumulative bytes count twice.
            key = (owner, descriptor[1] if descriptor else None, *fields[3:5]) if owner in pids else None
        elif line and not line[0].isspace():
            key = owner = None
        if key is not None:
            received = re.search(r"\bbytes_received:(\d+)", line)
            sent = re.search(r"\bbytes_sent:(\d+)", line)
            if received or sent:
                result[key] = (owner, int(received[1]) if received else 0,
                               int(sent[1]) if sent else 0)
    return result


def futex_wait_address(text, machine):
    """Return only a known futex WAIT/WAIT_BITSET address, not an inferred owner."""
    number = {"x86_64": 202, "amd64": 202, "aarch64": 98, "arm64": 98}.get(machine.lower())
    parts = text.split()
    try:
        if number is None or len(parts) < 3 or int(parts[0], 0) != number:
            return None
        operation = int(parts[2], 0) & 0x7f
        return hex(int(parts[1], 0)) if operation in (0, 9) else None
    except ValueError:
        return None


def observe(root_pid, output_root, *, seconds=20, interval=2):
    if not 1 <= seconds <= 60 or not 0.5 <= interval <= seconds:
        raise ValueError("observe duration must be1–60s and interval0.5s–duration")
    root = psutil.Process(root_pid)
    identity = root.create_time()
    records = {}
    waits = defaultdict(Counter)
    states = defaultdict(Counter)
    socket_bytes = defaultdict(lambda: [0, 0])
    previous_sockets = {}
    thread_records = {}
    wait_addresses = defaultdict(Counter)
    address_threads = defaultdict(set)
    machine = platform.machine()
    notices = set()
    probe_cpu = time.process_time()
    started = time.monotonic()
    nic_start = psutil.net_io_counters(pernic=True)
    samples = 0
    ss = shutil.which("ss")
    while True:
        try:
            if not root.is_running():
                notices.add("root exited or PID identity changed during probe")
                break
            processes = [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            notices.add("root exited during probe")
            break
        observed = set()
        for process in processes:
            try:
                pid = process.pid
                stamp = process.create_time()
                cpu = process.cpu_times()
                io = process.io_counters() if hasattr(process, "io_counters") else None
                current = {"user": cpu.user, "system": cpu.system,
                    "read_bytes": io.read_bytes if io else 0,
                    "write_bytes": io.write_bytes if io else 0,
                    "read_chars": io.read_chars if io else 0}
                record = records.setdefault((pid, stamp), {"pid": pid,
                    "name": process.name(), "command": process.cmdline()[:3],
                    "first": current, "last": current, "max_threads": 0})
                record["last"] = current
                record["max_threads"] = max(record["max_threads"], process.num_threads())
                observed.add(pid)
                # Only trace/coordinator roots: avoid hundreds of uninteresting
                # model library thread waits. PID group summaries still cover all.
                if pid == root_pid or process.name().startswith("vc_fiber_trace"):
                    for thread in process.threads():
                        directory = Path(f"/proc/{pid}/task/{thread.id}")
                        try:
                            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                            state = fields[0]
                            thread_key = (pid, thread.id, fields[19])
                            total_cpu = thread.user_time + thread.system_time
                            entry = thread_records.setdefault(thread_key, {"tid": thread.id, "first": total_cpu,
                                "last": total_cpu, "states": Counter(), "waits": Counter()})
                            entry["last"] = total_cpu
                            entry["states"][state] += 1
                            wait = (directory / "wchan").read_text().strip()
                            states[pid][state] += 1
                            waits[pid][wait] += 1
                            entry["waits"][wait] += 1
                            if "futex" in wait:
                                try:
                                    address = futex_wait_address((directory / "syscall").read_text(), machine)
                                    if address:
                                        wait_addresses[pid][address] += 1
                                        address_threads[(pid, address)].add(thread_key)
                                except OSError:
                                    notices.add("per-thread syscall addresses unavailable; wait-object attribution not possible")
                        except (OSError, IndexError):
                            notices.add("some Linux per-thread wait sites unavailable")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                notices.add("some processes exited or were inaccessible")
        if ss:
            try:
                sockets = tcp_counters(subprocess.check_output([ss, "-tinpH"], text=True, timeout=3), observed)
                if samples:
                    for key, (pid, rx, tx) in sockets.items():
                        old = previous_sockets.get(key, (pid, 0, 0))
                        socket_bytes[pid][0] += max(0, rx - old[1])
                        socket_bytes[pid][1] += max(0, tx - old[2])
                previous_sockets = sockets
            except (OSError, subprocess.SubprocessError):
                notices.add("per-process TCP counters unavailable")
        else:
            notices.add("ss unavailable; only host NIC rates measured")
        samples += 1
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    elapsed = time.monotonic() - started
    nic_end = psutil.net_io_counters(pernic=True)
    rows = []
    for record in records.values():
        delta = {key: record["last"][key] - record["first"][key] for key in record["first"]}
        pid = record["pid"]
        threads = [{"tid": t["tid"], "cpu_seconds": t["last"] - t["first"],
                    "states": dict(t["states"]), "waits": dict(t["waits"])}
                   for key, t in thread_records.items() if key[0] == pid]
        threads.sort(key=lambda t: t["cpu_seconds"], reverse=True)
        rows.append({k: v for k, v in record.items() if k not in ("first", "last")} |
            {"delta": delta, "cpu_core_equivalents": (delta["user"] + delta["system"]) / elapsed,
             "sampled_thread_states": dict(states[pid]),
             "threads_with_cpu_delta": sum(t["cpu_seconds"] > 0 for t in threads),
             "top_cpu_threads": threads[:24],
             "futex_wait_objects": [{"address": address, "samples": count,
                 "distinct_threads": len(address_threads[(pid, address)])}
                 for address, count in wait_addresses[pid].most_common(16)], "sampled_wait_sites": dict(waits[pid]),
             "observed_tcp_received_bytes": socket_bytes[pid][0],
             "observed_tcp_sent_bytes": socket_bytes[pid][1]})
    report = {"root_pid": root_pid, "root_created": identity, "wall_seconds": elapsed,
        "samples": samples, "probe_cpu_seconds": time.process_time() - probe_cpu,
        "processes": rows, "notices": sorted(notices), "host_nics": {
            name: {"received_bytes": value.bytes_recv - nic_start[name].bytes_recv,
                   "sent_bytes": value.bytes_sent - nic_start[name].bytes_sent}
            for name, value in nic_end.items() if name in nic_start and name != "lo"},
        "limitations": ["Host NIC counters include unrelated traffic.",
            "TCP counters cover observed live sockets; sockets opening/closing between samples can be missed.",
            "Thread sleep/futex counts cannot distinguish network, decode, cache locks, and idle workers.",
            "Futex addresses identify shared wait objects, not their purpose or owning C++ object.",
            "Per-thread CPU deltas cover observed thread lifetimes within the sample, not full trace runtime.",
            "CPU time includes all native threads; it is not exclusively inference or decompression.",
            "Physical I/O counters do not fully attribute mmap writeback; zero read_bytes does not imply no cached I/O."]}
    output = Path(output_root) / ("process_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(f"Process probe: {output}", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report
