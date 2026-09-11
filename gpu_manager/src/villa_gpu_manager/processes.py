"""Process identities and peer credentials without creating a CUDA context."""

import os
import socket
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

from .protocol import require


def process_identity(pid=None):
    p = psutil.Process(os.getpid() if pid is None else pid)
    return {"pid": p.pid, "start_us": round(p.create_time() * 1_000_000),
            "boot_us": round(psutil.boot_time() * 1_000_000)}


def process_state(identity, uid):
    """Return alive/dead/unknown; never interpret inaccessible as dead."""
    try:
        p = psutil.Process(identity["pid"])
        if process_identity(p.pid) != identity:
            return "dead"
        if p.uids().real != uid:
            return "unknown"
        return "dead" if p.status() == psutil.STATUS_ZOMBIE else "alive"
    except psutil.NoSuchProcess:
        return "dead"
    except (psutil.AccessDenied, OSError):
        return "unknown"


def descendants(identity, uid):
    if process_state(identity, uid) != "alive":
        return []
    try:
        return [process_identity(p.pid) for p in psutil.Process(identity["pid"]).children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return []


def peer_uid(sock):
    if sys.platform.startswith("linux"):
        return struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
    if sys.platform == "darwin":
        # LOCAL_PEERCRED / xucred: version:uint, uid:uint, ngroups:short, groups[16].
        credentials = sock.getsockopt(0, 1, 76)
        return struct.unpack_from("II", credentials)[1]
    raise RuntimeError("peer credential adapter supports Linux and macOS only")


def nvidia_inventory(cpu_cores, gpu_uuids, reserve_bytes):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=10)
    found = {}
    for line in output.splitlines():
        gpu, mib = (part.strip() for part in line.split(","))
        if gpu in gpu_uuids:
            found[gpu] = int(mib) * 1024**2 - reserve_bytes
    require(set(found) == set(gpu_uuids), "managed GPU UUID not found")
    require(all(v > 0 for v in found.values()), "reserve exceeds GPU memory")
    logical = psutil.cpu_count(logical=True)
    require(logical is not None and 0 < cpu_cores <= logical, "CPU budget exceeds logical CPU inventory")
    return {"cpu_cores": cpu_cores, "gpus": found}


def parse_nvidia_memory(data, gpu_uuids=None):
    """Shared driver-memory inventory for policy monitoring and calibration."""
    def mib(text):
        value, unit = text.split()
        if unit != "MiB":
            raise ValueError("unknown driver memory unit")
        return int(value) * 1024**2
    return {device.findtext("uuid"): {
        "total": mib(device.findtext("fb_memory_usage/total")),
        "used": mib(device.findtext("fb_memory_usage/used")),
        "processes": {int(p.findtext("pid")): mib(p.findtext("used_memory"))
                      for p in device.findall("processes/process_info")},
    } for device in ET.fromstring(data).findall("gpu")
      if gpu_uuids is None or device.findtext("uuid") in gpu_uuids}


class NvidiaMonitor:
    """Poll a diagnostic subprocess without blocking the socket event loop."""

    def __init__(self, inventory, reserve_bytes, interval=60):
        self.inventory, self.reserve = inventory, reserve_bytes
        self.interval = interval
        self.process = None
        self.next_probe = 0
        self.started = 0
        self.buffer = bytearray()
        self.blocked = set(inventory["gpus"])  # fail closed until first probe
        self.last_snapshot = None
        self.last_snapshot_at = None

    def poll(self, owned_pids):
        now = time.monotonic()
        if self.process is None and now >= self.next_probe:
            try:
                self.process = subprocess.Popen(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                os.set_blocking(self.process.stdout.fileno(), False)
                self.buffer.clear()
                self.started = now
            except OSError:
                self.blocked = set(self.inventory["gpus"])
                self.next_probe = now + self.interval
        if self.process is not None:
            while True:
                try:
                    data = os.read(self.process.stdout.fileno(), 65536)
                except BlockingIOError:
                    break
                if not data:
                    break
                self.buffer.extend(data)
                if len(self.buffer) > 1024**2:
                    break
            if now - self.started > 10 or len(self.buffer) > 1024**2:
                self.process.kill()
                self.blocked = set(self.inventory["gpus"])
            if self.process.poll() is not None:
                try:
                    if self.process.returncode != 0:
                        raise ValueError("driver query failed")
                    self.blocked = self.parse(bytes(self.buffer), owned_pids() if callable(owned_pids) else owned_pids)
                except (ValueError, ET.ParseError, TypeError):
                    self.blocked = set(self.inventory["gpus"])
                self.process.stdout.close()
                self.process = None
                self.next_probe = now + self.interval
        return self.blocked

    def parse(self, data, owned_pids):
        self.last_snapshot = parse_nvidia_memory(data, self.inventory["gpus"])
        self.last_snapshot_at = time.monotonic()
        blocked = set(self.inventory["gpus"])
        for key, device in self.last_snapshot.items():
            if key not in blocked:
                continue
            total, used = device["total"], device["used"]
            owned = sum(size for pid, size in device["processes"].items() if pid in owned_pids)
            if total >= self.inventory["gpus"][key] + self.reserve and used - owned <= self.reserve:
                blocked.remove(key)
        return blocked

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
            self.process.stdout.close()
            self.process = None
