import os
import re
import subprocess
from functools import lru_cache
from typing import Optional


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_GPU_ROW_RE = re.compile(
    r"^GPU(?P<gpu>\d+)\s+.*?\s+(?P<cpu>[0-9,\-]+)\s+(?P<numa>\S+)\s+(?P<gpu_numa>\S+)\s*$"
)


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _parse_cpu_affinity(spec: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    return tuple(sorted(cpus))


@lru_cache(maxsize=1)
def get_gpu_cpu_affinity_map() -> dict[int, tuple[int, ...]]:
    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return {}

    mapping: dict[int, tuple[int, ...]] = {}
    for raw_line in topo.splitlines():
        line = _strip_ansi(raw_line).strip()
        if not line.startswith("GPU"):
            continue
        match = _GPU_ROW_RE.match(line)
        if not match:
            continue
        gpu_id = int(match.group("gpu"))
        cpu_affinity = _parse_cpu_affinity(match.group("cpu"))
        if cpu_affinity:
            mapping[gpu_id] = cpu_affinity
    return mapping


def apply_numa_affinity(
    mode: str,
    assigned_gpu_id: Optional[int],
) -> Optional[dict[str, object]]:
    if str(mode).lower() != "auto":
        return None
    if assigned_gpu_id is None:
        return None
    if not hasattr(os, "sched_setaffinity"):
        return None

    affinity_map = get_gpu_cpu_affinity_map()
    cpu_affinity = affinity_map.get(int(assigned_gpu_id))
    if not cpu_affinity:
        return None

    os.sched_setaffinity(0, set(cpu_affinity))
    return {
        "gpu_id": int(assigned_gpu_id),
        "cpu_count": len(cpu_affinity),
        "cpu_affinity": cpu_affinity,
    }
