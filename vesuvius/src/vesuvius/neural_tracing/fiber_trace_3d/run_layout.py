"""Shared logging layout, including a supervisor-pinned refinement run date."""
from datetime import datetime
from pathlib import Path
import os
import re

RUN_DATE_ENV = "VILLA_FIBER_RUN_DATE"


def sanitize_run_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "fiber_trace_3d").strip())
    return name.strip("._-") or "fiber_trace_3d"


def resolve_run_date(config):
    training = config.get("training", {})
    return str(os.environ.get(RUN_DATE_ENV) or training.get("run_datestr")
               or datetime.now().strftime("%Y%m%d_%H%M%S"))


def resolve_run_layout(config, *, date_str_override=None):
    training = config.get("training", {})
    run_path = Path(str(training.get("run_path", config.get("run_path", "runs/fiber_trace_3d"))))
    run_name = sanitize_run_name(training.get("run_name", config.get("run_name", "fiber_trace_3d")))
    date_str = str(date_str_override) if date_str_override is not None else resolve_run_date(config)
    run_dir = run_path / f"{run_name}_{date_str}"
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, snapshot_dir
