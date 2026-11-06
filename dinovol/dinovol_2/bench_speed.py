from __future__ import annotations

from dinovol_2.training.config import as_legacy_config

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


RUN_DIR = Path("/ephemeral/sean/dinovol_perf/runs/pretrain_8_9um_objects_meshes_ink_160k")
BENCH_ROOT = RUN_DIR.with_name(f"{RUN_DIR.name}_benchmarks")
CODE_DIR = Path("/ephemeral/sean/dinovol_perf/code")
VENV_PY = "/home/ubuntu/dinovol/.venv/bin/python"


def launch_training() -> Path:
    checkpoint = (RUN_DIR / "checkpoint_step_160000.pt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Benchmark seed checkpoint is missing: {checkpoint}")
    cfg = as_legacy_config(json.loads((RUN_DIR / "config.json").read_text()))
    cfg["batch_size"] = 10
    cfg["grad_accumulation_steps"] = 1
    cfg["model"]["grad_checkpointing"] = False
    cfg["model"]["deeper_embed_batch_chunk_size"] = None
    cfg["dataset"]["num_local_crops"] = 4
    cfg["dataset"]["object_points"].pop("slab_thickness", None)
    cfg["dataset"]["object_points"]["mesh_cache_dir"] = "/ephemeral/sean/dinovol_perf/mesh_cache_8_9um"
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=BENCH_ROOT)).resolve()
    cfg.update(output_dir=str(output_dir), resume_from=str(checkpoint), resume=True, auto_resume=False)
    cfg_path = output_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    log_path = output_dir / "train.log"
    pid_path = output_dir / "train.pid"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "6"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = "source /ephemeral/sean/dinovol_perf/env.sh && exec " + shlex.join(
        [VENV_PY, "-u", "-m", "dinovol_2.pretrain", str(cfg_path)]
    )
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            ["/bin/bash", "-c", cmd],
            cwd=str(CODE_DIR),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{proc.pid}\n")
    (BENCH_ROOT / "latest-run.txt").write_text(str(output_dir) + "\n")
    print(f"Launched training PID={proc.pid} log={log_path}", flush=True)
    return output_dir


def check_status() -> None:
    smi = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "6",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", "6", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    print(f"GPU 6: {smi} | Apps: {apps}", flush=True)
    latest_run = BENCH_ROOT / "latest-run.txt"
    if not latest_run.exists():
        print("No benchmark has been launched.", flush=True)
        return
    log_path = Path(latest_run.read_text().strip()) / "train.log"
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        print(f"--- train.log ({len(lines)} lines, last 25) ---", flush=True)
        for line in lines[-25:]:
            print(line, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "launch":
        launch_training()
    elif mode == "status":
        check_status()
