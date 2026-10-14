from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


RUN_DIR = Path("/ephemeral/sean/dinovol_perf/runs/pretrain_8_9um_objects_meshes_ink_160k")
CODE_DIR = Path("/ephemeral/sean/dinovol_perf/code")
VENV_PY = "/home/ubuntu/dinovol/.venv/bin/python"


def kill_gpu6_procs() -> None:
    my_pid = os.getpid()
    out = subprocess.check_output(
        ["nvidia-smi", "-i", "6", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    )
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid = int(line)
        if pid != my_pid:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"Killed GPU 6 PID {pid}", flush=True)
            except ProcessLookupError:
                pass


def launch_training() -> None:
    kill_gpu6_procs()
    time.sleep(1.0)

    cfg_path = RUN_DIR / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["batch_size"] = 10
    cfg["grad_accumulation_steps"] = 1
    cfg["model"]["grad_checkpointing"] = False
    cfg["model"]["deeper_embed_batch_chunk_size"] = None
    cfg["dataset"]["num_local_crops"] = 4
    cfg["dataset"]["object_points"].pop("slab_thickness", None)
    cfg["dataset"]["object_points"]["mesh_cache_dir"] = "/ephemeral/sean/dinovol_perf/mesh_cache_8_9um"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    for ckpt in sorted(RUN_DIR.glob("checkpoint_step_*.pt")):
        if ckpt.name != "checkpoint_step_160000.pt":
            ckpt.unlink()
            print(f"Removed intermediate {ckpt.name}", flush=True)

    log_path = RUN_DIR / "train.log"
    pid_path = RUN_DIR / "train.pid"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "6"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = f"source /ephemeral/sean/dinovol_perf/env.sh && exec {VENV_PY} -u -m dinovol_2.pretrain {cfg_path}"
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
    print(f"Launched training PID={proc.pid} log={log_path}", flush=True)


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
    log_path = RUN_DIR / "train.log"
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
