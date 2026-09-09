"""GPU supervisor for canonical 3D inference and local OME-Zarr outputs."""

import argparse
from copy import deepcopy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numcodecs
import numpy as np
import zarr

from download_surface_volumes import write_json
from surface3d_validity import ensure_depth_validity
from vesuvius.ink_detection.inference.infer_surface3d import connect_queue
from vesuvius.label_zarr import create_v2_array


def available_gpus():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                           "--format=csv,noheader,nounits"], text=True).splitlines())
    return [int(r[0]) for r in csv.reader(io.StringIO(rows))
            if r[1].strip() not in occupied and int(r[2]) < 1024 and int(r[3]) < 5]


def process_alive(pid):
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def initialize(plan):
    root = Path(plan["root"])
    connection = connect_queue(root)
    connection.execute("CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, ready INTEGER NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, "
                       "y0 INTEGER,y1 INTEGER,x0 INTEGER,x1 INTEGER,status TEXT NOT NULL,owner INTEGER,"
                       "started REAL,finished REAL,metrics TEXT)")
    connection.execute("CREATE INDEX IF NOT EXISTS pending_jobs ON jobs(status,id)")
    saved = root / "execution-plan.json"
    if saved.exists():
        if json.loads(saved.read_text()) != plan:
            raise ValueError("Refusing to mix inference settings in an existing output directory")
        running = connection.execute("SELECT DISTINCT owner FROM jobs WHERE status='running'").fetchall()
        if any(process_alive(pid) for (pid,) in running):
            raise RuntimeError("Previous workers still own output tiles")
        connection.execute("UPDATE jobs SET status='pending',owner=NULL WHERE status IN ('running','failed')")
    else:
        tile = plan["tile_size"]
        if tile % 256:
            raise ValueError("Output tiles must align with 256-pixel output chunks")
        for item in plan["datasets"]:
            connection.execute("INSERT INTO datasets VALUES (?,0)", (item["id"],))
            _, height, width = item["shape"]
            connection.executemany("INSERT INTO jobs(dataset,y0,y1,x0,x1,status) VALUES (?,?,?,?,?,'pending')",
                                   [(item["id"], y, min(y+tile, height), x, min(x+tile, width))
                                    for y in range(0, height, tile) for x in range(0, width, tile)])
    connection.commit()
    if not saved.exists():
        write_json(saved, plan)
    return connection


def prepare_output(item, plan):
    source_attrs = json.loads((Path(item["input"]) / ".zattrs").read_text())
    source_array = json.loads((Path(item["input"]) / "0/.zarray").read_text())
    if source_array["shape"] != item["shape"] or source_array["dtype"] != "|u1":
        raise ValueError("Downloaded array does not match inference plan")
    path = Path(item["output"])
    kwargs = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
    group = zarr.open_group(str(path), mode="a", **kwargs)
    if "0" not in group:
        create_v2_array(group, "0", shape=tuple(item["shape"]), chunks=(item["shape"][0], 256, 256),
                        dtype=np.uint8, compressor=numcodecs.Blosc(cname="lz4", clevel=1, shuffle=1),
                        fill_value=0)
    if list(group["0"].shape) != item["shape"] or group["0"].dtype != np.uint8:
        raise ValueError("Existing output has incompatible geometry or dtype")
    attrs = {"multiscales": deepcopy(source_attrs["multiscales"]),
             "_ARRAY_DIMENSIONS": ["z", "y", "x"], "complete": False,
             "source_s3": item["source_s3"], "source_shape": item["shape"],
             "prediction": "3D ink probability; uint8 = round(255 * probability)",
             "probability_scale": 1/255, "projection_executed": False,
             "checkpoint": plan["checkpoint"], "checkpoint_sha256": plan["checkpoint_sha256"],
             "optimizer_update": plan["expected_update"], "weights": "EMA 3D backbone only",
             "patch_zyx": plan["patch_zyx"], "stride_zyx": plan["stride_zyx"],
             "blend": "floored Hann window; float32 probability accumulation",
             "amp": "bf16", "normalization": "checkpoint normalization, independently per patch",
             "missing_support": "all-zero CT columns in each patch contribute zero probability"}
    if plan.get("evaluation_split"):
        attrs.update(evaluation_split=plan["evaluation_split"],
                     training_record_index=item.get("training_record_index"))
    if plan.get('architecture') == 'canonical_logits':
        attrs.update(architecture='canonical_logits', attention_executed=False, projection_executed=False,
            weights=f"Canonical 3D student EMA at update {plan['expected_update']}",
            input_depth_reversed=item.get('reverse_depth', False), output_depth_order='original_source',
            depth_preprocessing=('Full-segment reversal before central crop; predictions reversed back for source alignment'
                if item.get('reverse_depth', False) else
                'Central crop in original source depth order; no depth reversal'))
    for key in ("physical_spacing_source", "metadata_normalization"):
        if key in source_attrs:
            attrs[key] = source_attrs[key]
    for scale in attrs["multiscales"]:
        scale["name"] = item["id"] + "-3d-ink"
        scale["datasets"] = [d for d in scale["datasets"] if d["path"] == "0"]
    group.attrs.update(attrs)
    if plan.get("depth_mode") == "centered":
        ensure_depth_validity(group, item["shape"], plan["patch_zyx"][0], plan.get('valid_depth_margin', 0),
                              item.get('reverse_depth', False))
    group["0"].attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["root"])
    lock = (root / "inference.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with Path(plan["checkpoint"]).open("rb") as checkpoint_file:
        plan["checkpoint_sha256"] = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    connection = initialize(plan)
    execution_path = root / "execution-plan.json"
    import wandb
    id_path = root / "wandb-run.json"
    run_id = json.loads(id_path.read_text())["id"] if id_path.exists() else wandb.util.generate_id()
    run = wandb.init(entity=plan["wandb_entity"], project=plan["wandb_project"],
                     id=run_id, resume="allow", job_type="inference-3d",
                     name=plan.get("run_name", "distilled-3d-inference"), dir=str(root), config=plan, mode=plan.get("wandb_mode", "online"))
    write_json(id_path, {"id": run.id, "url": run.url})
    artifact = wandb.Artifact("distilled-3d-inference-inputs", type="inference-config")
    artifact.add_file(str(execution_path))
    run.log_artifact(artifact)
    processes = {}
    finished_datasets = set()
    last_finished = 0.0
    try:
        while True:
            if shutil.disk_usage(root).free < plan.get("min_free_gb", 500) * 1e9:
                raise RuntimeError("Inference stopped at the disk reserve")
            for item in plan["datasets"]:
                name = item["id"]
                ready = connection.execute("SELECT ready FROM datasets WHERE id=?", (name,)).fetchone()[0]
                if not ready and (Path(item["input"]) / "0/.zarray").is_file():
                    prepare_output(item, plan)
                    connection.execute("UPDATE datasets SET ready=1 WHERE id=?", (name,))
                    connection.commit()
            failed = connection.execute("SELECT id,metrics FROM jobs WHERE status='failed'").fetchall()
            if failed:
                raise RuntimeError(f"Inference worker failed: {failed[:3]}")
            for gpu, (process, log) in list(processes.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del processes[gpu]
                    if code:
                        raise RuntimeError(f"GPU {gpu} worker exited with code {code}")
            for gpu in available_gpus():
                if 'allowed_gpus' in plan and gpu not in plan['allowed_gpus']:
                    continue
                if gpu in processes or len(processes) >= plan["max_gpus"]:
                    continue
                remaining = connection.execute("SELECT COUNT(*) FROM jobs WHERE status!='done'").fetchone()[0]
                if not remaining:
                    break
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                log = (root / f"gpu-{gpu}.log").open("a")
                process = subprocess.Popen([sys.executable, "-u", "-m",
                    "vesuvius.ink_detection.inference.infer_surface3d", str(execution_path), "--gpu", str(gpu)],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
                processes[gpu] = (process, log)
            counts = dict(connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
            progress = {"updated_unix": time.time(), "jobs": counts, "gpus": sorted(processes),
                        "wandb_url": run.url, "free_disk_gb": shutil.disk_usage(root).free / 1e9}
            write_json(root / "status.json", progress)
            metrics = {"jobs/" + k: v for k, v in counts.items()}
            metrics["resources/gpus"] = len(processes)
            metrics["resources/free_disk_gb"] = progress["free_disk_gb"]
            for item in plan["datasets"]:
                name = item["id"]
                n, total = connection.execute("SELECT SUM(status='done'),COUNT(*) FROM jobs WHERE dataset=?", (name,)).fetchone()
                metrics[f"progress/{name}"] = n / total
                if n == total and name not in finished_datasets:
                    group = zarr.open_group(item["output"], mode="a")
                    if group.attrs.get("complete") is not True:
                        group.attrs.update(complete=True, completed_unix=time.time())
                    finished_datasets.add(name)
                    run.summary[f"output/{name}"] = item["output"]
                    print(json.dumps({"complete": name, "output": item["output"]}), flush=True)
            recent = connection.execute("SELECT finished,metrics FROM jobs WHERE status='done' AND finished>? ORDER BY finished", (last_finished,)).fetchall()
            if recent:
                stats = [json.loads(v) for _, v in recent]
                seconds = sum(v["seconds"] + v["write_seconds"] for v in stats)
                metrics["throughput/worker_Mvox_s"] = sum(v["voxels"] for v in stats) / seconds / 1e6
                last_finished = max(finished for finished, _ in recent)
            run.log(metrics)
            if len(finished_datasets) == len(plan["datasets"]):
                for process, log in processes.values():
                    process.wait(timeout=60)
                    log.close()
                write_json(root / "complete.json", {**progress, "outputs": [d["output"] for d in plan["datasets"]]})
                break
            time.sleep(2)
        run.finish()
    except BaseException as error:
        for process, log in processes.values():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=60)
            log.close()
        write_json(root / "failure.json", {"error": repr(error), "updated_unix": time.time()})
        run.finish(exit_code=1)
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
