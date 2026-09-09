"""Portable commands for canonical ink distillation and EMA-only 3D inference.

Experiment settings and asset locations are supplied by a private release bundle.
No credentials or private service locations are embedded in this module.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vesuvius" / "src"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    from download_surface_volumes import write_json as atomic_write
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, value)


def bundle(args):
    root = args.assets_root.resolve()
    release = read_json(root / "release.json")
    if release.get("version") != 1:
        raise ValueError("Unsupported release manifest")
    return root, release, read_json(root / "recipe/training.json")


def asset_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release asset escapes the asset directory")
    return path


def verify(args):
    root, release, _ = bundle(args)
    for name, expected in release["files"].items():
        path = asset_path(root, name)
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Missing or changed release asset: {name}")
    for name, expected in release["code_files"].items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Code differs from the released recipe: {name}")
    print(f"Verified {len(release['files'])} assets and {len(release['code_files'])} code files", flush=True)


def checkpoint_path(root, release, name):
    relative = release["aliases"].get(name, name)
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Choose a released checkpoint alias or relative asset path")
    result = asset_path(root, relative)
    if relative not in release["files"] or sha256(result) != release["files"][relative]["sha256"]:
        raise ValueError("Checkpoint is not a verified asset in this release")
    return result


def check_data(root, recipe, data_root):
    """Validate recorded schemas and source label rasters; preserve the original split."""
    manifest = read_json(root / "recipe/dataset.json")
    old_root = Path(recipe["original_data_root"])
    excluded = set(recipe["stages"][0]["config"]["excluded_segments"])
    for record in manifest["segments"]:
        if record['scroll']+'/'+record['segment'] in excluded:
            continue
        for kind, fingerprint in record["fingerprints"].items():
            path = data_root / Path(record[kind]).relative_to(old_root)
            # The original manifest records a combined schema hash (see data module).
            from vesuvius.ink_detection.data.multiteacher import asset_fingerprint
            actual = asset_fingerprint(path, include_tiff=kind != "image")
            if actual != fingerprint:
                raise ValueError(f"Dataset fingerprint changed: {record['scroll']}/{record['segment']}/{kind}")
    return manifest


def data(args):
    verify(args)
    root, _, recipe = bundle(args)
    manifest = read_json(root / "recipe/dataset.json")
    excluded = set(recipe["stages"][0]["config"]["excluded_segments"])
    for record in manifest["segments"]:
        if record['scroll']+'/'+record['segment'] in excluded:
            continue
        destination = args.data_root.resolve() / record["scroll"] / record["segment"]
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(["hf", "buckets", "sync", record["source"], str(destination)], check=True)
    check_data(root, recipe, args.data_root.resolve())
    print("Dataset fingerprints verified; archived patch split retained")


def gpu_environment(ids, required=None):
    from run_surface3d_inference import available_gpus
    selected = [int(v) for v in ids.split(",")]
    if len(set(selected)) != len(selected) or (required is not None and len(selected) != required):
        raise ValueError(f"Select {required or 'distinct'} GPUs")
    idle = set(available_gpus())
    if not set(selected) <= idle:
        raise RuntimeError(f"Requested GPUs are not idle: {sorted(set(selected)-idle)}")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, selected)),
               PYTHONPATH=str(REPO / "vesuvius/src"), PYTHONDONTWRITEBYTECODE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    return selected, env


def stage_config(root, recipe, stage, data_root, output):
    cfg = deepcopy(stage["config"])
    for key in ("resume", "resume_transition", "teacher_handoff", "experiment_restart",
                "stop_after_updates", "resume_replace_coarse_teacher", "resume_rebalance"):
        cfg.pop(key, None)
    cfg.update(manifest=str(root / "recipe/dataset.json"), out_dir=str(output),
               initialization_checkpoint=str(root / recipe["initialization"]),
               background_calibration_dir=str(root / "recipe/background"),
               orientation_audit=str(root / "recipe/orientation.json"),
               canonical_source_dir=str(REPO / "ink-detection/optimized_inference"),
               path_roots={recipe["original_data_root"]: str(data_root)},
               provenance_dir=str(output / "provenance"), new_wandb_run=True)
    cfg["datasets"] = [{"segments_path":str(data_root), "volume_scale":"0"}]
    cfg["distillation"]["paris4"]["checkpoint"] = str(root / recipe["precise_teacher"])
    cfg["distillation"]["coarse"]["checkpoint"] = str(root / stage["coarse_teacher"])
    return cfg


def launch_training(cfg, config_path, env):
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    write_json(config_path, cfg)
    subprocess.run([sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
                    "--num_processes", str(cfg["expected_world_size"]), "--main_process_port", str(port),
                    "--num_cpu_threads_per_process", "4", "--mixed_precision", cfg["mixed_precision"],
                    "--module", "vesuvius.ink_detection.training.train", str(config_path)], env=env, check=True)


def train(args):
    verify(args)
    root, release, recipe = bundle(args)
    output, data_root = args.output.resolve(), args.data_root.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    check_data(root, recipe, data_root)
    _, env = gpu_environment(args.gpus, recipe["stages"][0]["config"]["expected_world_size"])
    output.mkdir(parents=True)
    if args.command == "resume":
        checkpoint = checkpoint_path(root, release, args.checkpoint)
        import torch
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        update = int(payload["optimizer_step"])
        stage = next(s for s in recipe["stages"] if s["start"] <= update < s["stop"] or
                     update == s["stop"] == recipe["stages"][-1]["stop"])
        if update >= stage["stop"]:
            raise ValueError("Released checkpoint already completed the recorded training schedule")
        cfg = stage_config(root, recipe, stage, data_root, output)
        cfg.update(resume=str(checkpoint), description="Full-state continuation from a released student checkpoint")
        if update == stage["start"]:
            from vesuvius.ink_detection.training.multiteacher_handoff import export_backbone_teacher
            teacher = output / "teachers" / "coarse.pth"
            report = export_backbone_teacher(checkpoint, teacher, update)
            cfg.update(resume_replace_coarse_teacher=True, resume_rebalance=True, teacher_handoff=report)
            cfg["distillation"]["coarse"].update(checkpoint=str(teacher), sha256=report["teacher_sha256"])
        launch_training(cfg, output / "config.json", env)
        return
    from vesuvius.ink_detection.training.multiteacher_handoff import export_backbone_teacher
    previous = None
    for stage in recipe["stages"]:
        stage_out = output / stage["name"]
        cfg = stage_config(root, recipe, stage, data_root, stage_out)
        cfg["stop_after_updates"] = stage["stop"]
        if previous is not None:
            teacher = output / "teachers" / (stage["name"]+".pth")
            report = export_backbone_teacher(previous, teacher, stage["start"])
            cfg.update(resume=str(previous), resume_replace_coarse_teacher=True, resume_rebalance=True,
                       teacher_handoff=report)
            cfg["distillation"]["coarse"].update(checkpoint=str(teacher), sha256=report["teacher_sha256"])
        launch_training(cfg, output / (stage["name"]+".json"), env)
        status = read_json(stage_out / "status.json")
        if status["optimizer_update"] != stage["stop"]:
            raise RuntimeError("Training stopped before the next stage boundary; refusing teacher replacement")
        previous = Path(status["checkpoint"])


def infer(args):
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    verify(args)
    root, release, recipe = bundle(args)
    checkpoint = checkpoint_path(root, release, args.checkpoint)
    source, output = args.input.resolve(), args.output.resolve()
    state = output.with_name(output.name+".work")
    if output.exists() or state.exists():
        raise FileExistsError("Use a new output path (existing output is never overwritten)")
    header, attrs = read_json(source / "0/.zarray"), read_json(source / ".zattrs")
    ms, = attrs["multiscales"]
    if [a["name"] for a in ms["axes"]] != ["z", "y", "x"] or header["dtype"] != "|u1":
        raise ValueError("Input must be a uint8 spatial ZYX OME-Zarr")
    if header["shape"][0] < recipe["inference"]["patch_zyx"][0]:
        raise ValueError("Input depth is shorter than the trained model window")
    selected, env = gpu_environment(args.gpus)
    import torch
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["config"]
    plan = {**recipe["inference"], "root":str(state), "checkpoint":str(checkpoint),
            "checkpoint_sha256":sha256(checkpoint), "expected_update":int(payload["optimizer_step"]),
            "architecture":"canonical_logits", "prediction_mode":"ink3d", "allowed_gpus":selected,
            "wandb_mode":args.wandb_mode,
            "max_gpus":len(selected), "wandb_entity":cfg["wandb_entity"], "wandb_project":cfg["wandb_project"],
            "datasets":[{"id":source.stem, "input":str(source), "output":str(output),
                         "source_s3":str(source), "shape":header["shape"], "reverse_depth":args.orientation=="reverse"}]}
    if args.batch_size is not None:
        plan["batch_size"] = args.batch_size
    state.mkdir(parents=True)
    write_json(state / "plan.json", plan)
    # Supervisor allocates the requested physical GPU IDs; workers see one each.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run([sys.executable, str(Path(__file__).with_name("run_surface3d_inference.py")),
                    str(state / "plan.json")], env=env, check=True)
    from build_surface3d_multiscales import build_one
    for item in plan["datasets"]:
        build_one(item, state, workers=recipe["inference"].get("pyramid_workers", 8))
    print(f"Verified six-level 3D probability OME-Zarr: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "data", "train", "resume", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--assets-root", type=Path, required=True)
        if name in ("data", "train", "resume"):
            p.add_argument("--data-root", type=Path, required=True)
        if name in ("train", "resume", "infer"):
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--gpus", required=True, help="Comma-separated idle physical GPU IDs")
        if name in ("resume", "infer"):
            p.add_argument("--checkpoint", default="latest", help="Release alias or relative checkpoint asset")
        if name == "infer":
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--orientation", choices=("forward", "reverse"), required=True)
            p.add_argument("--batch-size", type=int)
            p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args(argv)
    {"verify":verify, "data":data, "train":train, "resume":train, "infer":infer}[args.command](args)


if __name__ == "__main__":
    main()
