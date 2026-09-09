"""Export an immutable EMA 3D teacher and prepare a full-state continuation."""

from copy import deepcopy
import json
import os
from pathlib import Path

import torch

from vesuvius.ink_detection.data.multiteacher import file_sha256
from vesuvius.ink_detection.models.checkpoint import select_inference_weights


def backbone_teacher_payload(source, source_path, source_sha256):
    config = source["config"]
    if config.get("model_type") != "multiteacher_3d_projection":
        raise ValueError("Teacher export requires a joint 3D/projection checkpoint")
    weight_source, state = select_inference_weights(source)
    if weight_source != "ema_model":
        raise ValueError("Teacher export requires the checkpoint's EMA weights")
    if config.get("projection", {}).get("kind") == "canonical_logits":
        # Preserve the shared canonical encoder/decoder/classifier exactly.
        # FrozenInkTeachers dispatches forward_3d, so attention and 2D pooling
        # never execute despite their tensors remaining in the strict state.
        return {"config": deepcopy(config), "ema_model": state,
                "teacher_source": {"checkpoint": str(Path(source_path).resolve()),
                    "sha256": source_sha256, "optimizer_update": int(source["optimizer_step"]),
                    "weights": weight_source, "component": "canonical_3d"}}
    backbone = {key.removeprefix("backbone."): value for key,value in state.items()
                if key.startswith("backbone.")}
    if not backbone:
        raise ValueError("Joint checkpoint contains no EMA 3D backbone")
    backbone_config = deepcopy(config["model_config"]["backbone_config"])
    backbone_config["image_normalization"] = deepcopy(config["image_normalization"])
    return {"config": backbone_config, "ema_model": backbone,
            "teacher_source": {"checkpoint": str(Path(source_path).resolve()),
                "sha256": source_sha256, "optimizer_update": int(source["optimizer_step"]),
                "weights": weight_source, "component": "backbone"}}


def verify_exported_teacher(source_path, teacher_path):
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    expected = backbone_teacher_payload(source, source_path, file_sha256(source_path))
    actual = torch.load(teacher_path, map_location="cpu", weights_only=False, mmap=True)
    if actual.get("teacher_source") != expected["teacher_source"] or actual["config"] != expected["config"]:
        raise ValueError("Replacement teacher provenance/config does not match the resumed checkpoint")
    if actual["ema_model"].keys() != expected["ema_model"].keys() or any(
            not torch.equal(actual["ema_model"][key], value) for key,value in expected["ema_model"].items()):
        raise ValueError("Replacement teacher is not exactly the resumed checkpoint's EMA 3D model")
    return {**expected["teacher_source"], "teacher_sha256": file_sha256(teacher_path),
            "backbone_tensors_bitwise_equal": True}


def export_backbone_teacher(source_path, teacher_path, expected_update):
    source_path, teacher_path = Path(source_path), Path(teacher_path)
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if source["optimizer_step"] != expected_update:
        raise ValueError("Source checkpoint is not at the requested teacher handoff update")
    if teacher_path.exists():
        return verify_exported_teacher(source_path, teacher_path)
    payload = backbone_teacher_payload(source, source_path, file_sha256(source_path))
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = teacher_path.with_suffix(".partial")
    torch.save(payload, temporary)
    os.replace(temporary, teacher_path)
    return verify_exported_teacher(source_path, teacher_path)


def prepare_handoff_config(base_config, source_path, teacher_path, out_dir, provenance_dir,
                           expected_update, *, coarse_weight=None, paris4_per_global_update=None):
    """Preserve resume state, with explicitly requested teacher/objective changes."""
    if coarse_weight is not None and not 0 < float(coarse_weight) < float('inf'):
        raise ValueError("Coarse weight must be finite and positive")
    if paris4_per_global_update is not None:
        total = base_config['batch_size']*base_config['grad_acc_steps']*base_config['expected_world_size']
        if type(paris4_per_global_update) is not int or not 0 < paris4_per_global_update < total:
            raise ValueError("Global Paris 4 quota must be an integer strictly inside the effective batch")
    report = export_backbone_teacher(source_path, teacher_path, expected_update)
    config = deepcopy(base_config)
    if Path(out_dir).resolve() == Path(source_path).resolve().parent or Path(out_dir).exists():
        raise FileExistsError("Teacher handoff requires a new output directory")
    for key in ("resume_remove_depth_prior", "resume_rebalance", "resume_add_ct_context",
                "stop_after_updates", "resume_transition"):
        config.pop(key, None)
    config.update(resume=str(Path(source_path).resolve()), resume_replace_coarse_teacher=True,
                  initialization_sha256=file_sha256(config["initialization_checkpoint"]),
                  out_dir=str(Path(out_dir).resolve()), provenance_dir=str(Path(provenance_dir).resolve()),
                  initial_validation=True, wandb_run_name=f"flat_multiteacher_selfteacher{expected_update}_seed{config['seed']}")
    config["description"] = (f"Full-state continuation from update {expected_update} with its frozen EMA "
                             "3D model replacing the coarse teacher; original 50k schedule retained.")
    if config.get("projection", {}).get("kind") == "canonical_logits":
        config["wandb_run_name"] = f"canonical_r152_oriented_selfteacher{expected_update}_seed{config['seed']}"
    if "experiment_restart" in config:
        config["initial_restart_history"] = config.pop("experiment_restart")
    config["distillation"]["coarse"].update(checkpoint=str(Path(teacher_path).resolve()),
                                             sha256=report["teacher_sha256"])
    if coarse_weight is not None:
        config['distillation']['coarse']['weight'] = float(coarse_weight)
        config['resume_rebalance'] = True
    if paris4_per_global_update is not None:
        config.update(paris4_per_batch=0, paris4_per_rank_update=0,
                      paris4_per_global_update=paris4_per_global_update, resume_rebalance=True)
    if config.get('resume_rebalance'):
        config['description'] += (f" Coarse weight={config['distillation']['coarse']['weight']}; "
                                  f"Paris4 global quota={config.get('paris4_per_global_update', 0)}.")
    config["teacher_handoff"] = report
    Path(provenance_dir).mkdir(parents=True, exist_ok=True)
    (Path(provenance_dir)/"teacher-export-verification.json").write_text(json.dumps(report, indent=2)+"\n")
    # The run-inputs W&B artifact includes this exact frozen teacher as well as its hash.
    link = Path(provenance_dir)/"coarse-teacher.pth"
    temporary = link.with_suffix(".partial")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(Path(teacher_path).resolve())
    os.replace(temporary, link)
    return config
