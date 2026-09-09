"""Joint flat 3D/2D training; counters and schedules use optimizer updates."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from vesuvius.ink_detection.config import InkConfig, OptimizerConfig, SchedulerConfig
from vesuvius.ink_detection.data.multiteacher import (
    FlatDistillationDataset, RankDrawSampler, file_sha256,
)
from vesuvius.ink_detection.models.checkpoint import (
    load_checkpoint, load_model_state, select_inference_weights,
)
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.training.multiteacher_loss import (
    FrozenInkTeachers, joint_loss, finish_loss_metrics, suppression_statistics,
)
from vesuvius.ink_detection.training.multiteacher_reporting import ScrollMetrics, save_preview
from vesuvius.ink_detection.training.optimizers import create_training_optimizer


def resolve_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("mode") != "flat" or cfg.get("model_type") != "multiteacher_3d_projection":
        raise ValueError("Joint distillation requires mode=flat and multiteacher_3d_projection")
    if cfg.get("patch_size") != [64, 256, 256]:
        raise ValueError("This flat recipe requires patch_size=[64,256,256]")
    if cfg.get("dynamic_label") or cfg.get("force_full_supervision"):
        raise ValueError("Joint training must preserve human labels and supervision")
    for key in ("manifest", "out_dir", "initialization_checkpoint", "resume", "background_calibration_dir"):
        if cfg.get(key):
            p = Path(cfg[key]).expanduser()
            cfg[key] = str(p if p.is_absolute() else path.parent/p)
    if cfg.get("projection", {}).get("kind") != "canonical_logits":
        raise ValueError("Only the canonical student is supported by this recipe")
    normalizations = []
    teacher_payload = None
    for key in ("paris4", "coarse"):
        settings = cfg["distillation"][key]
        p = Path(settings["checkpoint"]).expanduser()
        p = p if p.is_absolute() else path.parent/p
        settings["checkpoint"] = str(p)
        actual = file_sha256(p)
        if actual != settings["sha256"]:
            raise ValueError(f"Teacher checksum mismatch: {key}")
        teacher_payload = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        normalizations.append(teacher_payload["config"].get("image_normalization", "percentile_minmax"))
    canonical = cfg.get("projection", {}).get("kind") == "canonical_logits"
    if canonical:
        if file_sha256(cfg["initialization_checkpoint"]) != cfg.get("initialization_sha256"):
            raise ValueError("Canonical initializer checksum mismatch")
    elif cfg["initialization_checkpoint"] != cfg["distillation"]["coarse"]["checkpoint"]:
        if not (cfg.get("resume") and cfg.get("resume_replace_coarse_teacher")
                and cfg.get("initialization_sha256")):
            raise ValueError("A separate coarse teacher requires an explicit verified resume transition")
        if file_sha256(cfg["initialization_checkpoint"]) != cfg["initialization_sha256"]:
            raise ValueError("Original student initializer checksum mismatch")
        teacher_payload = torch.load(cfg["initialization_checkpoint"], map_location="cpu",
                                     weights_only=False, mmap=True)
    # Embed only network construction/input settings, not the teacher's resume state.
    source = teacher_payload["config"]
    if not canonical and normalizations[1] != source.get("image_normalization", "percentile_minmax"):
        raise ValueError("Replacement teacher must preserve the student's input normalization")
    keys = ("model_type", "model_config", "patch_size", "crop_size", "in_channels",
            "targets", "batch_size", "enable_deep_supervision", "autoconfigure",
            "spacing", "model_name")
    backbone = {k: deepcopy(source[k]) for k in keys if k in source}
    backbone.update(mode="flat", datasets=deepcopy(cfg["datasets"]),
                    patch_overlap=cfg["patch_overlap"],
                    patch_min_labeled_coverage=cfg["patch_min_labeled_coverage"])
    cfg["model_config"] = {"backbone_config": backbone,
                           "projection": cfg.get("projection", {})}
    cfg["teacher_normalizations"] = normalizations
    cfg["image_normalization"] = normalizations[1]
    if canonical:
        from vesuvius.ink_detection.models.canonical_projection import canonical_source_dir
        source_dir = canonical_source_dir()
        cfg["model_config"] = {"projection": cfg["projection"], "canonical": {
            "source_dir": str(source_dir), "with_norm": True,
            "freeze_batchnorm_stats": True}}
        cfg["image_normalization"] = {"mode": "clip_zscore", "clip_min": 0.,
                                      "clip_max": 200., "mean": 0., "std": 200.}
        cfg["valid_depth_margin"] = 1
    cfg["manifest_sha256"] = file_sha256(cfg["manifest"])
    if cfg.get("background_calibration_dir"):
        calibration_path = Path(cfg["background_calibration_dir"])/"calibration.json"
        settings = cfg["distillation"]["coarse"]
        if file_sha256(calibration_path) != settings["background_calibration_sha256"]:
            raise ValueError("Background calibration checksum mismatch")
        calibration = json.loads(calibration_path.read_text())
        thresholds = {s: row["proposed_threshold"] for s, row in calibration["scrolls"].items()
                      if s != "phercparis4"}
        if (calibration["manifest_sha256"] != cfg["manifest_sha256"]
                or thresholds != settings["background_thresholds"]):
            raise ValueError("Background thresholds do not match the calibrated dataset")
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    if cfg.get("segment_depth_reversals") is not None:
        if not canonical:
            raise ValueError("Segment depth orientation is supported by the canonical student only")
        mapping = cfg["segment_depth_reversals"]
        expected = {r['scroll']+'/'+r['segment'] for r in manifest['segments']}
        if set(mapping) != expected or any(type(v) is not bool for v in mapping.values()):
            raise ValueError("Depth orientation requires an explicit boolean for every segment")
        audit = Path(cfg["orientation_audit"])
        if file_sha256(audit) != cfg["orientation_audit_sha256"]:
            raise ValueError("Orientation audit checksum mismatch")
        audit_data = json.loads(audit.read_text())
        if audit_data["segment_depth_reversals"] != mapping:
            raise ValueError("Orientation settings differ from the reviewed audit")
        if audit_data.get("orientation_calibration_exclusions", {}) != cfg.get("orientation_calibration_exclusions", {}):
            raise ValueError("Orientation calibration patches must be excluded from validation")
        if audit_data.get("excluded_segments", []) != cfg.get("excluded_segments", []):
            raise ValueError("Ambiguous segments must be excluded according to the reviewed audit")
    if file_sha256(Path(cfg["manifest"]).parent/"patches.npz") != manifest["patches_sha256"]:
        raise ValueError("Patch manifest checksum mismatch")
    if manifest["seed"] != cfg["seed"]:
        raise ValueError("Training seed must match prepared validation/sampling seed")
    for key in ("optimizer_updates", "batch_size", "grad_acc_steps", "val_every", "save_every", "log_every"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < float(cfg.get("human_2d_weight", 1.)) < float("inf"):
        raise ValueError("human_2d_weight must be finite and positive")
    diagnostics_every = int(cfg.get("diagnostics_every", 0))
    if diagnostics_every < 0 or diagnostics_every % int(cfg["log_every"]):
        raise ValueError("diagnostics_every must be zero or a multiple of log_every")
    return cfg


def training_signature(cfg):
    """Fields that may not silently change during a full-state resume."""
    keys = ("manifest_sha256", "model_config", "distillation", "seed", "optimizer_updates",
            "batch_size", "grad_acc_steps", "world_size", "learning_rate", "warmup_steps",
            "weight_decay", "optimizer", "optimizer_momentum", "optimizer_nesterov", "grad_clip",
            "ema_decay", "ema_start", "mixed_precision", "image_normalization")
    return {**{key: cfg[key] for key in keys}, "paris4_per_batch": cfg.get("paris4_per_batch", 0),
            "paris4_per_rank_update": cfg.get("paris4_per_rank_update", 0),
            "paris4_per_global_update": cfg.get("paris4_per_global_update", 0),
            "human_2d_weight": cfg.get("human_2d_weight", 1.),
            "segment_depth_reversals": cfg.get("segment_depth_reversals"),
            "excluded_segments": cfg.get("excluded_segments", []),
            "orientation_calibration_exclusions": cfg.get("orientation_calibration_exclusions", {})}


def validate_resume_signature(previous, current, *, remove_depth_prior=False, rebalance=False,
                              add_ct_context=False, replace_coarse_teacher=False):
    """Check explicit, bounded architecture/objective transitions and reject unrelated changes."""
    defaults = {"paris4_per_rank_update": 0, "paris4_per_global_update": 0,
                "human_2d_weight": 1., "segment_depth_reversals": None,
                "orientation_calibration_exclusions": {}, "excluded_segments": []}
    previous, current = deepcopy({**defaults, **previous}), deepcopy({**defaults, **current})
    # Paths are deployment details. Teacher content is independently SHA256 checked;
    # manifests keep their original bytes and model code comes from this checkout.
    for signature in (previous, current):
        canonical = signature.get("model_config", {}).get("canonical")
        if canonical is not None:
            canonical["source_dir"] = "<bundled-canonical-runtime>"
        for teacher in signature.get("distillation", {}).values():
            if teacher.get("sha256"):
                teacher["checkpoint"] = teacher["sha256"]
    for signature in (previous, current):
        if "projection" in signature.get("model_config", {}):
            signature["model_config"]["projection"].setdefault("ct_context", False)
    if previous == current:
        return False
    migrated = deepcopy(previous)
    projection = migrated.get("model_config", {}).get("projection", {})
    old_sigma = projection.get("prior_sigma")
    new_sigma = current.get("model_config", {}).get("projection", {}).get("prior_sigma")
    if remove_depth_prior and old_sigma is not None and old_sigma > 0 and new_sigma is None:
        projection["prior_sigma"] = None
    if (add_ct_context and not projection.get("ct_context", False)
            and current.get("model_config", {}).get("projection", {}).get("ct_context") is True):
        projection["ct_context"] = True
    if rebalance:
        for key in ("paris4_per_batch", "paris4_per_rank_update", "paris4_per_global_update", "human_2d_weight"):
            if key in current:
                migrated[key] = current[key]
        if "coarse" in current.get("distillation", {}):
            weight = current["distillation"]["coarse"].get("weight")
            if weight is not None and 0 < float(weight) < float("inf"):
                migrated["distillation"]["coarse"]["weight"] = weight
    if replace_coarse_teacher:
        old_teacher = migrated.get("distillation", {}).get("coarse", {})
        new_teacher = current.get("distillation", {}).get("coarse", {})
        for key in ("checkpoint", "sha256"):
            if key in old_teacher and new_teacher.get(key):
                old_teacher[key] = new_teacher[key]
    if migrated == current:
        return True
    raise ValueError("Resume configuration changes the model, data, batch, or schedule")


def load_transition_model(model, state, *, add_ct_context=False):
    if add_ct_context:
        initialized = model.state_dict()
        missing = initialized.keys() - state.keys()
        if not missing or any(not name.startswith("ct_attention.") for name in missing) or state.keys() - initialized.keys():
            raise ValueError("Only new CT attention parameters may be missing")
        initialized.update(state)
        state = initialized
    load_model_state(model, state)


def load_transition_optimizer(optimizer, state, model, *, add_ct_context=False):
    if add_ct_context:
        current = optimizer.state_dict()
        if len(state["param_groups"]) != 1 or len(current["param_groups"]) != 1:
            raise ValueError("CT context transition requires the single-group optimizer recipe")
        old_group, new_group = state["param_groups"][0], current["param_groups"][0]
        count = len(old_group["params"])
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        if (new_group["params"][:count] != old_group["params"] or len(names) <= count
                or any(name.startswith("ct_attention.") for name in names[:count])
                or any(not name.startswith("ct_attention.") for name in names[count:])):
            raise ValueError("Existing optimizer parameter order changed")
        state = {**state, "param_groups": [{**old_group, "params": new_group["params"]}]}
    optimizer.load_state_dict(state)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def seed_worker(_):
    torch.set_num_threads(1)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.lerp_(source.detach(), 1-decay)
    for target, source in zip(ema.buffers(), model.buffers(), strict=True):
        target.copy_(source)


def train(config_path):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from vesuvius.ink_detection.training.train import create_training_scheduler, _write_json_replace

    authored = json.loads(Path(config_path).read_text())
    accelerator = Accelerator(mixed_precision=authored.get("mixed_precision", "bf16"),
                              kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])
    cfg = resolve_config(config_path)
    cfg["world_size"] = accelerator.num_processes
    cfg["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    cfg["torch_version"] = str(torch.__version__)
    cfg["cuda_version"] = torch.version.cuda
    cfg["gpu_model"] = torch.cuda.get_device_name(accelerator.device)
    if accelerator.num_processes != int(cfg.get("expected_world_size", 4)):
        raise ValueError("Launch with the configured GPU count; do not silently change the batch")
    batch_size, accumulation = int(cfg["batch_size"]), int(cfg["grad_acc_steps"])
    effective_batch = batch_size * accumulation * accelerator.num_processes
    if effective_batch != int(cfg.get("effective_batch_size", 16)):
        raise ValueError(f"Effective batch mismatch: got {effective_batch}")
    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = Path(cfg["out_dir"])
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        if (out/"resolved_config.json").exists() and not cfg.get("resume"):
            raise FileExistsError("Existing run requires explicit resume; refusing to overwrite")
    accelerator.wait_for_everyone()

    ink_config = InkConfig.from_mapping(cfg)
    model = make_model(ink_config)
    initial = torch.load(cfg["initialization_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if cfg.get("projection", {}).get("kind") == "canonical_logits":
        model.load_canonical_state(initial)
    else:
        _, weights = select_inference_weights(initial)
        load_model_state(model.backbone, weights)
        del weights
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    del initial
    views = SimpleNamespace(ink=ink_config, optimizer=OptimizerConfig.from_mapping(cfg),
                            scheduler=SchedulerConfig.from_mapping(cfg, max_steps=cfg["optimizer_updates"]),
                            max_steps=cfg["optimizer_updates"])
    optimizer = create_training_optimizer(model, views)
    scheduler = create_training_scheduler(optimizer, views)
    completed, best = 0, -float("inf")
    resume = None
    transitioned = False
    added_ct_context = False
    if cfg.get("resume"):
        resume = load_checkpoint(cfg["resume"])
        transitioned = validate_resume_signature(
            resume["training_signature"], training_signature(cfg),
            remove_depth_prior=cfg.get("resume_remove_depth_prior", False),
            rebalance=cfg.get("resume_rebalance", False),
            add_ct_context=cfg.get("resume_add_ct_context", False),
            replace_coarse_teacher=cfg.get("resume_replace_coarse_teacher", False))
        added_ct_context = (transitioned and getattr(model, "ct_attention", None) is not None
                            and not resume["config"].get("projection", {}).get("ct_context", False))
        if transitioned:
            if cfg.get("resume_replace_coarse_teacher"):
                from vesuvius.ink_detection.training.multiteacher_handoff import verify_exported_teacher
                verify_exported_teacher(cfg["resume"], cfg["distillation"]["coarse"]["checkpoint"])
            if out.resolve() == Path(cfg["resume"]).resolve().parent or (out/"latest.pth").exists():
                raise ValueError("A training transition requires a new output directory")
            if not cfg.get("initial_validation", True):
                raise ValueError("A training transition requires a new validation baseline")
            cfg["resume_transition"] = {
                "reason": ("Replace coarse teacher with the resumed checkpoint's frozen EMA 3D backbone"
                           if cfg.get("resume_replace_coarse_teacher")
                           else "Add CT residual attention and rebalance PH4/human supervision" if added_ct_context
                           else "Rebalance PH4/human supervision" if cfg.get("resume_rebalance")
                           else "Remove fixed Gaussian depth penalty"),
                "parent_checkpoint": str(Path(cfg["resume"]).resolve()),
                "parent_wandb_run_id": resume.get("wandb_run_id"),
                "optimizer_update": int(resume["optimizer_step"]),
                "previous_signature": resume["training_signature"],
                "previous_transition": resume["config"].get("resume_transition"),
                "objective_rebalanced": bool(cfg.get("resume_rebalance")),
            }
        elif resume["config"].get("resume_transition"):
            cfg["resume_transition"] = deepcopy(resume["config"]["resume_transition"])
        load_transition_model(model, resume["model"], add_ct_context=added_ct_context)
        load_transition_optimizer(optimizer, resume["optimizer"], model, add_ct_context=added_ct_context)
        scheduler.load_state_dict(resume["lr_scheduler"])
        completed, best = int(resume["optimizer_step"]), float(resume["best_metric"])
        if transitioned:
            best = -float("inf")
    elif any(cfg.get(key) for key in ("resume_remove_depth_prior", "resume_rebalance",
                                     "resume_add_ct_context", "resume_replace_coarse_teacher")):
        raise ValueError("A training transition requires a source checkpoint")
    if accelerator.is_main_process:
        _write_json_replace(out/"resolved_config.json", cfg)
    # Copy before Accelerate replaces forward with an autocast closure. Copying
    # that closure can leave EMA inference bound to the live student's weights.
    ema = deepcopy(model).to(accelerator.device).eval()
    assert getattr(ema.forward, "__self__", None) is ema
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    if resume:
        load_transition_model(ema, resume["ema_model"], add_ct_context=added_ct_context)
    model, optimizer = accelerator.prepare(model, optimizer)
    unwrapped = accelerator.unwrap_model(model)
    teachers = FrozenInkTeachers(cfg["distillation"], accelerator.device)

    sampling_options = dict(world_size=accelerator.num_processes, batch_size=batch_size,
                            paris4_per_batch=int(cfg.get("paris4_per_batch", 0)),
                            grad_acc_steps=accumulation,
                            paris4_per_rank_update=int(cfg.get("paris4_per_rank_update", 0)),
                            paris4_per_global_update=int(cfg.get("paris4_per_global_update", 0)))
    input_options = dict(path_roots=cfg.get("path_roots", {}), student_normalization=cfg["image_normalization"],
                         valid_depth_margin=cfg.get("valid_depth_margin", 0),
                         segment_depth_reversals=cfg.get("segment_depth_reversals"),
                         excluded_segments=cfg.get("excluded_segments"),
                         orientation_calibration_exclusions=cfg.get("orientation_calibration_exclusions"))
    dataset = FlatDistillationDataset(cfg["manifest"], cfg["teacher_normalizations"],
                                      **input_options, **sampling_options)
    val_dataset = FlatDistillationDataset(cfg["manifest"], cfg["teacher_normalizations"],
                                        validation=True, val_patches_per_scroll=cfg.get("val_patches_per_scroll", 128),
                                        **input_options)
    workers = int(cfg.get("dataloader_workers", 4))
    loader_options = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(cfg["seed"] + accelerator.process_index))
    if workers:
        loader_options.update(multiprocessing_context="spawn", persistent_workers=True,
                              prefetch_factor=int(cfg.get("prefetch_factor", 2)))
    stop_at = min(int(cfg["optimizer_updates"]), int(cfg.get("stop_after_updates", cfg["optimizer_updates"])))
    sampler = RankDrawSampler(completed*effective_batch, stop_at*effective_batch,
                              accelerator.process_index, accelerator.num_processes)
    # Do not prepare this loader with Accelerate: RankDrawSampler already shards it.
    loader = DataLoader(dataset, sampler=sampler, **loader_options)
    val_sampler = RankDrawSampler(0, len(val_dataset), accelerator.process_index, accelerator.num_processes)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_options)
    resume_rng = None
    if resume:
        resume_rng = resume["rng_by_rank"][accelerator.process_index]
    else:
        set_seed(int(cfg["seed"]) + accelerator.process_index)
    del resume

    run = None
    if accelerator.is_main_process and cfg.get("wandb_mode", "online") != "disabled":
        import wandb
        kwargs = dict(entity=cfg["wandb_entity"], project=cfg["wandb_project"],
                      name=cfg["wandb_run_name"], config=cfg, dir=str(out),
                      mode=cfg.get("wandb_mode", "online"))
        if cfg.get("resume") and not transitioned and not cfg.get("new_wandb_run", False):
            previous = torch.load(cfg["resume"], map_location="cpu", weights_only=False, mmap=True)
            kwargs.update(id=previous["wandb_run_id"], resume="must")
            del previous
        run = wandb.init(**kwargs)
        if cfg.get("resume"):
            parent = torch.load(cfg["resume"], map_location="cpu", weights_only=False, mmap=True)
            run.summary["parent_wandb_run_id"] = parent.get("wandb_run_id")
            run.summary["parent_optimizer_update"] = parent["optimizer_step"]
            del parent
        artifact = wandb.Artifact("flat-multiteacher-inputs", type="run-inputs")
        artifact.add_file(str(out/"resolved_config.json"))
        artifact.add_file(cfg["manifest"])
        artifact.add_file(str(Path(cfg["manifest"]).parent/"patches.npz"))
        for path in Path(cfg.get("provenance_dir", out/"provenance")).glob("*"):
            if path.is_file():
                artifact.add_file(str(path))
        if cfg.get("background_calibration_dir"):
            artifact.add_dir(cfg["background_calibration_dir"], name="background-calibration")
            if not cfg.get("resume") or transitioned:
                for path in sorted(Path(cfg["background_calibration_dir"]).glob("*.png")):
                    run.log({f"calibration/{path.stem}": wandb.Image(str(path))}, step=completed)
        run.log_artifact(artifact)
        _write_json_replace(out/"wandb_run.json", {"id": run.id, "url": run.url})
        print(f"WANDB_URL={run.url}", flush=True)

    def log(values, step):
        if accelerator.is_main_process:
            values = {k: float(v) for k, v in values.items()}
            with (out/"metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"optimizer_update": step, **values})+"\n")
            if run:
                run.log(values, step=step)

    @torch.no_grad()
    def validate(step):
        # The unwrapped EMA avoids DDP collectives for unequal validation lengths.
        ema.eval()
        metrics = ScrollMetrics(val_dataset.records, accelerator.device)
        for batch in val_loader:
            batch = move_batch(batch, accelerator.device)
            with accelerator.autocast():
                targets, _, _ = teachers.generate(batch)
                output = ema(batch["image"], batch["valid_3d"])
            metrics.add(output, batch, targets)
        result = metrics.compute(accelerator)
        log(result, step)
        return result

    @torch.no_grad()
    def previews(step):
        if accelerator.is_main_process:
            for split in ("train", "validation"):
                # Stable draws and no photometric/geometric randomness for comparison.
                preview_dataset = (val_dataset if split == "validation" else
                                   FlatDistillationDataset(cfg["manifest"], cfg["teacher_normalizations"],
                                                           augment=False, **input_options, **sampling_options))
                selected = {}
                for draw in range(min(len(preview_dataset), 10000)):
                    record_id, _, _ = preview_dataset.locate(draw)
                    record = preview_dataset.records[record_id]
                    scroll = record["scroll"]
                    if scroll in selected:
                        continue
                    sample = preview_dataset[draw]
                    # Prefer ink-containing examples to make depth localization inspectable.
                    if not (sample["labels_2d"]*sample["mask_2d"]).any():
                        continue
                    selected[scroll] = True
                    batch = move_batch(default_collate([sample]), accelerator.device)
                    with accelerator.autocast():
                        targets, _, original = teachers.generate(batch)
                        output = ema(batch["image"], batch["valid_3d"])
                    path = out/"previews"/f"{step:06d}_{split}_{scroll}.png"
                    caption = save_preview(path, batch, output, targets, original, preview_dataset.records)
                    if run:
                        import wandb
                        run.log({f"previews/{split}/{scroll}": wandb.Image(str(path), caption=caption)}, step=step)
                    if len(selected) == len(preview_dataset.scrolls):
                        break
                if len(selected) != len(preview_dataset.scrolls):
                    raise RuntimeError(f"Missing labeled previews in {split}: {selected.keys()}")
        accelerator.wait_for_everyone()

    def save(step, metrics=None, *, is_best=False, final=False):
        states = [None] * accelerator.num_processes
        if accelerator.num_processes > 1:
            torch.distributed.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if accelerator.is_main_process:
            payload = {"model": unwrapped.state_dict(), "ema_model": ema.state_dict(),
                       "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
                       "step": step-1, "optimizer_step": step, "config": cfg,
                       "rng_by_rank": states, "best_metric": best,
                       "training_signature": training_signature(cfg),
                       "wandb_run_id": run.id if run else None, "validation_metrics": metrics}
            target = out/f"ckpt_{step:06d}.pth"
            temporary = target.with_suffix(".partial")
            torch.save(payload, temporary)
            os.replace(temporary, target)
            for alias, enabled in (("latest.pth", True), ("best.pth", is_best), ("final.pth", final)):
                if enabled:
                    temporary_link = out/(alias+".partial")
                    temporary_link.unlink(missing_ok=True)
                    temporary_link.symlink_to(target.name)
                    os.replace(temporary_link, out/alias)
            checkpoints = sorted(out.glob("ckpt_*.pth"))
            protected = {p.resolve() for p in out.glob("*.pth") if p.is_symlink()}
            for old in checkpoints[:-3]:
                if int(old.stem.split("_")[-1]) % 10000 and old.resolve() not in protected:
                    old.unlink()
            _write_json_replace(out/"status.json", {"optimizer_update": step,
                                "target_updates": cfg["optimizer_updates"], "checkpoint": str(target),
                                "finished": final and step == cfg["optimizer_updates"]})
        accelerator.wait_for_everyone()

    stopping = False
    def request_stop(signum, _):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    initial_metrics = validate(completed) if cfg.get("initial_validation", True) else None
    if cfg.get("previews", True):
        previews(completed)
    if initial_metrics and (not cfg.get("resume") or transitioned):
        best = initial_metrics["val/macro/pr_auc"]
        if resume_rng is not None:
            restore_rng(resume_rng)
        save(completed, initial_metrics, is_best=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if resume_rng is not None:
        restore_rng(resume_rng)
    iterator = iter(loader)
    durations = []
    last_metrics = initial_metrics
    try:
        while completed < stop_at:
            started = time.perf_counter()
            losses = {}
            diagnostics_every = int(cfg.get("diagnostics_every", 0))
            diagnose = bool(diagnostics_every and (completed+1) % diagnostics_every == 0)
            seen = torch.zeros(len(dataset.records), device=accelerator.device)
            for microstep in range(accumulation):
                batch = move_batch(next(iterator), accelerator.device)
                seen += torch.bincount(batch["record_id"], minlength=len(seen))
                context = accelerator.no_sync(model) if microstep+1 < accumulation else nullcontext()
                with context:
                    with accelerator.autocast():
                        targets, weights, original = teachers.generate(batch)
                        output = model(batch["image"], batch["valid_3d"])
                    loss, parts = joint_loss(output, batch, targets, weights,
                                             human_2d_weight=float(cfg.get("human_2d_weight", 1.)),
                                             gradient_diagnostics=diagnose)
                    if diagnose:
                        parts.update(suppression_statistics(output, batch, targets))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite joint loss")
                    accelerator.backward(loss/accumulation)
                for key, value in parts.items():
                    losses[key] = losses.get(key, 0.) + value/accumulation
                losses["teacher/mean_probability"] = original.mean().detach()
                coarse = batch["teacher_id"] == 1
                dark = teachers.dark_mask(batch)[coarse]
                losses["teacher/coarse_dark_fraction"] = (dark.float().mean() if dark.numel()
                                                           else torch.zeros((), device=accelerator.device))
            norm = accelerator.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite student gradients")
            optimizer.step()
            if accelerator.optimizer_step_was_skipped:
                raise RuntimeError("Unexpected skipped bf16 optimizer update")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            update_ema(ema, unwrapped, cfg["ema_decay"] if completed > cfg["ema_start"] else 0.)
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            durations.append(elapsed)
            if completed % cfg["log_every"] == 0 or completed == 1:
                values = {k: accelerator.reduce(v, reduction="mean").item() for k, v in losses.items()}
                values = finish_loss_metrics(values)
                values.update({"train/lr": optimizer.param_groups[0]["lr"], "train/grad_norm": norm.item(),
                               "train/update_seconds": elapsed, "train/samples_per_second": effective_batch/elapsed,
                               "train/peak_gpu_gb": torch.cuda.max_memory_allocated()/1e9})
                counts = accelerator.reduce(seen, reduction="sum").cpu().tolist()
                for scroll in dataset.scrolls:
                    values[f"sampling/{scroll}"] = sum(counts[i] for i in dataset.groups[scroll])
                log(values, completed)
                accelerator.print(f"update={completed}/{cfg['optimizer_updates']} loss2d={values['loss/2d']:.5f} "
                                  f"loss3d={values['loss/3d']:.5f} seconds={elapsed:.3f}", flush=True)
            is_best = False
            if completed % cfg["val_every"] == 0:
                last_metrics = validate(completed)
                score = last_metrics["val/macro/pr_auc"]
                if score > best:
                    best, is_best = score, True
                if cfg.get("previews", True):
                    previews(completed)
            low_disk = shutil.disk_usage(out).free < float(cfg.get("min_free_gb", 200))*1e9
            stop = torch.tensor(int(stopping or low_disk), device=accelerator.device)
            stop = bool(accelerator.reduce(stop, reduction="sum").item())
            if completed % cfg["save_every"] == 0 or is_best or completed == stop_at or stop:
                save(completed, last_metrics, is_best=is_best, final=completed == cfg["optimizer_updates"])
            if stop:
                accelerator.print("Stopped safely at an optimizer boundary (signal or low disk)")
                break
        if accelerator.is_main_process:
            measured = durations[min(5, max(len(durations)-1, 0)):]
            _write_json_replace(out/"timing.json", {"updates_measured": len(measured),
                "effective_batch": effective_batch, "gpus": accelerator.num_processes,
                "mean_seconds": float(np.mean(measured)) if measured else None,
                "p50_seconds": float(np.median(measured)) if measured else None,
                "p95_seconds": float(np.percentile(measured, 95)) if measured else None})
            if run:
                import wandb
                artifact = wandb.Artifact("flat-multiteacher-student", type="model")
                for name in ("best.pth", "final.pth"):
                    if (out/name).exists():
                        artifact.add_file(str((out/name).resolve()), name=name)
                run.log_artifact(artifact)
                run.finish()
    except BaseException as exc:
        if accelerator.is_main_process:
            _write_json_replace(out/"failure.json", {"update": completed, "error": repr(exc)})
        raise
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return 0
