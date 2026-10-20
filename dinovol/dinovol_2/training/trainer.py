# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in LICENSES/Apache-2.0.txt.
#
# Modified for Dinovol's three-dimensional training pipeline.

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from dinovol_2.dataset.cuda_pipeline import CudaBatchProcessor, CudaDataLoader
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset

from dinovol_2.eval.task_eval import TaskEvalRunner
from dinovol_2.ops.distributed_utils import build_distributed_sampler, resolve_distributed_config
from dinovol_2.ops.weighted_loader import WeightedCombinedLoader
from dinovol_2.ops.precision import resolve_amp_dtype


from dinovol_2.methods import build_method
from .config import normalize_config
from dinovol_2.training.schedules import CosineScheduler

def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    return float(value[0]), float(value[1])


def _as_3tuple(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"Expected 3 values, got {result}")
    return result


def _config_get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return default


class Trainer:
    def __init__(self, config: Mapping[str, Any], *, method=None) -> None:
        self.config = normalize_config(config)
        self.runtime_config = self.config["runtime"]
        self.optimizer_config = self.config["optimizer"]
        self.data_config = self.config["data"]
        distributed_config = resolve_distributed_config(self.runtime_config)
        self.is_distributed = bool(distributed_config["use_ddp"])
        self.rank = int(distributed_config["rank"])
        self.local_rank = int(distributed_config["local_rank"])
        self.world_size = int(distributed_config["world_size"])

        configured_device_name = self.runtime_config.get("device")
        configured_device = torch.device(configured_device_name) if configured_device_name is not None else None
        prefers_cuda = configured_device is None or configured_device.type == "cuda"
        if self.is_distributed and prefers_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        elif configured_device is not None:
            self.device = configured_device
            if self.device.type == "cuda":
                torch.cuda.set_device(0 if self.device.index is None else self.device.index)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.is_distributed:
            if self.world_size < 2 and not dist.is_initialized():
                raise ValueError(
                    "DDP requested but WORLD_SIZE is 1. Launch with torchrun, for example: "
                    "torchrun --nproc_per_node=NUM_GPUS pretrain.py ..."
                )
            if not dist.is_initialized():
                backend = "nccl" if self.device.type == "cuda" else "gloo"
                dist.init_process_group(backend=backend, init_method="env://")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        if self.device.type == "cuda" and bool(self.runtime_config.get("cuda_expandable_segments", False)):
            # Expandable segments avoid fragmentation-driven cache flushes near full
            # memory. They apply to allocations made after this call.
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        if self.device.type == "cuda" and bool(self.runtime_config.get("cudnn_benchmark", False)):
            # Autotune convolution algorithms per input shape (crop shapes are fixed).
            torch.backends.cudnn.benchmark = True
        self.use_amp = bool(self.runtime_config.get("use_amp", self.device.type == "cuda"))
        self.amp_dtype = resolve_amp_dtype(self.runtime_config.get("amp_dtype", "bfloat16"))
        self.runtime_config["amp_dtype"] = str(self.amp_dtype).removeprefix("torch.")
        self.fused_optimizer = bool(self.optimizer_config.get("fused_optimizer", False)) and self.device.type == "cuda"
        self.max_iterations = int(self.runtime_config.get("max_iterations", self.runtime_config.get("num_iterations", 1000000)))
        self.total_steps = self.max_iterations
        self.base_lr = float(self.optimizer_config.get("lr", 1e-4))
        self.min_lr = float(self.optimizer_config.get("min_lr", 1e-6))
        self.lr_schedule_type = str(self.optimizer_config.get("lr_schedule", "constant")).lower()
        if self.lr_schedule_type not in {"cosine", "constant"}:
            raise ValueError("lr_schedule must be 'cosine' or 'constant'")
        self.base_weight_decay = float(self.optimizer_config.get("weight_decay", 0.04))
        self.final_weight_decay = float(self.optimizer_config.get("weight_decay_end", self.base_weight_decay))
        self.betas = tuple(self.optimizer_config.get("betas", (0.9, 0.999)))
        self.clip_grad = float(self.optimizer_config.get("clip_grad", 3.0))
        self.layer_decay = float(self.optimizer_config.get("layer_decay", self.optimizer_config.get("layerwise_decay", 1.0)))
        self.patch_embed_lr_mult = float(self.optimizer_config.get("patch_embed_lr_mult", 0.2))

        warmup_ratio = float(self.optimizer_config.get("warmup_ratio", 0.1))
        default_warmup_steps = 0
        if self.total_steps > 0 and warmup_ratio > 0.0:
            default_warmup_steps = max(1, round(self.total_steps * warmup_ratio))
        self.warmup_steps = int(self.optimizer_config["warmup_steps"]) if "warmup_steps" in self.optimizer_config else default_warmup_steps
        self.grad_accumulation_steps = int(self.runtime_config.get("grad_accumulation_steps", 1))
        if self.grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be at least 1")
        self.method = method if method is not None else build_method(
            self.config, device=self.device, is_distributed=self.is_distributed, rank=self.rank)
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.device.type == "cuda" and self.amp_dtype == torch.float16)
        self.lr_schedule, self.wd_schedule = self._build_schedulers()

        self.log_every = int(self.runtime_config.get("log_every", 20))
        self.val_every_n = int(self.runtime_config.get("val_every_n", 0))
        # Save a monitor image from the current training batch every N steps (0 disables).
        self.train_monitor_every_n = int(self.runtime_config.get("train_monitor_every_n", 0))
        self.save_every_n = int(self.runtime_config.get("save_every_n", self.runtime_config.get("save_every", 0)))
        self.monitor_batch_size = max(5, int(self.runtime_config.get("monitor_batch_size", 5)))
        # Monitor PCA panels: components 1-3, 4-6, ... as RGB, `monitor_pca_sets` of them.
        self.monitor_pca_sets = max(1, int(self.runtime_config.get("monitor_pca_sets", 4)))
        self.monitor_pool_size = max(1, int(self.runtime_config.get("monitor_pool_size", 5000)))
        self.monitor_seed = int(self.runtime_config.get("monitor_seed", 0))
        self.output_dir = Path(self.runtime_config.get("output_dir", "./dinov2_pretrain_runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_dir = self.output_dir / "monitor"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_project = _config_get(self.runtime_config, "wandb-project", "wandb_project")
        self.wandb_entity = _config_get(self.runtime_config, "wandb-entity", "wandb_entity")
        self.wandb_run_name = _config_get(self.runtime_config, "wandb-run-name", "wandb_run_name")
        self.wandb_run_id = _config_get(self.runtime_config, "wandb-run-id", "wandb_run_id")
        self._wandb: Any | None = None
        self._warned_missing_val_dataset = False
        self.resume = bool(self.runtime_config.get("resume", False))
        self.auto_resume = bool(self.runtime_config.get("auto_resume", self.resume or bool(self.runtime_config.get("resume_from"))))
        self.resume_path = self._resolve_resume_path()
        configured_wandb_resume = _config_get(self.runtime_config, "wandb-resume", "wandb_resume")
        self.wandb_resume = self._normalize_wandb_resume_mode(
            configured_wandb_resume if configured_wandb_resume is not None else ("allow" if self.resume_path is not None else None)
        )
        self._monitor_dataset: SSLZarrDataset | None = None
        self._monitor_collate_fn: Any | None = None
        self._monitor_seed_pool = tuple(self.monitor_seed + offset for offset in range(self.monitor_pool_size))
        self._monitor_selection_rng = random.Random(self.monitor_seed)
        self._train_sampler_epoch = 0
        self._val_sampler_epoch = 0
        self.task_eval_every = int(self.runtime_config.get("task_eval_every", 0))
        self._task_evaluator = (
            TaskEvalRunner(
                self.runtime_config,
                output_dir=self.output_dir,
                device=self.device,
                use_amp=self.use_amp,
            )
            if self.task_eval_every > 0
            else None
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @staticmethod
    def _normalize_wandb_resume_mode(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "allow" if value else None

        normalized = str(value).strip().lower()
        if normalized in {"", "false", "none", "off"}:
            return None
        if normalized == "true":
            return "allow"
        if normalized not in {"allow", "must", "never", "auto"}:
            raise ValueError(
                "wandb_resume must be one of allow, must, never, auto, true, false, or null."
            )
        return normalized

    def _wandb_metadata_path(self) -> Path:
        return self.output_dir / "wandb_run.json"

    def _read_wandb_metadata(self) -> Mapping[str, Any]:
        metadata_path = self._wandb_metadata_path()
        if not metadata_path.exists():
            return {}

        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            if self.is_main_process:
                print(f"warning: failed to read wandb metadata from {metadata_path}: {exc}")
            return {}

        return payload if isinstance(payload, dict) else {}

    def _current_wandb_run_id(self) -> str | None:
        if self._wandb_enabled():
            run_id = getattr(self._wandb.run, "id", None)
            if run_id:
                return str(run_id)
        if self.wandb_run_id:
            return str(self.wandb_run_id)
        return None

    def _write_wandb_metadata(self) -> None:
        if not self.is_main_process:
            return

        run_id = self._current_wandb_run_id()
        if not run_id:
            return

        run = getattr(self._wandb, "run", None) if self._wandb is not None else None
        payload = {
            "id": run_id,
            "name": getattr(run, "name", None) or self.wandb_run_name,
            "project": self.wandb_project,
            "entity": self.wandb_entity,
            "resume": self.wandb_resume,
        }
        metadata_path = self._wandb_metadata_path()
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _resolve_wandb_run_id(self) -> str | None:
        if self.wandb_run_id:
            return str(self.wandb_run_id)
        if self.resume_path is None:
            return None

        metadata = self._read_wandb_metadata()
        run_id = metadata.get("id")
        if run_id:
            return str(run_id)
        return None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        param_groups = self.method.parameter_groups(
            lr_decay_rate=self.layer_decay,
            patch_embed_lr_mult=self.patch_embed_lr_mult,
        )
        return torch.optim.AdamW(
            param_groups,
            lr=self.base_lr,
            betas=self.betas,
            fused=True if self.fused_optimizer else None,
        )

    def _build_dataloader_kwargs(self) -> dict[str, Any]:
        num_workers = int(_config_get(self.data_config, "dataloader-workers", "dataloader_workers", "num_workers", default=0))
        if num_workers < 0:
            raise ValueError(f"dataloader workers must be non-negative, got {num_workers}")

        kwargs: dict[str, Any] = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
            "drop_last": True,
        }

        prefetch_factor = _config_get(self.data_config, "prefetch-factor", "prefetch_factor")
        if prefetch_factor is not None:
            prefetch_factor = int(prefetch_factor)
            if prefetch_factor <= 0:
                raise ValueError(f"prefetch factor must be positive, got {prefetch_factor}")
            if num_workers <= 0:
                raise ValueError("prefetch factor requires dataloader workers greater than zero")
            kwargs["prefetch_factor"] = prefetch_factor

        persistent_workers = _config_get(self.data_config, "persistent-workers", "persistent_workers")
        if persistent_workers is not None:
            persistent_workers = bool(persistent_workers)
            if persistent_workers and num_workers <= 0:
                raise ValueError("persistent workers requires dataloader workers greater than zero")
            kwargs["persistent_workers"] = persistent_workers

        return kwargs

    def _resolved_dataset_configs(
        self,
        key: str,
        *,
        enable_point_supervision: bool = False,
    ) -> list[dict[str, Any]] | None:
        dataset_config = self.data_config.get(key)
        if dataset_config is None:
            return None

        base_config = dict(dataset_config)
        variants = base_config.pop("variants", None)
        if not variants:
            return [self._prepare_dataset_config(
                base_config,
                enable_point_supervision=enable_point_supervision,
            )]

        resolved_configs: list[dict[str, Any]] = []
        for variant in variants:
            variant_overrides = {name: value for name, value in dict(variant).items() if name != "ratio"}
            merged_config = dict(base_config)
            merged_config.update(variant_overrides)
            resolved_configs.append(self._prepare_dataset_config(
                merged_config,
                enable_point_supervision=enable_point_supervision,
            ))
        return resolved_configs

    def _dataset_variant_weights(self, key: str) -> tuple[float, ...] | None:
        dataset_config = self.data_config.get(key)
        if dataset_config is None:
            return None
        variants = dataset_config.get("variants")
        if not variants:
            return None
        return tuple(float(dict(variant).get("ratio", 1.0)) for variant in variants)

    def _dataset_config(self, key: str) -> Mapping[str, Any] | None:
        resolved_configs = self._resolved_dataset_configs(key, enable_point_supervision=False)
        if resolved_configs is None:
            return None
        return dict(resolved_configs[0])

    def _build_ssl_dataloader(self, dataset_config: Mapping[str, Any], *, shuffle: bool) -> DataLoader:
        dataset = SSLZarrDataset(dataset_config, do_augmentations=True)
        collate_fn = self.method.build_collate(dataset)
        shuffle = bool(self.data_config.get("shuffle", shuffle))
        sampler = build_distributed_sampler(
            dataset,
            is_distributed=self.is_distributed,
            rank=self.rank,
            world_size=self.world_size,
            shuffle=shuffle,
        )
        return CudaDataLoader(
            dataset,
            device=self.device,
            batch_size=int(self.data_config.get("batch_size", 2)),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate_fn,
            **self._build_dataloader_kwargs(),
        )

    def _build_loader_from_configs(
        self,
        dataset_configs: list[dict[str, Any]],
        *,
        shuffle: bool,
        weights: tuple[float, ...] | None = None,
    ) -> DataLoader | WeightedCombinedLoader:
        if len(dataset_configs) == 1:
            return self._build_ssl_dataloader(dataset_configs[0], shuffle=shuffle)

        dataloaders = [self._build_ssl_dataloader(dataset_config, shuffle=shuffle) for dataset_config in dataset_configs]
        return WeightedCombinedLoader(
            dataloaders,
            weights=weights if weights is not None else tuple(1.0 for _ in dataloaders),
            seed=int(self.data_config.get("seed", 0)),
        )

    def build_dataloader(self) -> DataLoader | WeightedCombinedLoader:
        dataset_configs = self._resolved_dataset_configs("dataset", enable_point_supervision=True)
        if dataset_configs is None:
            raise ValueError("dataset configuration is required")
        return self._build_loader_from_configs(
            dataset_configs,
            shuffle=bool(self.data_config.get("shuffle", False)),
            weights=self._dataset_variant_weights("dataset"),
        )

    def build_val_dataloader(self) -> DataLoader | WeightedCombinedLoader | None:
        val_dataset_configs = self._resolved_dataset_configs("val_dataset", enable_point_supervision=False)
        val_weights = self._dataset_variant_weights("val_dataset")
        if val_dataset_configs is None:
            train_dataset_configs = self._resolved_dataset_configs("dataset", enable_point_supervision=False)
            if train_dataset_configs is None or len(train_dataset_configs) <= 1:
                return None
            val_dataset_configs = [dict(train_dataset_configs[0])]
            val_weights = None
        return self._build_loader_from_configs(
            val_dataset_configs,
            shuffle=False,
            weights=val_weights,
        )

    @staticmethod
    def _set_sampler_epoch(dataloader: DataLoader | WeightedCombinedLoader, epoch: int) -> None:
        set_epoch = getattr(dataloader, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
            return
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

    @staticmethod
    def _close_dataloader(dataloader: DataLoader | WeightedCombinedLoader | None) -> None:
        if dataloader is None:
            return
        close_loader = getattr(dataloader, "close", None)
        if callable(close_loader):
            close_loader()
            return
        iterator = getattr(dataloader, "_iterator", None)
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        dataset = getattr(dataloader, "dataset", None)
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    def _close_auxiliary_datasets(self) -> None:
        if self._monitor_dataset is None:
            pass
        else:
            self._monitor_dataset.close()
            self._monitor_dataset = None
            self._monitor_collate_fn = None
        if self._task_evaluator is not None:
            self._task_evaluator.close()

    def _average_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        averaged = {key: float(value) for key, value in metrics.items()}
        if not self.is_distributed:
            return averaged
        keys = list(averaged)
        values = torch.tensor([averaged[key] for key in keys], device=self.device, dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        return {key: float(values[index].item()) for index, key in enumerate(keys)}

    def _initialize_wandb(self) -> None:
        if self._wandb_enabled() or not self.is_main_process or not self.wandb_project:
            return
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "wandb logging requested via config but wandb is not installed. "
                "Install `wandb[media]` to enable it."
            ) from exc

        run_id = self._resolve_wandb_run_id()
        if run_id:
            self.runtime_config["wandb_run_id"] = run_id
        if self.wandb_resume is not None:
            self.runtime_config["wandb_resume"] = self.wandb_resume

        init_kwargs: dict[str, Any] = {
            "project": self.wandb_project,
            "entity": self.wandb_entity,
            "config": self.config,
            "dir": str(self.output_dir),
            "settings": wandb.Settings(start_method="thread"),
        }
        if self.wandb_run_name:
            init_kwargs["name"] = self.wandb_run_name
        if run_id:
            init_kwargs["id"] = run_id
        if self.wandb_resume is not None:
            init_kwargs["resume"] = self.wandb_resume
        elif run_id:
            init_kwargs["resume"] = "allow"
        wandb.init(**init_kwargs)
        self._wandb = wandb
        self._wandb.define_metric("trainer/step", hidden=True)
        self._wandb.define_metric("train/*", step_metric="trainer/step")
        self._wandb.define_metric("val/*", step_metric="trainer/step")
        self._wandb.define_metric("task_eval/*", step_metric="trainer/step")
        self.wandb_run_id = self._current_wandb_run_id()
        if self.wandb_run_id is not None:
            self.runtime_config["wandb_run_id"] = self.wandb_run_id
        self._write_wandb_metadata()

    def _wandb_enabled(self) -> bool:
        return self._wandb is not None and getattr(self._wandb, "run", None) is not None

    def _log_wandb_metrics(
        self,
        prefix: str,
        metrics: Mapping[str, Any],
        *,
        step: int,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._wandb_enabled():
            return

        payload: dict[str, Any] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            payload[f"{prefix}/{key}"] = float(value) if isinstance(value, (int, float, np.number)) else value
        if extra:
            payload.update(extra)
        if payload:
            payload["trainer/step"] = int(step)
            self._wandb.log(payload)

    def _finish_wandb(self) -> None:
        if self._wandb_enabled():
            self._wandb.finish()

    def _get_monitor_source(self) -> tuple[SSLZarrDataset, Any]:
        if self._monitor_dataset is None or self._monitor_collate_fn is None:
            monitor_dataset_config = self.data_config.get("monitor_dataset")
            if monitor_dataset_config is None:
                monitor_dataset_config = self.data_config.get("val_dataset", self.data_config["dataset"])
            monitor_dataset_config = self._dataset_config("monitor_dataset") or self._dataset_config("val_dataset") or self._dataset_config("dataset")
            dataset = SSLZarrDataset(monitor_dataset_config, do_augmentations=True)
            collate_fn = self.method.build_collate(dataset)
            self._monitor_dataset = dataset
            self._monitor_collate_fn = CudaBatchProcessor(dataset.config, collate_fn, device=self.device)
        return self._monitor_dataset, self._monitor_collate_fn

    def build_monitor_batch(self, seed: int | None = None) -> dict[str, Any]:
        dataset, collate_fn = self._get_monitor_source()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            sample_count = min(len(dataset), self.monitor_batch_size)
            samples = [dataset[i] for i in range(sample_count)]
            return collate_fn(samples)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _next_monitor_seed(self) -> int:
        return self._monitor_seed_pool[self._monitor_selection_rng.randrange(len(self._monitor_seed_pool))]

    def sample_monitor_batch(self) -> dict[str, Any]:
        return self.build_monitor_batch(seed=self._next_monitor_seed())

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _optimizer_to_device(self) -> None:
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)

    def _find_latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(self.output_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        return checkpoints[-1]

    def _resolve_resume_path(self) -> Path | None:
        resume_from = self.runtime_config.get("resume_from")
        if resume_from:
            return Path(resume_from)
        if self.auto_resume:
            return self._find_latest_checkpoint()
        return None

    def _monitor_subset(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """The first monitor_batch_size samples of a collated batch, for save_monitor_image."""
        batch_size = int(batch["batch_size"])
        n_global_views = int(batch["n_global_views"])
        sample_count = min(batch_size, self.monitor_batch_size)
        index = torch.tensor([view * batch_size + sample for view in range(n_global_views) for sample in range(sample_count)])
        return {
            "collated_global_crops": batch["collated_global_crops"].index_select(0, index),
            "n_global_views": n_global_views,
            "batch_size": sample_count,
        }

    def _run_task_evals(self, step: int) -> None:
        if self._task_evaluator is None:
            return

        rng_state = self._capture_rng_state()
        try:
            results = self._task_evaluator.run(
                backbone=self.method.evaluation_backbone(),
                step=step,
            )
        finally:
            self._restore_rng_state(rng_state)

        if not self.is_main_process:
            return

        for task_name, result in results.items():
            image_path = result.get("image_path")
            metrics_to_log = {
                key: value
                for key, value in result.items()
                if key not in {"image_path", "val_sample"}
            }
            print(
                f"step={step} task_eval={task_name} "
                f"val_loss={result['val_loss']:.4f} "
                f"val_fg_dice={result['val_fg_dice']:.4f} "
                f"sample={result['val_sample']}"
            )
            self._log_wandb_metrics(
                f"task_eval/{task_name}",
                metrics_to_log,
                step=step,
                extra=(
                    {f"task_eval/{task_name}/validation_image": self._wandb.Image(str(image_path))}
                    if image_path is not None and self._wandb_enabled()
                    else None
                ),
            )

    def fit(self) -> None:
        start_step = 0
        resume_path = self.resume_path
        if resume_path is not None:
            start_step = self.load_checkpoint(resume_path) + 1

        dataloader = self.build_dataloader()
        self._set_sampler_epoch(dataloader, self._train_sampler_epoch)
        dataloader_iter: Iterator[Any] = iter(dataloader)
        val_dataloader = self.build_val_dataloader()
        if val_dataloader is not None:
            self._set_sampler_epoch(val_dataloader, self._val_sampler_epoch)
        val_dataloader_iter: Iterator[Any] | None = iter(val_dataloader) if val_dataloader is not None else None
        self._initialize_wandb()
        progress = tqdm(total=self.max_iterations, initial=start_step, desc="training", unit="iter") if self.is_main_process else None
        try:
            for step in range(start_step, self.max_iterations):
                micro_batches = []
                for _ in range(self.grad_accumulation_steps):
                    try:
                        micro_batches.append(next(dataloader_iter))
                    except StopIteration:
                        self._train_sampler_epoch += 1
                        self._set_sampler_epoch(dataloader, self._train_sampler_epoch)
                        dataloader_iter = iter(dataloader)
                        micro_batches.append(next(dataloader_iter))
                batch = micro_batches[-1]

                metrics = self._average_metrics(self.train_step(micro_batches, step))
                if progress is not None:
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}")
                    progress.update(1)
                if self.is_main_process:
                    self._log_wandb_metrics("train", metrics, step=step)

                if self.is_main_process and step % self.log_every == 0:
                    print(f"step={step} loss={metrics['loss']:.4f} lr={metrics['lr']:.2e}")
                if self.val_every_n and step > 0 and step % self.val_every_n == 0:
                    if val_dataloader_iter is None:
                        if self.is_main_process and not self._warned_missing_val_dataset:
                            print("val_every_n is set but no val_dataset is configured; skipping validation.")
                            self._warned_missing_val_dataset = True
                    else:
                        try:
                            val_batch = next(val_dataloader_iter)
                        except StopIteration:
                            self._val_sampler_epoch += 1
                            self._set_sampler_epoch(val_dataloader, self._val_sampler_epoch)
                            val_dataloader_iter = iter(val_dataloader)
                            val_batch = next(val_dataloader_iter)
                        val_metrics = self._average_metrics(self.validate(val_batch, step))
                        if self.is_main_process:
                            print(f"step={step} val_loss={val_metrics['loss']:.4f}")
                            image_path = self.save_monitor_image(val_batch, step, val_metrics)
                            self._log_wandb_metrics(
                                "val",
                                val_metrics,
                                step=step,
                                extra={"val/monitor_image": self._wandb.Image(str(image_path))} if self._wandb_enabled() and image_path is not None else None,
                            )
                if (
                    self.is_main_process
                    and self.train_monitor_every_n
                    and step > 0
                    and (step == start_step or step % self.train_monitor_every_n == 0)
                ):
                    image_path = self.save_monitor_image(self._monitor_subset(batch), step, metrics)
                    if self._wandb_enabled() and image_path is not None:
                        self._log_wandb_metrics(
                            "train",
                            {},
                            step=step,
                            extra={"train/monitor_image": self._wandb.Image(str(image_path))},
                        )
                if self.task_eval_every and step > 0 and step % self.task_eval_every == 0:
                    self._run_task_evals(step)
                if self.is_main_process and self.save_every_n and step > 0 and step % self.save_every_n == 0:
                    self.save_checkpoint(step)
        finally:
            if progress is not None:
                progress.close()
            self._close_dataloader(dataloader)
            self._close_dataloader(val_dataloader)
            self._close_auxiliary_datasets()

    def _prepare_dataset_config(self, dataset_config, *, enable_point_supervision=False):
        config = dict(dataset_config)
        config.setdefault("epoch_length", int(self.data_config.get("official_epoch_length", self.data_config.get("epoch_length", self.max_iterations))))
        return self.method.dataset_config(config, training=enable_point_supervision)

    def _build_schedulers(self):
        final_lr = self.base_lr if self.lr_schedule_type == "constant" else self.min_lr
        return (
            CosineScheduler(base_value=self.base_lr, final_value=final_lr, total_iters=self.total_steps,
                            warmup_iters=self.warmup_steps, start_warmup_value=0.0),
            CosineScheduler(base_value=self.base_weight_decay, final_value=self.final_weight_decay, total_iters=self.total_steps),
        )

    def _apply_optim_scheduler(self, step):
        lr, weight_decay = self.lr_schedule[step], self.wd_schedule[step]
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_multiplier", 1.0)
            group["weight_decay"] = weight_decay * group.get("wd_multiplier", 1.0)
        self.method.adjust_optimizer_groups(self.optimizer.param_groups, step)
        return lr, weight_decay

    def train_step(self, batch, step: int) -> dict[str, float]:
        micro_batches = list(batch) if isinstance(batch, (list, tuple)) else [batch]
        if not micro_batches:
            raise ValueError("An optimizer step requires at least one microbatch")
        self.method.train()
        lr, weight_decay = self._apply_optim_scheduler(step)
        self.optimizer.zero_grad(set_to_none=True)
        totals = None
        names = None
        for micro_batch in micro_batches:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.method(micro_batch, step)
            self.scaler.scale(output.loss / len(micro_batches)).backward()
            current_names = tuple(output.metrics)
            if names is not None and names != current_names:
                raise ValueError("Metric names must be consistent across microbatches")
            names = current_names
            values = torch.stack([value.detach().float().reshape(()) for value in output.metrics.values()])
            totals = values if totals is None else totals + values
        self.scaler.unscale_(self.optimizer)
        self.method.before_optimizer_step(step)
        clip_grad_norm_(self.method.gradient_parameters(), self.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.method.after_optimizer_step(step)
        metrics = dict(zip(names, (totals / len(micro_batches)).tolist()))
        return {**metrics, **self.method.step_metrics(step), "lr": lr, "weight_decay": weight_decay}

    def validate(self, batch, step: int) -> dict[str, float]:
        self.method.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            return dict(self.method.evaluate(batch, step))

    def save_monitor_image(self, batch, step, metrics):
        return self.method.save_monitor_image(batch, step, metrics, directory=self.monitor_dir, config=self.runtime_config)

    def save_checkpoint(self, step: int) -> Path:
        path = self.output_dir / f"checkpoint_step_{step:06d}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save({
            "step": step, "config": self.config, "wandb_run_id": self._current_wandb_run_id(),
            **self.method.checkpoint_state(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "rng_state": self._capture_rng_state(),
        }, tmp_path)
        tmp_path.replace(path)
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        step = int(checkpoint.get("step", -1))
        added = self.method.load_checkpoint_state(checkpoint, step)
        if "optimizer" in checkpoint:
            self.method.load_optimizer_state(self.optimizer, checkpoint["optimizer"], added)
            self._optimizer_to_device()
        if checkpoint.get("scaler") is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("wandb_run_id") and not self.wandb_run_id:
            self.wandb_run_id = str(checkpoint["wandb_run_id"])
            self.runtime_config["wandb_run_id"] = self.wandb_run_id
        return step
