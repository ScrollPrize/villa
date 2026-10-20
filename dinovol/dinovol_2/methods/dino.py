from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch._functorch import config as functorch_config
from torch.nn.parallel import DistributedDataParallel as DDP

from dinovol_2.loss import (
    DINOLoss, GramLoss, KoLeoLoss, ObjectLoss, ObjectLossResult, PointCosineLoss, PointLossResult, iBOTPatchLoss,
)
from dinovol_2.model.model import DinoVitStudentTeacher, _upgrade_weight_norm_state_dict_keys
from dinovol_2.ops.point_embeddings import gather_variable_points, sample_normalized_patch_embeddings
from .dino_data import build_dino_ibot_collate_fn
from dinovol_2.training.schedules import CosineScheduler
from dinovol_2.training.config import dino_model_config
from dinovol_2.training.checkpoint import method_state
from .base import TrainingMethod, StepOutput

def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    return float(value[0]), float(value[1])


def dino_loss_term_count(n_local_views: int, n_global_views: int = 2) -> int:
    if n_local_views < 0:
        raise ValueError(f"n_local_views must be non-negative, got {n_local_views}")
    if n_global_views <= 0:
        raise ValueError(f"n_global_views must be positive, got {n_global_views}")

    n_local_terms = n_local_views * n_global_views
    n_global_terms = (n_global_views - 1) * n_global_views
    return n_global_terms + n_local_terms


class DinoMethod(TrainingMethod):
    """DINO/iBOT, its teachers, and optional volumetric supervision."""
    name = "dino"

    def __init__(self, config: Mapping[str, Any], *, device, is_distributed=False, rank=0) -> None:
        super().__init__()
        self.config = dict(config["method"])
        runtime = config["runtime"]
        self.data_config = config["data"]
        self.compile_config = runtime.get("compile", False)
        self.device = torch.device(device)
        self.is_distributed = is_distributed
        self.rank = rank
        self.use_amp = bool(runtime.get("use_amp", self.device.type == "cuda"))
        from dinovol_2.ops.precision import resolve_amp_dtype
        self.amp_dtype = resolve_amp_dtype(runtime.get("amp_dtype", "bfloat16"))
        self.total_steps = int(runtime.get("max_iterations", runtime.get("num_iterations", 1000000)))
        self.model_config = dino_model_config(config)
        self.freeze_last_layer_steps = self._resolve_freeze_last_layer_steps()
        # object_losses: dense point head + object/ink/region losses on pseudo-object
        # points (dataset.object_points). Weights are relative to the DINO loss.
        self.object_loss_config = dict(self.config.get("object_losses") or {})
        self.do_object_losses = bool(self.object_loss_config.get("enabled", False))
        if self.do_object_losses:
            self.model_config["dense_head"] = {"enabled": True, **dict(self.object_loss_config.get("head") or {})}
        self.object_point_weight = float(self.object_loss_config.get("point_weight", 0.0))
        self.object_weight = float(self.object_loss_config.get("object_weight", 0.1))
        self.object_ink_weight = float(self.object_loss_config.get("ink_weight", 0.1))
        self.object_region_weight = float(self.object_loss_config.get("region_weight", 0.1))
        # The object-loss weights ramp up linearly over ramp_steps from the step the
        # heads were added (0 for a new run, the resume step when added to a run).
        self.object_ramp_steps = int(self.object_loss_config.get("ramp_steps", 0))
        self.object_heads_start_step = 0
        head_cfg = dict(self.object_loss_config.get("head") or {})
        out_dim = int(head_cfg.get("out_dim", 64))
        ink_dim = int(head_cfg.get("ink_dim", min(32, out_dim)))
        self.object_loss = ObjectLoss(
            object_temperature=float(self.object_loss_config.get("object_temperature", 0.15)),
            region_temperature=float(self.object_loss_config.get("region_temperature", 0.15)),
            ink_temperature=float(self.object_loss_config.get("ink_temperature", 0.15)),
            pos_margin=float(self.object_loss_config.get("pos_margin", 0.75)),
            ink_queue_size=int(self.object_loss_config.get("ink_queue_size", 512)),
            ink_dim=ink_dim,
        ).to(self.device)
        self.model = DinoVitStudentTeacher(self.model_config).to(self.device)
        if self.is_distributed:
            ddp_kwargs: dict[str, Any] = {
                "broadcast_buffers": False,
                # A local-only norm may be unused when a batch has no local crops.
                "find_unused_parameters": bool(self.model_config.get("untie_global_and_local_cls_norm", False)),
            }
            if self.device.type == "cuda":
                ddp_kwargs.update(
                    {
                        "device_ids": [self.device.index],
                        "output_device": self.device.index,
                    }
                )
            if "region_head" in self.model.student:
                # Ignoring the head also skips DDP's initialization broadcast.
                # Start its manually averaged gradients and EMA from shared weights.
                region_head = self.model.student["region_head"]
                for param in region_head.parameters():
                    dist.broadcast(param.detach(), src=0)
                self.model.teacher["region_head"].load_state_dict(region_head.state_dict())
                DDP._set_params_and_buffers_to_ignore_for_model(
                    self.model,
                    [f"student.region_head.{k}" for k, _ in self.model.student["region_head"].named_parameters()],
                )
            self.model = DDP(self.model, **ddp_kwargs)
        dino_out_dim = int(self.model_config.get("dino_out_dim", 131072))
        ibot_out_dim = int(self.model_config.get("ibot_out_dim", dino_out_dim))
        self.dino_loss = DINOLoss(dino_out_dim).to(self.device)
        # Form iBOT logits, Sinkhorn-Knopp targets, and loss gradients chunk by chunk from
        # the head bottleneck instead of materializing [n_masked, ibot_out_dim] tensors.
        self.ibot_fused_last_layer = bool(self.config.get("ibot_fused_last_layer", False))
        masked_loss_chunk_size = self.config.get("ibot_masked_loss_chunk_size")
        if masked_loss_chunk_size is not None:
            masked_loss_chunk_size = int(masked_loss_chunk_size)
        self.ibot_patch_loss = iBOTPatchLoss(
            ibot_out_dim,
            masked_loss_chunk_size=masked_loss_chunk_size,
            kernel_backend=str(self.config.get("ibot_kernel_backend", "torch")),
        ).to(self.device)
        self.koleo_loss = KoLeoLoss(
            distributed=bool(self.config.get("koleo_distributed", True)),
            loss_group_size=self.config.get("koleo_loss_group_size", 16),
        ).to(self.device)
        self.gram_config = dict(self.config.get("gram") or {})
        self.do_gram = bool(self.gram_config.get("enabled", False))
        self.gram_loss = (
            GramLoss(
                apply_norm=bool(self.gram_config.get("normalized", True)),
                remove_neg=bool(self.gram_config.get("remove_neg", False)),
                remove_only_teacher_neg=bool(self.gram_config.get("remove_only_teacher_neg", False)),
            ).to(self.device)
            if self.do_gram
            else None
        )

        self.dino_loss_weight = float(self.config.get("dino_loss_weight", 1.0))
        self.ibot_loss_weight = float(self.config.get("ibot_loss_weight", 1.0))
        self.koleo_loss_weight = float(self.config.get("koleo_loss_weight", 0.1))
        self.gram_loss_weight = float(self.gram_config.get("loss_weight", 2.0))
        self.point_config = {
            "enabled": True,
            "sampling_probability": 0.1,
            "loss_weight": 0.05,
            "different_type_margin": 0.0,
            "max_points_per_view": 64,
        }
        self.point_config.update(dict(self.config.get("point_supervision") or {}))
        self.config["point_supervision"] = dict(self.point_config)
        configured_train_dataset = self.data_config.get("dataset") or {}
        train_datasets = list(configured_train_dataset.get("datasets", []))
        for variant in configured_train_dataset.get("variants", []) or []:
            train_datasets.extend(dict(variant).get("datasets", []))
        has_point_collections = any(dataset.get("point_collections") for dataset in train_datasets)
        self.do_point_supervision = bool(self.point_config["enabled"]) and has_point_collections
        self.point_loss_weight = float(self.point_config["loss_weight"])
        self.point_loss = PointCosineLoss(
            different_type_margin=float(self.point_config["different_type_margin"])
        ).to(self.device)
        self.do_dino = self.dino_loss_weight > 0.0
        self.do_ibot = self.ibot_loss_weight > 0.0
        # ibot_targets: "same_view" reads the teacher target of a masked token at the same
        # token of the same (unmasked) view; "cross_view" reads it at the same physical
        # location in the other global view. Same-view targets can be predicted from a
        # token's position in the crop (which the network can infer from the crop borders),
        # and that position code took over low-content crops; cross-view targets carry
        # no information about the student's position in its crop.
        self.ibot_targets = str(self.config.get("ibot_targets", "same_view"))
        if self.ibot_targets not in ("same_view", "cross_view"):
            raise ValueError(f"ibot_targets must be 'same_view' or 'cross_view', got {self.ibot_targets!r}")
        self.do_koleo = self.koleo_loss_weight > 0.0 and self.do_dino
        self.centering = str(self.config.get("centering", "sinkhorn_knopp"))
        self.gram_img_level = bool(self.gram_config.get("img_level", True))
        self.gram_teacher_checkpoint = self.gram_config.get("teacher_checkpoint")
        self.gram_teacher_refresh_every = int(self.gram_config.get("teacher_refresh_every", 10000))
        self.gram_teacher_refresh_start_step = int(self.gram_config.get("teacher_refresh_start_step", 0))
        self.gram_teacher_backbone = None
        if self.do_gram:
            self.gram_teacher_backbone = deepcopy(self.model_module.teacher.backbone).to(self.device)
            self.gram_teacher_backbone.requires_grad_(False)
            if self.gram_teacher_checkpoint:
                self._load_gram_teacher_from_checkpoint(self.gram_teacher_checkpoint)
            else:
                self._refresh_gram_teacher_from_teacher()
            self.gram_teacher_backbone.eval()

        self.teacher_temp = float(self.config.get("teacher_temp", 0.07))
        self.warmup_teacher_temp = float(self.config.get("warmup_teacher_temp", 0.04))
        self.warmup_teacher_temp_steps = int(
            self.config.get("warmup_teacher_temp_steps", round(self.total_steps * 0.3))
        )
        self.momentum_teacher = float(self.config.get("momentum_teacher", 0.994))
        self.final_momentum_teacher = float(self.config.get("final_momentum_teacher", self.momentum_teacher))
        self.momentum_schedule = CosineScheduler(
            base_value=self.momentum_teacher, final_value=self.final_momentum_teacher, total_iters=self.total_steps)
        self.teacher_temp_schedule = CosineScheduler(
            base_value=self.teacher_temp, final_value=self.teacher_temp, total_iters=self.total_steps,
            warmup_iters=self.warmup_teacher_temp_steps, start_warmup_value=self.warmup_teacher_temp)
        self._configure_model_compilation()

    def _configure_model_compilation(self) -> None:
        settings = self.compile_config
        if isinstance(settings, bool):
            settings = {"enabled": settings}
        elif isinstance(settings, Mapping):
            settings = dict(settings)
        else:
            raise ValueError("compile must be a boolean or a settings mapping")
        unknown = set(settings) - {"enabled", "backend", "mode", "dynamic", "fullgraph"}
        if unknown:
            raise ValueError(f"Unknown compile settings: {sorted(unknown)}")
        enabled = settings.pop("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("compile.enabled must be a boolean")
        self.compile_enabled = enabled
        if not enabled:
            return
        options = {"backend": "inductor", "mode": "default", "dynamic": None, "fullgraph": False, **settings}
        # dynamic may also map module names (backbone, dino_head, ibot_head) to settings;
        # unlisted modules use null. Backbones see fixed global/local crop shapes, while
        # the iBOT head sees a different number of masked tokens every step.
        dynamic = options.pop("dynamic")
        module_names = ("backbone", "dino_head", "ibot_head")
        if isinstance(dynamic, Mapping):
            unknown_modules = set(dynamic) - set(module_names)
            if unknown_modules:
                raise ValueError(f"Unknown compile.dynamic modules: {sorted(unknown_modules)}")
            dynamic_by_module = {name: dynamic.get(name) for name in module_names}
        else:
            dynamic_by_module = {name: dynamic for name in module_names}
        if any(value is not None and not isinstance(value, bool) for value in dynamic_by_module.values()):
            raise ValueError("compile.dynamic must be a boolean, null, or a mapping of module names to those")
        if not isinstance(options["fullgraph"], bool):
            raise ValueError("compile.fullgraph must be a boolean")
        if not isinstance(options["backend"], str) or not isinstance(options["mode"], str):
            raise ValueError("compile.backend and compile.mode must be strings")
        if options["backend"] == "inductor":
            # Inductor's mix-order reduction (torch 2.10) fuses the mask-token gradient with
            # LayerNormNd parameter gradients in the CNN backward and then returns it as [C]
            # instead of [1, 1, C] (seen with 8+ masked 128^3 crops). Disabling it cost no
            # measurable CNN throughput.
            import torch._inductor.config as inductor_config
            inductor_config.triton.mix_order_reduction = False
        # Compile inner modules rather than DDP or the variable masked-token selection.
        # In-place compilation preserves parameter identities and state_dict keys for
        # optimizer groups, EMA updates, checkpoint loading, and downstream inference.
        # The small point heads (object losses) stay eager.
        for branch in (self.model_module.student, self.model_module.teacher):
            for name in module_names:
                branch[name].compile(**options, dynamic=dynamic_by_module[name])
        if self.gram_teacher_backbone is not None:
            self.gram_teacher_backbone.compile(**options, dynamic=dynamic_by_module["backbone"])
        self.ibot_patch_loss.compile_kernels(**options)

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def model_module(self) -> DinoVitStudentTeacher:
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _extract_backbone_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        state_dict = dict(state_dict)
        if any(key.startswith("backbone.") for key in state_dict):
            return {key.replace("backbone.", "", 1): value for key, value in state_dict.items() if key.startswith("backbone.")}
        return state_dict

    def _load_gram_teacher_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        if self.gram_teacher_backbone is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "format_version" in checkpoint:
            checkpoint = method_state(checkpoint, expected_method=self.name)
        if checkpoint.get("gram_teacher_backbone") is not None:
            state_dict = checkpoint["gram_teacher_backbone"]
        elif "teacher" in checkpoint:
            state_dict = self._extract_backbone_state_dict(checkpoint["teacher"])
        elif "model" in checkpoint:
            state_dict = self._extract_backbone_state_dict(checkpoint["model"])
        else:
            state_dict = self._extract_backbone_state_dict(checkpoint)
        self.gram_teacher_backbone.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def _refresh_gram_teacher_from_teacher(self) -> None:
        if self.gram_teacher_backbone is None:
            return
        self.gram_teacher_backbone.load_state_dict(self.model_module.teacher.backbone.state_dict(), strict=True)
        self.gram_teacher_backbone.requires_grad_(False)
        self.gram_teacher_backbone.eval()

    def _maybe_refresh_gram_teacher(self, step: int) -> None:
        if self.gram_teacher_backbone is None:
            return
        if self.gram_teacher_refresh_every <= 0:
            return
        if (step + 1) < self.gram_teacher_refresh_start_step:
            return
        if ((step + 1) - self.gram_teacher_refresh_start_step) % self.gram_teacher_refresh_every != 0:
            return
        self._refresh_gram_teacher_from_teacher()

    def _resolve_step_count(self, steps_key: str, epochs_key: str, *, default: int) -> int:
        if steps_key in self.config:
            return int(self.config[steps_key])
        if epochs_key in self.config:
            epoch_length = self.data_config.get("official_epoch_length", self.data_config.get("epoch_length"))
            if epoch_length is None:
                raise ValueError(f"{epochs_key} requires official_epoch_length or epoch_length in the config.")
            return int(self.config[epochs_key]) * int(epoch_length)
        return int(default)

    def _resolve_freeze_last_layer_steps(self) -> int:
        if "freeze_last_layer_steps" in self.config:
            return int(self.config["freeze_last_layer_steps"])
        if "freeze_last_layer_epochs" in self.config:
            return self._resolve_step_count("freeze_last_layer_steps", "freeze_last_layer_epochs", default=0)

        freeze_ratio = float(self.config.get("freeze_last_layer_ratio", 0.01))
        if self.total_steps <= 0 or freeze_ratio <= 0.0:
            return 0
        return max(1, min(self.total_steps, round(self.total_steps * freeze_ratio)))

    def _teacher_temp(self, step: int) -> float:
        return self.teacher_temp_schedule[step]

    def _center_teacher_cls(
            self,
            teacher_cls: torch.Tensor,
            teacher_temp: float,
            *,
            update_centers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.centering == "sinkhorn_knopp":
            teacher_targets = self.dino_loss.sinkhorn_knopp_teacher(teacher_cls, teacher_temp)
        else:
            teacher_targets = self.dino_loss.softmax_center_teacher(teacher_cls, teacher_temp)
            if update_centers:
                self.dino_loss.update_center(teacher_cls)
        return teacher_targets.chunk(2)

    def _center_teacher_patch(
            self,
            teacher_patch: torch.Tensor,
            teacher_temp: float,
            *,
            update_centers: bool = True,
    ) -> torch.Tensor:
        if teacher_patch.numel() == 0:
            return teacher_patch
        if self.centering == "sinkhorn_knopp":
            n_masked = torch.tensor([teacher_patch.shape[0]], device=teacher_patch.device, dtype=torch.long)
            return self.ibot_patch_loss.sinkhorn_knopp_teacher(teacher_patch, teacher_temp, n_masked)
        teacher_patch_batched = teacher_patch.unsqueeze(0)
        targets = self.ibot_patch_loss.softmax_center_teacher(teacher_patch_batched, teacher_temp).squeeze(0)
        if update_centers:
            self.ibot_patch_loss.update_center(teacher_patch_batched)
        return targets

    def _feature_map_shape(
        self,
        backbone: Any,
        input_spatial_shape: tuple[int, int, int],
        *,
        view_kind: str,
    ) -> tuple[int, int, int]:
        patch_size = tuple(int(size) for size in backbone.patch_size)
        return tuple(int(size) // int(patch) for size, patch in zip(input_spatial_shape, patch_size))

    @staticmethod
    def _patch_tokens_to_grid(
        patch_tokens: torch.Tensor,
        feature_map_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if len(feature_map_shape) != 3:
            raise ValueError(f"expected a 3D feature map shape, got {feature_map_shape}")
        batch_size, _, channels = patch_tokens.shape
        return patch_tokens.reshape(batch_size, *feature_map_shape, channels).permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _grid_to_patch_tokens(feature_grid: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = feature_grid.shape
        return feature_grid.permute(0, 2, 3, 4, 1).reshape(batch_size, depth * height * width, channels).contiguous()

    def _resize_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        source_feature_map_shape: tuple[int, int, int],
        target_feature_map_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if source_feature_map_shape == target_feature_map_shape:
            return patch_tokens
        feature_grid = self._patch_tokens_to_grid(patch_tokens, source_feature_map_shape)
        feature_grid = F.interpolate(
            feature_grid,
            size=target_feature_map_shape,
            mode="trilinear",
            align_corners=False,
        )
        return self._grid_to_patch_tokens(feature_grid)

    def _gram_teacher_patch_tokens(self, gram_teacher_crops: torch.Tensor | None) -> torch.Tensor | None:
        if self.gram_teacher_backbone is None or gram_teacher_crops is None:
            return None
        with torch.no_grad():
            outputs = self.gram_teacher_backbone(gram_teacher_crops, is_training=self.model.training, view_kind="global")
        return outputs["x_norm_patchtokens"]

    def _compute_gram_loss(
        self,
        *,
        student_patch_tokens: torch.Tensor,
        global_crops: torch.Tensor,
        gram_teacher_crops: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.do_gram:
            return global_crops.new_zeros(()), None, None
        if gram_teacher_crops is None:
            raise ValueError("Gram loss is enabled but the batch does not contain collated_gram_teacher_crops.")
        gram_teacher_patch_tokens = self._gram_teacher_patch_tokens(gram_teacher_crops)
        if gram_teacher_patch_tokens is None:
            return global_crops.new_zeros(()), None, None

        student_backbone = self.model_module.student.backbone
        student_feature_map_shape = self._feature_map_shape(
            student_backbone,
            tuple(int(dim) for dim in global_crops.shape[2:]),
            view_kind="global",
        )
        gram_teacher_feature_map_shape = self._feature_map_shape(
            self.gram_teacher_backbone,
            tuple(int(dim) for dim in gram_teacher_crops.shape[2:]),
            view_kind="global",
        )
        resized_teacher_patch_tokens = self._resize_patch_tokens(
            gram_teacher_patch_tokens,
            gram_teacher_feature_map_shape,
            student_feature_map_shape,
        )
        gram_loss = self.gram_loss(
            student_patch_tokens,
            resized_teacher_patch_tokens,
            img_level=self.gram_img_level,
        )
        return gram_loss, gram_teacher_patch_tokens, resized_teacher_patch_tokens

    def _compute_point_loss(
        self,
        *,
        student_patch_tokens: torch.Tensor,
        batch: Mapping[str, Any],
        global_crops: torch.Tensor,
    ) -> tuple[PointLossResult, torch.Tensor]:
        zero = global_crops.new_zeros(())
        coordinates = batch.get("collated_point_coordinates")
        type_ids = batch.get("collated_point_type_ids")
        row_indices = batch.get("collated_point_rows")
        if not self.do_point_supervision or coordinates is None or type_ids is None or row_indices is None:
            return PointLossResult(zero, zero, zero, 0, 0, 0), student_patch_tokens.new_empty(
                (0, student_patch_tokens.shape[-1])
            )

        coordinates = coordinates.to(self.device, non_blocking=True)
        type_ids = type_ids.to(self.device, non_blocking=True)
        row_indices = row_indices.to(self.device, non_blocking=True)
        backbone = self.model_module.student.backbone
        feature_map_shape = self._feature_map_shape(
            backbone,
            tuple(int(dim) for dim in global_crops.shape[2:]),
            view_kind="global",
        )
        embeddings = sample_normalized_patch_embeddings(
            student_patch_tokens,
            row_indices,
            coordinates,
            tuple(int(value) for value in backbone.patch_size),
            feature_map_shape,
        )
        embeddings, type_ids = gather_variable_points(embeddings, type_ids)
        return self.point_loss(embeddings, type_ids), embeddings

    def _forward_model(
        self,
        *,
        global_crops: torch.Tensor,
        local_crops: torch.Tensor | None,
        masks: torch.Tensor,
        mask_indices: torch.Tensor,
        n_masked: int,
        return_teacher: bool,
        masked_patch_features: bool = False,
        point_coordinates: torch.Tensor | None = None,
        teacher_target_coordinates: torch.Tensor | None = None,
        teacher_target_rows: torch.Tensor | None = None,
    ) -> Mapping[str, Any]:
        # Backward runs outside autocast. AOTAutograd must use that same
        # assumption when it captures backward graphs during these forwards.
        context = functorch_config.patch(backward_pass_autocast="off") if self.compile_enabled else nullcontext()
        with context:
            return self.model(
                student_input=global_crops,
                teacher_input=global_crops if return_teacher else None,
                student_masks=masks,
                local_student_input=local_crops,
                mask_indices_list=mask_indices,
                n_masked_patches=n_masked,
                return_teacher=return_teacher,
                masked_patch_features=masked_patch_features,
                point_coordinates=point_coordinates,
                teacher_target_coordinates=teacher_target_coordinates,
                teacher_target_rows=teacher_target_rows,
            )

    def _cross_view_ibot_selection(
        self,
        batch: Mapping[str, Any],
        global_crops: torch.Tensor,
        mask_indices: torch.Tensor,
        n_global_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-view iBOT: masked tokens whose location lies in the other global view.

        Returns the kept mask indices, their weights (1 / kept count of their view, as
        the collate does for all masked tokens), the per-row target coordinates and rows
        for the teacher, and the kept fraction of masked tokens.
        """
        if n_global_views != 2:
            raise ValueError("cross-view iBOT targets need exactly 2 global views")
        coordinates = batch["global_token_targets"].to(self.device, non_blocking=True)
        n_rows, n_tokens = coordinates.shape[:2]
        size = coordinates.new_tensor(tuple(float(dim) for dim in global_crops.shape[2:]))
        inside = torch.all((coordinates >= -0.5) & (coordinates <= size - 0.5), dim=-1).flatten()
        keep = inside[mask_indices]
        kept = mask_indices[keep]
        kept_rows = torch.div(kept, n_tokens, rounding_mode="floor")
        counts = torch.bincount(kept_rows, minlength=n_rows).clamp_min(1).to(torch.float32)
        weights = (1.0 / counts)[kept_rows]
        batch_size = n_rows // n_global_views
        rows = (torch.arange(n_rows, device=self.device) + batch_size) % n_rows
        fraction = keep.float().mean() if keep.numel() else coordinates.new_zeros(())
        return kept, weights, coordinates, rows, fraction

    def _logit_dtype(self) -> torch.dtype:
        if torch.is_autocast_enabled(self.device.type):
            return torch.get_autocast_dtype(self.device.type)
        return torch.float32

    def _object_loss_ramp(self, step: int) -> float:
        if self.object_ramp_steps <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (step - self.object_heads_start_step + 1) / self.object_ramp_steps)))

    def _object_point_validity(
        self, batch: Mapping[str, Any], masks: torch.Tensor, global_crops: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Point coordinates, teacher validity (inside the view) and student validity
        (inside the view and not in an iBOT-masked token, whose voxels are hidden)."""
        coordinates = batch["object_point_coordinates"].to(self.device, non_blocking=True)
        teacher_valid = batch["object_point_valid"].to(self.device, non_blocking=True)
        patch = coordinates.new_tensor(tuple(float(size) for size in self.model_module.student.backbone.patch_size))
        token_shape = [int(size) // int(p) for size, p in zip(global_crops.shape[2:], patch.tolist())]
        token = (coordinates / patch).floor().long()
        for axis, size in enumerate(token_shape):
            token[..., axis] = token[..., axis].clamp(0, size - 1)
        flat = (token[..., 0] * token_shape[1] + token[..., 1]) * token_shape[2] + token[..., 2]
        student_valid = teacher_valid & ~torch.gather(masks, 1, flat)
        return coordinates, teacher_valid, student_valid

    _TRAIN_METRICS = (
        "loss", "dino_global_loss", "dino_local_loss", "ibot_loss", "koleo_loss", "gram_loss",
        "point_loss", "point_same_type_loss", "point_different_type_loss",
        "point_count", "point_same_pair_count", "point_different_pair_count",
        "object_point_loss", "object_loss", "object_region_loss", "object_ink_loss",
        "object_point_anchors", "object_anchors", "object_region_anchors", "object_ink_anchors",
        "ibot_target_fraction",
    )

    def forward(self, batch: Mapping[str, Any], step: int) -> StepOutput:
        return self._compute_objective(batch, step, training=True)

    def _compute_objective(self, batch: Mapping[str, Any], step: int, *, training: bool) -> StepOutput:
        teacher_temp = self._teacher_temp(step)
        object_ramp = self._object_loss_ramp(step)
        if int(batch["n_global_views"]) != 2:
            raise ValueError("DINO requires exactly two global views")
        global_crops = batch["collated_global_crops"].to(self.device, non_blocking=True)
        local_crops = batch["collated_local_crops"].to(self.device, non_blocking=True)
        gram_teacher_crops = batch.get("collated_gram_teacher_crops")
        if gram_teacher_crops is not None:
            gram_teacher_crops = gram_teacher_crops.to(self.device, non_blocking=True)
        masks = batch["collated_masks"].to(self.device, non_blocking=True)
        mask_indices = batch["mask_indices_list"].to(self.device, non_blocking=True)
        masks_weight = batch["masks_weight"].to(self.device, non_blocking=True)
        n_global_views = int(batch["n_global_views"])
        n_local_views = int(batch["n_local_views"])
        n_masked = int(batch["n_masked_patches"].item())
        use_objects = training and self.do_object_losses and "object_point_coordinates" in batch
        point_coordinates = teacher_point_valid = student_point_valid = None
        if use_objects:
            point_coordinates, teacher_point_valid, student_point_valid = self._object_point_validity(
                batch, masks, global_crops
            )

        teacher_target_coordinates = teacher_target_rows = None
        ibot_target_fraction = global_crops.new_ones(())
        if training and self.ibot_targets == "cross_view" and "global_token_targets" in batch:
            mask_indices, masks_weight, teacher_target_coordinates, teacher_target_rows, ibot_target_fraction = (
                self._cross_view_ibot_selection(batch, global_crops, mask_indices, n_global_views)
            )
            n_masked = int(mask_indices.numel())

        fused_ibot = training and self.ibot_fused_last_layer and self.do_ibot and self.centering == "sinkhorn_knopp"
        model_outputs = self._forward_model(
            global_crops=global_crops,
            local_crops=local_crops if n_local_views else None,
            masks=masks,
            mask_indices=mask_indices,
            n_masked=n_masked,
            return_teacher=self.do_dino or self.do_ibot or use_objects,
            masked_patch_features=fused_ibot,
            point_coordinates=point_coordinates,
            teacher_target_coordinates=teacher_target_coordinates,
            teacher_target_rows=teacher_target_rows,
        )
        teacher_branch = model_outputs.get("teacher")
        if self.do_dino and teacher_branch is not None:
            teacher_cls_0, teacher_cls_1 = self._center_teacher_cls(
                teacher_branch["global_cls_projections"],
                teacher_temp,
                update_centers=training,
            )
        else:
            teacher_cls_0 = teacher_cls_1 = None
        teacher_patch_targets = None
        if fused_ibot and teacher_branch is not None and n_masked > 0:
            logit_dtype = self._logit_dtype()
            teacher_patch_features = teacher_branch["global_masked_patch_features"].to(logit_dtype)
            teacher_last_layer_weight = teacher_branch["ibot_last_layer_weight"].to(logit_dtype)
            log_prototype_scales = self.ibot_patch_loss.sinkhorn_knopp_prototype_scales(
                teacher_patch_features,
                teacher_last_layer_weight,
                teacher_temp,
                torch.tensor([n_masked], device=self.device, dtype=torch.long),
            )
        elif self.do_ibot and teacher_branch is not None:
            # Release the half-precision teacher logits once their targets exist.
            teacher_patch_targets = self._center_teacher_patch(
                teacher_branch.pop("global_masked_patch_projections"),
                teacher_temp,
                update_centers=training,
            )

        student_branch = model_outputs["student"]
        student_global = student_branch["global"]
        student_global_cls = student_branch["global_cls_projections"]
        student_patch = student_branch.get("global_masked_patch_projections")
        global_cls_0, global_cls_1 = student_global_cls.chunk(n_global_views)

        total_terms = dino_loss_term_count(n_local_views, n_global_views=n_global_views)
        if self.do_dino:
            dino_global_loss = (
                self.dino_loss([global_cls_0], [teacher_cls_1]) +
                self.dino_loss([global_cls_1], [teacher_cls_0])
            ) / total_terms

            if n_local_views:
                student_local = student_branch["local"]
                local_cls_chunks = list(student_local["cls_projections"].chunk(n_local_views))
                dino_local_loss = self.dino_loss(local_cls_chunks, [teacher_cls_0, teacher_cls_1]) / total_terms
            else:
                dino_local_loss = global_crops.new_zeros(())
        else:
            dino_global_loss = global_crops.new_zeros(())
            dino_local_loss = global_crops.new_zeros(())

        if fused_ibot and n_masked > 0:
            ibot_loss = self.ibot_patch_loss.forward_masked_from_features(
                student_branch["global_masked_patch_features"].to(logit_dtype),
                student_branch["ibot_last_layer_weight"].to(logit_dtype),
                teacher_patch_features,
                teacher_last_layer_weight,
                log_prototype_scales,
                teacher_temp,
                student_masks_flat=masks,
                masks_weight=masks_weight[:n_masked],
            )
        elif self.do_ibot and n_masked > 0:
            ibot_loss = self.ibot_patch_loss.forward_masked(
                student_patch,
                teacher_patch_targets,
                student_masks_flat=masks,
                n_masked_patches=n_masked,
                masks_weight=masks_weight,
            )
        else:
            ibot_loss = global_crops.new_zeros(())

        if self.do_koleo:
            koleo_loss = sum(self.koleo_loss(chunk) for chunk in student_global["cls_tokens"].chunk(n_global_views))
        else:
            koleo_loss = global_crops.new_zeros(())
        gram_loss, _, _ = self._compute_gram_loss(
            student_patch_tokens=student_global["patch_tokens"],
            global_crops=global_crops,
            gram_teacher_crops=gram_teacher_crops,
        )
        point_result, _ = self._compute_point_loss(
            student_patch_tokens=student_global["patch_tokens"],
            batch=batch,
            global_crops=global_crops,
        )
        zero = global_crops.new_zeros(())
        if use_objects:
            student_points = student_branch["object_points"]
            teacher_points = teacher_branch["object_points"]
            ink_tensor = (
                batch["object_point_ink"].to(self.device, non_blocking=True)
                if "object_point_ink" in batch
                else None
            )
            object_result = self.object_loss(
                student_dense=student_points["dense"],
                teacher_dense=teacher_points["dense"],
                student_tokens=student_points["tokens"],
                teacher_tokens=teacher_points["tokens"],
                student_valid=student_point_valid,
                teacher_valid=teacher_point_valid,
                objects=batch["object_point_objects"].to(self.device, non_blocking=True),
                negatives=batch["object_negatives"].to(self.device, non_blocking=True),
                student_region_head=self.model_module.student.region_head,
                teacher_region_head=self.model_module.teacher.region_head,
                student_ink=student_points.get("ink"),
                teacher_ink=teacher_points.get("ink"),
                ink=ink_tensor,
                n_views=n_global_views,
            )
        else:
            object_result = ObjectLossResult(zero, zero, zero, zero, zero, zero, zero, zero)

        loss = (
            self.dino_loss_weight * (dino_global_loss + dino_local_loss) +
            self.ibot_loss_weight * ibot_loss +
            self.koleo_loss_weight * koleo_loss +
            self.gram_loss_weight * gram_loss +
            self.point_loss_weight * point_result.loss +
            object_ramp * (
                self.object_point_weight * object_result.point +
                self.object_weight * object_result.object +
                self.object_region_weight * object_result.region +
                self.object_ink_weight * object_result.ink
            )
        )

        values = torch.stack([
            torch.as_tensor(value, device=self.device).detach().float().reshape(())
            for value in (
                loss, dino_global_loss, dino_local_loss, ibot_loss, koleo_loss, gram_loss,
                point_result.loss, point_result.same_type, point_result.different_type,
                point_result.point_count, point_result.same_pair_count, point_result.different_pair_count,
                object_result.point, object_result.object, object_result.region, object_result.ink,
                object_result.point_anchors, object_result.object_anchors, object_result.region_anchors,
                object_result.ink_anchors,
                ibot_target_fraction,
            )
        ])
        return StepOutput(loss, dict(zip(self._TRAIN_METRICS, values.unbind())))

    def train(self, mode=True):
        super().train(mode)
        if self.gram_teacher_backbone is not None:
            self.gram_teacher_backbone.eval()
        return self

    def parameter_groups(self, **kwargs):
        return self.model_module.get_params_groups(**kwargs)

    def gradient_parameters(self):
        return self.model_module.student.parameters()

    def before_optimizer_step(self, step: int) -> None:
        if self.is_distributed and "region_head" in self.model_module.student:
            for param in self.model_module.student["region_head"].parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    def after_optimizer_step(self, step: int) -> None:
        self.model_module.update_teacher(self.momentum_schedule[step])
        self._maybe_refresh_gram_teacher(step)

    def adjust_optimizer_groups(self, groups, step: int) -> None:
        if step < self.freeze_last_layer_steps:
            for group in groups:
                if group.get("is_last_layer", False):
                    group["lr"] = 0.0

    def step_metrics(self, step: int) -> dict[str, float]:
        return {"gram_loss_weight": float(self.gram_loss_weight),
                "point_loss_weight": float(self.point_loss_weight),
                "teacher_temp": self._teacher_temp(step),
                "object_loss_ramp": self._object_loss_ramp(step)}

    def evaluation_backbone(self):
        return self.model_module.teacher.backbone

    def build_collate(self, dataset):
        return build_dino_ibot_collate_fn({
            "global_crop_size": dataset.global_crop_size,
            "patch_size": self.model_config["patch_size"],
            "mask_ratio_min_max": _as_float_pair(self.config.get("mask_ratio_min_max"), (0.1, 0.5)),
            "mask_sample_probability": float(self.config.get("mask_sample_probability", 0.5)),
            "dtype": torch.float32,
        })

    def save_monitor_image(self, batch, step, metrics, *, directory, config):
        from dinovol_2.training.monitoring import DinoMonitor
        return DinoMonitor(self, directory, config).save_monitor_image(batch, step, metrics)

    def dataset_config(
        self,
        dataset_config: Mapping[str, Any],
        *,
        training: bool = False,
    ) -> dict[str, Any]:
        resolved_config = dict(dataset_config)
        if int(resolved_config.get("num_global_crops", 2)) != 2:
            raise ValueError("DINO requires exactly two global views")
        if self.do_gram:
            resolved_config.setdefault(
                "gram_teacher_crop_size",
                resolved_config.get("global_crop_size", resolved_config.get("crop_size")),
            )
            resolved_config.setdefault("gram_teacher_no_augmentations", True)
        if training and self.do_point_supervision:
            resolved_config["point_supervision"] = dict(self.point_config)
        else:
            resolved_config.pop("point_supervision", None)
        # Object points are only sampled for the training loader of an object-loss run.
        if not (training and self.do_object_losses):
            resolved_config.pop("object_points", None)
        # Token correspondences between the global views, for cross-view iBOT targets.
        if training and self.ibot_targets == "cross_view" and self.do_ibot:
            resolved_config["cross_view_token_patch_size"] = list(self.model_config.get("patch_size", (16, 16, 16)))
        else:
            resolved_config.pop("cross_view_token_patch_size", None)
        return resolved_config

    def checkpoint_state(self):
        # Centering delays its reduction until the next forward. Finish it here so
        # saving/resuming preserves the next targets without serializing work handles.
        self.dino_loss.apply_center_update()
        self.ibot_patch_loss.apply_center_update()
        return {
            "student": self.model_module.student.state_dict(),
            "teacher": self.model_module.teacher.state_dict(),
            "dino_loss": self.dino_loss.state_dict(),
            "ibot_patch_loss": self.ibot_patch_loss.state_dict(),
            "object_loss": self.object_loss.state_dict() if self.do_object_losses else None,
            "gram_teacher_backbone": self.gram_teacher_backbone.state_dict() if self.gram_teacher_backbone is not None else None,
            "object_heads_start_step": self.object_heads_start_step,

        }

    def rank_checkpoint_state(self):
        return {
            name: getattr(self.object_loss, name).detach().cpu()
            for name in ("ink_pos_queue", "ink_neg_queue")
        } if self.do_object_losses else {}

    def load_rank_checkpoint_state(self, state):
        if not self.do_object_losses:
            return
        for name in ("ink_pos_queue", "ink_neg_queue"):
            queue = state.get(name)
            if queue is None:
                queue = torch.empty((0, self.object_loss.ink_dim))
            if queue.ndim != 2 or queue.shape[1] != self.object_loss.ink_dim:
                raise ValueError(f"Checkpoint {name} does not match the ink head dimension")
            setattr(self.object_loss, name, queue.to(self.device).clone())

    _ADDABLE_MODULES = ("dense_head.", "region_head.")
    _GROUP_KEYS = ("lr_multiplier", "wd_multiplier", "is_last_layer")

    def _load_branch(self, branch: torch.nn.Module, state_dict: Mapping[str, Any]) -> list[str]:
        """Load one branch; returns the keys the checkpoint lacked.

        Besides added heads, a checkpoint written with axial RoPE may be resumed with
        mixed RoPE: the per-block frequencies start at the axial layout (identical
        outputs) and the axial-only `rope_embed.periods` buffer is dropped.
        If the checkpoint contains addable-module parameters with an older layout or
        different shapes, those addable modules are re-initialized from scratch.
        """
        state_dict = _upgrade_weight_norm_state_dict_keys(state_dict)
        backbone = branch.backbone
        if backbone.rope_embed is None:
            state_dict = {key: value for key, value in state_dict.items() if not key.startswith("backbone.rope_embed.")}
        branch_state = branch.state_dict()
        if any(key.startswith(self._ADDABLE_MODULES) for key in state_dict):
            branch_addable = {
                key: val.shape for key, val in branch_state.items() if key.startswith(self._ADDABLE_MODULES)
            }
            ckpt_addable = {
                key: val.shape for key, val in state_dict.items() if key.startswith(self._ADDABLE_MODULES)
            }
            if branch_addable != ckpt_addable:
                state_dict = {
                    key: val for key, val in state_dict.items() if not key.startswith(self._ADDABLE_MODULES)
                }
        missing, unexpected = branch.load_state_dict(state_dict, strict=False)
        backbone_missing = [key.removeprefix("backbone.") for key in missing if key.startswith("backbone.")]
        allowed = {f"backbone.{key}" for key in backbone.warm_start_missing_rope(backbone_missing)}
        required = [key for key in missing if not key.startswith(self._ADDABLE_MODULES) and key not in allowed]
        if required or unexpected:
            raise RuntimeError(f"Checkpoint does not match the model: missing={required} unexpected={list(unexpected)}")
        return list(missing)

    def load_optimizer_state(self, optimizer, state: Mapping[str, Any], added: Sequence[str] = ()) -> None:
        current = optimizer.state_dict()
        old_groups, new_groups = state["param_groups"], current["param_groups"]
        if not added:
            optimizer.load_state_dict(state)
            return
        # Parameters added since the checkpoint (heads, mixed-RoPE frequencies) may sit
        # anywhere inside a fused group, so old ids are matched to parameters by name:
        # within a group, the checkpoint's parameters are the current ones minus the added.
        names = {id(parameter): name for name, parameter in self.model_module.student.named_parameters()}
        added_set = set(added)
        has_replaced_addable = any(name.startswith(self._ADDABLE_MODULES) for name in added_set)
        group_key = lambda group: tuple(group.get(key) for key in self._GROUP_KEYS)
        old_by_key = {group_key(group): group for group in old_groups}
        if len(old_by_key) != len(old_groups):
            raise RuntimeError("Checkpoint optimizer groups are not distinguishable by their multipliers.")
        remapped, matched = {}, set()
        for live_group, new_group in zip(optimizer.param_groups, new_groups):
            kept = [new_id for parameter, new_id in zip(live_group["params"], new_group["params"])
                    if names[id(parameter)] not in added_set]
            old = old_by_key.get(group_key(new_group))
            if old is None:
                if kept:
                    raise RuntimeError("Checkpoint optimizer state lacks a group of existing parameters.")
                continue
            if len(kept) != len(old["params"]):
                if not (has_replaced_addable and len(kept) <= len(old["params"])):
                    raise RuntimeError("Checkpoint optimizer state does not match the model's parameter groups.")
            matched.add(group_key(new_group))
            remapped.update(
                {
                    new_id: state["state"][old_id]
                    for old_id, new_id in zip(old["params"][: len(kept)], kept)
                    if old_id in state["state"]
                }
            )
        if matched != set(old_by_key):
            raise RuntimeError("Checkpoint optimizer state has groups the model lacks.")
        optimizer.load_state_dict({"state": remapped, "param_groups": new_groups})
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                moments = optimizer.state.get(parameter, {}).get("exp_avg")
                if moments is not None and moments.shape != parameter.shape:
                    raise RuntimeError("Remapped optimizer state does not match its parameter's shape.")

    def load_checkpoint_state(self, checkpoint, step):
        for loss in (self.dino_loss, self.ibot_patch_loss):
            loss.apply_center_update()
            loss.updated = True
            loss.reduce_handle = None
            loss.async_batch_center = None
        student = self.model_module.student
        teacher = self.model_module.teacher
        added = self._load_branch(student, checkpoint["student"])
        teacher_added = self._load_branch(teacher, checkpoint["teacher"])
        # Added heads start from the student's initialization in both branches (added
        # RoPE frequencies were already warm-started identically in each branch).
        for name in {key.split(".", 1)[0] for key in teacher_added if key.startswith(self._ADDABLE_MODULES)}:
            getattr(teacher, name).load_state_dict(getattr(student, name).state_dict())
        added_rope = sorted({key for key in added if key.endswith("rope_embed.mix_frequencies")})
        if added_rope and self.is_main_process:
            print(f"Warm-started {len(added_rope)} mixed-RoPE frequency tensors from the axial layout")
        if any(key.startswith(self._ADDABLE_MODULES) for key in added):
            # Heads added to an existing run: their loss ramp starts at the next step.
            self.object_heads_start_step = step + 1
            if self.is_main_process:
                print(
                    f"Initialized modules absent from the checkpoint: "
                    f"{sorted({key.split('.', 1)[0] for key in added if key.startswith(self._ADDABLE_MODULES)})} "
                    f"(start step {self.object_heads_start_step})"
                )
        else:
            self.object_heads_start_step = int(checkpoint.get("object_heads_start_step", 0))
        if "dino_loss" in checkpoint:
            self.dino_loss.load_state_dict(checkpoint["dino_loss"])
        if "ibot_patch_loss" in checkpoint:
            self.ibot_patch_loss.load_state_dict(checkpoint["ibot_patch_loss"])
        if self.do_object_losses and checkpoint.get("object_loss") is not None:
            self.object_loss.load_state_dict(checkpoint["object_loss"], strict=False)
        if self.gram_teacher_backbone is not None:
            if checkpoint.get("gram_teacher_backbone") is not None:
                self.gram_teacher_backbone.load_state_dict(checkpoint["gram_teacher_backbone"])
                self.gram_teacher_backbone.requires_grad_(False)
                self.gram_teacher_backbone.eval()
            else:
                # Older or base-pretraining checkpoints do not carry a
                # dedicated Gram teacher. On resume, align the frozen Gram
                # teacher with the just-restored EMA teacher state instead of
                # keeping any stale initialization from __init__.
                self._refresh_gram_teacher_from_teacher()
        return added

    def evaluate(self, batch: Mapping[str, Any], step: int) -> dict[str, float]:
        """Retain the established validation objective, using the shared computation.

        Validation uses same-view, unfused iBOT and excludes object losses. It
        consumes pending centers but does not enqueue center or queue updates.
        These differences are explicit rather than a second loss implementation.
        """
        output = self._compute_objective(batch, step, training=False)
        names = self._TRAIN_METRICS[:12]
        values = torch.stack([output.metrics[name] for name in names]).tolist()
        return {
            **dict(zip(names, values)),
            "gram_loss_weight": float(self.gram_loss_weight),
            "point_loss_weight": float(self.point_loss_weight),
            "teacher_temp": self._teacher_temp(step),
        }
