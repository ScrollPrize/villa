from pathlib import Path
from copy import deepcopy
import os
from time import perf_counter
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from vesuvius.models.training.lr_schedulers import get_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from vesuvius.models.utils import InitWeights_He
from vesuvius.models.datasets import ZarrDataset
from vesuvius.utils.plotting import save_debug, convert_slice_to_bgr, _compute_display_value_range, add_text_label
from vesuvius.models.build.build_network_from_config import NetworkFromConfig

from vesuvius.models.training.loss.losses import _create_loss
from vesuvius.models.training.loss.nnunet_losses import DeepSupervisionWrapper
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.save_checkpoint import (
    save_checkpoint,
    manage_checkpoint_history,
    manage_debug_gifs,
    cleanup_old_configs,
    save_final_checkpoint
)

from vesuvius.models.utilities.load_checkpoint import load_checkpoint
from vesuvius.models.utilities.get_accelerator import get_accelerator
from vesuvius.models.utilities.s3_utils import detect_s3_paths, setup_multiprocessing_for_s3
from vesuvius.models.training.wandb_logging import save_train_val_filenames
from vesuvius.models.evaluation.connected_components import ConnectedComponentsMetric
from vesuvius.models.evaluation.critical_components import CriticalComponentsMetric
from vesuvius.models.evaluation.iou_dice import IOUDiceMetric
from vesuvius.models.evaluation.voi import VOIMetric
from vesuvius.models.augmentation.transforms.cucim_dilate import dilate_label_batch_with_cucim
from contextlib import nullcontext
from collections import deque, defaultdict



class BaseTrainer:
    def __init__(self,
                 mgr=None,
                 verbose: bool = True):
        """
        Initialize the trainer with a config manager instance

        Parameters
        ----------
        mgr : ConfigManager, optional
            If provided, use this config manager instance instead of creating a new one
        verbose : bool
            Whether to print verbose output
        """
        if mgr is not None:
            self.mgr = mgr
        else:
            from vesuvius.models.configuration.config_manager import ConfigManager
            self.mgr = ConfigManager(verbose)

        # --- DDP and GPU selection setup --- #
        self.is_distributed = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1

        # Parse requested GPU IDs from config (from --gpus)
        gpu_ids = getattr(self.mgr, 'gpu_ids', None)
        if isinstance(gpu_ids, str):
            gpu_ids = [int(x) for x in gpu_ids.split(',') if x.strip() != '']
        self.gpu_ids = gpu_ids if gpu_ids else None

        # Determine if DDP is requested by config or env (torchrun)
        env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
        want_ddp = bool(getattr(self.mgr, 'use_ddp', False)) or env_world_size > 1

        # Set device early (before init) if CUDA available
        if torch.cuda.is_available():
            # Determine local rank from env (torchrun) or default 0
            env_local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
            self.local_rank = env_local_rank
            if want_ddp and self.gpu_ids:
                # Map this process to the user-specified GPU list
                if len(self.gpu_ids) < env_world_size:
                    raise ValueError(
                        f"--gpus specifies {len(self.gpu_ids)} devices, but WORLD_SIZE={env_world_size}. "
                        f"Launch with torchrun --nproc_per_node={len(self.gpu_ids)} or adjust --gpus."
                    )
                assigned_gpu = int(self.gpu_ids[env_local_rank])
            elif want_ddp:
                assigned_gpu = env_local_rank
            elif self.gpu_ids:
                assigned_gpu = int(self.gpu_ids[0])
            else:
                assigned_gpu = 0

            torch.cuda.set_device(assigned_gpu)
            self.device = torch.device('cuda', assigned_gpu)
            self.assigned_gpu_id = assigned_gpu
        else:
            self.device = get_accelerator()
            self.assigned_gpu_id = None

        # Initialize process group if needed
        if want_ddp and dist.is_available():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            if not dist.is_initialized():
                dist.init_process_group(backend=backend, init_method='env://')
            self.is_distributed = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            # Validate GPU mapping length matches world size when provided
            if torch.cuda.is_available() and self.gpu_ids and len(self.gpu_ids) != self.world_size:
                raise ValueError(
                    f"In DDP, number of GPUs in --gpus ({len(self.gpu_ids)}) must equal WORLD_SIZE ({self.world_size})."
                )

        # Friendly prints
        if self.is_distributed and (not self.rank or self.rank == 0):
            if torch.cuda.is_available():
                used = self.gpu_ids if self.gpu_ids else list(range(self.world_size))
                print(f"DDP enabled (world size={self.world_size}). Using GPUs: {used}")
            else:
                print(f"DDP enabled on CPU/MPS (world size={self.world_size})")
        elif not self.is_distributed and torch.cuda.is_available() and self.gpu_ids:
            if len(self.gpu_ids) > 1:
                print(f"Multiple GPUs specified {self.gpu_ids} without DDP; using GPU {self.gpu_ids[0]} only.")
            else:
                print(f"Using GPU {self.gpu_ids[0]}")

        # Default AMP dtype; resolved during training initialization
        self.amp_dtype = torch.float16
        self.amp_dtype_str = 'float16'
        self._profile_augmentations = bool(getattr(self.mgr, 'profile_augmentations', False))
        self._augmentation_names = None
        self._epoch_aug_time = None
        self._epoch_aug_count = None
        self._on_device_transforms = None
        ema_cfg = getattr(self.mgr, 'ema_config', {}) or {}
        self.ema_enabled = bool(getattr(self.mgr, 'ema_enabled', ema_cfg.get('enabled', False)))
        self.ema_decay = float(getattr(self.mgr, 'ema_decay', ema_cfg.get('decay', 0.999)))
        self.ema_start_step = int(getattr(self.mgr, 'ema_start_step', ema_cfg.get('start_step', 0)))
        self.ema_update_every_steps = max(
            1,
            int(getattr(self.mgr, 'ema_update_every_steps', ema_cfg.get('update_every_steps', 1)))
        )
        self.ema_validate = bool(
            getattr(self.mgr, 'ema_validate', ema_cfg.get('validate', self.ema_enabled))
        )
        self.ema_save_in_checkpoint = bool(
            getattr(
                self.mgr,
                'ema_save_in_checkpoint',
                ema_cfg.get('save_in_checkpoint', self.ema_enabled),
            )
        )
        self.ema_model = None
        self._ema_optimizer_step = 0
        self._checkpoint_ema_state = None
        self._checkpoint_ema_optimizer_step = None
        self._printed_ema_validation_mode = False
        self._volume_dilate_by_name = {}
        self.guide_loss_weight = float(getattr(self.mgr, "guide_loss_weight", 0.0))
        self.guide_supervision_target = getattr(self.mgr, "guide_supervision_target", None)
        model_config = getattr(self.mgr, "model_config", {}) or {}
        self.guide_fusion_stage = str(model_config.get("guide_fusion_stage", "input")).strip().lower()
        if self.guide_fusion_stage in {"feature_encoder", "feature_skip_concat", "direct_segmentation"} and self.guide_loss_weight > 0.0:
            raise ValueError(
                "guide_loss_weight must be 0.0 when model_config.guide_fusion_stage is "
                "'feature_encoder', 'feature_skip_concat', or 'direct_segmentation'"
            )
        self._current_aux_outputs = {}
        self.compile_policy = str(getattr(self.mgr, "compile_policy", "auto")).strip().lower()
        self.startup_timing = bool(getattr(self.mgr, "startup_timing", False))
        self._startup_timing_records = []
        self._startup_timing_logged = False
        self._first_optimizer_timing_recorded = False

    # --- build model --- #
    def _build_model(self):
        if not hasattr(self.mgr, 'model_config') or self.mgr.model_config is None:
            print("Initializing model_config with defaults")
            self.mgr.model_config = {
                "train_patch_size": self.mgr.train_patch_size,
                "in_channels": self.mgr.in_channels,
                "model_name": self.mgr.model_name,
                "autoconfigure": self.mgr.autoconfigure,
                "conv_op": "nn.Conv2d" if len(self.mgr.train_patch_size) == 2 else "nn.Conv3d"
            }

        model = NetworkFromConfig(self.mgr)
        return model

    def _get_additional_checkpoint_data(self):
        """
        Return additional data to include in checkpoint saves.

        Subclasses can override this to save extra state (e.g., EMA model).
        Returns a dict that will be merged into the checkpoint.
        """
        if self.ema_model is not None and self.ema_save_in_checkpoint:
            return {
                'ema_model': self.ema_model.state_dict(),
                'ema_optimizer_step': int(self._ema_optimizer_step),
            }
        return {}

    def _unwrap_model(self, model):
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, '_orig_mod'):
            try:
                model = model._orig_mod
            except Exception:
                pass
        return model

    def _wrap_model_for_distributed_training(self, model):
        if not self.is_distributed:
            return model

        raw_model = self._unwrap_model(model)
        guide_enabled = bool(getattr(raw_model, "guide_enabled", False))

        find_unused_cfg = getattr(self.mgr, "ddp_find_unused_parameters", "auto")
        if isinstance(find_unused_cfg, str):
            find_unused_norm = find_unused_cfg.strip().lower()
            if find_unused_norm == "auto":
                find_unused_parameters = not guide_enabled
            else:
                find_unused_parameters = find_unused_norm == "true"
        else:
            find_unused_parameters = bool(find_unused_cfg)

        static_graph_cfg = getattr(self.mgr, "ddp_static_graph", "auto")
        if isinstance(static_graph_cfg, str):
            static_graph_norm = static_graph_cfg.strip().lower()
            if static_graph_norm == "auto":
                static_graph = guide_enabled
            else:
                static_graph = static_graph_norm == "true"
        else:
            static_graph = bool(static_graph_cfg)

        ddp_kwargs = {
            "find_unused_parameters": find_unused_parameters,
            "static_graph": static_graph,
            "gradient_as_bucket_view": bool(getattr(self.mgr, "ddp_gradient_as_bucket_view", False)),
        }
        if self.device.type == 'cuda':
            ddp_kwargs.update(
                device_ids=[self.assigned_gpu_id],
                output_device=self.assigned_gpu_id,
            )
        if not self.is_distributed or self.rank == 0:
            print(
                "DDP kwargs: "
                f"find_unused_parameters={ddp_kwargs['find_unused_parameters']}, "
                f"static_graph={ddp_kwargs['static_graph']}, "
                f"gradient_as_bucket_view={ddp_kwargs['gradient_as_bucket_view']}"
            )
        return DDP(model, **ddp_kwargs)

    def _record_startup_timing(self, label, duration_seconds):
        if not self.startup_timing:
            return
        label = str(label)
        duration_seconds = float(duration_seconds)
        self._startup_timing_records.append((label, duration_seconds))
        if not self.is_distributed or self.rank == 0:
            print(f"[Timing] {label}: {duration_seconds:.3f}s")

    def _flush_startup_timing(self, prefix="startup"):
        if not self.startup_timing or self._startup_timing_logged:
            return
        self._startup_timing_logged = True

    def _resolve_compile_policy(self, model):
        policy = self.compile_policy
        if policy == "auto":
            raw_model = self._unwrap_model(model)
            return "module" if getattr(raw_model, "guide_enabled", False) else "ddp_wrapper"
        return policy

    def _compile_module_in_place(self, model):
        compile_method = getattr(model, "compile", None)
        if callable(compile_method):
            compile_method()
            return model
        return torch.compile(model)

    def _maybe_compile_model(self, model):
        if self.device.type != "cuda":
            return model

        compile_policy = self._resolve_compile_policy(model)
        if compile_policy == "off":
            if not self.is_distributed or self.rank == 0:
                print("Compile policy set to 'off'; using eager mode.")
            return model

        compile_start = perf_counter()
        if not self.is_distributed or self.rank == 0:
            print(f"Compiling model with policy '{compile_policy}'")

        try:
            if compile_policy == "module":
                model = self._compile_module_in_place(model)
            elif compile_policy == "ddp_wrapper":
                model = torch.compile(model)
            else:
                raise ValueError(f"Unsupported compile policy: {compile_policy}")
        except Exception as e:
            if not self.is_distributed or self.rank == 0:
                print(f"Compile policy '{compile_policy}' failed; continuing without compile. Reason: {e}")
            self._record_startup_timing("compile_failed", perf_counter() - compile_start)
            return model

        self._record_startup_timing("compile", perf_counter() - compile_start)
        return model

    def _create_ema_model(self, model):
        ema_model = deepcopy(self._unwrap_model(model))
        ema_model = ema_model.to(self.device)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        return ema_model

    def _initialize_ema_model(self, model):
        if not self.ema_enabled:
            self.ema_model = None
            return None

        self.ema_model = self._create_ema_model(model)
        if self._checkpoint_ema_state is not None:
            try:
                self.ema_model.load_state_dict(self._checkpoint_ema_state)
                self._ema_optimizer_step = int(self._checkpoint_ema_optimizer_step or 0)
                print(
                    "Restored EMA model from checkpoint "
                    f"(optimizer_step={self._ema_optimizer_step})"
                )
            except Exception as exc:
                print(f"Warning: Failed to restore EMA model from checkpoint: {exc}")
                print("Using freshly initialized EMA model")
                self._ema_optimizer_step = 0
            finally:
                self._checkpoint_ema_state = None
                self._checkpoint_ema_optimizer_step = None
        else:
            print(
                "Created EMA model "
                f"(decay={self.ema_decay}, start_step={self.ema_start_step}, "
                f"update_every_steps={self.ema_update_every_steps})"
            )
        return self.ema_model

    def _update_ema_model(self, model):
        if self.ema_model is None:
            return

        self._ema_optimizer_step += 1
        if self._ema_optimizer_step < self.ema_start_step:
            return
        if ((self._ema_optimizer_step - self.ema_start_step) % self.ema_update_every_steps) != 0:
            return

        ema_state = self.ema_model.state_dict()
        for name, model_value in self._unwrap_model(model).state_dict().items():
            ema_value = ema_state[name]
            model_value = model_value.detach()
            if torch.is_floating_point(ema_value):
                ema_value.lerp_(model_value.to(dtype=ema_value.dtype), 1.0 - self.ema_decay)
            else:
                ema_value.copy_(model_value)

    def _get_validation_model(self, model):
        if self.ema_model is not None and self.ema_validate:
            if not self._printed_ema_validation_mode:
                print("Validation will use the EMA model")
                self._printed_ema_validation_mode = True
            return self.ema_model
        return model

    # --- configure dataset --- #
    def _configure_dataset(self, is_training=True):
        dataset = self._build_dataset_for_mgr(self.mgr, is_training=is_training)
        for vol in getattr(dataset, '_volumes', []):
            if vol.dilate not in (None, 0):
                self._volume_dilate_by_name[vol.volume_id] = float(vol.dilate)
                vol.dilate = None
        print(f"Using ZarrDataset ({'training' if is_training else 'validation'})")
        return dataset

    def _build_dataset_for_mgr(self, mgr, *, is_training: bool) -> ZarrDataset:
        """Build a ZarrDataset for the given config manager."""
        return ZarrDataset(mgr=mgr, is_training=is_training)

    # --- hooks for subclasses --------------------------------------------------------------- #

    def _prepare_sample(self, sample: dict, *, is_training: bool) -> dict:
        """Allow subclasses to inject additional data into a single-sample dictionary."""
        return sample

    def _prepare_batch(self, batch: dict, *, is_training: bool) -> dict:
        """Allow subclasses to inject additional data into a batch dictionary."""
        return batch

    def _reset_epoch_aug_timers(self) -> None:
        if not self._profile_augmentations:
            return
        self._epoch_aug_time = defaultdict(float)
        self._epoch_aug_count = defaultdict(int)

    def _reduce_aug_value(self, value):
        if torch.is_tensor(value):
            return float(value.sum().item()), int(value.numel())
        if isinstance(value, np.ndarray):
            return float(value.sum()), int(value.size)
        if isinstance(value, (list, tuple)):
            total = 0.0
            count = 0
            for item in value:
                t, c = self._reduce_aug_value(item)
                total += t
                count += c
            return total, count
        try:
            return float(value), 1
        except (TypeError, ValueError):
            return 0.0, 0

    def _accumulate_aug_timers(self, data_dict: dict) -> None:
        if not self._profile_augmentations:
            return
        if not data_dict:
            return
        perf = data_dict.get('_aug_perf')
        if perf is None:
            return
        if isinstance(perf, list):
            for item in perf:
                if isinstance(item, dict):
                    self._accumulate_aug_timers({'_aug_perf': item})
            return
        if not isinstance(perf, dict):
            return
        if self._epoch_aug_time is None or self._epoch_aug_count is None:
            self._reset_epoch_aug_timers()
        for name, value in perf.items():
            total, count = self._reduce_aug_value(value)
            if count == 0:
                continue
            self._epoch_aug_time[name] += total
            self._epoch_aug_count[name] += count

    def _report_epoch_aug_timers(self, epoch: int) -> None:
        if not self._profile_augmentations:
            return
        if not self._epoch_aug_time:
            return
        if self.is_distributed and self.rank != 0:
            return

        total_time = sum(self._epoch_aug_time.values())
        if total_time <= 0:
            return
        print(f"\n[Perf] Augmentation timing for epoch {epoch + 1}: total={total_time:.2f}s")
        for name, elapsed in sorted(self._epoch_aug_time.items(), key=lambda kv: kv[1], reverse=True):
            count = max(1, self._epoch_aug_count.get(name, 1))
            avg_ms = (elapsed / count) * 1000.0
            pct = (elapsed / total_time) * 100.0 if total_time > 0 else 0.0
            print(f"  {name}: {elapsed:.2f}s ({pct:.1f}%) avg={avg_ms:.2f}ms/sample")

    def _should_include_target_in_loss(self, target_name: str) -> bool:
        if target_name == 'is_unlabeled':
            return False
        if target_name.endswith('_skel'):
            return False
        if target_name.endswith('_mask') or target_name.startswith('mask_') or target_name == 'plane_mask':
            return False
        return True

    def _compute_loss_value(
        self,
        loss_fn,
        prediction,
        ground_truth,
        *,
        target_name: str,
        targets_dict: dict,
        outputs: dict,
    ):
        skeleton_data = targets_dict.get(f'{target_name}_skel')
        base_loss = getattr(loss_fn, 'loss', loss_fn)
        skeleton_losses = {'DC_SkelREC_and_CE_loss', 'SoftSkeletonRecallLoss'}

        extra_loss_kwargs = {}
        if self.guide_fusion_stage == "direct_segmentation":
            prediction, ground_truth, extra_loss_kwargs = self._adapt_direct_segmentation_loss_inputs(
                base_loss,
                prediction,
                ground_truth,
                target_name=target_name,
            )

        if skeleton_data is not None and base_loss.__class__.__name__ in skeleton_losses:
            return loss_fn(prediction, ground_truth, skeleton_data, **extra_loss_kwargs)

        return loss_fn(prediction, ground_truth, **extra_loss_kwargs)

    def _resolve_target_ignore_value(self, target_name: str):
        target_info = self.mgr.targets.get(target_name, {}) or {}
        for alias in ("ignore_index", "ignore_label", "ignore_value"):
            value = target_info.get(alias)
            if value is not None:
                return value
        return None

    @staticmethod
    def _make_ignore_mask(target: torch.Tensor, ignore_label):
        if ignore_label is None:
            return None
        if isinstance(ignore_label, float) and np.isnan(ignore_label):
            return torch.isnan(target)
        return target == float(ignore_label)

    def _adapt_direct_segmentation_loss_inputs(self, base_loss, prediction, ground_truth, *, target_name: str):
        loss_name = base_loss.__class__.__name__
        ce_style_losses = {"DC_and_CE_loss", "DC_SkelREC_and_CE_loss"}
        bce_region_losses = {"DC_and_BCE_loss", "LabelSmoothedDCAndBCELoss"}
        bce_single_channel_losses = {"BCEWithLogitsLoss", "MemoryEfficientSoftDiceLoss", "SoftDiceLoss"}
        supported_losses = ce_style_losses | bce_region_losses | bce_single_channel_losses
        if loss_name not in supported_losses:
            raise ValueError(
                f"Loss '{loss_name}' is not supported with guide_fusion_stage='direct_segmentation'."
            )

        if not torch.is_tensor(ground_truth):
            raise TypeError("direct_segmentation requires tensor targets")

        target = ground_truth
        if target.ndim == 4:
            target = target.unsqueeze(1)
        if target.ndim != 5:
            raise ValueError(
                "direct_segmentation expects 5D targets [B, C, D, H, W] or 4D label maps, "
                f"got shape {tuple(ground_truth.shape)}"
            )

        ignore_label = self._resolve_target_ignore_value(target_name)
        ignore_mask = self._make_ignore_mask(target, ignore_label)

        if loss_name in ce_style_losses:
            if target.shape[1] > 1:
                target = torch.argmax(target, dim=1, keepdim=True)
            return prediction, target, {}

        if prediction.ndim != 5 or prediction.shape[1] < 2:
            raise ValueError(
                "direct_segmentation expects synthesized 2-channel logits for BCE-style losses, "
                f"got shape {tuple(prediction.shape)}"
            )
        fg_prediction = prediction[:, 1:2]

        if target.shape[1] > 1:
            foreground = target[:, -1:].float()
            ignore_mask = None
        else:
            foreground = (target > 0).float()
            if ignore_mask is not None:
                foreground = foreground.masked_fill(ignore_mask, 0.0)

        if loss_name in bce_region_losses:
            ignore_channel = ignore_mask.float() if ignore_mask is not None else torch.zeros_like(foreground)
            region_target = torch.cat([foreground, ignore_channel], dim=1)
            return fg_prediction, region_target, {}

        if loss_name == "BCEWithLogitsLoss":
            if ignore_mask is not None:
                bce_target = target.float().clone()
                bce_target = torch.where(ignore_mask, bce_target, foreground)
            else:
                bce_target = foreground
            return fg_prediction, bce_target, {}

        extra_loss_kwargs = {}
        if ignore_mask is not None:
            extra_loss_kwargs["loss_mask"] = (~ignore_mask).float()
        return fg_prediction, foreground, extra_loss_kwargs

    def _resolve_guide_supervision_target(self, targets_dict):
        if self.guide_supervision_target is not None:
            return self.guide_supervision_target
        for target_name, target_info in self.mgr.targets.items():
            if not target_info.get("auxiliary_task", False) and target_name in targets_dict:
                return target_name
        return None

    def _compute_guide_alignment_loss(self, targets_dict):
        if self.guide_loss_weight <= 0.0:
            return None

        guide_mask = self._current_aux_outputs.get("guide_mask")
        if guide_mask is None:
            return None

        target_name = self._resolve_guide_supervision_target(targets_dict)
        if target_name is None or target_name not in targets_dict:
            return None

        guide_target = targets_dict[target_name]
        if isinstance(guide_target, (list, tuple)):
            guide_target = guide_target[0]
        if not torch.is_tensor(guide_target):
            return None

        ignore_label = None
        target_info = self.mgr.targets.get(target_name, {}) or {}
        for key in ("ignore_label", "ignore_index", "ignore_value"):
            value = target_info.get(key)
            if value is not None:
                ignore_label = value
                break

        guide_target = guide_target.float()
        if guide_target.ndim != 5:
            return None

        if guide_target.shape[1] == 1:
            ignore_mask = torch.zeros_like(guide_target, dtype=torch.bool)
            if ignore_label is not None:
                if isinstance(ignore_label, float) and np.isnan(ignore_label):
                    ignore_mask = torch.isnan(guide_target)
                else:
                    ignore_mask = guide_target == float(ignore_label)
            foreground = (guide_target > 0).float()
            if ignore_label is not None:
                foreground = foreground.masked_fill(ignore_mask, 0.0)
            valid_mask = (~ignore_mask).float()
        else:
            foreground = (guide_target > 0).any(dim=1, keepdim=True).float()
            valid_mask = torch.ones_like(foreground)

        target_resize = F.interpolate(foreground, size=guide_mask.shape[2:], mode="nearest")
        valid_resize = F.interpolate(valid_mask, size=guide_mask.shape[2:], mode="nearest")
        device_type = guide_mask.device.type
        autocast_enabled = device_type in {"cuda", "cpu"}
        with torch.amp.autocast(device_type=device_type, enabled=False) if autocast_enabled else nullcontext():
            guide_mask_fp32 = guide_mask.float().clamp(1e-6, 1.0 - 1e-6)
            target_resize_fp32 = target_resize.float()
            loss_map = F.binary_cross_entropy(guide_mask_fp32, target_resize_fp32, reduction="none")
        weighted_loss = loss_map * valid_resize
        denom = valid_resize.sum().clamp_min(1.0)
        return weighted_loss.sum() / denom

    def _resolve_guide_supervision_target_name(self) -> str:
        if self.guide_supervision_target is not None:
            return str(self.guide_supervision_target)
        for target_name, target_info in self.mgr.targets.items():
            if not target_info.get("auxiliary_task", False):
                return str(target_name)
        return "target"

    def _slice_aux_outputs_for_debug(self, aux_outputs, batch_index: int):
        if not aux_outputs:
            return {}
        sliced = {}
        for name, value in aux_outputs.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > batch_index:
                sliced[name] = value[batch_index: batch_index + 1]
        return sliced

    @staticmethod
    def _select_debug_sample_index_from_targets(targets_dict):
        if not targets_dict:
            return 0, False
        first_target_any = next(iter(targets_dict.values()))
        first_target = first_target_any[0] if isinstance(first_target_any, (list, tuple)) else first_target_any
        if not torch.is_tensor(first_target) or first_target.ndim == 0:
            return 0, False
        for b in range(first_target.shape[0]):
            if torch.any(first_target[b] != 0):
                return b, True
        return 0, False

    @staticmethod
    def _detach_debug_tensors(data_dict):
        detached = {}
        for name, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                detached[name] = value.detach().cpu()
        return detached

    def _capture_validation_debug_snapshot(self, inputs, targets_dict, outputs, aux_outputs, batch_index: int):
        inputs_first = inputs[batch_index: batch_index + 1].detach().cpu()

        targets_dict_first_all = {}
        for t_name, t_val in targets_dict.items():
            if isinstance(t_val, (list, tuple)):
                targets_dict_first_all[t_name] = t_val[0][batch_index: batch_index + 1].detach().cpu()
            else:
                targets_dict_first_all[t_name] = t_val[batch_index: batch_index + 1].detach().cpu()

        outputs_dict_first = {}
        for t_name, p_val in outputs.items():
            if isinstance(p_val, (list, tuple)):
                outputs_dict_first[t_name] = p_val[0][batch_index: batch_index + 1].detach().cpu()
            else:
                outputs_dict_first[t_name] = p_val[batch_index: batch_index + 1].detach().cpu()

        raw_aux_outputs_dict_first = self._detach_debug_tensors(
            self._slice_aux_outputs_for_debug(aux_outputs, batch_index)
        )

        return {
            "input": inputs_first,
            "targets_all": targets_dict_first_all,
            "outputs": outputs_dict_first,
            "raw_aux_outputs": raw_aux_outputs_dict_first,
        }

    @staticmethod
    def _select_validation_debug_snapshot_for_epoch(candidates, fallback, epoch: int):
        if candidates:
            return candidates[int(epoch) % len(candidates)]
        return fallback

    def _prepare_aux_debug_outputs(self, aux_outputs, reference_input):
        if not aux_outputs or not isinstance(reference_input, torch.Tensor):
            return {}

        prepared = {}
        guide_mask = aux_outputs.get("guide_mask")
        if guide_mask is not None:
            upsampled = F.interpolate(
                guide_mask.float(),
                size=reference_input.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).clamp_(0.0, 1.0)
            prepared[f"guide_{self._resolve_guide_supervision_target_name()}"] = upsampled.detach()
        return prepared

    @staticmethod
    def _is_feature_encoder_aux_output(name: str) -> bool:
        raw = str(name).strip().lower()
        if not raw.startswith("enc_"):
            return False
        try:
            int(raw.split("_", 1)[1])
        except (IndexError, ValueError):
            return False
        return True

    @staticmethod
    def _stack_preview_rows(rows):
        valid_rows = [row for row in rows if row is not None]
        if not valid_rows:
            return None
        max_width = max(row.shape[1] for row in valid_rows)
        padded_rows = []
        for row in valid_rows:
            if row.shape[1] < max_width:
                row = np.pad(
                    row,
                    ((0, 0), (0, max_width - row.shape[1]), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            padded_rows.append(np.ascontiguousarray(row, dtype=np.uint8))
        return np.ascontiguousarray(np.vstack(padded_rows), dtype=np.uint8)

    def _make_feature_encoder_guide_preview_row(self, aux_outputs_dict, *, row_prefix: str | None = None):
        if not aux_outputs_dict:
            return None

        stage_items = [
            (name, tensor)
            for name, tensor in aux_outputs_dict.items()
            if self._is_feature_encoder_aux_output(name) and isinstance(tensor, torch.Tensor)
        ]
        if not stage_items:
            return None

        stage_items.sort(key=lambda item: int(str(item[0]).split("_", 1)[1]))
        rendered_panels = []
        max_height = 0
        max_width = 0

        for aux_name, aux_tensor in stage_items:
            aux_tensor = aux_tensor.float() if aux_tensor.dtype == torch.bfloat16 else aux_tensor
            aux_np = aux_tensor.detach().cpu().numpy()
            while aux_np.ndim > 4:
                aux_np = aux_np[0]
            if aux_np.ndim == 4:
                aux_np = aux_np[0]

            if aux_np.ndim == 2:
                preview_slice = aux_np
                is_2d = True
            else:
                preview_slice = aux_np[max(aux_np.shape[0] // 2, 0)]
                is_2d = False

            value_range = _compute_display_value_range(
                aux_np,
                is_2d_run=is_2d,
                task_name=aux_name,
                task_cfg={},
            )
            preview = convert_slice_to_bgr(preview_slice, value_range=value_range)
            rendered_panels.append((aux_name, np.ascontiguousarray(preview, dtype=np.uint8)))
            max_height = max(max_height, preview.shape[0])
            max_width = max(max_width, preview.shape[1])

        labeled_panels = []
        for aux_name, preview in rendered_panels:
            if preview.shape[0] != max_height or preview.shape[1] != max_width:
                preview = np.array(
                    Image.fromarray(preview).resize((max_width, max_height), resample=Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            label = aux_name if not row_prefix else f"{row_prefix} {aux_name}"
            labeled_panels.append(add_text_label(preview, label))

        return np.ascontiguousarray(np.hstack(labeled_panels), dtype=np.uint8)

    def _make_feature_encoder_guide_preview_image(self, aux_outputs_dict, *, train_aux_outputs_dict=None):
        val_row = self._make_feature_encoder_guide_preview_row(aux_outputs_dict, row_prefix="Val")
        train_row = self._make_feature_encoder_guide_preview_row(train_aux_outputs_dict, row_prefix="Train")
        return self._stack_preview_rows([val_row, train_row])

    @staticmethod
    def _save_preview_image(preview_image, save_path):
        if preview_image is None or save_path is None:
            return
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(preview_image, dtype=np.uint8)).save(out_path)

    def _make_aux_preview_image(self, aux_outputs_dict):
        if not aux_outputs_dict:
            return None

        aux_name = sorted(aux_outputs_dict.keys())[0]
        aux_tensor = aux_outputs_dict[aux_name]
        if not isinstance(aux_tensor, torch.Tensor):
            return None
        if aux_tensor.dtype == torch.bfloat16:
            aux_tensor = aux_tensor.float()

        aux_np = aux_tensor.detach().cpu().numpy()
        while aux_np.ndim > 4:
            aux_np = aux_np[0]
        if aux_np.ndim == 4:
            aux_np = aux_np[0]

        if aux_np.ndim == 2:
            preview_slice = aux_np
            is_2d = True
        else:
            preview_slice = aux_np[max(aux_np.shape[0] // 2, 0)]
            is_2d = False

        value_range = _compute_display_value_range(
            aux_np,
            is_2d_run=is_2d,
            task_name=aux_name,
            task_cfg={},
        )
        preview = convert_slice_to_bgr(preview_slice, value_range=value_range)
        return np.ascontiguousarray(preview, dtype=np.uint8)

    def _build_debug_media_payload(self, debug_preview_image=None, guide_preview_image=None):
        payload = {}
        for key, image in (("debug_image", debug_preview_image), ("debug_guide_image", guide_preview_image)):
            if image is None:
                continue
            image_to_log = image
            if image_to_log.ndim == 3 and image_to_log.shape[2] == 3:
                image_to_log = image_to_log[..., ::-1]
            payload[key] = np.ascontiguousarray(image_to_log)
        return payload

    @staticmethod
    def _attach_debug_media_to_wandb_metrics(metrics, media_payload, wandb_module):
        if "debug_image" in media_payload:
            metrics["debug_image"] = wandb_module.Image(media_payload["debug_image"])
        if "debug_guide_image" in media_payload:
            metrics["debug_guide_image"] = wandb_module.Image(media_payload["debug_guide_image"])
        return metrics

    # --- losses ---- #
    def _build_loss(self):
        loss_fns = {}
        self._deferred_losses = {}  # losses that may be added later in training (at a selected epoch)
        
        def _pretty_loss_name(loss_fn, fallback_name: str):
            """Return a human-friendly loss name including base loss under wrappers."""
            try:
                names = []
                lf = loss_fn
                # Unwrap nested wrappers exposing `.loss`
                while hasattr(lf, 'loss'):
                    names.append(lf.__class__.__name__)
                    lf = lf.loss
                base = lf.__class__.__name__
                if names:
                    return f"{' + '.join(names)} ({base})"
                return base
            except Exception:
                return fallback_name

        for task_name, task_info in self.mgr.targets.items():
            task_losses = []
            deferred_losses = []

            target_ignore_value = None
            for alias in ("ignore_index", "ignore_label", "ignore_value"):
                value = task_info.get(alias)
                if value is not None:
                    target_ignore_value = value
                    break

            if "losses" in task_info:
                print(f"Target {task_name} using multiple losses:")
                for loss_cfg in task_info["losses"]:
                    loss_name = loss_cfg["name"]
                    loss_weight = loss_cfg.get("weight", 1.0)
                    loss_kwargs = dict(loss_cfg.get("kwargs", {}))
                    start_epoch = loss_cfg.get("start_epoch", 0)

                    for key, value in loss_cfg.items():
                        if key not in {"name", "weight", "kwargs", "start_epoch"}:
                            loss_kwargs.setdefault(key, value)

                    if target_ignore_value is not None and "ignore_index" not in loss_kwargs:
                        loss_kwargs["ignore_index"] = target_ignore_value

                    weight = loss_kwargs.get("weight", None)
                    ignore_index = loss_kwargs.get("ignore_index", -100)
                    pos_weight = loss_kwargs.get("pos_weight", None)

                    try:
                        loss_fn = _create_loss(
                            name=loss_name,
                            loss_config=loss_kwargs,
                            weight=weight,
                            ignore_index=ignore_index,
                            pos_weight=pos_weight,
                            mgr=self.mgr
                        )
                        # If deep supervision is enabled, wrap the loss in nnUNet-style DS wrapper
                        skip_ds = loss_cfg.get("skip_deep_supervision", False)
                        if getattr(self.mgr, 'enable_deep_supervision', False) and getattr(self, '_ds_weights', None) is not None and not skip_ds:
                            # Wrap all losses incl. skeleton-aware; wrapper now forwards extra args
                            loss_fn = DeepSupervisionWrapper(loss_fn, self._ds_weights)
                        
                        if start_epoch > 0:
                            # Store for later addition
                            deferred_losses.append({
                                'loss_fn': loss_fn,
                                'weight': loss_weight,
                                'start_epoch': start_epoch,
                                'name': loss_name
                            })
                            print(f"  - {_pretty_loss_name(loss_fn, loss_name)} (weight: {loss_weight}) - will start at epoch {start_epoch}")
                        else:
                            # Add immediately
                            task_losses.append((loss_fn, loss_weight))
                            print(f"  - {_pretty_loss_name(loss_fn, loss_name)} (weight: {loss_weight})")
                    except RuntimeError as e:
                        raise ValueError(
                            f"Failed to create loss function '{loss_name}' for target '{task_name}': {str(e)}")

            loss_fns[task_name] = task_losses
            if deferred_losses:
                self._deferred_losses[task_name] = deferred_losses

        return loss_fns

    def _capture_loss_overrides(self):
        """
        Snapshot loss-related configuration for each target so it can be restored
        after loading a checkpoint that may overwrite mgr.targets.
        """
        targets = getattr(self.mgr, 'targets', None)
        if not targets:
            return {}

        overrides = {}
        for target_name, cfg in targets.items():
            if not isinstance(cfg, dict):
                continue
            target_override = {}
            if cfg.get("losses"):
                target_override["losses"] = deepcopy(cfg["losses"])
            elif cfg.get("loss_fn"):
                target_override["loss_fn"] = cfg["loss_fn"]
            # Preserve ignore label/index so CE doesn't crash after checkpoint overrides
            for alias in ("ignore_index", "ignore_label", "ignore_value"):
                if alias in cfg:
                    target_override[alias] = cfg[alias]
            if target_override:
                overrides[target_name] = target_override
        return overrides

    def _apply_loss_overrides(self, overrides):
        """
        Reapply stored loss configuration after checkpoint load so CLI/config
        overrides take precedence over persisted checkpoint values.
        """
        if not overrides:
            return

        targets = getattr(self.mgr, 'targets', None)
        if not targets:
            return

        applied = False
        for target_name, override in overrides.items():
            if target_name not in targets:
                continue
            if override.get("losses"):
                targets[target_name]["losses"] = deepcopy(override["losses"])
                targets[target_name].pop("loss_fn", None)
                applied = True
            elif override.get("loss_fn"):
                loss_name = override["loss_fn"]
                targets[target_name]["loss_fn"] = loss_name
                targets[target_name]["losses"] = [{
                    "name": loss_name,
                    "weight": 1.0,
                    "kwargs": {}
                }]
                applied = True
            # Reapply ignore_label/index/value if provided in overrides
            for alias in ("ignore_index", "ignore_label", "ignore_value"):
                if alias in override:
                    targets[target_name][alias] = override[alias]
                    applied = True

        if applied:
            if isinstance(getattr(self.mgr, 'model_config', None), dict):
                self.mgr.model_config["targets"] = deepcopy(targets)
            if isinstance(getattr(self.mgr, 'dataset_config', None), dict):
                self.mgr.dataset_config["targets"] = deepcopy(targets)
            # Always print when config overrides checkpoint loss values
            print("Config loss parameters override checkpoint values:")
            for target_name, override in overrides.items():
                if target_name in targets and override.get("losses"):
                    for loss_cfg in override["losses"]:
                        loss_name = loss_cfg.get("name", "unknown")
                        params = {k: v for k, v in loss_cfg.items()
                                  if k not in ("name", "weight", "kwargs")}
                        if params:
                            print(f"  {target_name}/{loss_name}: {params}")

    # --- deep supervision helpers --- #
    def _set_deep_supervision_enabled(self, model, enabled: bool):
        if not hasattr(model, 'task_decoders'):
            return
        for _, dec in model.task_decoders.items():
            if hasattr(dec, 'deep_supervision'):
                dec.deep_supervision = enabled

    def _get_deep_supervision_scales(self, model):
        cfg = getattr(model, 'final_config', {})
        pool_kernels = cfg.get('pool_op_kernel_sizes', None)
        if pool_kernels is None:
            return None
        arr = np.vstack(pool_kernels)
        # 1 / cumprod of pooling kernels, drop the last (lowest resolution not used for logits loss weight)
        scales = list(list(i) for i in 1 / np.cumprod(arr, axis=0))[:-1]
        return scales

    def _compute_ds_weights(self, n: int):
        if n <= 0:
            return None
        weights = np.array([1 / (2 ** i) for i in range(n)], dtype=np.float32)
        # Do not use the lowest resolution output
        weights[-1] = 0.0
        s = weights.sum()
        if s > 0:
            weights = weights / s
        return weights.tolist()

    def _get_interp_mode_for_target(self, target_name: str, ndim: int):
        """Return (mode, align_corners) for F.interpolate based on YAML ds_interpolation and dims.
        Supported values:
          - 'nearest' (default)
          - 'linear' (mapped to 'bilinear' in 2D, 'trilinear' in 3D)
          - 'bilinear' (2D only; 3D -> 'trilinear')
          - 'trilinear' (3D only; 2D -> 'bilinear')
          - 'area' (2D only; 3D falls back to 'nearest')
        """
        cfg = self.mgr.targets.get(target_name, {}) if hasattr(self.mgr, 'targets') else {}
        req = str(cfg.get('ds_interpolation', 'nearest')).lower()

        # Default
        mode = 'nearest'
        align = None

        if req == 'nearest':
            return 'nearest', None

        if req in ('linear', 'bilinear', 'trilinear'):
            if ndim == 4:  # BCHW (2D)
                mode = 'bilinear'
            elif ndim == 5:  # BCDHW (3D)
                mode = 'trilinear'
            align = False
            return mode, align

        if req == 'area':
            if ndim == 4:
                return 'area', None
            else:
                # area not supported for 3D, fall back safe
                return 'nearest', None

        # Fallback safe
        return 'nearest', None

    def _downsample_targets_for_ds(self, outputs, targets_dict):
        """Downsample ground truth targets to match deep supervision outputs.
        Only modifies keys that are predicted (present in outputs).
        Returns a copy of targets_dict with lists of tensors per key.
        """
        if getattr(self, '_ds_scales', None) is None:
            return targets_dict
        new_targets = dict(targets_dict)
        for t_name, pred in outputs.items():
            # Skip if no ground truth for this prediction (e.g., auxiliary outputs not supervised)
            if t_name not in targets_dict:
                continue
            # Only act if the network returns deep supervision (list) for this output
            if isinstance(pred, (list, tuple)):
                base_t = targets_dict[t_name]
                if base_t.ndim not in (4, 5):  # BCHW or BCDHW
                    continue
                ds_targets = []
                mode, align_corners = self._get_interp_mode_for_target(t_name, base_t.ndim)
                for s in self._ds_scales:
                    # interpolate targets per selected mode
                    if align_corners is None:
                        ds_t = F.interpolate(base_t.float(), scale_factor=s, mode=mode)
                    else:
                        ds_t = F.interpolate(base_t.float(), scale_factor=s, mode=mode, align_corners=align_corners)
                    ds_targets.append(ds_t.to(base_t.dtype))
                new_targets[t_name] = ds_targets

                # Also downsample associated skeleton target if present
                skel_key = f"{t_name}_skel"
                if skel_key in targets_dict:
                    base_skel = targets_dict[skel_key]
                    if base_skel.ndim in (4, 5):
                        ds_skels = []
                        for s in self._ds_scales:
                            # keep skeletons as nearest to preserve topology
                            ds_s = F.interpolate(base_skel.float(), scale_factor=s, mode='nearest')
                            ds_skels.append(ds_s.to(base_skel.dtype))
                        new_targets[skel_key] = ds_skels
        return new_targets

    def _update_scheduler_for_epoch(self, scheduler, optimizer, epoch):
        """
        Update the learning rate scheduler for the current epoch.
        Override this method in subclasses to implement epoch-based scheduler switching.
        
        Args:
            scheduler: Current scheduler
            optimizer: Current optimizer
            epoch: Current epoch number
            
        Returns:
            tuple: (scheduler, is_per_iteration_scheduler)
        """
        # By default, just return the existing scheduler
        # Subclasses can override to switch schedulers at specific epochs
        return scheduler, getattr(self, '_is_per_iteration_scheduler', False)
    
    def _update_loss_for_epoch(self, loss_fns, epoch):
        if hasattr(self, '_deferred_losses') and self._deferred_losses:
            task_names = list(self._deferred_losses.keys())
            
            for task_name in task_names:
                deferred_list = self._deferred_losses[task_name]

                losses_to_add = []
                remaining_deferred = []
                
                for deferred_loss in deferred_list:
                    if epoch >= deferred_loss['start_epoch']:
                        losses_to_add.append(deferred_loss)
                    else:
                        remaining_deferred.append(deferred_loss)

                for loss_info in losses_to_add:
                    loss_fns[task_name].append((loss_info['loss_fn'], loss_info['weight']))
                    print(f"\nEpoch {epoch}: Adding {loss_info['name']} to task '{task_name}' (weight: {loss_info['weight']})")

                if remaining_deferred:
                    self._deferred_losses[task_name] = remaining_deferred
                else:
                    del self._deferred_losses[task_name]
        
        return loss_fns

    def _update_dataloaders_for_epoch(self,
                                      train_dataloader,
                                      val_dataloader,
                                      train_dataset,
                                      val_dataset,
                                      epoch):
        """
        Optionally update/rebuild dataloaders for the current epoch.

        By default, returns the provided dataloaders unchanged. Trainers can override
        this to switch sampling strategies across epochs (e.g., warmup phases).

        Args:
            train_dataloader: The current training dataloader
            val_dataloader: The current validation dataloader
            train_dataset: The training dataset instance
            val_dataset: The validation dataset instance
            epoch: Current epoch number (0-indexed)

        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        return train_dataloader, val_dataloader

    # --- optimizer ---- #
    def _get_optimizer(self, model):

        optimizer_config = {
            'name': self.mgr.optimizer,
            'learning_rate': self.mgr.initial_lr,
            'weight_decay': self.mgr.weight_decay
        }

        return create_optimizer(optimizer_config, model)

    # --- scheduler --- #
    def _get_scheduler(self, optimizer):

        scheduler_type = getattr(self.mgr, 'scheduler', 'poly')
        scheduler_kwargs = getattr(self.mgr, 'scheduler_kwargs', {})

        scheduler = get_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            initial_lr=self.mgr.initial_lr,
            max_steps=self.mgr.max_epoch,
            **scheduler_kwargs
        )

        print(f"Using {scheduler_type} learning rate scheduler")
        
        # set some per iteration schedulers so we can easily step them once per iter vs once per epoch
        per_iter_schedulers = ['onecycle', 'cyclic', 'cosine_warmup', 'diffusers_cosine_warmup']
        is_per_iteration = scheduler_type.lower() in per_iter_schedulers
        
        return scheduler, is_per_iteration

    # --- scaler --- #
    def _initialize_evaluation_metrics(self):

        metrics = {}
        for task_name, task_config in self.mgr.targets.items():
            task_metrics = []

            num_classes = task_config.get('num_classes', 2)
            target_ignore_value = None
            for alias in ("ignore_index", "ignore_label", "ignore_value"):
                value = task_config.get(alias)
                if value is not None:
                    target_ignore_value = value
                    break

            if target_ignore_value is not None:
                task_metrics.append(ConnectedComponentsMetric(num_classes=num_classes, ignore_index=target_ignore_value))
            else:
                task_metrics.append(ConnectedComponentsMetric(num_classes=num_classes))

            # if num_classes == 2:
            #     task_metrics.append(CriticalComponentsMetric())

            if target_ignore_value is not None:
                task_metrics.append(IOUDiceMetric(num_classes=num_classes, ignore_index=target_ignore_value))
                task_metrics.append(VOIMetric(ignore_index=target_ignore_value))
            else:
                task_metrics.append(IOUDiceMetric(num_classes=num_classes))
                task_metrics.append(VOIMetric())
            # task_metrics.append(SkeletonBranchPointsMetric(num_classes=num_classes))
            # task_metrics.append(HausdorffDistanceMetric(num_classes=num_classes))
            metrics[task_name] = task_metrics
        
        return metrics
    
    def _get_scaler(self, device_type='cuda', use_amp=True, amp_dtype=torch.float16):
        # for cuda, we can use a grad scaler for mixed precision training if amp is enabled
        # for mps or cpu, or when amp is disabled, we create a dummy scaler that does nothing

        class DummyScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass


        if device_type == 'cuda' and use_amp and amp_dtype == torch.float16:
            # Use standard GradScaler when AMP is enabled on CUDA with float16
            print("Using GradScaler with CUDA AMP (float16)")
            return torch.amp.GradScaler('cuda')
        else:
            # Not using amp or not on cuda - no gradient scaling needed
            return DummyScaler()

    def _autocast_context(self, use_amp: bool):
        if not use_amp:
            return nullcontext()
        if self.device.type == 'cuda':
            return torch.amp.autocast('cuda', dtype=self.amp_dtype)
        if self.device.type == 'cpu':
            return torch.amp.autocast('cpu')
        if self.device.type in ['mlx', 'mps']:
            return torch.amp.autocast(self.device.type, dtype=self.amp_dtype)
        return torch.amp.autocast(self.device.type)

    # --- dataloaders --- #
    def _configure_dataloaders(self, train_dataset, val_dataset=None):

        # If no separate validation dataset provided, or both datasets point to the same source,
        # fall back to a random split of the training dataset.
        same_source = False
        if val_dataset is not None:
            try:
                train_path = getattr(train_dataset, 'data_path', None)
                val_path = getattr(val_dataset, 'data_path', None)
                same_source = (train_path is not None and val_path is not None and train_path == val_path)
            except Exception:
                same_source = False

        if val_dataset is None or val_dataset is train_dataset or same_source:
            dataset_size = len(train_dataset)

            # Get number of FG patches (patches with labels) - BG patches go to training only
            n_fg = getattr(train_dataset, 'n_fg', dataset_size)
            fg_indices = list(range(n_fg))
            bg_indices = list(range(n_fg, dataset_size))

            if hasattr(self.mgr, 'seed'):
                np.random.seed(self.mgr.seed)
                if self.mgr.verbose:
                    print(f"Using seed {self.mgr.seed} for train/val split")

            np.random.shuffle(fg_indices)

            train_val_split = self.mgr.tr_val_split
            split = int(np.floor(train_val_split * len(fg_indices)))

            # Train gets split FG + ALL BG patches, val only gets FG patches
            train_fg = fg_indices[:split]
            val_fg = fg_indices[split:]
            train_indices = train_fg + bg_indices
            val_indices = val_fg

            # Store counts for weighted sampling epoch size calculation
            n_train_fg = len(train_fg)
            n_train_bg = len(bg_indices)

            if same_source and self.mgr.verbose:
                print("Validation dataset shares the same source as training; using random split")
            if self.mgr.verbose:
                print(f"Split: {len(train_fg)} FG + {len(bg_indices)} BG patches for training, {len(val_fg)} FG patches for validation")
        else:
            # Separate validation dataset provided.
            # Check for patch position overlap to avoid data leakage when volumes are shared.
            val_indices = list(range(len(val_dataset)))

            # Build set of (volume_name, position) tuples from validation patches
            val_patch_positions = set()
            for vp in getattr(val_dataset, 'valid_patches', []):
                name = vp.volume_name or 'default'
                val_patch_positions.add((name, vp.position))

            # Include training patches that don't have exact position overlap with validation
            train_indices = []
            excluded_count = 0
            for i, vp in enumerate(getattr(train_dataset, 'valid_patches', [])):
                name = vp.volume_name or 'default'
                if (name, vp.position) in val_patch_positions:
                    # Exact same patch position in same volume - exclude to prevent leakage
                    excluded_count += 1
                else:
                    train_indices.append(i)

            # Count FG/BG in training indices for weighted sampling
            n_fg_dataset = getattr(train_dataset, 'n_fg', len(train_dataset))
            n_train_fg = sum(1 for i in train_indices if i < n_fg_dataset)
            n_train_bg = len(train_indices) - n_train_fg

            if self.mgr.verbose:
                print(f"Using external validation set: {len(val_indices)} val patches")
                if excluded_count > 0:
                    print(f"Excluded {excluded_count} train patches with same position as validation (leakage prevention)")
                print(f"Training with {len(train_indices)} patches ({n_train_fg} FG, {n_train_bg} BG)")

        # Batch size semantics: in all modes, --batch-size is per-GPU (per process)
        per_device_batch = self.mgr.train_batch_size

        # Build subset datasets so DistributedSampler can partition without overlap
        train_base = train_dataset
        val_base = val_dataset if val_dataset is not None else train_dataset
        train_subset = Subset(train_base, train_indices)
        val_subset = Subset(val_base, val_indices)

        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_subset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=False
            )
            # For validation we only run on rank 0; sampler unused there, but keep a sequential sampler for completeness
            val_sampler = None
        else:
            if hasattr(train_base, 'patch_weights') and isinstance(getattr(train_base, 'patch_weights', None), list):
                if train_base.patch_weights and len(train_base.patch_weights) >= len(train_base):
                    subset_weights = [train_base.patch_weights[idx] for idx in train_indices]
                    total_weight = float(sum(subset_weights))
                    if total_weight > 0:
                        weight_tensor = torch.tensor(subset_weights, dtype=torch.double)
                        generator = None
                        if hasattr(self.mgr, 'seed') and self.mgr.seed is not None:
                            generator = torch.Generator()
                            generator.manual_seed(int(self.mgr.seed))

                        # Calculate epoch size: all FG patches + enough BG so that BG is bg_to_fg_ratio of total
                        # If bg_to_fg_ratio=0.1, we want 10% of total samples to be BG
                        # bg_samples = n_fg * ratio / (1 - ratio)
                        bg_to_fg_ratio = float(getattr(self.mgr, 'bg_to_fg_ratio', 0.5))
                        if bg_to_fg_ratio < 1.0:
                            n_bg_samples = int(n_train_fg * bg_to_fg_ratio / (1.0 - bg_to_fg_ratio))
                        else:
                            n_bg_samples = n_train_bg  # ratio >= 1 means use all BG
                        n_bg_samples = min(n_bg_samples, n_train_bg)  # Can't sample more BG than exists
                        num_samples = n_train_fg + n_bg_samples

                        train_sampler = WeightedRandomSampler(
                            weights=weight_tensor,
                            num_samples=num_samples,
                            replacement=False,
                            generator=generator
                        )
                        if self.mgr.verbose:
                            bg_percent = 100.0 * n_bg_samples / num_samples if num_samples > 0 else 0
                            print(f"Using WeightedRandomSampler: {n_train_fg} FG + {n_bg_samples} BG = {num_samples} samples/epoch ({bg_percent:.1f}% BG)")
                    else:
                        train_sampler = SubsetRandomSampler(list(range(len(train_subset))))
                else:
                    train_sampler = SubsetRandomSampler(list(range(len(train_subset))))
            else:
                train_sampler = SubsetRandomSampler(list(range(len(train_subset))))
            val_sampler = SubsetRandomSampler(list(range(len(val_subset))))

        pin_mem = True if self.device.type == 'cuda' else False
        dl_kwargs = {}
        if self.mgr.train_num_dataloader_workers and self.mgr.train_num_dataloader_workers > 0:
            dl_kwargs['prefetch_factor'] = 2
            dl_kwargs['persistent_workers'] = True

        train_dataloader = DataLoader(
            train_subset,
            batch_size=per_device_batch,
            sampler=train_sampler,
            shuffle=False,
            pin_memory=pin_mem,
            num_workers=self.mgr.train_num_dataloader_workers,
            **dl_kwargs
        )

        # Validation dataloader will only be iterated on rank 0 in DDP
        val_dataloader = DataLoader(
            val_subset,
            batch_size=1,
            sampler=val_sampler,
            shuffle=False,
            pin_memory=pin_mem,
            num_workers=self.mgr.train_num_dataloader_workers,
            **dl_kwargs
        )

        return train_dataloader, val_dataloader, train_indices, val_indices

    def _initialize_training(self):
        if detect_s3_paths(self.mgr):
            print("\nDetected S3 paths in configuration")
            setup_multiprocessing_for_s3()

        # By default the training dataset applies augmentations in __getitem__ so
        # workers can parallelize them. When augment_on_device=True we preserve the
        # same pipeline here and disable dataset-side augmentation to avoid applying
        # transforms twice.
        stage_start = perf_counter()
        train_dataset = self._configure_dataset(is_training=True)
        self._record_startup_timing("train_dataset_init", perf_counter() - stage_start)
        self._train_dataset = train_dataset
        self._on_device_transforms = None
        if getattr(self.mgr, 'augment_on_device', False):
            self._on_device_transforms = getattr(train_dataset, 'transforms', None)
            train_dataset.transforms = None
        if self._profile_augmentations:
            self._augmentation_names = getattr(train_dataset, '_augmentation_names', None)

        if hasattr(self.mgr, 'val_data_path') and self.mgr.val_data_path is not None:
            from copy import deepcopy
            from vesuvius.models.utilities.data_format_utils import detect_data_format as _detect_df

            val_mgr = deepcopy(self.mgr)
            val_mgr.data_path = Path(self.mgr.val_data_path)

            detected_val_fmt = _detect_df(val_mgr.data_path)
            if detected_val_fmt is None:
                raise ValueError(f"Could not determine data format for validation directory: {val_mgr.data_path}")
            val_mgr.data_format = detected_val_fmt

            stage_start = perf_counter()
            val_dataset = self._build_dataset_for_mgr(val_mgr, is_training=False)
            self._record_startup_timing("val_dataset_init", perf_counter() - stage_start)
            print(f"Using {val_mgr.data_format} dataset format (validation from --val-dir)")
        else:
            # Reuse same source for validation without re-running expensive image checks
            from copy import deepcopy
            val_mgr = deepcopy(self.mgr)
            setattr(val_mgr, 'skip_image_checks', True)

            stage_start = perf_counter()
            val_dataset = self._build_dataset_for_mgr(val_mgr, is_training=False)
            self._record_startup_timing("val_dataset_init", perf_counter() - stage_start)
        
        stage_start = perf_counter()
        autodetect_sample = self._prepare_sample(train_dataset[0], is_training=True) if len(train_dataset) > 0 else None
        self._record_startup_timing("sample_autodetect", perf_counter() - stage_start)
        self.mgr.auto_detect_channels(dataset=train_dataset, sample=autodetect_sample)
        stage_start = perf_counter()
        model = self._build_model()
        self._record_startup_timing("model_build", perf_counter() - stage_start)

        self._ds_scales = None
        self._ds_weights = None
        stage_start = perf_counter()
        optimizer = self._get_optimizer(model)
        scheduler, is_per_iteration_scheduler = self._get_scheduler(optimizer)
        self._is_per_iteration_scheduler = is_per_iteration_scheduler  # Store for later use

        model.apply(InitWeights_He(neg_slope=0.2))
        model = model.to(self.device)

        use_amp = not getattr(self.mgr, 'no_amp', False)
        if not use_amp:
            print("Automatic Mixed Precision (AMP) is disabled")

        amp_dtype_setting = getattr(self.mgr, 'amp_dtype', 'float16')
        if amp_dtype_setting is None:
            amp_dtype_setting = 'float16'

        if isinstance(amp_dtype_setting, torch.dtype):
            resolved_amp_dtype = amp_dtype_setting
            amp_dtype_str = 'bfloat16' if amp_dtype_setting == torch.bfloat16 else 'float16'
        else:
            amp_dtype_str = str(amp_dtype_setting).lower()
            if amp_dtype_str in ('bfloat16', 'bf16'):
                resolved_amp_dtype = torch.bfloat16
                amp_dtype_str = 'bfloat16'
            elif amp_dtype_str in ('float16', 'fp16', 'half'):
                resolved_amp_dtype = torch.float16
                amp_dtype_str = 'float16'
            else:
                if not self.is_distributed or self.rank == 0:
                    print(f"Unrecognized amp_dtype '{amp_dtype_setting}', defaulting to float16")
                resolved_amp_dtype = torch.float16
                amp_dtype_str = 'float16'

        self.amp_dtype = resolved_amp_dtype
        self.amp_dtype_str = amp_dtype_str

        if self.device.type in ['mlx', 'mps'] and self.amp_dtype == torch.bfloat16:
            if not self.is_distributed or self.rank == 0:
                print("bfloat16 autocast not supported on this backend; falling back to float16")
            self.amp_dtype = torch.float16
            self.amp_dtype_str = 'float16'

        if use_amp and self.device.type == 'cuda' and self.amp_dtype == torch.bfloat16:
            if not self.is_distributed or self.rank == 0:
                print("Using CUDA AMP with bfloat16 (GradScaler disabled)")

        scaler = self._get_scaler(self.device.type, use_amp=use_amp, amp_dtype=self.amp_dtype)
        self._record_startup_timing("optimizer_scheduler_scaler_init", perf_counter() - stage_start)
        stage_start = perf_counter()
        train_dataloader, val_dataloader, train_indices, val_indices = self._configure_dataloaders(train_dataset,
                                                                                                   val_dataset)
        self._record_startup_timing("dataloader_build", perf_counter() - stage_start)

        ckpt_out_base = str(self.mgr.ckpt_out_base)
        os.makedirs(ckpt_out_base, exist_ok=True)
        model_ckpt_dir = os.path.join(ckpt_out_base, self.mgr.model_name)
        os.makedirs(model_ckpt_dir, exist_ok=True)

        now = datetime.now()
        date_str = now.strftime('%m%d%y')
        time_str = now.strftime('%H%M')
        ckpt_dir = os.path.join(ckpt_out_base, f"{self.mgr.model_name}_{date_str}{time_str}")
        os.makedirs(ckpt_dir, exist_ok=True)

        loss_overrides = self._capture_loss_overrides()

        start_epoch = 0
        checkpoint_loaded = False
        if hasattr(self.mgr, 'checkpoint_path') and self.mgr.checkpoint_path:
            model, optimizer, scheduler, start_epoch, checkpoint_loaded = load_checkpoint(
                checkpoint_path=self.mgr.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mgr=self.mgr,
                device=self.device,
                load_weights_only=getattr(self.mgr, 'load_weights_only', False)
            )

            if checkpoint_loaded:
                self._apply_loss_overrides(loss_overrides)
                # Store any additional state (e.g., EMA model) for subclasses to use
                try:
                    ckpt = torch.load(self.mgr.checkpoint_path, map_location=self.device, weights_only=False)
                    if isinstance(ckpt, dict) and 'ema_model' in ckpt:
                        self._checkpoint_ema_state = ckpt['ema_model']
                        self._checkpoint_ema_optimizer_step = int(ckpt.get('ema_optimizer_step', 0))
                        print("Found EMA model state in checkpoint")
                    del ckpt
                except Exception:
                    pass

            if checkpoint_loaded and self.mgr.load_weights_only:
                scheduler, is_per_iteration_scheduler = self._get_scheduler(optimizer)

        ds_enabled = bool(getattr(self.mgr, 'enable_deep_supervision', False))
        self._set_deep_supervision_enabled(model, ds_enabled)
        if ds_enabled:
            self._ds_scales = self._get_deep_supervision_scales(model)
            if self._ds_scales is not None:
                self._ds_weights = self._compute_ds_weights(len(self._ds_scales))
        else:
            self._ds_scales = None
            self._ds_weights = None
        loss_fns = self._build_loss()
        self._initialize_ema_model(model)
        raw_model = self._unwrap_model(model)
        if hasattr(raw_model, "_compile_guidance_submodules"):
            stage_start = perf_counter()
            compiled_guide_modules = raw_model._compile_guidance_submodules(device_type=self.device.type)
            if compiled_guide_modules:
                self._record_startup_timing("guide_submodule_compile", perf_counter() - stage_start)
                if not self.is_distributed or self.rank == 0:
                    print("Compiled guide submodules: " + ", ".join(compiled_guide_modules))

        compile_policy = self._resolve_compile_policy(model)
        if compile_policy == "module":
            model = self._maybe_compile_model(model)
        stage_start = perf_counter()
        model = self._wrap_model_for_distributed_training(model)
        self._record_startup_timing("ddp_wrap", perf_counter() - stage_start)
        if compile_policy == "ddp_wrapper":
            model = self._maybe_compile_model(model)

        return {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'is_per_iteration_scheduler': is_per_iteration_scheduler,
            'loss_fns': loss_fns,
            'scaler': scaler,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'use_amp': use_amp,
            'start_epoch': start_epoch,
            'ckpt_dir': ckpt_dir,
            'model_ckpt_dir': model_ckpt_dir
        }

    def _initialize_wandb(self, train_dataset, val_dataset, train_indices, val_indices, ckpt_dir=None):
        """Initialize Weights & Biases logging if configured."""
        # Only rank 0 should initialize wandb in DDP
        if self.mgr.wandb_project and (not self.is_distributed or self.rank == 0):
            import wandb  # lazy import in case it's not available
            import json
            import os
            from datetime import datetime

            train_val_splits = save_train_val_filenames(self, train_dataset, val_dataset, train_indices, val_indices)

            save_dir = ckpt_dir if ckpt_dir else os.getcwd()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            splits_filename = f"train_val_splits_{self.mgr.model_name}_{timestamp}.json"
            splits_filepath = os.path.join(save_dir, splits_filename)

            # Save to local file
            with open(splits_filepath, 'w') as f:
                json.dump(train_val_splits, f, indent=2)
            print(f"Saved train/val splits to: {splits_filepath}")

            mgr_config = self.mgr.convert_to_dict()
            mgr_config['train_val_splits_file'] = splits_filepath
            mgr_config['train_patch_count'] = len(train_indices)
            mgr_config['val_patch_count'] = len(val_indices)

            wandb_init_kwargs = {
                "entity": self.mgr.wandb_entity,
                "project": self.mgr.wandb_project,
                "group": self.mgr.model_name,
                "config": mgr_config,
            }
            wandb_resume = getattr(self.mgr, "wandb_resume", None)
            if wandb_resume:
                resume_arg = str(wandb_resume).strip()
                resume_lower = resume_arg.lower()
                known_resume_modes = {"allow", "auto", "must", "never"}
                if resume_lower in known_resume_modes:
                    wandb_init_kwargs["resume"] = resume_lower
                else:
                    wandb_init_kwargs["resume"] = "allow"
                    wandb_init_kwargs["id"] = resume_arg
            run_name = getattr(self.mgr, "wandb_run_name", None)
            if run_name:
                wandb_init_kwargs["name"] = run_name

            wandb.init(**wandb_init_kwargs)

            # Log the splits file as an artifact for reference
            artifact = wandb.Artifact(f"train_val_splits_{timestamp}", type="dataset")
            artifact.add_file(splits_filepath)
            wandb.log_artifact(artifact)

    def _extract_targets(self, data_dict):
        # Only include tensor targets; skip metadata and lists (e.g., 'regression_keys')
        non_blocking = self.device.type == 'cuda'
        return {
            k: v.to(self.device, non_blocking=non_blocking)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled", "regression_keys"]
            and hasattr(v, "to")
        }

    def _get_model_outputs(self, model, data_dict):
        non_blocking = self.device.type == 'cuda'
        inputs = data_dict["image"].to(self.device, non_blocking=non_blocking)
        targets_dict = self._extract_targets(data_dict)
        if self.device.type == 'cuda' and self._volume_dilate_by_name:
            padding_mask = data_dict["padding_mask"].to(self.device, non_blocking=True)
            volume_names = data_dict["patch_info"]["volume_name"]
            for i, volume_name in enumerate(volume_names):
                distance = self._volume_dilate_by_name.get(volume_name)
                if distance in (None, 0):
                    continue
                for target_name in self.mgr.targets:
                    if target_name not in targets_dict:
                        continue
                    target_tensor = targets_dict[target_name]
                    target_info = self.mgr.targets.get(target_name, {})
                    ignore_label = target_info.get("ignore_label", target_info.get("ignore_index", target_info.get("ignore_value")))
                    targets_dict[target_name][i:i + 1] = dilate_label_batch_with_cucim(
                        target_tensor[i:i + 1],
                        padding_mask[i:i + 1],
                        distance,
                        ignore_label,
                    )
        
        raw_model = self._unwrap_model(model)
        if getattr(raw_model, "guide_enabled", False):
            outputs, aux_outputs = model(inputs, return_aux=True)
            self._current_aux_outputs = aux_outputs
        else:
            outputs = model(inputs)
            self._current_aux_outputs = {}

        # If deep supervision is enabled, prepare lists of downsampled targets
        if getattr(self.mgr, 'enable_deep_supervision', False):
            targets_dict = self._downsample_targets_for_ds(outputs, targets_dict)
        
        return inputs, targets_dict, outputs

    def _apply_transforms_per_sample(self, tfm, batched_dict):
        """Apply a ComposeTransforms pipeline to each sample in a batched dict.
        Expects tensors shaped [B, C, ...]. Returns a new batched dict with tensors stacked on dim 0.
        Preserves nested metadata structures such as patch_info produced by the default collate_fn.
        """
        if 'image' not in batched_dict or not isinstance(batched_dict['image'], torch.Tensor):
            return batched_dict

        def _extract_sample(value, batch_size: int, sample_idx: int):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                return value[sample_idx]
            if isinstance(value, dict):
                return {
                    key: _extract_sample(subvalue, batch_size, sample_idx)
                    for key, subvalue in value.items()
                }
            if isinstance(value, list):
                if len(value) == batch_size:
                    return value[sample_idx]
                return [_extract_sample(item, batch_size, sample_idx) for item in value]
            if isinstance(value, tuple):
                if len(value) == batch_size:
                    return value[sample_idx]
                return tuple(_extract_sample(item, batch_size, sample_idx) for item in value)
            return value

        def _collate_samples(values):
            first = values[0]
            if isinstance(first, torch.Tensor):
                return torch.stack(values, dim=0)
            if isinstance(first, dict):
                return {
                    key: _collate_samples([value[key] for value in values])
                    for key in first
                }
            if isinstance(first, list):
                return [
                    _collate_samples([value[idx] for value in values])
                    for idx in range(len(first))
                ]
            if isinstance(first, tuple):
                return tuple(
                    _collate_samples([value[idx] for value in values])
                    for idx in range(len(first))
                )
            return values

        B = batched_dict['image'].shape[0]
        out_accum = {}
        perf_names = self._augmentation_names if (self._profile_augmentations and self._augmentation_names) else None
        for b in range(B):
            sample = {}
            for k, v in batched_dict.items():
                sample[k] = _extract_sample(v, B, b)
            if perf_names is not None:
                sample['_aug_perf'] = {name: 0.0 for name in perf_names}
            sample_out = tfm(**sample)
            for k, v in sample_out.items():
                out_accum.setdefault(k, []).append(v)

        batched_out = {}
        for k, vals in out_accum.items():
            batched_out[k] = _collate_samples(vals)
        return batched_out

    def _train_step(self, model, data_dict, loss_fns, use_amp, autocast_ctx, epoch, step, verbose=False,
                    scaler=None, optimizer=None, num_iters=None, grad_accumulate_n=1):
        """Execute a single training step including gradient updates."""
        global_step = step

        data_dict = self._prepare_batch(data_dict, is_training=True)

        if epoch == 0 and step == 0 and verbose:
            print("Items from the first batch -- Double check that your shapes and values are expected:")
            for item, val in data_dict.items():
                if isinstance(val, dict):
                    print(f"{item}: (dictionary with keys: {list(val.keys())})")
                    for sub_key, sub_val in val.items():
                        print(
                            f"  {sub_key}: {sub_val.dtype}, {sub_val.shape}, min {sub_val.min()} max {sub_val.max()}")
                else:
                    print(f"{item}: {val.dtype}, {val.shape}, min {val.min()} max {val.max()}")

        # Optionally run augmentations on the model device instead of Dataset workers
        if getattr(self.mgr, 'augment_on_device', False):
            tfm = self._on_device_transforms
            if tfm is None:
                data_for_forward = data_dict
            else:
                dd = {}
                for k, v in data_dict.items():
                    dd[k] = v.to(self.device, non_blocking=(self.device.type == 'cuda')) if isinstance(v, torch.Tensor) else v

                # Apply transforms per-sample (transforms expect unbatched (C, ...) tensors)
                try:
                    data_for_forward = self._apply_transforms_per_sample(tfm, dd)
                except Exception as e:
                    raise RuntimeError(f"On-device augmentation failed: {e}")
        else:
            data_for_forward = data_dict

        self._accumulate_aug_timers(data_for_forward)

        should_time_first_step = self.startup_timing and epoch == 0 and step == 0
        if should_time_first_step:
            step_stage_start = perf_counter()
        with autocast_ctx:
            inputs, targets_dict, outputs = self._get_model_outputs(model, data_for_forward)
            if should_time_first_step:
                self._record_startup_timing("first_forward", perf_counter() - step_stage_start)
                step_stage_start = perf_counter()
            total_loss, task_losses = self._compute_train_loss(outputs, targets_dict, loss_fns)
            if should_time_first_step:
                self._record_startup_timing("first_loss_compute", perf_counter() - step_stage_start)

        # Handle gradient accumulation, clipping, and optimizer step
        # Scale loss by accumulation steps to maintain same effective batch size
        scaled_loss = total_loss / grad_accumulate_n

        # backward
        if should_time_first_step:
            step_stage_start = perf_counter()
        scaler.scale(scaled_loss).backward()
        if should_time_first_step:
            self._record_startup_timing("first_backward", perf_counter() - step_stage_start)

        optimizer_stepped = False
        if (step + 1) % grad_accumulate_n == 0 or (step + 1) == num_iters:
            should_time_optimizer = self.startup_timing and not self._first_optimizer_timing_recorded
            if should_time_optimizer:
                step_stage_start = perf_counter()
            scaler.unscale_(optimizer)
            grad_clip = getattr(self.mgr, 'gradient_clip', 12.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = True
            if should_time_optimizer:
                self._record_startup_timing("first_optimizer_step", perf_counter() - step_stage_start)
                self._first_optimizer_timing_recorded = True

        return total_loss, task_losses, inputs, targets_dict, outputs, optimizer_stepped

    def _compute_task_losses(self, outputs, targets_dict, loss_fns, *, apply_task_weights: bool):
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        task_losses = {}

        for t_name, t_gt in targets_dict.items():
            if not self._should_include_target_in_loss(t_name):
                continue

            t_pred = outputs[t_name]
            task_loss_fns = loss_fns[t_name]
            task_weight = self.mgr.targets[t_name].get("weight", 1.0) if apply_task_weights else 1.0

            ref_tensor = t_pred[0] if isinstance(t_pred, (list, tuple)) else t_pred
            task_total_loss = torch.zeros((), device=ref_tensor.device, dtype=torch.float32)
            for loss_fn, loss_weight in task_loss_fns:
                pred_for_loss, gt_for_loss = t_pred, t_gt
                if isinstance(t_pred, (list, tuple)) and not isinstance(loss_fn, DeepSupervisionWrapper):
                    pred_for_loss = t_pred[0]
                    if isinstance(t_gt, (list, tuple)):
                        gt_for_loss = t_gt[0]

                loss_value = self._compute_loss_value(
                    loss_fn,
                    pred_for_loss,
                    gt_for_loss,
                    target_name=t_name,
                    targets_dict=targets_dict,
                    outputs=outputs,
                )
                # Some losses (e.g., Betti) return (loss, aux_dict); extract loss
                if isinstance(loss_value, tuple) and len(loss_value) == 2:
                    loss_value = loss_value[0]
                # DEBUG: Accumulate raw loss values for epoch-end summary
                loss_fn_name = type(loss_fn).__name__
                if hasattr(loss_fn, 'loss'):  # Unwrap DeepSupervisionWrapper
                    loss_fn_name = type(loss_fn.loss).__name__
                key = f"{t_name}/{loss_fn_name}"
                if not hasattr(self, '_debug_raw_losses'):
                    self._debug_raw_losses = {}
                if key not in self._debug_raw_losses:
                    self._debug_raw_losses[key] = {'values': [], 'weight': loss_weight}
                self._debug_raw_losses[key]['values'].append(loss_value.item())

                task_total_loss += loss_weight * loss_value

            weighted_loss = task_weight * task_total_loss
            total_loss = total_loss + weighted_loss.to(total_loss.dtype)
            task_losses[t_name] = weighted_loss.detach().cpu().item()

        if self.guide_loss_weight > 0.0:
            guide_loss = self._compute_guide_alignment_loss(targets_dict)
            if guide_loss is not None:
                weighted_guide_loss = self.guide_loss_weight * guide_loss
                total_loss = total_loss + weighted_guide_loss.to(total_loss.dtype)
                task_losses["guide_mask"] = weighted_guide_loss.detach().cpu().item()

        return total_loss, task_losses

    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        return self._compute_task_losses(outputs, targets_dict, loss_fns, apply_task_weights=True)

    def _validation_step(self, model, data_dict, loss_fns, use_amp):
        data_dict = self._prepare_batch(data_dict, is_training=False)
        with self._autocast_context(use_amp):
            inputs, targets_dict, outputs = self._get_model_outputs(model, data_dict)
            task_losses = self._compute_validation_loss(outputs, targets_dict, loss_fns)

        return task_losses, inputs, targets_dict, outputs

    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        _, task_losses = self._compute_task_losses(
            outputs, targets_dict, loss_fns, apply_task_weights=False
        )
        return task_losses

    def _on_epoch_end(self, epoch, model, optimizer, scheduler, train_dataset,
                       ckpt_dir, model_ckpt_dir, checkpoint_history, best_checkpoints,
                       avg_val_loss):
        """Handle end-of-epoch operations: checkpointing, cleanup, etc."""
        ckpt_path = os.path.join(
            ckpt_dir,
            f"{self.mgr.model_name}_epoch{epoch + 1}.pth"
        )

        checkpoint_data = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            checkpoint_path=ckpt_path,
            model_config=getattr(model, 'final_config', None),
            train_dataset=train_dataset,
            additional_data=self._get_additional_checkpoint_data()
        )

        checkpoint_history.append((epoch, ckpt_path))

        del checkpoint_data

        # Manage checkpoint history
        checkpoint_history, best_checkpoints = manage_checkpoint_history(
            checkpoint_history=checkpoint_history,
            best_checkpoints=best_checkpoints,
            epoch=epoch,
            checkpoint_path=ckpt_path,
            validation_loss=avg_val_loss,
            checkpoint_dir=ckpt_dir,
            model_name=self.mgr.model_name,
            max_recent=3,
            max_best=2
        )

        cleanup_old_configs(
            model_ckpt_dir=model_ckpt_dir,
            model_name=self.mgr.model_name,
            keep_latest=1
        )

        return checkpoint_history, best_checkpoints, ckpt_path

    def _prepare_metrics_for_logging(self, epoch, step, epoch_losses, current_lr=None, val_losses=None):

        # this is a separate method just so i have an easy way to accumulate metrics
        # TODO: make this easier

        metrics = {"epoch": epoch, "step": step}

        # Add training losses, including auxiliary entries such as guide_mask
        for t_name, losses in epoch_losses.items():
            if len(losses) > 0:
                metrics[f"train_loss_{t_name}"] = np.mean(losses[-100:])

        # Add total training loss
        if epoch_losses:
            recent_losses = [np.mean(losses[-100:]) for losses in epoch_losses.values() if len(losses) > 0]
            if recent_losses:
                metrics["train_loss_total"] = np.mean(recent_losses)

        # Add learning rate if provided
        if current_lr is not None:
            metrics["learning_rate"] = current_lr

        # Add validation losses if provided, including auxiliary entries such as guide_mask
        if val_losses is not None:
            total_val_loss = 0.0
            num_val_entries = 0
            for t_name, losses in val_losses.items():
                if len(losses) > 0:
                    val_avg = np.mean(losses)
                    metrics[f"val_loss_{t_name}"] = val_avg
                    total_val_loss += val_avg
                    num_val_entries += 1

            # Add total validation loss
            if num_val_entries > 0:
                metrics["val_loss_total"] = total_val_loss / num_val_entries

        return metrics

    def train(self):

        training_state = self._initialize_training()

        # Unpack the state
        model = training_state['model']
        optimizer = training_state['optimizer']
        scheduler = training_state['scheduler']
        is_per_iteration_scheduler = training_state['is_per_iteration_scheduler']
        loss_fns = training_state['loss_fns']
        scaler = training_state['scaler']
        train_dataset = training_state['train_dataset']
        val_dataset = training_state['val_dataset']
        train_dataloader = training_state['train_dataloader']
        val_dataloader = training_state['val_dataloader']
        train_indices = training_state['train_indices']
        val_indices = training_state['val_indices']
        use_amp = training_state['use_amp']
        start_epoch = training_state['start_epoch']
        ckpt_dir = training_state['ckpt_dir']
        model_ckpt_dir = training_state['model_ckpt_dir']

        self._initialize_wandb(train_dataset, val_dataset, train_indices, val_indices, ckpt_dir)

        val_loss_history = {}  # {epoch: validation_loss}
        checkpoint_history = deque(maxlen=3)
        best_checkpoints = []
        debug_gif_history = deque(maxlen=3)
        best_debug_gifs = []  # List of (val_loss, epoch, gif_path)

        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation

        early_stopping_patience = getattr(self.mgr, 'early_stopping_patience', 20)
        if early_stopping_patience > 0:
            best_val_loss = float('inf')
            patience_counter = 0
            print(f"Early stopping enabled with patience: {early_stopping_patience} epochs")
        else:
            print("Early stopping disabled")

        # ---- training! ----- #
        for epoch in range(start_epoch, self.mgr.max_epoch):
            # Ensure each rank shuffles differently per epoch
            if self.is_distributed and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            # Update loss functions for this epoch
            loss_fns = self._update_loss_for_epoch(loss_fns, epoch)
            
            # Update scheduler for this epoch (for epoch-based scheduler switching)
            scheduler, is_per_iteration_scheduler = self._update_scheduler_for_epoch(scheduler, optimizer, epoch)
            step_scheduler_at_epoch_begin = getattr(scheduler, 'step_on_epoch_begin', False) and not is_per_iteration_scheduler

            if step_scheduler_at_epoch_begin:
                scheduler.step(epoch)

            # Optionally update dataloaders for this epoch (e.g., warmup strategies)
            train_dataloader, val_dataloader = self._update_dataloaders_for_epoch(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epoch=epoch
            )

            model.train()

            if getattr(self.mgr, 'max_steps_per_epoch', None) is not None and self.mgr.max_steps_per_epoch > 0:
                num_iters = min(len(train_dataloader), self.mgr.max_steps_per_epoch)
            else:
                num_iters = len(train_dataloader)

            self._reset_epoch_aug_timers()
            epoch_losses = {t_name: [] for t_name in self.mgr.targets}
            train_iter = iter(train_dataloader)
            # Progress bar for training iterations
            pbar = tqdm(total=num_iters, desc=f'Epoch {epoch + 1}/{self.mgr.max_epoch}') if (not self.is_distributed or self.rank == 0) else None
            
            # Variables to store train samples for debug visualization
            train_sample_input = None
            train_sample_targets = None
            train_sample_outputs = None
            train_sample_aux_outputs = None
            train_sample_raw_aux_outputs = None

            print(f"Using optimizer : {optimizer.__class__.__name__}")
            print(f"Using scheduler : {scheduler.__class__.__name__} (per-iteration: {is_per_iteration_scheduler})")
            print(f"Gradient accumulation steps : {grad_accumulate_n}")

            for i in range(num_iters):
                if i % grad_accumulate_n == 0:
                    optimizer.zero_grad(set_to_none=True)

                fetch_start = perf_counter() if (self.startup_timing and epoch == 0 and i == 0) else None
                data_dict = next(train_iter)
                if fetch_start is not None:
                    self._record_startup_timing("first_batch_fetch", perf_counter() - fetch_start)
                global_step += 1
                
                # Setup autocast context (dtype resolved based on CLI/config)
                autocast_ctx = self._autocast_context(use_amp)

                # Execute training step
                total_loss, task_losses, inputs, targets_dict, outputs, optimizer_stepped = self._train_step(
                    model=model,
                    data_dict=data_dict,
                    loss_fns=loss_fns,
                    use_amp=use_amp,
                    autocast_ctx=autocast_ctx,
                    epoch=epoch,
                    step=i,
                    verbose=self.mgr.verbose,
                    scaler=scaler,
                    optimizer=optimizer,
                    num_iters=num_iters,
                    grad_accumulate_n=grad_accumulate_n
                )

                for t_name, loss_value in task_losses.items():
                    if t_name not in epoch_losses:
                        epoch_losses[t_name] = []
                    epoch_losses[t_name].append(loss_value)
                

                if i == 0 and train_sample_input is None:
                    # Prefer choosing a labeled sample for debug when using semi-supervised trainers
                    first_target_key = list(targets_dict.keys())[0]
                    first_target_any = targets_dict[first_target_key]
                    first_target_tensor = first_target_any[0] if isinstance(first_target_any, (list, tuple)) else first_target_any

                    # If the trainer exposes labeled_batch_size, labeled samples are first in the batch
                    labeled_limit = None
                    if hasattr(self, 'labeled_batch_size') and inputs.shape[0] == self.mgr.train_batch_size:
                        labeled_limit = min(self.labeled_batch_size, first_target_tensor.shape[0])

                    # Search for a non-zero target within labeled region (if known), else whole batch
                    search_range = range(labeled_limit) if labeled_limit is not None else range(first_target_tensor.shape[0])
                    b_idx = 0
                    for b in search_range:
                        if torch.any(first_target_tensor[b] != 0):
                            b_idx = b
                            break

                    # Fallback: if labeled region found no positives and we limited search,
                    # keep the first labeled sample rather than drifting into unlabeled indices
                    if labeled_limit is None:
                        pass  # already searched full batch
                    else:
                        b_idx = min(b_idx, labeled_limit - 1)

                    train_sample_input = inputs[b_idx: b_idx + 1]
                    train_sample_targets_all = {}
                    for t_name, t_val in targets_dict.items():
                        if isinstance(t_val, (list, tuple)):
                            train_sample_targets_all[t_name] = t_val[0][b_idx: b_idx + 1]
                        else:
                            train_sample_targets_all[t_name] = t_val[b_idx: b_idx + 1]
                    train_sample_targets = {}
                    for t_name, t_tensor in train_sample_targets_all.items():
                        if t_name not in ['skel', 'is_unlabeled']:
                            train_sample_targets[t_name] = t_tensor
                    train_sample_outputs = {}
                    for t_name, p_val in outputs.items():
                        if isinstance(p_val, (list, tuple)):
                            train_sample_outputs[t_name] = p_val[0][b_idx: b_idx + 1]
                        else:
                            train_sample_outputs[t_name] = p_val[b_idx: b_idx + 1]
                    train_sample_raw_aux_outputs = self._slice_aux_outputs_for_debug(
                        getattr(self, "_current_aux_outputs", {}),
                        b_idx,
                    )
                    train_sample_aux_outputs = self._prepare_aux_debug_outputs(
                        train_sample_raw_aux_outputs,
                        train_sample_input,
                    )

                if optimizer_stepped and is_per_iteration_scheduler:
                    self._update_ema_model(model)
                    scheduler.step()
                elif optimizer_stepped:
                    self._update_ema_model(model)

                if self.startup_timing and self._first_optimizer_timing_recorded and not self._startup_timing_logged:
                    self._flush_startup_timing(prefix="startup_and_first_step")

                if pbar is not None:
                    loss_str = " | ".join([f"{t}: {np.mean(epoch_losses[t][-100:]):.4f}"
                                           for t in epoch_losses.keys() if len(epoch_losses[t]) > 0])
                    pbar.set_postfix_str(loss_str)
                    pbar.update(1)

                current_lr = optimizer.param_groups[0]['lr']

                if self.mgr.wandb_project and (not self.is_distributed or self.rank == 0):
                    metrics = self._prepare_metrics_for_logging(
                        epoch=epoch,
                        step=global_step,
                        epoch_losses=epoch_losses,
                        current_lr=current_lr
                    )
                    import wandb
                    wandb.log(metrics)

                del data_dict, inputs, targets_dict, outputs

            if pbar is not None:
                pbar.close()

            self._report_epoch_aug_timers(epoch)

            if not is_per_iteration_scheduler and not step_scheduler_at_epoch_begin:
                scheduler.step()

            # Report the effective learning rate(s) after all scheduler updates for this epoch.
            current_lrs = [group['lr'] for group in optimizer.param_groups]

            if not self.is_distributed or self.rank == 0:
                print(f"\n[Train] Epoch {epoch + 1} completed.")
                lr_str = ", ".join(f"{lr:.8f}" for lr in current_lrs)
                print(f"  Learning rate(s) = {lr_str}")
                for t_name in self.mgr.targets:
                    avg_loss = np.mean(epoch_losses[t_name]) if epoch_losses[t_name] else 0
                    print(f"  {t_name}: Avg Loss = {avg_loss:.4f}")

                # DEBUG: Print average raw loss values per loss function
                if hasattr(self, '_debug_raw_losses') and self._debug_raw_losses:
                    print("  [DEBUG] Raw loss averages:")
                    for key, data in self._debug_raw_losses.items():
                        avg_raw = np.mean(data['values']) if data['values'] else 0
                        print(f"    {key}: raw={avg_raw:.6f}, weight={data['weight']}, weighted={avg_raw * data['weight']:.6f}")
                    self._debug_raw_losses = {}  # Reset for next epoch

            # ---- validation ----- #
            val_every_n = int(getattr(self.mgr, 'val_every_n', 1))
            do_validate = ((epoch + 1) % max(1, val_every_n) == 0)
            if do_validate and (not self.is_distributed or self.rank == 0):
                validation_model = self._get_validation_model(model)
                # For MAE training, don't set to eval mode to keep patch dropping active
                if not hasattr(self, '_is_mae_training'):
                    model.eval()
                if validation_model is not model or not hasattr(self, '_is_mae_training'):
                    validation_model.eval()
                with torch.no_grad():
                    val_losses = {t_name: [] for t_name in self.mgr.targets}
                    debug_preview_image = None
                    debug_guide_preview_image = None
                    debug_preview_fallback = None
                    debug_preview_candidates = []
                    
                    # Initialize evaluation metrics
                    evaluation_metrics = self._initialize_evaluation_metrics()

                    val_dataloader_iter = iter(val_dataloader)

                    if hasattr(self.mgr, 'max_val_steps_per_epoch') and self.mgr.max_val_steps_per_epoch is not None and self.mgr.max_val_steps_per_epoch > 0:
                        num_val_iters = min(len(val_indices), self.mgr.max_val_steps_per_epoch)
                    else:
                        num_val_iters = len(val_indices)

                    val_pbar = tqdm(range(num_val_iters), desc=f'Validation {epoch + 1}')

                    for i in val_pbar:
                        try:
                            data_dict = next(val_dataloader_iter)
                        except StopIteration:
                            val_dataloader_iter = iter(val_dataloader)
                            data_dict = next(val_dataloader_iter)

                        task_losses, inputs, targets_dict, outputs = self._validation_step(
                            model=validation_model,
                            data_dict=data_dict,
                            loss_fns=loss_fns,
                            use_amp=use_amp
                        )

                        for t_name, loss_value in task_losses.items():
                            # Ensure we have a slot for dynamically introduced tasks (e.g., 'mae')
                            if t_name not in val_losses:
                                val_losses[t_name] = []
                            val_losses[t_name].append(loss_value)
                        
                        # Compute evaluation metrics for each task (handle deep supervision lists)
                        for t_name in self.mgr.targets:
                            if t_name in outputs and t_name in targets_dict:
                                pred_val = outputs[t_name]
                                gt_val = targets_dict[t_name]
                                if isinstance(pred_val, (list, tuple)):
                                    pred_val = pred_val[0]
                                if isinstance(gt_val, (list, tuple)):
                                    gt_val = gt_val[0]
                                # If no metrics configured for this task (e.g., MAE), skip safely
                                mask_tensor = targets_dict.get(f"{t_name}_mask")
                                if isinstance(mask_tensor, (list, tuple)):
                                    mask_tensor = mask_tensor[0]
                                for metric in evaluation_metrics.get(t_name, []):
                                    if isinstance(metric, CriticalComponentsMetric) and i >= 10:
                                        continue
                                    metric.update(pred=pred_val, gt=gt_val, mask=mask_tensor)

                        if debug_preview_image is None:
                                b_idx, found_non_zero = self._select_debug_sample_index_from_targets(targets_dict)
                                snapshot = self._capture_validation_debug_snapshot(
                                    inputs,
                                    targets_dict,
                                    outputs,
                                    getattr(self, "_current_aux_outputs", {}),
                                    b_idx,
                                )

                                if i == 0:
                                    debug_preview_fallback = snapshot

                                if found_non_zero:
                                    debug_preview_candidates.append(snapshot)

                        loss_str = " | ".join([f"{t}: {np.mean(val_losses[t]):.4f}"
                                               for t in self.mgr.targets if len(val_losses[t]) > 0])
                        val_pbar.set_postfix_str(loss_str)

                        del outputs, inputs, targets_dict

                    selected_debug_snapshot = self._select_validation_debug_snapshot_for_epoch(
                        debug_preview_candidates,
                        debug_preview_fallback,
                        epoch,
                    )

                    if debug_preview_image is None and selected_debug_snapshot is not None:
                        inputs_first = selected_debug_snapshot["input"]
                        targets_dict_first_all = selected_debug_snapshot["targets_all"]
                        outputs_dict_first = selected_debug_snapshot["outputs"]
                        raw_aux_outputs_dict_first = selected_debug_snapshot["raw_aux_outputs"]
                        aux_outputs_dict_first = self._prepare_aux_debug_outputs(
                            raw_aux_outputs_dict_first,
                            inputs_first,
                        )
                        debug_img_path = f"{ckpt_dir}/{self.mgr.model_name}_debug_epoch{epoch + 1}.gif"
                        guide_debug_img_path = f"{ckpt_dir}/{self.mgr.model_name}_guide_epoch{epoch + 1}.png"
                        skeleton_dict = None
                        train_skeleton_dict = None
                        if 'skel' in targets_dict_first_all:
                            skeleton_dict = {'segmentation': targets_dict_first_all.get('skel')}
                        if 'train_sample_targets_all' in locals() and train_sample_targets_all and 'skel' in train_sample_targets_all:
                            train_skeleton_dict = {'segmentation': train_sample_targets_all.get('skel')}
                        targets_dict_first = {}
                        for t_name, t_tensor in targets_dict_first_all.items():
                            if t_name not in ['skel', 'is_unlabeled']:
                                targets_dict_first[t_name] = t_tensor
                        save_debug_media = bool(getattr(self.mgr, 'save_gifs', True))
                        unlabeled_input = getattr(self, '_debug_unlabeled_input', None)
                        unlabeled_pseudo = getattr(self, '_debug_unlabeled_pseudo_label', None)
                        unlabeled_pred = getattr(self, '_debug_unlabeled_student_pred', None)
                        _, debug_preview_image = save_debug(
                            input_volume=inputs_first,
                            targets_dict=targets_dict_first,
                            outputs_dict=outputs_dict_first,
                            aux_outputs_dict=aux_outputs_dict_first,
                            tasks_dict=self.mgr.targets,
                            epoch=epoch,
                            save_path=debug_img_path,
                            train_input=train_sample_input,
                            train_targets_dict=train_sample_targets,
                            train_outputs_dict=train_sample_outputs,
                            train_aux_outputs_dict=train_sample_aux_outputs,
                            skeleton_dict=skeleton_dict,
                            train_skeleton_dict=train_skeleton_dict,
                            unlabeled_input=unlabeled_input,
                            unlabeled_pseudo_dict=unlabeled_pseudo,
                            unlabeled_outputs_dict=unlabeled_pred,
                            save_media=save_debug_media,
                        )
                        debug_guide_preview_image = self._make_feature_encoder_guide_preview_image(
                            raw_aux_outputs_dict_first,
                            train_aux_outputs_dict=train_sample_raw_aux_outputs,
                        )
                        if save_debug_media and debug_guide_preview_image is not None:
                            self._save_preview_image(debug_guide_preview_image, guide_debug_img_path)
                        if save_debug_media:
                            debug_gif_history.append((epoch, debug_img_path))

                    print(f"\n[Validation] Epoch {epoch + 1} summary:")
                    total_val_loss = 0.0
                    for t_name in self.mgr.targets:
                        val_avg = np.mean(val_losses[t_name]) if val_losses[t_name] else 0
                        print(f"  Task '{t_name}': Avg validation loss = {val_avg:.4f}")
                        total_val_loss += val_avg

                    avg_val_loss = total_val_loss / len(self.mgr.targets) if self.mgr.targets else 0
                    val_loss_history[epoch] = avg_val_loss
                    
                    print("\n[Validation Metrics]")
                    metric_results = {}
                    for t_name in self.mgr.targets:
                        if t_name in evaluation_metrics:
                            print(f"  Task '{t_name}':")
                            for metric in evaluation_metrics[t_name]:
                                aggregated = metric.aggregate()
                                for metric_name, value in aggregated.items():
                                    full_metric_name = f"{t_name}_{metric_name}"
                                    metric_results[full_metric_name] = value
                                    display_name = f"{metric.name}_{metric_name}"
                                    print(f"    {display_name}: {value:.4f}")

                    if self.mgr.wandb_project:
                        val_metrics = {"epoch": epoch, "step": global_step}
                        for t_name in self.mgr.targets:
                            if t_name in val_losses and len(val_losses[t_name]) > 0:
                                val_metrics[f"val_loss_{t_name}"] = np.mean(val_losses[t_name])
                        val_metrics["val_loss_total"] = avg_val_loss
                        
                        # Add evaluation metrics to wandb
                        for metric_name, value in metric_results.items():
                            val_metrics[f"val_{metric_name}"] = value

                        import wandb

                        media_payload = self._build_debug_media_payload(
                            debug_preview_image=debug_preview_image,
                            guide_preview_image=debug_guide_preview_image,
                        )
                        val_metrics = self._attach_debug_media_to_wandb_metrics(
                            val_metrics,
                            media_payload,
                            wandb,
                        )

                        wandb.log(val_metrics)

                    # Early stopping check
                    if early_stopping_patience > 0:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            print(f"[Early Stopping] New best validation loss: {best_val_loss:.4f}")
                        else:
                            patience_counter += 1
                            print(f"[Early Stopping] No improvement for {patience_counter}/{early_stopping_patience} epochs")

                        if patience_counter >= early_stopping_patience:
                            print(f"\n[Early Stopping] Validation loss did not improve for {early_stopping_patience} epochs.")
                            print(f"Best validation loss: {best_val_loss:.4f}")
                            print("Stopping training early.")
                            break
                    
                    # Handle epoch end operations (checkpointing, cleanup)
                    checkpoint_history, best_checkpoints, ckpt_path = self._on_epoch_end(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_dataset=train_dataset,
                        ckpt_dir=ckpt_dir,
                        model_ckpt_dir=model_ckpt_dir,
                        checkpoint_history=checkpoint_history,
                        best_checkpoints=best_checkpoints,
                        avg_val_loss=avg_val_loss
                    )

                    # Manage debug videos
                    if getattr(self.mgr, 'save_gifs', True) and epoch in [e for e, _ in debug_gif_history]:
                        debug_gif_history, best_debug_gifs = manage_debug_gifs(
                            debug_gif_history=debug_gif_history,
                            best_debug_gifs=best_debug_gifs,
                            epoch=epoch,
                            gif_path=next(p for e, p in debug_gif_history if e == epoch),
                            validation_loss=avg_val_loss,
                            checkpoint_dir=ckpt_dir,
                            model_name=self.mgr.model_name,
                            max_recent=3,
                            max_best=2
                        )

        # Synchronize all ranks before finalization
        if self.is_distributed:
            dist.barrier()

        if not self.is_distributed or self.rank == 0:
            print('Training Finished!')

            final_model_path = save_final_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                max_epoch=self.mgr.max_epoch,
                model_ckpt_dir=model_ckpt_dir,
                model_name=self.mgr.model_name,
                model_config=getattr(model, 'final_config', None),
                train_dataset=train_dataset,
                additional_data=self._get_additional_checkpoint_data(),
            )

        # Clean up DDP process group
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

def main():
    from vesuvius.models.training.cli import main as cli_main
    cli_main()


if __name__ == '__main__':
    import multiprocessing
    import sys

    if len(sys.argv) > 1:
        # Quick check for S3 paths in command line args
        if any('s3://' in str(arg) for arg in sys.argv) or '--config-path' in sys.argv:
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
    main()
