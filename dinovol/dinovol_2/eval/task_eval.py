from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tifffile import imread
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from dinovol_2.eval.download_data import download_tasks
from dinovol_2.model.patch_encode_decode import LayerNormNd, PatchDecode


_TASK_NAMES = ("fibers", "surfaces")
_DECODER_ALIASES = {
    "simple": "simple",
    "minimal": "simple",
    "patch_encode_decode": "patch_encode_decode",
    "primus_patch_decode": "patch_encode_decode",
}
_IGNORE_LABEL = 2
_LABEL_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [235, 87, 87],
        [39, 174, 96],
        [47, 128, 237],
        [242, 153, 74],
        [155, 81, 224],
    ],
    dtype=np.uint8,
)


def resolve_eval_tasks(value: Any) -> tuple[str, ...]:
    if value is None:
        return _TASK_NAMES
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "both", "all"}:
            return _TASK_NAMES
        tasks = (normalized,)
    elif isinstance(value, Sequence):
        tasks = tuple(str(item).strip().lower() for item in value)
    else:
        raise ValueError(f"Unsupported eval_task value: {value!r}")

    invalid = [task for task in tasks if task not in _TASK_NAMES]
    if invalid:
        raise ValueError(f"eval_task must be one of both, fibers, or surfaces. Got {invalid}")
    return tasks


def resolve_eval_decoder_type(value: Any) -> str:
    normalized = str(value or "simple").strip().lower()
    if normalized not in _DECODER_ALIASES:
        expected = ", ".join(sorted(_DECODER_ALIASES))
        raise ValueError(f"Unknown eval_task_decoder_type {value!r}. Expected one of: {expected}")
    return _DECODER_ALIASES[normalized]


def resolve_resize_task_data(value: Any) -> float:
    if value is None:
        return 0.0
    factor = float(value)
    if factor < 0.0:
        raise ValueError(f"resize_task_data must be non-negative, got {factor}")
    return factor


def _read_volume(path: Path) -> np.ndarray:
    return np.asarray(imread(path))


def _scaled_shape(shape: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(1, int(round(dim * factor))) for dim in shape)


def _resize_volume(volume: np.ndarray, *, factor: float, is_label: bool) -> np.ndarray:
    if factor in {0.0, 1.0}:
        return np.asarray(volume)

    tensor = torch.from_numpy(np.asarray(volume))
    tensor = tensor[None, None].float()
    size = _scaled_shape(tuple(int(dim) for dim in volume.shape), factor)
    if is_label:
        resized = F.interpolate(tensor, size=size, mode="nearest")
        return resized[0, 0].round().to(dtype=torch.int64).cpu().numpy()
    resized = F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
    return resized[0, 0].cpu().numpy()


def _normalize_image(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array - min_value) / (max_value - min_value)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def _colorize_labels(array: np.ndarray) -> np.ndarray:
    labels = array.astype(np.int64, copy=False)
    if labels.size == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    if int(labels.max()) < len(_LABEL_PALETTE):
        return _LABEL_PALETTE[labels]
    grayscale = _normalize_image(labels.astype(np.float32))
    return np.stack([grayscale, grayscale, grayscale], axis=-1)


@dataclass(frozen=True)
class TaskSample:
    name: str
    image_path: Path
    label_path: Path


class TaskVolumeSet:
    def __init__(
        self,
        task_name: str,
        data_root: Path,
        crop_size: tuple[int, int, int],
        *,
        resize_factor: float = 0.0,
    ) -> None:
        self.task_name = task_name
        self.data_root = Path(data_root)
        self.crop_size = tuple(int(dim) for dim in crop_size)
        self.resize_factor = float(resize_factor)
        task_root = self.data_root / task_name
        image_dir = task_root / "images"
        label_dir = task_root / "labels"
        image_paths = sorted(image_dir.glob("*.tif"))
        if not image_paths:
            raise FileNotFoundError(f"No task-eval images found in {image_dir}")

        label_paths = {path.name: path for path in sorted(label_dir.glob("*.tif"))}
        samples: list[TaskSample] = []
        missing_labels: list[str] = []
        for image_path in image_paths:
            label_path = label_paths.get(image_path.name)
            if label_path is None:
                missing_labels.append(image_path.name)
                continue
            samples.append(TaskSample(image_path.stem, image_path, label_path))

        if missing_labels:
            raise FileNotFoundError(
                f"Missing task-eval labels for {task_name}: {', '.join(missing_labels[:5])}"
            )
        if len(samples) < 2:
            raise ValueError(
                f"Task {task_name!r} needs at least 2 samples so the first can be validation and the rest training."
            )

        self.validation_sample = samples[0]
        self.training_samples = tuple(samples[1:])
        self._cache: dict[tuple[Path, bool], np.ndarray] = {}
        self._num_classes: int | None = None

    def close(self) -> None:
        self._cache.clear()

    def _load_cached(self, path: Path, *, is_label: bool) -> np.ndarray:
        cache_key = (path, is_label)
        cached = self._cache.get(cache_key)
        if cached is None:
            cached = _read_volume(path)
            cached = _resize_volume(cached, factor=self.resize_factor, is_label=is_label)
            if cached.ndim != 3:
                raise ValueError(f"Expected a 3D TIFF at {path}, got shape {cached.shape}")
            self._cache[cache_key] = cached
        return cached

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            max_label = 0
            for sample in (self.validation_sample, *self.training_samples):
                max_label = max(max_label, int(self._load_cached(sample.label_path, is_label=True).max()))
            self._num_classes = max_label + 1
        return self._num_classes

    def _crop_starts(self, shape: tuple[int, int, int], rng: np.random.Generator | None) -> tuple[int, int, int]:
        starts: list[int] = []
        for dim_size, crop_dim in zip(shape, self.crop_size):
            if dim_size <= crop_dim:
                starts.append(0)
            elif rng is None:
                starts.append((dim_size - crop_dim) // 2)
            else:
                starts.append(int(rng.integers(0, dim_size - crop_dim + 1)))
        return tuple(starts)

    def _crop_or_pad(self, volume: np.ndarray, starts: tuple[int, int, int]) -> np.ndarray:
        slices = []
        pad_width = []
        for dim_size, crop_dim, start in zip(volume.shape, self.crop_size, starts):
            if dim_size >= crop_dim:
                slices.append(slice(start, start + crop_dim))
                pad_width.append((0, 0))
            else:
                slices.append(slice(0, dim_size))
                total_pad = crop_dim - dim_size
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_width.append((pad_before, pad_after))

        cropped = np.asarray(volume[tuple(slices)])
        if any(pad_before or pad_after for pad_before, pad_after in pad_width):
            cropped = np.pad(cropped, pad_width=pad_width, mode="constant")
        return cropped

    def _materialize_sample(
        self,
        sample: TaskSample,
        *,
        rng: np.random.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        image = self._load_cached(sample.image_path, is_label=False)
        label = self._load_cached(sample.label_path, is_label=True)
        if image.shape != label.shape:
            raise ValueError(
                f"Image/label shape mismatch for {sample.name}: image={image.shape}, label={label.shape}"
            )

        starts = self._crop_starts(tuple(int(dim) for dim in image.shape), rng)
        image_crop = self._crop_or_pad(image, starts).astype(np.float32, copy=False)
        label_crop = self._crop_or_pad(label, starts).astype(np.int64, copy=False)
        if image_crop.max() > 1.0:
            image_crop = image_crop / 255.0

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_crop[None]))
        label_tensor = torch.from_numpy(np.ascontiguousarray(label_crop))
        return image_tensor, label_tensor, sample.name

    def sample_training_crop(self, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample_index = int(rng.integers(0, len(self.training_samples)))
        return self._materialize_sample(self.training_samples[sample_index], rng=rng)

    def validation_crop(self) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self._materialize_sample(self.validation_sample, rng=None)


class MinimalTaskDecoder(nn.Module):
    def __init__(self, ndim: int, patch_size: tuple[int, int, int], embed_dim: int, num_classes: int) -> None:
        super().__init__()
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        conv_t = nn.ConvTranspose2d if ndim == 2 else nn.ConvTranspose3d
        self.project = conv(embed_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsample = None
        if any(step > 1 for step in patch_size):
            self.upsample = conv_t(
                num_classes,
                num_classes,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEncodeDecodeTaskDecoder(nn.Module):
    def __init__(self, patch_size: tuple[int, int, int], embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.patch_decoder = PatchDecode(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=num_classes,
            norm=LayerNormNd,
            activation=nn.GELU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_decoder(x)


class Dinov2TaskModel(nn.Module):
    def __init__(self, teacher_backbone: nn.Module, num_classes: int, decoder_type: str) -> None:
        super().__init__()
        self.backbone = deepcopy(teacher_backbone)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.patch_size = tuple(int(v) for v in self.backbone.patch_size)
        self.target_spatial = tuple(int(v) for v in self.backbone.global_crops_size)
        self.ndim = len(self.patch_size)
        self.embed_dim = int(self.backbone.embed_dim)
        if decoder_type == "simple":
            self.decoder = MinimalTaskDecoder(self.ndim, self.patch_size, self.embed_dim, num_classes)
        elif decoder_type == "patch_encode_decode":
            self.decoder = PatchEncodeDecodeTaskDecoder(self.patch_size, self.embed_dim, num_classes)
        else:
            raise ValueError(f"Unknown decoder_type {decoder_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x, masks=None, view_kind="global")
        patch_tokens = features["x_norm_patchtokens"]
        patch_grid = tuple(size // patch for size, patch in zip(self.target_spatial, self.patch_size))
        feature_map = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[-1],
            *patch_grid,
        ).contiguous()
        return self.decoder(feature_map)


class TaskEvalRunner:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        output_dir: Path,
        device: torch.device,
        use_amp: bool,
    ) -> None:
        self.config = dict(config)
        self.output_dir = Path(output_dir) / "task_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.tasks = resolve_eval_tasks(self.config.get("eval_task", "both"))
        self.train_iters = int(self.config.get("eval_task_train_iters", 500))
        if self.train_iters < 0:
            raise ValueError(f"eval_task_train_iters must be non-negative, got {self.train_iters}")
        self.decoder_type = resolve_eval_decoder_type(
            self.config.get("eval_task_decoder_type", self.config.get("eval_decoder_type", "simple"))
        )
        self.learning_rate = float(self.config.get("eval_task_lr", 1e-4))
        self.weight_decay = float(self.config.get("eval_task_weight_decay", 0.0))
        self.seed = int(self.config.get("eval_task_seed", 0))
        self.resize_task_data = resolve_resize_task_data(self.config.get("resize_task_data", 0))
        self.download_timeout_s = float(self.config.get("eval_task_download_timeout_s", 3600.0))
        if self.download_timeout_s <= 0:
            raise ValueError(f"eval_task_download_timeout_s must be positive, got {self.download_timeout_s}")
        self.data_root = Path(
            self.config.get("eval_task_data_root", Path(__file__).resolve().parent / "data")
        )
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._downloaded = False
        self._datasets: dict[tuple[str, tuple[int, int, int]], TaskVolumeSet] = {}

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.close()
        self._datasets.clear()

    def _download_sentinel_path(self) -> Path:
        task_key = "_".join(sorted(self.tasks))
        return self.data_root / f".task_eval_ready_{task_key}"

    def _task_data_ready(self) -> bool:
        for task_name in self.tasks:
            task_root = self.data_root / task_name
            image_dir = task_root / "images"
            label_dir = task_root / "labels"
            if not image_dir.exists() or not label_dir.exists():
                return False
            if next(image_dir.glob("*.tif"), None) is None:
                return False
            if next(label_dir.glob("*.tif"), None) is None:
                return False
        return True

    def _ensure_data(self) -> None:
        if self._downloaded:
            return
        if self._task_data_ready():
            self._downloaded = True
            return

        sentinel_path = self._download_sentinel_path()
        if self.world_size <= 1:
            download_tasks(self.tasks, data_root=self.data_root)
        elif self.rank == 0:
            sentinel_path.unlink(missing_ok=True)
            download_tasks(self.tasks, data_root=self.data_root)
            sentinel_path.write_text("ready\n", encoding="utf-8")
        else:
            deadline = time.monotonic() + self.download_timeout_s
            while True:
                if self._task_data_ready() and sentinel_path.exists():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {self.download_timeout_s:.0f}s waiting for task-eval data in {self.data_root}"
                    )
                time.sleep(1.0)
        self._downloaded = True

    def _dataset_for(self, task_name: str, crop_size: tuple[int, int, int]) -> TaskVolumeSet:
        key = (task_name, crop_size)
        dataset = self._datasets.get(key)
        if dataset is None:
            dataset = TaskVolumeSet(
                task_name,
                self.data_root,
                crop_size,
                resize_factor=self.resize_task_data,
            )
            self._datasets[key] = dataset
        return dataset

    @staticmethod
    def _task_seed(base_seed: int, task_name: str, step: int) -> int:
        task_offset = sum(ord(ch) for ch in task_name)
        return int(base_seed + 1009 * step + task_offset)

    @staticmethod
    def _binary_target_and_mask(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = target.detach()
        valid_mask = target != _IGNORE_LABEL
        binary_target = (target == 1).to(dtype=torch.float32)
        return binary_target, valid_mask

    @staticmethod
    def _task_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        binary_target, valid_mask = TaskEvalRunner._binary_target_and_mask(target)
        masked_logits = logits[:, 0][valid_mask]
        masked_target = binary_target[valid_mask]
        if masked_target.numel() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(masked_logits, masked_target)

    @staticmethod
    def _ddp_kwargs(device: torch.device) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            kwargs["device_ids"] = [device.index]
            kwargs["output_device"] = device.index
        return kwargs

    def _distributed_mean(self, *values: float) -> tuple[float, ...]:
        if self.world_size <= 1:
            return tuple(float(value) for value in values)
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tuple(float(value.item()) for value in tensor)

    @staticmethod
    def _foreground_mean_dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
        prediction = prediction.detach()
        target = target.detach()
        valid_mask = target != _IGNORE_LABEL
        pred_mask = (prediction == 1) & valid_mask
        target_mask = (target == 1) & valid_mask
        denom = int(pred_mask.sum().item() + target_mask.sum().item())
        if denom == 0:
            return 1.0
        intersection = int((pred_mask & target_mask).sum().item())
        return (2.0 * intersection) / denom

    @staticmethod
    def _center_slice_image(image: torch.Tensor) -> np.ndarray:
        volume = image.detach().cpu().float()
        depth_index = volume.shape[1] // 2
        return volume[0, depth_index].numpy()

    @staticmethod
    def _center_slice_label(label: torch.Tensor) -> np.ndarray:
        volume = label.detach().cpu()
        depth_index = volume.shape[0] // 2
        return volume[depth_index].numpy()

    def _save_validation_image(
        self,
        task_name: str,
        step: int,
        image: torch.Tensor,
        label: torch.Tensor,
        prediction_probability: torch.Tensor,
    ) -> Path:
        image_slice = self._center_slice_image(image)
        label_slice = self._center_slice_label(label)
        prediction_slice = self._center_slice_image(prediction_probability.unsqueeze(0))

        image_rgb = np.stack([_normalize_image(image_slice)] * 3, axis=-1)
        label_rgb = _colorize_labels(label_slice)
        prediction_rgb = np.stack([_normalize_image(prediction_slice)] * 3, axis=-1)
        canvas = np.concatenate([image_rgb, label_rgb, prediction_rgb], axis=1)

        path = self.output_dir / f"{task_name}_step_{step:06d}.png"
        Image.fromarray(canvas).save(path)
        return path

    def _run_single_task(
        self,
        *,
        task_name: str,
        dataset: TaskVolumeSet,
        teacher_backbone: nn.Module,
        step: int,
    ) -> dict[str, Any]:
        task_seed = self._task_seed(self.seed, task_name, step)
        torch.manual_seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)
        rng = np.random.default_rng(task_seed + 1000003 * self.rank)
        model = Dinov2TaskModel(teacher_backbone, 1, self.decoder_type).to(self.device)
        ddp_model: nn.Module = model
        if self.world_size > 1:
            ddp_model = DDP(model, **self._ddp_kwargs(self.device))
        optimizer = torch.optim.AdamW(
            model.decoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        train_loss_total = 0.0
        ddp_model.train()
        model.backbone.eval()
        with tqdm(
            range(self.train_iters),
            desc=f"task_eval/{task_name} step={step}",
            unit="iter",
            leave=False,
            disable=self.train_iters <= 0 or self.rank != 0,
        ) as progress:
            for _ in progress:
                image, target, _ = dataset.sample_training_crop(rng)
                image = image.unsqueeze(0).to(self.device, non_blocking=True)
                target = target.unsqueeze(0).to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = ddp_model(image)
                    loss = self._task_loss(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_value = float(loss.detach().item())
                train_loss_total += loss_value
                progress.set_postfix(loss=f"{loss_value:.4f}")

        val_image, val_target, val_name = dataset.validation_crop()
        val_batch = val_image.unsqueeze(0).to(self.device, non_blocking=True)
        val_target_batch = val_target.unsqueeze(0).to(self.device, non_blocking=True)

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            val_logits = model(val_batch)
            val_loss = self._task_loss(val_logits, val_target_batch)
        val_probability = torch.sigmoid(val_logits[:, 0]).detach().cpu()
        val_prediction = (val_logits[:, 0] > 0).to(dtype=torch.int64).detach().cpu()
        image_path = (
            self._save_validation_image(
                task_name,
                step,
                val_image,
                val_target,
                val_probability[0],
            )
            if self.rank == 0
            else None
        )
        foreground_dice = self._foreground_mean_dice(val_prediction, val_target_batch.cpu())
        train_loss_mean = train_loss_total / max(1, self.train_iters)
        train_loss_mean, val_loss_mean, foreground_dice = self._distributed_mean(
            train_loss_mean,
            float(val_loss.detach().item()),
            foreground_dice,
        )

        return {
            "train_loss_mean": train_loss_mean,
            "val_loss": val_loss_mean,
            "val_fg_dice": foreground_dice,
            "ignore_label": _IGNORE_LABEL,
            "val_sample": val_name,
            "image_path": image_path,
        }

    def run(self, *, teacher_backbone: nn.Module, step: int) -> dict[str, dict[str, Any]]:
        self._ensure_data()
        crop_size = tuple(int(v) for v in teacher_backbone.global_crops_size)
        results: dict[str, dict[str, Any]] = {}
        for task_name in self.tasks:
            dataset = self._dataset_for(task_name, crop_size)
            try:
                results[task_name] = self._run_single_task(
                    task_name=task_name,
                    dataset=dataset,
                    teacher_backbone=teacher_backbone,
                    step=step,
                )
            except Exception:
                print(f"task_eval failed for {task_name} at step={step} rank={self.rank}")
                traceback.print_exc()
        return results
