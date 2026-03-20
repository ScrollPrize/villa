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


_DEFAULT_TASK_NAMES = ("surfaces", "ink")
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


@dataclass(frozen=True)
class TaskSpec:
    name: str
    validation_count: int = 1
    resize_factor: float = 1.0
    supervision_subdir: str | None = None


_TASK_SPECS = {
    "surfaces": TaskSpec("surfaces", validation_count=10, resize_factor=2.0),
    "ink": TaskSpec("ink", validation_count=10, resize_factor=1.0, supervision_subdir="supervision_masks"),
}
_TASK_NAMES = tuple(_TASK_SPECS)


def resolve_eval_tasks(value: Any) -> tuple[str, ...]:
    if value is None:
        return _DEFAULT_TASK_NAMES
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "both"}:
            return _DEFAULT_TASK_NAMES
        tasks = (normalized,)
    elif isinstance(value, Sequence):
        tasks = tuple(str(item).strip().lower() for item in value)
    else:
        raise ValueError(f"Unsupported eval_task value: {value!r}")

    invalid = [task for task in tasks if task not in _TASK_NAMES]
    if invalid:
        expected = ", ".join(("both", *_TASK_NAMES))
        raise ValueError(f"eval_task must be one of {expected}. Got {invalid}")
    return tasks


def resolve_eval_decoder_type(value: Any) -> str:
    normalized = str(value or "simple").strip().lower()
    if normalized not in _DECODER_ALIASES:
        expected = ", ".join(sorted(_DECODER_ALIASES))
        raise ValueError(f"Unknown eval_task_decoder_type {value!r}. Expected one of: {expected}")
    return _DECODER_ALIASES[normalized]


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


def _render_binary_label_with_ignore(array: np.ndarray) -> np.ndarray:
    labels = np.asarray(array, dtype=np.int64)
    rendered = np.zeros(labels.shape, dtype=np.uint8)
    rendered[labels == _IGNORE_LABEL] = 127
    rendered[labels == 1] = 255
    return np.stack([rendered, rendered, rendered], axis=-1)


@dataclass(frozen=True)
class TaskSample:
    name: str
    image_path: Path
    label_path: Path
    supervision_mask_path: Path | None = None


@dataclass(frozen=True)
class TaskChunk:
    sample: TaskSample
    starts: tuple[int, int, int]
    valid_count: int
    foreground_count: int
    background_count: int


class TaskVolumeSet:
    def __init__(
        self,
        task_spec: TaskSpec,
        data_root: Path,
        crop_size: tuple[int, int, int],
    ) -> None:
        self.task_spec = task_spec
        self.task_name = task_spec.name
        self.data_root = Path(data_root)
        self.crop_size = tuple(int(dim) for dim in crop_size)
        self.resize_factor = float(task_spec.resize_factor)
        task_root = self.data_root / self.task_name
        image_dir = task_root / "images"
        label_dir = task_root / "labels"
        supervision_dir = (
            task_root / task_spec.supervision_subdir
            if task_spec.supervision_subdir is not None
            else None
        )
        image_paths = sorted(image_dir.glob("*.tif"))
        if not image_paths:
            raise FileNotFoundError(f"No task-eval images found in {image_dir}")

        label_paths = {path.name: path for path in sorted(label_dir.glob("*.tif"))}
        supervision_paths = (
            {path.name: path for path in sorted(supervision_dir.glob("*.tif"))}
            if supervision_dir is not None
            else {}
        )
        samples: list[TaskSample] = []
        missing_labels: list[str] = []
        missing_supervision: list[str] = []
        for image_path in image_paths:
            label_path = label_paths.get(image_path.name)
            if label_path is None:
                missing_labels.append(image_path.name)
                continue
            supervision_mask_path = None
            if supervision_dir is not None:
                supervision_mask_path = supervision_paths.get(image_path.name)
                if supervision_mask_path is None:
                    missing_supervision.append(image_path.name)
                    continue
            samples.append(
                TaskSample(
                    image_path.stem,
                    image_path,
                    label_path,
                    supervision_mask_path=supervision_mask_path,
                )
            )

        if missing_labels:
            raise FileNotFoundError(
                f"Missing task-eval labels for {self.task_name}: {', '.join(missing_labels[:5])}"
            )
        if missing_supervision:
            raise FileNotFoundError(
                f"Missing task-eval supervision masks for {self.task_name}: "
                f"{', '.join(missing_supervision[:5])}"
            )
        self._cache: dict[tuple[Path, bool], np.ndarray] = {}
        candidate_chunks: list[list[TaskChunk]] = []
        for sample in samples:
            sample_chunks = self._chunk_candidates(sample)
            if sample_chunks:
                candidate_chunks.append(sample_chunks)

        validation_count = int(task_spec.validation_count)
        if len(candidate_chunks) <= validation_count:
            raise ValueError(
                f"Task {self.task_name!r} needs more than {validation_count} eligible chunked samples "
                f"with >=50% background so the first {validation_count} can be validation and the rest training."
            )
        self.validation_chunks = tuple(sample_chunks[0] for sample_chunks in candidate_chunks[:validation_count])
        self.training_chunks = tuple(
            chunk
            for sample_chunks in candidate_chunks[validation_count:]
            for chunk in sample_chunks
        )
        if not self.training_chunks:
            raise ValueError(
                f"Task {self.task_name!r} has no eligible training chunks after filtering for >=50% background."
            )
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

    @staticmethod
    def _apply_supervision_mask(label: np.ndarray, supervision_mask: np.ndarray) -> np.ndarray:
        if label.shape != supervision_mask.shape:
            raise ValueError(
                f"Label/supervision mask shape mismatch: label={label.shape}, "
                f"supervision_mask={supervision_mask.shape}"
            )
        valid_supervision = np.asarray(supervision_mask) > 0
        target = np.full(label.shape, _IGNORE_LABEL, dtype=np.int64)
        target[valid_supervision] = (np.asarray(label)[valid_supervision] > 0).astype(np.int64)
        return target

    def _load_target(self, sample: TaskSample) -> np.ndarray:
        label = self._load_cached(sample.label_path, is_label=True)
        if sample.supervision_mask_path is None:
            return label
        supervision_mask = self._load_cached(sample.supervision_mask_path, is_label=True)
        return self._apply_supervision_mask(label, supervision_mask)

    @staticmethod
    def _chunk_name(chunk: TaskChunk) -> str:
        start_str = ",".join(str(int(value)) for value in chunk.starts)
        return f"{chunk.sample.name}@{start_str}"

    @staticmethod
    def _axis_chunk_starts(dim_size: int, crop_dim: int) -> list[int]:
        if dim_size <= crop_dim:
            return [0]
        max_start = dim_size - crop_dim
        starts = list(range(0, max_start + 1, crop_dim))
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts

    @staticmethod
    def _chunk_counts(label_crop: np.ndarray) -> tuple[int, int, int]:
        valid_mask = np.asarray(label_crop) != _IGNORE_LABEL
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            return 0, 0, 0
        foreground_count = int(((np.asarray(label_crop) == 1) & valid_mask).sum())
        background_count = valid_count - foreground_count
        return valid_count, foreground_count, background_count

    @staticmethod
    def _chunk_sort_key(chunk: TaskChunk) -> tuple[int, int, int, tuple[int, int, int]]:
        return (
            -chunk.foreground_count,
            chunk.background_count - chunk.foreground_count,
            -chunk.valid_count,
            chunk.starts,
        )

    def _chunk_candidates(self, sample: TaskSample) -> list[TaskChunk]:
        label = self._load_target(sample)
        axis_starts = [
            self._axis_chunk_starts(int(dim_size), int(crop_dim))
            for dim_size, crop_dim in zip(label.shape, self.crop_size)
        ]
        chunks: list[TaskChunk] = []
        for z_start in axis_starts[0]:
            for y_start in axis_starts[1]:
                for x_start in axis_starts[2]:
                    starts = (z_start, y_start, x_start)
                    label_crop = self._crop_or_pad(label, starts)
                    valid_count, foreground_count, background_count = self._chunk_counts(label_crop)
                    if valid_count == 0 or foreground_count == 0 or background_count < foreground_count:
                        continue
                    chunks.append(
                        TaskChunk(
                            sample=sample,
                            starts=starts,
                            valid_count=valid_count,
                            foreground_count=foreground_count,
                            background_count=background_count,
                        )
                    )
        chunks.sort(key=self._chunk_sort_key)
        return chunks

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            max_label = 0
            seen_samples: set[TaskSample] = set()
            for chunk in (*self.validation_chunks, *self.training_chunks):
                if chunk.sample in seen_samples:
                    continue
                seen_samples.add(chunk.sample)
                target = self._load_target(chunk.sample)
                valid_target = target[target != _IGNORE_LABEL]
                if valid_target.size:
                    max_label = max(max_label, int(valid_target.max()))
            self._num_classes = max_label + 1
        return self._num_classes

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

    def _materialize_chunk(
        self,
        chunk: TaskChunk,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        image = self._load_cached(chunk.sample.image_path, is_label=False)
        label = self._load_target(chunk.sample)
        if image.shape != label.shape:
            raise ValueError(
                f"Image/label shape mismatch for {chunk.sample.name}: image={image.shape}, label={label.shape}"
            )

        image_crop = self._crop_or_pad(image, chunk.starts).astype(np.float32, copy=False)
        label_crop = self._crop_or_pad(label, chunk.starts).astype(np.int64, copy=False)
        if image_crop.max() > 1.0:
            image_crop = image_crop / 255.0

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_crop[None]))
        label_tensor = torch.from_numpy(np.ascontiguousarray(label_crop))
        return image_tensor, label_tensor, self._chunk_name(chunk)

    def sample_training_crop(self, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, str]:
        chunk_index = int(rng.integers(0, len(self.training_chunks)))
        return self._materialize_chunk(self.training_chunks[chunk_index])

    def validation_crops(self) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        return [self._materialize_chunk(chunk) for chunk in self.validation_chunks]


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
            task_spec = _TASK_SPECS[task_name]
            task_root = self.data_root / task_name
            required_dirs = [task_root / "images", task_root / "labels"]
            if task_spec.supervision_subdir is not None:
                required_dirs.append(task_root / task_spec.supervision_subdir)
            for directory in required_dirs:
                if not directory.exists():
                    return False
                if next(directory.glob("*.tif"), None) is None:
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
                _TASK_SPECS[task_name],
                self.data_root,
                crop_size,
            )
            self._datasets[key] = dataset
        return dataset

    @staticmethod
    def _task_seed(base_seed: int, task_name: str, step: int) -> int:
        task_offset = sum(ord(ch) for ch in task_name)
        return int(base_seed + 1009 * step + task_offset)

    @staticmethod
    def _project_ink_target(target: torch.Tensor) -> torch.Tensor:
        target = target.detach()
        depth_dim = 1 if target.ndim == 4 else 0
        valid_mask = target != _IGNORE_LABEL
        foreground_mask = (target == 1) & valid_mask
        projected_valid = torch.any(valid_mask, dim=depth_dim)
        projected_foreground = torch.any(foreground_mask, dim=depth_dim)
        projected = torch.full(
            projected_valid.shape,
            _IGNORE_LABEL,
            dtype=target.dtype,
            device=target.device,
        )
        projected[projected_valid] = 0
        projected[projected_foreground] = 1
        return projected

    @staticmethod
    def _task_logits(task_name: str, logits: torch.Tensor) -> torch.Tensor:
        logits = logits[:, 0]
        if task_name == "ink":
            return torch.amax(logits, dim=1)
        return logits

    @classmethod
    def _task_target(cls, task_name: str, target: torch.Tensor) -> torch.Tensor:
        if task_name == "ink":
            return cls._project_ink_target(target)
        return target.detach()

    @classmethod
    def _binary_target_and_mask(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = target.detach()
        valid_mask = target != _IGNORE_LABEL
        binary_target = (target == 1).to(dtype=torch.float32)
        return binary_target, valid_mask

    @classmethod
    def _task_loss(cls, task_name: str, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        task_target = cls._task_target(task_name, target)
        task_logits = cls._task_logits(task_name, logits)
        binary_target, valid_mask = cls._binary_target_and_mask(task_target)
        masked_logits = task_logits[valid_mask]
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

    @classmethod
    def _label_preview(cls, task_name: str, label: torch.Tensor) -> np.ndarray:
        if task_name == "ink":
            label_2d = cls._project_ink_target(label).detach().cpu().numpy()
            return _render_binary_label_with_ignore(label_2d)
        label_slice = cls._center_slice_label(label)
        return _colorize_labels(label_slice)

    @classmethod
    def _probability_preview(cls, prediction_probability: torch.Tensor) -> np.ndarray:
        prediction = prediction_probability.detach().cpu()
        if prediction.ndim == 2:
            prediction_slice = prediction.numpy()
        else:
            prediction_slice = cls._center_slice_image(prediction.unsqueeze(0))
        return np.stack([_normalize_image(prediction_slice)] * 3, axis=-1)

    def _save_validation_image(
        self,
        task_name: str,
        step: int,
        rows: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Path:
        canvas_rows: list[np.ndarray] = []
        for image, label, prediction_probability in rows:
            image_slice = self._center_slice_image(image)
            image_rgb = np.stack([_normalize_image(image_slice)] * 3, axis=-1)
            label_rgb = self._label_preview(task_name, label)
            prediction_rgb = self._probability_preview(prediction_probability)
            canvas_rows.append(np.concatenate([image_rgb, label_rgb, prediction_rgb], axis=1))

        canvas = np.concatenate(canvas_rows, axis=0)

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
                    loss = self._task_loss(task_name, logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_value = float(loss.detach().item())
                train_loss_total += loss_value
                progress.set_postfix(loss=f"{loss_value:.4f}")

        model.eval()
        validation_crops = dataset.validation_crops()
        val_loss_total = 0.0
        foreground_dice_total = 0.0
        val_names: list[str] = []
        image_path: Path | None = None
        validation_image_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for val_image, val_target, val_name in validation_crops:
                val_names.append(val_name)
                val_batch = val_image.unsqueeze(0).to(self.device, non_blocking=True)
                val_target_batch = val_target.unsqueeze(0).to(self.device, non_blocking=True)
                val_logits = model(val_batch)
                val_loss = self._task_loss(task_name, val_logits, val_target_batch)
                val_task_logits = self._task_logits(task_name, val_logits)
                val_probability = torch.sigmoid(val_task_logits).detach().cpu()
                val_prediction = (val_task_logits > 0).to(dtype=torch.int64).detach().cpu()
                val_metric_target = self._task_target(task_name, val_target_batch).cpu()
                val_loss_total += float(val_loss.detach().item())
                foreground_dice_total += self._foreground_mean_dice(val_prediction, val_metric_target)
                if self.rank == 0:
                    validation_image_rows.append((val_image, val_target, val_probability[0]))
        if self.rank == 0 and validation_image_rows:
            image_path = self._save_validation_image(
                task_name,
                step,
                validation_image_rows,
            )
        val_count = max(1, len(validation_crops))
        val_loss_mean_local = val_loss_total / val_count
        foreground_dice_local = foreground_dice_total / val_count
        train_loss_mean = train_loss_total / max(1, self.train_iters)
        train_loss_mean, val_loss_mean, foreground_dice = self._distributed_mean(
            train_loss_mean,
            val_loss_mean_local,
            foreground_dice_local,
        )

        return {
            "train_loss_mean": train_loss_mean,
            "val_loss": val_loss_mean,
            "val_fg_dice": foreground_dice,
            "ignore_label": _IGNORE_LABEL,
            "val_sample": ",".join(val_names),
            "val_sample_count": val_count,
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
