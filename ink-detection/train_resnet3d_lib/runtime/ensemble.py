import json
import math
import os
import os.path as osp
import re
from typing import Dict, Iterable, List, Sequence

import torch

from train_resnet3d_lib.runtime.checkpointing import load_state_dict_from_checkpoint


EPOCH_CHECKPOINT_PATTERN = re.compile(r"^epochepoch=(\d+)\.ckpt$")
AVG_MODE_EQUAL = "equal"
AVG_MODE_FAKE_EMA_POWER = "fake_ema_power"
SUPPORTED_AVG_MODES = (AVG_MODE_EQUAL, AVG_MODE_FAKE_EMA_POWER)


def resolve_training_run_dir(training_run_dir: str) -> str:
    run_dir = osp.expanduser(str(training_run_dir))
    if not osp.isabs(run_dir):
        run_dir = osp.abspath(run_dir)
    if not osp.isdir(run_dir):
        raise FileNotFoundError(f"training run directory not found: {run_dir}")
    return run_dir


def resolve_output_dir(training_run_dir: str, output_dir: str | None) -> str:
    if output_dir is None:
        training_run_dir_base = training_run_dir.rstrip("/\\")
        return f"{training_run_dir_base}_ensembled"
    resolved = osp.expanduser(str(output_dir))
    if not osp.isabs(resolved):
        resolved = osp.abspath(resolved)
    return resolved


def parse_epochs_csv(epoch_csv: str) -> List[int]:
    if not isinstance(epoch_csv, str):
        raise TypeError(f"epochs csv must be a string, got {type(epoch_csv).__name__}")
    epochs: List[int] = []
    for raw_value in [part.strip() for part in epoch_csv.split(",")]:
        if not raw_value:
            continue
        if not raw_value.isdigit():
            raise ValueError(f"epochs csv contains non-integer value: {raw_value!r}")
        epochs.append(int(raw_value))
    if not epochs:
        raise ValueError("epochs csv must contain at least one integer epoch")
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"epochs csv contains duplicates: {epochs!r}")
    return epochs


def discover_epoch_checkpoints(training_run_dir: str) -> Dict[int, str]:
    checkpoints_dir = osp.join(training_run_dir, "checkpoints")
    if not osp.isdir(checkpoints_dir):
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoints_dir}")
    checkpoints_by_epoch: Dict[int, str] = {}
    for filename in sorted(os.listdir(checkpoints_dir)):
        match = EPOCH_CHECKPOINT_PATTERN.fullmatch(filename)
        if match is None:
            continue
        epoch = int(match.group(1))
        checkpoint_path = osp.join(checkpoints_dir, filename)
        if epoch in checkpoints_by_epoch:
            raise ValueError(
                f"duplicate epoch checkpoint detected for epoch={epoch}: "
                f"{checkpoints_by_epoch[epoch]} and {checkpoint_path}"
            )
        checkpoints_by_epoch[epoch] = checkpoint_path
    if not checkpoints_by_epoch:
        raise FileNotFoundError(
            f"no epoch checkpoints found under {checkpoints_dir}; "
            "expected files matching epochepoch=<N>.ckpt"
        )
    return checkpoints_by_epoch


def _validate_available_epochs(available_epochs: Sequence[int]) -> List[int]:
    epochs = sorted(int(epoch) for epoch in available_epochs)
    if not epochs:
        raise ValueError("available_epochs must contain at least one epoch")
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"available_epochs contains duplicates: {epochs!r}")
    return epochs


def _evenly_spaced_indices(length: int, count: int) -> List[int]:
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    if count > length:
        raise ValueError(f"count must be <= length, got count={count} length={length}")
    if count == 1:
        return [length - 1]

    indices: List[int] = []
    last_idx = -1
    for i in range(count):
        raw = int(round(i * (length - 1) / (count - 1)))
        min_allowed = last_idx + 1
        max_allowed = length - (count - i)
        idx = min(max(raw, min_allowed), max_allowed)
        indices.append(idx)
        last_idx = idx
    return indices


def select_auto_epochs(
    available_epochs: Sequence[int],
    *,
    target_count: int = 10,
    dense_prefix: int = 4,
) -> List[int]:
    epochs = _validate_available_epochs(available_epochs)
    target_count = int(target_count)
    dense_prefix = int(dense_prefix)
    if target_count <= 0:
        raise ValueError(f"target_count must be > 0, got {target_count}")
    if dense_prefix < 0:
        raise ValueError(f"dense_prefix must be >= 0, got {dense_prefix}")

    if len(epochs) <= target_count:
        return epochs
    if target_count == 1:
        return [epochs[-1]]

    dense_count = min(dense_prefix, target_count, len(epochs))
    if dense_count == target_count:
        selected = epochs[:target_count]
        selected[-1] = epochs[-1]
        return selected

    dense_epochs = epochs[:dense_count]
    remaining_pool = epochs[dense_count:]
    remaining_needed = target_count - dense_count
    if remaining_needed > len(remaining_pool):
        remaining_needed = len(remaining_pool)
    tail_indices = _evenly_spaced_indices(len(remaining_pool), remaining_needed)
    tail_epochs = [remaining_pool[idx] for idx in tail_indices]

    selected = dense_epochs + tail_epochs
    if epochs[-1] not in selected:
        selected[-1] = epochs[-1]
    return selected


def resolve_selected_epochs(
    available_epochs: Sequence[int],
    *,
    epochs_csv: str | None = None,
    target_count: int = 10,
    dense_prefix: int = 4,
) -> List[int]:
    epochs = _validate_available_epochs(available_epochs)
    if epochs_csv is not None:
        selected_epochs = parse_epochs_csv(epochs_csv)
        missing = [epoch for epoch in selected_epochs if epoch not in set(epochs)]
        if missing:
            raise FileNotFoundError(
                f"requested epochs not found in run checkpoints: {missing}; "
                f"available epochs: {epochs}"
            )
        return selected_epochs
    return select_auto_epochs(
        epochs,
        target_count=target_count,
        dense_prefix=dense_prefix,
    )


def resolve_checkpoint_paths_for_epochs(
    checkpoints_by_epoch: Dict[int, str],
    selected_epochs: Sequence[int],
) -> List[str]:
    paths: List[str] = []
    for epoch in selected_epochs:
        epoch_i = int(epoch)
        if epoch_i not in checkpoints_by_epoch:
            raise FileNotFoundError(f"missing checkpoint for epoch {epoch_i}")
        checkpoint_path = checkpoints_by_epoch[epoch_i]
        if not osp.isfile(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        paths.append(checkpoint_path)
    return paths


def std_to_exp(std: float) -> float:
    std_value = float(std)
    if not (0.0 < std_value < 0.289):
        raise ValueError("sigma_rel must be > 0 and < 0.289")

    def _exp_to_std(exp):
        exp = float(exp)
        return math.sqrt((exp + 1.0) / (((exp + 2.0) ** 2) * (exp + 3.0)))

    lo = 0.0
    hi = 1.0
    while _exp_to_std(hi) > std_value:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError(f"failed to bracket std_to_exp root for sigma_rel={std_value}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _exp_to_std(mid) > std_value:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_fake_ema_power_weights(selected_epochs: Sequence[int], *, sigma_rel: float) -> List[float]:
    epochs = _validate_available_epochs(selected_epochs)
    exp = std_to_exp(float(sigma_rel))
    times = [float(epoch) + 1.0 for epoch in epochs]
    t_end = float(times[-1])
    if t_end <= 0.0:
        raise ValueError(f"invalid final epoch time: {t_end}")

    kernel = [((time / t_end) ** exp) for time in times]
    if len(times) > 1:
        edges = [0.0] * (len(times) + 1)
        edges[0] = 0.0
        edges[-1] = t_end
        for idx in range(1, len(times)):
            edges[idx] = 0.5 * (times[idx - 1] + times[idx])
        kernel = [value * (edges[idx + 1] - edges[idx]) for idx, value in enumerate(kernel)]
    kernel_sum = float(sum(kernel))
    if not math.isfinite(kernel_sum) or kernel_sum <= 0.0:
        raise ValueError("failed to compute fake EMA weights from selected epochs")
    return [float(weight / kernel_sum) for weight in kernel]


def compute_averaging_weights(
    selected_epochs: Sequence[int],
    *,
    avg_mode: str,
    sigma_rel: float = 0.10,
) -> List[float]:
    avg_mode_text = str(avg_mode).strip().lower()
    count = len(selected_epochs)
    if count <= 0:
        raise ValueError("selected_epochs must contain at least one epoch")
    if avg_mode_text == AVG_MODE_EQUAL:
        return [1.0 / float(count)] * count
    if avg_mode_text == AVG_MODE_FAKE_EMA_POWER:
        return compute_fake_ema_power_weights(selected_epochs, sigma_rel=float(sigma_rel))
    raise ValueError(f"unsupported avg_mode={avg_mode!r}; expected one of {SUPPORTED_AVG_MODES!r}")


def _normalized_weights(weights: Sequence[float], *, expected_count: int) -> List[float]:
    if len(weights) != int(expected_count):
        raise ValueError(f"weights length mismatch: expected={expected_count} got={len(weights)}")
    normalized = [float(weight) for weight in weights]
    if any(weight < 0.0 for weight in normalized):
        raise ValueError(f"weights must be >= 0, got {normalized!r}")
    total = float(sum(normalized))
    if total <= 0.0:
        raise ValueError(f"sum(weights) must be > 0, got {total}")
    return [weight / total for weight in normalized]


def average_checkpoints(
    checkpoint_paths: Sequence[str],
    *,
    weights: Sequence[float] | None = None,
) -> Dict[str, torch.Tensor]:
    ckpt_paths = [str(path) for path in checkpoint_paths]
    if not ckpt_paths:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")

    state_dicts = [load_state_dict_from_checkpoint(path) for path in ckpt_paths]
    key_order = list(state_dicts[0].keys())
    for idx, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict.keys()) != set(key_order):
            missing_keys = sorted(set(key_order) - set(state_dict.keys()))
            extra_keys = sorted(set(state_dict.keys()) - set(key_order))
            raise ValueError(
                f"state dict key mismatch at checkpoint index={idx} path={ckpt_paths[idx]!r}; "
                f"missing={missing_keys!r} extra={extra_keys!r}"
            )

    if weights is None:
        normalized_weights = [1.0 / float(len(state_dicts))] * len(state_dicts)
    else:
        normalized_weights = _normalized_weights(weights, expected_count=len(state_dicts))

    averaged: Dict[str, torch.Tensor] = {}
    for key in key_order:
        first_tensor = state_dicts[0][key]
        if not isinstance(first_tensor, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] is not a tensor in first checkpoint")

        for idx, state_dict in enumerate(state_dicts[1:], start=1):
            tensor = state_dict[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"state_dict[{key!r}] is not a tensor in checkpoint index={idx} path={ckpt_paths[idx]!r}"
                )
            if tuple(tensor.shape) != tuple(first_tensor.shape):
                raise ValueError(
                    f"shape mismatch for key={key!r} at checkpoint index={idx} path={ckpt_paths[idx]!r}; "
                    f"expected={tuple(first_tensor.shape)!r} got={tuple(tensor.shape)!r}"
                )

        if bool(first_tensor.is_floating_point()):
            accumulator = first_tensor.detach().to(device="cpu", dtype=torch.float64) * float(
                normalized_weights[0]
            )
            for idx, state_dict in enumerate(state_dicts[1:], start=1):
                tensor = state_dict[key]
                if not bool(tensor.is_floating_point()):
                    raise ValueError(
                        f"dtype mismatch for key={key!r}: first checkpoint is floating "
                        f"({first_tensor.dtype}) but checkpoint index={idx} dtype={tensor.dtype}"
                    )
                accumulator.add_(
                    tensor.detach().to(device="cpu", dtype=torch.float64),
                    alpha=float(normalized_weights[idx]),
                )
            averaged[key] = accumulator.to(dtype=first_tensor.dtype)
        else:
            # BatchNorm running-step counters are expected to differ between
            # checkpoints; keep the latest value instead of enforcing equality.
            if str(key).endswith("num_batches_tracked"):
                averaged[key] = state_dicts[-1][key].detach().to(device="cpu").clone()
                continue

            first_value = first_tensor.detach().to(device="cpu")
            for idx, state_dict in enumerate(state_dicts[1:], start=1):
                tensor = state_dict[key]
                if tensor.dtype != first_tensor.dtype:
                    raise ValueError(
                        f"dtype mismatch for non-floating key={key!r} at checkpoint index={idx} "
                        f"path={ckpt_paths[idx]!r}: expected {first_tensor.dtype}, got {tensor.dtype}"
                    )
                if not torch.equal(tensor.detach().to(device="cpu"), first_value):
                    raise ValueError(
                        f"non-floating tensor mismatch for key={key!r} at checkpoint index={idx} "
                        f"path={ckpt_paths[idx]!r}"
                    )
            averaged[key] = first_value.clone()
    return averaged


def default_ensemble_checkpoint_filename(*, avg_mode: str) -> str:
    avg_mode_text = str(avg_mode).strip().lower()
    if avg_mode_text == AVG_MODE_EQUAL:
        return "ensemble_equal_avg.pt"
    return f"ensemble_{avg_mode_text}.pt"


def write_ensemble_checkpoint(output_dir: str, state_dict: Dict[str, torch.Tensor], *, avg_mode: str) -> str:
    checkpoints_dir = osp.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    filename = default_ensemble_checkpoint_filename(avg_mode=avg_mode)
    checkpoint_path = osp.join(checkpoints_dir, filename)
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def write_manifest(output_dir: str, manifest: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = osp.join(output_dir, "ensemble_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def copy_metadata_snapshot(metadata_path: str, output_dir: str) -> str:
    if not metadata_path:
        raise ValueError("metadata_path must be non-empty")
    src = osp.abspath(osp.expanduser(str(metadata_path)))
    if not osp.isfile(src):
        raise FileNotFoundError(f"metadata source file not found: {src}")
    os.makedirs(output_dir, exist_ok=True)
    dst = osp.join(output_dir, "metadata.snapshot.json")
    with open(src, "r") as src_f:
        contents = src_f.read()
    with open(dst, "w") as dst_f:
        dst_f.write(contents)
    return dst


def as_absolute_paths(paths: Iterable[str]) -> List[str]:
    out = []
    for path in paths:
        resolved = osp.expanduser(str(path))
        if not osp.isabs(resolved):
            resolved = osp.abspath(resolved)
        out.append(resolved)
    return out
