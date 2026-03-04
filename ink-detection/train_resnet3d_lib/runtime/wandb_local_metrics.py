import datetime
import json
import math
import os
import os.path as osp
import re
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger


def _to_json_value(value, *, key_path):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{key_path} contains non-finite float {value!r}")
        return value
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _to_json_value(value.item(), key_path=key_path)
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist(), key_path=key_path)
    if isinstance(value, torch.Tensor):
        return _to_json_value(value.detach().cpu().tolist(), key_path=key_path)
    if _is_wandb_summary_container(value):
        normalized_container = _normalize_wandb_summary_container(value, key_path=key_path)
        return _to_json_value(normalized_container, key_path=key_path)
    if isinstance(value, Mapping):
        normalized = {}
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise TypeError(
                    f"{key_path} mapping keys must be strings, got {type(nested_key).__name__}: {nested_key!r}"
                )
            nested_key_path = f"{key_path}.{nested_key}"
            normalized[nested_key] = _to_json_value(nested_value, key_path=nested_key_path)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _to_json_value(item, key_path=f"{key_path}[{idx}]")
            for idx, item in enumerate(value)
        ]
    raise TypeError(f"{key_path} has unsupported type {type(value).__name__}")


def _is_wandb_summary_container(value):
    cls = type(value)
    module_name = str(getattr(cls, "__module__", ""))
    class_name = str(getattr(cls, "__name__", ""))
    return module_name.startswith("wandb.") and class_name.startswith("SummarySub")


def _normalize_wandb_summary_container(value, *, key_path):
    as_dict_method = getattr(type(value), "_as_dict", None)
    if callable(as_dict_method):
        normalized_as_dict = as_dict_method(value)
        if not isinstance(normalized_as_dict, Mapping):
            raise TypeError(
                f"{key_path} _as_dict() must return a mapping, got {type(normalized_as_dict).__name__}"
            )
        return dict(normalized_as_dict)

    items_method = getattr(type(value), "items", None)
    if callable(items_method):
        normalized = {}
        for nested_key, nested_value in items_method(value):
            if not isinstance(nested_key, str):
                raise TypeError(
                    f"{key_path} mapping keys must be strings, got {type(nested_key).__name__}: {nested_key!r}"
                )
            normalized[nested_key] = nested_value
        return normalized

    iter_method = getattr(type(value), "__iter__", None)
    if callable(iter_method):
        return list(iter_method(value))

    raise TypeError(
        f"{key_path} has unsupported W&B summary container type {type(value).__name__}; "
        "expected _as_dict(), items(), or iterable behavior"
    )


def _is_wandb_media_object(value):
    cls = type(value)
    module_name = str(getattr(cls, "__module__", ""))
    if not module_name.startswith("wandb."):
        return False
    class_name = str(getattr(cls, "__name__", ""))
    if not class_name:
        return False
    if ".data_types" in module_name:
        return True
    media_class_names = {
        "Image",
        "Audio",
        "Video",
        "Table",
        "Html",
        "Object3D",
        "Molecule",
        "Plotly",
        "Bokeh",
        "Histogram",
    }
    return class_name in media_class_names


def _contains_wandb_media(value):
    if _is_wandb_media_object(value):
        return True
    if isinstance(value, Mapping):
        for nested_value in value.values():
            if _contains_wandb_media(nested_value):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if _contains_wandb_media(item):
                return True
        return False
    return False


def _normalize_metrics(metrics):
    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics must be a mapping, got {type(metrics).__name__}")
    normalized = {}
    dropped_media_metric_keys = []
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError(f"metrics keys must be strings, got {type(key).__name__}: {key!r}")
        try:
            normalized[key] = _to_json_value(value, key_path=f"metrics.{key}")
        except TypeError:
            if _contains_wandb_media(value):
                dropped_media_metric_keys.append(key)
                continue
            raise
    if dropped_media_metric_keys:
        normalized["_local/dropped_wandb_media_metric_keys"] = sorted(dropped_media_metric_keys)
    return normalized


def _normalize_step(step):
    if step is None:
        return None
    normalized = _to_json_value(step, key_path="step")
    if isinstance(normalized, bool):
        raise TypeError(f"step must be an integer or null, got {normalized!r}")
    if isinstance(normalized, float):
        if not normalized.is_integer():
            raise ValueError(f"step must be an integer or null, got {normalized!r}")
        normalized = int(normalized)
    if not isinstance(normalized, int):
        raise TypeError(f"step must be an integer or null, got {type(normalized).__name__}: {normalized!r}")
    return normalized


def _sanitize_path_component(value, *, key_path):
    if not isinstance(value, str):
        raise TypeError(f"{key_path} must be a string, got {type(value).__name__}")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{key_path} must be a non-empty string")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError(f"{key_path} became empty after sanitization: {value!r}")
    return cleaned


def _normalize_image_array(image, *, key_path):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif callable(getattr(type(image), "__array__", None)) and not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"{key_path} must be a numpy array or torch tensor, got {type(image).__name__}")

    if image.ndim == 2:
        arr = image
    elif image.ndim == 3:
        if image.shape[2] in {1, 3, 4}:
            arr = image
        elif image.shape[0] in {1, 3, 4}:
            arr = np.transpose(image, (1, 2, 0))
        else:
            raise ValueError(
                f"{key_path} has unsupported channel shape {tuple(image.shape)!r}; "
                "expected HxW, HxWx{1,3,4}, or {1,3,4}xHxW"
            )
    else:
        raise ValueError(f"{key_path} must be 2D or 3D, got ndim={image.ndim}")

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.uint8) * 255

    if np.issubdtype(arr.dtype, np.floating):
        arr_f = arr.astype(np.float32, copy=False)
        finite_mask = np.isfinite(arr_f)
        if not finite_mask.all():
            arr_f = arr_f.copy()
            arr_f[~finite_mask] = 0.0
        max_value = float(arr_f.max()) if arr_f.size else 0.0
        min_value = float(arr_f.min()) if arr_f.size else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            arr_f = arr_f * 255.0
        arr_u8 = np.clip(np.rint(arr_f), 0.0, 255.0).astype(np.uint8, copy=False)
        return arr_u8

    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    raise TypeError(f"{key_path} has unsupported dtype {arr.dtype}")


def _save_png_image(path, image_u8, *, key_path):
    if image_u8.dtype != np.uint8:
        raise TypeError(f"{key_path} must be uint8 after normalization, got {image_u8.dtype}")
    if image_u8.ndim == 2:
        mode = "L"
    elif image_u8.ndim == 3 and image_u8.shape[2] == 3:
        mode = "RGB"
    elif image_u8.ndim == 3 and image_u8.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError(
            f"{key_path} must have shape HxW, HxWx3, or HxWx4 after normalization, got {tuple(image_u8.shape)!r}"
        )

    from PIL import Image

    Image.fromarray(image_u8, mode=mode).save(path, format="PNG")


class LocalMetricsWandbLogger(WandbLogger):
    def configure_local_persistence(self, *, log_dir):
        if not isinstance(log_dir, str) or not log_dir.strip():
            raise ValueError(f"log_dir must be a non-empty string, got {log_dir!r}")
        resolved_log_dir = log_dir.strip()
        os.makedirs(resolved_log_dir, exist_ok=True)
        metrics_path = osp.join(resolved_log_dir, "wandb_metrics.jsonl")
        summary_path = osp.join(resolved_log_dir, "wandb_summary.json")
        media_dir = osp.join(resolved_log_dir, "wandb_media")
        media_manifest_path = osp.join(resolved_log_dir, "wandb_media_manifest.jsonl")
        os.makedirs(media_dir, exist_ok=True)

        current_metrics_fh = getattr(self, "_local_metrics_fh", None)
        if current_metrics_fh is not None:
            current_metrics_fh.close()
        current_media_manifest_fh = getattr(self, "_local_media_manifest_fh", None)
        if current_media_manifest_fh is not None:
            current_media_manifest_fh.close()

        self._local_metrics_path = metrics_path
        self._local_summary_path = summary_path
        self._local_log_dir = resolved_log_dir
        self._local_media_dir = media_dir
        self._local_media_manifest_path = media_manifest_path
        self._local_metrics_fh = open(metrics_path, "a", encoding="utf-8", buffering=1)
        self._local_media_manifest_fh = open(media_manifest_path, "a", encoding="utf-8", buffering=1)
        self._local_media_record_id = int(getattr(self, "_local_media_record_id", 0))

    def _require_local_metrics_file(self):
        metrics_fh = getattr(self, "_local_metrics_fh", None)
        if metrics_fh is None:
            raise RuntimeError("configure_local_persistence() must be called before metrics can be logged.")
        return metrics_fh

    def _require_local_summary_path(self):
        summary_path = getattr(self, "_local_summary_path", None)
        if summary_path is None:
            raise RuntimeError("configure_local_persistence() must be called before summary can be persisted.")
        return summary_path

    def _require_experiment(self):
        experiment = getattr(self, "_experiment", None)
        if experiment is None:
            raise RuntimeError("W&B experiment is not initialized.")
        return experiment

    def _require_local_log_dir(self):
        log_dir = getattr(self, "_local_log_dir", None)
        if log_dir is None:
            raise RuntimeError("configure_local_persistence() must be called before local media can be persisted.")
        return log_dir

    def _require_local_media_dir(self):
        media_dir = getattr(self, "_local_media_dir", None)
        if media_dir is None:
            raise RuntimeError("configure_local_persistence() must be called before local media can be persisted.")
        return media_dir

    def _require_local_media_manifest_file(self):
        media_manifest_fh = getattr(self, "_local_media_manifest_fh", None)
        if media_manifest_fh is None:
            raise RuntimeError("configure_local_persistence() must be called before local media can be persisted.")
        return media_manifest_fh

    def log_metrics(self, metrics, step=None):
        normalized_metrics = _normalize_metrics(metrics)
        normalized_step = _normalize_step(step)
        record = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "step": normalized_step,
            "metrics": normalized_metrics,
        }
        metrics_fh = self._require_local_metrics_file()
        metrics_fh.write(json.dumps(record, sort_keys=True, allow_nan=False))
        metrics_fh.write("\n")
        metrics_fh.flush()
        super().log_metrics(metrics, step=step)

    def log_image(self, key, images, step=None, **kwargs):
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"image key must be a non-empty string, got {key!r}")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
            raise TypeError(f"images must be a sequence, got {type(images).__name__}")

        captions = kwargs.get("caption")
        if captions is not None and (
            (not isinstance(captions, Sequence)) or isinstance(captions, (str, bytes, bytearray))
        ):
            raise TypeError(f"caption must be a sequence when provided, got {type(captions).__name__}")
        if captions is not None and len(captions) != len(images):
            raise ValueError(
                "caption length must match images length, "
                f"got len(caption)={len(captions)} len(images)={len(images)}"
            )

        normalized_step = _normalize_step(step)
        media_dir = self._require_local_media_dir()
        log_dir = self._require_local_log_dir()
        manifest_records = []

        key_components = [_sanitize_path_component(part, key_path="key") for part in key.split("/") if part]
        if not key_components:
            raise ValueError(f"image key has no valid path components after splitting: {key!r}")
        key_dir = osp.join(media_dir, *key_components)
        os.makedirs(key_dir, exist_ok=True)

        for idx, image in enumerate(images):
            image_u8 = _normalize_image_array(image, key_path=f"images[{idx}]")
            record_id = int(self._local_media_record_id)
            step_tag = "none" if normalized_step is None else str(normalized_step)
            filename = f"record_{record_id:08d}_step_{step_tag}_idx_{idx:04d}.png"
            out_path = osp.join(key_dir, filename)
            _save_png_image(out_path, image_u8, key_path=f"images[{idx}]")

            record = {
                "index": int(idx),
                "path": osp.relpath(out_path, start=log_dir),
                "shape": [int(v) for v in image_u8.shape],
                "caption": None if captions is None else _to_json_value(captions[idx], key_path=f"caption[{idx}]"),
            }
            manifest_records.append(record)
            self._local_media_record_id = record_id + 1

        manifest_entry = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "step": normalized_step,
            "key": key,
            "images": manifest_records,
        }
        media_manifest_fh = self._require_local_media_manifest_file()
        media_manifest_fh.write(json.dumps(manifest_entry, sort_keys=True, allow_nan=False))
        media_manifest_fh.write("\n")
        media_manifest_fh.flush()
        if step is None:
            super().log_image(key=key, images=images, **kwargs)
        else:
            super().log_image(key=key, images=images, step=step, **kwargs)

    def persist_summary(self):
        summary_path = self._require_local_summary_path()
        summary_payload = _to_json_value(dict(self._require_experiment().summary), key_path="summary")
        with open(summary_path, "w", encoding="utf-8") as summary_fh:
            json.dump(summary_payload, summary_fh, sort_keys=True, allow_nan=False)
            summary_fh.write("\n")

    def persist_local_state(self):
        self.persist_summary()
        metrics_fh = self._require_local_metrics_file()
        metrics_fh.flush()
        metrics_fh.close()
        self._local_metrics_fh = None
        media_manifest_fh = self._require_local_media_manifest_file()
        media_manifest_fh.flush()
        media_manifest_fh.close()
        self._local_media_manifest_fh = None
