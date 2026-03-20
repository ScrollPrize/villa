from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import uuid

import numpy as np

from ink.recipes.stitch.artifact_primitives import (
    normalize_skeleton_method as _normalize_skeleton_method,
    pseudo_weight_maps,
    skeletonize_binary,
)

STITCH_EVAL_ARTIFACT_SCHEMA_VERSION = 1


def _as_bool_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={tuple(array.shape)}")
    return array.astype(bool, copy=False)


def _readonly_bool(array: np.ndarray) -> np.ndarray:
    out = np.array(_as_bool_2d(array), dtype=bool, copy=True)
    out.setflags(write=False)
    return out


def _readonly_float32(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=np.float32, copy=True)
    if out.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={tuple(out.shape)}")
    out.setflags(write=False)
    return out


def _segment_store_key(segment_id: str) -> str:
    return str(segment_id).replace("/", "__")


@dataclass(frozen=True)
class StitchEvalArtifactKey:
    segment_id: str
    source_fingerprint: str
    metric_bbox: tuple[int, int, int, int]
    component_bbox: tuple[int, int, int, int]
    downsample: int
    connectivity: int
    component_pad: int


def _stitch_eval_artifact_key_payload(key: StitchEvalArtifactKey) -> dict[str, object]:
    return {
        "schema_version": int(STITCH_EVAL_ARTIFACT_SCHEMA_VERSION),
        "segment_id": str(key.segment_id),
        "source_fingerprint": str(key.source_fingerprint),
        "metric_bbox": [int(v) for v in key.metric_bbox],
        "component_bbox": [int(v) for v in key.component_bbox],
        "downsample": int(key.downsample),
        "connectivity": int(key.connectivity),
        "component_pad": int(key.component_pad),
    }


def _stitch_eval_artifact_key_digest(key: StitchEvalArtifactKey) -> str:
    payload = json.dumps(_stitch_eval_artifact_key_payload(key), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _save_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_npy(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        loaded = np.load(handle, allow_pickle=False)
    return np.asarray(loaded)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


@dataclass
class StitchEvalArtifactStore:
    cache_root: str | Path | None = None
    _prepared: dict[StitchEvalArtifactKey, PreparedStitchEvalArtifacts] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.cache_root is None:
            return
        root = Path(self.cache_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.cache_root = root

    def _artifact_path(self, *, key: StitchEvalArtifactKey, artifact_name: str, ext: str) -> Path | None:
        if self.cache_root is None:
            return None
        return Path(self.cache_root) / _segment_store_key(key.segment_id) / _stitch_eval_artifact_key_digest(key) / (
            f"{artifact_name}.{ext}"
        )

    def _load_or_compute_bool_array(
        self,
        *,
        key: StitchEvalArtifactKey,
        artifact_name: str,
        compute,
    ) -> np.ndarray:
        path = self._artifact_path(key=key, artifact_name=artifact_name, ext="npy")
        if path is not None and path.exists():
            try:
                return _readonly_bool(_load_npy(path))
            except Exception:
                pass
        out = _readonly_bool(compute())
        if path is not None:
            _save_npy_atomic(path, out)
        return out

    def _load_or_compute_npz(
        self,
        *,
        key: StitchEvalArtifactKey,
        artifact_name: str,
        compute,
    ) -> dict[str, np.ndarray]:
        path = self._artifact_path(key=key, artifact_name=artifact_name, ext="npz")
        if path is not None and path.exists():
            try:
                return _load_npz(path)
            except Exception:
                pass
        out = compute()
        if path is not None:
            _save_npz_atomic(path, **out)
        return out

    def skeleton(
        self,
        *,
        key: StitchEvalArtifactKey,
        selected_component_gt: np.ndarray,
        method: str,
    ) -> np.ndarray:
        method_name = _normalize_skeleton_method(method)
        return self._load_or_compute_bool_array(
            key=key,
            artifact_name=f"skeleton_{method_name}",
            compute=lambda: skeletonize_binary(selected_component_gt, method=method_name),
        )

    def selected_component_gt(
        self,
        *,
        key: StitchEvalArtifactKey,
        labels: np.ndarray,
        supervision: np.ndarray,
        component_mask: np.ndarray,
    ) -> np.ndarray:
        return self._load_or_compute_bool_array(
            key=key,
            artifact_name="selected_component_gt",
            compute=lambda: np.asarray(labels, dtype=bool)
            & np.asarray(supervision, dtype=bool)
            & np.asarray(component_mask, dtype=bool),
        )

    def pfm_weights(
        self,
        *,
        key: StitchEvalArtifactKey,
        selected_component_gt: np.ndarray,
        connectivity: int,
        method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        method_name = _normalize_skeleton_method(method)
        packed = self._load_or_compute_npz(
            key=key,
            artifact_name=f"pfm_weights_{method_name}",
            compute=lambda: _pack_pfm_weights(
                pseudo_weight_maps(
                    selected_component_gt,
                    connectivity=int(connectivity),
                    skel_gt=self.skeleton(
                        key=key,
                        selected_component_gt=selected_component_gt,
                        method=method_name,
                    ),
                )
            ),
        )
        return (
            _readonly_float32(packed["recall_weights"]),
            _readonly_float32(packed["precision_weights"]),
        )

    def get_prepared(
        self,
        *,
        key: StitchEvalArtifactKey,
        labels: np.ndarray,
        supervision: np.ndarray,
        component_mask: np.ndarray,
    ) -> PreparedStitchEvalArtifacts:
        prepared = self._prepared.get(key)
        if prepared is None:
            prepared = PreparedStitchEvalArtifacts(
                key=key,
                labels=labels,
                supervision=supervision,
                component_mask=component_mask,
                store=self,
            )
            self._prepared[key] = prepared
        return prepared


def _pack_pfm_weights(weights: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    recall_weights, precision_weights = weights
    return {
        "recall_weights": np.asarray(recall_weights, dtype=np.float32),
        "precision_weights": np.asarray(precision_weights, dtype=np.float32),
    }


@dataclass(kw_only=True)
class PreparedStitchEvalArtifacts:
    store: StitchEvalArtifactStore
    key: StitchEvalArtifactKey
    labels: np.ndarray
    supervision: np.ndarray
    component_mask: np.ndarray

    _selected_component_gt: np.ndarray | None = field(default=None, init=False, repr=False)
    _skeleton_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _pfm_cache: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.labels = _readonly_bool(self.labels)
        self.supervision = _readonly_bool(self.supervision)
        self.component_mask = _readonly_bool(self.component_mask)
        if self.labels.shape != self.supervision.shape or self.labels.shape != self.component_mask.shape:
            raise ValueError(
                "prepared stitch artifacts require labels, supervision, and component_mask to share one shape"
            )

    def selected_component_gt(self) -> np.ndarray:
        if self._selected_component_gt is None:
            self._selected_component_gt = self.store.selected_component_gt(
                key=self.key,
                labels=self.labels,
                supervision=self.supervision,
                component_mask=self.component_mask,
            )
        return self._selected_component_gt

    def skeleton(self, *, method: str = "guo_hall") -> np.ndarray:
        method_name = _normalize_skeleton_method(method)
        skeleton = self._skeleton_cache.get(method_name)
        if skeleton is None:
            skeleton = self.store.skeleton(
                key=self.key,
                selected_component_gt=self.selected_component_gt(),
                method=method_name,
            )
            self._skeleton_cache[method_name] = skeleton
        return skeleton

    def pfm_weights(self, *, method: str = "guo_hall") -> tuple[np.ndarray, np.ndarray]:
        method_name = _normalize_skeleton_method(method)
        weights = self._pfm_cache.get(method_name)
        if weights is None:
            recall_weights, precision_weights = self.store.pfm_weights(
                key=self.key,
                selected_component_gt=self.selected_component_gt(),
                connectivity=int(self.key.connectivity),
                method=method_name,
            )
            weights = (
                _readonly_float32(recall_weights),
                _readonly_float32(precision_weights),
            )
            self._pfm_cache[method_name] = weights
        return weights
