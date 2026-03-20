from __future__ import annotations

from contextlib import nullcontext
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ink.core.types import Batch
from ink.recipes.stitch.data import StitchData, normalize_component_key
from ink.recipes.stitch.ops import (
    accumulate_to_buffers as _accumulate_to_buffers_impl,
    allocate_segment_buffers,
    compose_segment_from_roi_buffers,
    gaussian_weights,
    normalize_xyxy_rows,
    resolve_buffer_crop,
)
from ink.recipes.stitch.runtime import (
    StitchRuntimeState,
    _log,
    _noop_log,
)
from ink.recipes.stitch.terms import (
    StitchLossBatch,
    _requires_boundary_dist_map,
    _stitch_loss_terms,
    compute_stitched_loss_components,
)
from ink.recipes.stitch.train_component_runtime import (
    _allocate_train_loss_roi_buffers,
    _dataset_component_group_idx,
    _resolve_model_device,
    _resolve_boundary_dist_map,
    _run_component_patch_pass,
    _summarize_train_stitched_component_losses,
)


def _coerce_component_dataset_map(raw_datasets) -> dict[tuple[str, int], Any]:
    if raw_datasets is None:
        return {}
    return {
        normalize_component_key(component_key): dataset
        for component_key, dataset in dict(raw_datasets).items()
    }


def _segment_spec_ids(segment_specs) -> tuple[str, ...]:
    return tuple(str(spec.segment_id) for spec in (segment_specs or ()))


def _should_run_every_n_epochs(*, epoch: int, every_n_epochs: int) -> bool:
    cadence = max(1, int(every_n_epochs))
    return cadence == 1 or ((int(epoch) + 1) % cadence) == 0


def _segment_loader_pairs(*, loaders, segment_ids, mode_name: str) -> tuple[tuple[Any, str], ...]:
    normalized_ids = tuple(str(segment_id) for segment_id in (segment_ids or ()))
    normalized_loaders = tuple(loaders or ())
    if len(normalized_loaders) != len(normalized_ids):
        raise ValueError(
            f"{mode_name} stitch loaders/segment_ids length mismatch "
            f"({len(normalized_loaders)} vs {len(normalized_ids)})"
        )
    return tuple(zip(normalized_loaders, normalized_ids))


@dataclass
class TrainStitchRuntime:
    data: StitchData
    state: StitchRuntimeState
    patch_loss: object = None
    patch_loss_weight: float = 1.0
    gradient_checkpointing: bool = False
    save_on_cpu: bool = False
    log: object = _noop_log
    precision_context: object = None
    loaders: list[Any] = field(default_factory=list)
    log_only_loaders: list[Any] = field(default_factory=list)
    component_datasets: dict[tuple[str, int], Any] = field(default_factory=dict)
    _warned_checkpoint_vs_offload: bool = False

    def __post_init__(self) -> None:
        self.patch_loss_weight = float(self.patch_loss_weight)
        self.gradient_checkpointing = bool(self.gradient_checkpointing)
        self.save_on_cpu = bool(self.save_on_cpu)
        self.component_datasets = _coerce_component_dataset_map(self.component_datasets)

    def _run_segment_viz_pass(
        self,
        model,
        *,
        epoch: int,
        enabled: bool,
        every_n_epochs: int,
        loaders,
        segment_ids,
        meta_for_segment,
        loss_component_names=(),
        mode_name: str,
    ):
        if not enabled:
            return None
        segment_pairs = _segment_loader_pairs(
            loaders=loaders,
            segment_ids=segment_ids,
            mode_name=mode_name,
        )
        if not segment_pairs:
            return None
        if not _should_run_every_n_epochs(epoch=int(epoch), every_n_epochs=int(every_n_epochs)):
            return None

        t0 = time.perf_counter()
        _log(self, f"{mode_name} stitch pass start epoch={epoch}")
        segment_viz = {}

        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                for loader, sid in segment_pairs:
                    meta = meta_for_segment(sid)
                    if meta is None:
                        continue
                    segment_entry = _run_train_segment(
                        self,
                        model,
                        data=self.data,
                        state=self.state,
                        precision_context=self.precision_context,
                        loader=loader,
                        segment_id=sid,
                        meta=meta,
                        loss_component_names=loss_component_names,
                    )
                    if segment_entry is not None:
                        segment_viz[sid] = segment_entry
        finally:
            if was_training:
                model.train()

        _log(self, f"{mode_name} stitch pass done epoch={epoch} elapsed={time.perf_counter() - t0:.1f}s")
        return segment_viz

    def run_viz_pass(self, model, *, epoch: int):
        return self._run_segment_viz_pass(
            model,
            epoch=int(epoch),
            enabled=bool(self.data.train.viz.enabled),
            every_n_epochs=int(self.data.train.viz.every_n_epochs),
            loaders=self.loaders,
            segment_ids=_segment_spec_ids(self.data.train.segments),
            meta_for_segment=self.state.train_segment_meta,
            loss_component_names=tuple(self.data.train.viz.loss_components),
            mode_name="train",
        )

    def run_log_only_viz_pass(self, model, *, epoch: int):
        return self._run_segment_viz_pass(
            model,
            epoch=int(epoch),
            enabled=bool(self.data.log_only.segment_ids),
            every_n_epochs=int(self.data.log_only.every_n_epochs),
            loaders=self.log_only_loaders,
            segment_ids=self.data.log_only.segment_ids,
            meta_for_segment=self.state.log_only_segment_meta,
            loss_component_names=(),
            mode_name="log_only",
        )

    def compute_component_loss(self, model, *, component_key):
        terms = _stitch_loss_terms(self.data.train.loss.terms)
        if len(terms) <= 0:
            raise RuntimeError("stitched training requires stitch.train.loss.terms")

        if not self.component_datasets or not self.data.train.component_keys:
            raise RuntimeError("train component stitch datasets are not configured")
        if len(self.component_datasets) != len(self.data.train.component_keys):
            raise ValueError(
                "train component datasets/keys length mismatch "
                f"({len(self.component_datasets)} vs {len(self.data.train.component_keys)})"
            )

        component_key = normalize_component_key(component_key)
        dataset = self.dataset_for_component(component_key)
        if dataset is None:
            raise KeyError(f"missing train component stitch dataset for component_key={component_key!r}")

        meta = self.state.component_meta_for(component_key)
        if meta is None:
            raise KeyError(f"missing train component ROI metadata for component_key={component_key!r}")

        segment_id, component_idx = component_key
        group_idx = _dataset_component_group_idx(dataset, component_key=component_key)
        batch_size = int(self.data.train.loss.patch_batch_size or self.data.train.loss.valid_batch_size or 1)
        roi_buffers = _allocate_train_loss_roi_buffers(meta, device=_resolve_model_device(model))

        patch_components = _run_component_patch_pass(
            self,
            model,
            data=self.data,
            state=self.state,
            dataset=dataset,
            component_key=component_key,
            batch_size=batch_size,
            roi_buffers=roi_buffers,
        )
        stitch_components = _summarize_train_stitched_component_losses(
            self,
            state=self.state,
            terms=terms,
            component_key=component_key,
            roi_buffers=roi_buffers,
        )
        return {
            "component_key": component_key,
            "segment_id": segment_id,
            "component_idx": component_idx,
            "group_idx": int(group_idx),
            "patch": patch_components,
            "stitch": stitch_components,
        }

    def set_loaders(self, loaders) -> None:
        self.loaders[:] = list(loaders or [])

    def set_log_only_loaders(self, loaders) -> None:
        self.log_only_loaders[:] = list(loaders or [])

    def set_component_datasets(self, datasets, component_keys=None) -> None:
        normalized_keys = [normalize_component_key(key) for key in (component_keys or [])]
        if component_keys is not None:
            component_specs_by_key = {spec.component_key: spec for spec in self.data.train.components}
            if normalized_keys and all(component_key in component_specs_by_key for component_key in normalized_keys):
                self.data.train.components = [
                    component_specs_by_key[component_key] for component_key in normalized_keys
                ]

        dataset_pairs = zip(normalized_keys, list(datasets or [])) if component_keys is not None else []
        self.component_datasets.clear()
        if component_keys is not None:
            self.component_datasets.update({component_key: dataset for component_key, dataset in dataset_pairs})
        else:
            self.component_datasets.update(_coerce_component_dataset_map(datasets))
        self.state.clear_boundary_caches()

    def dataset_for_component(self, component_key):
        return self.component_datasets.get(normalize_component_key(component_key))


def _accumulate_to_buffers(
    *,
    data,
    state,
    outputs,
    xyxys,
    pred_buf,
    count_buf,
    offset=(0, 0),
):
    return _accumulate_to_buffers_impl(
        outputs=outputs,
        xyxys=xyxys,
        pred_buf=pred_buf,
        count_buf=count_buf,
        downsample=int(data.layout.downsample),
        offset=offset,
        gaussian_cache=state._gaussian_cache,
        gaussian_sigma_scale=float(state._gaussian_sigma_scale),
        gaussian_min_weight=float(state._gaussian_min_weight),
    )


def _accumulate_tensor_to_numpy_buffers(
    *,
    downsample,
    state,
    values,
    xyxys,
    pred_buf,
    count_buf,
    offset=(0, 0),
    mode: str,
):
    ds = int(downsample)
    values = values.detach().to("cpu", dtype=torch.float32)
    for i, xyxy in enumerate(xyxys):
        crop = resolve_buffer_crop(
            xyxy=xyxy,
            downsample=ds,
            offset=offset,
            buffer_shape=pred_buf.shape,
        )
        if crop is None:
            continue

        patch = values[i].unsqueeze(0)
        if patch.shape[-2:] != (crop["target_h"], crop["target_w"]):
            if mode == "nearest":
                patch = F.interpolate(patch, size=(crop["target_h"], crop["target_w"]), mode=mode)
            else:
                patch = F.interpolate(
                    patch,
                    size=(crop["target_h"], crop["target_w"]),
                    mode=mode,
                    align_corners=False,
                )

        patch_weights = gaussian_weights(
            state._gaussian_cache,
            h=crop["target_h"],
            w=crop["target_w"],
            sigma_scale=float(state._gaussian_sigma_scale),
            min_weight=float(state._gaussian_min_weight),
        )
        patch_crop = patch[..., crop["py0"]:crop["py1"], crop["px0"]:crop["px1"]]
        weight_crop = patch_weights[crop["py0"]:crop["py1"], crop["px0"]:crop["px1"]]

        pred_buf[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]] += (
            patch_crop.squeeze(0).squeeze(0).numpy() * weight_crop
        )
        count_buf[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]] += weight_crop


def _unpack_segment_batch(batch):
    if isinstance(batch, Batch):
        if batch.meta.patch_xyxy is None:
            raise ValueError("stitch segment batch requires meta.patch_xyxy")
        return batch.x, batch.y, batch.meta.patch_xyxy, batch.meta.group_idx
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        return batch
    raise ValueError("stitch segment batch must be Batch or (x, y, xyxys, group_idx)")


def _compose_average_from_roi_buffers(roi_buffers, *, full_shape):
    full_shape = tuple(int(v) for v in full_shape)
    full_sum = np.zeros(full_shape, dtype=np.float32)
    full_count = np.zeros(full_shape, dtype=np.float32)
    for pred_buf, count_buf, offset in roi_buffers:
        y0, x0 = [int(v) for v in offset]
        h, w = pred_buf.shape
        full_sum[y0:y0 + h, x0:x0 + w] += pred_buf
        full_count[y0:y0 + h, x0:x0 + w] += count_buf
    covered = full_count > 0
    averaged = np.divide(
        full_sum,
        full_count,
        out=np.zeros_like(full_sum, dtype=np.float32),
        where=covered,
    )
    return averaged, covered


def _compute_train_viz_loss_components(
    *,
    data,
    state,
    segment_id,
    pred_buffers,
    target_buffers,
    full_shape,
    loss_component_names,
):
    if not loss_component_names:
        return {}

    terms = _stitch_loss_terms(data.train.loss.terms)
    if len(terms) <= 0:
        raise RuntimeError("train stitch viz loss components require stitch.train.loss.terms")

    logits_np, valid_np = _compose_average_from_roi_buffers(pred_buffers, full_shape=full_shape)
    targets_np, _ = _compose_average_from_roi_buffers(target_buffers, full_shape=full_shape)
    valid_mask = torch.from_numpy(valid_np)
    if not bool(valid_mask.any().item()):
        return {"covered_px": 0}

    stitched_logits = torch.from_numpy(logits_np)
    stitched_targets = torch.from_numpy(targets_np)
    boundary_dist_map = None
    if _requires_boundary_dist_map(terms):
        boundary_dist_map = _resolve_boundary_dist_map(
            state,
            cache_key=("viz", str(segment_id)),
            stitched_targets=stitched_targets,
            device=torch.device("cpu"),
        )
    components = compute_stitched_loss_components(
        terms,
        StitchLossBatch(
            logits=stitched_logits,
            targets=stitched_targets,
            valid_mask=valid_mask,
            boundary_dist_map=boundary_dist_map,
        ),
    )
    requested = {str(name) for name in loss_component_names}
    return {
        key: value
        for key, value in components.items()
        if key == "covered_px" or key in requested
    }


def _run_train_segment(
    owner,
    model,
    *,
    data,
    state,
    precision_context,
    loader,
    segment_id,
    meta,
    loss_component_names,
):
    full_shape = tuple(int(v) for v in meta.get("full_shape", (0, 0)))
    pred_buffers = allocate_segment_buffers(meta)
    if not pred_buffers:
        return None

    needs_loss_components = bool(loss_component_names)
    target_buffers = allocate_segment_buffers(meta) if needs_loss_components else None
    model_device = _resolve_model_device(model)

    for batch in loader:
        x, y, xyxys, _group_idx = _unpack_segment_batch(batch)
        xyxys = normalize_xyxy_rows(xyxys)
        x = x.to(model_device, non_blocking=True)
        context = precision_context(device=x.device) if callable(precision_context) else nullcontext()
        with context:
            outputs = model(x)
        for pred_buf, count_buf, offset in pred_buffers:
            _accumulate_to_buffers(
                data=data,
                state=state,
                outputs=outputs,
                xyxys=xyxys,
                pred_buf=pred_buf,
                count_buf=count_buf,
                offset=offset,
            )
        if target_buffers is None:
            continue
        if y is None:
            raise ValueError("train stitch viz loss components require targets")
        for pred_buf, count_buf, offset in target_buffers:
            _accumulate_tensor_to_numpy_buffers(
                downsample=data.layout.downsample,
                state=state,
                values=y,
                xyxys=xyxys,
                pred_buf=pred_buf,
                count_buf=count_buf,
                offset=offset,
                mode="nearest",
            )

    stitched, covered = compose_segment_from_roi_buffers(pred_buffers, full_shape)
    covered_px = int(covered.sum())
    total_px = int(covered.size)
    coverage = float(covered_px) / float(max(1, total_px))
    if covered_px > 0:
        vals = stitched[covered]
        prob_mean = float(vals.mean()) if vals.size else float("nan")
        prob_max = float(vals.max()) if vals.size else float("nan")
    else:
        prob_mean = float("nan")
        prob_max = float("nan")

    _log(
        owner,
        f"train stitch summary segment={segment_id} "
        f"coverage={coverage:.4f} "
        f"covered_px={covered_px}/{total_px} "
        f"prob_mean={prob_mean:.4f} prob_max={prob_max:.4f}",
    )

    segment_viz = {
        "img_u8": (np.clip(stitched, 0.0, 1.0) * 255.0).astype(np.uint8),
        "has": covered,
        "meta": {
            "offset": (0, 0),
            "full_shape": tuple(int(v) for v in full_shape),
        },
    }
    if target_buffers is not None:
        segment_loss_components = _compute_train_viz_loss_components(
            data=data,
            state=state,
            segment_id=segment_id,
            pred_buffers=pred_buffers,
            target_buffers=target_buffers,
            full_shape=full_shape,
            loss_component_names=loss_component_names,
        )
        if segment_loss_components:
            segment_viz["loss_components"] = segment_loss_components
    return segment_viz
