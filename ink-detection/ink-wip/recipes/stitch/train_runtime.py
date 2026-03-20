from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ink.recipes.losses.boundary import binary_mask_to_signed_distance_map
from ink.recipes.stitch.data import StitchData, normalize_component_key
from ink.recipes.stitch.ops import (
    accumulate_to_buffers as _accumulate_to_buffers_impl,
    allocate_segment_buffers,
    compose_segment_from_roi_buffers,
    gaussian_weights,
    resolve_buffer_crop,
)
from ink.recipes.stitch.state import (
    StitchExecutionContext,
    _StitchState,
    _coerce_component_dataset_map,
    _log,
    _noop_log,
)
from ink.recipes.stitch.terms import (
    _requires_boundary_dist_map,
    _stitch_loss_terms,
    compute_stitched_component_loss,
)
from ink.recipes.stitch.train_component_runtime import (
    compute_train_stitch_loss as _compute_train_stitch_loss_impl,
    stitch_saved_tensors_context as _stitch_saved_tensors_context_impl,
)


def _owner_data(owner) -> StitchData:
    data = getattr(owner, "data", None)
    if not isinstance(data, StitchData):
        raise TypeError("stitch runtime requires an owner with .data: StitchData")
    return data


def _owner_state(owner) -> _StitchState:
    state = getattr(owner, "state", None)
    if not isinstance(state, _StitchState):
        raise TypeError("stitch runtime requires an owner with .state")
    return state


def _owner_execution(owner) -> StitchExecutionContext:
    execution = getattr(owner, "execution", None)
    if not isinstance(execution, StitchExecutionContext):
        raise TypeError("stitch runtime requires an owner with .execution")
    return execution


@dataclass
class TrainStitchRuntime:
    data: StitchData
    state: _StitchState
    execution: StitchExecutionContext
    patch_loss: object = None
    patch_loss_weight: float = 1.0
    gradient_checkpointing: bool = False
    save_on_cpu: bool = False
    log: object = _noop_log
    loaders: list[Any] = field(default_factory=list)
    component_datasets: dict[tuple[str, int], Any] = field(default_factory=dict)
    _warned_checkpoint_vs_offload: bool = False

    def __post_init__(self) -> None:
        self.patch_loss_weight = float(self.patch_loss_weight)
        self.gradient_checkpointing = bool(self.gradient_checkpointing)
        self.save_on_cpu = bool(self.save_on_cpu)
        self.component_datasets = _coerce_component_dataset_map(self.component_datasets)

    def run_viz_pass(self, model):
        return run_train_stitch_pass(self, model)

    def compute_component_loss(self, model, *, component_key):
        return compute_train_stitch_loss(self, model, component_key=component_key)

    def set_loaders(self, loaders, segment_ids=None) -> None:
        self.loaders[:] = list(loaders or [])
        if segment_ids is not None:
            self.data.train.viz.segment_ids[:] = [str(segment_id) for segment_id in (segment_ids or [])]

    def loader_for_segment(self, segment_id):
        sid = str(segment_id)
        for loader, candidate_id in zip(self.loaders, self.data.train.viz.segment_ids):
            if str(candidate_id) == sid:
                return loader
        return None

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
    owner,
    *,
    outputs,
    xyxys,
    pred_buf,
    count_buf,
    offset=(0, 0),
):
    state = _owner_state(owner)
    data = _owner_data(owner)
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


def _normalize_xyxys(xyxys) -> list[tuple[int, int, int, int]]:
    if isinstance(xyxys, torch.Tensor):
        if xyxys.ndim == 1:
            if int(xyxys.numel()) != 4:
                raise ValueError(f"stitch xyxys tensor must have 4 values, got shape={tuple(xyxys.shape)}")
            return [tuple(int(v) for v in xyxys.tolist())]
        if xyxys.ndim != 2 or int(xyxys.shape[1]) != 4:
            raise ValueError(f"stitch xyxys tensor must have shape (N,4), got shape={tuple(xyxys.shape)}")
        return [tuple(int(v) for v in row.tolist()) for row in xyxys]

    if isinstance(xyxys, (list, tuple)):
        if len(xyxys) == 4 and all(isinstance(value, torch.Tensor) for value in xyxys):
            columns = [value.detach().reshape(-1) for value in xyxys]
            batch_size = int(columns[0].numel())
            return [
                tuple(int(column[row_idx].item()) for column in columns)
                for row_idx in range(batch_size)
            ]

        if len(xyxys) == 4 and all(not isinstance(value, (list, tuple)) for value in xyxys):
            return [
                tuple(
                    int(value.item()) if isinstance(value, torch.Tensor) else int(value)
                    for value in xyxys
                )
            ]

        normalized = []
        for item in xyxys:
            if isinstance(item, torch.Tensor):
                flat = item.detach().reshape(-1)
                if int(flat.numel()) != 4:
                    raise ValueError("stitch xyxy tensor entries must have 4 values")
                normalized.append(tuple(int(v.item()) for v in flat))
                continue
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError("stitch xyxy entries must have 4 values")
            normalized.append(
                tuple(
                    int(value.item()) if isinstance(value, torch.Tensor) else int(value)
                    for value in item
                )
            )
        return normalized

    raise TypeError(f"unsupported stitch xyxys value: {xyxys!r}")


def _accumulate_tensor_to_numpy_buffers(
    owner,
    *,
    values,
    xyxys,
    pred_buf,
    count_buf,
    offset=(0, 0),
    mode: str,
):
    data = _owner_data(owner)
    state = _owner_state(owner)
    ds = int(data.layout.downsample)
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


def _resolve_boundary_dist_map(owner, *, cache_key, stitched_targets, device):
    state = _owner_state(owner)

    dist_map_np = state._boundary_dist_maps_cpu.get(cache_key)
    if dist_map_np is None:
        target_mask = stitched_targets.detach().cpu().numpy() > 0.5
        dist_map_np = binary_mask_to_signed_distance_map(target_mask)
        state._boundary_dist_maps_cpu[cache_key] = dist_map_np

    device_key = (cache_key, str(device.type), getattr(device, "index", None))
    dist_map_t = state._boundary_dist_maps_torch.get(device_key)
    if dist_map_t is None:
        dist_map_t = torch.from_numpy(dist_map_np).to(device=device, dtype=torch.float32)
        state._boundary_dist_maps_torch[device_key] = dist_map_t
    return dist_map_t


def _stitch_saved_tensors_context(owner):
    return _stitch_saved_tensors_context_impl(owner, log=_log)


def compute_train_stitch_loss(owner, model, *, component_key):
    data = _owner_data(owner)
    terms = _stitch_loss_terms(data.train.loss.terms)
    if len(terms) <= 0:
        raise RuntimeError("stitched training requires stitch.train.loss.terms")

    return _compute_train_stitch_loss_impl(
        owner,
        model,
        component_key=component_key,
        terms=terms,
        normalize_xyxys=_normalize_xyxys,
        resolve_boundary_dist_map=_resolve_boundary_dist_map,
        compute_stitched_component_loss=compute_stitched_component_loss,
        requires_boundary_dist_map=_requires_boundary_dist_map,
        stitch_saved_tensors_context=_stitch_saved_tensors_context,
    )


def _should_run_train_viz(owner, model):
    data = _owner_data(owner)
    if not data.train.viz.enabled:
        return False, int(getattr(model, "current_epoch", 0))
    if not owner.loaders or not data.train.viz.segment_ids:
        return False, int(getattr(model, "current_epoch", 0))
    if len(owner.loaders) != len(data.train.viz.segment_ids):
        raise ValueError(
            "train stitch loaders/segment_ids length mismatch "
            f"({len(owner.loaders)} vs {len(data.train.viz.segment_ids)})"
        )
    epoch = int(getattr(model, "current_epoch", 0))
    if data.train.viz.every_n_epochs > 1 and ((epoch + 1) % data.train.viz.every_n_epochs) != 0:
        return False, epoch
    return True, epoch


def _compute_train_viz_metrics(owner, *, data, segment_id, pred_buffers, target_buffers, full_shape, metric_names):
    if not metric_names:
        return {}

    terms = _stitch_loss_terms(data.train.loss.terms)
    if len(terms) <= 0:
        raise RuntimeError("train stitch viz metrics require stitch.train.loss.terms")

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
            owner,
            cache_key=("viz", str(segment_id)),
            stitched_targets=stitched_targets,
            device=torch.device("cpu"),
        )
    metrics = compute_stitched_component_loss(
        terms,
        stitched_logits,
        stitched_targets,
        valid_mask=valid_mask,
        boundary_dist_map=boundary_dist_map,
    )
    requested = {str(name) for name in metric_names}
    return {
        key: value
        for key, value in metrics.items()
        if key == "covered_px" or key in requested
    }


def _run_train_segment(
    owner,
    model,
    *,
    data,
    loader,
    segment_id,
    meta,
    metric_names,
):
    full_shape = tuple(int(v) for v in meta.get("full_shape", (0, 0)))
    pred_buffers = allocate_segment_buffers(meta)
    if not pred_buffers:
        return None

    needs_metrics = bool(metric_names)
    target_buffers = allocate_segment_buffers(meta) if needs_metrics else None

    for batch in loader:
        x, y, xyxys, _group_idx = batch
        xyxys = _normalize_xyxys(xyxys)
        x = x.to(model.device, non_blocking=True)
        outputs = model(x)
        for pred_buf, count_buf, offset in pred_buffers:
            _accumulate_to_buffers(
                owner,
                outputs=outputs,
                xyxys=xyxys,
                pred_buf=pred_buf,
                count_buf=count_buf,
                offset=offset,
            )
        if target_buffers is None:
            continue
        for pred_buf, count_buf, offset in target_buffers:
            _accumulate_tensor_to_numpy_buffers(
                owner,
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
        f"coverage={coverage:.4f} covered_px={covered_px}/{total_px} "
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
        segment_metrics = _compute_train_viz_metrics(
            owner,
            data=data,
            segment_id=segment_id,
            pred_buffers=pred_buffers,
            target_buffers=target_buffers,
            full_shape=full_shape,
            metric_names=metric_names,
        )
        if segment_metrics:
            segment_viz["metrics"] = segment_metrics
    return segment_viz


def run_train_stitch_pass(owner, model):
    should_run, epoch = _should_run_train_viz(owner, model)
    if not should_run:
        return None

    data = _owner_data(owner)
    state = _owner_state(owner)
    metric_names = tuple(data.train.viz.metrics)

    t0 = time.perf_counter()
    _log(owner, f"train stitch pass start epoch={epoch}")
    segment_viz = {}

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode(), _owner_execution(owner).forward_context():
            for loader, segment_id in zip(owner.loaders, data.train.viz.segment_ids):
                sid = str(segment_id)
                meta = state.roi_meta_by_split["train"].get(sid)
                if meta is None:
                    continue
                segment_entry = _run_train_segment(
                    owner,
                    model,
                    data=data,
                    loader=loader,
                    segment_id=sid,
                    meta=meta,
                    metric_names=metric_names,
                )
                if segment_entry is not None:
                    segment_viz[sid] = segment_entry
    finally:
        if was_training:
            model.train()

    _log(owner, f"train stitch pass done epoch={epoch} elapsed={time.perf_counter() - t0:.1f}s")
    return segment_viz
