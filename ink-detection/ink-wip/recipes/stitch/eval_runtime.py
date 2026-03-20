from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ink.recipes.stitch.ops import compose_segment_from_roi_buffers


def _owner_data(owner):
    data = getattr(owner, "data", None)
    if data is None:
        raise TypeError("stitch eval runtime requires an owner with .data")
    return data


def _owner_state(owner):
    state = getattr(owner, "state", None)
    if state is None:
        raise TypeError("stitch eval runtime requires an owner with .state")
    return state


def _owner_execution(owner):
    execution = getattr(owner, "execution", None)
    if execution is None:
        raise TypeError("stitch eval runtime requires an owner with .execution")
    return execution


def _import_optional_attr(module_name: str, attr_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    return getattr(module, attr_name, None)


def accumulate_val(
    owner,
    *,
    outputs,
    xyxys,
    dataloader_idx,
    normalize_xyxys: Callable[[Any], list[tuple[int, int, int, int]]],
    accumulate_to_buffers: Callable[..., Any],
):
    data = _owner_data(owner)
    state = _owner_state(owner)
    if not state.roi_buffers_by_split["eval"]:
        return

    xyxys = normalize_xyxys(xyxys)
    sid = data.eval.loader_to_segment.get(int(dataloader_idx))
    if sid is None:
        return

    for pred_buf, count_buf, offset in state.roi_buffers_by_split["eval"].get(str(sid), []):
        accumulate_to_buffers(
            owner,
            outputs=outputs,
            xyxys=xyxys,
            pred_buf=pred_buf,
            count_buf=count_buf,
            offset=offset,
        )


def collect_eval_segments_for_logging(*, data, state):
    segment_to_val = {}
    segment_to_val_meta = {}
    for _loader_idx, segment_id in data.eval.loader_to_segment.items():
        sid = str(segment_id)
        roi_buffers = state.roi_buffers_by_split["eval"].get(sid)
        if not roi_buffers:
            continue
        meta = state.roi_meta_by_split["eval"].get(sid, {})
        full_shape = tuple(meta.get("full_shape", roi_buffers[0][0].shape))
        base, has = compose_segment_from_roi_buffers(roi_buffers, full_shape)
        segment_to_val[sid] = (base, has)
        segment_to_val_meta[sid] = {
            "offset": (0, 0),
            "full_shape": full_shape,
        }
    return segment_to_val, segment_to_val_meta


def finalize_validation_epoch(owner, model):
    data = _owner_data(owner)
    state = _owner_state(owner)
    execution = _owner_execution(owner)
    if not state.roi_buffers_by_split["eval"]:
        return

    sanity_checking = bool(execution.sanity_checking)
    train_segment_viz = _resolve_train_segment_viz(
        owner,
        data=data,
        sanity_checking=sanity_checking,
        model=model,
    )

    segment_to_val, segment_to_val_meta = collect_eval_segments_for_logging(data=data, state=state)
    _log_eval_media(
        model=model,
        sanity_checking=sanity_checking,
        train_segment_viz=train_segment_viz,
        segment_to_val=segment_to_val,
        data=data,
    )
    _log_eval_metrics(
        model=model,
        sanity_checking=sanity_checking,
        segment_to_val=segment_to_val,
        segment_to_val_meta=segment_to_val_meta,
        data=data,
    )
    state.reset_split_buffers("eval")


def _resolve_train_segment_viz(owner, *, data, sanity_checking: bool, model):
    if owner.train is None:
        return {}
    if not data.train.viz.enabled:
        return {}
    if sanity_checking:
        return {}
    return owner.train.run_viz_pass(model) or {}


def _log_eval_media(
    *,
    model,
    sanity_checking: bool,
    train_segment_viz,
    segment_to_val,
    data,
):
    log_media = _import_optional_attr("ink.recipes.stitch.wandb_media", "log_stitched_wandb_media")
    if callable(log_media):
        log_media(
            model=model,
            sanity_checking=sanity_checking,
            log_train_stitch=bool(train_segment_viz),
            segment_to_val=segment_to_val,
            train_segment_viz=train_segment_viz,
            borders_by_split=data.layout.borders_by_split,
            downsample=int(data.layout.downsample),
            compose_segment_from_roi_buffers=compose_segment_from_roi_buffers,
        )


def _log_eval_metrics(
    *,
    model,
    sanity_checking: bool,
    segment_to_val,
    segment_to_val_meta,
    data,
):
    if not data.eval.metrics:
        return
    log_metrics = _import_optional_attr("ink.recipes.stitch.metrics_runtime", "log_stitched_validation_metrics")
    if callable(log_metrics):
        log_metrics(
            model=model,
            sanity_checking=sanity_checking,
            segment_to_val=segment_to_val,
            segment_to_val_meta=segment_to_val_meta,
            downsample=int(data.layout.downsample),
            metrics=list(data.eval.metrics),
        )
