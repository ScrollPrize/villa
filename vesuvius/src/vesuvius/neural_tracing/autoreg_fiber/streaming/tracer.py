"""End-to-end bidirectional fiber tracer.

Glues together :class:`AutoregFiberModel` (with the KV-cache from M1) and
:class:`WindowedVolumeReader` (M2) into the core algorithm that extends one
or two ends of an annotated fiber to the volume boundary.

Per-step cost is dominated by one decoder layer-stack on a single token
(O(seq) attention via cached K/V) plus a chunk LRU lookup; the costly DINO
backbone re-encode and ``init_kv_cache`` reprime only fire when the window
needs to slide.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import resample_polyline_uniform
from vesuvius.neural_tracing.autoreg_fiber.model import AutoregFiberModel, FiberKVCache
from vesuvius.neural_tracing.autoreg_fiber.serialization import IGNORE_INDEX
from vesuvius.neural_tracing.autoreg_fiber.streaming.window import WindowedVolumeReader
from vesuvius.neural_tracing.autoreg_mesh.serialization import quantize_local_xyz


@dataclasses.dataclass
class TraceResult:
    """Output of :meth:`FiberTracer.trace_one_direction`.

    `polyline_world_zyx` always starts with the input prompt's world points and
    is appended one-per-step until `stop_reason` fires.
    """

    polyline_world_zyx: np.ndarray
    stop_probabilities: np.ndarray
    stop_reason: str
    steps: int
    reanchors: int


@dataclasses.dataclass
class BidirectionalResult:
    polyline_world_zyx: np.ndarray
    forward: TraceResult
    backward: TraceResult


class FiberTracer:
    """Greedy autoregressive tracer with KV-cache + sliding window streaming.

    Parameters
    ----------
    model:
        A trained :class:`AutoregFiberModel`. Must be on ``device`` and in eval
        mode; the tracer forces ``.eval()`` defensively.
    reader:
        The :class:`WindowedVolumeReader` over the source volume.
    max_steps:
        Per-direction step budget.
    stop_prob_threshold:
        Greedy stop trigger on the head's stop-probability. ``None`` disables
        the trigger (relies purely on `max_steps` and volume bounds).
    min_steps:
        Minimum number of steps before `stop_prob_threshold` can fire.
    dtype:
        Autocast dtype for the streaming forward. Defaults to ``bfloat16`` to
        match the training run's ``mixed_precision: "bf16"``.
    """

    def __init__(
        self,
        model: AutoregFiberModel,
        reader: WindowedVolumeReader,
        *,
        max_steps: int = 5000,
        stop_prob_threshold: float | None = 0.5,
        min_steps: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        max_steps_per_window: int | None = None,
        snap_stride: float | None = None,
    ) -> None:
        model.eval()
        self.model = model
        self.reader = reader
        self.max_steps = int(max_steps)
        self.stop_prob_threshold = None if stop_prob_threshold is None else float(stop_prob_threshold)
        self.min_steps = int(min_steps)
        self.dtype = dtype
        self.device = next(model.parameters()).device
        self.prompt_length = int(model.config["prompt_length"])
        self.input_shape = tuple(int(v) for v in model.config["input_shape"])
        if tuple(self.input_shape) != tuple(self.reader.crop_size):
            raise ValueError(
                f"model.input_shape ({self.input_shape}) and reader.crop_size "
                f"({self.reader.crop_size}) must match for streaming inference"
            )
        self.max_fiber_position_embeddings = int(model.config["max_fiber_position_embeddings"])
        # The model was trained with target_length steps after the prompt; beyond that
        # the stop head saturates to 1.0 and the position embedding leaves its trained
        # range. To extend a trace past target_length we force a re-anchor (rebuilds
        # the prompt from the last prompt_length predicted points, resets positions).
        # Default to target_length - prompt_length so the model always operates inside
        # its trained range.
        target_length = int(model.config["target_length"])
        default_window = max(1, target_length - self.prompt_length)
        self.max_steps_per_window = (
            int(max_steps_per_window) if max_steps_per_window is not None else default_window
        )
        # Cache stride (training distribution): the fiber-cache pipeline resamples
        # every fiber to exact `densify_step` voxels of arc length between
        # consecutive points (typically 4.0 — see
        # `coordinate_convention: wk_xyz_to_new_zyx_resampled_4` in the cache
        # metadata). The model only ever saw inputs at this stride during
        # training, so when ``snap_stride`` is set we project each prediction
        # onto the sphere of radius ``snap_stride`` centred at the previous
        # input. This keeps the autoregressive feedback on the training manifold
        # and prevents the loop from degenerating into a sub-stride fixed point.
        self.snap_stride = None if snap_stride is None else float(snap_stride)
        # Step tangent estimate used as a fallback for the snap when the model's
        # raw direction is degenerate (norm < 1 voxel).
        self._fallback_tangent: np.ndarray | None = None

    # --- public API ---------------------------------------------------- #

    def trace_one_direction(
        self,
        prompt_world_zyx: np.ndarray,
        *,
        prefetch: bool = True,
        progress: bool | str = False,
    ) -> TraceResult:
        """Extend ``prompt_world_zyx`` (an `(N, 3)` array of world ``zyx``) forward.

        The prompt's last :attr:`prompt_length` points are used as the model's
        input; the tracer then yields one new point per step until a stop
        condition fires. The returned polyline includes the prompt at its head.

        ``progress``: pass ``True`` to draw a tqdm progress bar, or a string to
        use as the bar's description (defaults off so the function remains
        silent for library use).
        """

        if prompt_world_zyx.ndim != 2 or prompt_world_zyx.shape[1] != 3:
            raise ValueError(
                f"prompt_world_zyx must have shape (N, 3); got {tuple(prompt_world_zyx.shape)}"
            )
        if prompt_world_zyx.shape[0] < self.prompt_length:
            raise ValueError(
                f"prompt has {prompt_world_zyx.shape[0]} points but the model needs at least "
                f"prompt_length={self.prompt_length} to prime the cache"
            )

        # The trainer anchors each 128^3 window on the AABB midpoint of the
        # 8 prompt + 32 target points, so the prompt sits at *one edge* of the
        # window with empty space ahead for the predicted trajectory. Inference
        # has to mimic that or the model loses its sense of direction. We
        # estimate the prompt's tangent from the last two prompt points and
        # push the anchor forward by half the crop along that tangent.
        prompt_tail = prompt_world_zyx[-self.prompt_length :]
        anchor_offset_world = self._anchor_offset_along_prompt(prompt_tail)
        self.reader.anchor_on(prompt_tail[-1] + anchor_offset_world)
        prefetch and self.reader.prefetch_anchor(prompt_tail[-1] + anchor_offset_world)

        # Prime KV-cache.
        outputs, cache, step_position = self._init_at_current_anchor(prompt_tail)

        polyline: list[np.ndarray] = [point.astype(np.float32, copy=True) for point in prompt_world_zyx]
        stop_probabilities: list[float] = []
        stop_reason = "max_steps"
        reanchors = 0

        bar_desc = progress if isinstance(progress, str) else "trace"
        bar = tqdm(
            total=self.max_steps,
            desc=bar_desc,
            unit="step",
            leave=True,
            dynamic_ncols=True,
            disable=not progress,
        )

        # Seed the fallback tangent from the prompt's own stride direction
        # so the very first step has something sensible to snap onto when
        # the model's raw output is degenerate.
        self._fallback_tangent = self._unit_tangent(prompt_tail)

        try:
            for step_idx in range(self.max_steps):
                coarse_id, offset_bins, local_xyz = self._sample_step(outputs)
                stop_prob = float(torch.sigmoid(outputs["stop_logits"][0, 0].float()).item())
                stop_probabilities.append(stop_prob)
                world_xyz = self.reader.local_to_world(local_xyz)
                # In-distribution snap: force the next polyline point to be
                # exactly `snap_stride` voxels of arc length from the previous
                # input, along the direction the model emitted. This matches
                # the training cache's uniform 4-voxel spacing.
                if self.snap_stride is not None and self.snap_stride > 0.0:
                    prev_input = np.asarray(polyline[-1], dtype=np.float32)
                    delta = world_xyz.astype(np.float32, copy=False) - prev_input
                    norm = float(np.linalg.norm(delta))
                    if norm >= 1.0:
                        unit = delta / norm
                    elif self._fallback_tangent is not None:
                        unit = self._fallback_tangent
                    else:
                        unit = None
                    if unit is not None:
                        world_xyz = (prev_input + unit * float(self.snap_stride)).astype(np.float32)
                        local_xyz = self.reader.world_to_local(world_xyz)
                        # Re-quantise so the model gets coarse/offset bins
                        # that match the snapped world position exactly.
                        coarse_id, offset_bins = self._quantize_local_xyz(local_xyz)
                        # Update the fallback tangent for the next step.
                        self._fallback_tangent = unit.astype(np.float32, copy=False)
                polyline.append(world_xyz.astype(np.float32, copy=False))

                bar.update(1)
                bar.set_postfix(
                    reanchors=reanchors,
                    stop_p=f"{stop_prob:.2f}",
                    z=int(world_xyz[0]),
                    y=int(world_xyz[1]),
                    x=int(world_xyz[2]),
                )

                actual_steps = step_idx + 1
                # 1) explicit stop signal
                if (
                    self.stop_prob_threshold is not None
                    and actual_steps >= self.min_steps
                    and stop_prob >= self.stop_prob_threshold
                ):
                    stop_reason = "stop_probability"
                    break
                # 2) leading point exited the volume
                if not self.reader.in_volume_bounds(world_xyz):
                    stop_reason = "out_of_volume"
                    break
                if actual_steps >= self.max_steps:
                    stop_reason = "max_steps"
                    break

                # 3) re-anchor if the leading point is near a face, OR if we've taken
                # too many steps inside the current window (so the position embedding
                # never leaves the model's trained range).
                steps_in_window = step_position - self.prompt_length
                force_reanchor = steps_in_window >= self.max_steps_per_window
                if force_reanchor or self.reader.needs_reanchor(local_xyz):
                    last_k_world = np.asarray(polyline[-self.prompt_length :], dtype=np.float32)
                    # Bring the prompt back into the training distribution:
                    # resample to uniform `snap_stride` arc length, matching
                    # the cache's `wk_xyz_to_new_zyx_resampled_4` convention.
                    if self.snap_stride is not None and self.snap_stride > 0.0:
                        last_k_world = self._resample_prompt(last_k_world, self.snap_stride)
                    # Same edge-anchor strategy as the initial prompt: push the
                    # window forward of the leading point so the rebuilt prompt
                    # sits at one edge of the new crop, matching training.
                    anchor_offset = self._anchor_offset_along_prompt(last_k_world)
                    self.reader.anchor_on(world_xyz + anchor_offset)
                    outputs, cache, step_position = self._init_at_current_anchor(last_k_world)
                    reanchors += 1
                    self._fallback_tangent = self._unit_tangent(last_k_world)
                    if prefetch:
                        forecast = world_xyz + anchor_offset * 2.0
                        self.reader.prefetch_anchor(forecast)
                    continue

                # 4) one cached step.
                next_position = step_position
                step_position = min(step_position + 1, self.max_fiber_position_embeddings - 1)
                outputs, cache = self._cached_step(
                    coarse_id=coarse_id,
                    offset_bins=offset_bins,
                    local_xyz=local_xyz,
                    position=next_position,
                    cache=cache,
                )
            else:
                # for-else fires when the loop completes without break.
                stop_reason = "max_steps"
        finally:
            bar.close()

        polyline_arr = np.asarray(polyline, dtype=np.float32)
        return TraceResult(
            polyline_world_zyx=polyline_arr,
            stop_probabilities=np.asarray(stop_probabilities, dtype=np.float32),
            stop_reason=stop_reason,
            steps=len(stop_probabilities),
            reanchors=reanchors,
        )

    def trace_bidirectional(
        self,
        fiber_world_zyx: np.ndarray,
        *,
        prefetch: bool = True,
        progress: bool = False,
    ) -> BidirectionalResult:
        """Trace forward from the fiber's last `prompt_length` points and
        backward from its first `prompt_length` points (reversed); concatenate
        as ``reverse(backward_extension) + fiber + forward_extension``.

        Backward extension is performed by reversing the prompt — the model
        does not know about direction explicitly, so reversing the prompt
        causes the tracer to continue in the opposite tangent direction.
        """

        fiber = np.asarray(fiber_world_zyx, dtype=np.float32)
        if fiber.ndim != 2 or fiber.shape[1] != 3:
            raise ValueError(f"fiber_world_zyx must have shape (N, 3); got {tuple(fiber.shape)}")
        if fiber.shape[0] < self.prompt_length:
            raise ValueError(
                f"fiber has {fiber.shape[0]} points but prompt_length={self.prompt_length} is required"
            )

        forward = self.trace_one_direction(
            fiber, prefetch=prefetch, progress="forward" if progress else False
        )

        backward_prompt = fiber[: self.prompt_length][::-1].copy()
        backward = self.trace_one_direction(
            backward_prompt, prefetch=prefetch, progress="backward" if progress else False
        )

        # The forward trace polyline is [..fiber, fwd_1, fwd_2, ...]. The backward
        # trace polyline is [reversed_prompt, back_1, back_2, ...] (the reversed
        # prompt = fiber[: prompt_length][::-1]). The new-points portion is
        # backward.polyline[prompt_length:] which extends *before* fiber[0].
        backward_new = backward.polyline_world_zyx[self.prompt_length :]
        backward_extension_reversed = backward_new[::-1]
        combined = np.concatenate([backward_extension_reversed, forward.polyline_world_zyx], axis=0)

        return BidirectionalResult(
            polyline_world_zyx=combined.astype(np.float32, copy=False),
            forward=forward,
            backward=backward,
        )

    # --- internals ----------------------------------------------------- #

    def _init_at_current_anchor(
        self,
        prompt_world_zyx: np.ndarray,
    ) -> tuple[dict, FiberKVCache, int]:
        """Encode the conditioning at the current anchor and prime the KV cache.

        Returns ``(outputs_for_first_step, cache, next_position)`` where
        ``next_position`` is the sequence index that the *next* generated token
        will use as its position embedding input (i.e. the START's position + 1).
        """

        prompt_local = prompt_world_zyx[-self.prompt_length :] - self.reader.min_corner.astype(np.float32)
        prompt_local = prompt_local.astype(np.float32, copy=False)
        coarse_ids, offset_bins, valid_mask = quantize_local_xyz(
            prompt_local,
            volume_shape=self.input_shape,
            patch_size=tuple(self.model.patch_size),
            offset_num_bins=tuple(self.model.offset_num_bins),
        )
        device = self.device
        prompt_tokens = {
            "coarse_ids": torch.from_numpy(coarse_ids.astype(np.int64, copy=False)).to(device).unsqueeze(0),
            "offset_bins": torch.from_numpy(offset_bins.astype(np.int64, copy=False)).to(device).unsqueeze(0),
            "xyz": torch.from_numpy(prompt_local.astype(np.float32, copy=False)).to(device).unsqueeze(0),
            "positions": torch.arange(self.prompt_length, dtype=torch.long, device=device).unsqueeze(0),
            "valid_mask": torch.from_numpy(valid_mask).to(device).unsqueeze(0),
            "mask": torch.ones((1, self.prompt_length), dtype=torch.bool, device=device),
        }
        prompt_anchor_xyz = prompt_tokens["xyz"][:, -1, :]
        prompt_anchor_valid = prompt_tokens["valid_mask"][:, -1]

        volume = torch.from_numpy(self.reader.fetch_crop()).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.dtype, enabled=device.type == "cuda"):
            encoded = self.model.encode_conditioning(volume)
            outputs, cache = self.model.init_kv_cache(
                prompt_tokens=prompt_tokens,
                prompt_anchor_xyz=prompt_anchor_xyz,
                prompt_anchor_valid=prompt_anchor_valid,
                target_start_position=self.prompt_length,
                memory_tokens=encoded["memory_tokens"],
                memory_patch_centers=encoded["memory_patch_centers"],
            )
        self._memory_tokens = encoded["memory_tokens"]  # held for the cached steps
        next_position = self.prompt_length + 1
        return outputs, cache, next_position

    def _sample_step(self, outputs: dict) -> tuple[int, list[int], np.ndarray]:
        """Greedy sample one prediction from the model outputs.

        Returns ``(coarse_id, offset_bins_list, local_xyz_np)``.
        ``local_xyz_np`` includes the refine residual where present, clamped to
        the crop bounds.
        """

        if outputs.get("coarse_logits") is not None:
            coarse_id = int(outputs["coarse_logits"][0, 0].argmax(dim=-1).item())
        else:
            axis_logits = outputs["coarse_axis_logits"]
            z_id = axis_logits["z"][0, 0].argmax(dim=-1)
            y_id = axis_logits["y"][0, 0].argmax(dim=-1)
            x_id = axis_logits["x"][0, 0].argmax(dim=-1)
            coarse_id = int(self.model._flatten_coarse_axis_ids(z_id, y_id, x_id).item())

        offset_bins: list[int] = []
        for axis, bins in enumerate(self.model.offset_num_bins):
            axis_logits = outputs["offset_logits"][0, 0, axis, :bins]
            offset_bins.append(int(axis_logits.argmax(dim=-1).item()))

        coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=self.device)
        offset_tensor = torch.tensor([offset_bins], dtype=torch.long, device=self.device).view(1, 1, 3)
        bin_center_xyz = self.model.decode_local_xyz(coarse_tensor, offset_tensor)[0, 0]
        residual = outputs.get("pred_refine_residual")
        if residual is not None:
            res = residual[0, 0].float()
            patch_diag = float(np.linalg.norm(self.model.patch_size))
            res_norm = float(res.norm().item())
            if res_norm > patch_diag:
                res = res * (patch_diag / max(res_norm, 1e-6))
            local_xyz = (bin_center_xyz + res).float()
        else:
            local_xyz = bin_center_xyz.float()
        crop_max = np.asarray(self.input_shape, dtype=np.float32) - 1e-4
        local_np = local_xyz.detach().to(torch.float32).cpu().numpy()
        local_np = np.clip(local_np, 0.0, crop_max)
        return coarse_id, offset_bins, local_np

    def _cached_step(
        self,
        *,
        coarse_id: int,
        offset_bins: list[int],
        local_xyz: np.ndarray,
        position: int,
        cache: FiberKVCache,
    ) -> tuple[dict, FiberKVCache]:
        device = self.device
        next_coarse = torch.tensor([[int(coarse_id)]], dtype=torch.long, device=device)
        next_offset = torch.tensor([offset_bins], dtype=torch.long, device=device).view(1, 1, 3)
        next_xyz = torch.from_numpy(local_xyz.astype(np.float32, copy=False)).view(1, 1, 3).to(device)
        next_position = torch.tensor([[int(position)]], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.dtype, enabled=device.type == "cuda"):
            outputs, new_cache = self.model.step_from_encoded_cached(
                next_coarse_ids=next_coarse,
                next_offset_bins=next_offset,
                next_xyz=next_xyz,
                next_position=next_position,
                cache=cache,
                memory_tokens=self._memory_tokens,
            )
        return outputs, new_cache

    @staticmethod
    def _tangent_estimate(polyline: Iterable[np.ndarray]) -> np.ndarray | None:
        polyline_list = list(polyline)
        if len(polyline_list) < 2:
            return None
        a = polyline_list[-1]
        b = polyline_list[-2]
        diff = (a - b).astype(np.float32)
        norm = float(np.linalg.norm(diff))
        if norm < 1e-6:
            return None
        return diff / norm

    def _quantize_local_xyz(self, local_xyz: np.ndarray) -> tuple[int, list[int]]:
        """Re-quantise a (3,) local xyz back to (coarse_id, [bin_z, bin_y, bin_x]).

        Used after the per-step snap so the model's next-step input has
        coarse/offset tokens that agree with the snapped world position.
        """

        coarse_ids, offset_bins, _valid = quantize_local_xyz(
            local_xyz.reshape(1, 3).astype(np.float32, copy=False),
            volume_shape=tuple(self.input_shape),
            patch_size=tuple(self.model.patch_size),
            offset_num_bins=tuple(self.model.offset_num_bins),
        )
        return int(coarse_ids[0]), [int(v) for v in offset_bins[0]]

    @staticmethod
    def _unit_tangent(polyline_world_zyx: np.ndarray) -> np.ndarray | None:
        """Mean unit-tangent across the polyline's consecutive segments."""

        arr = np.asarray(polyline_world_zyx, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return None
        diffs = np.diff(arr, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        unit = diffs / np.clip(norms, 1e-6, None)
        tangent = unit.mean(axis=0)
        n = float(np.linalg.norm(tangent))
        if n < 1e-6:
            return None
        return (tangent / n).astype(np.float32, copy=False)

    @staticmethod
    def _resample_prompt(prompt_world_zyx: np.ndarray, target_spacing: float) -> np.ndarray:
        """Resample the prompt to exact ``target_spacing`` arc-length spacing.

        Mirrors what ``fiber_geometry.resample_polyline_uniform`` does on the
        cache: produces a polyline where every consecutive pair is exactly
        ``target_spacing`` voxels apart in 3D arc length. Returns the LAST
        ``prompt_length`` resampled points (or the full resample if shorter).
        """

        resampled = resample_polyline_uniform(prompt_world_zyx, target_spacing=float(target_spacing))
        return np.asarray(resampled[-prompt_world_zyx.shape[0] :], dtype=np.float32)

    def _anchor_offset_along_prompt(self, prompt_world_zyx: np.ndarray) -> np.ndarray:
        """Offset from the prompt's leading point to the desired window anchor.

        Reproduces the trainer's window placement. ``AutoregFiberDataset``
        picks ``min_corner`` so the AABB midpoint of all 40 points (8 prompt
        + 32 target) lands at the crop's centre. For a roughly straight
        fiber at the cache's uniform ``stride`` arc-length spacing, that
        AABB midpoint sits exactly ``(target_length - prompt_length + 1) *
        stride / 2`` voxels *forward* of ``prompt[-1]`` along the prompt's
        tangent — which for the standard config (8 prompt, 32 target, 4-voxel
        stride) is 50 voxels. We push the window anchor by that amount along
        the prompt's mean unit-tangent so the rebuilt prompt lands at the
        same in-distribution position the model saw at training time.
        """

        arr = np.asarray(prompt_world_zyx, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            return np.zeros(3, dtype=np.float32)
        # Mean of the per-segment unit tangents — robust to a single noisy
        # point and matches the smoothed tangent the trainer's
        # `_tangent_vectors` averages over the whole prompt.
        diffs = np.diff(arr, axis=0).astype(np.float32)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        unit = diffs / np.clip(norms, 1e-6, None)
        tangent = unit.mean(axis=0)
        tnorm = float(np.linalg.norm(tangent))
        if tnorm < 1e-6:
            return np.zeros(3, dtype=np.float32)
        tangent = tangent / tnorm
        target_length = int(self.model.config["target_length"])
        # Use the actual cache stride if known (snap_stride matches the
        # cache's densify_step); otherwise estimate from the prompt itself
        # so the formula still holds for non-uniform-cache inputs.
        if self.snap_stride is not None and self.snap_stride > 0.0:
            stride = float(self.snap_stride)
        else:
            stride = float(np.mean(norms.flatten()))
            if stride < 1e-3:
                stride = float(min(self.reader.crop_size)) / float(target_length + self.prompt_length)
        # AABB midpoint distance forward of prompt[-1].
        push = (target_length - self.prompt_length + 1) * stride * 0.5
        # Never push so far that the prompt would land outside the window.
        max_push = float(min(self.reader.crop_size)) * 0.5
        push = float(min(push, max_push))
        return (tangent * push).astype(np.float32)


__all__ = ["BidirectionalResult", "FiberTracer", "TraceResult"]
