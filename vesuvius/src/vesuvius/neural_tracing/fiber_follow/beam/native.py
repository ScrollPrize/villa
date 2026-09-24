"""The volume-cartographer beam tracer, driven from fiber_follow trace-grid units.

Everything the C++ tracer does (cone candidates, hand loss, lookahead, pruning,
target planes, meeting fusion, restarts) stays in ``vc.fiber_trace``. This
module only converts coordinates (fiber_follow grid voxels <-> VC trace voxels)
and hands the tracer's prune-time candidate pools to Python hooks.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BeamSpec:
    """Which prediction/normal datasets the beam reads and how it is configured."""
    prediction_manifest: str
    normal_manifest: str | None = None
    cache_bytes: int = 512 << 20
    scaledown_power: int = 2
    config: dict = field(default_factory=dict)  # ``TraceConfig`` overrides, VC key names
    hook_every_rounds: int = 4
    hook_pool_size: int = 32
    parallel_threads: int = 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "BeamSpec":
        return cls(**values)


@dataclass
class Pool:
    """One prune-time candidate pool in fiber_follow grid coordinates (xyz)."""
    round: int
    step: int
    phase: str
    start: np.ndarray
    target: np.ndarray
    paths: list  # (n_i, 3) arrays from the trace start through each candidate endpoint
    losses: np.ndarray  # float32 cumulative hand loss, ascending
    depth: np.ndarray
    traced_length: np.ndarray
    reached: np.ndarray
    step_directions: np.ndarray  # (P, 3) unit direction of each candidate's last step

    def __len__(self):
        return len(self.losses)


@dataclass
class SpanResult:
    points: np.ndarray  # grid xyz, start first
    reached: bool
    reason: str
    steps: int
    endpoint_error_base: float


def import_fiber_trace():
    try:
        from vc import fiber_trace
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError('vc.fiber_trace is required for the beam method; rebuild and reinstall '
                          'volume-cartographer with VC_BUILD_PYTHON=ON') from exc
    return fiber_trace


def fiber_input_from_traced(fiber, grid_scale: float):
    """``FiberInput`` (base voxels) for a loaded ``TracedFiber``: its controlled
    line plus the control points recorded as span boundaries."""
    ft = import_fiber_trace()
    arcs = [span.start for span in fiber.spans] + [fiber.spans[-1].end]
    indices = []
    for arc in arcs:
        index = int(np.abs(fiber.s - arc).argmin())
        if abs(fiber.s[index] - arc) > 1e-6:
            raise ValueError(f'{fiber.name}: control arclength {arc} is not a line vertex')
        indices.append(index)
    line = np.asarray(fiber.points, np.float64) * grid_scale
    return ft.FiberInput(line, line[indices], np.asarray(indices, np.int64), fiber.name)


class NativeBeam:
    """Lazily opened prediction field + normal sampler, one per process.

    Open it inside the worker that uses it: the underlying readers keep a
    process-global thread pool and chunk cache that must not cross a fork.
    """

    def __init__(self, spec: BeamSpec, grid_scale: float):
        self.spec = spec
        self.grid_scale = float(grid_scale)
        self._ft = None
        self._field = None
        self._normals = None

    def _ensure(self):
        if self._field is None:
            ft = import_fiber_trace()
            self._field = ft.open_prediction_field(self.spec.prediction_manifest, cache_bytes=self.spec.cache_bytes,
                                                  scaledown_power=self.spec.scaledown_power)
            if self.spec.normal_manifest:
                self._normals = ft.open_normal_sampler(self.spec.normal_manifest, self._field.trace_to_base_scale,
                                                      cache_bytes=self.spec.cache_bytes)
            self._ft = ft
        return self._ft

    @property
    def trace_to_base(self) -> float:
        self._ensure()
        return float(self._field.trace_to_base_scale)

    @property
    def grid_to_trace(self) -> float:
        return self.grid_scale / self.trace_to_base

    def config(self):
        ft = self._ensure()
        values = dict(self.spec.config)
        values.setdefault('parallel_threads', self.spec.parallel_threads)
        values['trace_to_base_scale'] = self.trace_to_base
        return ft.TraceConfig(**values)

    def to_trace(self, grid):
        return np.asarray(grid, np.float64) * self.grid_to_trace

    def to_grid(self, trace):
        return np.asarray(trace, np.float64) / self.grid_to_trace

    def _pool(self, raw) -> Pool:
        scale = self.grid_to_trace
        return Pool(round=int(raw.round), step=int(raw.step), phase=str(raw.phase),
                    start=np.asarray(raw.start) / scale, target=np.asarray(raw.target) / scale,
                    paths=[np.asarray(p) / scale for p in raw.paths()],
                    losses=np.asarray(raw.losses, np.float32), depth=np.asarray(raw.depth),
                    traced_length=np.asarray(raw.traced_length) / scale, reached=np.asarray(raw.reached, bool),
                    step_directions=np.asarray(raw.previous_step_direction))

    def _hook_kwargs(self, hook):
        if hook is None:
            return dict(hook=None, hook_every_rounds=self.spec.hook_every_rounds, hook_pool_size=self.spec.hook_pool_size)

        def wrapped(raw):
            return hook(self._pool(raw))
        return dict(hook=wrapped, hook_every_rounds=self.spec.hook_every_rounds, hook_pool_size=self.spec.hook_pool_size)

    def trace_span(self, line_grid, start_index: int, target_index: int, *, start_point=None,
                   initial_direction=None, hook=None, accept_threshold_base: float | None = None) -> SpanResult:
        """One-way trace toward ``line[target_index]`` with VC's target planes.

        The start may be perturbed away from the line (``start_point`` /
        ``initial_direction``); the planes still come from the line indices.
        """
        ft = self._ensure()
        line = self.to_trace(line_grid)
        start = line[start_index] if start_point is None else self.to_trace(start_point)
        target = line[target_index]
        if initial_direction is None:
            direction = np.asarray(ft.reference_tangent_toward(line, start_index, target_index))
        else:
            direction = np.asarray(initial_direction, np.float64)
        planes = ft.target_local_planes(self._field, line, target_index, start_index, target)
        span = float(np.linalg.norm(target - line[start_index]))
        cfg = self.config()
        if accept_threshold_base is None:
            accept_threshold_base = ft.effective_endpoint_accept_threshold_base_voxels(cfg, span * self.trace_to_base)
        result = ft.trace_one_way(self._field, start, target, direction, planes,
                                  accept_threshold_voxels=accept_threshold_base / self.trace_to_base,
                                  budget_span_voxels=span, config=cfg, normal_sampler=self._normals,
                                  snap_trace_to_selected_crossing=False, **self._hook_kwargs(hook))
        return SpanResult(points=self.to_grid(result.points), reached=bool(result.reached_target_plane),
                          reason=result.reason, steps=int(result.steps),
                          endpoint_error_base=float(result.selected_target_plane_error_voxels) * self.trace_to_base)

    def trace_segment(self, line_grid, start_index: int, target_index: int, hook=None):
        """Bidirectional span trace with meeting fusion, as the VC3D window does."""
        ft = self._ensure()
        result = ft.trace_segment(self._field, self.to_trace(line_grid), start_index, target_index, self.config(),
                                  normal_sampler=self._normals, **self._hook_kwargs(hook))
        return dict(points=self.to_grid(result.fused_line), accepted=bool(result.accepted), reason=result.reason,
                    detail=result.detail, meeting_error_base=float(result.meeting_error_base_voxels))

    def trace_open(self, start_grid, direction, distance_grid: float, hook=None):
        """Open-ended trace (no target planes) for ``distance_grid`` grid voxels."""
        ft = self._ensure()
        result = ft.trace_extrapolation(self._field, self.to_trace(start_grid), np.asarray(direction, np.float64),
                                        float(distance_grid) * self.grid_to_trace, self.config(),
                                        normal_sampler=self._normals, **self._hook_kwargs(hook))
        return self.to_grid(result.points), result.reason, bool(result.reached_trace_length)

    def whole_fiber_metric(self, fiber, hook=None, error_threshold_base: float = 20.0) -> dict:
        """VC's restart metric: chain control point to control point, restarting at a miss.

        ``fiber`` is a ``FiberInput`` (base voxels) or a fiber JSON path.
        """
        ft = self._ensure()
        if isinstance(fiber, str):
            fiber = ft.load_fiber_json(fiber)
        result = ft.trace_whole_fiber_metric(self._field, fiber, working_to_base_scale=self.trace_to_base,
                                             error_threshold_base_voxels=error_threshold_base, config=self.config(),
                                             normal_sampler=self._normals, **self._hook_kwargs(hook))
        return dict(restart_count=int(result.restart_count), segment_count=int(result.segment_count),
                    restarts_per_kvx=float(result.restarts_per_kvx),
                    reference_length_grid=float(result.reference_length_voxels) / self.grid_to_trace,
                    segments=[dict(success=bool(seg.success), reason=seg.reason,
                                   in_plane_error_base=float(seg.in_plane_error_base_voxels),
                                   arc_grid=float(seg.reference_arc_distance_voxels) / self.grid_to_trace)
                              for seg in result.segments],
                    stitched=self.to_grid(result.stitched_trace))
