from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ink.core.types import EvalReport, ModelOutputBatch
from ink.recipes.metrics import merge_metric_reports


@dataclass(kw_only=True)
class PatchEval:
    metrics: tuple[Any, ...] = ()
    n_groups: int | None = None
    _states: tuple[Any, ...] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.metrics = tuple(self.metrics)
        if self.n_groups is not None:
            self.n_groups = int(self.n_groups)

    def build(self, *, data, runtime=None, logger=None) -> PatchEval:
        if not self.metrics:
            raise ValueError("PatchEval requires at least one metric")
        group_counts = getattr(data, "group_counts", None)
        return replace(
            self,
            metrics=tuple(
                metric.build(data=data, runtime=runtime, logger=logger)
                if callable(getattr(metric, "build", None))
                else metric
                for metric in self.metrics
            ),
            n_groups=None if group_counts is None else int(len(group_counts)),
        )

    def _require_states(self) -> tuple[Any, ...]:
        states = self._states
        if states is None:
            raise ValueError("PatchEval requires begin_epoch() before observe_batch/finalize_epoch")
        return states

    def begin_epoch(self) -> None:
        if not self.metrics:
            raise ValueError("PatchEval requires at least one metric")
        self._states = tuple(metric.empty_state(n_groups=self.n_groups) for metric in self.metrics)

    def observe_batch(self, batch: ModelOutputBatch) -> None:
        if not isinstance(batch, ModelOutputBatch):
            raise TypeError("PatchEval requires ModelOutputBatch")
        if batch.targets is None:
            raise ValueError("patch evaluation requires batch.y")
        states = self._require_states()
        shared = {}
        next_states = []
        for metric, state in zip(self.metrics, states):
            next_states.append(metric.update(state, batch, shared=shared))
        self._states = tuple(next_states)

    def finalize_epoch(self) -> EvalReport:
        states = self._require_states()
        reports = [metric.finalize(state) for metric, state in zip(self.metrics, states)]
        self._states = None
        return merge_metric_reports(reports)
