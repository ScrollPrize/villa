from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pytest

from vesuvius.neural_tracing.ordered_sample_pipeline import OrderedTwoStageSamplePipeline


@dataclass(frozen=True)
class _StageOne:
    slow_first: bool = False
    hard_exit_index: int = -1
    hard_exit_code: int = 7

    def __call__(self, index: int) -> int:
        if index == self.hard_exit_index:
            os._exit(self.hard_exit_code)
        if self.slow_first and index == 0:
            time.sleep(0.15)
        elif self.slow_first:
            time.sleep(0.005)
        return index + 10


@dataclass(frozen=True)
class _StageOneFactory:
    slow_first: bool = False
    hard_exit_index: int = -1
    hard_exit_code: int = 7

    def __call__(self) -> _StageOne:
        return _StageOne(self.slow_first, self.hard_exit_index, self.hard_exit_code)


@dataclass(frozen=True)
class _StageTwo:
    fail_value: int = -1

    def __call__(self, value: int) -> int:
        if value == self.fail_value:
            raise ValueError(f"rejected {value}")
        return value * 2


@dataclass(frozen=True)
class _StageTwoFactory:
    fail_value: int = -1

    def __call__(self) -> _StageTwo:
        return _StageTwo(self.fail_value)


def _pipeline(**overrides: object) -> OrderedTwoStageSamplePipeline:
    args: dict[str, object] = {
        "stage1_factory": _StageOneFactory(),
        "stage2_factory": _StageTwoFactory(),
        "start_index": 0,
        "item_count": 12,
        "stage1_workers": 4,
        "stage2_workers": 3,
        "raw_queue_capacity": 1,
        "result_queue_capacity": 1,
        "max_in_flight": 6,
        "multiprocessing_context": "spawn",
        "startup_timeout_seconds": 30.0,
        "no_progress_timeout_seconds": 5.0,
    }
    args.update(overrides)
    return OrderedTwoStageSamplePipeline(**args)  # type: ignore[arg-type]


def test_pipeline_drains_out_of_order_results_and_delivers_in_order() -> None:
    pipeline = _pipeline(stage1_factory=_StageOneFactory(slow_first=True))
    try:
        results = [pipeline.next_ordered() for _ in range(12)]
        assert [result.index for result in results] == list(range(12))
        assert [result.value for result in results] == [(index + 10) * 2 for index in range(12)]
        assert 1 < pipeline.reorder_highwater <= 6
        assert len({result.stage1_worker for result in results}) > 1
    finally:
        processes = list(pipeline.processes)
        pipeline.close()
    del processes


def test_pipeline_propagates_stage_exception_and_cleans_up() -> None:
    pipeline = _pipeline(stage2_factory=_StageTwoFactory(fail_value=13))
    processes = list(pipeline.processes)
    with pytest.raises(RuntimeError, match=r"augment worker .*ValueError: rejected 13"):
        try:
            while True:
                pipeline.next_ordered()
        finally:
            pipeline.close()
    del processes


def test_pipeline_detects_hard_worker_exit() -> None:
    pipeline = _pipeline(
        stage1_factory=_StageOneFactory(hard_exit_index=2),
        item_count=8,
    )
    processes = list(pipeline.processes)
    with pytest.raises(RuntimeError, match=r"exited unexpectedly with code 7"):
        try:
            while True:
                pipeline.next_ordered()
        finally:
            pipeline.close()
    del processes


def test_pipeline_detects_unexpected_clean_worker_exit() -> None:
    pipeline = _pipeline(
        stage1_factory=_StageOneFactory(hard_exit_index=2, hard_exit_code=0),
        item_count=8,
    )
    processes = list(pipeline.processes)
    with pytest.raises(RuntimeError, match=r"exited unexpectedly with code 0"):
        try:
            while True:
                pipeline.next_ordered()
        finally:
            pipeline.close()
    del processes


def test_pipeline_early_close_with_saturated_queues_is_idempotent() -> None:
    pipeline = _pipeline(item_count=100, max_in_flight=8)
    time.sleep(0.2)
    pipeline.close()
    pipeline.close()
    assert pipeline.closed
