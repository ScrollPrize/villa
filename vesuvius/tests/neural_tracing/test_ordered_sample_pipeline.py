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
    hard_exit_value: int = -1

    def __call__(self, value: int) -> int:
        if value == self.hard_exit_value:
            os._exit(9)
        if value == self.fail_value:
            raise ValueError(f"rejected {value}")
        return value * 2


@dataclass(frozen=True)
class _StageTwoFactory:
    fail_value: int = -1
    hard_exit_value: int = -1

    def __call__(self) -> _StageTwo:
        return _StageTwo(self.fail_value, self.hard_exit_value)


@dataclass(frozen=True)
class _FailingFactory:
    def __call__(self) -> _StageOne:
        raise RuntimeError("factory startup rejected")


@dataclass(frozen=True)
class _TupleTransport:
    slot_count: int = 6

    def _location(self, index: int) -> tuple[int, int]:
        return index % self.slot_count, index // self.slot_count

    def encode(self, index: int, value: int, *, final: bool) -> tuple[int, bool, int]:
        return int(index), bool(final), int(value)

    def decode(
        self, index: int, payload: tuple[int, bool, int], *, final: bool
    ) -> int:
        payload_index, payload_final, value = payload
        assert payload_index == index
        assert payload_final is final
        return value


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


def test_elastic_shared_pool_parks_resumes_and_one_worker_finishes_both_stages():
    with _pipeline(stage1_workers=0, stage2_workers=0, shared_workers=4,
                   shared_stage1_affinity_workers=3, elastic_workers=True,
                   initial_active_workers=0, item_count=30) as pipeline:
        pids = [p.pid for p in pipeline.processes]
        assert pipeline.workers_within_limit()
        assert pipeline.try_next_ordered() is None
        pipeline.set_worker_limit(4)
        first = pipeline.next_ordered()
        assert first.index == 0
        pipeline.set_worker_limit(0)
        deadline = time.monotonic() + 10
        while not pipeline.workers_within_limit():
            pipeline.poll_ready()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        frontier = int(pipeline.next_index.value)
        time.sleep(0.05)
        assert int(pipeline.next_index.value) == frontier
        assert pipeline.received <= pipeline.max_in_flight + 1
        pipeline.set_worker_limit(1)
        results = [first] + [pipeline.next_ordered() for _ in range(29)]
        assert [r.index for r in results] == list(range(30))
        assert [r.value for r in results] == [(i + 10) * 2 for i in range(30)]
        assert [p.pid for p in pipeline.processes] == pids


def test_try_next_ordered_keeps_pending_samples_and_lease_credit():
    with _pipeline(stage1_factory=_StageOneFactory(slow_first=True),
                   stage1_workers=0, stage2_workers=0, shared_workers=2,
                   elastic_workers=True, shared_stage1_affinity_workers=1) as pipeline:
        result = None
        deadline = time.monotonic() + 10
        while result is None:
            result = pipeline.try_next_ordered(release_credit=False)
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result.index == 0 and pipeline.leased == {0}
        pipeline.release(0)
        assert [pipeline.next_ordered().index for _ in range(11)] == list(range(1, 12))


def _shared_pipeline(**overrides: object) -> OrderedTwoStageSamplePipeline:
    args: dict[str, object] = {
        "stage1_factory": _StageOneFactory(),
        "stage2_factory": _StageTwoFactory(),
        "start_index": 0,
        "item_count": 12,
        "stage1_workers": 0,
        "stage2_workers": 0,
        "shared_workers": 4,
        "initial_in_flight": 1,
        "raw_queue_capacity": 2,
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


def test_pipeline_can_deliver_stage_one_values_without_second_stage() -> None:
    pipeline = _pipeline(
        stage2_factory=None,
        stage2_workers=0,
        stage1_factory=_StageOneFactory(slow_first=True),
    )
    try:
        results = [pipeline.next_ordered() for _ in range(12)]
        assert [result.index for result in results] == list(range(12))
        assert [result.value for result in results] == [index + 10 for index in range(12)]
        assert all(result.stage2_worker == -1 for result in results)
        assert all(result.stage2_service_ms == 0.0 for result in results)
        assert 1 < pipeline.reorder_highwater <= 6
    finally:
        pipeline.close()


def test_pipeline_rejects_inconsistent_second_stage_configuration() -> None:
    with pytest.raises(ValueError, match="stage2_factory must be None"):
        _pipeline(stage2_factory=None, stage2_workers=1)
    with pytest.raises(ValueError, match="stage2_factory must be None"):
        _pipeline(stage2_factory=_StageTwoFactory(), stage2_workers=0)


def test_pipeline_transport_round_trip_and_deferred_credit_release() -> None:
    pipeline = _pipeline(
        item_count=4,
        max_in_flight=2,
        transport=_TupleTransport(),
    )
    try:
        first = pipeline.next_ordered(release_credit=False)
        second = pipeline.next_ordered(release_credit=False)
        assert first.value == 20
        assert second.value == 22
        assert pipeline.leased == {0, 1}
        pipeline.release(first.index)
        third = pipeline.next_ordered(release_credit=False)
        assert third.value == 24
        with pytest.raises(RuntimeError, match="is not leased"):
            pipeline.release(first.index)
        pipeline.release(second.index)
        pipeline.release(third.index)
        fourth = pipeline.next_ordered()
        assert fourth.value == 26
        assert not pipeline.leased
    finally:
        pipeline.close()


def test_shared_pool_gradually_admits_and_delivers_exactly_once() -> None:
    pipeline = _shared_pipeline(stage1_factory=_StageOneFactory(slow_first=True))
    try:
        results = [pipeline.next_ordered() for _ in range(12)]
        pipeline.close()
        assert [result.index for result in results] == list(range(12))
        assert [result.value for result in results] == [
            (index + 10) * 2 for index in range(12)
        ]
        assert sum(pipeline.shared_stage1_counts.values()) == 12
        assert sum(pipeline.shared_stage2_counts.values()) == 12
        assert 1 < pipeline.admission_highwater <= 6
        assert pipeline.shared_stage1_counts
        assert pipeline.shared_stage2_counts
    finally:
        pipeline.close()


def test_shared_pool_handles_short_final_growth_and_small_stream() -> None:
    pipeline = _shared_pipeline(item_count=3, initial_in_flight=6)
    try:
        values = [pipeline.next_ordered().value for _ in range(3)]
        pipeline.close()
        assert values == [20, 22, 24]
        assert pipeline.admission_highwater == 3
    finally:
        pipeline.close()


def test_one_shared_worker_can_execute_both_runnable_stages() -> None:
    pipeline = _shared_pipeline(
        shared_workers=1,
        initial_in_flight=1,
        item_count=3,
        raw_queue_capacity=1,
    )
    try:
        values = [pipeline.next_ordered().value for _ in range(3)]
        pipeline.close()
        assert values == [20, 22, 24]
        assert pipeline.shared_stage1_counts == {0: 3}
        assert pipeline.shared_stage2_counts == {0: 3}
    finally:
        pipeline.close()


def test_shared_pool_propagates_each_stage_failure_with_index() -> None:
    for stage1_factory, stage2_factory, match in (
        (_StageOneFactory(hard_exit_index=-1), _StageTwoFactory(fail_value=13), r"augment.*sample 3"),
        (_StageOneFactory(hard_exit_index=3), _StageTwoFactory(), r"exited unexpectedly.*code 7"),
        (_StageOneFactory(), _StageTwoFactory(hard_exit_value=13), r"exited unexpectedly.*code 9"),
    ):
        pipeline = _shared_pipeline(
            stage1_factory=stage1_factory,
            stage2_factory=stage2_factory,
            item_count=8,
        )
        with pytest.raises(RuntimeError, match=match):
            try:
                while True:
                    pipeline.next_ordered()
            finally:
                pipeline.close()


def test_shared_pool_reports_factory_startup_failure() -> None:
    with pytest.raises(RuntimeError, match=r"startup worker .*factory startup rejected"):
        _shared_pipeline(stage1_factory=_FailingFactory())


def test_shared_pool_deferred_release_and_saturated_close() -> None:
    pipeline = _shared_pipeline(item_count=100, max_in_flight=2, raw_queue_capacity=1)
    try:
        first = pipeline.next_ordered(release_credit=False)
        second = pipeline.next_ordered(release_credit=False)
        assert pipeline.leased == {0, 1}
        pipeline.release(second.index)
        pipeline.release(first.index)
    finally:
        pipeline.close()
        pipeline.close()


def test_shared_pool_does_not_reuse_a_leased_transport_slot() -> None:
    pipeline = _shared_pipeline(
        item_count=4,
        max_in_flight=2,
        initial_in_flight=2,
        transport=_TupleTransport(slot_count=2),
    )
    try:
        first = pipeline.next_ordered(release_credit=False)
        second = pipeline.next_ordered(release_credit=False)
        pipeline.release(second.index)
        time.sleep(0.05)
        # Index 2 maps to the slot still leased by index 0, so generic credit
        # availability alone must not permit its dispatch.
        assert sum(int(value) for value in pipeline._shared_stage1_count_array) == 2
        pipeline.release(first.index)
        assert pipeline.next_ordered().value == 24
        assert pipeline.next_ordered().value == 26
    finally:
        pipeline.close()


def test_shared_pool_validates_configuration() -> None:
    with pytest.raises(ValueError, match="must be 0 in shared-pool mode"):
        _shared_pipeline(stage1_workers=1)
    with pytest.raises(ValueError, match="requires stage2_factory"):
        _shared_pipeline(stage2_factory=None)
    with pytest.raises(ValueError, match="initial_in_flight"):
        _shared_pipeline(initial_in_flight=7)
    with pytest.raises(ValueError, match="slot_count"):
        _shared_pipeline(transport=_TupleTransport(slot_count=5))
