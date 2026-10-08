from concurrent.futures import Future
from types import SimpleNamespace
import os
import time
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.demand_gpu import DemandGpuLease, DemandCohort


def test_idle_timer_slow_read_and_demand_unpause():
    calls = []
    future = Future()
    lease = DemandGpuLease.__new__(DemandGpuLease)
    lease.started = True
    now = [0.]
    lease.clock = lambda: now[0]
    lease.pool = SimpleNamespace(active={}, loading={1: ('s', (), 0, future, 0)}, last_gpu_activity=0)
    lease.idle_paused = False
    lease.last_status = None
    supervisor = SimpleNamespace(connected=True, stopping=False, next_poll=1,
        client=SimpleNamespace(pause=lambda flag: calls.append(flag), task_id='test'),
        tick=lambda: None, error=None, status='running')
    lease.supervisor = supervisor
    now[0] = 29.99
    lease.tick()
    assert not calls
    now[0] = 30
    lease.tick()
    assert calls == [True]
    now[0] = 60
    lease.tick()
    assert calls == [True]  # slow read doesn't cause regrant churn
    future.set_result(None)
    lease.tick()
    assert calls == [True, False]
    # Only self pause is touched; no administrative set_paused operation.


def test_cache_extension_validates_parameters_not_code_hashes():
    from vesuvius.neural_tracing.fiber_trace_3d.inference_cache import validate_cached_model
    model = dict(checkpoint_sha256='same', precision='bf16', tile=256, scale=4,
                 max_batch_size=8, embedding_channels='signed-float32-v2')
    settings = dict(precision='bf16', tile_size=256, scaledown_power=2,
                    max_batch_size=8, emit_embeddings=True)
    for record in (model, dict(model, implementation={'runner': 'old code'})):
        validate_cached_model(record, 'same', settings)
        resolved = validate_cached_model(record, 'same', {'precision':'auto'})
        assert resolved['precision'] == 'bf16'
        assert resolved['max_batch_size'] == 8
        assert resolved['emit_embeddings'] is True
        with pytest.raises(ValueError, match='checkpoint'):
            validate_cached_model(record, 'changed', settings)
        for key, value in [('precision','float32'), ('tile_size',128),
                           ('scaledown_power',1), ('max_batch_size',4), ('emit_embeddings',False)]:
            with pytest.raises(ValueError):
                validate_cached_model(record, 'same', dict(settings, **{key:value}))


def test_initial_request_waits_for_ready_input(monkeypatch):
    import vesuvius.neural_tracing.fiber_trace_3d.demand_gpu as module
    calls = []
    monkeypatch.setattr(module, 'resolve_gpu_uuids', lambda devices: ['GPU-test'])
    monkeypatch.setattr(module, 'ensure_manager', lambda socket: (calls.append('bootstrap') or 'test.sock', None))
    monkeypatch.setattr(module, 'Client', lambda socket: SimpleNamespace(
        close=lambda: calls.append('close'), task_id='test'))
    def supervisor(client, **kwargs):
        return SimpleNamespace(client=client, connected=False, stopping=False,
            tick=lambda: calls.append('register'), error=None, status='waiting')
    monkeypatch.setattr(module, 'Supervisor', supervisor)
    future = Future()
    pool = SimpleNamespace(active={}, loading={1: ('s', (), 0, future, 0)}, last_gpu_activity=0)
    lease = DemandGpuLease(pool, dict(vram_bytes_per_gpu=4 << 30), clock=lambda: 100)
    for _ in range(3):
        lease.tick()
    assert calls == [] and not lease.started
    lease.close()
    assert calls == ['close']  # Closing unused lease must not register either.
    calls.clear()
    lease = DemandGpuLease(pool, dict(vram_bytes_per_gpu=4 << 30), clock=lambda: 100)
    future.set_result(None)
    lease.tick()
    assert calls == ['bootstrap', 'register'] and lease.started


def test_drain_waits_for_submitted_tiles_but_not_cpu_results():
    calls = []
    pool = SimpleNamespace(start_gpu=lambda *a, **k: calls.append('start'),
        authorize_gpu=lambda: calls.append('authorize'), poll_stop_gpu=lambda: calls.append('join') or True,
        active={1: 'tile'}, input_owners={}, accumulating={1: 'cpu-result'}, gpu_authorized=True)
    cohort = DemandCohort(pool, {'gpus': {'GPU-test': 10}})
    cohort.authorize()
    cohort.drain()
    assert not pool.gpu_authorized
    cohort.poll()
    assert not cohort.closed
    pool.active.clear()
    cohort.poll()
    assert cohort.closed and calls == ['start', 'authorize', 'join']
    assert pool.accumulating  # CPU accumulation survives GPU join


def test_exclusive_cohort_checks_external_occupancy_before_authorize(monkeypatch):
    import vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation as managed
    calls = []
    pool = SimpleNamespace(
        processes=[],
        start_gpu=lambda devices, **kw: calls.append(("start", tuple(devices))),
        stop_gpu=lambda: calls.append(("stop",)),
    )
    monkeypatch.setattr(managed, "assert_gpus_idle",
                        lambda devices: calls.append(("idle", tuple(devices))))
    cohort = DemandCohort(pool, {"gpus": {"GPU-b": 20, "GPU-a": 10}},
                          exclusive=True)
    assert cohort.gpu_uuids == ("GPU-a", "GPU-b")
    assert calls == [("start", ("GPU-a", "GPU-b")),
                     ("idle", ("GPU-a", "GPU-b"))]
    calls.clear()
    DemandCohort(pool, {"gpus": {"GPU-a": 10}}, exclusive=False)
    assert calls == [("start", ("GPU-a",))]


def test_exclusive_occupancy_failure_stops_gated_workers(monkeypatch):
    import vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation as managed
    calls = []
    pool = SimpleNamespace(
        processes=[],
        start_gpu=lambda *a, **k: calls.append("start"),
        stop_gpu=lambda: calls.append("stop"),
    )
    monkeypatch.setattr(managed, "assert_gpus_idle",
                        lambda devices: (_ for _ in ()).throw(RuntimeError("occupied")))
    with pytest.raises(RuntimeError, match="occupied"):
        DemandCohort(pool, {"gpus": {"GPU-a": 10}}, exclusive=True)
    assert calls == ["start", "stop"]


@pytest.mark.skipif(os.environ.get('FIBER_DEMAND_GPU_SMOKE') != '1', reason='explicit manager GPU smoke only')
def test_live_manager_idle_reacquire(tmp_path):
    import numpy as np
    import zarr
    from lasagna.tests.test_demand_predict3d import outputs
    from lasagna.demand_predict3d import DemandGeometry, DemandStore, DemandWorkerPool
    geometry = DemandGeometry((8,4,4), 4, 0, 0, 1, 4)
    adapter, output = outputs(tmp_path, geometry.shape, 4)
    with DemandStore(tmp_path, {'smoke': True}, geometry, adapter, output) as store:
        with DemandWorkerPool({'s': store}, {'s': lambda tile: np.full((4,4,4), 255, np.uint8)},
                              ['cuda:0'], deferred_gpu=True, input_workers=1) as pool:
            lease = DemandGpuLease(pool, dict(exclusive=False, priority=-100,
                cpu_cores=1, vram_bytes_per_gpu=4 << 30, label='demand-pause-smoke'))
            try:
                for _ in range(3):
                    lease.tick()
                    pool.pump()
                assert not lease.started and not pool.processes
                assert lease.supervisor.client.task_id is None
                for z in range(2):
                    if z == 1:
                        lease.supervisor.client.task_call('set_paused', paused=True, scope='controller')
                    store.request((z,0,0))
                    if z == 1:
                        deadline = time.monotonic() + 1
                        while time.monotonic() < deadline:
                            lease.tick()
                            pool.pump()
                            time.sleep(.01)
                        assert not pool.processes and pool.loading
                        assert any(row[3].done() for row in pool.loading.values())
                        assert lease.supervisor.client.task['paused']['controller']
                        lease.supervisor.client.task_call('set_paused', paused=False, scope='controller')
                    deadline = time.monotonic() + 90
                    while pool.busy and time.monotonic() < deadline:
                        lease.tick()
                        pool.pump()
                        time.sleep(.01)
                    if pool.busy and lease.supervisor.cohort is None:
                        pytest.skip('manager capacity unavailable within 90s; no GPU work authorized')
                    assert not pool.busy
                    # Fake only elapsed idle time, not manager/worker lifecycle.
                    pool.last_gpu_activity = time.monotonic() - 31
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        lease.tick()
                        pool.pump()
                        if lease.supervisor.cohort is None and lease.supervisor.status == 'paused':
                            break
                        time.sleep(.01)
                    assert not pool.processes and lease.supervisor.status == 'paused'
                    assert lease.supervisor.client.task['held']['gpus'] == {}
                np.testing.assert_array_equal(zarr.open_array(str(tmp_path / 'value.zarr/0'), mode='r')[:],
                                              np.full(geometry.shape, 255, np.uint8))
            finally:
                lease.close()


def test_cleanup_aborts_on_pump_error_and_still_reconciles():
    lease = DemandGpuLease.__new__(DemandGpuLease)
    lease.started = True
    calls = []
    supervisor = SimpleNamespace(error=None, finished=False, stop=lambda: calls.append('stop'))
    def fail(error):
        supervisor.error = error
        calls.append('abort-and-join')
    supervisor.fail = fail
    def tick():
        supervisor.finished = True
        calls.append('release')
    supervisor.tick = tick
    lease.supervisor = supervisor
    lease.clock = lambda: 0
    def broken_pump():
        raise RuntimeError('read failed')
    lease.pool = SimpleNamespace(pump=broken_pump)
    lease.close()
    assert calls == ['stop', 'abort-and-join', 'release']


def test_idle_release_threshold_is_configurable(monkeypatch):
    from types import SimpleNamespace
    from concurrent.futures import Future
    from vesuvius.neural_tracing.fiber_trace_3d import demand_gpu as module
    import pytest
    monkeypatch.setattr(module, 'resolve_gpu_uuids', lambda devices: ['GPU-test'])
    monkeypatch.setattr(module, 'ensure_manager', lambda socket: ('/sock', None))
    monkeypatch.setattr(module, 'Client', lambda path: SimpleNamespace(path=path, close=lambda: None,
                                                                   pause=lambda idle: paused.append(idle),
                                                                   task_id='t'))
    monkeypatch.setattr(module, 'Supervisor', lambda client, **kwargs: SimpleNamespace(
        client=client, connected=True, stopping=False, tick=lambda: None, error=None, status='running'))
    paused = []
    future = Future(); future.set_result(None)
    pool = SimpleNamespace(active={}, loading={1: ('s', (), 0, future, 0)}, last_gpu_activity=0)
    now = [0]
    lease = DemandGpuLease(pool, dict(vram_bytes_per_gpu=4 << 30, idle_release_seconds=600), clock=lambda: now[0])
    lease.tick()  # registers; a ready read means GPU work is needed
    pool.loading = {}
    now[0] = 100
    lease.tick()
    assert paused == []  # 100 s idle is below the configured 600 s
    now[0] = 700
    lease.tick()
    assert paused == [True]
    with pytest.raises(ValueError, match='idle_release_seconds'):
        DemandGpuLease(pool, dict(vram_bytes_per_gpu=4 << 30, idle_release_seconds=0))
