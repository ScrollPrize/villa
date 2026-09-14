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


def test_compatibility_is_exact_not_a_fingerprint_bypass():
    from vesuvius.neural_tracing.fiber_trace_3d.inference_compatibility import compatible_implementation, SCHEDULING_ONLY
    old = {k: v[0] for k, v in SCHEDULING_ONLY.items()}
    new = {k: v[1] for k, v in SCHEDULING_ONLY.items()}
    assert compatible_implementation(old, new)
    assert not compatible_implementation(old, dict(new, model='different'))
    assert not compatible_implementation(old, dict(new, runner='unreviewed edit'))


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
