"""Demand GPU lifetime adapter for the existing cooperative Supervisor.

CPU tracing, reads and accumulation stay with their existing owner. Only GPU
processes are handed off; pending tiles/results never leave the shared pool.
"""
import time

from villa_gpu_manager.client import Client
from villa_gpu_manager.bootstrap import ensure_manager
from villa_gpu_manager.processes import process_identity
from villa_gpu_manager.supervisor import Supervisor

from .managed_evaluation import resolve_gpu_uuids


class DemandCohort:
    ready = True
    idle = False  # The demand loop, not start_unit(), pumps tile work.
    done = False
    state = None

    def __init__(self, pool, grant):
        self.pool = pool
        self.authorized = self.closed = self.draining = False
        pool.start_gpu(sorted(grant["gpus"]),
                       vram_budget=min(grant["gpus"].values()))

    @property
    def processes(self):
        return [process_identity(p.pid) for p in self.pool.processes]

    def authorize(self):
        self.pool.authorize_gpu()
        self.authorized = True

    def start_unit(self):
        raise AssertionError("demand loop owns tile admission")

    def drain(self):
        self.draining = True
        self.pool.gpu_authorized = False

    def poll(self):
        if self.closed:
            return
        if self.draining and not self.pool.active and not self.pool.input_owners:
            self.closed = self.pool.poll_stop_gpu()

    def abort(self):
        # Fatal error invalidates this evaluation; pool.close joins every worker
        # before releasing shared buffers. Do not publish interrupted results.
        self.pool.close()
        self.closed = True


class DemandGpuLease:
    def __init__(self, pool, settings, *, clock=time.monotonic):
        selected = resolve_gpu_uuids(settings.get("devices", "0"))
        budget = settings.get("vram_bytes_per_gpu")
        if type(budget) is not int or budget <= (2 << 30):
            raise ValueError("adaptive inference requires VRAM budget >2GiB")
        exclusive = settings.get("exclusive", False)
        if type(exclusive) is not bool:
            raise ValueError("gpu_manager.exclusive must be boolean")
        self.manager_socket = settings.get("socket")
        request = dict(max_cpu_cores=int(settings.get("cpu_cores", 16)),
                       max_gpus=len(selected), min_vram_bytes_per_gpu=budget,
                       max_vram_bytes_per_gpu=budget, allowed_gpu_uuids=selected,
                       exclusive=exclusive,
                       priority=int(settings.get("priority", 100)))
        self.pool, self.clock = pool, clock
        self.started = False
        self.idle_paused = False
        self.last_status = None
        self.supervisor = Supervisor(Client(self.manager_socket or ""), request=request,
            label=settings.get("label", "demand-inference"),
            cohort_factory=lambda grant, state: DemandCohort(pool, grant),
            clock=clock, poll_interval=.2)

    def tick(self):
        supervisor, pool = self.supervisor, self.pool
        # Slow input I/O is not GPU work. Reacquire only when a read is ready;
        # this also avoids idle pause/reacquire thrashing on queued slow reads.
        need = bool(pool.active or any(row[3].done() for row in pool.loading.values()))
        if not self.started:
            if not need:
                return  # No registration/grant while the first input is loading.
            # Bootstrap now, not during slow CPU reads: an unused manager may
            # otherwise idle-exit before our first registration.
            socket, _ = ensure_manager(self.manager_socket)
            supervisor.client.path = str(socket)
            self.started = True
        idle = self.idle_paused
        if need:
            idle = False
        elif self.clock() - pool.last_gpu_activity >= 30:
            idle = True
        if supervisor.connected and not supervisor.stopping and idle != self.idle_paused:
            # Self-pause is distinct from administrative pause/other contenders.
            # On disconnect Supervisor drains and reconciles; retry this update.
            try:
                supervisor.client.pause(idle)
                self.idle_paused = idle
                supervisor.next_poll = 0
            except OSError as error:
                supervisor._lost_connection(error)
        supervisor.tick()
        if supervisor.error is not None:
            raise supervisor.error
        status = (supervisor.status, self.idle_paused)
        if status != self.last_status:
            print(f"demand GPU task={supervisor.client.task_id} phase={status[0]} idle_release={status[1]}", flush=True)
            self.last_status = status

    def close(self):
        if not self.started:
            self.supervisor.client.close()
            return
        self.supervisor.stop()
        deadline = self.clock() + 15
        while not self.supervisor.finished and self.clock() < deadline:
            if self.supervisor.error is None:
                try:
                    self.pool.pump()
                except BaseException as error:
                    self.supervisor.fail(error)
            self.supervisor.tick()
            time.sleep(.01)
        if not self.supervisor.finished:
            self.supervisor.fail(RuntimeError("demand GPU cleanup timed out"))
            # Workers are joined; leave the server reservation conservative if
            # communication cannot be restored rather than claiming release.
            self.supervisor.client.close()
