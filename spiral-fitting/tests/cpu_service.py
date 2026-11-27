"""Services whose runtime is real on the CPU with the device stages stubbed
(the conftest's ``device_stubs`` fixture), over the reference fit's settings."""
import time

import object_store_reference as reference

# The reference fit's settings, without the z window a request carries.
RUN = {key: value for key, value in reference.CONFIG_OVERRIDES.items()
       if key not in ('z_begin', 'z_end')}
REQUEST = {'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}


def request(**config):
    """REQUEST with ``config`` overriding the run's settings."""
    return {'run': {'z_begin': 0, 'z_end': 64, 'config': {**RUN, **config}}}


def stub_device_stages(fitter):
    """Leave everything but the CUDA-only stages of ``fitter`` (fit_spiral) real."""
    def build(self):
        self.start_iteration = 0
    fitter.FitContext.build_device_state = build
    fitter.FitContext.rebuild_model_state = build
    fitter.FitContext.release_setup_only_tracks = lambda self: None


def service_bindings(tmp_path):
    import spiral_service
    return spiral_service.bind_service_inputs(tmp_path / 'output', tmp_path / 'cache')


def service_state(binding, tmp_path, **kwargs):
    """A service over ``binding`` (an object-store binding) whose startup
    run is the reference fit's."""
    import spiral_service
    return spiral_service.ServiceState(
        bindings=service_bindings(tmp_path), object_store=binding,
        startup_run={'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}, **kwargs)


def write_inputs(root):
    """A directory holding the umbilicus a session built directly (not
    through the service, whose project supplies it) reads."""
    root.mkdir()
    reference.write_umbilicus(str(root / 'umbilicus.json'))
    return root


def wait_until(predicate, timeout=30, message='condition'):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(0.02)


def settle(state, timeout=120):
    """Wait until the service is not building and its session is Idle or
    Error; returns its status."""
    wait_until(lambda: not state._building
               and state.status()['state'] in ('Idle', 'Error'), timeout,
               'the session did not settle')
    return state.status()


def two_ranks(worker, tmp_path, *args):
    """Run ``worker(rank, 2, rendezvous, *args, result_path)`` on two
    spawned ranks; returns each rank's JSON result."""
    import json
    import torch.multiprocessing as mp
    results = [str(tmp_path / f'rank{rank}.json') for rank in range(2)]
    rendezvous = tmp_path / 'rendezvous'
    rendezvous.mkdir()
    mp.spawn(worker, args=(2, str(rendezvous / 'store'), *args, results), nprocs=2)
    outcome = []
    for path in results:
        with open(path) as stream:
            outcome.append(json.load(stream))
    return outcome
