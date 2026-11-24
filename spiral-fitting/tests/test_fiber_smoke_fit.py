"""A short CPU fit with the fiber CC loss: one patch, two fibers in two
components of a winding run, 20 steps through the equal-winding and DT
phases with the vertical-fiber offset on; every loss stays finite, and the
end-of-stage fiber satisfaction pass covers both fibers."""
import math

import numpy as np
import pytest


def test_a_short_fit_with_the_fiber_loss_has_finite_losses(tmp_path, monkeypatch):
    pytest.importorskip('vc_object_store')
    from vc_object_store import ObjectStore
    import object_store_reference as reference
    from object_store_fixture import VOLUME, put_patch, write_winding_solve
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')
    store = str(tmp_path / 'fit.vcobj')
    patch = reference.PATCHES[0]
    with ObjectStore.create(store, author='ingest', hostname='host') as db:
        put_patch(db, patch['zyxs'], patch['scale'], name=patch['name'])
    umbilicus = str(tmp_path / 'umbilicus.json')
    reference.write_umbilicus(umbilicus)

    def sheet(winding, theta0, theta1, z0, z1, count=40):
        """Stored XYZ along the reference spiral's sheet ``winding``."""
        return np.array([reference.spiral_zyx(theta, winding, z) for theta, z in zip(
            np.linspace(theta0, theta1, count), np.linspace(z0, z1, count))])[:, ::-1].copy()

    # A horizontal fiber over most of a turn past theta = 0 and a vertical one.
    solve = write_winding_solve(
        tmp_path, [(sheet(8, 0.4, 5.0, 8, 56), 0, 0, 0),
                   (sheet(9, 0.2, 1.4, 10, 50), 0, 1, 1)],
        handedness=1 if reference.scroll_spec().spiral_outward_sense == 'CW' else -1)
    config = {**reference.fit_config(), 'object_store_location': store,
              'optimizer_num_training_steps': 20, 'model_flow_bounds_radius': 256,
              'loss_start_fiber_dt': 10, 'fiber_dt_target_update_interval': 5,
              'fiber_vertical_radial_offset_enabled': True,
              'sample_count_fiber_ccs': 8, 'sample_count_points_per_fiber_cc': 8}
    lines = []
    fit = FitContext(
        FitConfig(config), scroll=reference.scroll_spec(),
        paths=SpiralInputPaths(volume_uuid=VOLUME, umbilicus=umbilicus,
                               winding_solve_fibers=solve['fibers'],
                               winding_solve_run_store=solve['run_store'],
                               winding_solve_run_uuid=solve['run_uuid']),
        cache_dir=str(tmp_path / 'cache'), out_base_dir=str(tmp_path / 'out'),
        input_log=lines.append)
    fit.load_host_inputs()
    fit.prepare_output()
    fit.build_device_state()
    assert fit.fiber_cc_loss is not None and fit.fiber_cc_loss.has_offsets
    for iteration in range(20):
        _, losses, metrics, _ = fit.step(iteration)
        assert all(math.isfinite(float(value)) for value in losses.values()), losses
        if iteration == 0:
            assert float(losses['fiber']) > 0
            assert metrics['fiber_rejected_fraction'] == 0.0
            assert metrics['fiber_label_step_mismatch'] == 0.0
    # The other component's whole-winding target was refreshed in the DT phase.
    assert any(line.startswith('fiber DT targets (step 11)') for line in lines)
    # The end-of-stage fiber satisfaction pass over both fibers.
    summary = fit._fiber_satisfaction('fitted')
    assert summary['total_fibers'] == 2 and summary['total_fiber_other_ccs'] == 1
    assert summary['fiber_cached_targets'] == 1
    with np.load(f'{fit.out_path}/fiber_satisfied_fitted.npz') as saved:
        assert len(saved['fiber_uuid']) == 2
        assert saved['point_count'].sum() == summary['fiber_points']
    # A checkpoint names the solve, and resumes only against it.
    payload = fit._checkpoint_payload(20)
    assert payload['input_manifest']['winding_solve']['run_uuid'] == solve['run_uuid']
    assert fit.inspect_checkpoint(payload).accepted
    payload['input_manifest']['winding_solve']['run_uuid'] = '0' * 32
    assert not fit.inspect_checkpoint(payload).accepted
