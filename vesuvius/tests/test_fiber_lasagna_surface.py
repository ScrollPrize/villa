import json
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d import fiber_lasagna_surface as adapter


def test_reversed_piece_uses_actual_curve_and_no_normals(tmp_path, monkeypatch):
    data = dict(source_id=np.array([7, 8]), start=np.array([0., 0.]), end=np.array([20., 20.]),
                family=np.array([0, 1]), node_ds=np.array([[0., -10., -20.], [0., 10., 20.]]))
    np.savez(tmp_path/'tensors.npz', **data)
    np.savez(tmp_path/'solution.npz', **data, state=np.zeros((2, 4)), widths=np.array([100.]))
    (tmp_path/'config.json').write_text(json.dumps(dict(tensor_cache=str(tmp_path/'tensors.npz'), afv='unused')))
    monkeypatch.setattr(adapter.uv, 'hard_geometry', lambda *_: (
        np.array([[.9, .1], [.9, .1]]), None, np.array([True, True]), np.array([0, 0]),
        np.array([100., 100.]), np.array([50., 50.]), None))
    class Bank:
        def __init__(self, afv, ids):
            np.testing.assert_array_equal(ids, [7])
            self.ids = np.array([7]); self.keys = np.array([0., 10., 20.])
            self.first = np.array([0]); self.last = np.array([2]); self.base = np.array([0.])
        def sample_xyz(self, source, arc):
            return np.column_stack((arc, arc**2, arc*0+500))
    monkeypatch.setattr(adapter, 'PolylineBank', Bank)
    curves, provenance = adapter.load_h_curves(tmp_path)
    assert len(curves) == 1 and curves[0]['source'] == 7
    np.testing.assert_array_equal(curves[0]['u'], [80., 90., 100.])
    np.testing.assert_array_equal(curves[0]['xyz'][:, 1], [400., 100., 0.])
    assert provenance['joint_threshold'] == .5


def test_cpu_never_requests_gpu(monkeypatch):
    monkeypatch.setattr(adapter, 'run_local', lambda cfg: 'cpu-result')
    assert adapter.run(dict(device='cpu')) == 'cpu-result'


def test_cuda_requires_manager_and_one_gpu(tmp_path):
    with pytest.raises(ValueError, match='gpu_manager'):
        adapter.run(dict(device='cuda'))
    with pytest.raises(ValueError, match='one GPU'):
        adapter.run(dict(device='cuda', devices='0,1', gpu_manager={'exclusive': False}))


def test_managed_launch_uses_module_and_no_local_cuda(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import managed_evaluation
    out = tmp_path/'surface'
    def launch(command, *, devices, settings):
        assert command[1:3] == ['-m', adapter.__name__]
        assert devices == '3'
        cfg = json.loads((tmp_path/'surface.launch.json').read_text())
        assert cfg['output_root'] == str(out)
        out.mkdir()
        (out/'summary.json').write_text('{"done": true}')
        return 0
    monkeypatch.setattr(managed_evaluation, 'launch_managed', launch)
    assert adapter.run(dict(device='cuda', devices='3', gpu_manager={'exclusive': False}, output_root=str(out))) == {'done': True}
