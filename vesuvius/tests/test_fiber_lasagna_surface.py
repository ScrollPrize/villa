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


def test_implicit_mask_offsets_and_start_arc(tmp_path, monkeypatch):
    data = dict(source_id=np.array([7, 8]), start=np.array([10., 0.]), end=np.array([30., 20.]),
                family=np.array([0, 0]), node_ds=np.array([[10., 20., 30.], [0., 10., 20.]]))
    state = np.array([[1., .25, 50., 1.], [0., 0., 10., 1.]])
    widths = np.array([100., 200., 300.])
    np.savez(tmp_path/'tensors.npz', **data)
    np.savez(tmp_path/'solution.npz', **data, state=state, widths=widths,
             fixed_winding=[1., 0.], retained=[True, False],
             v_offset_source=[7, 7], v_offset_arc=[10., 30.],
             u_offsets=[0., 8.], v_offsets=[2., 6.])
    (tmp_path/'config.json').write_text(json.dumps(dict(tensor_cache=str(tmp_path/'tensors.npz'), afv='unused')))
    class Bank:
        def __init__(self, afv, ids):
            np.testing.assert_array_equal(ids, [7])
            self.ids = np.array([7]); self.keys = np.array([0., 20., 40.])
            self.first = np.array([0]); self.last = np.array([2]); self.base = np.array([0.])
        def sample_xyz(self, source, arc):
            return np.column_stack((arc, arc*0, arc*0))
    monkeypatch.setattr(adapter, 'PolylineBank', Bank)
    original, _ = adapter.load_h_curves(tmp_path, deformation_root=tmp_path)
    deformed, _ = adapter.load_h_curves(tmp_path, deformation_root=tmp_path, apply_offsets=True)
    assert [c['piece'] for c in original] == [c['piece'] for c in deformed] == [0]
    np.testing.assert_allclose(original[0]['u'], [160., 170., 180.])
    np.testing.assert_allclose(deformed[0]['u'], [160., 174., 188.])
    np.testing.assert_allclose(deformed[0]['v'], [52., 54., 56.])
    np.testing.assert_array_equal(original[0]['xyz'], deformed[0]['xyz'])


def test_cuda_requires_manager_and_one_gpu(tmp_path):
    with pytest.raises(ValueError, match='gpu_manager'):
        adapter.run(dict(device='cuda'))
    with pytest.raises(ValueError, match='one GPU'):
        adapter.run(dict(device='cuda', devices='0,1', gpu_manager={'exclusive': False}))


def test_connected_mapping_uses_v_arc_and_float64_anchor(tmp_path, monkeypatch):
    data = dict(source_id=np.array([7, 8]), start=np.array([0.,0.]), end=np.array([20.,20.]),
        family=np.array([0,1]), node_ds=np.array([[0.,10.,20.],[0.,10.,20.]]),
        i=np.array([0]),j=np.array([1]),ds_a=np.array([10.]),ds_b=np.array([10.]))
    phase = np.float32(.1234567)
    state=np.array([[14,phase,50,1],[14,phase,40,1]], dtype=np.float32)
    np.savez(tmp_path/'tensors.npz', **data)
    np.savez(tmp_path/'solution.npz', **data,state=state,widths=np.full(20,8400.),
        fixed_winding=np.array([14,14],dtype=np.float32),retained=[True,True],
        v_offset_source=[7,8],v_offset_arc=[10.,10.],v_offset_a=[0],v_offset_b=[1],
        fixed_event_indices=[0],u_offsets=[0.,10.],v_offsets=[0.,0.])
    (tmp_path/'config.json').write_text(json.dumps(dict(tensor_cache=str(tmp_path/'tensors.npz'),afv='unused')))
    class Bank:
        def __init__(self, *_):
            self.ids=np.array([7,8]); self.keys=np.array([0.,10.,20.,0.,10.,20.])
            self.first=np.array([0,3]);self.last=np.array([2,5]);self.base=np.array([0.,0.])
        def sample_xyz(self, source, arc):
            return np.column_stack([arc,arc*0,arc*0])
    monkeypatch.setattr(adapter,'PolylineBank',Bank)
    curves,_=adapter.load_h_curves(tmp_path,deformation_root=tmp_path,apply_offsets=True,include_v=True)
    h,v=curves
    np.testing.assert_allclose(h['u'][1],(14+float(phase))*8400+10,rtol=0,atol=1e-9)
    np.testing.assert_allclose(v['u'],np.full(3,h['u'][1]))
    np.testing.assert_allclose(v['v'],[40.,50.,60.])
    assert h['u'][1]==v['u'][1] and h['v'][1]==v['v'][1]


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
