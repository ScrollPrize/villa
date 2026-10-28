import json
import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_afv_transform import mapped_bounds, run
from vesuvius.neural_tracing.tracers.shared.annotations.afv import write_afv, read_afv_polylines


def test_inverse_registration_keeps_whole_fibers(tmp_path):
    src=tmp_path/'source.afv'; target=tmp_path/'target.afv'
    def meta(space):
        return dict(frame=dict(vc_open_data_coordinate_space=space), root={}, uuid=space)
    source_id=2**53+7  # Preserve full SQLite integer IDs through bbox filtering.
    row=dict(id=source_id,name='seven',family='H',annotation='{}')
    xyz=np.array([[1.,2.,3.],[1.,2.,9.]])
    write_afv(None,src,[(row,xyz)],meta('old'))
    write_afv(None,target,[],meta('new'))
    transform=dict(fixed_volume='old_volume',transformation_matrix=[[2,0,0,0],[0,2,0,0],[0,0,2,1]])
    path=tmp_path/'transform.json';path.write_text(json.dumps(transform))
    run(dict(afv=str(src),frame_template=str(target),transform=str(path),invert=True,
        fixed_volume='old_volume',source_coordinate_space='old',target_coordinate_space='new',
        output_root=str(tmp_path/'output'),z_range=[2,3]))
    rows=read_afv_polylines(tmp_path/'output/fibers.afv')
    assert len(rows)==1 and rows[0][0]['id']==source_id
    np.testing.assert_array_equal(rows[0][1],[[.5,1,1],[.5,1,4]])


def test_bounds_reflection():
    m=np.array([[-2,1,0,4],[0,1,0,0],[0,0,3,1],[0,0,0,1.]])
    lo,hi=mapped_bounds(np.array([[0,0,0]]),np.array([[2,3,4]]),m)
    np.testing.assert_array_equal(lo,[[0,0,1]])
    np.testing.assert_array_equal(hi,[[7,3,13]])
