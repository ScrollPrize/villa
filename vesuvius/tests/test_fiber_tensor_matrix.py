import json
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign import tensor_matrix_jobs


def test_lane_binding_and_duplicate_validation(tmp_path):
    paths=[]
    for i in range(3):
        p=tmp_path/f'{i}.json'
        p.write_text(json.dumps({'fiber_uv_layout':{'output_root':str(tmp_path/f'out{i}')}}))
        paths.append(str(p))
    config={'gpu_manager':{'exclusive':True},'experiment_matrix':[paths[:2],paths[2:]]}
    jobs=tensor_matrix_jobs(config,['GPU-A','GPU-B'])
    assert [(j[0],j[1]) for j in jobs]==[(0,'GPU-A'),(0,'GPU-A'),(1,'GPU-B')]
    with pytest.raises(ValueError,match='Insufficient'):tensor_matrix_jobs(config,['GPU-A'])
    config['experiment_matrix']=[[paths[0]],[paths[0]]]
    with pytest.raises(ValueError,match='Duplicate'):tensor_matrix_jobs(config,['GPU-A','GPU-B'])
    config['gpu_manager']['exclusive']=False
    with pytest.raises(ValueError,match='exclusive'):tensor_matrix_jobs(config,['GPU-A','GPU-B'])


def test_matrix_child_launch_under_parent_lease(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign as module
    config_path=tmp_path/'child.json'
    config_path.write_text(json.dumps({'fiber_uv_layout':{'output_root':str(tmp_path/'child')}}))
    config={'gpu_manager':{'exclusive':True},'experiment_matrix':[[str(config_path)]],
            'output_root':str(tmp_path)}
    calls=[]
    class Process:
        def __init__(self,command,**kwargs):calls.append((command,kwargs['env']))
        def wait(self,timeout=None):return 0
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-A')
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    result=module.solve_tensor_matrix(config,'test.module')
    assert result['jobs'][0]['exit_code']==0
    assert calls[0][0][-3:]==['--unmanaged','--devices','0']
    assert calls[0][1]['CUDA_VISIBLE_DEVICES']=='GPU-A'
