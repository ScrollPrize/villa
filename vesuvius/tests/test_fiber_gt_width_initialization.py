import numpy as np
import pytest
import json
from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_width_initialization import perturb_widths


def test_scale_does_not_mutate_initialization():
    widths=np.array([1000.,1200.,1400.]);before=widths.copy()
    got=perturb_widths(widths,dict(kind='scale',factor=.5),np.ones(3))
    np.testing.assert_array_equal(got,before*.5)
    np.testing.assert_array_equal(widths,before)


def test_radial_replaces_without_rescaling_or_fitting():
    radial=np.array([500.,600.,700.])
    got=perturb_widths(np.array([1000.,1200.,1400.]),dict(kind='radial'),radial)
    np.testing.assert_array_equal(got,radial)
    assert not np.shares_memory(got,radial)


@pytest.mark.parametrize('factor',[0.,-1.,float('nan'),float('inf')])
def test_invalid_widths_rejected(factor):
    with pytest.raises(ValueError):
        perturb_widths(np.ones(3),dict(kind='scale',factor=factor),np.ones(3))


def test_preparer_preserves_state_and_writes_required_initial_metadata(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_width_initialization import run
    source=tmp_path/'source.npz';state=np.arange(8).reshape(2,4)
    np.savez(source,widths=np.array([1000.,1200.]),state=state)
    recipe=tmp_path/'base.json'
    recipe.write_text(json.dumps({'fiber_implicit_winding':{'initial_solution':str(source),'lr':.01}}))
    result=run(dict(base_recipe=str(recipe),radial_initial_solution=str(source),
        output_root=str(tmp_path/'out'),parameters={'width_lr':200,'lr':.02},
        trials=[dict(name='half',kind='scale',factor=.5,device='2',parameters={'lr':.03})]))
    row=result['half'];path=tmp_path/'out/half'
    with np.load(row['initial_solution']) as saved:
        np.testing.assert_array_equal(saved['state'],state)
        np.testing.assert_array_equal(saved['widths'],[500.,600.])
    config=json.loads((path/'config.json').read_text())
    assert config['lr']==.03
    assert config['width_lr']==200
    assert config['devices']=='2'
    assert json.loads((path/'recipe.json').read_text())['fiber_implicit_winding']['initial_solution']==row['initial_solution']
