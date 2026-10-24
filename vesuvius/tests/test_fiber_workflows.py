"""Restored entry points stay portable and delegate to the shared algorithms."""
import ast
import importlib
import inspect
import json
from pathlib import Path
import sqlite3
import pytest
from vesuvius.neural_tracing.fiber_trace_3d import fiber_workflows as workflows


def test_every_workflow_import_resolves():
    tree=ast.parse(inspect.getsource(workflows))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.level==1:
            if node.module is None:
                for name in node.names:importlib.import_module('.'+name.name,workflows.__package__)
                continue
            module=importlib.import_module('.'+node.module,workflows.__package__)
            for name in node.names:assert hasattr(module,name.name),(node.module,name.name)


def test_restored_modules_import():
    root=Path(workflows.__file__).parent
    for path in sorted(root.glob('fiber_*.py')):
        importlib.import_module('.'+path.stem,workflows.__package__)


def test_dispatch_calls_preparation_without_changing_settings(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_selected_dataset_eval as selected
    supplied={'output_root':'unused','parameters':{'rejection_cost':2.5}}
    monkeypatch.setattr(selected,'run',lambda cfg:cfg)
    assert workflows.run({'fiber_selected_dataset_eval':supplied}) is supplied


def test_trial_override_delegates_through_same_workflow(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_selected_dataset_eval as selected
    path=tmp_path/'recipe.json'
    path.write_text(json.dumps({'fiber_selected_dataset_eval':{'parameters':{'lr':.1,'width_lr':2},'output_root':'old'}}))
    monkeypatch.setattr(selected,'run',lambda cfg:cfg)
    result=workflows.run({'fiber_solver_trial':{'recipe':str(path),'output_root':'new','parameters':{'lr':.05}}})
    assert result=={'parameters':{'lr':.05,'width_lr':2},'output_root':'new'}


def test_dispatch_rejects_ambiguous_or_unknown_tasks():
    with pytest.raises(ValueError,match='exactly one'):workflows.run({'a':{},'b':{}})
    with pytest.raises(ValueError,match='Unknown'):workflows.run({'unknown':{}})


def test_shared_helpers_not_reimplemented():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_current_gt_solve as gt
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_enclosure_width_init as enclosure
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_curve_bank as curves
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_surface as surface
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_winding_order as order
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_density_sign_report as report
    assert gt.enclosure_brackets is enclosure.enclosure_brackets
    assert surface.CurveBank is curves.CurveBank
    assert report.ordering is order.ordering


def test_native_exporter_uses_standalone_command(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import constraint_extraction as native
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_native_workflow import resolve_command
    monkeypatch.setattr(native.shutil,'which',lambda name:'/bin/exporter' if name=='custom-exporter' else None)
    assert resolve_command({'binary':'vc_fiber_trace_chunk','args':['afv-constraints','input.json']},
        {'native_binary':'custom-exporter'})==['/bin/exporter','input.json']


def test_native_input_regression_can_be_run_first(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import constraint_extraction as native
    calls=[]
    monkeypatch.setattr(native,'resolve_native_binary',lambda name,settings:name)
    monkeypatch.setattr(native.subprocess,'run',lambda command,**kwargs:calls.append(command))
    (tmp_path/'extract.json').write_text('{}')
    native.run_native_constraints(tmp_path/'extract.json',{'native_regression_binary':'test_fiber_trace_winding_bp'})
    assert calls==[['test_fiber_trace_winding_bp'],['vc_fiber_constraints',str(tmp_path/'extract.json')]]


def test_native_exporter_reads_the_afv_object_store(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import constraint_extraction as native
    afv=tmp_path/'fibers.afv'
    with sqlite3.connect(afv) as db:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute("INSERT INTO metadata VALUES('uuid','\"1482bcc8-f93e-4c3b-b380-2ddf369318f4\"')")
    importer=tmp_path/'import_afv.py';importer.write_text('')
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[1]==str(importer):Path(command[3]).mkdir()
    monkeypatch.setattr(native,'resolve_native_binary',lambda name,settings:name)
    monkeypatch.setattr(native.subprocess,'run',run)
    config=tmp_path/'extract.json';config.write_text(json.dumps(dict(afv=str(afv),full_volume=True)))
    settings=dict(afv_importer=str(importer))
    native.run_native_constraints(config,settings)
    native.run_native_constraints(config,settings)
    imports=[command for command in calls if command[1]==str(importer)]
    assert len(imports)==1 and imports[0][2]==str(afv) and imports[0][4]=='--unbound'
    resolved=json.loads((tmp_path/'extract.store.json').read_text())
    assert resolved['afv']==str(afv) and resolved['object_store']==imports[0][3]
    assert resolved['collection']=='1482bcc8-f93e-4c3b-b380-2ddf369318f4'
    assert calls[-1]==['vc_fiber_constraints',str(tmp_path/'extract.store.json')]


def test_usage_examples_do_not_depend_on_local_experiments():
    root=Path(__file__).parents[1]/'docs/examples/fiber_workflows'
    for path in root.glob('*.json'):
        text=path.read_text();json.loads(text)
        assert '/home/' not in text and '/ephemeral/' not in text,path


def test_shared_workflows_and_docs_have_no_embedded_run_identity():
    """Supplement behavioral config tests with guards against copied run metadata."""
    import re
    source=Path(workflows.__file__).parent
    docs=Path(__file__).parents[1]/'docs'
    paths=list(source.glob('fiber_*.py'))+list(source.glob('paired_*.py'))
    paths+=[docs/'fiber_workflows.md',docs/'fiber_layout_benchmark.md']
    paths+=list((docs/'examples/fiber_workflows').glob('*.json'))
    forbidden=re.compile(r'/(?:home|ephemeral)/[^\s\x27\x22]+|\b20\d{6}[_T]\d{6}\b|\b[a-f0-9]{40,64}\b')
    for path in paths:
        assert not forbidden.search(path.read_text()),path


def test_reviewed_json_requires_explicit_scan_identity(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_json_solve import convert
    with pytest.raises(TypeError,match='expected_manifest_tokens'):
        convert(tmp_path,tmp_path/'source.afv',tmp_path/'out.afv')
