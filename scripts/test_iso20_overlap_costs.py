import iso20_overlap_costs as sweep
from test_iso20_iterative import row


def test_all_zero_candidates_and_cost_only_grid():
    assert len(set(sweep.VARIANTS))==12
    assert all(v.endswith('-f0') for v in sweep.VARIANTS)
    neighbours=sweep.neighbours(sweep.BASE)
    assert len(neighbours)==1+sum(len(v)-1 for v in sweep.GRIDS.values())
    for p in neighbours:
        assert sum(p[k]!=sweep.BASE[k] for k in p)<=1
        u=sweep.updates(p)
        assert not any(k in u for k in ('--local-min-piece-length','--orthogonal-probe-span','--winding-message-iterations','--threads'))
        assert u['--winding-sign-weights']=='1,1'
        assert u['--winding-weights'].split(',')[-1]=='0'
    assert sweep.identity('a',sweep.BASE)!=sweep.identity('b',sweep.BASE)


def test_zero_magnitude_does_not_disable_signs():
    p={**sweep.BASE,'near':0,'far':0,'p0':0,'p1':0}
    u=sweep.updates(p)
    assert all(float(x)==0 for x in u['--winding-weights'].split(','))
    assert u['--winding-sign-cost']=='hard'


def test_resume_plateau_is_terminal():
    item=dict(stage='round 1',rounds=[dict(previous='a',winner='a')])
    assert list(sweep.remaining_rounds(item))==[]
    item['rounds'][0]['winner']='b'
    assert list(sweep.remaining_rounds(item))==[1]
    item['stage']='complete'
    assert list(sweep.remaining_rounds(item))==[]


def test_selection_keeps_incumbent_on_exact_tie():
    rows={'base':row(),'a':row(),'b':row(correct=17)}
    assert sweep.choose(['base','a'],rows,'base')=='base'
    assert sweep.choose(list(rows),rows,'base')=='b'


def test_parameter_override_shared_stage(monkeypatch,tmp_path):
    import iso20_iterative as shared
    seen=[]
    monkeypatch.setattr(shared.shared,'configured',lambda t,n,s,u:seen.append(u) or {})
    monkeypatch.setattr(shared.shared,'execute',lambda c:{})
    monkeypatch.setattr(shared.isolation,'collect',lambda *a:({'independent_percent':50},None))
    monkeypatch.setattr(shared,'signature',lambda r:{})
    shared.run_stage('x',{},'ind',{},[],parameter_updates={'--winding-defect-cost':0})
    assert seen==[{'--winding-defect-cost':0}]


def test_completed_report_has_twelve_before_after_rows(tmp_path,monkeypatch):
    import json
    r=row()
    r['params']=dict(sweep.BASE)
    r['independent']['observed_ladder_shift']['ladder_shift_percent']=87.5
    r['oracle'].update(constraint_score_percent=-29.5,reference_score_percent=80.77)
    monkeypatch.setattr(sweep,'OUT',tmp_path)
    monkeypatch.setattr(sweep,'SOURCE',tmp_path)
    monkeypatch.setattr(sweep,'REPORT',tmp_path/'report.json')
    (tmp_path/'results.json').write_text(json.dumps({v:r for v in ['control',*sweep.VARIANTS]}))
    state=dict(stage='complete',rows={v:r for v in sweep.VARIANTS},variants={
        v:dict(winner=v,stage='complete') for v in sweep.VARIANTS})
    sweep.publish(state)
    report=(tmp_path/'report.md').read_text()
    assert 'No-dedup baseline' in report and 'Before accuracy' in report
    assert 'All12 selections complete' in report
    assert all('| '+v+' | complete |' in report for v in sweep.VARIANTS)
