from iso20_iterative import BASE,GRIDS,case_id,neighbours,updates,primary,rank,finalists,choose


def row(correct=16,shifts=3,problematic=55,fulfilled=188,**params):
    return dict(params={**BASE,**params},independent=dict(independent_percent=100*correct/26,correct_references=correct,
        observed_ladder_shift=dict(references=24,ladder_shifts=shifts)),
        oracle=dict(problematic_unique_constraints=problematic,retained_fulfilled_unique_constraints=fulfilled))


def test_neighbours_and_full_identity():
    n=neighbours(BASE)
    assert len(n)==1+sum(len(v)-1 for v in GRIDS.values())
    assert len({a for a,_ in n})==len(n)
    assert any(p['minimum']==1024 for _,p in n)
    assert all(sum(p[k]!=BASE[k] for k in GRIDS)<=1 for _,p in n)
    assert case_id(BASE)==case_id(dict(reversed(list(BASE.items()))))
    assert updates(BASE)['--winding-fixed-phase']==.5


def test_accuracy_then_shift_then_removal():
    base=row()
    assert rank(base)>rank(row(correct=15,shifts=0,problematic=0))
    assert rank(base)>rank(row(shifts=4,problematic=0))
    assert rank(row(problematic=54))>rank(base)
    unknown=row(correct=17);unknown.pop('oracle')
    assert rank(unknown)>rank(base)


def test_ties_cached_finalists_and_stable_choice():
    rows={'base':row(),'cached':row(problematic=54,minimum=384),'inferior':row(correct=15)}
    assert set(finalists(rows,rows))=={'base','cached'}
    assert choose(list(rows),rows,'base')=='cached'
    rows['cached']=row(minimum=384)
    assert choose(list(rows),rows,'base')=='base'
    assert choose(['cached','base'],rows,'other')=='base'


def test_unavailable_not_zero():
    assert primary({'independent':{'independent_percent':None}})<primary(row(correct=0))


def test_completed_report(tmp_path,monkeypatch):
    import iso20_iterative as module
    monkeypatch.setattr(module,'OUT',tmp_path)
    monkeypatch.setattr(module,'REPORT',tmp_path/'report.json')
    r=row()
    r['independent']['observed_ladder_shift']['ladder_shift_percent']=87.5
    r['oracle'].update(constraint_score_percent=-29.5,counts={'problematic_percent_initial':22.8})
    state=dict(stage='complete',incumbent=case_id(BASE),rows={case_id(BASE):r},rounds=[],repeat={'oracle_counts_identical':True})
    module.publish(state)
    report=(tmp_path/'report.md').read_text()
    assert '## Outcome' in report and '61.5385%' in report
    assert 'No accuracy or shift improvement' in report
