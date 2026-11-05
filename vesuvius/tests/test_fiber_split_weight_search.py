from vesuvius.neural_tracing.fiber_trace_3d.fiber_split_weight_search import ranking,better,acceptance_limit_reached




def test_unlimited_search_continues_past_reporting_milestone():
    for count in (0,3,4,100):
        assert not acceptance_limit_reached(None,count)
    assert not acceptance_limit_reached(3,2)
    assert acceptance_limit_reached(3,3)




def test_reuse_key_when_fixed_cost_becomes_tunable(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_split_weight_search import historical_trial_key
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_paired_weight_search import numeric_key
    old=dict(split_target_length_vx=0,continuation_weight=10)
    (tmp_path/'manifest.json').write_text(json.dumps({'settings':{'initial_parameters':old}}))
    assert historical_trial_key(tmp_path,dict(old,rejection_cost=2.5))==numeric_key(old)




def test_priority_is_count_divided_by_retained_fraction_then_l1():
    a=dict(count_per_kept_fraction=20,perp_l1_sum=2)
    assert better(dict(a,count_per_kept_fraction=19,perp_l1_sum=100),a)
    assert not better(dict(a,count_per_kept_fraction=21,perp_l1_sum=0),a)
    assert better(dict(a,perp_l1_sum=1),a)
    assert not better(a,a)


def test_retention_counts_length_at_half_not_piece_count_or_probability_mass():
    r=dict(signs=dict(perpendicular=dict(violations=2,percent=3)),
           score=dict(gt_perpendicular=dict(l1=.2)))
    reports={k:dict(r) for k in ('free','joint','selectable')}
    reports['selectable']['gt_enablement']=[
        dict(length_vx=1000,enabled_probability=.5),
        dict(length_vx=100,enabled_probability=.99),
        dict(length_vx=9000,enabled_probability=.4999)]
    s=ranking(reports)
    assert s['gt_retained_vx']==1100 and s['gt_total_vx']==10100
    assert s['perp_sign_sum']==6 and s['perp_sign_percent_sum']==9
    assert s['count_per_kept_fraction']==6*10100/1100
    reports['selectable']['gt_enablement'][2]['enabled_probability']=.9
    assert better(ranking(reports),s)  # More retained length must improve score.
    for piece in reports['selectable']['gt_enablement']:piece['enabled_probability']=0
    empty=ranking(reports)
    assert empty['count_per_kept_fraction'] is None
    assert not better(empty,s)
    assert better(s,empty)
    assert not better(empty,empty)


def test_rescore_ignores_historical_ranking_and_selects_high_retention(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_split_weight_search import rescore_completed
    for index,kept in enumerate((25,100)):
        root=tmp_path/'trials'/str(index)
        root.mkdir(parents=True)
        (root/'free_base.json').write_text(json.dumps({'fiber_implicit_winding':{'rejection_cost':index}}))
        for name in ('free','joint','selectable'):
            (root/name).mkdir()
            report=dict(signs=dict(perpendicular=dict(violations=2,percent=3)),
                        score=dict(gt_perpendicular=dict(l1=.2)),gt_enablement=[
                            dict(length_vx=kept,enabled_probability=1),
                            dict(length_vx=100-kept,enabled_probability=0)])
            (root/name/'search_report.json').write_text(json.dumps(report))
        (root/'result.json').write_text(json.dumps({'ranking':{'count_times_kept_fraction':0}}))
    results=rescore_completed(dict(reuse_roots=[str(tmp_path)],initial_parameters={'rejection_cost':0}))
    assert results[0]['parameters']['rejection_cost']==1
    assert results[0]['ranking']['count_per_kept_fraction']==6
    assert results[1]['ranking']['count_per_kept_fraction']==24


def test_caller_selected_joint_moves_preserve_other_parameters():
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_split_weight_search import focused_proposals
    best=dict(rejection_cost=3.,continuation_weight=7.,split_min_perpendicular=3,phase_weight=11.)
    cfg=dict(weights=['rejection_cost','continuation_weight','split_min_perpendicular'],
        bounds=dict(rejection_cost=[0,10],continuation_weight=[0,20],split_min_perpendicular=[2,6]),
        proposal_groups=[dict(name='costs',axes=[dict(parameters=['rejection_cost','continuation_weight'],
            mode='factor',values=[.5,1,2])]),dict(name='splits',axes=[dict(parameters=['split_min_perpendicular'],
            mode='absolute',values=[2,3,6])])])
    rows=focused_proposals(cfg,best,0)
    assert len(rows)==4
    assert all(values['phase_weight']==11 for values,_ in rows)
    assert any(values['rejection_cost']==6 and values['continuation_weight']==14 for values,_ in rows)
    assert focused_proposals(cfg,best,1)==[]
    cfg['weights'].remove('continuation_weight')
    with pytest.raises(ValueError,match='fixed parameter'):focused_proposals(cfg,best,0)


def test_minimal_sample_proposals_are_valid():
    import json
    from pathlib import Path
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_split_weight_search import focused_proposals
    path=Path(__file__).resolve().parents[1]/'docs/examples/fiber_workflows/split_search.json'
    cfg=json.loads(path.read_text())['fiber_split_weight_search']
    assert len(focused_proposals(cfg,cfg['initial_parameters'],0))==2
