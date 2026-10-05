from iso20_constraint_isolation import matrix,key,percent


def test_matrix_is_unique_and_keeps_reference_changes_explicit():
    cases=list(matrix())
    assert len(cases)>=20
    assert len({c[0] for c in cases})==len(cases)
    for name,updates,extra,_ in cases:
        assert '--winding-message-iterations' not in updates
        if '--reference-measurements' in updates:
            assert name=='reference-events' and not extra
        assert '--winding-gap-repair' not in updates
    assert cases[0][0]=='baseline'


def test_identity_uses_source_and_full_precision_arcs():
    p=dict(original_trace='123',begin_arc_base='0.125',end_arc_base='511.99999999')
    assert key(p)==(123,.125,511.99999999)


def test_mass_and_class_controls_are_separate():
    rows={name:(updates,extra) for name,updates,extra,_ in matrix()}
    for suffix,expected in [('mass',{'--freeze-piece-mass'}),('classes',{'--freeze-piece-classes'}),
                            ('classes-mass',{'--freeze-piece-classes','--freeze-piece-mass'})]:
        updates,extra=rows['events-frozen-'+suffix]
        assert '--frozen-piece-prior' in updates
        assert {x for x in extra if x.startswith('--freeze-')}==expected


def test_unavailable_score_is_not_zero():
    assert percent(None)=='N/A'
    assert percent(0)=='0.00%'


def test_factor_inventory_excludes_same_source_and_tracks_sign_conflicts(tmp_path,monkeypatch):
    import iso20_constraint_isolation as module
    monkeypatch.setattr(module,'OUT',tmp_path)
    (tmp_path/'x_consistency.csv').write_text('piece,original_trace\n0,10\n1,10\n2,20\n')
    (tmp_path/'x_winding_factors.csv').write_text(
        'piece_a,piece_b,evidence_weight,parallel,perpendicular,effective_signed_parallel_delta,effective_perpendicular_signed_delta,hard_parallel_sign,hard_perpendicular_sign\n'
        '0,1,1,1,0,0,NA,1,0\n'
        '0,2,0.5,1,0,1,NA,1,0\n'
        '0,2,0.5,1,0,-1,NA,1,0\n')
    stats,pairs=module.factor_inventory('x')
    assert stats==dict(measurements=2,evidence_mass=1.,pairs=1,contradictory_hard_sign_pairs=1)
    assert pairs[('0','2')]==[1.,0.,0.,1.,1,1]
