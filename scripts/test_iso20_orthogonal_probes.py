from iso20_orthogonal_probes import matrix,followup_matrix


def test_probe_matrix_is_bounded_and_target_only():
    rows=list(matrix())
    assert len(rows)==9 and len({r[0] for r in rows})==9
    assert rows[0]==('baseline',{},[])
    for name,params,flags in rows[1:]:
        assert params['--orthogonal-probe-span'] in (16,32,64)
        assert set(params)<={'--orthogonal-probe-span','--orthogonal-probe-blend'}
        assert set(flags)<={'--orthogonal-probe-consensus'}
        if name.startswith('half'):assert not flags


def test_followup_names_have_no_native_extension_collisions():
    from pathlib import Path
    rows=list(followup_matrix())
    assert len(rows)==4
    assert len({Path(name).stem for name,_,_ in rows})==4
    assert all(Path(name).stem==name for name,_,_ in rows)
