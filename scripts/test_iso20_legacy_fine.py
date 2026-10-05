from iso20_legacy_fine import matrix


def test_port_matrix_preserves_legacy_extraction():
    rows=list(matrix())
    assert len(rows)==10
    assert rows[0]==('baseline',{},[])
    for name,params,flags in rows[1:]:
        assert '--legacy-fine-pieces' in flags
        assert '--local-constraint-events' not in flags
        assert '--constraint-local-scorer' not in flags
        assert '--winding-message-iterations' not in params
        assert params['--local-min-piece-length'] in (64,128,256)
        if 'length' in name:
            assert '--partition-length-cost' in flags
            assert params['--winding-defect-cost']==150
