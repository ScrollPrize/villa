import numpy as np
from iso20_probe_correctness import sample_at,match_pieces,score


def test_strict_matches_and_reverse_tangent():
    refs={0:[np.array([[0.,0,0],[100,0,0]])]}
    pieces=[dict(original_trace='0',begin_arc_base='0',end_arc_base='100')]
    line=np.array([[100.,1,0],[0,1,0]])
    assert match_pieces([line],pieces,refs,(4,))[4].tolist()==[0]
    refs[1]=[np.array([[0.,2,0],[100,2,0]])]
    assert match_pieces([line],pieces,refs,(4,))[4].tolist()==[-1]


def test_sample_at():
    x,t=sample_at([[0,0,0],[0,0,0],[10,0,0]],[2,8])
    np.testing.assert_allclose(x,[[2,0,0],[8,0,0]])
    np.testing.assert_allclose(t,[[1,0,0],[1,0,0]])


def test_score_freezes_polarity_and_checks_contacts():
    row=dict(piece_a='0',piece_b='1',original_signed_delta='-.5',
             hard_perpendicular_sign='1',normal_component='0',constraint='0')
    stats,_,pol=score([row],[0,1])
    assert stats['wrong']==0 and pol=={'0':-1}
    stats,_,_=score([{**row,'original_signed_delta':'.5'}],[0,1],pol)
    assert stats['wrong']==1
    stats,_,_=score([row],[0,1],pol,{'0':(-1,1)})
    assert stats['assessable']==0


def test_tied_calibration_unknown():
    row=dict(piece_a='0',piece_b='1',original_signed_delta='-.5',
             hard_perpendicular_sign='1',normal_component='0',constraint='0')
    stats,_,pol=score([row,{**row,'constraint':'1','original_signed_delta':'.5'}],[0,1])
    assert pol=={'0':0} and stats['assessable']==0


def test_local_competitor_rejects_whole_piece():
    refs={0:[np.array([[0.,0,0],[100,0,0]])],
          1:[np.array([[90.,2,0],[100,2,0]])]}
    pieces=[dict(original_trace='0',begin_arc_base='0',end_arc_base='100')]
    assert match_pieces([np.array([[0.,1,0],[100,1,0]])],pieces,refs,(4,))[4].tolist()==[-1]
