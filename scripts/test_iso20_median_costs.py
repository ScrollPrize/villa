from iso20_median_costs import matrix,combinations


def test_matrix_costs_and_names():
    rows=dict(matrix())
    assert len(rows)==18
    assert all('.' not in name for name in rows)
    assert rows['p0']['--winding-weights']=='0,0,0,0,0'
    for name in ('p0p005','p0p01','p0p025','p0p05','p0p1','d0','d75','d300','b100','b500','perp-near','perp-far','parallel-same','sign22','sign88','no-parallel-sign','no-perp-sign'):
        assert name in rows
    assert all('--reference-oracle-magnitude-weight' not in u for u in rows.values())
    assert rows['sign22']['--winding-hard-signs']=='both'
    assert rows['sign22']['--winding-hard-sign-angle']=='off'


def test_only_improving_disjoint_combinations():
    def row(score,updates):return dict(oracle=dict(constraint_score_percent=score),updates=updates)
    rows={'p0p05':row(-30,{'p':'0.05'}),'x':row(-29,{'p':'0.01'}),'y':row(-28,{'d':'75'}),'z':row(-31,{'b':'500'})}
    assert combinations(rows)==[('combo1',{'d':'75','p':'0.01'})]
    assert combinations({'p0p05':rows['p0p05']})==[]
