from vesuvius.neural_tracing.fiber_trace_3d.fiber_label_layout import place_labels


def test_dense_labels_do_not_overlap_and_repeat_deterministically():
    items=[dict(uv=[100,100],text=str(i),family=i%2) for i in range(30)]
    args=(items,lambda p:tuple(p),(0,0,500,500),lambda s:(len(s)*4,6))
    rows=place_labels(*args)
    assert rows==place_labels(*args)
    assert len(rows)==30
    for i,a in enumerate(rows):
        x,y,r,b=a['box']
        for other in rows[:i]:
            xx,yy,rr,bb=other['box']
            assert not (x<rr and r>xx and y<bb and b>yy)


def test_labels_point_away_from_partner():
    rows=place_labels([dict(uv=[100,100],peer=[120,120],text='1'),
                       dict(uv=[120,120],peer=[100,100],text='2')],
                      tuple,(0,0,500,500),lambda s:(5,6))
    assert rows[0]['box'][2]<100 and rows[0]['box'][3]<100
    assert rows[1]['box'][0]>120 and rows[1]['box'][1]>120


def test_shared_anchor_uses_distinct_quadrants():
    rows=place_labels([dict(uv=[100,100],text=str(i)) for i in range(4)],
                      tuple,(0,0,300,300),lambda s:(5,6))
    quadrants={(r['box'][0]>100,r['box'][1]>100) for r in rows}
    assert len(quadrants)==4
