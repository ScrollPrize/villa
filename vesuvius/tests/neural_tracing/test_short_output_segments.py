import pytest
from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import filter_output_segments,filtered_output_statistics


def fixture():
    segments=[]
    for i,(length,covered,reason) in enumerate([(511,700,'distance_above_threshold'),(512,100,'presence_filtered'),(600,900,'reference_end'),(0,0,'reverse_consistency_filtered')]):
        segments.append(dict(trace_points_base_xyz=[[0,0,0],[length,0,0]],matches=[dict(search_begin_arc_base=0)] if covered else [],end_reference_arc_base=covered,termination_reason=reason))
    return dict(provenance=dict(base_voxel_size_um=2,source_id='test',case_index=0),replay=dict(reference_begin_arc_base=0,reference_end_arc_base=2000,completed_reference_arc_base=2000,segments=segments,failures=[dict(segment_index=0,reference_arc_base=700)],filtered_stops=[[],[]]))


def test_boundary_geometry_and_failure_mapping():
    doc=fixture()
    before=filtered_output_statistics([filter_output_segments(doc,0)])
    after=filtered_output_statistics([filter_output_segments(doc,512)])
    assert before['ordinary_restarts']==1 and after['ordinary_restarts']==0
    assert after['discarded_error_events']==1
    assert after['kept_segments']==2 and after['discarded_segments']==2
    assert after['presence_filtered']==1 and after['reverse_filtered']==0
    assert after['retained_mm']==pytest.approx(2) and after['discarded_mm']==pytest.approx(1.4)
    assert after['mean_mm']==1 and after['piece_mean_mm']==1
    assert after['retained_mm']+after['discarded_mm']==pytest.approx(before['retained_mm'])
    assert len(doc['replay']['segments'])==4  # no mutation


def test_empty_invalid_and_episode_regrouping():
    doc=fixture()
    empty=filtered_output_statistics([filter_output_segments(doc,1000)])
    assert empty['retained_percent']==0 and empty['mean_mm'] is None and empty['median_mm'] is None
    with pytest.raises(ValueError): filter_output_segments(doc,-1)
    doc['replay']['failures']=[dict(segment_index=0,reference_arc_base=700),dict(segment_index=2,reference_arc_base=750)]
    assert filter_output_segments(doc,0)['ordinary_restarts']==1
    assert filter_output_segments(doc,512)['ordinary_restarts']==1
    doc['replay']['failures'][0]['segment_index']=50
    with pytest.raises(ValueError): filter_output_segments(doc,512)
