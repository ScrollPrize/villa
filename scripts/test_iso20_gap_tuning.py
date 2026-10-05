import pytest
import subprocess
from pathlib import Path
from iso20_gap_tuning import fragmentation


def pieces(states):
    return [dict(original_trace='0', begin_arc_base=i*10,
                 end_arc_base=(i+1)*10, winding_final_class=s)
            for i,s in enumerate(states)]


def test_gap_boundaries_and_length_not_piece_count():
    result=fragmentation(pieces(['h','mixed','mixed','v']))
    assert result['gap_runs']==1
    assert result['boundaries']==2
    assert result['interior_gaps']==1
    assert result['disabled_percent']==50
    subdivided=pieces(['h','mixed','mixed','v'])
    subdivided[1]['end_arc_base']=15
    subdivided.insert(2,dict(original_trace='0',begin_arc_base=15,
        end_arc_base=20,winding_final_class='mixed'))
    assert fragmentation(subdivided)==result


def test_endpoints_and_whole_disabled():
    result=fragmentation(pieces(['mixed','h','mixed']))
    assert result['gap_runs']==2
    assert result['boundaries']==2
    assert result['endpoint_gaps']==2
    assert result['interior_gaps']==0
    result=fragmentation(pieces(['mixed','mixed']))
    assert result['gap_runs']==1
    assert result['whole_disabled_traces']==1
    assert result['boundaries']==0


def test_refuse_gaps_in_clean_piece_geometry():
    rows=pieces(['h','v'])
    rows[1]['begin_arc_base']=11
    with pytest.raises(AssertionError):fragmentation(rows)


@pytest.mark.parametrize('unit', ['0', '-1', 'nan', 'inf'])
def test_native_cli_rejects_invalid_length_unit(unit):
    binary=Path(__file__).resolve().parents[1]/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_chunk'
    if not binary.exists():pytest.skip('native benchmark binary not built')
    result=subprocess.run([str(binary),'direction-ablation','unused',
        '--winding-defect-length-unit',unit],text=True,capture_output=True)
    assert result.returncode!=0
    message=result.stdout+result.stderr
    assert '--winding-defect-length-unit' in message
    assert ('requires a finite number' if unit in ('nan','inf') else 'finite and positive') in message


def test_report_exposes_units_and_observed_reference_count(tmp_path,monkeypatch):
    import iso20_gap_tuning as sweep
    monkeypatch.setattr(sweep,'OUT',tmp_path/'artifacts')
    monkeypatch.setattr(sweep.shared,'REPO',tmp_path)
    (tmp_path/'volume-cartographer/docs/fiber_benchmark_runs').mkdir(parents=True)
    row=dict(minimum=64,boundary_cost=1000,defect_cost=300,length_unit=128,messages=500,
        independent_percent=42.3077,observed_ladder_shift=dict(ladder_shift_percent=62.5,references=23),
        disabled_percent=15.77,gap_runs=568,interior_gaps=5,whole_disabled_traces=507,
        boundaries=66,active_components=113,projected_defects=0,tied_rows=4,unbounded_rows=3,seconds=32)
    sweep.report({'example':row},[])
    report=(tmp_path/'volume-cartographer'/sweep.REPORT).read_text()
    assert '128 base vx' in report
    assert '62.50% (23)' in report
    assert 'Unavailable' in report
    assert 'iso20_gap_length_tuning' in report
