from iso20_event_bp import configuration


def test_only_local_events_added_to_objective_and_source_unchanged():
    args=['direction-ablation','trace.zarr','--output','old',
          '--piece-length','512','--winding-hard-signs','both','--threads','128']
    source=dict(commands=[dict(name='old',args=args)],input_files=['trace.zarr/.zattrs'])
    config=configuration(source)
    result=config['commands'][0]['args']
    assert result[1]=='trace.zarr'
    assert result[-1]=='--local-constraint-events'
    assert '--local-no-parallel' not in result
    result[result.index('--output')+1]='old'
    assert result[:-1]==args
    assert config['input_files']==source['input_files']
