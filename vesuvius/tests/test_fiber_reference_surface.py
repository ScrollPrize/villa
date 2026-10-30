import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_surface import relative_key,digest


def test_relative_keys_and_digest(tmp_path):
    assert str(relative_key('prefix/x.tif','prefix/'))=='x.tif'
    assert str(relative_key('prefix/sub/mask.tif','prefix/'))=='sub/mask.tif'
    for key in ('outside/x.tif','prefix/../x.tif','prefix//x.tif','prefix/./x.tif','prefix/'):
        with pytest.raises(ValueError):relative_key(key,'prefix/')
    path=tmp_path/'empty';path.touch()
    assert digest(path)=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


@pytest.fixture
def fake_s3(monkeypatch):
    from datetime import datetime,timezone
    from types import SimpleNamespace
    boto3=pytest.importorskip('boto3')
    remote=SimpleNamespace(objects={'prefix/x.tif':b'x','prefix/y.tif':b'y'},downloads=[])
    class Client:
        def get_paginator(self,name):return self
        def paginate(self,**kwargs):
            return [{'Contents':[dict(Key=k,Size=len(v),ETag=v.hex(),
                LastModified=datetime(2026,1,1,tzinfo=timezone.utc)) for k,v in remote.objects.items()]}]
        def download_file(self,bucket,key,path,Config):
            from pathlib import Path
            remote.downloads.append(key);Path(path).write_bytes(remote.objects[key])
    monkeypatch.setattr(boto3,'client',lambda *a,**kw:Client())
    return remote


def test_download_reuse_and_update(tmp_path,fake_s3):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_surface import run
    cfg=dict(source='s3://bucket/prefix/',output_root=str(tmp_path))
    first=run(cfg)
    assert first['object_count']==2 and len(fake_s3.downloads)==2
    assert run(dict(cfg,source='s3://bucket/prefix'))['objects']==first['objects']
    assert len(fake_s3.downloads)==2
    fake_s3.objects['prefix/x.tif']=b'new-x'
    run(cfg)
    assert len(fake_s3.downloads)==3 and (tmp_path/'x.tif').read_bytes()==b'new-x'


@pytest.mark.parametrize('change', ['removed','different_bucket','different_prefix','untracked'])
def test_refresh_rejects_mixed_source_before_writes(tmp_path,fake_s3,change):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_surface import run
    cfg=dict(source='s3://bucket/prefix/',output_root=str(tmp_path))
    run(cfg)
    if change=='removed':del fake_s3.objects['prefix/y.tif']
    elif change=='different_bucket':cfg['source']='s3://other/prefix/'
    elif change=='different_prefix':
        cfg['source']='s3://bucket/other/'
        fake_s3.objects={'other/x.tif':b'other-x'}
    else:(tmp_path/'stale.tif').write_bytes(b'stale')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='source changed|Stale/unlisted'):
        run(cfg)
    assert len(fake_s3.downloads)==2
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_resume_partial_but_reject_stale_partial(tmp_path,fake_s3):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_surface import run
    cfg=dict(source='s3://bucket/prefix/',output_root=str(tmp_path))
    (tmp_path/'x.tif.download-part').write_bytes(b'interrupted')
    run(cfg)
    assert not (tmp_path/'x.tif.download-part').exists()
    (tmp_path/'removed.tif.download-part').write_bytes(b'stale')
    with pytest.raises(ValueError,match='Stale/unlisted'):run(cfg)
