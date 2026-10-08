import json
import pytest
from reviewed_cache_export import CHANNELS, export_source
from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import export_union_lasagna_manifests


@pytest.mark.parametrize('level', [1, 3])
def test_export_excludes_scratch_and_incomplete_chunks(tmp_path, level):
    root, output = tmp_path/'cache', tmp_path/'export'
    root.mkdir()
    (root/'inference.json').write_text(json.dumps(dict(source=dict(locator='s3://bucket/PHercParis4/volumes/test.zarr'))))
    groups = {}
    for channel in CHANNELS:
        array = root/channel/str(level)
        (array/'0/0').mkdir(parents=True)
        (array/'0/0/0').write_bytes(b'chunk')
        (array/'.scratch').mkdir()
        (array/'.scratch/largefile').write_bytes(b'exclude')
        (root/channel/'.hidden').mkdir()
        (root/channel/'.hidden/scratch').write_bytes(b'exclude')
        (array/'.zarray').write_text(json.dumps(dict(shape=[8,8,8], chunks=[4,4,4], dimension_separator='/')))
        (root/channel/'.zgroup').write_text('{"zarr_format":2}')
        groups[channel] = dict(zarr=f'{channel}/{level}', scaledown=level)
    (root/CHANNELS[0]/str(level)/'0/0/1').write_bytes(b'incomplete')
    (root/'inference.lasagna.json').write_text(json.dumps(dict(groups=groups, base_shape_zyx=[8*2**level]*3)))
    entry = export_source((root, output))
    assert entry['chunk_count'] == 1
    assert entry['incomplete_chunks_excluded'] == 1
    assert sum(entry['channel_bytes'].values()) == 15
    (output/'manifest.json').write_text(json.dumps(dict(vc3d_channels=[entry])))
    export_union_lasagna_manifests(output)
    dest = output/'vc3d_channels/PHercParis4'
    assert (dest/'inference.lasagna.json').is_file()
    for channel in CHANNELS:
        assert (dest/channel/str(level)/'0/0/0').is_symlink()
        assert not (dest/channel/str(level)/'0/0/1').exists()
        assert not (dest/channel/'.hidden').exists()
        assert not (dest/channel/str(level)/'.scratch').exists()
    export_source((root, output))  # same inventory resumes
    for channel in CHANNELS:
        (root/channel/str(level)/'0/0/1').write_bytes(b'new')
    with pytest.raises(ValueError, match='cache changed'):
        export_source((root, output))
