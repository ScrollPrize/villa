"""A fit's Lasagna dataset: its identity, its stores and its packs' place."""
import hashlib
import json

import pytest

from lasagna_source import (LasagnaSource, LasagnaSourceError, array_shape_matches,
                            dataset_url, dyadic_coordinate_scale,
                            find_dataset_manifest, lasagna_scale_for, load_lasagna_source,
                            manifest_identity, read_lasagna_manifest, url_identity,
                            write_lasagna_source)


def write_manifest(directory, name='s.lasagna.json', **extra):
    document = {'version': 2, 'groups': {
        channel: {'zarr': f's_{channel}.ome.zarr/4', 'scaledown': 4, 'channels': [channel]}
        for channel in ('nx', 'ny', 'grad_mag')}, **extra}
    path = directory / name
    path.write_text(json.dumps(document))
    return path


def test_a_manifest_without_provenance_is_identified_by_its_bytes(tmp_path):
    path = write_manifest(tmp_path)
    source = read_lasagna_manifest(path)
    assert source.identity == 'sha256-' + hashlib.sha256(path.read_bytes()).hexdigest()
    assert source.group == '4'
    assert source.stores == {key: str(tmp_path / f's_{channel}.ome.zarr')
                             for key, channel in (('normal_x', 'nx'), ('normal_y', 'ny'),
                                                  ('gradient_magnitude', 'grad_mag'))}
    # The same bytes anywhere are the same dataset.
    copy = tmp_path / 'copy'
    copy.mkdir()
    (copy / 's.lasagna.json').write_bytes(path.read_bytes())
    assert read_lasagna_manifest(copy / 's.lasagna.json').identity == source.identity


def test_a_published_manifest_is_identified_by_its_run(tmp_path):
    (tmp_path / 'inference.json').write_text(json.dumps(
        {'run_uuid': '5b17ff6c-3c47-4f7f-8042-9326a683933b', 'manifest': 's.lasagna.json'}))
    path = write_manifest(tmp_path, provenance='inference.json')
    assert read_lasagna_manifest(path).identity == '5b17ff6c-3c47-4f7f-8042-9326a683933b'


def test_a_run_of_another_manifest_does_not_identify_this_one():
    data = b'{"version": 2}'
    digest = 'sha256-' + hashlib.sha256(data).hexdigest()
    assert manifest_identity(data, 's.lasagna.json', {'run_uuid': 'r', 'manifest': 'o.lasagna.json'}) == digest
    assert manifest_identity(data, 's.lasagna.json', {'run_uuid': 'r'}) == 'r'
    assert manifest_identity(data, 's.lasagna.json', {'run_uuid': '../r'}) == digest


def test_a_named_but_missing_provenance_is_refused(tmp_path):
    path = write_manifest(tmp_path, provenance='inference.json')
    with pytest.raises(LasagnaSourceError, match='provenance record'):
        read_lasagna_manifest(path)


def test_a_remote_dataset_is_identified_by_its_url():
    url = 's3://Bucket/scroll/s.lasagna.json'
    identity = url_identity(url)
    assert identity.startswith('url-') and LasagnaSource(identity, '', '4').identity == identity
    # One URL, however spelled; another URL is another dataset.
    for spelling in ('S3://bucket/scroll/s.lasagna.json/', ' s3://BUCKET/scroll/s.lasagna.json#x'):
        assert url_identity(spelling) == identity
    assert dataset_url(url) == 's3://bucket/scroll/s.lasagna.json'
    assert url_identity('https://bucket.s3.amazonaws.com/scroll/s.lasagna.json') != identity
    assert url_identity('s3://bucket/scroll/S.lasagna.json') != identity
    with pytest.raises(LasagnaSourceError, match='not a remote'):
        url_identity('/data/s.lasagna.json')


def test_a_local_copy_of_a_remote_dataset_is_that_dataset(tmp_path):
    # Its provenance record need not have been copied: the URL identifies it.
    path = write_manifest(tmp_path, provenance='inference.json')
    url = 'https://data.example/s.lasagna.json'
    source = read_lasagna_manifest(path, url=url)
    assert (source.identity, source.location) == (url_identity(url), url)
    assert source.stores['normal_x'] == str(tmp_path / 's_nx.ome.zarr')
    assert source.dataset_directory('/cache') == f'/cache/lasagna/{url_identity(url)}'


def test_channels_must_share_one_single_channel_array(tmp_path):
    path = tmp_path / 'n.lasagna.json'
    path.write_text(json.dumps({'version': 2, 'groups': {
        'normals': {'zarr': 'n.zarr/0', 'scaledown': 2, 'channels': ['nx', 'ny']}}}))
    with pytest.raises(LasagnaSourceError, match='single-channel'):
        read_lasagna_manifest(path)
    path.write_text(json.dumps({'version': 2, 'groups': {
        'nx': {'zarr': 'nx.zarr/4', 'scaledown': 4, 'channels': ['nx']},
        'ny': {'zarr': 'ny.zarr/3', 'scaledown': 3, 'channels': ['ny']}}}))
    with pytest.raises(LasagnaSourceError, match='one array'):
        read_lasagna_manifest(path)


def test_a_service_s_source_document_round_trips(tmp_path):
    source = LasagnaSource(identity='run-1', location='https://data/s.lasagna.json', group='4',
                           stores={'normal_x': 'https://data/s_nx.ome.zarr'})
    path = write_lasagna_source(tmp_path / 'project-inputs', source)
    assert path.endswith('lasagna-run-1.json')
    assert load_lasagna_source(path) == source
    # A manifest loads through the same call.
    assert load_lasagna_source(write_manifest(tmp_path)).group == '4'


def test_packs_live_under_the_identity():
    source = LasagnaSource(identity='run-1', location='', group='4',
                           stores={'normal_x': '/nx', 'normal_y': '/ny'})
    assert source.pack_directory('/cache', 'normals') == '/cache/lasagna/run-1/normals_g4'
    assert source.arrays('normals') == ['/nx/4', '/ny/4']
    with pytest.raises(LasagnaSourceError, match='grad_mag'):
        source.arrays('grad_mag')
    with pytest.raises(LasagnaSourceError, match='identity'):
        LasagnaSource(identity='../x', location='', group='4')


def test_a_dataset_s_one_manifest_is_found(tmp_path):
    assert find_dataset_manifest(tmp_path) == ''
    path = write_manifest(tmp_path)
    assert find_dataset_manifest(tmp_path) == str(path)
    write_manifest(tmp_path, name='t.lasagna.json')
    assert find_dataset_manifest(tmp_path) == ''


def test_the_link_refresh_reads_the_normals_pack_when_the_cache_has_it(tmp_path):
    from fit_session import lasagna_normal_inputs
    source = LasagnaSource(identity='run-1', location='', group='4',
                           stores={'normal_x': '/nx', 'normal_y': '/ny'})
    config = {'input_use_normals': True}
    assert lasagna_normal_inputs(source, config, 4, tmp_path) is None
    pack = tmp_path / 'lasagna' / 'run-1' / 'normals_g4'
    pack.mkdir(parents=True)
    (pack / 'meta.json').write_text('{}')
    assert lasagna_normal_inputs(source, config, 4, tmp_path) == {'pool': str(pack), 'scale': 4}
    assert lasagna_normal_inputs(source, {'input_use_normals': False}, 4, tmp_path) is None
    assert lasagna_normal_inputs(None, config, 4, tmp_path) is None


def test_shape_pairing_matches_vc3d_s_dyadic_rule():
    # The cases of core/test/test_lasagna_manifest.cpp.
    l0, l2 = (75784, 32694, 32694), (18946, 8174, 8174)
    assert dyadic_coordinate_scale(l2, l0, 5) == 4.0
    assert dyadic_coordinate_scale(l0, l2, 5) == 0.25
    inclusive = (75784, 32693, 32693)
    assert dyadic_coordinate_scale(inclusive, l0, 5) == 1.0
    assert dyadic_coordinate_scale(l0, inclusive, 5) == 1.0
    with pytest.raises(ValueError):
        dyadic_coordinate_scale((75784, 32692, 32694), l0, 5)
    assert dyadic_coordinate_scale((13, 25, 38), (100, 200, 300), 5) == 8.0
    with pytest.raises(ValueError):
        dyadic_coordinate_scale((18946, 8175, 8174), l0, 5)
    with pytest.raises(ValueError):
        dyadic_coordinate_scale(l2, l2, -1)
    assert array_shape_matches(l0, l2, 4) and not array_shape_matches(l0, l2, 2)


def test_a_manifest_s_frame_gives_the_arrays_downsampling(tmp_path):
    path = write_manifest(tmp_path, base_shape_zyx=[400, 800, 1200])
    source = read_lasagna_manifest(path)
    # scaledown 4 of the fixture: 16 base voxels per array voxel.
    assert (source.base_shape_zyx, source.base_voxels_per_array_voxel) == ((400, 800, 1200), 16.0)
    assert lasagna_scale_for(source, (400, 800, 1200))[0] == 16
    assert lasagna_scale_for(source, (399, 800, 1199))[0] == 16   # inclusive convention
    assert lasagna_scale_for(source, (800, 1600, 2400))[0] == 32  # a 2x finer fit volume
    # The scale survives a local copy of a remote dataset and its source document.
    copy = read_lasagna_manifest(path, url='https://data.example/s.lasagna.json')
    assert load_lasagna_source(write_lasagna_source(tmp_path / 'c', copy)) == copy
    assert lasagna_scale_for(copy, (400, 800, 1200))[0] == 16

    scaled = read_lasagna_manifest(write_manifest(
        tmp_path, name='t.lasagna.json', base_shape_zyx=[400, 800, 1200], source_to_base=0.5))
    assert lasagna_scale_for(scaled, (400, 800, 1200))[0] == 8


def test_the_downsampling_is_not_derived_without_a_frame_or_a_matching_shape(tmp_path):
    bare = read_lasagna_manifest(write_manifest(tmp_path))
    assert bare.base_shape_zyx is None
    assert lasagna_scale_for(bare, (400, 800, 1200)) == (
        None, f'{bare.location} does not state its base_shape_zyx')
    framed = read_lasagna_manifest(write_manifest(
        tmp_path, name='t.lasagna.json', base_shape_zyx=[400, 800, 1200]))
    assert lasagna_scale_for(framed, None)[0] is None
    scale, why = lasagna_scale_for(framed, (300, 800, 1200))
    assert scale is None and 'not levels of one grid' in why
    # A fit volume coarser than the arrays.
    scale, why = lasagna_scale_for(framed, (13, 25, 38))
    assert scale is None and 'not a whole downsampling' in why
    for bad in ({'base_shape_zyx': [1, 2]}, {'source_to_base': 0}):
        with pytest.raises(LasagnaSourceError):
            read_lasagna_manifest(write_manifest(tmp_path, name='u.lasagna.json', **bad))


def test_a_stated_scale_is_derived_cross_checked_or_required(tmp_path):
    from dataclasses import replace

    from fit_session import ScrollSpec, ScrollSpecError, resolve_lasagna_scale
    framed = read_lasagna_manifest(write_manifest(tmp_path, base_shape_zyx=[400, 800, 1200]))
    bare = read_lasagna_manifest(write_manifest(tmp_path, name='t.lasagna.json'))
    spec = ScrollSpec(name='s', voxel_size_um=8.0, spiral_outward_sense='CW',
                      base_shape_zyx=(400, 800, 1200))
    assert resolve_lasagna_scale(spec, framed).lasagna_scale == 16
    assert resolve_lasagna_scale(replace(spec, lasagna_scale=16), framed).lasagna_scale == 16
    with pytest.raises(ScrollSpecError, match='lasagna_scale 4 contradicts'):
        resolve_lasagna_scale(replace(spec, lasagna_scale=4), framed)
    # Underivable: the stated scale stands, and without one the fit is refused.
    assert resolve_lasagna_scale(replace(spec, lasagna_scale=4), bare).lasagna_scale == 4
    with pytest.raises(ScrollSpecError, match='state lasagna_scale'):
        resolve_lasagna_scale(spec, bare)
    with pytest.raises(ScrollSpecError, match='base_shape_zyx'):
        resolve_lasagna_scale(replace(spec, base_shape_zyx=None), framed)
