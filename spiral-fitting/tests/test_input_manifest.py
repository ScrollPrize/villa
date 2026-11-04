"""The command-line fit's explicit input manifest (fit_session.load_input_manifest)
and the shared input validation (validate_fit_inputs)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from config import Config
from fit_session import (ScrollSpecError, SpiralInputPaths, load_input_manifest,
                         scroll_spec_config_defaults, validate_fit_inputs)
from lasagna_source import load_lasagna_source, url_identity

SPIRAL_DIR = Path(__file__).resolve().parents[1]
URL = 'https://data.example.org/scroll/predictions.lasagna.json'
# The configuration reading none of the supervision inputs.
NO_SUPERVISION = {'input_use_normals': False, 'input_use_gradient_magnitude': False,
                  'input_use_fiber_directions': False,
                  'dense_spacing_mode': 'grad_mag'}


def write_lasagna(directory, name='predictions.lasagna.json', **extra):
    directory.mkdir(parents=True, exist_ok=True)
    document = {'version': 2, 'groups': {
        'nx': {'zarr': 'nx.ome.zarr/4', 'channels': ['nx'], 'scaledown': 4},
        'ny': {'zarr': 'ny.ome.zarr/4', 'channels': ['ny'], 'scaledown': 4},
        'grad_mag': {'zarr': 'gm.ome.zarr/4', 'channels': ['grad_mag'], 'scaledown': 4}},
        **extra}
    path = directory / name
    path.write_text(json.dumps(document))
    return path


def write_manifest(directory, filename='fit-inputs.json', **entries):
    """A manifest beside arbitrarily named inputs in no conventional layout."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'centre-line.json').write_text(json.dumps({'control_points': [
        {'z': 0, 'y': 1, 'x': 2}, {'z': 10, 'y': 1, 'x': 2}]}))
    document = {'schema_version': 1, 'volume_uuid': 'volume-1',
                'scroll': {'name': 's1', 'voxel_size_um': 8.0,
                           'spiral_outward_sense': 'CW'},
                'umbilicus': {'path': 'centre-line.json'}}
    for key, value in entries.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    path = directory / filename
    path.write_text(json.dumps(document))
    return path


def load(path, tmp_path, **kwargs):
    return load_input_manifest(path, cache_directory=tmp_path / 'cache', **kwargs)


def test_relative_paths_resolve_against_the_manifest_not_the_working_directory(
        tmp_path, monkeypatch):
    inputs = tmp_path / 'anywhere' / 'nested'
    write_lasagna(inputs / 'lasagna-copy')
    (inputs / 'fd.npz').write_bytes(b'')
    (inputs / 'winding').mkdir()
    manifest = write_manifest(
        inputs, lasagna={'manifest': 'lasagna-copy/predictions.lasagna.json'},
        fiber_directions='fd.npz', winding_inference='winding',
        scroll_zarr='https://data.example.org/scroll.zarr/0')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    scroll, paths = load(Path(os.path.relpath(manifest, elsewhere)), tmp_path)
    root = inputs.resolve()
    assert scroll.name == 's1' and scroll.spiral_outward_sense == 'CW'
    assert paths.volume_uuid == 'volume-1'
    assert paths.umbilicus == str(root / 'centre-line.json')
    assert paths.lasagna == str(root / 'lasagna-copy' / 'predictions.lasagna.json')
    assert paths.fiber_directions == str(root / 'fd.npz')
    assert paths.winding_inference == str(root / 'winding')
    # URLs remain URLs.
    assert paths.scroll_zarr == 'https://data.example.org/scroll.zarr/0'
    assert paths.provenance['input_manifest'] == str(manifest.resolve())
    assert len(paths.provenance['sha256']) == 64
    # The resolved inputs are what session and checkpoint metadata record.
    assert SpiralInputPaths.from_mapping(paths.manifest()) == paths


def test_supervision_entries_are_optional_and_nothing_is_discovered(tmp_path):
    inputs = tmp_path / 'inputs'
    # Conventionally named files beside the manifest are not inputs.
    write_lasagna(inputs / 'lasagna_inputs')
    (inputs / 'fiber_directions.npz').write_bytes(b'')
    (inputs / 'winding_inference').mkdir(parents=True)
    _, paths = load(write_manifest(inputs), tmp_path)
    assert (paths.lasagna, paths.fiber_directions, paths.winding_inference,
            paths.scroll_zarr) == ('', '', '', '')


@pytest.mark.parametrize('entries, message', [
    ({'schema_version': 2}, 'schema_version must be 1'),
    ({'volume_uuid': None}, 'volume_uuid'),
    ({'scroll': None}, 'scroll block is required'),
    ({'umbilicus': {'path': 'https://example.org/u.json'}}, 'must be a local path'),
    ({'scroll': {'name': 's1', 'spiral_outward_sense': 'CW'}}, 'voxel_size_um'),
    ({'lasagna': {'manifest': URL}}, 'must be a local path'),
    ({'lasagna': {'manifest': 'l.json', 'original_url': '/local/path'}}, 'original_url'),
])
def test_malformed_manifests_are_refused(tmp_path, entries, message):
    with pytest.raises(ScrollSpecError, match=message):
        load(write_manifest(tmp_path / 'inputs', **entries), tmp_path)


@pytest.mark.parametrize('key, guidance', [
    ('name', 'belongs in the scroll block'),
    ('verified_patches', 'geometry is read from --object-store only'),
])
def test_retired_and_geometry_keys_are_refused_with_guidance(tmp_path, key, guidance):
    with pytest.raises(ScrollSpecError, match=guidance):
        load(write_manifest(tmp_path / 'inputs', **{key: 'x'}), tmp_path)


def test_orientation_is_validated_with_catalog_flags_and_the_sense_setting(tmp_path):
    inputs = tmp_path / 'inputs'
    oriented = {'name': 's1', 'voxel_size_um': 8.0, 'z_direction_is_top_to_bottom': True,
                'left_handed_coordinates': False}
    scroll, _ = load(write_manifest(inputs, scroll=oriented), tmp_path)
    assert scroll.spiral_outward_sense == 'ACW'
    assert scroll.z_direction_is_top_to_bottom is True
    with pytest.raises(ScrollSpecError, match='contradicts the catalog'):
        load(write_manifest(inputs, scroll={**oriented, 'spiral_outward_sense': 'CW'}),
             tmp_path)
    with pytest.raises(ScrollSpecError, match='contradicts the catalog'):
        load(write_manifest(inputs, scroll=oriented), tmp_path, outward_sense='CW')
    bare = {'name': 's1', 'voxel_size_um': 8.0}
    with pytest.raises(ScrollSpecError, match='set the fit setting spiral_outward_sense'):
        load(write_manifest(inputs, scroll=bare), tmp_path)
    scroll, _ = load(write_manifest(inputs, scroll=bare), tmp_path, outward_sense='ACW')
    assert scroll.spiral_outward_sense == 'ACW'
    with pytest.raises(ScrollSpecError, match='contradicts the fit setting'):
        load(write_manifest(inputs, scroll={**bare, 'spiral_outward_sense': 'CW'}),
             tmp_path, outward_sense='ACW')


def test_a_local_copy_of_a_remote_dataset_takes_its_cache_identity(tmp_path):
    inputs = tmp_path / 'inputs'
    write_lasagna(inputs / 'copy')
    manifest = write_manifest(inputs, lasagna={
        'manifest': 'copy/predictions.lasagna.json', 'original_url': URL})
    _, paths = load(manifest, tmp_path)
    source = load_lasagna_source(paths.lasagna)
    # The URL's identity, so its packs are a service's fitting the URL; the
    # stores stay the local copy's, and nothing is downloaded.
    assert source.identity == url_identity(URL)
    assert source.location == URL
    assert Path(paths.lasagna).parent == Path(source.dataset_directory(tmp_path / 'cache'))
    assert source.stores['normal_x'] == str(inputs / 'copy' / 'nx.ome.zarr')
    assert paths.provenance['lasagna_original_url'] == URL
    # Without the URL the copy is a local dataset of its own identity.
    _, local = load(write_manifest(inputs, lasagna={
        'manifest': 'copy/predictions.lasagna.json'}), tmp_path)
    assert load_lasagna_source(local.lasagna).identity != url_identity(URL)


def test_inputs_the_configuration_needs_must_be_named_and_exist(tmp_path):
    inputs = tmp_path / 'inputs'
    _, paths = load(write_manifest(inputs), tmp_path)
    paths = SpiralInputPaths.from_mapping({**paths.manifest(),
                                           'cache_directory': str(tmp_path / 'cache')})
    hints = {'lasagna': 'name it as lasagna.manifest in the input manifest'}
    assert validate_fit_inputs(paths, NO_SUPERVISION, missing=hints) == []

    normals = {**NO_SUPERVISION, 'input_use_normals': True,
               'loss_weight_dense_normals': 1.0}
    [error] = validate_fit_inputs(paths, normals, missing=hints)
    assert error['field'] == 'lasagna'
    assert 'lasagna.manifest' in error['message']
    fibers = {**NO_SUPERVISION, 'input_use_fiber_directions': True,
              'loss_weight_fiber_directions': 1.0}
    assert [error['field'] for error in validate_fit_inputs(paths, fibers)] == [
        'fiber_directions']
    winding = {**NO_SUPERVISION, 'dense_spacing_mode': 'winding_model'}
    assert [error['field'] for error in validate_fit_inputs(paths, winding)] == [
        'winding_inference']
    # A named input that is not there is reported with its path.
    missing = SpiralInputPaths.from_mapping({**paths.manifest(),
                                             'fiber_directions': str(inputs / 'nope.npz')})
    [error] = validate_fit_inputs(missing, fibers)
    assert 'nope.npz' in error['message']
    # The umbilicus is always required.
    [error] = validate_fit_inputs(
        SpiralInputPaths.from_mapping({**paths.manifest(), 'umbilicus': ''}),
        NO_SUPERVISION, missing={'umbilicus': 'name it as umbilicus.path'})
    assert error['field'] == 'umbilicus' and 'umbilicus.path' in error['message']


def test_a_scroll_winding_count_is_a_configuration_default(tmp_path):
    inputs = tmp_path / 'inputs'
    scroll, _ = load(write_manifest(inputs), tmp_path)
    assert scroll_spec_config_defaults(scroll, Config().as_dict()) == {}
    scroll, _ = load(write_manifest(inputs, scroll={
        'name': 's1', 'voxel_size_um': 8.0, 'spiral_outward_sense': 'CW',
        'num_windings': 70}), tmp_path)
    defaults = scroll_spec_config_defaults(scroll, Config().as_dict())
    assert defaults['shell_outer_winding_idx'] == 70
    assert defaults['model_gap_expander_num_windings'] == 70
    assert scroll_spec_config_defaults(
        {'num_windings': 70}, {'model_gap_expander_capacity_windings': 60})[
            'model_gap_expander_capacity_windings'] == 73


def test_the_cli_fit_refuses_store_objects_of_another_volume(tmp_path, monkeypatch):
    pytest.importorskip('vc_object_store')
    from vc_object_store import ObjectStore
    import object_store_reference as reference
    from object_store_fixture import VOLUME, plane, put_patch
    from config import FitConfig
    from fit_spiral import FitContext
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')
    store = str(tmp_path / 'fit.vcobj')
    with ObjectStore.create(store, author='ingest', hostname='host') as db:
        put_patch(db, plane(), (0.25, 0.25), name='near')
    umbilicus = str(tmp_path / 'umbilicus.json')
    reference.write_umbilicus(umbilicus)

    def context(volume):
        return FitContext(
            FitConfig({**reference.fit_config(), 'object_store_path': store,
                       'object_store_author': 'cli'}),
            scroll=reference.scroll_spec(),
            paths=SpiralInputPaths(volume_uuid=volume, umbilicus=umbilicus),
            cache_dir=str(tmp_path / 'cache'), out_base_dir=str(tmp_path / 'out'),
            input_log=lambda line: None)
    with pytest.raises(ScrollSpecError, match=f'selected store objects are fitted in {VOLUME}'):
        context('another-volume').load_host_inputs()
    context(VOLUME).load_host_inputs()


def test_a_retired_cli_flag_is_refused_with_guidance():
    result = subprocess.run(
        [sys.executable, str(SPIRAL_DIR / 'fit_spiral.py'), '--dataset', '/data/scroll'],
        capture_output=True, text=True, cwd=SPIRAL_DIR, timeout=300)
    assert result.returncode == 2
    assert '--dataset is retired' in result.stderr
    assert '--inputs MANIFEST' in result.stderr


def test_the_cli_reports_every_missing_input_before_fitting(tmp_path):
    manifest = write_manifest(tmp_path / 'inputs', fiber_directions='absent.npz')
    result = subprocess.run(
        [sys.executable, str(SPIRAL_DIR / 'fit_spiral.py'), '--inputs', str(manifest),
         '--object-store', str(tmp_path / 'absent.vcobj'),
         '--cache', str(tmp_path / 'cache')],
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
        env={**os.environ, 'WANDB_MODE': 'disabled', 'FIT_SPIRAL_CONFIG_OVERRIDES':
             json.dumps({'input_use_fiber_directions': True,
                         'loss_weight_fiber_directions': 1.0,
                         'input_use_normals': True,
                         'loss_weight_dense_normals': 1.0})})
    assert result.returncode != 0
    assert 'cannot be fitted' in result.stderr
    assert 'absent.npz' in result.stderr
    assert 'lasagna: No Lasagna dataset' in result.stderr
    assert 'lasagna.manifest in the input manifest' in result.stderr


