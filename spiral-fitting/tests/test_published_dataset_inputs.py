"""Datasets that ship packed resident pools instead of the Lasagna OME-Zarr
stores they were packed from, which is what dl.ash2txt.org publishes."""

import json

from config import Config
from fit_session import (DEFAULT_NORMAL_ZARR_GROUP, SpiralInputPaths,
                         SpiralRunConfig, conventional_input_paths,
                         fit_input, lasagna_sidecar_path, load_scroll_spec,
                         resolve_dataset_root, validate_session_request)
from pack_resident_pools import sidecar_path


LASAGNA = 'lasagna_inputs'
NORMAL_X = f'{LASAGNA}/las_008_nx.ome.zarr'
GRAD_MAG = f'{LASAGNA}/las_008_grad_mag.ome.zarr'


def write_scroll_spec(root, **extra):
    document = {
        'schema_version': 1,
        'name': 'PHercParis4',
        'voxel_size_um': 9.6,
        'spiral_outward_sense': 'CW',
        **extra,
    }
    (root / 'spiral-scroll.json').write_text(json.dumps(document))


def write_sidecar(root, relative):
    sidecar = root / relative
    sidecar.mkdir(parents=True, exist_ok=True)
    (sidecar / 'meta.json').write_text(json.dumps({
        'format': 'respool', 'version': 2,
        'array_shape': [4737, 2044, 2044], 'dtype': 'u1',
    }))
    return sidecar


def published_dataset(root, group=DEFAULT_NORMAL_ZARR_GROUP):
    """The published PHercParis4 layout: sidecars, no base stores."""
    write_scroll_spec(root)
    (root / 'umbilicus.json').write_text('{}')
    for name in ('abs_winding.json', 'relative_windings.json',
                 'same_windings.json'):
        (root / name).write_text('{}')
    (root / 'verified_patches').mkdir()
    (root / 'outer_shell').mkdir()
    inference = root / 'winding_inference'
    inference.mkdir()
    (inference / 'manifest.json').write_text(json.dumps({
        'artifact_type': 'winding_inference_crossings', 'format_version': 1}))
    (root / LASAGNA).mkdir()
    write_sidecar(root, f'{NORMAL_X}.respool_g{group}_pair')
    write_sidecar(root, f'{GRAD_MAG}.respool_g{group}')
    for name in ('output', 'cache'):
        (root / name).mkdir()
    return root


def request_for(root):
    spec = load_scroll_spec(root)
    paths = conventional_input_paths(
        root, spec, output_directory=str(root / 'output'),
        cache_directory=str(root / 'cache'))
    config = Config().as_dict()
    run = SpiralRunConfig(z_begin=config['z_begin'], z_end=config['z_end'],
                          config=config)
    return paths, run, spec


class TestSidecarNaming:
    def test_names_match_the_published_directories(self):
        assert lasagna_sidecar_path(NORMAL_X, '4', pair=True) == \
            f'{NORMAL_X}.respool_g4_pair'
        assert lasagna_sidecar_path(GRAD_MAG, '4') == f'{GRAD_MAG}.respool_g4'
        assert lasagna_sidecar_path(f'{NORMAL_X}/', '2', pair=True) == \
            f'{NORMAL_X}.respool_g2_pair'

    def test_the_packer_and_the_catalog_use_one_definition(self):
        assert sidecar_path is lasagna_sidecar_path

    def test_the_catalog_points_the_normal_pair_at_the_nx_store(self):
        assert fit_input('normal_x').sidecar_store == 'normal_x'
        assert fit_input('normal_y').sidecar_store == 'normal_x'
        assert fit_input('normal_x').sidecar_pair is True
        assert fit_input('normal_y').sidecar_pair is True
        assert fit_input('gradient_magnitude').sidecar_store == \
            'gradient_magnitude'
        assert fit_input('gradient_magnitude').sidecar_pair is False


class TestResolution:
    def test_a_packed_dataset_resolves_every_lasagna_input(self, tmp_path):
        resolution = resolve_dataset_root(published_dataset(tmp_path))
        assert resolution.ok
        for key in ('normal_x', 'normal_y', 'gradient_magnitude'):
            assert key not in resolution.missing_optional
            assert resolution.resolved[key].endswith(
                {'normal_x': 'las_008_nx.ome.zarr',
                 'normal_y': 'las_008_ny.ome.zarr',
                 'gradient_magnitude': 'las_008_grad_mag.ome.zarr'}[key])

    def test_an_unpacked_store_still_resolves_on_its_own(self, tmp_path):
        published_dataset(tmp_path)
        for relative in (f'{NORMAL_X}.respool_g4_pair',
                         f'{GRAD_MAG}.respool_g4'):
            (tmp_path / relative / 'meta.json').unlink()
        for relative in (NORMAL_X, f'{LASAGNA}/las_008_ny.ome.zarr', GRAD_MAG):
            (tmp_path / relative).mkdir()
        resolution = resolve_dataset_root(tmp_path)
        assert resolution.ok
        for key in ('normal_x', 'normal_y', 'gradient_magnitude'):
            assert key not in resolution.missing_optional

    def test_the_scroll_spec_chooses_which_sidecar_group_counts(self, tmp_path):
        published_dataset(tmp_path, group='2')
        write_scroll_spec(tmp_path, normal_zarr_group='4')
        missing = resolve_dataset_root(tmp_path).missing_optional
        assert {'normal_x', 'normal_y', 'gradient_magnitude'} <= set(missing)
        write_scroll_spec(tmp_path, normal_zarr_group='2')
        missing = resolve_dataset_root(tmp_path).missing_optional
        assert not {'normal_x', 'normal_y', 'gradient_magnitude'} & set(missing)

    def test_a_pool_without_its_metadata_is_not_a_sidecar(self, tmp_path):
        published_dataset(tmp_path)
        (tmp_path / f'{NORMAL_X}.respool_g4_pair' / 'meta.json').unlink()
        missing = resolve_dataset_root(tmp_path).missing_optional
        assert {'normal_x', 'normal_y'} <= set(missing)
        assert 'gradient_magnitude' not in missing


class TestValidation:
    def test_a_packed_dataset_needs_no_base_stores(self, tmp_path):
        paths, run, spec = request_for(published_dataset(tmp_path))
        fields = {error['field']
                  for error in validate_session_request(paths, run, spec)}
        assert not {'normal_x', 'normal_y', 'gradient_magnitude'} & fields

    def test_a_store_that_is_neither_present_nor_packed_still_errors(
            self, tmp_path):
        published_dataset(tmp_path)
        (tmp_path / f'{NORMAL_X}.respool_g4_pair' / 'meta.json').unlink()
        paths, run, spec = request_for(tmp_path)
        fields = {error['field']
                  for error in validate_session_request(paths, run, spec)}
        assert {'normal_x', 'normal_y'} <= fields

    def test_a_service_shaped_request_accepts_the_published_layout(
            self, tmp_path):
        root = published_dataset(tmp_path)
        resolution = resolve_dataset_root(root)
        paths = SpiralInputPaths.from_mapping({
            'dataset_root': resolution.root,
            **{key: value for key, value in resolution.resolved.items()},
            'pcls': resolution.pcl_inputs,
            'output_directory': str(root / 'output'),
            'cache_directory': str(root / 'cache'),
        })
        config = Config().as_dict()
        run = SpiralRunConfig(z_begin=config['z_begin'],
                              z_end=config['z_end'], config=config)
        assert validate_session_request(
            paths, run, load_scroll_spec(root)) == []

    def test_without_a_scroll_spec_the_schema_default_group_is_assumed(
            self, tmp_path):
        paths, run, _ = request_for(published_dataset(tmp_path))
        fields = {error['field'] for error in validate_session_request(paths, run)}
        assert not {'normal_x', 'normal_y', 'gradient_magnitude'} & fields
