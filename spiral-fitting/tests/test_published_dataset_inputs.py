"""Datasets that ship packed resident pools instead of the Lasagna OME-Zarr
stores they were packed from, which is what dl.ash2txt.org publishes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from config import Config
from fit_session import (DEFAULT_NORMAL_ZARR_GROUP, SCROLL_SPEC_FILENAME,
                         SCROLL_SPEC_UNDERIVABLE, ScrollSpecError,
                         SpiralInputPaths, SpiralRunConfig,
                         conventional_input_paths, fit_input,
                         lasagna_sidecar_path, load_scroll_spec,
                         parse_scroll_spec, preflight_dataset,
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


class TestPreflight:
    def test_a_published_dataset_reports_the_sidecar_it_will_read(
            self, tmp_path):
        report = preflight_dataset(published_dataset(tmp_path),
                                   Config().as_dict())
        assert report.ok
        rows = {item.key: item for item in report.inputs}
        assert rows['normal_x'].path.endswith('.respool_g4_pair')
        assert rows['normal_y'].path == rows['normal_x'].path
        assert rows['gradient_magnitude'].path.endswith('.respool_g4')
        assert 'This dataset can start a fit.' in report.report()

    def test_requirements_follow_the_fit_configuration(self, tmp_path):
        root = published_dataset(tmp_path)
        config = Config().as_dict()
        rows = {item.key: item.requirement
                for item in preflight_dataset(root, config).inputs}
        assert rows['winding_inference'] == 'required'
        assert rows['gradient_magnitude'] == 'optional'
        assert rows['tracks_dbm'] == 'off'
        config.update({'dense_spacing_mode': 'grad_mag',
                       'input_use_tracks': True})
        rows = {item.key: item.requirement
                for item in preflight_dataset(root, config).inputs}
        assert rows['winding_inference'] == 'off'
        assert rows['gradient_magnitude'] == 'required'
        assert rows['tracks_dbm'] == 'optional'

    def test_a_missing_scroll_spec_is_reported_with_a_filled_template(
            self, tmp_path):
        root = published_dataset(tmp_path)
        (root / 'spiral-scroll.json').unlink()
        report = preflight_dataset(root, Config().as_dict())
        assert not report.ok
        template = report.scroll_spec_template
        assert template['name'] == root.name
        assert template['normal_zarr_group'] == '4'
        assert template['schema_version'] == 1
        assert SCROLL_SPEC_FILENAME in report.report()

    def test_the_template_placeholders_are_refused_rather_than_guessed(
            self, tmp_path):
        root = published_dataset(tmp_path)
        (root / 'spiral-scroll.json').unlink()
        template = preflight_dataset(root, Config().as_dict()) \
            .scroll_spec_template
        for key in SCROLL_SPEC_UNDERIVABLE:
            assert str(template[key]).startswith('<')
        with pytest.raises(ScrollSpecError):
            parse_scroll_spec(template, root)

    def test_a_track_store_off_the_conventional_name_is_named_in_the_template(
            self, tmp_path):
        root = published_dataset(tmp_path)
        (root / 'spiral-scroll.json').unlink()
        tracks = root / 'tracks'
        tracks.mkdir()
        (tracks / 'PHerc0826_surface_m7_L0_th0.2.dbm').write_text('')
        template = preflight_dataset(root, Config().as_dict()) \
            .scroll_spec_template
        assert template['paths']['tracks_dbm'] == \
            'tracks/PHerc0826_surface_m7_L0_th0.2.dbm'

    def test_an_empty_lasagna_z_roi_blocks_the_fit_before_any_gpu(
            self, tmp_path):
        root = published_dataset(tmp_path)
        write_scroll_spec(root, lasagna_scale=1)
        config = Config().as_dict()
        config.update({'z_begin': 10000, 'z_end': 11000})
        report = preflight_dataset(root, config)
        assert not report.ok
        assert report.missing == ()
        assert any('z-ROI [10000, 4737) is empty' in problem
                   for problem in report.problems)

    def test_a_sidecar_packed_from_other_stores_is_reported_not_hidden(
            self, tmp_path):
        root = published_dataset(tmp_path)
        sidecar = root / f'{NORMAL_X}.respool_g4_pair' / 'meta.json'
        document = json.loads(sidecar.read_text())
        document['channel_names'] = ['other_nx.ome.zarr/4',
                                     'other_ny.ome.zarr/4']
        sidecar.write_text(json.dumps(document))
        report = preflight_dataset(root, Config().as_dict())
        rows = {item.key: item for item in report.inputs}
        assert 'other_nx.ome.zarr/4' in rows['normal_x'].note
        assert rows['normal_x'].blocks is False
        assert report.ok


class TestCheckFlag:
    def run_check(self, root):
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1]
                                 / 'fit_spiral.py'),
             '--dataset', str(root), '--check'],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]))

    def test_a_runnable_dataset_exits_zero(self, tmp_path):
        result = self.run_check(published_dataset(tmp_path))
        assert result.returncode == 0, result.stderr
        assert 'This dataset can start a fit.' in result.stdout

    def test_a_dataset_without_a_scroll_spec_exits_one(self, tmp_path):
        root = published_dataset(tmp_path)
        (root / 'spiral-scroll.json').unlink()
        result = self.run_check(root)
        assert result.returncode == 1
        assert '"schema_version": 1' in result.stdout
