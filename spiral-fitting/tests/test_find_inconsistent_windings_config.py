import json

import pytest
import click
from click.testing import CliRunner

from find_inconsistent_windings import main


@pytest.fixture
def cli_config(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoint.ckpt'
    checkpoint.touch()
    pcl = tmp_path / 'pcl.json'
    pcl.write_text('{}')
    config = {
        'checkpoint': str(checkpoint),
        'patches_dir': str(tmp_path),
        'patch-id': 'seed',
        'pcl': [str(pcl)],
        'z_range': '100,200',
        'fibers': '',
        'detect_loops': False,
        'plot': True,
        'max_hops': 3,
    }
    path = tmp_path / 'params.json'
    captured = {}
    # Exercise real CLI parsing without loading scan data or requiring CUDA.
    monkeypatch.setattr(main, 'callback', lambda **kwargs: captured.update(kwargs))
    return path, config, captured


def test_config_supplies_required_options_and_defaults(cli_config):
    path, config, captured = cli_config
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(main, ['--config', str(path)])
    assert result.exit_code == 0, result.output
    assert captured['checkpoint'] == config['checkpoint']
    assert captured['pcl_paths'] == tuple(config['pcl'])
    assert captured['fibers_flag'] == ''
    assert captured['detect_loops'] is False
    assert captured['plot'] is True
    assert captured['max_hops'] == 3
    assert captured['step_size'] == 1.0


@pytest.mark.parametrize('config_first', [True, False])
def test_cli_overrides_config(cli_config, config_first):
    path, config, captured = cli_config
    path.write_text(json.dumps(config))
    overrides = ['--max-hops', '0', '--detect-loops', '--no-plot',
                 '--pcl', config['checkpoint']]
    config_args = ['--config', str(path)]
    result = CliRunner().invoke(main, config_args + overrides if config_first else overrides + config_args)
    assert result.exit_code == 0, result.output
    assert captured['max_hops'] == 0
    assert captured['detect_loops'] is True
    assert captured['plot'] is False
    assert captured['pcl_paths'] == (config['checkpoint'],)


@pytest.mark.parametrize(('contents', 'error'), [
    ('{', 'cannot read JSON configuration'),
    ('[]', 'must contain a JSON object'),
    ('{"max_hop": 2}', 'unknown option'),
    ('{"max_hops": 2, "max-hops": 3}', 'duplicate option'),
])
def test_bad_config_reports_cli_error(cli_config, contents, error):
    path, _, captured = cli_config
    path.write_text(contents)
    result = CliRunner().invoke(main, ['--config', str(path)])
    assert result.exit_code == 2
    assert error in result.output
    assert not captured


@pytest.mark.parametrize(('key', 'value'), [
    ('max_hops', 'bad-integer'),
    ('checkpoint', '/nonexistent/checkpoint.ckpt'),
    ('pcl', 'must-be-an-array'),
])
def test_config_values_receive_click_validation(cli_config, key, value):
    path, config, captured = cli_config
    config[key] = value
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(main, ['--config', str(path)])
    assert result.exit_code == 2, result.output
    assert not captured


def test_cli_without_config(cli_config):
    _, config, captured = cli_config
    result = CliRunner().invoke(main, [
        '--checkpoint', config['checkpoint'], '--patches-dir', config['patches_dir'],
        '--patch-id', 'seed', '--pcl', config['pcl'][0], '--z-range', '100,200',
    ])
    assert result.exit_code == 0, result.output
    assert captured['detect_loops'] is True
    assert captured['plot'] is False
    assert captured['max_hops'] is None


def test_json_only_linking_config(cli_config, monkeypatch):
    path, config, captured = cli_config
    config['linking'] = {'distance': 1.75, 'min_points': 3, 'fiber_stride': 4}
    path.write_text(json.dumps(config))

    def capture(**kwargs):
        captured.update(kwargs)
        captured['linking'] = click.get_current_context().meta['analysis_linking']

    monkeypatch.setattr(main, 'callback', capture)
    result = CliRunner().invoke(main, ['--config', str(path)])
    assert result.exit_code == 0, result.output
    assert captured['linking'].distance == 1.75
    assert captured['linking'].min_points == 3
    assert captured['linking'].fiber_stride == 4
    result = CliRunner().invoke(main, ['--config', str(path), '--linking-distance', '2'])
    assert result.exit_code == 2
    assert 'No such option' in result.output


@pytest.mark.parametrize('linking', [
    None, [], {'distance': -1}, {'window_points': 2}, {'min_points': 6},
    {'horizontal_max_front_fraction': 2}, {'fiber_stride': False}, {'min_point': 3},
])
def test_bad_linking_config_fails_before_main(cli_config, linking):
    path, config, captured = cli_config
    config['linking'] = linking
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(main, ['--config', str(path)])
    assert result.exit_code == 2
    assert 'linking' in result.output
    assert not captured
