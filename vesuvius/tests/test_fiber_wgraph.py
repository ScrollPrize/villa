"""Public command routing; no data extraction, native library or GPU required."""
import json
from pathlib import Path

import pytest

from vesuvius import wgraph as cli


def write_config(path, document):
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize('stage', cli.STAGES)
def test_input_output_flags_override_profile_for_every_stage(stage, tmp_path):
    input_key, output_key = cli.STORE_KEYS[stage]
    profile = write_config(tmp_path/'settings.json', {input_key: 'old-input', output_key: 'old-output'})
    config = cli.stage_config(stage, profile, input_path='input-file', output_path='new.vcobj')
    assert config[input_key] == 'input-file'
    assert config[output_key] == 'new.vcobj'
    assert config['output_root'] == 'new.vcobj.work'


def test_extraction_output_flag_means_store_not_report(tmp_path):
    profile = write_config(tmp_path/'settings.json', {'output': 'chosen-report.json', 'normal_prior_output': 'chosen-priors.json'})
    config = cli.stage_config('extract', profile, input_path='fibers.vcobj', output_path='relations.vcobj')
    assert config['database'] == 'relations.vcobj'
    assert config['output'] == 'chosen-report.json'
    assert config['normal_prior_output'] == 'chosen-priors.json'
    assert config['chunk_size_base_voxels'] == 2048
    assert 'threads' not in config  # Native runtime physical-core detection.


def test_derived_outputs_and_native_sidecars_have_distinct_paths():
    imported = cli.stage_config('import', input_path='source.afv', output_path='fibers.vcobj')
    extracted = cli.stage_config('extract', input_path='fibers.vcobj', output_path='constraints.vcobj')
    derived = cli.stage_config('hvb', input_path='constraints.vcobj', output_path='classes.vcobj')
    assert imported['report'] == 'fibers.import.json'
    assert extracted['output'] == 'constraints.extraction.json'
    assert extracted['normal_prior_output'] == 'constraints.outward-priors.json'
    assert derived['result_store'] == 'classes.vcobj'
    assert 'output' not in derived


def test_workflow_profile_merges_common_and_selected_stage_only(tmp_path):
    profile = write_config(tmp_path/'settings.json', {
        'version': 1,
        'common': {'native_binaries': {'first': 'a', 'second': 'b'}, 'seed': 7},
        'stages': {
            'extract': {'native_binaries': {'second': 'override'}, 'chunk_size_base_voxels': 1024},
            'hvb': {'seed': 99},
        },
    })
    config = cli.stage_config('extract', profile, input_path='in', output_path='out')
    assert config['seed'] == 7
    assert config['native_binaries'] == {'first': 'a', 'second': 'override'}
    assert config['chunk_size_base_voxels'] == 1024
    assert 'stages' not in config and 'common' not in config


def test_named_profiles_search_explicit_then_environment_then_xdg(tmp_path, monkeypatch):
    explicit, environment, xdg = (tmp_path/name for name in ('explicit', 'env', 'xdg'))
    for path in (explicit, environment, xdg/'vc_wgraph'/'profiles'):
        path.mkdir(parents=True)
    monkeypatch.setenv('VC_WGRAPH_PROFILE_PATH', str(environment))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(xdg))
    for path in (explicit, environment, xdg/'vc_wgraph'/'profiles'):
        write_config(path/'example.json', {})
    assert cli.profile_path('example', [explicit]) == explicit/'example.json'
    assert cli.profile_path('example') == environment/'example.json'
    (environment/'example.json').unlink()
    assert cli.profile_path('example') == xdg/'vc_wgraph'/'profiles'/'example.json'
    assert 'default' in cli.available_profiles()


def test_unknown_profile_and_missing_json_have_distinct_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match='Configuration file does not exist'):
        cli.profile_path(str(tmp_path/'missing.json'))
    with pytest.raises(FileNotFoundError, match='Unknown profile'):
        cli.profile_path('not-a-known-profile')


@pytest.mark.parametrize('document', [[], {'common': []}, {'stages': {'unknown': {}}}, {'stages': {'hvb': []}}, {'common': {}, 'typo': 1}, {'version': 2, 'stages': {}}])
def test_malformed_config_is_rejected(document, tmp_path):
    profile = write_config(tmp_path/'bad.json', document)
    with pytest.raises(ValueError):
        cli.stage_config('hvb', profile, input_path='in', output_path='out')


def test_input_and_output_are_required_and_must_differ():
    with pytest.raises(ValueError, match='requires --input'):
        cli.stage_config('import')
    with pytest.raises(ValueError, match='requires --output'):
        cli.stage_config('import', input_path='in')
    with pytest.raises(ValueError, match='must be different'):
        cli.stage_config('sign', input_path='same', output_path='./same')


def test_print_config_does_not_dispatch_or_import_solver(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'run_stage', lambda *_: pytest.fail('print-config must not execute'))
    assert cli.main(['extract', '-i', 'fibers.vcobj', '-o', 'graph.vcobj', '--print-config']) == 0
    config = json.loads(capsys.readouterr().out)
    assert config['fibers'] == 'fibers.vcobj' and config['database'] == 'graph.vcobj'
    assert not (tmp_path/'graph.vcobj').exists()


def test_cli_dispatches_exact_config_and_reports_actual_store(monkeypatch, capsys):
    calls = []
    def run(stage, config):
        calls.append((stage, config))
        return {'run_uuid': 'e0a2080a-a70a-40b4-8569-70c26b1de861'}
    monkeypatch.setattr(cli, 'run_stage', run)
    assert cli.main(['sign', 'default', '--input', 'classes.vcobj', '--output', 'signs.vcobj']) == 0
    assert calls[0][0] == 'sign'
    assert calls[0][1]['database'] == 'classes.vcobj'
    assert calls[0][1]['result_store'] == 'signs.vcobj'
    report = json.loads(capsys.readouterr().out.removeprefix('WGRAPH_STAGE_COMPLETE '))
    assert report['output'] == 'signs.vcobj' and report['wall_seconds'] >= 0
    assert report['run_uuid'] == 'e0a2080a-a70a-40b4-8569-70c26b1de861'


def test_profiles_command_lists_profiles_without_running(monkeypatch, capsys):
    monkeypatch.setattr(cli, 'run_stage', lambda *_: pytest.fail('profiles must not execute'))
    assert cli.main(['profiles']) == 0
    assert 'default\t' in capsys.readouterr().out


def test_native_dispatch_reuses_existing_native_workflow(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wrapped_full
    calls = []
    monkeypatch.setattr(fiber_wrapped_full, 'native_step', lambda *args, **kwargs: calls.append((args, kwargs)))
    config = cli.stage_config('import', input_path='source.afv', output_path=str(tmp_path/'geometry.vcobj'))
    cli.run_stage('import', config)
    (workflow, name, binary, settings), kwargs = calls[0]
    assert name == 'import' and binary == 'vc_object_import_afv'
    assert workflow['output_root'] == settings['output_root']
    assert settings['output'] == str(tmp_path/'geometry.vcobj')
    assert kwargs == {'timeout': None}
    config['native_timeout_seconds'] = 30
    cli.run_stage('import', config)
    assert calls[1][1] == {'timeout': 30}
