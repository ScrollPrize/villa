"""Configurable winding-graph commands over shared native and solver stages.

Profiles contain settings, not implementations. The native importer/extractor
and downstream adapters own all data access, computation and store writes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import time


STAGES = ('import', 'extract', 'hvb', 'sign', 'winding', 'uv')
STORE_KEYS = {
    'import': ('source', 'output'),
    'extract': ('fibers', 'database'),
    **{stage: ('database', 'result_store') for stage in ('hvb', 'sign', 'winding', 'uv')},
}
BUILTIN_PROFILES = Path(__file__).with_name('wgraph_profiles')


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _read_object(path: Path) -> dict:
    with path.open() as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f'Configuration must be a JSON object: {path}')
    return document


def profile_directories(extra=()) -> list[Path]:
    """First match wins; local profiles never need to be installed in the repo."""
    paths = [Path(p).expanduser() for p in extra]
    paths.extend(Path(p).expanduser() for p in os.environ.get('VC_WGRAPH_PROFILE_PATH', '').split(os.pathsep) if p)
    config = Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home()/'.config'))).expanduser()
    paths.extend((config/'vc_wgraph'/'profiles', BUILTIN_PROFILES))
    return list(dict.fromkeys(paths))


def profile_path(selector: str, extra=()) -> Path:
    candidate = Path(selector).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.suffix.lower() == '.json' or candidate.is_absolute() or '/' in selector or '\\' in selector:
        raise FileNotFoundError(f'Configuration file does not exist: {candidate}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', selector):
        raise ValueError(f'Invalid profile name: {selector}')
    for directory in profile_directories(extra):
        path = directory/(selector+'.json')
        if path.is_file():
            return path
    searched = ', '.join(str(p) for p in profile_directories(extra))
    raise FileNotFoundError(f'Unknown profile {selector!r}; searched {searched}')


def available_profiles(extra=()) -> dict[str, Path]:
    result = {}
    for directory in profile_directories(extra):
        if directory.is_dir():
            for path in sorted(directory.glob('*.json')):
                result.setdefault(path.stem, path)
    return result


def _stage_document(document: dict, stage: str) -> dict:
    if 'common' not in document and 'stages' not in document:
        return deepcopy(document)
    unknown = document.keys()-{'common', 'stages', 'version'}
    if unknown:
        raise ValueError('Workflow profile has unknown root fields: '+', '.join(sorted(unknown)))
    if document.get('version', 1) != 1:
        raise ValueError('Unsupported workflow profile version')
    common, stages = document.get('common', {}), document.get('stages', {})
    if not isinstance(common, dict) or not isinstance(stages, dict):
        raise ValueError('Profile common and stages fields must be JSON objects')
    unknown_stages = stages.keys()-set(STAGES)
    if unknown_stages:
        raise ValueError('Unknown profile stages: '+', '.join(sorted(unknown_stages)))
    specific = stages.get(stage, {})
    if not isinstance(specific, dict):
        raise ValueError(f'Profile stage {stage!r} must be a JSON object')
    return _merge(common, specific)


def stage_config(stage: str, selector='default', *, input_path=None, output_path=None, profile_dirs=()) -> dict:
    """Resolve settings and overrides without importing a solver or touching data."""
    if stage not in STAGES:
        raise ValueError(f'Unknown winding-graph stage: {stage}')
    defaults = _stage_document(_read_object(BUILTIN_PROFILES/'default.json'), stage)
    selected = _stage_document(_read_object(profile_path(str(selector), profile_dirs)), stage)
    config = _merge(defaults, selected)
    input_key, output_key = STORE_KEYS[stage]
    if input_path is not None:
        config[input_key] = str(Path(input_path).expanduser())
    if output_path is not None:
        config[output_key] = str(Path(output_path).expanduser())
    for key, option in ((input_key, '--input'), (output_key, '--output')):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f'{stage} requires {option}, or a nonempty {key!r} in its configuration')
    if Path(config[input_key]).expanduser().resolve() == Path(config[output_key]).expanduser().resolve():
        raise ValueError('Input and output must be different; stages create new immutable stores')
    output = Path(config[output_key]).expanduser()
    config.setdefault('output_root', str(output.with_name(output.name+'.work')))
    if stage == 'import':
        config.setdefault('report', str(output.with_name(output.stem+'.import.json')))
    elif stage == 'extract':
        config.setdefault('output', str(output.with_name(output.stem+'.extraction.json')))
        config.setdefault('normal_prior_output', str(output.with_name(output.stem+'.outward-priors.json')))
    return config


def _run_native(stage: str, config: dict):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_wrapped_full import native_step
    settings = deepcopy(config)
    if stage == 'extract' and settings.get('umbilicus') and not settings.get('umbilicus_sha256'):
        from lasagna.inference_provenance import sha256_file
        settings['umbilicus_sha256'] = sha256_file(settings['umbilicus'])
    binary = {'import': 'vc_object_import_afv', 'extract': 'vc_fiber_constraints'}[stage]
    return native_step(settings, stage, binary, settings, timeout=settings.get('native_timeout_seconds'))


def run_stage(stage: str, config: dict):
    if stage in ('import', 'extract'):
        return _run_native(stage, config)
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_wgraph_stages import run_stage as run_derived_stage
    return run_derived_stage(stage, config)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog='vc_wgraph', description=__doc__)
    command.add_argument('--profile-dir', action='append', default=[], metavar='DIRECTORY',
                         help='additional named-profile directory (highest priority; repeatable)')
    subcommands = command.add_subparsers(dest='stage', required=True)
    for stage in STAGES:
        subcommand = subcommands.add_parser(stage, help=f'run the {stage} stage')
        subcommand.add_argument('profile', nargs='?', default='default', metavar='PROFILE_OR_JSON',
                                help='named profile or arbitrary JSON configuration (default: default)')
        subcommand.add_argument('-i', '--input', dest='input_path', metavar='FILE', help='override the input AFV/vcobj store')
        subcommand.add_argument('-o', '--output', dest='output_path', metavar='FILE', help='override the new output vcobj store')
        subcommand.add_argument('--print-config', action='store_true', help='print resolved settings without running the stage')
    subcommands.add_parser('profiles', help='list available named profiles and their source files')
    return command


def main(argv=None) -> int:
    command = parser()
    args = command.parse_args(argv)
    if args.stage == 'profiles':
        for name, path in sorted(available_profiles(args.profile_dir).items()):
            print(f'{name}\t{path}')
        return 0
    try:
        config = stage_config(args.stage, args.profile, input_path=args.input_path,
                              output_path=args.output_path, profile_dirs=args.profile_dir)
    except (OSError, ValueError) as error:
        command.error(str(error))
    if args.print_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    started = time.monotonic()
    result = run_stage(args.stage, config)
    input_key, output_key = STORE_KEYS[args.stage]
    report = dict(stage=args.stage, input=config[input_key], output=config[output_key],
                  wall_seconds=time.monotonic()-started)
    if isinstance(result, dict) and result.get('run_uuid'):
        report['run_uuid'] = result['run_uuid']
    print('WGRAPH_STAGE_COMPLETE '+json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
