"""wandb sweep orchestrator for fit_spiral.py.

Subcommands:

  dump-config   Write the full default Config as JSON (the base a sweep
                builds on; edit a copy and pass it as the spec's/run's
                --base-config).
  create        Validate a sweep spec (JSON) against the Config schema and
                register the sweep with wandb; prints the sweep id.
  launch        Partition the available GPUs into slots of --gpus-per-run and
                run one `wandb agent` per slot, each pinned to its GPUs via
                CUDA_VISIBLE_DEVICES.
  run           create + launch in one call.

Any Config key can be swept: parameters in the spec are sweep dimensions,
entries in the spec's "fixed" dict are constants applied to every run, and
loss families are enabled/disabled by sweeping their loss_weight_* key with
0.0 among the values. Every sweep-assigned config runs the full fit+eval
chain once per seed in the spec's "seeds" list (default [1, 2, 3]) before the
agent moves to the next config; the sweep objective is the across-seed mean.
The spec's "eval" section configures the post-fit render_ink +
get_ink_metrics chain (see paris4_sweep.json in this directory).

``--reuse-root`` may be repeated on ``create``/``run`` to reuse exact-matching
completed seeds from older local sweep output directories.  It is independent
of the new wandb sweep id: ``run`` always creates a new sweep before launch.

Example (from scripts/spiral/):

  python sweep/sweep_fit_spiral.py run --spec sweep/paris4_sweep.json \
      --dataset /data/spiral_dataset --gpus 0,1,2,3,4,5,6,7 --gpus-per-run 2
"""

import argparse
import difflib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import Config
DEFAULT_METRIC = {'name': 'ink/total_fg_pixels', 'goal': 'maximize'}

# spec "eval" key -> sweep_run_wrapper.py flag (value-taking options)
EVAL_OPTION_FLAGS = {
    'ink_volume': '--ink-volume',
    'vc_render_bin': '--vc-render-bin',
    'ink_python': '--ink-python',
    'render_procs': '--render-procs',
    'flatboi_threads': '--flatboi-threads',
    'ink_preview_downsample': '--ink-preview-downsample',
}
EVAL_BOOL_FLAGS = {'tta': '--tta', 'skip_eval': '--skip-eval'}


def parameter_values(param):
    """The concrete values a sweep parameter definition can assign (for
    validation); distribution bounds are checked separately."""
    if 'value' in param:
        return [param['value']]
    return list(param.get('values', []))


def is_group_parameter(fields, key, param):
    """A group parameter is a non-Config-key name whose every value is a dict
    of Config-key -> value: one composite sweep value setting several keys at
    once. This is the conditional-parameter workaround — wandb sweeps have no
    native conditionals, so each dict enumerates one branch (e.g. a
    dense_spacing_mode together with the knobs that only matter in that mode)
    and the grid crosses branches instead of the inert full product."""
    values = parameter_values(param)
    return (key not in fields and values
            and all(isinstance(v, dict) for v in values))


def check_key_value(fields, errors, key, value, where):
    if key == 'optimizer_random_seed':
        errors.append(f'{where}optimizer_random_seed cannot be swept or '
                      'fixed: the wrapper runs every config once per seed '
                      'in the spec\'s "seeds" list')
    elif key not in fields:
        close = difflib.get_close_matches(key, fields, n=3)
        hint = f' (did you mean {", ".join(close)}?)' if close else ''
        errors.append(f'{where}unknown Config key: {key!r}{hint}')
    else:
        try:
            Config({key: value})
        except ValueError as e:
            errors.append(f'{where}{key}: invalid value {value!r} ({e})')


def raise_if_errors(errors):
    if errors:
        raise SystemExit('sweep spec validation failed:\n  '
                         + '\n  '.join(errors)
                         + '\n(dump-config lists every valid Config key)')


def validate_spec_parameters(parameters):
    fields = Config.catalog()['schema']['fields']
    errors = []
    plain_keys = {key for key, param in parameters.items()
                  if not is_group_parameter(fields, key, param)}
    for key, param in parameters.items():
        if is_group_parameter(fields, key, param):
            for branch in parameter_values(param):
                for inner_key, inner_value in branch.items():
                    check_key_value(fields, errors, inner_key, inner_value, f'{key}: ')
                    if inner_key in plain_keys:
                        errors.append(f'{key}: inner key {inner_key!r} collides '
                                      'with another swept/fixed parameter')
            continue
        values = parameter_values(param)
        if key == 'optimizer_random_seed':
            check_key_value(fields, errors, key, None, '')
            continue
        if key not in fields:
            if any(isinstance(v, dict) for v in values):
                errors.append(f'{key}: group parameters need EVERY value to be '
                              'a dict of Config keys')
            else:
                check_key_value(fields, errors, key, None, '')
            continue
        for value in values:
            check_key_value(fields, errors, key, value, '')
        if fields[key]['type'] in ('integer', 'number'):
            spec = fields[key]
            for bound in ('min', 'max'):
                if bound in param and not (
                        spec['minimum'] <= param[bound] <= spec['maximum']):
                    errors.append(
                        f'{key}: {bound}={param[bound]} outside '
                        f"[{spec['minimum']}, {spec['maximum']}]")
    raise_if_errors(errors)


def expand_parameter_grid(parameters):
    """Every config in the grid cross-product, as flat Config-key dicts
    (group branches flattened). Requires enumerable values on every parameter."""
    fields = Config.catalog()['schema']['fields']
    keys = list(parameters)
    per_key_values = []
    for key in keys:
        values = parameter_values(parameters[key])
        if not values:
            raise SystemExit(f'{key}: "extra_configs" needs an enumerable grid, '
                             'so every parameter must use value/values, not a '
                             'distribution')
        per_key_values.append(values)
    configs = []
    for combo in itertools.product(*per_key_values):
        flat = {}
        for key, value in zip(keys, combo):
            if key not in fields and isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        configs.append(flat)
    return configs


def validate_extra_configs(extras, fixed):
    fields = Config.catalog()['schema']['fields']
    errors = []
    for i, extra in enumerate(extras):
        where = f'extra_configs[{i}]: '
        if not isinstance(extra, dict):
            errors.append(f'{where}must be a dict of Config keys '
                          '(an empty dict {} runs the plain baseline)')
            continue
        for key, value in extra.items():
            check_key_value(fields, errors, key, value, where)
        overlap = set(extra) & set(fixed)
        if overlap:
            errors.append(f'{where}overrides "fixed" keys {sorted(overlap)}')
    raise_if_errors(errors)


def build_wrapper_command(spec, args):
    cmd = [sys.executable, str(SCRIPT_DIR / 'sweep_run_wrapper.py'),
           '--dataset', str(Path(args.dataset).resolve()),
           '--out-root', str(Path(args.out_root).resolve())]
    if args.scroll_spec:
        cmd += ['--scroll-spec', str(Path(args.scroll_spec).resolve())]
    if args.cache:
        cmd += ['--cache', str(Path(args.cache).resolve())]
    if args.base_config:
        cmd += ['--base-config', str(Path(args.base_config).resolve())]
    for reuse_root in args.reuse_root:
        cmd += ['--reuse-root', str(Path(reuse_root).resolve())]
    eval_cfg = spec.get('eval', {})
    unknown = set(eval_cfg) - set(EVAL_OPTION_FLAGS) - set(EVAL_BOOL_FLAGS)
    if unknown:
        raise SystemExit(f'unknown eval keys in spec: {sorted(unknown)}; '
                         f'valid: {sorted(EVAL_OPTION_FLAGS) + sorted(EVAL_BOOL_FLAGS)}')
    for key, flag in EVAL_OPTION_FLAGS.items():
        if key in eval_cfg:
            cmd += [flag, str(eval_cfg[key])]
    for key, flag in EVAL_BOOL_FLAGS.items():
        if eval_cfg.get(key):
            cmd.append(flag)
    if not eval_cfg.get('skip_eval') and 'ink_volume' not in eval_cfg:
        raise SystemExit('spec eval section needs "ink_volume" '
                         '(or "skip_eval": true to fit without the ink objective)')
    seeds = spec.get('seeds', [1, 2, 3])
    if (not isinstance(seeds, list) or not seeds
            or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)):
        raise SystemExit(f'spec "seeds" must be a non-empty list of distinct '
                         f'integers, got {seeds!r}')
    cmd += ['--seeds', ','.join(map(str, seeds))]
    objective = spec.get('objective')
    if objective:
        bad = [name for name in objective
               if not name.startswith(('final/', 'ink/'))]
        if bad:
            raise SystemExit(f'objective terms must be final/* or ink/* metrics, '
                             f'got: {bad}')
        if eval_cfg.get('skip_eval') and any(n.startswith('ink/') for n in objective):
            raise SystemExit('objective uses ink/* terms but the spec sets '
                             'eval.skip_eval')
        cmd += ['--objective-json', json.dumps(objective)]
    cmd += ['--params-json', '${args_json}']
    return cmd


def build_sweep_dict(spec, args):
    swept = dict(spec.get('parameters', {}))
    extras = spec.get('extra_configs', [])
    if not swept and not extras:
        raise SystemExit('sweep spec needs "parameters" (crossed dimensions) '
                         'and/or "extra_configs" (explicit single configs)')
    parameters = dict(swept)
    fixed = spec.get('fixed', {})
    Config(fixed)  # validates keys and values in one shot
    for key, value in fixed.items():
        if key in parameters:
            raise SystemExit(f'{key!r} appears in both "fixed" and "parameters"')
        parameters[key] = {'value': value}
    validate_spec_parameters(parameters)

    if extras:
        # wandb grids can only cross parameters, so single extra runs (e.g. a
        # baseline plus one modification) require enumerating the whole grid
        # ourselves: cross-product + extras become explicit configs under one
        # composite "run_config" group parameter that the wrapper flattens.
        if spec.get('method', 'bayes') != 'grid':
            raise SystemExit('"extra_configs" requires "method": "grid"')
        validate_extra_configs(extras, fixed)
        # With no swept parameters the sweep is a pure enumeration: exactly
        # the listed extra configs, nothing crossed.
        configs = expand_parameter_grid(parameters) if swept else []
        num_crossed = len(configs)
        seen = {json.dumps(c, sort_keys=True) for c in configs}
        for extra in extras:
            config = fixed | extra
            fingerprint = json.dumps(config, sort_keys=True)
            if fingerprint in seen:
                print(f'note: extra config {extra} duplicates a grid config; skipped')
                continue
            seen.add(fingerprint)
            configs.append(config)
        print(f'expanded grid: {len(configs)} configs '
              f'({num_crossed} crossed + {len(configs) - num_crossed} extra)')
        parameters = {'run_config': {'values': configs}}

    metric = spec.get('metric')
    if metric is None:
        # A combined-objective spec optimizes the wrapper-computed weighted
        # sum by default; otherwise fall back to the raw ink objective.
        metric = ({'name': 'objective', 'goal': 'maximize'}
                  if spec.get('objective') else DEFAULT_METRIC)
    if metric['name'].startswith('ink/') and spec.get('eval', {}).get('skip_eval'):
        raise SystemExit(f"metric {metric['name']!r} needs the eval chain, "
                         'but the spec sets eval.skip_eval')
    return {
        'name': spec.get('name', 'fit-spiral-sweep'),
        'method': spec.get('method', 'bayes'),
        'metric': metric,
        'parameters': parameters,
        'command': ['${env}'] + build_wrapper_command(spec, args),
    }


def cmd_dump_config(args):
    out = Path(args.out)
    out.write_text(json.dumps(Config().as_dict(), indent=2) + '\n')
    print(f'wrote {len(Config().as_dict())} default config keys to {out}')
    print('note: files in configs/ are auto-published as interactive-service '
          'presets, so keep sweep base configs elsewhere unless that is intended')


def cmd_create(args):
    spec = json.loads(Path(args.spec).read_text())
    sweep_dict = build_sweep_dict(spec, args)
    if args.dry_run:
        print(json.dumps(sweep_dict, indent=2))
        return None
    import wandb
    sweep_id = wandb.sweep(sweep_dict, project=args.project, entity=args.entity)
    print(f'created sweep {sweep_id} (project={args.project}, entity={args.entity})')
    print(f'launch with: python {Path(__file__).name} launch --sweep-id {sweep_id} '
          f'--project {args.project} --gpus <ids> --gpus-per-run <n>')
    return sweep_id


def make_gpu_slots(gpus_arg, gpus_per_run):
    gpus = [t.strip() for t in gpus_arg.split(',') if t.strip()]
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f'duplicate GPU ids in --gpus: {gpus_arg}')
    if gpus_per_run < 1 or len(gpus) % gpus_per_run != 0:
        raise SystemExit(f'{len(gpus)} GPUs are not divisible into slots of {gpus_per_run}')
    return [gpus[i:i + gpus_per_run] for i in range(0, len(gpus), gpus_per_run)]


def spawn_agent(slot, sweep_id, args, env_base):
    env = env_base | {'CUDA_VISIBLE_DEVICES': ','.join(slot)}
    env.setdefault('WANDB_MODE', 'online')
    cmd = [sys.executable, '-m', 'wandb', 'agent']
    if args.entity:
        cmd += ['-e', args.entity]
    if args.project:
        cmd += ['-p', args.project]
    if args.runs_per_agent:
        cmd += ['--count', str(args.runs_per_agent)]
    cmd.append(sweep_id)
    print(f'[launch] agent on GPUs {",".join(slot)}: {" ".join(cmd)}', flush=True)
    # Own session: the agent and every descendant (run wrapper, torchrun, fit,
    # eval chain) share one process group, so retiring the slot can kill the
    # whole tree. Killing only the agent leaves ~10 GiB orphaned fits holding
    # the slot's GPUs, and successive relaunches stack them until CUDA OOM.
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def kill_agent_tree(agent, signum=signal.SIGTERM):
    """Signal the agent's whole process group; returns False if already gone."""
    try:
        os.killpg(agent['proc'].pid, signum)
        return True
    except ProcessLookupError:
        return False


def cmd_launch(args, sweep_id=None):
    sweep_id = sweep_id or args.sweep_id
    if not sweep_id:
        raise SystemExit('--sweep-id is required')
    slots = make_gpu_slots(args.gpus, args.gpus_per_run)
    env_base = dict(os.environ)
    agents = {i: {'proc': spawn_agent(slot, sweep_id, args, env_base),
                  'slot': slot, 'restarts': 0}
              for i, slot in enumerate(slots)}
    try:
        while agents:
            time.sleep(5)
            for i in list(agents):
                agent = agents[i]
                code = agent['proc'].poll()
                if code is None:
                    continue
                if code == 0:
                    print(f'[launch] agent on GPUs {",".join(agent["slot"])} '
                          'finished (sweep exhausted or --runs-per-agent reached)',
                          flush=True)
                    del agents[i]
                elif agent['restarts'] < args.max_agent_restarts:
                    agent['restarts'] += 1
                    print(f'[launch] agent on GPUs {",".join(agent["slot"])} exited '
                          f'with {code}; restart {agent["restarts"]}/'
                          f'{args.max_agent_restarts}', flush=True)
                    # A crashed agent can leave its run wrapper / fit alive;
                    # reap them before a new agent claims the same GPUs.
                    kill_agent_tree(agent)
                    time.sleep(10)
                    kill_agent_tree(agent, signal.SIGKILL)
                    agent['proc'] = spawn_agent(agent['slot'], sweep_id, args, env_base)
                else:
                    print(f'[launch] agent on GPUs {",".join(agent["slot"])} exited '
                          f'with {code}; restart budget exhausted, retiring slot',
                          flush=True)
                    kill_agent_tree(agent)
                    del agents[i]
    except KeyboardInterrupt:
        print('[launch] interrupted; stopping agents', flush=True)
        for agent in agents.values():
            kill_agent_tree(agent, signal.SIGINT)
        deadline = time.time() + 60
        for agent in agents.values():
            try:
                agent['proc'].wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                kill_agent_tree(agent, signal.SIGKILL)
        raise SystemExit(130)
    print('[launch] all agents done', flush=True)


def cmd_run(args):
    sweep_id = cmd_create(args)
    if sweep_id is not None:
        cmd_launch(args, sweep_id=sweep_id)


def add_create_args(p):
    p.add_argument('--spec', required=True, help='sweep spec JSON')
    p.add_argument('--dataset', required=True, help='dataset root for fit_spiral.py')
    p.add_argument('--scroll-spec', default=None)
    p.add_argument('--cache', default=None)
    p.add_argument('--base-config', default=None,
                   help='Config-overrides JSON applied under the sweep params '
                        '(start from dump-config output)')
    p.add_argument('--out-root', default='sweep_out',
                   help='per-run outputs go to <out-root>/<sweep_id>/<run_id>')
    p.add_argument('--reuse-root', action='append', default=[],
                   help='old local sweep output directory used as a completed-seed '
                        'cache; repeat for multiple sources (a new wandb sweep is '
                        'still created)')
    p.add_argument('--project', default='scrolls')
    p.add_argument('--entity', default=None)
    p.add_argument('--dry-run', action='store_true',
                   help='print the wandb sweep dict and exit')


def add_launch_args(p, require_sweep_id):
    if require_sweep_id:
        p.add_argument('--sweep-id', required=True)
        p.add_argument('--project', default='scrolls')
        p.add_argument('--entity', default=None)
    p.add_argument('--gpus', required=True,
                   help='comma list of GPU ids available to the sweep, e.g. 0,1,2,3')
    p.add_argument('--gpus-per-run', type=int, default=1)
    p.add_argument('--runs-per-agent', type=int, default=None,
                   help='stop each agent after N runs (default: run until the '
                        'sweep is exhausted)')
    p.add_argument('--max-agent-restarts', type=int, default=3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('dump-config', help='write the full default Config as JSON')
    p.add_argument('--out', default='sweep_base_config.json')
    p.set_defaults(func=cmd_dump_config)

    p = sub.add_parser('create', help='register a sweep from a spec JSON')
    add_create_args(p)
    p.set_defaults(func=cmd_create)

    p = sub.add_parser('launch', help='run wandb agents pinned to GPU slots')
    add_launch_args(p, require_sweep_id=True)
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser('run', help='create + launch')
    add_create_args(p)
    add_launch_args(p, require_sweep_id=False)
    p.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
