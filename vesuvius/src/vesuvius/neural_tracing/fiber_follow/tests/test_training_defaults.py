"""The coordinate_regression launcher's configuration, removed options and fresh-location sampling."""
from pathlib import Path
import shlex
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder, IdentitySampling, LOCATION_SOURCES
from vesuvius.neural_tracing.fiber_follow.train import train
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.train.train import build_parser, main
from vesuvius.neural_tracing.fiber_follow.data.data import TASKS, TaskBudget
from model_fixtures import REQUIRED

ROOT = Path(__file__).resolve().parents[1]


def launcher_argv():
    text = (ROOT/'scripts'/'train_coordinate_regression.sh').read_text().replace('\\\n', ' ')
    command = shlex.split(next(line for line in text.splitlines() if line.startswith('exec ')))
    argv = command[command.index('vesuvius.neural_tracing.fiber_follow.train.train')+1:]
    substitute = {'$task_root': str(ROOT), '$task_run': 'coordinate_regression', '$source_checkpoint': '/unused/ckpt.pt'}
    for name, value in substitute.items():
        argv = [arg.replace(name, value) for arg in argv]
    return [arg for arg in argv if arg != '$@']


def test_coordinate_regression_launcher_parses_to_the_planned_model_and_budget():
    args = build_parser().parse_args(launcher_argv())
    assert args.dataset_config == str(ROOT/'configs'/'mixed_ct_datasets_paris50.json')
    assert args.init_weights == '/unused/ckpt.pt' and args.resume is None
    assert (args.batch, args.grad_steps, args.n_commit, args.steps) == (4, 3, 16, 40000)
    cfg = train.model_config_from_args(args)
    assert cfg.input_channels == 1
    assert cfg.model_type == 'coordinate_regression'
    assert cfg.input_channels == 1 and cfg.recurrent_refinement_steps == 3
    budget = TaskBudget.parse(args.task_share, terminal_fallback_cap=args.terminal_fallback_cap,
                              replay_max_age=args.replay_max_age, replay_event_cap=args.replay_event_cap)
    assert dict(zip(TASKS, budget.shares)) == dict(
        fresh=.40, live=.25, dagger_pre_excursion=.08, dagger_recoverable=.06, dagger_terminal=.08,
        dagger_premature_stop=.03, dagger_ordinary=.05, synthetic_terminal=.05)
    assert (budget.replay_max_age, budget.replay_event_cap, budget.terminal_fallback_cap) == (12000, 64, .5)
    for option, value in (('--batch', '0'), ('--grad-steps', '0'), ('--grad-steps', '-1')):
        with pytest.raises(ValueError, match='Positive counts'):
            main(REQUIRED+[option, value])


def test_removed_options_are_rejected():
    removed = ['--contacts', '--hard-spans', '--contact-fraction', '--hard-span-fraction', '--fresh-fraction',
               '--clean-fraction', '--decision-fraction', '--decision-choice-fraction', '--candidate-weight',
               '--pair-rank-weight', '--gt-perturb-probability', '--replay-continuation-fraction',
               '--replay-failure-fraction', '--memory-switch-probability', '--bank-following-probability',
               '--bank-wrong-continuation-probability', '--no-history-prob', '--short-history-prob', '--dagger-seeds',
               '--following-bank', '--microbatch', '--feature-replay-weight', '--memory-slots', '--memory-steps',
               '--memory-stride', '--feature-sequence-length', '--memory-version', '--proposal-step',
               '--proposal-warmup-steps']
    switches = ['--live-continuation', '--live-continuation-stratified', '--correct-replay-only',
                '--prefer-real-wrong-turns', '--prefer-replay-for-light-gt', '--compile', '--no-compile',
                '--history-encoder-checkpointing']
    parser = build_parser()
    assert 'compile' not in vars(parser.parse_args(REQUIRED))
    for argv in [[option, '1'] for option in removed]+[[flag] for flag in switches]:
        with pytest.raises(SystemExit) as error:
            parser.parse_args(REQUIRED+argv)
        assert error.value.code == 2, argv


def test_task_budget_must_be_a_complete_distribution():
    for shares in (['fresh=.5'], ['unknown=.1'], ['fresh=-.1', 'live=.75']):
        with pytest.raises(ValueError):
            TaskBudget.parse(shares)


def test_fresh_location_only_oversamples_available_lateral_history():
    builder = IdentityObservationBuilder(CoordinateRegressionConfig(), [SimpleNamespace(length=100.)],
                                         IdentitySampling(lateral_fraction=1.))
    rng = np.random.default_rng(0)
    assert builder.fresh_location(rng) is None
    builder.lateral.append((0, 50., False))
    location = builder.fresh_location(rng)
    assert location['fiber'] == 0 and 34 <= location['t'] <= 66
    assert LOCATION_SOURCES[location['source']] == 'lateral'
    assert 'contact' not in LOCATION_SOURCES and 'hard_span' not in LOCATION_SOURCES


def test_initialized_run_can_resume_with_lower_lr_without_changing_training_contract():
    original = build_parser().parse_args(launcher_argv())
    resumed = SimpleNamespace(**vars(original))
    resumed.init_weights = None
    resumed.resume = '/unused/run/last.pt'
    resumed.lr = 5e-5
    train.validate_resume_options(resumed, vars(original))
    resumed.steps = original.steps + 50000
    train.validate_resume_options(resumed, vars(original))
    # The confidence-label tolerance may change on resume; the departure threshold is a constant.
    train.validate_resume_options(SimpleNamespace(**dict(vars(resumed), tolerance=original.tolerance+.5)), vars(original))
    for key, value in [('warmup', 0), ('n_commit', 8)]:
        changed = SimpleNamespace(**vars(resumed))
        setattr(changed, key, value)
        with pytest.raises(ValueError, match=f'Resume option differs: {key}'):
            train.validate_resume_options(changed, vars(original))


def test_lr_step_offset_continues_the_original_cosine_after_its_warmup():
    import math
    from vesuvius.neural_tracing.fiber_follow.train.runloop import lr_at
    original = lambda step: lr_at(step, 1e-4, 1000, 100000)
    offset, warmup, steps = 32000, 500, 68000
    continued = lambda own: lr_at(own, 1e-4, warmup, steps, offset)
    for own in (500, 1000, 20000, 68000):
        assert math.isclose(continued(own), original(offset+own), rel_tol=1e-12)
    assert continued(1) < original(offset+1)
