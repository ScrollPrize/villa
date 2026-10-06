"""The trainer's command line (a run configuration only), removed options and fresh-location sampling."""
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder, IdentitySampling, LOCATION_SOURCES
from vesuvius.neural_tracing.fiber_follow.train import run_config
from vesuvius.neural_tracing.fiber_follow.train.train import build_parser
from vesuvius.neural_tracing.fiber_follow.data.data import TaskBudget
from model_fixtures import follower_config, run_document


def test_the_command_line_takes_a_run_configuration_and_an_optional_resume():
    args = build_parser().parse_args(['--config', 'run.json'])
    assert vars(args) == dict(config='run.json', resume=None)
    assert build_parser().parse_args(['--config', 'run.json', '--resume', 'last.pt']).resume == 'last.pt'
    for argv in (['--name', 'x'], ['--config', 'run.json', '--batch', '4'], ['--config', 'run.json', '--model', 'flow'], []):
        with pytest.raises(SystemExit) as error:
            build_parser().parse_args(argv)
        assert error.value.code == 2, argv


@pytest.mark.parametrize('section, key', [
    ('training', 'memory'), ('training', 'identity_weight'), ('training', 'history_grad_clip'),
    ('training', 'rest_grad_clip'), ('training', 'task_share'), ('training', 'chain_seed_fraction'),
    ('training', 'tube_weight'), ('training', 'memory_augmentation'), ('training', 'identity_switch_tail'),
    ('model', 'memory'), ('model', 'identity_dim'), ('model', 'path_planes'), ('model', 'tube_head'),
    ('model', 'stem_channels'), ('model', 'decoder_layers'), ('model', 'scorer_layers'), ('model', 'encoder_ffn')])
def test_removed_options_are_rejected(section, key):
    document = run_document()
    document.setdefault(section, {})[key] = 1
    with pytest.raises(ValueError, match=f'Unknown {section} settings: {key}'):
        run_config.resolve(document)


@pytest.mark.parametrize('key', ['batch', 'grad_steps', 'steps'])
def test_counts_must_be_positive(key):
    for value in (0, -1):
        with pytest.raises(ValueError, match='Positive counts'):
            run_config.resolve(run_document(training={key: value}))


def test_task_budget_must_be_a_complete_distribution():
    for shares in (['fresh=.5'], ['unknown=.1'], ['fresh=-.1', 'live=.75'], ['synthetic_identity=0']):
        with pytest.raises(ValueError):
            TaskBudget.parse(shares)
    with pytest.raises(ValueError):
        run_config.resolve(run_document(training=dict(task_shares=dict(fresh=.5))))


def test_fresh_location_only_oversamples_available_lateral_history():
    builder = IdentityObservationBuilder(follower_config(), [SimpleNamespace(length=100.)],
                                         IdentitySampling(lateral_fraction=1.))
    rng = np.random.default_rng(0)
    assert builder.fresh_location(rng) is None
    builder.lateral.append((0, 50., False))
    location = builder.fresh_location(rng)
    assert location['fiber'] == 0 and 34 <= location['t'] <= 66
    assert LOCATION_SOURCES[location['source']] == 'lateral'
    assert 'contact' not in LOCATION_SOURCES and 'hard_span' not in LOCATION_SOURCES


def test_lr_step_offset_continues_the_original_cosine_after_its_warmup():
    import math
    from vesuvius.neural_tracing.fiber_follow.train.runloop import lr_at
    original = lambda step: lr_at(step, 1e-4, 1000, 100000)
    offset, warmup, steps = 32000, 500, 68000
    continued = lambda own: lr_at(own, 1e-4, warmup, steps, offset)
    for own in (500, 1000, 20000, 68000):
        assert math.isclose(continued(own), original(offset+own), rel_tol=1e-12)
    assert continued(1) < original(offset+1)
