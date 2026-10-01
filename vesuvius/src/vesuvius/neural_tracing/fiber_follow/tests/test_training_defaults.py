"""Production training defaults and removal of mined-location inputs."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, LOCATION_SOURCES,
)
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser, main, options_argv
from vesuvius.neural_tracing.fiber_follow.shared.training_options import normalize_batch_options
from vesuvius.neural_tracing.fiber_follow.shared.training_log import format_training_log


REQUIRED = ['--name', 'test', '--fiber-zarrs', 'unused', '--fibers', 'unused',
            '--ct', 'unused', '--manifest', 'unused']


def test_production_defaults_and_explicit_overrides():
    parser = build_parser()
    args = vars(parser.parse_args(REQUIRED))
    expected = dict(steps=100000, batch=4, grad_steps=2, workers=8, n_commit=16,
                    direction_inputs=True,
                    memory_switch_probability=.3, decision_fraction=.3, clean_fraction=.8,
                    bank_wrong_continuation_probability=0., bank_following_probability=.2,
                    presence_dropout=0., diag_every=5000, dagger_after=96.,
                    recurrent_refinement_steps=2, warmup=500,remote_prefetch_lookahead=16)
    assert {key: args[key] for key in expected} == expected
    assert 'compile' not in args
    for option in ('--microbatch', '--compile', '--no-compile', '--feature-replay-weight', '--memory-slots', '--memory-steps', '--memory-stride', '--feature-sequence-length', '--history-encoder-checkpointing'):
        with pytest.raises(SystemExit):
            parser.parse_args(REQUIRED+[option])
    root = Path(__file__).resolve().parents[1]
    assert args['negative_bank'] == str(root/'output'/'neighbor_samples_r0_32_l80_160_v2')
    custom = parser.parse_args(REQUIRED+['--batch', '16', '--no-direction-inputs',
                                      '--negative-bank', '/tmp/custom-bank','--remote-prefetch-lookahead','0'])
    assert custom.batch == 16 and not custom.direction_inputs
    assert custom.negative_bank == '/tmp/custom-bank'
    assert custom.remote_prefetch_lookahead==0


@pytest.mark.parametrize('option', ['--contacts', '--hard-spans', '--contact-fraction', '--hard-span-fraction'])
def test_removed_location_options_are_rejected(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(REQUIRED+[option, '0'])


@pytest.mark.parametrize('options', [dict(batch=24, microbatch=12), dict(batch=12, grad_steps=2)])
def test_saved_batch_options_preserve_effective_batch(options):
    original = dict(options)
    args = build_parser().parse_args(REQUIRED+options_argv(options))
    assert (args.batch, args.grad_steps) == (12, 2)
    assert options == original
    text = format_training_log(dict(event='resume_configuration', step=1000,
                                    checkpoint='checkpoint.pt', training_options=options))
    assert 'batch 12 / grad steps 2' in text
    assert 'microbatch' not in text


@pytest.mark.parametrize('options', [dict(batch=24, microbatch=0), dict(batch=25, microbatch=12),
                                   dict(batch=24, microbatch=12, grad_steps=1)])
def test_invalid_legacy_batch_options_are_rejected(options):
    with pytest.raises(ValueError):
        normalize_batch_options(options)


@pytest.mark.parametrize('option,value', [('--batch', '0'), ('--grad-steps', '0'), ('--grad-steps', '-1')])
def test_nonpositive_accumulation_settings_are_rejected(option, value):
    with pytest.raises(ValueError, match='Positive counts'):
        main(REQUIRED+[option, value])


def test_fresh_location_only_oversamples_available_lateral_history():
    builder = IdentityObservationBuilder(DirectConfig(), [SimpleNamespace(length=100.)],
                                         IdentitySampling(lateral_fraction=1.))
    rng = np.random.default_rng(0)
    assert builder.fresh_location(rng) is None
    builder.lateral.append((0, 50., False))
    location = builder.fresh_location(rng)
    assert location['fiber'] == 0 and 34 <= location['t'] <= 66
    assert LOCATION_SOURCES[location['source']] == 'lateral'
    assert 'contact' not in LOCATION_SOURCES and 'hard_span' not in LOCATION_SOURCES
