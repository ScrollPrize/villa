"""Production training defaults and removal of mined-location inputs."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, LOCATION_SOURCES,
)
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser


REQUIRED = ['--name', 'test', '--fiber-zarrs', 'unused', '--fibers', 'unused',
            '--ct', 'unused', '--manifest', 'unused']


def test_production_defaults_and_explicit_overrides():
    parser = build_parser()
    args = vars(parser.parse_args(REQUIRED))
    expected = dict(steps=100000, batch=8, microbatch=4, workers=8, n_commit=16,
                    direction_inputs=True, feature_switch_crop_fraction=.15,
                    memory_switch_probability=.3, decision_fraction=.3,
                    bank_wrong_continuation_probability=0., bank_following_probability=.2,
                    presence_dropout=0., diag_every=5000, dagger_after=96.,
                    recurrent_refinement_steps=2, warmup=500)
    assert {key: args[key] for key in expected} == expected
    assert 'compile' not in args
    for option in ('--compile', '--no-compile'):
        with pytest.raises(SystemExit):
            parser.parse_args(REQUIRED+[option])
    root = Path(__file__).resolve().parents[1]
    assert args['negative_bank'] == str(root/'output'/'neighbor_samples_r0_32_l80_160_v2')
    custom = parser.parse_args(REQUIRED+['--batch', '16', '--no-direction-inputs',
                                      '--negative-bank', '/tmp/custom-bank'])
    assert custom.batch == 16 and not custom.direction_inputs
    assert custom.negative_bank == '/tmp/custom-bank'


@pytest.mark.parametrize('option', ['--contacts', '--hard-spans', '--contact-fraction', '--hard-span-fraction'])
def test_removed_location_options_are_rejected(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(REQUIRED+[option, '0'])


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
