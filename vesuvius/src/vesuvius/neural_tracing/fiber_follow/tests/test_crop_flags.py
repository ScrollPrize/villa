"""Model crop size/spacing from the run configuration."""
import pytest

from model_fixtures import run_document
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train import run_config


def model(kind='regression', **fields):
    return run_config.model_config(run_config.resolve(run_document(kind, **fields)))


def test_crop_defaults_are_each_model_types_crop():
    assert model().fine == CropSpec(depth=144, width=104, behind=72, spacing=.5)
    assert model('flow').fine == CropSpec(depth=144, width=104, behind=72, spacing=.5)
    assert model('sequence').fine == CropSpec(depth=80, width=64, behind=16, spacing=.5)


@pytest.mark.parametrize('kind', ['regression', 'flow', 'sequence'])
def test_configured_crop_sets_the_model_crop(kind):
    cfg = model(kind, fine=dict(depth=96, width=80, behind=32, spacing=1.), n_future=16, gate_plane=None)
    assert cfg.fine == CropSpec(depth=96, width=80, behind=32, spacing=1.)
    assert cfg.token_shape == (12, 10, 10)


@pytest.mark.parametrize('fine', [dict(depth=98, width=104, behind=72, spacing=.5),
                                  dict(depth=144, width=104, behind=200, spacing=.5),
                                  dict(depth=144, width=104, behind=72, spacing=0.)])
def test_invalid_crops_are_rejected(fine):
    with pytest.raises(ValueError):
        model(fine=fine)
