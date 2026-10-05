"""Model crop size/spacing from the training command line."""
import pytest

from model_fixtures import REQUIRED
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train.train import build_parser, model_config_from_args


def test_crop_flag_defaults_are_the_current_model_crop():
    args = build_parser().parse_args(REQUIRED)
    assert model_config_from_args(args).fine == CoordinateRegressionConfig().fine


@pytest.mark.parametrize('model', ['coordinate_regression', 'flow_matching'])
def test_crop_flags_set_the_model_crop(model):
    args = build_parser().parse_args(REQUIRED+['--model', model, '--crop-depth', '96', '--crop-width', '80',
                                               '--crop-behind', '32', '--crop-spacing', '1.0'])
    cfg = model_config_from_args(args)
    assert cfg.fine == CropSpec(depth=96, width=80, behind=32, spacing=1.)
    assert cfg.token_shape == (24, 20, 20)


def test_invalid_crop_flags_are_rejected():
    for flags in (['--crop-depth', '98'], ['--crop-behind', '200'], ['--crop-spacing', '0']):
        with pytest.raises(ValueError):
            model_config_from_args(build_parser().parse_args(REQUIRED+flags))
