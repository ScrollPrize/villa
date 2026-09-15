import pytest
import torch

from vesuvius.neural_tracing.inference.displacement_helpers import predict_displacement
from vesuvius.neural_tracing.inference.displacement_tta import (
    TTA_FLIP_COMBOS,
    TTA_ROTATE3_PERMS,
    _transform_input_vector_channels,
    run_model_tta,
)


# Copy checkpoints with direction priors take 8 input channels:
# [volume, conditioning, +normal (z, y, x), -normal (z, y, x)].
PRIOR_STARTS = (2, 5)


class ReturnPriors:
    """A stand-in model whose "displacement" is the two input direction priors.

    If TTA transports the prior components together with the spatial transform,
    every variant maps back to the untransformed priors and the merge returns
    them unchanged. If it does not, the variants disagree and the merge drifts.
    """

    def __call__(self, inputs):
        return inputs[:, 2:8]


def _forward(model, inputs, amp_enabled, amp_dtype):
    return model(inputs)


def _inputs():
    torch.manual_seed(205)
    return torch.randn(1, 8, 5, 5, 5)


@pytest.mark.parametrize("mode", ["mirror", "rotate3"])
def test_transported_priors_survive_tta(mode):
    x = _inputs()
    saved = x.clone()
    out = run_model_tta(
        ReturnPriors(), x, False, torch.float32, _forward,
        transform_mode=mode, tta_batch_size=1,
        input_vector_channel_starts=PRIOR_STARTS,
    )
    torch.testing.assert_close(out, x[:, 2:8], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x, saved, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["mirror", "rotate3"])
def test_untransported_priors_do_not_survive_tta(mode):
    # The behaviour this change fixes: without component transport the merged
    # "displacement" is not the prior that went in.
    x = _inputs()
    out = run_model_tta(
        ReturnPriors(), x, False, torch.float32, _forward,
        transform_mode=mode, tta_batch_size=1,
    )
    assert float((out - x[:, 2:8]).abs().max()) > 0.5


def test_no_vector_channels_is_a_no_op():
    x = _inputs()
    assert _transform_input_vector_channels(x) is x
    assert _transform_input_vector_channels(x, vector_channel_starts=()) is x


def test_scalar_channels_are_left_alone():
    x = _inputs()
    for flip_dims in TTA_FLIP_COMBOS:
        y = _transform_input_vector_channels(x, vector_channel_starts=PRIOR_STARTS, flip_dims=flip_dims)
        torch.testing.assert_close(y[:, :2], x[:, :2], rtol=0, atol=0)
    for perm in TTA_ROTATE3_PERMS:
        y = _transform_input_vector_channels(x, vector_channel_starts=PRIOR_STARTS, axis_perm=perm)
        torch.testing.assert_close(y[:, :2], x[:, :2], rtol=0, atol=0)


def test_flip_negates_only_the_flipped_component():
    x = _inputs()
    y = _transform_input_vector_channels(x, vector_channel_starts=PRIOR_STARTS, flip_dims=(-3,))
    # dim -3 is depth (z): channel 0 of each vector group flips sign, the rest are untouched.
    torch.testing.assert_close(y[:, 2], -x[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(y[:, 3:5], x[:, 3:5], rtol=0, atol=0)
    torch.testing.assert_close(y[:, 5], -x[:, 5], rtol=0, atol=0)
    torch.testing.assert_close(y[:, 6:8], x[:, 6:8], rtol=0, atol=0)


@pytest.mark.parametrize("start", [-1, 6, 8])
def test_out_of_range_vector_group_is_rejected(start):
    x = _inputs()
    with pytest.raises(ValueError, match="vector channel start"):
        _transform_input_vector_channels(x, vector_channel_starts=(start,), flip_dims=(-1,))


class ReturnPriorsAsDict:
    def __call__(self, inputs):
        return {"displacement": inputs[:, 2:8]}


def _model_state(model_config):
    return {
        "model": ReturnPriorsAsDict(),
        "amp_enabled": False,
        "amp_dtype": torch.float32,
        "model_config": model_config,
    }


class _Args:
    tta = True
    tta_batch_size = 1


def test_predict_displacement_transports_priors_for_direction_conditioned_checkpoints():
    x = _inputs()
    out = predict_displacement(_Args(), _model_state({"use_triplet_direction_priors": True}), x)
    torch.testing.assert_close(out, x[:, 2:8], rtol=1e-5, atol=1e-5)


def test_predict_displacement_leaves_other_checkpoints_unchanged():
    x = _inputs()
    legacy = run_model_tta(ReturnPriors(), x, False, torch.float32, _forward, tta_batch_size=1)
    out = predict_displacement(_Args(), _model_state({}), x)
    torch.testing.assert_close(out, legacy, rtol=0, atol=0)


def test_predict_displacement_does_not_guess_a_layout_for_other_widths():
    # Direction priors are enabled but the tensor is not the 8-channel layout:
    # nothing is transported rather than transporting the wrong channels.
    torch.manual_seed(205)
    x = torch.randn(1, 6, 5, 5, 5)

    class ReturnTail:
        def __call__(self, inputs):
            return {"displacement": inputs[:, 3:6]}

    state = _model_state({"use_triplet_direction_priors": True})
    state["model"] = ReturnTail()
    out = predict_displacement(_Args(), state, x)

    class ReturnTailPlain:
        def __call__(self, inputs):
            return inputs[:, 3:6]

    legacy = run_model_tta(ReturnTailPlain(), x, False, torch.float32, _forward, tta_batch_size=1)
    torch.testing.assert_close(out, legacy, rtol=0, atol=0)
