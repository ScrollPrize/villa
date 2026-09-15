"""The standalone MemoryEfficientSoftDiceLoss must consume probabilities.

Regression test for the case where it consumed raw logits instead: under the
documented raw-logits setup (activation: none alongside BCEWithLogitsLoss) the
dice denominator crosses zero as the mean logit drifts negative, and the loss
explodes to ~1e13 -- which dominates the combined gradient and erases BCE
supervision for as long as the regime lasts.
"""
import pytest

torch = pytest.importorskip("torch")

from vesuvius.models.training.loss.losses import _create_loss, _resolve_nonlin


def _logits(mean, shape=(2, 1, 16, 16, 16), seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * 0.3 + mean


def _target(shape=(2, 1, 16, 16, 16), positive=0.15, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=generator) < positive).float()


def _dice(config=None):
    return _create_loss("MemoryEfficientSoftDiceLoss", config or {}, 1.0, None, None)


@pytest.mark.parametrize("mean", [0.1, 0.0, -0.1, -0.15, -0.16, -1.0])
def test_dice_stays_bounded_across_the_logit_range(mean):
    """The reported explosion happened between mu=-0.10 and mu=-0.15."""
    loss = _dice()(_logits(mean), _target())
    assert torch.isfinite(loss)
    assert abs(float(loss)) <= 1.0, f"dice out of range at mu={mean}: {float(loss)}"


def test_the_default_applies_sigmoid():
    """Same convention DC_and_BCE_loss already uses when it builds this class."""
    logits = _logits(-1.0)
    target = _target()
    default = _dice()(logits, target)
    explicit = _dice({"apply_nonlin": "sigmoid"})(logits, target)
    assert torch.allclose(default, explicit)


def test_a_config_can_turn_the_nonlinearity_off():
    """For a model that already emits probabilities."""
    probs = torch.sigmoid(_logits(-1.0))
    target = _target()
    off = _dice({"apply_nonlin": "none"})(probs, target)
    assert torch.isfinite(off) and abs(float(off)) <= 1.0


def test_a_named_nonlinearity_resolves_to_a_callable():
    assert _resolve_nonlin("sigmoid") is torch.sigmoid
    assert _resolve_nonlin(None) is None
    assert _resolve_nonlin("none") is None
    assert callable(_resolve_nonlin("softmax"))
    assert _resolve_nonlin(torch.sigmoid) is torch.sigmoid


def test_an_unknown_nonlinearity_is_rejected_rather_than_called():
    """A string passed straight through used to raise deep inside the forward
    pass, if it did not silently do nothing first."""
    with pytest.raises(ValueError, match="unknown apply_nonlin"):
        _resolve_nonlin("selu")
