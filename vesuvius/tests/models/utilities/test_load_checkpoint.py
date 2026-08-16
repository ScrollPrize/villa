"""Tests for models/utilities/load_checkpoint.py's success-reporting contract.

load_checkpoint() returns (model, optimizer, scheduler, start_epoch, success),
and callers rely on the boolean: models/training/train.py gates restoring EMA
model state and applying loss overrides on `if checkpoint_loaded:`.
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

from vesuvius.models.utilities.load_checkpoint import load_checkpoint


class _ToyModel(nn.Module):
    """Two layers so a partial load has something to genuinely miss.

    layer_a always matches the checkpoint; layer_b's shape can be made to
    drift, simulating a real architecture change between when a checkpoint
    was saved and when it's later resumed from.
    """

    def __init__(self, drifted_shape: bool = False):
        super().__init__()
        self.layer_a = nn.Linear(8, 8)
        self.layer_b = nn.Linear(8, 8 if not drifted_shape else 16)


def _make_mgr():
    return SimpleNamespace(
        scheduler="poly",
        scheduler_kwargs={},
        initial_lr=0.01,
        max_epoch=10,
        load_weights_only=False,
    )


def _save_checkpoint(tmp_path):
    saved = _ToyModel(drifted_shape=False)
    with torch.no_grad():
        saved.layer_a.weight.fill_(1.0); saved.layer_a.bias.fill_(1.0)
        saved.layer_b.weight.fill_(2.0); saved.layer_b.bias.fill_(2.0)
    path = tmp_path / "ckpt.pt"
    # No 'model_config' key: a real, common checkpoint shape, and it also
    # keeps this test from needing to construct the model-rebuild machinery
    # (NetworkFromConfig/create_optimizer), which this code path never enters.
    torch.save({"model": saved.state_dict()}, path)
    return path


def test_partial_load_reports_failure(tmp_path):
    """A shape-mismatched layer forces the strict load to fail and fall back
    to a partial, non-strict load -- the function's own success flag must
    reflect that, not silently report True.

    Reproduces a real bug: the function's final return hardcoded True on
    every path that did not raise, including this one, so a caller checking
    the flag had no way to know layer_b was left at random initialization.
    """
    ckpt_path = _save_checkpoint(tmp_path)
    target_model = _ToyModel(drifted_shape=True)
    optimizer = torch.optim.SGD(target_model.parameters(), lr=0.01)
    mgr = _make_mgr()

    _, _, _, _, checkpoint_loaded = load_checkpoint(
        checkpoint_path=str(ckpt_path),
        model=target_model,
        optimizer=optimizer,
        scheduler=None,
        mgr=mgr,
        device="cpu",
        load_weights_only=False,
    )

    assert checkpoint_loaded is False, (
        "load_checkpoint() must report failure when it fell back to a "
        "partial load -- a caller gating real behaviour (EMA restore, loss "
        "overrides) on this flag would otherwise proceed as if the "
        "checkpoint had loaded fully, while part of the model is still at "
        "random initialization."
    )

    # And confirm the actual mechanism: layer_a (matched) loaded from the
    # checkpoint; layer_b (drifted) did not.
    after = target_model.state_dict()
    assert torch.allclose(after["layer_a.weight"], torch.full((8, 8), 1.0))
    assert not torch.allclose(
        after["layer_b.weight"][:8], torch.full((8, 8), 2.0)
    ), "layer_b should still be at its random init, not the checkpoint's saved value"


def test_clean_match_still_reports_success(tmp_path):
    """No-regression check: an exact shape match must still report True."""
    ckpt_path = _save_checkpoint(tmp_path)
    target_model = _ToyModel(drifted_shape=False)
    optimizer = torch.optim.SGD(target_model.parameters(), lr=0.01)
    mgr = _make_mgr()

    _, _, _, _, checkpoint_loaded = load_checkpoint(
        checkpoint_path=str(ckpt_path),
        model=target_model,
        optimizer=optimizer,
        scheduler=None,
        mgr=mgr,
        device="cpu",
        load_weights_only=False,
    )

    assert checkpoint_loaded is True
    after = target_model.state_dict()
    assert torch.allclose(after["layer_a.weight"], torch.full((8, 8), 1.0))
    assert torch.allclose(after["layer_b.weight"], torch.full((8, 8), 2.0))
