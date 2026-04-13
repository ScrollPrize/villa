from types import SimpleNamespace

import pytest
import torch

from vesuvius.models.build.build_network_from_config import NetworkFromConfig


def _make_mgr(
    patch_size,
    targets,
    architecture_type="primus_s",
    patch_drop_rate=0.0,
):
    return SimpleNamespace(
        targets=targets,
        train_patch_size=patch_size,
        train_batch_size=2,
        in_channels=1,
        autoconfigure=False,
        model_name="primus_test",
        enable_deep_supervision=False,
        model_config={
            "architecture_type": architecture_type,
            "input_shape": patch_size,
            "patch_embed_size": tuple([8] * len(patch_size)),
            "patch_drop_rate": patch_drop_rate,
        },
    )


def test_primus_forward_shape_3d():
    mgr = _make_mgr(
        patch_size=(16, 16, 16),
        targets={"ink": {"out_channels": 2, "activation": "none"}},
    )
    model = NetworkFromConfig(mgr)
    out = model(torch.randn(2, 1, 16, 16, 16))

    assert set(out.keys()) == {"ink"}
    assert out["ink"].shape == (2, 2, 16, 16, 16)


def test_primus_forward_shape_2d():
    mgr = _make_mgr(
        patch_size=(32, 32),
        targets={"ink": {"out_channels": 3, "activation": "none"}},
    )
    model = NetworkFromConfig(mgr)
    out = model(torch.randn(2, 1, 32, 32))

    assert set(out.keys()) == {"ink"}
    assert out["ink"].shape == (2, 3, 32, 32)


def test_primus_mae_mask_shape_and_dtype():
    torch.manual_seed(0)
    mgr = _make_mgr(
        patch_size=(16, 16, 16),
        targets={"mae": {"out_channels": 1, "activation": "none"}},
        patch_drop_rate=0.75,
    )
    model = NetworkFromConfig(mgr)
    model.train()

    out, mask = model(torch.randn(2, 1, 16, 16, 16), return_mae_mask=True)

    assert out["mae"].shape == (2, 1, 16, 16, 16)
    assert mask.shape == (2, 1, 16, 16, 16)
    assert mask.dtype == torch.bool
    assert 0 < mask.float().mean().item() < 1


def test_primus_single_train_step_backward():
    mgr = _make_mgr(
        patch_size=(16, 16, 16),
        targets={"ink": {"out_channels": 1, "activation": "none"}},
    )
    model = NetworkFromConfig(mgr)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 1, 16, 16, 16)
    y = torch.randn(2, 1, 16, 16, 16)
    optim.zero_grad(set_to_none=True)
    pred = model(x)["ink"]
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    assert model.shared_encoder.patch_embed.stem.blocks[0].conv1.conv.weight.grad is not None
    assert model.task_heads["ink"].weight.grad is not None
    optim.step()


def test_primus_rejects_runtime_spatial_mismatch_with_clear_error():
    mgr = _make_mgr(
        patch_size=(16, 16, 16),
        targets={"ink": {"out_channels": 1, "activation": "none"}},
    )
    model = NetworkFromConfig(mgr)

    with pytest.raises(ValueError, match="configured input_shape"):
        model(torch.randn(1, 1, 24, 24, 24))


def test_primus_disables_deep_supervision_with_warning(capsys):
    mgr = _make_mgr(
        patch_size=(16, 16, 16),
        targets={"ink": {"out_channels": 1, "activation": "none"}},
    )
    mgr.enable_deep_supervision = True

    model = NetworkFromConfig(mgr)
    captured = capsys.readouterr()

    assert "Disabling deep supervision for this run" in captured.out
    assert mgr.enable_deep_supervision is False

    out = model(torch.randn(1, 1, 16, 16, 16))
    assert out["ink"].shape == (1, 1, 16, 16, 16)
