"""3D (transposed) convolutions through 2D kernels on MPS: identical results, identical modules."""

import copy

import pytest
import torch
from torch import nn

from vesuvius.models.utilities.mps_conv3d import (
    ENV_KEEP_NATIVE,
    MPSConv3d,
    MPSConvTranspose3d,
    conv3d_as_conv2d,
    conv_transpose3d_by_blocks,
    speed_up_3d_convs,
    speed_up_3d_convs_on_mps,
)


def _mps_works() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        (torch.ones(1, device="mps") + 1).cpu()
        return True
    except RuntimeError:
        return False


requires_mps = pytest.mark.skipif(not _mps_works(), reason="requires a working MPS device")

CONV_CASES = [
    dict(in_channels=3, out_channels=5, kernel_size=3, padding=1),
    dict(in_channels=4, out_channels=6, kernel_size=3, stride=2, padding=1),
    dict(in_channels=2, out_channels=3, kernel_size=(3, 1, 5), stride=(2, 1, 1), padding=(1, 0, 2)),
    dict(in_channels=4, out_channels=4, kernel_size=3, dilation=2, padding=2, bias=False),
    dict(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=3),
    dict(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=6),
    dict(in_channels=5, out_channels=7, kernel_size=1),
    dict(in_channels=3, out_channels=4, kernel_size=(2, 3, 3), stride=(2, 2, 2)),
    dict(in_channels=3, out_channels=4, kernel_size=(3, 3, 3), stride=(2, 1, 1), dilation=(2, 1, 1), padding=(3, 1, 1)),
    dict(in_channels=2, out_channels=3, kernel_size=1, padding=(2, 0, 0)),  # edge planes read only padding
]


@pytest.mark.parametrize("kwargs", CONV_CASES)
def test_depth_tap_conv_equals_conv3d(kwargs):
    torch.manual_seed(0)
    conv = nn.Conv3d(**kwargs).double()
    x = torch.randn(2, kwargs["in_channels"], 9, 10, 11, dtype=torch.float64)
    depth_major = x.permute(2, 0, 1, 3, 4).contiguous().permute(1, 2, 0, 3, 4)  # same values, [D, B, C, H, W] memory
    plane = 2 * kwargs["in_channels"] * 10 * 11 * 8
    # per-tap path (one output plane per chunk, three per chunk), stacked path
    for limit in (0, 3 * plane * conv.stride[0], 2**40):
        for inp in (x, depth_major):
            torch.testing.assert_close(conv3d_as_conv2d(conv, inp, limit), conv(x), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(conv3d_as_conv2d(conv, x[0], limit), conv(x[0]), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kernel", [2, (1, 2, 2), (2, 3, 1)])
def test_block_transposed_conv_equals_conv_transpose3d(kernel):
    torch.manual_seed(0)
    conv = nn.ConvTranspose3d(6, 4, kernel, stride=kernel).double()
    x = torch.randn(2, 6, 3, 4, 5, dtype=torch.float64)
    torch.testing.assert_close(conv_transpose3d_by_blocks(conv, x), conv(x), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(conv_transpose3d_by_blocks(conv, x[0]), conv(x[0]), rtol=1e-10, atol=1e-10)


def test_depth_tap_conv_gradients_match():
    torch.manual_seed(0)
    conv = nn.Conv3d(3, 4, 3, padding=1).double()
    alt = copy.deepcopy(conv)
    x = torch.randn(2, 3, 6, 7, 8, dtype=torch.float64, requires_grad=True)
    xa = x.detach().clone().requires_grad_(True)
    conv(x).square().sum().backward()
    for limit in (0, 2**40):
        alt.zero_grad(); xa.grad = None
        conv3d_as_conv2d(alt, xa, limit).square().sum().backward()
        torch.testing.assert_close(xa.grad, x.grad)
        torch.testing.assert_close(alt.weight.grad, conv.weight.grad)
        torch.testing.assert_close(alt.bias.grad, conv.bias.grad)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.InstanceNorm3d(4), nn.LeakyReLU(),
                                 nn.Conv3d(4, 8, 3, stride=2, padding=1))
        self.up = nn.ConvTranspose3d(8, 4, 2, stride=2)
        self.odd_up = nn.ConvTranspose3d(8, 4, 3, stride=2, padding=1, output_padding=1)  # not eligible
        self.circular = nn.Conv3d(4, 4, 3, padding=1, padding_mode="circular")  # not eligible
        self.same = nn.Conv3d(4, 4, 3, padding="same")  # not eligible (string padding)
        self.head = nn.Conv3d(8, 2, 1)

    def forward(self, x):
        z = self.enc(x)
        y = self.circular(self.up(z)) + self.same(self.odd_up(z))
        return self.head(torch.cat([y, y], 1))


def test_swap_keeps_classes_parameters_and_state_dict():
    torch.manual_seed(0)
    net = _Net()
    params = [id(p) for p in net.parameters()]
    keys = list(net.state_dict())
    x = torch.randn(1, 1, 8, 8, 8)
    before = net(x)
    assert speed_up_3d_convs(net) == 4
    assert type(net.enc[0]) is MPSConv3d and type(net.enc[3]) is MPSConv3d and type(net.head) is MPSConv3d
    assert type(net.up) is MPSConvTranspose3d
    assert type(net.odd_up) is nn.ConvTranspose3d and type(net.circular) is nn.Conv3d and type(net.same) is nn.Conv3d
    assert all(isinstance(m, nn.Conv3d) for m in (net.enc[0], net.head))
    assert [id(p) for p in net.parameters()] == params
    assert list(net.state_dict()) == keys
    torch.testing.assert_close(net(x), before)  # off MPS the swapped modules run PyTorch's kernels
    assert speed_up_3d_convs(net) == 0  # idempotent
    fresh = _Net()
    fresh.load_state_dict(net.state_dict())
    copy.deepcopy(net)


def test_swap_only_on_mps(monkeypatch):
    net = _Net()
    assert speed_up_3d_convs_on_mps(net, "cpu") == 0
    assert type(net.enc[0]) is nn.Conv3d
    monkeypatch.setenv(ENV_KEEP_NATIVE, "1")
    assert speed_up_3d_convs_on_mps(net, torch.device("mps")) == 0
    monkeypatch.delenv(ENV_KEEP_NATIVE)
    assert speed_up_3d_convs_on_mps(net, torch.device("mps")) == 4


@requires_mps
def test_swapped_net_on_mps_matches_cpu():
    torch.manual_seed(0)
    net = _Net()
    x = torch.randn(2, 1, 8, 8, 8)
    want = net(x)
    speed_up_3d_convs(net)
    got = net.to("mps")(x.to("mps")).cpu()
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@requires_mps
def test_swapped_transposed_conv_runs_in_half_precision_on_mps():
    conv = nn.ConvTranspose3d(8, 4, 2, stride=2)
    x = torch.randn(1, 8, 4, 4, 4)
    want = conv(x)
    speed_up_3d_convs(conv)
    got = conv.to("mps", torch.float16)(x.to("mps", torch.float16)).float().cpu()
    torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


@requires_mps
def test_swapped_training_step_on_mps_matches_cpu():
    torch.manual_seed(0)
    net = _Net()
    ref = copy.deepcopy(net)
    speed_up_3d_convs(net)
    net.to("mps")
    x = torch.randn(2, 1, 8, 8, 8)
    for model, dev in ((ref, "cpu"), (net, "mps")):
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        model(x.to(dev)).square().mean().backward()
        opt.step()
    for (name, a), b in zip(ref.named_parameters(), net.parameters()):
        torch.testing.assert_close(b.detach().cpu(), a.detach(), rtol=1e-4, atol=1e-5, msg=name)


def test_per_tap_path_sums_low_precision_taps_in_fp32():
    # Under bf16 autocast each tap's 2-D convolution rounds to bf16; summing those in bf16 doubled the error of a
    # native bf16 Conv3d. The taps are now summed in fp32 and rounded once.
    torch.manual_seed(0)
    conv = nn.Conv3d(32, 32, 3, padding=1)
    x = torch.randn(1, 32, 16, 24, 24)
    with torch.no_grad():
        want = conv(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            native = conv(x)
            got = conv3d_as_conv2d(conv, x, stack_limit_bytes=0)
    assert got.dtype == native.dtype == torch.bfloat16
    native_err = (native.float() - want).abs().max()
    assert (got.float() - want).abs().max() <= 1.5 * native_err
