import itertools

import numpy as np
import pytest
import torch

from vesuvius.afv_spline_generator.predict import MIN_STD, normalize, predict_fibers, window_starts


CPU = torch.device("cpu")


def pointwise_network():
    """Logits depending on each voxel only, so every window predicts the same value there."""
    network = torch.nn.Conv3d(1, 4, 1)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([-2.0, 1.5, 0.5, 1.0]).view(4, 1, 1, 1, 1))
        network.bias.copy_(torch.tensor([0.0, -0.5, 0.25, -1.0]))
    return network.eval()


def expected(network, ct):
    with torch.no_grad():
        logits = network(torch.from_numpy(normalize(ct))[None, None])[0]
    probability = logits.softmax(0)[1:].numpy()
    return np.moveaxis(np.rint(probability * 255), 0, -1)


def test_window_starts_cover_the_axis_with_half_overlap():
    assert window_starts(128, 128) == [0]
    assert window_starts(256, 128) == [0, 64, 128]
    assert window_starts(200, 128) == [0, 64, 72]


def test_normalize_uses_the_zone_statistics_with_a_deviation_floor():
    rng = np.random.default_rng(0)
    ct = rng.normal(100, 30, (20, 20, 20)).clip(0, 255).astype(np.uint8)
    image = normalize(ct)
    assert image.dtype == np.float32
    assert abs(image.mean()) < 1e-5 and image.std() == pytest.approx(1, abs=1e-5)
    flat = np.full((4, 4, 4), 7, dtype=np.uint8)
    flat[0, 0, 0] = 17
    assert normalize(flat).max() == pytest.approx((17 - flat.mean()) / MIN_STD)


@pytest.mark.parametrize("shape", [(40, 37, 50), (10, 20, 12)])
@pytest.mark.parametrize("mirror", [False, True])
def test_blending_reproduces_a_whole_zone_prediction(shape, mirror):
    rng = np.random.default_rng(1)
    ct = rng.integers(0, 256, shape, dtype=np.uint8)
    network = pointwise_network()
    reported = []
    probabilities = predict_fibers(network, ct, (16, 16, 16), device=CPU, mirror=mirror, progress=lambda d, t: reported.append((d, t)))
    assert probabilities.shape == shape + (3,) and probabilities.dtype == np.uint8
    assert np.abs(probabilities.astype(int) - expected(network, ct)).max() <= 1
    total = reported[-1][1]
    assert reported == [(i, total) for i in range(1, total + 1)]


def test_mirroring_averages_the_logits_of_the_eight_flips():
    torch.manual_seed(0)
    network = torch.nn.Conv3d(1, 4, 3, padding=1).eval()
    ct = np.random.default_rng(2).integers(0, 256, (16, 16, 16), dtype=np.uint8)
    x = torch.from_numpy(normalize(ct))[None, None]
    with torch.no_grad():
        flips = [dims for n in range(4) for dims in itertools.combinations((2, 3, 4), n)]
        logits = sum(torch.flip(network(torch.flip(x, dims)), dims) for dims in flips) / len(flips)
    reference = np.moveaxis(np.rint(logits[0].softmax(0)[1:].numpy() * 255), 0, -1)
    mirrored = predict_fibers(network, ct, (16, 16, 16), device=CPU, mirror=True)
    single = predict_fibers(network, ct, (16, 16, 16), device=CPU, mirror=False)
    assert np.abs(mirrored.astype(int) - reference).max() <= 1
    assert np.abs(single.astype(int) - reference).max() > 1
