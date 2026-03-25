import random

import torch

from vesuvius.models.augmentation.transforms.noise import LayerMixDropoutTransform


def test_layer_mix_dropout_relocates_and_drops_channels(monkeypatch):
    image = torch.arange(6 * 2 * 2, dtype=torch.float32).reshape(6, 2, 2)
    transform = LayerMixDropoutTransform(
        min_crop_ratio=0.5,
        max_crop_ratio=0.5,
        cutout_max_count=2,
        cutout_probability=1.0,
    )

    randint_values = iter([3, 1, 0, 2])
    monkeypatch.setattr(random, 'randint', lambda a, b: next(randint_values))
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    monkeypatch.setattr(torch, 'randperm', lambda n, device=None: torch.tensor([2, 0, 1], device=device))

    output = transform(image=image.clone())['image']

    expected = torch.zeros_like(image)
    expected[0:3] = image[1:4]
    expected[2] = 0
    expected[0] = 0

    assert torch.equal(output, expected)


def test_layer_mix_dropout_handles_rounding_when_min_crop_exceeds_max_crop(monkeypatch):
    image = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    transform = LayerMixDropoutTransform(
        min_crop_ratio=0.41,
        max_crop_ratio=0.41,
        cutout_max_count=0,
        cutout_probability=0.0,
    )

    randint_values = iter([3, 1, 0, 0])
    monkeypatch.setattr(random, 'randint', lambda a, b: next(randint_values))
    monkeypatch.setattr(random, 'random', lambda: 1.0)
    monkeypatch.setattr(torch, 'randperm', lambda n, device=None: torch.arange(n, device=device))

    output = transform(image=image.clone())['image']

    expected = torch.zeros_like(image)
    expected[0:3] = image[1:4]

    assert torch.equal(output, expected)
