import builtins

import pytest
import torch

pytest.importorskip("numba")

from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import SmearTransform

CASES = [
    ((1, 17, 128, 128), dict(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=3)),  # ink default
    ((1, 17, 128, 128), dict(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=1)),
    ((1, 17, 128, 128), dict(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=2)),
    ((3, 17, 32, 32), dict(shift=(10, 3), alpha=0.5, num_prev_slices=1, smear_axis=3)),
    ((2, 40, 40), dict(shift=(10, 0), alpha=0.5, num_prev_slices=2, smear_axis=1)),  # 1D slices
    ((2, 40, 40), dict(shift=(7, 2), alpha=0.35, num_prev_slices=4, smear_axis=2)),
    ((1, 5, 9, 7, 6), dict(shift=(-3, -2), alpha=0.3, num_prev_slices=2, smear_axis=4)),  # 4D spatial, negative shift
    ((1, 3, 16, 16), dict(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=1)),  # axis not longer than k
]


def _image(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(shape, generator=generator) * 3 + 0.5
    image.view(-1)[::7] = 0.0
    return image


@pytest.mark.parametrize(("shape", "kwargs"), CASES)
def test_compiled_smear_matches_torch_implementation_exactly(shape, kwargs):
    transform = SmearTransform(**kwargs)
    image = _image(shape)
    expected = transform._apply_torch(image.clone())
    original = image.clone()

    actual = transform._apply_to_image(image)

    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(image, original), "input must not be modified"


def test_non_float32_images_use_the_torch_implementation():
    transform = SmearTransform(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=3)
    image = _image((1, 6, 12, 12)).double()
    expected = transform._apply_torch(image.clone())

    assert torch.equal(transform._apply_to_image(image.clone()), expected)


def test_falls_back_when_numba_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.endswith("_smear_kernel"):
            raise ImportError("numba unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    transform = SmearTransform(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=3)
    image = _image((1, 6, 12, 12))
    expected = transform._apply_torch(image.clone())

    assert torch.equal(transform._apply_to_image(image.clone()), expected)


def test_invalid_axis_raises():
    transform = SmearTransform(smear_axis=9)
    with pytest.raises(ValueError, match="smear_axis"):
        transform._apply_to_image(_image((1, 6, 12, 12)))
